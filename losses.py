import torch
import torch.nn as nn
import numpy as np
import sys


def calc_iou(a, b):
    """
    Computes IoU over prediction (a) and groundtruth (b) bounding boxes.
    Input:
        - a: prediction bbox
        - b: groundtruth bbox
    Output:
        - IoU: ratio [0-1] representing intersection over union
    """
    # Compute area of annotations
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    
    # 1. Computes intersection bbox coordinates between prediction (a) and groundtruth (b)
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih

    # 2. Computes area of the union of bounding boxes a & b
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - intersection
    ua = torch.clamp(ua, min=1e-8)
    
    
    IoU = intersection / ua
    
    return IoU



class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0, training=True, malNet=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.training = training
        self.malNet = malNet
        
        
    def set_MALnet(self, max_iter, k=50):
        """ Initializes configutartion for MALnet. """
        self.malNet = True
        self.max_iter = max_iter
        self.k = k
        
    def update_bag_size(self, iteration):
        """ Reduces bag size on each training iteration. """
        lmbda = iteration / float(self.max_iter)
        self.bag_size = int(self.k * (1 - lmbda) + 1)
        
    
    def _compute_targets(self, classification, IoU_max, assigned_annotations, positive_indices):
        targets = torch.ones(classification.size()) * -1

        if torch.cuda.is_available():
            targets = targets.cuda()

        # Those with low IoU index are incorrect
        targets[torch.lt(IoU_max, 0.4), :] = 0
        targets[positive_indices, :] = 0
        targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
        
        return targets
    
    
    def _compute_cls_loss(self, targets, classification):
        """ 
        Computes Focal Loss as per formula described on the paper.
        Input:
            - targets: 
            - classification:
        Returns:
            - cls_loss:
        """
        if torch.cuda.is_available():
            alpha_factor = torch.ones(targets.size()).cuda() * self.alpha
        else:
            alpha_factor = torch.ones(targets.size()) * self.alpha

        # Adjust alpha and focal weight (1-p_t) for label
        alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
        
        # Compute loss prefix
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

        # Batch CE loss
        bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

        # Focal loss   
        cls_loss = focal_weight * bce
        
        if torch.cuda.is_available():
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        else:
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
        
        return cls_loss
    
    
    def _compute_rgs_loss(self, regression, anchor_tp, assigned_annotations, positive_indices):
        
        (anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y) = anchor_tp
        
        
        # Anchors/Predictions
        anchor_widths_pi = anchor_widths[positive_indices]
        anchor_heights_pi = anchor_heights[positive_indices]
        anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
        anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

        # Groundtruth
        gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
        gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
        gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths             # Center point
        gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

        # clip widths to 1
        gt_widths  = torch.clamp(gt_widths, min=1)
        gt_heights = torch.clamp(gt_heights, min=1)

        targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
        targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
        targets_dw = torch.log(gt_widths / anchor_widths_pi)
        targets_dh = torch.log(gt_heights / anchor_heights_pi)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
        targets = targets.t()

        if torch.cuda.is_available():
            targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
        else:
            targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

            
        regression_diff = torch.abs(targets - regression)

        # Where do they get this loss? Paper?
        regression_loss = torch.where(
            torch.le(regression_diff, 1.0 / 9.0),
            0.5 * 9.0 * torch.pow(regression_diff, 2),
            regression_diff - 0.5 / 9.0
        )
        
        return regression_loss
        
    
    
    def forward(self, classifications, regressions, anchors, annotations):
        
#         print(">>> CLASSIFICATION:", classifications.size())
#         print(">>> REGRESSION:", regressions.size())
#         print(">>> ANCHORS:", anchors.size())
#         print(">>> ANNOTATIONS:", annotations.size())

        batch_size = classifications.size(0)
        classification_losses = []
        regression_losses = []
        
        anchor = anchors[0,:,:]
        
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        
        anchor_tp = (anchor_widths, anchor_heights, anchor_ctr_x, anchor_ctr_y)
        
        for j in range(batch_size):

            classification = classifications[j]
            regression = regressions[j]

            bbox_annotation = annotations[j]
            # Retrieve bbox of EXISTING annotations
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            # If there are no annotations in the image...
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                    classification_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    classification_losses.append(torch.tensor(0).float())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # Computes IoU, each row being a prediction for each annotation [n_pred, n_annotations]
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
               
            if not self.malNet:
                # Retrieves annotation that most closely fits IoU for each prediction
                IoU_max, IoU_argmax = torch.max(IoU, dim=1)

                positive_indices = torch.ge(IoU_max, 0.5)

                num_positive_anchors = positive_indices.sum()
                
                # Retrieve groundtruth annotations based on prediction that most closely matches annotation
                assigned_annotations = bbox_annotation[IoU_argmax, :]

                targets = self._compute_targets(classification, IoU_max, assigned_annotations, positive_indices)
                
                cls_loss = self._compute_cls_loss(targets, classification)      # [n_anchors, n_classes]
                
                if positive_indices.sum() > 0:
                    assigned_annotations = assigned_annotations[positive_indices, :]
                    rgs = regression[positive_indices, :]
                    
                    rgs_loss = self._compute_rgs_loss(rgs, anchor_tp, assigned_annotations, positive_indices)
                else:
                    if torch.cuda.is_available():
                        rgs_loss = torch.tensor(0).float().cuda()
                    else:
                        rgs_loss = torch.tensor(0).float()
                        
                classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
                regression_losses.append(rgs_loss.mean())
                    
            
            elif self.malNet:
                n_annotations = bbox_annotation.size(0)
                lst = np.arange(n_annotations)
                
                #### CLASSIFICATION LOSS ###
                # 1. Retrieve top-k IoU bbox INDICES and SCORES
                IoU_bags_indices = torch.argsort(IoU, dim=0, descending=True)[:self.bag_size]
                IoU_topk = torch.stack([IoU[IoU_bags_indices[:,i], i] for i in range(n_annotations)])   # [n_anchors, k]
                IoU_bags_indices = IoU_bags_indices.t()

                # 2. Retrieve classification predictions from top-k bboxes
                clss = classification[IoU_bags_indices]

                # 3. Compute one-hot encoding of labels for top-k bbox predictions
                gt_labels = torch.zeros(clss.size()).cuda()
                gt_labels[lst,:,bbox_annotation[lst,4].long()] = 1

                # 4. Compute classification score.
                cls_loss = self._compute_cls_loss(gt_labels, clss)
                cls_loss = torch.sum(cls_loss, dim=(1,2))
        

                #### REGRESSION LOSS ####
                # Mask ( IoU_topk >= 0.5 ==> 1. ; IoU_topk < 0.5 ==> 0)
                mask = torch.ge(IoU_topk.view(1,-1).squeeze(0), 0.5)
                
                if mask.sum() > 0:
                    # Sequence of indices of top-k IoU bboxes that yield IoU > 0.5 (POSITIVE as per paper)
                    positive_indices = IoU_bags_indices.contiguous().view(1,-1).squeeze(0)[mask]

                    # 1. Retrieve POSITIVE groundtruth bounding-boxes
                    gt_bboxes = bbox_annotation[np.repeat(np.arange(n_annotations), self.bag_size), :4]
                    gt_bboxes = gt_bboxes[mask]  

                    # 2. Retrieve regression predictions from top-k POSITIVE bboxes.
                    reg = regression[positive_indices,:]

                    # 3. Compute Regression score
                    rgs_loss = self._compute_rgs_loss(reg, anchor_tp, gt_bboxes, positive_indices)

                else:
                    if torch.cuda.is_available():
                        rgs_loss = torch.tensor(0).float().cuda()
                    else:
                        rgs_loss = torch.tensor(0).float()
                        
                classification_losses.append(cls_loss.mean())
                regression_losses.append(rgs_loss.mean())
            

#             print(">>> CLS SIZE:",cls_loss.size())
#             print(">>> CLS:", cls_loss.mean)
#             print(">>> REG SIZE:", rgs_loss.size())
#             print(">>> REG:", rgs_loss.mean())
        
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)