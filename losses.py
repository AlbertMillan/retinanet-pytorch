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
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
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
            
            # Retrieves annotation that most closely fits IoU for each prediction
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            # Retrieve groundtruth annotations based on prediction that most closely matches annotation
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            
            targets = self._compute_targets(classification, IoU_max, assigned_annotations, positive_indices)

            cls_loss = self._compute_cls_loss(targets, classification)      # [n_anchors, n_classes]

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

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

                
#                 negative_indices = 1 + (~positive_indices)    # ~: invert bits

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                # Where do they get this loss? Paper?
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
        
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)