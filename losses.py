import torch
import torch.nn as nn
import numpy as np
import sys


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU




class FocalLoss(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    
    def forward(self, classifications, regressions, anchors, annotations):
        
        print(">>> CLASSIFICATION...")
        for i in range(len(classification)):
            print("Inputs:", classification[i].size())
            
        print(">>> REGRESSION...")
        for i in range(len(regression)):
            print("Inputs:", regression[i].size())
        print(">>> Anchors:", anchors.size())
#         print(">>> Annotations:",classification.size())


        n_levels = len(classifications)
        batch_size = classifications[0].size(0)
        classification_losses = []
        regression_losses = []
        
        anchor = anchors[0,:,:]
        
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        
        
        # Adjustment for varying ranges classification
        
        for i in range(n_levels):
            
            for j in range(batch_size):

                classification = classifications[i][j]
                regression = regressions[i][j]

                bbox_annotation = annotations[j]
                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

                # Condition?
                if bbox_annotation.shape[0] == 0:
                    if torch.cuda.is_available():
                        regression_losses.append(torch.tensor(0).float().cuda())
                        classification_losses.append(torch.tensor(0).float().cuda())
                    else:
                        regression_losses.append(torch.tensor(0).float())
                        classification_losses.append(torch.tensor(0).float())

                    continue

                classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
                
                IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
                
                IoU_max, IoU_argmax = torch.max(IoU, dim=1)
                
                    
        
        
        sys.exit()
        
        return out