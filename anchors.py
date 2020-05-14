import torch.nn as nn
import torch
import numpy as np
import sys


class Anchors(nn.Module):
    """ Implementation of RPN as per Faster R-CNN paper. """
    
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = torch.cuda.FloatTensor([0.5, 1, 2])
        if scales is None:
            self.scales = torch.cuda.FloatTensor([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
#             self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
            
        print(">>> Pyramid Levels:", self.pyramid_levels)
        print(">>> Strides:", self.strides)
        print(">>> Sizes:", self.sizes)
        print(">>> Ratios:", self.ratios)
        print(">>> Scales:", self.scales)
        
        
    def forward(self, image):
        image_shape = torch.tensor(image.shape[2:])
        image_shapes = torch.zeros([len(self.pyramid_levels), len(image_shape)])
        all_anchors = torch.zeros((0, 4), dtype=torch.float32).cuda()
        
        for i, x in  enumerate(self.pyramid_levels):
            image_shapes[i] = ( image_shape + (2. ** x) - 1. ) // (2. ** x)
            
            anchors = self._generate_anchors(base_size=self.sizes[i], ratios=self.ratios, scales=self.scales)
            shifted_anchors = self._shift(image_shapes[i], self.strides[i], anchors)
            all_anchors = torch.cat((all_anchors, shifted_anchors), dim=0)
            
        all_anchors = all_anchors.unsqueeze(0)
        
        return all_anchors
    
    
    def _generate_anchors(self, base_size, ratios, scales):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        """
        num_anchors = len(ratios) * len(scales)
        
        # Initialize output anchors
        anchors = torch.zeros((num_anchors, 4)).cuda('cuda:0')
        
        # Scale base_size
        anchors[:, 2:] = base_size * scales.repeat(2, len(ratios)).transpose(0,1)
        
        # Compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]
        
        # Correct for ratios
        ratios_t = torch.flatten(ratios.unsqueeze(1).repeat(1, len(scales)))
        
        anchors[:, 2] = torch.sqrt(areas / ratios_t)
        anchors[:, 3] = anchors[:, 2] * ratios_t
        
#         anchors[:, 2] = torch.sqrt(areas / torch.repeat_interleave(ratios, len(scales)))
#         anchors[:, 3] = anchors[:, 2] * torch.repeat_interleave(ratios, len(scales))  
        
        
        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= (anchors[:, 2] * 0.5).repeat(2, 1).transpose(0,1)
        anchors[:, 1::2] -= (anchors[:, 3] * 0.5).repeat(2, 1).transpose(0,1)

        return anchors
    
    def _shift(self, shape, stride, anchors):

        shift_x =  (torch.arange(0, shape[0]).cuda('cuda:0') + 0.5) * stride
        shift_y =  (torch.arange(0, shape[1]).cuda('cuda:0') + 0.5) * stride

        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
        
        shifts = torch.stack((
            torch.flatten(shift_x), torch.flatten(shift_y),
            torch.flatten(shift_x), torch.flatten(shift_y)
        )).transpose(0,1)
        
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchors.size(0)
        K = shifts.size(0)
        all_anchors = (anchors.contiguous().view((1,A,4)) + shifts.contiguous().view((1,K,4)).permute(1,0,2))
        all_anchors = all_anchors.contiguous().view((K * A, 4))
#         all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
#         all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors
    
    
    

class RPN(nn.Module):
    """ Implementation of RPN as per Faster R-CNN paper. """
    
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(RPN, self).__init__()
        
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
            
#         print(">>> Pyramid Levels:", self.pyramid_levels)
#         print(">>> Strides:", self.strides)
#         print(">>> Sizes:", self.sizes)
#         print(">>> Ratios:", self.ratios)
#         print(">>> Scales:", self.scales)
        
        
    def forward(self, image):
        
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2 ** x -1) // (2 ** x) for x in self.pyramid_levels]
        
        # Compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        
        # Anchors depend on size of pyramid.
        for idx, p in enumerate(self.pyramid_levels):
            anchors = self._generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = self._shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
            
        all_anchors = np.expand_dims(all_anchors, axis=0)
        
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))
    
    
    def _generate_anchors(self, base_size, ratios, scales):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        """
        
        num_anchors = len(ratios) * len(scales)
        
        # Initialize output anchors
        anchors = np.zeros((num_anchors, 4))
        
        # Scale base_size
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
#         print(anchors[:,2:])
        
        # Compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]
#         print(areas)
        
        # Correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))  
#         print(anchors[:,2:])
        
        
        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors
    
    def _shift(self, shape, stride, anchors):
        
        shift_x =  (np.arange(0, shape[0]) + 0.5) * stride
        shift_y =  (np.arange(0, shape[1]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        
        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
        )).transpose()
        
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        
        return all_anchors