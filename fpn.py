import torch
import torch.nn as nn
from utils import ConvBlock

class FPN(nn.Module):
    """ 
    Computes features throughout varying levels of the network.
    Contains adjustments as per RetinaNet paper; a) P6 obtained
    via 3x3 stride 2 convolution on C5; b) P7 computed by applying
    ReLU followed by a 3x3 stride 2 convolution on P6.
    
    Input:
        - features : list of output features at each level of the forward pass in decreasing order (C5, C4, etc.)
        - dimensions : list of feature map dimensions at each level of the forward pass in decreasing order.
        - max_plane : dimension of the largest feature map (last at the top of the network)
        - feature_size : features at each level of the backward pass (all have same fixed length)
        
    Output:
        - p_results : output values at each of the layers in descending order
    
    """
    
    # We use constant output features, although we can use each of the channels to preserve lenght & scales
    def __init__(self, features, feature_size=256):
        super(FPN, self).__init__()

        self.feature_size = feature_size
        self.n_layers = len(features)
        

        # Operations
        self.conv1 = self._make_conv(features, kernel=1, stride=1, padding=0)
        self.upsampler = self._make_upsampler(self.n_layers)
        self.conv2 = self._make_conv([feature_size] * self.n_layers, kernel=3, stride=1, padding=1)
        
        # Operations P6 & P7
        self.p6 = nn.Conv2d(features[-1], feature_size, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        )
        
        
        
    def _make_conv(self, features, kernel, stride, padding):
        """ Creates modules with convolutions for each layer."""
        layer = []
        for i in range(len(features)):
            layer.insert(0, nn.Conv2d(features[i], self.feature_size, kernel_size=kernel, stride=stride, padding=padding) )
            
        return nn.ModuleList(layer)
    
    def _make_upsampler(self, layers):
        """ Creates upsampler modules for each level."""
        layer = []
        for i in range(layers - 1):
            layer.insert(0, nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) )
            
        return nn.ModuleList(layer)
            
        
    def forward(self, x):
        p_results = []
        
        # There is no fusion on 1st layer
        x_lateral = self.conv1[0](x[0])
        x_out = self.conv2[0](x_lateral)
    
        p_results.append(x_out)
        
        for i in range(1, self.n_layers):
            # 1. Compute top input 
            x_top = self.upsampler[i-1](x_lateral)
            
            # 2. Retrieve lateral input
            x_lateral = self.conv1[i](x[i])
            
            # 3. Merge lateral and top input
            x_merged = x_lateral + x_top
            
            # 4. Compute PX output
            x_out = self.conv2[i](x_merged)
            
            # 5. Retain results
            p_results.append(x_out)
        
        # 6. Compute P6 & P7
        p6 = self.p6(x[0])
        p7 = self.p7(p6)
        p_results.insert(0, p6)
        p_results.insert(0, p7)
        
        
        # Need to verify that when results are added in array, information of the gradient is still not lost...
        print(">>> P7:", p_results[0].size())
        print(">>> P6:", p_results[1].size()) 
        print(">>> P5:", p_results[2].size())
        print(">>> P4:", p_results[3].size())
        print(">>> P3:", p_results[4].size())
        
        return p_results