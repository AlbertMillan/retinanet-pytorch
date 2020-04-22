import torch.nn as nn
import torch
import numpy as np
from utils import ConvBlock, BasicBlock, Bottleneck

import sys, os


def layers_dim(h_x, w_x, factor, n_layers):
    h = []
    w = []
    for i in range(n_layers - 1):
        h.append(h_x)
        w.append(w_x)
        h_x /= float(factor)
        w_x /= float(factor)
    return h, w



class FPN(nn.Module):
    """ 
    Computes features throughout varying levels of the network.
    
    Input:
        - features : list of output features at each level of the forward pass in decreasing order (C5, C4, etc.)
        - dimensions : list of feature map dimensions at each level of the forward pass in decreasing order.
        - max_plane : dimension of the largest feature map (last at the top of the network)
        - feature_size : features at each level of the backward pass (all have same fixed length)
        
    Output:
        - p_results : output values at each of the layers in descending order
    
    """
    
    # We use constant output features, although we can use each of the channels to preserve lenght & scales
    def __init__(self, features, dimensions, max_plane, feature_size=256):
        super(FPN, self).__init__()
        
        assert len(features) == len(dimensions), "Lateral connections do not match top-to-bottom connections."
        
        self.feature_size = feature_size
        self.n_layers = len(features)
        
        # Input Re-scaling
        self.lvl_conv = nn.Conv2d(max_plane, feature_size, kernel_size=1, stride=1, padding=0)
        
        # Operations
        self.conv1 = self._make_conv(features, kernel=1, stride=1, padding=0)
        self.upsampler = self._make_upsampler(dimensions)
        self.conv2 = self._make_conv([feature_size] * self.n_layers, kernel=3, stride=1, padding=1)
        
        
    def _make_conv(self, features, kernel, stride, padding):
        """ Creates modules with convolutions for each layer."""
        layer = []
        for i in range(len(features)):
            layer.insert(0, nn.Conv2d(features[i], self.feature_size, kernel_size=kernel, stride=stride, padding=padding) )
            
        return nn.ModuleList(layer)
        
        
    def _make_upsampler(self, dimensions):
        """ Creates upsampler modules for each level."""
        layer = []
        for i in range(len(dimensions)):
            layer.insert(0, nn.Upsample(size=(int(dimensions[i])), mode='bilinear', align_corners=False) )
            
        return nn.ModuleList(layer)
            
        
    def forward(self, x):
        p_results = []
        
        # Leveling Convolution
        out = self.lvl_conv(x[0])
        
        for i in range(self.n_layers):
            # 1. Re-size to [N, 256, h_i, w_i]
            x_lateral = self.conv1[i](x[i+1])
            
            # 2. Rescale (h_i, w_i)
            x_top = self.upsampler[i](out)
            
            # 3. Append lateral and top connections
            out = x_lateral + x_top
            
            # 4. Perform convolution on output
            out = self.conv2[i](out)
            
            # 5. Retain Results at each layer
            p_results.append(out)
            
        # Need to verify that when results are added in array, information of the gradient is still not lost...
            
        print(">>> Final outcome:", out.size())
        return p_results


class Classifier(self):
    
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
    
class RetinaNet(nn.Module):
    """ 
    Ideally, it should extend to allow more multiple type of backbones.
    """
    
    def __init__(self):
        super(RetinaNet, self).__init__()
        
    def forward(self):
        pass
        

        
class ResNet(nn.Module):
    
    def __init__(self, x_dim, layers, block, num_classes):
        super(ResNet, self).__init__()
        
        self.in_plane = 64
        
        # Compute H & W at each layer
        x_dim = x_dim / 4.
        h_x, w_x = layers_dim( x_dim[0], x_dim[1], factor=2, n_layers=len(layers))
        
        self.conv1 = ConvBlock(3, 64, kernel=7, stride=2, pad=3, bias=False)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Needs to be converted toa for loop
        self.conv_2 = self._make_layers(block, layers[0], 64)        
        self.conv_3 = self._make_layers(block, layers[1], 128, stride=2)
        self.conv_4 = self._make_layers(block, layers[2], 256, stride=2)
        self.conv_5 = self._make_layers(block, layers[3], 512, stride=2)
        
        self.fpn = FPN([64, 128, 256], h_x, 512)
        
#         self.avgpool = nn.AvgPool2d(kernel_size=self.conv_5[layers[-1]])
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        
    def _make_layers(self, block, n_layers, out_plane, stride=1):
        
        downsample = None
        
        # Assume downsampling is performed by adding an extra layer to perform downsampling
        if self.in_plane != out_plane or block == Bottleneck:
            downsample = ConvBlock(self.in_plane, out_plane * block.expansion, kernel=1, stride=stride, pad=0, bias=False, is_relu=False)
        
        layers = [ block(self.in_plane, out_plane, stride, downsample) ]
        self.in_plane = out_plane * block.expansion
        for i in range(1, n_layers):
            layers.append( block(self.in_plane, out_plane) )
            
        return nn.Sequential(*layers)
        
        
    def forward(self, x):
        
        c_outputs = []
        
        x = self.conv1(x)
        x = self.maxpool(x)
        
        # This could be for-loop
        x1 = self.conv_2(x)
        x2 = self.conv_3(x1)
        x3 = self.conv_4(x2)
        x4 = self.conv_5(x3)
        
        # Reverse order for top-to-bottom pass
        features = self.fpn([x4,x3,x2,x1])
        
        # Regression-Box
        
        return out
    
    
if __name__ == '__main__':
    print(">>> Start...")
    
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    
    x = torch.randn((10, 3, 224, 224)).cuda()
    x_dim = np.array([x.size(2), x.size(3)])
    layers = [2, 3, 6, 4]
    
    model = ResNet(x_dim, layers, BasicBlock, 1000).cuda()
    model.forward(x)