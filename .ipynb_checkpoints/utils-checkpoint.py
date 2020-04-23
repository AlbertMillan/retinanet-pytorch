import torch.nn as nn
import torch.nn.functional as F
import torch

import sys

class ConvBlock(nn.Module):
    """ Block consisting of convolutions, batchnorm & relu (if any) layers"""
    def __init__(self, in_plane, out_plane, kernel, stride=1, pad=1, bias=True, is_bn=True, is_relu=True):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=kernel, stride=stride, padding=pad, bias=bias)
        
        self.bn = None
        if is_bn:
            self.bn = nn.BatchNorm2d(out_plane)
    
        self.relu = None
        if is_relu:
            self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        out = self.conv(x)
        
        if self.bn:
            out = self.bn(out)
        
        if self.relu:
            out = self.relu(out)
        
        return out
    
    
        

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_plane, out_plane, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(in_plane, out_plane, kernel=3, stride=stride, pad=1)
        self.conv2 = ConvBlock(out_plane, out_plane, kernel=3, is_relu=False)
        self.downsample = downsample
        
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample:
            residual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        
        return out
    
    
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_plane, out_plane, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBlock(in_plane, out_plane, kernel=1, stride=1, pad=0, bias=False)
        self.conv2 = ConvBlock(out_plane, out_plane, kernel=3, stride=stride, pad=1, bias=False)
        self.conv3 = ConvBlock(out_plane, out_plane * 4, kernel=1, pad=0, is_relu=False, bias=False)
        self.downsample = downsample
        
        
    def forward(self, x):
        residual = x
        
#         print(">>> Forward:", x.size())
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.downsample:
#             print(x.size())
            residual = self.downsample(x)
#             print(out.size())
#             print(residual.size())
            
        out += residual
        out = F.relu(out)
        
        return out
    
    