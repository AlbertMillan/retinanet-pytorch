import torch.nn as nn
import torch.functional as F
import torch

class ConvBlock(nn.Module):
    """ Block consisting of convolutions, batchnorm & relu (if any) layers"""
    def __init__(self, in_plane, out_plane, kernel, stride=1, pad=1, bias=True, is_relu=True):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=kernel, stride=stride, padding=pad, bias=bias)
        self.bn = nn.BatchNorm(out_plane)
        
        if is_relu:
            self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        
        if self.relu:
            out = self.relu(out)
        
        return out
    
    
        

class BasicBlock(nn.Module):
    
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
            redidual = self.downsample(x)
            
        out += residual
        out = F.relu(out)
        
        return out
    
    
class Bottleneck(nn.Module):
    
    def __init__(self, in_planes, out_planes):
        super(Bottleneck, self).__init__()
        
        
    def forward(self, x):
        
        
        return out
    
    