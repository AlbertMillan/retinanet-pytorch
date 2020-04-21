import torch.nn as nn
import torch
from utils import ConvBlock, BasicBlock, Bottleneck

import sys, os

class FPN(nn.Module):
    
    def __init__(self):
        super(FPN, self).__init__()
        
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
    
    def __init__(self, layers, block, num_classes):
        super(ResNet, self).__init__()
        
        self.in_plane = 64
        
        self.conv1 = ConvBlock(3, 64, kernel=7, stride=2, pad=3, bias=False)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv_2 = self._make_layers(block, layers[0], 64)        
        self.conv_3 = self._make_layers(block, layers[1], 128, stride=2)
        self.conv_4 = self._make_layers(block, layers[2], 256, stride=2)
        self.conv_5 = self._make_layers(block, layers[3], 512, stride=2)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7)
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
        
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.conv_5(out)
        out = self.avgpool(out).view(x.size(0), -1)
        out = self.fc(out)
        
        return out
    
    
if __name__ == '__main__':
    print(">>> Start...")
    
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    
    x = torch.randn((10, 3, 224, 224)).cuda()
    layers = [2, 3, 6, 4]
    
    model = ResNet(layers, Bottleneck, 1000).cuda()
    model.forward(x)