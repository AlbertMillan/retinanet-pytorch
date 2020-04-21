import torch.nn as nn
import torch
from utils import ConvBlock, BasicBlock, Bottleneck

import sys

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
        super(FPN, self).__init__()
        
    def forward(self):
        pass
        

        
class ResNet(nn.Module):
    
    def __init__(self, layers, block):
        super(FPN, self).__init__()
        
        print(">>> In ResNet...")
        
        self.conv1 = ConvBlock(3, 64, kernel=7, stride=2, pad=3, bias=False)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv_2 = self._make_layers(self, block, layers[0], 64, 64)
        
        sys.exit()
        
        self.conv_3 = self._make_layers(self, block, layers[1], 64, 128)
        self.conv_4 = self._make_layers(self, block, layers[2], 128, 256)
        self.conv_5 = self._make_layers(self, block, layers[3], 256, 512)
        
        
    def _make_layers(self, block, n_layers, in_plane, out_plane, stride=1):
        
        downsample = None
        
        # Assume downsampling is performed by adding an extra layer to perform downsampling
        if in_plane != out_plane:
            downsample = ConvBlock(out_plane, out_plane, kernel=1, stride=stride, bias=False, is_relu=False)
        
        layers = [ block(in_plane, out_plane, stride, downsample) ]
        for i in range(1, n_layers):
            layers.append( block(in_plane, out_plane) )
            
        return nn.Sequential(layers)
        
    def forward(self):
        pass
    
    
if __name__ == '__main__':
    print(">>> Start...")
    
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    
    x = torch.randn((10, 3, 224, 224)).cuda()
    layers = [2, 3, 6, 4]
    
    model = ResNet(layers, BasicBlock).cuda()