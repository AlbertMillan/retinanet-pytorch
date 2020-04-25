import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import sys, os

from data_loader import CocoDataset, Normalizer, Augmenter, Resizer, collater, AspectRatioSampler
from retinanet import resnet18



if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    
    parser = argparse.ArgumentParser(description='Script for training a RetinaNet network.')
    
    parser.add_argument('--coco_path', default='datasets/COCO', type=str, help='Path to COCO directory')
    parser.add_argument('--coco_name', default='train2017', type=str, help='Name of directory containing images')
    
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    
    args = parser.parse_args()
    
    dataset_train = CocoDataset(args.coco_path, args.coco_name,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    
    sampler = AspectRatioSampler(dataset_train, batch_size=5, drop_last=False)
    
    data_loader = DataLoader(dataset_train, collate_fn=collater, batch_sampler=sampler)
    
    # Create Model Instance
    model = resnet18(80).cuda()
    
    for i, batch in enumerate(data_loader):
        classification_loss, regression_loss = model.forward((batch['img'].cuda(), batch['annot'].cuda()))

        print('Epoch: [{0}][{1}/{2}] \t'
              'Classification Loss: {3:.3f} \t'
              'Regression Loss: {4:.3f}'.format(
                  0, i, len(data_loader), 
                  classification_loss.item(),
                  regression_loss.item()))
        
    sys.exit()