import argparse

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import sys, os

from data_loader import CocoDataset, Normalizer, Augmenter, Resizer, collater, AspectRatioSampler
from retinanet import get_model



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
    
    sampler = AspectRatioSampler(dataset_train, batch_size=3, drop_last=False)
    
    data_loader = DataLoader(dataset_train, collate_fn=collater, batch_sampler=sampler)
    
    # Create Model Instance
    model = get_model(80).cuda()
    
    for i, batch in enumerate(data_loader):
        
        loss = model.forward((batch['img'].cuda(), batch['annot'].cuda()))
        
        print("ITER COMPLETED")
        
        sys.exit()
    sys.exit()