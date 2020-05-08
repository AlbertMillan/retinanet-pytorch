import argparse

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import sys, os, time

from data_loader import CocoDataset, Normalizer, Augmenter, Resizer, collater, AspectRatioSampler
from retinanet import resnet18


class AverageMeter():
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
    
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
#     model = torch.nn.DataParallel(model).cuda()
    
#     model.training = False
    
    for i in range(20):
    
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        
        cls_loss = AverageMeter()
        rgs_loss = AverageMeter()
        total_loss = AverageMeter()
        batch_time = AverageMeter()
        
        end = time.time()

        for j, batch in enumerate(data_loader):
            optimizer.zero_grad()
            classification_loss, regression_loss = model.forward((batch['img'].cuda(), batch['annot'].cuda()))
            
            loss = classification_loss + regression_loss

            cls_loss.update(classification_loss.item())
            rgs_loss.update(regression_loss.item())
            total_loss.update(loss.item())
            
            loss.backward()

            optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}] | '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                  'CLS Loss: {cls.val:.3f} ({cls.avg:.3f}) | '
                  'RGS Loss: {rgs.val:.3f} ({rgs.avg:.3f}) | '
                  'Running Loss: {total_ls.val:.3f} ({total_ls.avg:.3f})'
                  .format(
                      i, j, len(data_loader),
                      batch_time = batch_time,
                      cls=cls_loss,
                      rgs=rgs_loss,
                      total_ls=total_loss))
            
            del classification_loss
            del regression_loss
        
    sys.exit()