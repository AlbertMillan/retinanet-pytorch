import argparse

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import sys, os, time

from data_loader import CocoDataset, Normalizer, Augmenter, Resizer, collater, AspectRatioSampler
from retinanet import resnet18, resnet50, resnet101, resnet152

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

        
        
def get_model(depth):
    if depth == 18:
        return resnet18(80)
    
    elif depth == 34:
        return resnet34(80)
    
    elif depth == 50:
        return resnet50(80)
    
    elif depth == 101:
        return resnet101(80)
    
    elif depth == 152:
        return resnet152(80)
    
    else:
        print("Model not found. Exiting...")
        sys.exit(1)
        
        
def save_checkpoint(self, epoch, model_state, optimizer_state, loss, save_dir, base_name="chkpt"):
    """Saves checkpoint to disk"""
    directory = save_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + base_name + ".pth.tar"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'loss': loss
    }, filename)
#     if is_best:
#         shutil.copyfile(filename, directory + base_name + '__model_best.pth.tar')

        
        
if __name__ == '__main__':
    
    
    
    parser = argparse.ArgumentParser(description='Script for training a RetinaNet network.')
    
    parser.add_argument('--coco_path', default='datasets/COCO', type=str, help='Path to COCO directory')
    parser.add_argument('--coco_name', default='train2017', type=str, help='Name of directory containing images')
    
    parser.add_argument('--save_dir', default='chkpt/', type=str, help='Path to save checkpoint files.')
    
    parser.add_argument('--mal', help='Apply plain or MAL retinanet.', type=int, default=0)
    parser.add_argument('--bag_size', help='Initial bag size (k)', type=int, default=50)
    
    
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
    parser.add_argument('--batch_size', default=5, help='Batch size', type=int)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--print_freq', default=50, help='Frequency to print training info...')
    parser.add_argument('--gpus', default="0,1,2,3", help='GPU devices to use in string format, comma separated.', type=str)
        
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

    parallel = (torch.cuda.device_count() > 1)
    
    dataset_train = CocoDataset(args.coco_path, args.coco_name,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    
    sampler = AspectRatioSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
    
    data_loader = DataLoader(dataset_train, collate_fn=collater, batch_sampler=sampler)
    
    # Create Model Instance
    model = get_model(args.depth)
    
    
    if not parallel:
        model = model.cuda()
        if args.mal:
            model.focalLoss.set_MALnet(args.epochs, k=args.bag_size)
        print("Loading to GPU...")
    elif parallel:
        model = torch.nn.DataParallel(model).cuda()
        if args.mal:
            model.module.focalLoss.set_MALnet(args.epochs, k=args.bag_size)
        print("Loading to multiple GPUs...")
        
        
    cls_loss_hist = []
    rgs_loss_hist = []
    loss_hist = []
    
    
#     for i in range(args.epochs):
    
# #         optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
        
#         optimizer = optim.Adam(model.parameters(), lr=1e-5)
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        
#         if args.mal:
#             if parallel:
#                 model.module.focalLoss.update_bag_size(i)
#             else:
#                 model.focalLoss.update_bag_size(i)
        
#         cls_loss = AverageMeter()
#         rgs_loss = AverageMeter()
#         total_loss = AverageMeter()
#         batch_time = AverageMeter()
        
#         end = time.time()

#         for j, batch in enumerate(data_loader):
#             optimizer.zero_grad()
#             classification_loss, regression_loss = model.forward((batch['img'].cuda(), batch['annot'].cuda()))
#             loss = classification_loss.mean() + regression_loss.mean()

#             cls_loss.update(classification_loss.mean().item())
#             rgs_loss.update(regression_loss.mean().item())
#             total_loss.update(loss.item())
            
#             loss.backward()

#             optimizer.step()
            
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if (j % args.print_freq) == 0:
#                 print('Epoch: [{0}][{1}/{2}] | '
#                       'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
#                       'CLS Loss: {cls.val:.3f} ({cls.avg:.3f}) | '
#                       'RGS Loss: {rgs.val:.3f} ({rgs.avg:.3f}) | '
#                       'Running Loss: {total_ls.val:.3f} ({total_ls.avg:.3f})'
#                       .format(
#                           i, j, len(data_loader),
#                           batch_time = batch_time,
#                           cls=cls_loss,
#                           rgs=rgs_loss,
#                           total_ls=total_loss))
            
#             del classification_loss
#             del regression_loss
        
#         model_dict = None
#         if parallel:
#             model_dict = model.module.state_dict()
#         else:
#             model_dict = model.state_dict()
            
#         cls_loss_hist.append(cls_loss.avg)
#         rgs_loss_hist.append(rgs_loss.avg)
#         loss_hist.append(total_loss.avg)
            
#         scheduler.step(total_loss.avg)
            
#         save_checkpoint(True, i, model_dict, optimizer.state_dict(), loss.item(), args.save_dir)
    
    # Store loss numpy arrays
    if not os.path.exists('results/'):
        os.makedirs('results/')
    np.save('results/cls_loss_hist.np', cls_loss_hist)
    np.save('results/rgs_loss_hist.np', rgs_loss_hist)
    np.save('results/loss_hist.np', loss_hist)
    