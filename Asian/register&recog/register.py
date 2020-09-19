import argparse
import os,sys,shutil
import time
import struct as st
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from selfDefine_1 import CFPDataset, CaffeCrop

from ResNet import resnet18, resnet50, resnet101


#from test_recog import test_recog
#from test_verify import test_verify


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./data/resnet18.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--model_dir','-m', default='./model', type=str)



def extract_feat(arch, model_path, yaw_type,img_path):
    global args, best_prec1
    args = parser.parse_args()

    if arch.find('end2end')>=0:
        end2end=True
    else:
        end2end=False

    arch = arch.split('_')[0]
  
    #class_num = 87020
    class_num = 500
    #class_num = 13386
    
    model = None
    assert(arch in ['resnet18','resnet50','resnet101'])
    if arch == 'resnet18':
        model = resnet18(pretrained=False, num_classes=class_num, \
                extract_feature=True, end2end=end2end)
    if arch == 'resnet50':
        model = resnet50(pretrained=False, num_classes=class_num, \
                extract_feature=True, end2end=end2end)
    if arch == 'resnet101':
        model = resnet101(pretrained=False, num_classes=class_num, \
                extract_feature=True, end2end=end2end)

    #model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    
    assert(os.path.isfile(model_path))
    checkpoint = torch.load(model_path,map_location='cpu')
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
    for key in pretrained_state_dict:
        if key in model_state_dict:
            model_state_dict[key] = pretrained_state_dict[key]
    model.load_state_dict(model_state_dict)

    print('load trained model complete')

    caffe_crop = CaffeCrop('test')

    
    img_dir = './'
    img_dataset =  CFPDataset(img_path,
            transforms.Compose([caffe_crop,transforms.ToTensor()]))
    img_loader = torch.utils.data.DataLoader(
            img_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    data_num = len(img_dataset)
    img_feat_file = img_dir + 'resnet18_gallery_feat.bin'
    #print(img_feat_file)
    feat_dim = 256
    with open(img_feat_file, 'ab+') as bin_f:
        bin_f.write(st.pack('ii', data_num, feat_dim))
        for i, (input, yaw) in enumerate(img_loader):
            #print(i)
            #print(input)
            #print(yaw[0])
            yaw = yaw.float()
            #print(i)
            #yaw = torch.as_tensor(float(yaw[0]))
            yaw_var = torch.autograd.Variable(yaw)
            input_var = torch.autograd.Variable(input, volatile=True)
            output = model(input_var, yaw_var)
            output_data = output.cpu().data.numpy()
            print(output_data.shape)
            feat_num  = output.size(0)
            
            for j in range(feat_num):
                bin_f.write(st.pack('f'*feat_dim, *tuple(output_data[j,:])))

def get_last_line(filename='./search_gallery_1.csv'):
    if not os.path.exists(filename):
        return -1

    with open(filename, 'r') as f:
        lines = f.readlines() # 批量
        lastline = lines[-1]
    return lastline

    
def write_csv(img_path,csv_file):
    if get_last_line() == -1:
        with open(csv_file,'a+') as f:
            csv_write = csv.writer(f)
            data_row = [1,1,img_path]
            csv_write.writerow(data_row)
            data_row = [1,1,img_path]
            csv_write.writerow(data_row)
    else:

        template,name_id = get_last_line().split(',')[:2]

        with open(csv_file,'a+') as f:
            csv_write = csv.writer(f)
            data_row = [int(template)+1,int(name_id)+1,img_path]
            csv_write.writerow(data_row)


if __name__ == '__main__':

    arch = 'resnet18_end2end'
    model_path = '/Volumes/FreeAgent G/finetune/model/model_crop.pth.tar'
    yaw_type = 'nonli'
    img_path = '../DREAM/data/team/gallery_cropped/liuchenxi2.jpg'

    print(model_path.split('/')[-1])
    time_start=time.time()
    extract_feat(arch, model_path, yaw_type,img_path)
    write_csv(img_path,'./search_gallery_1.csv')
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
        
