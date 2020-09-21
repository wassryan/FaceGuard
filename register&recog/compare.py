import numpy as np
from scipy import spatial
from sklearn import metrics
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity

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


def load_meta_data(meta_file):
    meta_data = dict()
    with open(meta_file, 'r') as in_f:
        in_f.readline() # the first line is not data
        for idx, line in enumerate(in_f):
            record = line.strip().split(',')
            template, class_id = int(record[0]), int(record[1])
            img_path = record[2]
            if template not in meta_data:
                meta_data[template] = ( class_id, [img_path,] )
            else:
                meta_data[template][1].append(img_path)
        #print('load meta_dat')
        #print(meta_data)
    return meta_data

def get_last_line(filename='./search_gallery_1.csv'):
    with open(filename, 'r') as f:
        lines = f.readlines() # 批量
        lastline = lines[-1]
    return lastline

def load_feat(bin_file):
    mid_feats = dict()
    feat_dim = 256
    
    feas = os.listdir('./db/')
    for fea in feas:
        fea_p = './db/' +  fea
        temp = np.load(fea_p)
        #print(type(temp))
        idd = int(fea.split('.')[0])
        mid_feats[idd] = temp
    #print(mid_feats)
            
    return mid_feats, feat_dim

def update_meta_data(meta_data, feats, feat_dim):
    new_meta_data = dict()
    for template in meta_data.keys():
        class_id, img_names = meta_data[template]
        
        feat = feats[template]
        
        new_meta_data[template] = (class_id, feat)
    #print('update meta data:')
    #print(new_meta_data)
    return new_meta_data


def get_top(probe, gallery_data):
    score_info = list()
    probe_feat = probe[0]
    #print(gallery_data)
    for template in gallery_data.keys():
        gallery_id, gallery_feat = gallery_data[template]
        #print(gallery_feat)
        #print('*******************')
        #print(probe_feat)
        score = 1-spatial.distance.cosine(probe_feat, gallery_feat)
        #score = np.sqrt(np.sum(np.square(probe_feat - gallery_feat)))
        score_info.append((gallery_id, score))
    
    score_info = sorted(score_info, key=lambda a:a[1], reverse=True)
    #score_info = sorted(score_info, key=lambda a:a[1])
    #print('score:')
    #print(score_info[:5])
    top5_id = [item[0] for item in score_info[:5]]
    return top5_id
        
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

    #print('load trained model complete')

    caffe_crop = CaffeCrop('test')

    
    img_dir = './'
    img_dataset =  CFPDataset(img_path,
            transforms.Compose([caffe_crop,transforms.ToTensor()]))
    img_loader = torch.utils.data.DataLoader(
            img_dataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    data_num = len(img_dataset)
    #print(img_feat_file)
    feat_dim = 256
    
    for i, (input, yaw) in enumerate(img_loader):
        yaw = yaw.float()
        yaw_var = torch.autograd.Variable(yaw)
        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var, yaw_var)
        output_data = output.cpu().data.numpy()
    return output_data  
            

def test_recog(model_type,img_path):
    time_start=time.time()
      
    #model_type = 'resnet18'
    protocol_dir = './'
    top1s, top5s = list(), list()

    
    gallery_file = protocol_dir + 'search_gallery_1.csv'
    gallery_data = load_meta_data(gallery_file)
    #print(gallery_data)

    arch = 'resnet18_end2end'
    model_path = './model/model_best.pth.tar'
    yaw_type = 'nonli'
    

    probe_feat = extract_feat(arch, model_path, yaw_type,img_path)
    
    #feats = dict()
    feat_dim = 0
    
    bin_file = protocol_dir +  'resnet18_gallery_feat.bin'
    mid_feats, feat_dim = load_feat(bin_file)

    gallery_data = update_meta_data(gallery_data, mid_feats, feat_dim)
    #print(gallery_data)

    top5_id = get_top(probe_feat, gallery_data)

    time_end=time.time()
    
    return top5_id


if __name__ == '__main__':
    imgs = os.listdir('../../team/cropped_probe/')
    #imgs = ['LinXi.png','LiuChenxi.png','CaoWen.png','Zhengkai.png']
    for i in imgs:
        img_path = '../../team/cropped_probe/' + i
        print(img_path)
        top5_id = test_recog('resnet18',img_path)
        print(top5_id)
        #break


