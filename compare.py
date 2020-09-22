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


class FaceRecog():
    def __init__(self):
        self.db_path = './db/'
        self.batch_size = 32
        self.workers = 16
        self.gallery_file = './search_gallery_1.csv'

        def load_meta_data(meta_file):
            #read csv file
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
            return meta_data

        def load_feat():
            #load the feature of gallery set
            mid_feats = dict()
            feat_dim = 256
            feas = os.listdir(self.db_path)
            for fea in feas:
                fea_p = self.db_path +  fea
                temp = np.load(fea_p)
                
                idd = int(fea.split('.')[0])
                mid_feats[idd] = temp
                    
            return mid_feats, feat_dim

        def update_meta_data(meta_data, feats, feat_dim):
            new_meta_data = dict()
            for template in meta_data.keys():
                class_id, img_names = meta_data[template]
                feat = feats[template]
                new_meta_data[template] = (class_id, feat, img_names[0])
            return new_meta_data

        feat_dim = 0
        #load csv file
        self.gallery_data = load_meta_data(self.gallery_file)

        #load the gallery set
        mid_feats, feat_dim = load_feat()
        self.gallery_data = update_meta_data(self.gallery_data, mid_feats, feat_dim)

    


    def get_top(self,probe, gallery_data):
        #cal the cosine dist and return the top 5 results
        score_info = list()
        probe_feat = probe[0]
        for template in gallery_data.keys():
            gallery_id, gallery_feat, gallery_img_path = gallery_data[template]
            score = 1-spatial.distance.cosine(probe_feat, gallery_feat)
            #score = np.sqrt(np.sum(np.square(probe_feat - gallery_feat)))
            score_info.append((gallery_id,gallery_img_path,score))
        
        score_info = sorted(score_info, key=lambda a:a[2], reverse=True)
        #score_info = sorted(score_info, key=lambda a:a[1])
        top5_id = [score_info[0][0],score_info[0][1]]
        return top5_id
        
    def extract_feat(self, model_path, img_path):
        class_num = 500

        model = resnet18(pretrained=False, num_classes=class_num, \
                    extract_feature=True, end2end=True)
        model = torch.nn.DataParallel(model)
        model.eval()
        
        #load model
        checkpoint = torch.load(model_path,map_location='cpu')
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        for key in pretrained_state_dict:
            if key in model_state_dict:
                model_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(model_state_dict)

        #preprocess img
        caffe_crop = CaffeCrop('test')
        img_dir = './'
        img_dataset =  CFPDataset(img_path,
                transforms.Compose([caffe_crop,transforms.ToTensor()]))
        img_loader = torch.utils.data.DataLoader(
                img_dataset,
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.workers, pin_memory=True)

        data_num = len(img_dataset)
        feat_dim = 256
        
        #get img output
        for i, (input, yaw) in enumerate(img_loader):
            yaw = yaw.float()
            yaw_var = torch.autograd.Variable(yaw)
            input_var = torch.autograd.Variable(input, volatile=True)
            output = model(input_var, yaw_var)
            output_data = output.cpu().data.numpy()
        return output_data  
            

    def test_recog(self,img_path,model_path):
        top1s, top5s = list(), list()
        
        #get the img feature
        probe_feat = self.extract_feat(model_path, img_path)
        
        #get the result
        top5_id = self.get_top(probe_feat, self.gallery_data)
        
        return top5_id

    def run(self,img_path):
        model_path = './model_best.pth.tar'
        print(img_path)
        time_start=time.time()

        top5 = self.test_recog(img_path, model_path)
        print(top5)

        time_end=time.time()
        print('time cost',time_end-time_start,'s')



if __name__ == '__main__':
    imgs = os.listdir('./cropped_probe/')
    f = FaceRecog()
    #imgs = ['LinXi.png','LiuChenxi.png','CaoWen.png','Zhengkai.png']
    for i in imgs:
        img_path = './cropped_probe/' + i
        f.run(img_path)
        #break


