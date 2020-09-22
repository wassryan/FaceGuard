import argparse
import os,sys,shutil
import time
import struct as st
import csv
import numpy as np

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


class Register():
    def __init__(self):
        
        self.batch_size = 32
        self.workers = 16
        if not os.path.exists('./db/'):
            os.makedirs('./db/')

    def extract_feat(self, model_path, img_path,csv_file):
        #get the img's feature
        class_num = 500    
        model = resnet18(pretrained=False, num_classes=class_num, extract_feature=True, end2end=True)

        model = torch.nn.DataParallel(model)
        model.eval()

        #load model
        assert(os.path.isfile(model_path))
        checkpoint = torch.load(model_path,map_location='cpu')
        pretrained_state_dict = checkpoint['state_dict']
        model_state_dict = model.state_dict()
        for key in pretrained_state_dict:
    	    if key in model_state_dict:
    	        model_state_dict[key] = pretrained_state_dict[key]
        model.load_state_dict(model_state_dict)

        print('load trained model complete')

        # preprocess img
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
        #get the latest id
        template,name_id = self.get_last_line(csv_file).split(',')[:2]
        
        #put the img into the model and get the feature, store it to db/id.npy
        for i, (input, yaw) in enumerate(img_loader):
    	    yaw = yaw.float()
    	    yaw_var = torch.autograd.Variable(yaw)
    	    input_var = torch.autograd.Variable(input, volatile=True)
    	    output = model(input_var, yaw_var)
    	    output_data = output.cpu().data.numpy()
    	    #print(output_data[0,:])
    	    feat_num  = output.size(0)
    	    save_path = './db/' + str(name_id) + '.npy'
    	    np.save(save_path,output_data[0,:])
            
             
        #get the latest id to identify the new person
    def get_last_line(self,filename):
        if not os.path.exists(filename):
            return -1

        with open(filename, 'r') as f:
            lines = f.readlines() 
            lastline = lines[-1]
        return lastline

        #write to the csv file to map the img_path with its id
    def write_csv(self,img_path,csv_file):
        if self.get_last_line(csv_file) == -1:
            with open(csv_file,'a+') as f:
                csv_write = csv.writer(f)
                data_row = [1,1,img_path]
                csv_write.writerow(data_row)
                data_row = [1,1,img_path]
                csv_write.writerow(data_row)
        else:

            template,name_id = self.get_last_line(csv_file).split(',')[:2]
            with open(csv_file,'a+') as f:
                csv_write = csv.writer(f)
                data_row = [int(template)+1,int(name_id)+1,img_path]
                csv_write.writerow(data_row)

    def run(self,img_path):
        model_path = './model_best.pth.tar'
        csv_file = './search_gallery_1.csv'

        time_start=time.time()

        self.write_csv(img_path,csv_file)
        self.extract_feat(model_path, img_path, csv_file)
        
        time_end=time.time()
        print('time cost',time_end-time_start,'s')



if __name__ == '__main__':
    imgs = os.listdir('./cropped_gallery/')
    r = Register()

    for i in imgs:
        img_path = './cropped_gallery/' + i
        r.run(img_path)
        
