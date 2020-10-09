import sys
# import caffe
import os
import numpy as np
import cv2
import scipy.io
import copy
import core.model
import os
import torch.utils.data
from core import model
from dataloader.LFW_loader import LFW
from config import LFW_DATA_DIR

class register():

    def __init__(self,resume, gpu=True):
        self.net = model.MobileFacenet()
        if gpu:
            self.net = self.net.cuda()
        if resume:
            ckpt = torch.load(resume)
            self.net.load_state_dict(ckpt['net_state_dict'])
        self.net.eval()
        
    def run(self,root,new_img,gpu=True):
        if os.path.exists(root):
            result = scipy.io.loadmat(root)
            featureRs = result['fr']
            nr = result['nr']
        else:
            featureRs = None
            nr = None

        nr_new = [new_img]
        nl_new = [new_img]
        lfw_dataset = LFW(nl_new, nr_new)
        lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=1,
                                              shuffle=False, num_workers=8, drop_last=False)    

        count = 0
        for data in lfw_loader:
            if gpu:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            count += data[0].size(0)
        
            res = [self.net(d).data.cpu().numpy()for d in data]
        
            featureR = np.concatenate((res[2], res[3]), 1)
            nr_new = np.array([nr_new])
        
            if featureRs is None:
                featureRs = featureR
            else:
                featureRs = np.concatenate((featureRs, featureR), 0)
            if nr is None:
                nr = nr_new
            else:
                nr = np.concatenate((nr, nr_new), 0)
        

        result = {'fr': featureRs,'nr': nr}
        scipy.io.savemat(root, result)
    



if __name__ == '__main__':
    resume = './model/Asian/070.ckpt'
    root = './best_result.mat'
    gallery_path = './data/cropped_gallery/'
    imgs = os.listdir(gallery_path)
    for img in imgs:
        new_img = gallery_path + img

        r = register(resume)
        r.run(root,new_img)
    '''
    result = scipy.io.loadmat('./best_result.mat')

    featureRs = result['fr']
    nr = result['nr']
    print(featureRs)
    print(nr)
    '''
