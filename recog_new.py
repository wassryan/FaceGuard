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
import time

class recog():
    def __init__(self,resume,gpu=True):
        self.net = model.MobileFacenet()
        if gpu:
            self.net = self.net.cuda()
        if resume:
            ckpt = torch.load(resume)
            self.net.load_state_dict(ckpt['net_state_dict'])
        self.net.eval()
    
    def run(self,feature_save_dir,cur_img,resume=None, gpu=True):
        result = scipy.io.loadmat(feature_save_dir)
        featureRs = result['fr']
        nr = result['nr']

        nl_new = [cur_img]
        lfw_dataset = LFW(nl_new, nl_new)
        lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=1,
                                              shuffle=False, num_workers=8, drop_last=False)

        featureLs = None
        count = 0

        for data in lfw_loader:
            if gpu:
                for i in range(len(data)):
                    data[i] = data[i].cuda()
            count += data[0].size(0)
        
            res = [self.net(d).data.cpu().numpy()for d in data]
            featureL = np.concatenate((res[0], res[1]), 1)
            featureR = np.concatenate((res[2], res[3]), 1)
        
            if featureLs is None:
                featureLs = featureL
            else:
                featureLs = np.concatenate((featureLs, featureL), 0)
        length = featureRs.shape[0]
        for _ in range(length-1):
            featureLs = np.concatenate((featureLs, featureL), 0)
        
        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        max_idx = np.where(scores == np.max(scores))

        return np.max(scores), nr[max_idx[0][0]][0]
    



if __name__ == '__main__':
    
    resume = './model/Asian/070.ckpt'
    feature_save_dir = './best_result.mat'
    #cur_img = '/home/lin/Desktop/caowen.jpg'
    probe_path = './data/cropped_probe/'

    imgs = os.listdir(probe_path)

    for img in imgs:
        start_time = time.time()
        cur_img = probe_path + img
        rec = recog(resume)
        scores,matched_img = rec.run(feature_save_dir, cur_img)
        result = [cur_img,scores,matched_img]
        end_time = time.time()
        print('time used: ' + str(end_time - start_time))
        print(result)
    
