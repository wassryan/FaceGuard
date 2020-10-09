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
import argparse

def parseList(root):
    with open('pairs.txt') as f:
        pairs = f.read().splitlines()[1:]
    #folder_name = 'lfw-112X96'
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        nameL,nameR =  p.split(' ')

        nameLs.append(nameL)
        nameRs.append(nameR)
        folds.append(i)
        flags.append(1)
    #print(nameLs)
    return [nameLs, nameRs, folds, flags]



def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])

    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold


def evaluation_10_fold(nr,root='./result/pytorch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(root)

    fold = result['fold']
    flags = result['flag']
    featureLs = result['fl']
    featureRs = result['fr']

    flags = np.squeeze(flags)

    scores = np.sum(np.multiply(featureLs, featureRs), 1)
    max_idx = np.where(scores == np.max(scores))
    #print(np.max(scores))
    #print(max_idx)
    #print('find matched in gallery: ' + nr[max_idx[0][0]])

    return np.max(scores), nr[max_idx[0][0]]



def getFeatureFromTorch(lfw_dir, feature_save_dir, resume=None, gpu=True):
    net = model.MobileFacenet()
    if gpu:
        net = net.cuda()
    if resume:
        ckpt = torch.load(resume)
        net.load_state_dict(ckpt['net_state_dict'])
    net.eval()
    nl, nr, flods, flags = parseList(lfw_dir)
    lfw_dataset = LFW(nl, nr)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=1,
                                              shuffle=False, num_workers=8, drop_last=False)

    featureLs = None
    featureRs = None
    count = 0

    for data in lfw_loader:
        if gpu:
            for i in range(len(data)):
                data[i] = data[i].cuda()
        count += data[0].size(0)
        
        res = [net(d).data.cpu().numpy()for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        

    result = {'fl': featureLs, 'fr': featureRs, 'fold': flods, 'flag': flags}
    scipy.io.savemat(feature_save_dir, result)
    return nr



if __name__ == '__main__':
    
    lfw_dir = LFW_DATA_DIR
    resume = './model/best/068.ckpt'
    feature_save_dir = './result/best_result.mat'
    nr = getFeatureFromTorch(lfw_dir, feature_save_dir, resume)
    evaluation_10_fold(nr,feature_save_dir)
    
