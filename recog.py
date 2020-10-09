import os
import sys
from test_face import getFeatureFromTorch,evaluation_10_fold
import time

class recog():
    def __init__(self):
        self.gallery_path = './data/cropped_gallery/'
        self.lfw_dir = './'
        self.resume = './model/Asian+friends/070.ckpt'
        self.feature_save_dir = './result/best_result.mat'

    def create_pairs(self, cur_img):
        gallerys = os.listdir(self.gallery_path)
        if os.path.exists('./pairs.txt'):
            os.remove('./pairs.txt')

        for classmate in gallerys:
            img_path = self.gallery_path + classmate
            with open('pairs.txt','a+') as f:
                f.write(cur_img + ' ' + img_path + '\n')
 
    def get_matched(self,cur_img,):
        self.create_pairs(cur_img)

        nr = getFeatureFromTorch(self.lfw_dir, self.feature_save_dir, self.resume)
        score,matched = evaluation_10_fold(nr,self.feature_save_dir) 
        return [cur_img,score,matched]
        
matched_path = './data/cropped_probe/'
tests = os.listdir(matched_path)
r = recog()
for l_img in tests:
    time_start = time.time()

    cur_img = matched_path + l_img
    result = r.get_matched(cur_img)
    print(result)

    time_end = time.time()
    print('time cost: ' + str(time_end - time_start))


