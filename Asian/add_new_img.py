import os
import shutil
import cv2
import numpy as np
from skimage.util import random_noise
from skimage.filters import gaussian

root_path = '../team/cropped_classmate/'
imgs = os.listdir(root_path)
idx = 500
out_path = '../CROP_DB/test/'

for img in imgs:
    if img.split('.')[1] != 'png':
        continue
    img_path = root_path + img
    out_img = out_path + str(idx)
    idx += 1
    os.mkdir(out_img)
    #shutil.copy(img_path, out_img)
    
    imgg = cv2.imread(img_path)

    #filtered_img = gaussian(imgg, sigma=1, multichannel=True)
    #filtered_img = random_noise(imgg,'gaussian)
    filtered_img = random_noise(imgg,'s&p')

    noise_img = np.array(255*filtered_img, dtype = 'uint8')
 
    cv2.imwrite(out_img + '/s&p_'+img,noise_img)
    
