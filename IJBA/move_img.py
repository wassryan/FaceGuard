import os
import shutil

root_path = '../../data/IJBA/align_image_1N/split1/probe/'
target_path = '../../data/IJBA/align_image_1N/test/probe/'

if not os.path.exists(target_path):
    os.makedirs(target_path)

for i in range(1,1001):
	ori_img = root_path + str(i) + '.jpg'
	dst_img = target_path + str(i) + '.jpg'
	shutil.copy(ori_img,dst_img)
print('done')


