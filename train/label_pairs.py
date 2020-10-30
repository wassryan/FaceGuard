import os
import sys

#root_path = '/home/lin/Desktop/Asian/left_20/train/'
root_path = sys.argv[1]
idxs = os.listdir(root_path)
idd = 0

if os.path.exists('classmate_112x96.txt'):
    os.remove('classmate_112x96.txt')

with open('classmate_112x96.txt','a+') as f:
	for idx in idxs:
		idx_path = root_path + idx
		imgs = os.listdir(idx_path)
		for img in imgs:
			img_path = idx_path + '/' + img
			f.write(img_path + ' ' + str(idd) + '\n')
		idd += 1
