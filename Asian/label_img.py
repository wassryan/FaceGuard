import os

paths = [['/Volumes/FreeAgent G/64_CASIA-FaceV5/aligned/cropped/','train'],
		 ['/Volumes/FreeAgent G/64_CASIA-FaceV5/aligned/cropped/test/','test']]

for root_path, typ in paths:
	subs = os.listdir(root_path)
	for sub in subs:
		sub_p = root_path + sub
 		if sub == 'test' or not os.path.isdir(sub_p):
			continue
		imgs = os.listdir(sub_p)
		for img in imgs:
			if img[0] == '.' or img.split('.')[-1] != 'bmp':
				continue
			img_p = sub_p + '/' + img
			file_list = typ + '_list.txt'
			file_label = typ + '_label.txt'
			with open(file_list,'a+') as f:
				f.write(img_p + '\n')
			with open(file_label,'a+') as f:
				f.write(sub + ' 0.00\n')

