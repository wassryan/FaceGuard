import os

root_path = '../../data/team/'
img_type = ['gallery','probe']

for t in img_type:
	paths = root_path + t
	imgs = os.listdir(paths)
	dst_path = paths + '/img_name.txt'
	with open(dst_path,'w+') as f:
		for img in imgs:
			if img.split('.')[1] != 'jpg':
				continue
			img_path = paths + '/' + img
			print(img_path)
			f.write(img_path + ' 0.00\n')

