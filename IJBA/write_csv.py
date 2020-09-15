import csv
import os

root_path = '../../data/team/'
img_type = ['gallery','probe']
idx = 1
for t in img_type:
	img_path = root_path + t
	imgs = os.listdir(img_path)
	dst_path = root_path + 'search_' + t + '_1.csv'
	for img in imgs:
		if img.split('.')[1] != 'jpg':
			continue

		#python2可以用file替代open
		imgg = img_path + '/' + img
		print(imgg)
		with open(dst_path,"a") as csvfile: 
		    writer = csv.writer(csvfile)
		    #先写入columns_name
		    writer.writerow([str(idx),str(idx),str(imgg)])
		idx += 1
		    