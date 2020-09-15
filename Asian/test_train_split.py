import os
import shutil

root_path = '/Volumes/FreeAgent G/64_CASIA-FaceV5/aligned/cropped/'
dst_path = root_path + 'test/'
subs = os.listdir(root_path)

for sub in subs:
	img_p = root_path + sub
	imgs = os.listdir(img_p)
	if len(imgs) <= 1:
		continue
	src_img = img_p + '/' + imgs[0]

	dst_p = dst_path + sub 
	print(img_p)
	if not os.path.isdir(dst_p):
		os.makedirs(dst_p)
	dst_img = dst_p + '/' + imgs[0]
	shutil.move(src_img, dst_img)
