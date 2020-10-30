import os
import random
import sys

#root_path = './Asian/left_20/test/'
root_path = sys.argv[1]
#root_path = '/home/lin/Desktop/classmate_cropped/'
subs = os.listdir(root_path)

pos_pairs = []
neg_pairs = []
lock = 0


for sub in subs:
	#print(sub)
	#postive
	sub_p = root_path + sub + '/'
	imgs = os.listdir(sub_p)
	if len(imgs) > 1:
		pos_pairs += [(sub_p + imgs[i], sub_p + imgs[j])
                 		for i in range(len(imgs))
                 		for j in range(i + 1, len(imgs))]
	
	#negative
	temp = [i for i in subs]
	temp.remove(sub)
	#print(temp)
	
	for s in temp:
		temp_p = root_path + s + '/'
		for img1 in imgs:
			imgs2 = os.listdir(temp_p)
                        
			for img2 in imgs2:
				neg_pairs += [(sub_p + img1, temp_p + img2)]
			

random.shuffle(pos_pairs)
random.shuffle(neg_pairs)
neg_pairs = neg_pairs[:len(pos_pairs)]
if os.path.exists('pairs_train.txt'):
    os.remove('pairs_train.txt')

with open('pairs_train.txt','a+') as f:
	for img1, img2 in pos_pairs:
		f.write(img1 + ' ' + img2 + ' 1\n')
	for img1, img2 in neg_pairs:
		f.write(img1 + ' ' + img2 + ' -1\n')

	
