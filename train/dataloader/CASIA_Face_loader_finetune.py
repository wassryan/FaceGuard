import numpy as np
import scipy.misc
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

class CASIA_Face(object):
    def __init__(self, root):
        self.root = root

        img_txt_dir = os.path.join(root, 'classmate_112x96.txt')
        image_list = []
        label_list = []
        with open(img_txt_dir) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_dir, label_name = info.split(' ')
            image_list.append(image_dir)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        # img = scipy.misc.imread(img_path)
        img = Image.open(img_path)
        img = transforms.RandomApply([transforms.ColorJitter(0.2, 0.5, 0.5, 0.2)], p=0.2)(img)
        img = transforms.RandomRotation((-15,15))(img)
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = transforms.RandomHorizontalFlip()(img)

        # png_name = img_path.split('/')[-1]
        # if not (png_name.startswith('blur') or png_name.startswith('s&p') or png_name.startswith('gaussian')):
        #     img.show()
        img = np.array(img)

        # if len(img.shape) == 2:
        #     img = np.stack([img] * 3, 2)
        # flip = np.random.choice(2)*2-1
        # img = img[:, ::flip, :] # (H,W,C)
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1) # (C,H,W)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
    # data_dir = '/home/brl/USER/fzc/dataset/CASIA'
    data_dir = './'
    dataset = CASIA_Face(root=data_dir)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for i, data in enumerate(trainloader):
        print(data[0].shape)
