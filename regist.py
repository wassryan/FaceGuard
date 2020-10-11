import os
from tqdm import tqdm
from src.register import register

if __name__ == '__main__':
    resume = './lib/070.ckpt'
    pkl_path = './data/db_features.pkl'
    gallery_path = './data/cropped_gallery/'
    imgs = os.listdir(gallery_path)
    for img in tqdm(imgs):
        img_path = gallery_path + img

        r = register(resume, gpu=False)
        r.run(pkl_path, img_path)