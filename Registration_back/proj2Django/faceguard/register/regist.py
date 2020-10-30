import os
from tqdm import tqdm
from .register import register
current_path = os.path.dirname(__file__)

save_path = '/Users/liuchenxi/Desktop/faceguard_registration/data/'
gallery_path = '/Users/liuchenxi/Desktop/faceguard_registration/data/cropped_gallery/'
def user_reg(img):
    resume = current_path + '/lib/070.ckpt'
    pkl_path = save_path + 'db_features.pkl'
    #imgs = os.listdir(gallery_path)
    # for img in tqdm(imgs):
    img_path = gallery_path + img
    r = register(resume, gpu=False)
    r.run(pkl_path, img_path)
    print("registration succesully")