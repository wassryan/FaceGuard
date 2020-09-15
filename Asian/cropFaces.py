import dlib         
import numpy as np  
import cv2          
import os

def crop(img_path,path_save,dlib_path):
    img = cv2.imread(img_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_path)
    try:
        faces = detector(img, 1)
        for k, d in enumerate(faces):
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            height = d.bottom()-d.top()
            width = d.right()-d.left()

            img_blank = np.zeros((height, width, 3), np.uint8)

            for i in range(height):
                for j in range(width):
                        img_blank[i][j] = img[d.top()+i][d.left()+j]

            img_name = img_path.split('/')[-1]
            if not os.path.isdir(path_save):
                os.makedirs(path_save)
            out_path = path_save + img_name
            #print("Save to:", path_save+"img_face_"+str(k+1)+".jpg")
            print(out_path)
            cv2.imwrite(out_path, img_blank)
    except:
        print(img_path)

path_read = "/Volumes/FreeAgent G/64_CASIA-FaceV5/aligned/"
dlib_path = './models/shape_predictor_68_face_landmarks.dat'
subs = os.listdir(path_read)

for sub in subs:
    sub1 = sub
    if sub[0] == '0':
        sub1 = sub[1:]
    if sub1[0] == '0':
        sub1 = sub1[1:]
    sub_p = path_read + '/' + sub
    path_save = path_read + 'cropped/' + sub1 + '/'
    if os.path.exists(path_save):
        continue
    imgs = os.listdir(sub_p)
    for img in imgs:
        img_path = sub_p + '/' + img
        crop(img_path,path_save,dlib_path)

#img_path = './001_0.bmp'


