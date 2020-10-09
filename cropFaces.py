import dlib         
import numpy as np  
import cv2          
import os

def crop(img_path,path_save,dlib_path):
    img = cv2.imread(img_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_path)
    
    faces = detector(img, 1)
    for k, d in enumerate(faces):
        pos_start = tuple([d.left(), d.top()])
        pos_end = tuple([d.right(), d.bottom()])

        height = d.bottom()-d.top()
        width = d.right()-d.left()

        height += 20
        width = round(96*height/112)

        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            for j in range(width):
                img_blank[i][j] = img[d.top()-20+i][d.left()+j]

        img_name = img_path.split('/')[-1]
        if not os.path.isdir(path_save):
            os.makedirs(path_save)
        out_path = path_save + img_name
        img_blank = cv2.resize(img_blank,(96,112)) 
        cv2.imwrite(out_path, img_blank)


#img need to be crop
img_path = '/home/lin/Desktop/caowen.jpg'
#model path
dlib_path = './models/shape_predictor_68_face_landmarks.dat'
#where to store the img(just the dir, will store with the same name as the img_path)
out_path = './'

crop(img_path,out_path,dlib_path)
