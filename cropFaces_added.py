import dlib         
import numpy as np  
import cv2          
import os

def crop(img_path,out_crop_path,out_detect_path,dlib_path):
    img = cv2.imread(img_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_path)
    
    faces = detector(img, 1)
    for k, d in enumerate(faces):

        height = d.bottom()-d.top()
        width = d.right()-d.left()

        height += 20
        width = round(96*height/112)

        pos_start = tuple([d.left(), d.top()-20])
        pos_end = tuple([d.left()+width, d.top()-20+height])

        

        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            for j in range(width):
                img_blank[i][j] = img[d.top()-20+i][d.left()+j]

        img_name = img_path.split('/')[-1]

        #detect faces
        cv2.rectangle(img, pos_start, pos_end, color=(0, 0, 255), thickness=4)
        out_face_path = out_detect_path + img_name
        cv2.imwrite(out_face_path,img)

        
        if not os.path.isdir(out_crop_path):
            os.makedirs(out_crop_path)
        out_path = out_crop_path + img_name
        img_blank = cv2.resize(img_blank,(96,112)) 
        cv2.imwrite(out_path, img_blank)


#img need to be crop
img_path = '/home/lin/Desktop/001_0.bmp'
#model path
dlib_path = '/home/lin/Desktop/mobileFace/models/shape_predictor_68_face_landmarks.dat'
#where to store the img(just the dir, will store with the same name as the img_path)
out_crop_path = '/home/lin/Desktop/mobileFace/'
#store the raw img
out_detect_path = './'

crop(img_path,out_crop_path,out_detect_path,dlib_path)
