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
        #print(height)
        #print(width)
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
        #print("Save to:", path_save+"img_face_"+str(k+1)+".jpg")
        print(out_path)
        img_blank = cv2.resize(img_blank,(96,112))
        #shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  
        cv2.imwrite(out_path, img_blank)
    
        #print(img_path)
'''
#types = ['gallery/','probe/','classmate/']
types = ['test/','train/']

for typ in types:
    path_read = "/media/lin/FreeAgent G/64_CASIA-FaceV5/" + typ
    #path_read = typ
    out_path = '/home/lin/Desktop/Asian/' + typ 
    #out_path = './cropped_gallery/'
    dlib_path = './models/shape_predictor_68_face_landmarks.dat'

    subs = os.listdir(path_read)
    for sub in subs:
        sub_p = path_read + sub + '/'
        imgs = os.listdir(sub_p)
        out_p = out_path + sub + '/'
        for img in imgs:
            img_path = sub_p + img
            if img[0] == '.':
                continue
            print(img_path)
            #print(out_p)
            crop(img_path,out_p,dlib_path)
'''
img_path = '/home/lin/Desktop/caowen.jpg'
dlib_path = '../src/shape_predictor_68_face_landmarks.dat'
crop(img_path,'./',dlib_path)
