import dlib         
import numpy as np  
import cv2          
import os
import math

def face_alignment(img_path,path_save):
    img = cv2.imread(img_path)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
    try:
        faces = detector(img, 1)
        for k, face in enumerate(faces):
            #rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
            shape = predictor(img, face)
            # left eye, right eye, nose, left mouth, right mouth
            order = [36, 45, 30, 48, 54]
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
            # 计算两眼的中心坐标
            eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, (shape.part(36).y + shape.part(45).y) * 1./2)
            dx = (shape.part(45).x - shape.part(36).x)
            dy = (shape.part(45).y - shape.part(36).y)
            # 计算角度
            angle = math.atan2(dy, dx) * 180. / math.pi
            # 计算仿射矩阵
            RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
            # 进行仿射变换，即旋转
            RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0], img.shape[1]))
            print(path_save)
            if not os.path.isdir(path_save):
                os.makedirs(path_save)
            img_name = img_path.split('/')[-1]
            path_save = path_save + img_name
            cv2.imwrite(path_save, RotImg)
    except:
        print(img_path)

path_read = "/Volumes/FreeAgent G/64_CASIA-FaceV5/"
dlib_path = './models/shape_predictor_68_face_landmarks.dat'
parts_path = os.listdir(path_read)
for part in parts_path:
    if part == 'cropped':
        continue
    part_p = path_read + part
    if not os.path.isdir(part_p):
        continue
    subs = os.listdir(part_p)
    for sub in subs:
        sub_p = part_p + '/' + sub
        path_save = path_read + 'aligned/' + sub + '/'
        if os.path.exists(path_save):
            continue

        imgs = os.listdir(sub_p)
        for img in imgs:
            img_path = sub_p + '/' + img
            face_alignment(img_path,path_save)

