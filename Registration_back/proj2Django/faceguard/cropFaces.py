import dlib         
import numpy as np  
import cv2          
import os

current_path = os.path.dirname(__file__)
def crop(img_path, out_crop_path, out_detect_path, dlib_path):
    img = cv2.imread(img_path)
    dlib_path = '/Users/liuchenxi/Desktop/faceguard_registration/src/shape_predictor_68_face_landmarks.dat'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_path)


    has_face = False
    faces = detector(img, 1)
    for k, d in enumerate(faces):

        height = d.bottom() - d.top()
        width = d.right() - d.left()

        height += 20
        width = round(96 * height / 112)

        pos_start = tuple([d.left(), d.top() - 20])
        pos_end = tuple([d.left() + width, d.top() - 20 + height])

        img_blank = np.zeros((height, width, 3), np.uint8)

        for i in range(height):
            for j in range(width):
                img_blank[i][j] = img[d.top() - 20 + i][d.left() + j]

        img_name = img_path.split('/')[-1]

        # detect faces and frame this face
        cv2.rectangle(img, pos_start, pos_end, color=(0, 0, 255), thickness=4)
        out_face_path = current_path + out_detect_path + img_name
        cv2.imwrite(out_face_path, img)

        # crop face and save it
        if not os.path.isdir(out_crop_path):
            os.makedirs(out_crop_path)
        out_path = out_crop_path + img_name
        img_blank = cv2.resize(img_blank, (96, 112))
        cv2.imwrite(out_path, img_blank)
        has_face = True

        return has_face

# def crop(img_path,path_save,dlib_path):
#     img = cv2.imread(img_path)
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor(dlib_path)
#
#     faces = detector(img, 1)
#     has_face = False
#     for k, d in enumerate(faces):
#
#         pos_start = tuple([d.left(), d.top()])
#         pos_end = tuple([d.right(), d.bottom()])
#
#         height = d.bottom()-d.top()
#         width = d.right()-d.left()
#
#         height += 20
#         width = round(96*height/112)
#
#         img_blank = np.zeros((height, width, 3), np.uint8)
#
#         for i in range(height):
#             for j in range(width):
#                 img_blank[i][j] = img[d.top()-20+i][d.left()+j]
#
#         img_name = img_path.split('/')[-1]
#         if not os.path.isdir(path_save):
#             os.makedirs(path_save)
#         out_path = path_save + img_name
#         img_blank = cv2.resize(img_blank,(96,112))
#         cv2.imwrite(out_path, img_blank)
#         has_face = True
#
#     return has_face
#
# #img need to be crop
# img_path = '../images/6.jpg'
# #model path
# dlib_path = '../model/shape_predictor_68_face_landmarks.dat'
# #where to store the img(just the dir, will store with the same name as the img_path)
# out_path = '../images_crop/'
#
# crop(img_path,out_path,dlib_path)
