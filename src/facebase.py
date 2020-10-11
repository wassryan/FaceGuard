import os,sys
import numpy as np

import cv2
from PIL import Image
import torch
import torch.nn.functional as F
# import torchvision.transforms as transforms
import dlib
import time

from scipy import spatial

from imutils import face_utils
from PyQt5.QtCore import QThread, pyqtSignal

from .utils import eye_aspect_ratio #  CFPDataset, CaffeCrop
from lib import basemodel


class FRFace(QThread):
    recog_msg = pyqtSignal(object)

    def __init__(self):
        super(FRFace, self).__init__()
        self.root = 'data/'
        self.db_path = self.root + 'db_features.pkl'
        self.batch_size = 32
        self.workers = 16
        self.model_path = './lib/070.ckpt'
        self.db_imgs = self.root + 'cropped_gallery/'

        self.min_thresh = 0.3

        #load the database feature
        self.db_dict = torch.load(self.db_path)
        self.name_list = self.db_dict['name']

        print("=> Load Database Feature into CPU Memory...")

        self.model = self.init_model()
        print("=> Model Load & Init...")

        self.result_face = RtFace() # save result

    def init_model(self):
        model = basemodel.MobileFacenet()
        ckpt = torch.load(self.model_path, map_location='cpu')
        model.load_state_dict(ckpt['net_state_dict'])
        model.eval()

        return model

    def predict(self, image):
        """
        image: H,W,C, ndarray
        """
        # im = [image]
        # lfw_dataset = LFW(im, im)
        im_hflip = image[:, ::-1, :]
        im = (image - 127.5) / 128.0
        im_hflip = (im_hflip - 127.5) / 128.0
        im = np.expand_dims(im.transpose(2, 0, 1), axis=0) # (N,C,H,W)
        im_hflip = np.expand_dims(im_hflip.transpose(2, 0, 1), axis=0)

        # print(im.dtype)
        ims = [torch.from_numpy(im).float(), torch.from_numpy(im_hflip).float()]

        res = [self.model(d).data.cpu() for d in ims]
        pred_feat = torch.cat(res, dim=1) # (1,256)

        scores = torch.mm(F.normalize(pred_feat), F.normalize(self.db_dict['feat']).T).squeeze() # (k)
        max_score, max_id = torch.max(scores, dim=0)

        return max_score.item(), self.name_list[max_id]


    def run(self):
        face_conf = 0.0
        face_name = 'Unknown'

        image = self.image # ndarray
        time_start = time.time()
        # TODO: make sure image's size > (300, 300) or the program would crash in CNN forward
        if image.shape[0] < 80 or image.shape[1] < 30:
            print(" Discard face({})".format(image.shape))
            self.recog_msg.emit("invalid face")

        # print("=> ", image.shape)
        top1_score, top1_impath = self.predict(image)
        pred_name = top1_impath.split('/')[-1].split('.')[0] # database path -> name

        conf = top1_score
        time_end = time.time()
        # print("Time cost: {:.3f}s".format(time_end-time_start))

        if conf >= self.min_thresh:
            face_conf = round(conf,2)
            face_name = pred_name
            face_path = top1_impath
            
            # TODO: 把识别数据反馈到FaceResult类中，让main类里面定义的槽函数出发getresult类获取数据到UI
            self.result_face.setData(True, None, face_conf, face_name, face_path, '')
            print("[Valid] confidence: {}, face_name: {}, path: {}".format(conf, pred_name, top1_impath))
            self.recog_msg.emit("valid face")
        else:
            print("[Low Score] confidence: {} face_name: {}".format(conf, pred_name))
            self.recog_msg.emit("invalid face")



class FDFace():

    def __init__(self, fr, crop_size=(224, 224)):
        self.frameratio = fr
        self.crop_size = crop_size

        self.shape_detector_path = './src/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()  # 人脸检测器
        self.predictor = dlib.shape_predictor(self.shape_detector_path)  # 人脸特征点检测器

        # 定义一些参数
        self.EYE_AR_THRESH_HIGH = 0.3  # EAR阈值，大于0.3认为眼睛是睁开的
        self.EYE_AR_THRESH_LOW = 0.25  # EAR阈值，小于0.2认为眼睛是闭住的
        self.EYE_AR_CONSEC_FRAMES = 3  # 当EAR小于阈值时，接连多少帧才确保发生了发生眨眼动作
        # 对应特征点的序号
        self.RIGHT_EYE_START = 37 - 1
        self.RIGHT_EYE_END = 42 - 1
        self.LEFT_EYE_START = 43 - 1
        self.LEFT_EYE_END = 48 - 1

        self.blink_counter = 0  # 眨眼计数
        self.frame_counter = 0  # 连续帧计数

        self.Trueface = False
        self.faceDetected = False

    # detect
    def detectFace(self, frame):
        """
        output:
        - frame_output: frame with detected box on it
        - det_box: select biggest box and return crop box
        - self.faceDetected: True if has face in frame
        - self.Trueface: True if face over eye blink threshold
        """
        frame_output = frame
        det_box = None
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(frame_gray, 0)  # 调用dlib人脸检测

        # frame_counter = 0  # 连续帧计数
        blink_counter = 0  # 眨眼计数
        # count = 0
        # name = 'Unknown'
        # confidence = 0

        if len(rects) < 1:
            self.blink_counter = 0  # 眨眼计数清零
            self.frame_counter = 0  # 连续帧计数清零
            self.Trueface = False
            self.faceDetected = False
            return frame_output, det_box, self.faceDetected, self.Trueface

        # transform to list
        rect_list = [[i, x.left(), x.right(), x.top(), x.bottom()] for i, x in enumerate(rects)]
        # sort by areas
        rect_list.sort(key=lambda x: (x[2]-x[1]) * (x[4]-x[3]), reverse=True)

        rect = rect_list[0]
        self.faceDetected = True
        idx, left, right, top, bottom = rect[0], rect[1], rect[2], rect[3], rect[4]
        shape = self.predictor(frame_gray, rects[idx])  # 检测特征点

        points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        leftEye = points[self.LEFT_EYE_START:self.LEFT_EYE_END + 1]  # 取出左眼对应的特征点
        rightEye = points[self.RIGHT_EYE_START:self.RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
        leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR
        rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼EAR
        # print('leftEAR = {0}'.format(leftEAR))
        # print('rightEAR = {0}'.format(rightEAR))
        ear = (leftEAR + rightEAR) / 2.0  # 求左右眼EAR的均值
        
        # resize frame
        # h, w = frame_output.shape[0:2]
        # frame_output = cv2.resize(frame_output, (int(w * self.frameratio), int(h * self.frameratio)))

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        # TODO:  调整逻辑，人脸识别一次后 ，需要把flag都清零
        if ear < self.EYE_AR_THRESH_LOW: # 0.25
            self.frame_counter += 1
        # 只有闭眼到一定帧数，再睁开眼，才会检测到人脸
        elif ear > self.EYE_AR_THRESH_LOW and self.frame_counter >= self.EYE_AR_CONSEC_FRAMES:
            self.blink_counter += 1
            # count += 1
            self.Trueface = True
            self.frame_counter = 0
        # else:
        #     if self.frame_counter >= self.EYE_AR_CONSEC_FRAMES:
        #         self.blink_counter += 1
        #         # count += 1
        #         self.Trueface = True
        #     self.frame_counter = 0
        
        # crop det true box
        # TODO: 调整到112x96的倍数
        oh, ow = bottom - top, right - left
        newh = round(ow * 112 / 96)
        pad = newh - oh

        if pad > 0: # should expand, pad to top
            newleft, newright, newtop, newbot = left, right, max(int(top-pad),0), bottom 
        else: # should crop, -pad to bot
            newleft, newright, newtop, newbot = left, right, top, int(bottom+pad)

        maxh, maxw = frame.shape[0], frame.shape[1]
        # oh, ow = bottom - top, right - left
        # omax = max(oh, ow)
        # expand small one
        # hpad = (omax - ow) / 2
        # vpad = (omax - oh) / 2
        # newleft, newright, newtop, newbot = int(left-hpad), int(right+hpad), int(top-vpad), int(bottom-vpad)

        det_box = frame[newtop:newbot, newleft:newright] # H, W
        det_box = cv2.resize(det_box, (96, 112))
        # cv2.imshow("im", det_box)

        frame_output = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        blink_msg = "Blink Count: {}".format(self.blink_counter)
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_output = cv2.putText(frame_output, blink_msg, (50, 100), font, 1.2, (0,255,0), 2)

        return frame_output, det_box, self.faceDetected, self.Trueface

class RtFace():
    """
    Store result of face recognition
    """
    def __init__(self, faceExist=False, faces=None, p=0.0, faceName='', faceID=''):
        self.faceExist = False # face exist or not
        self.faces = None          # 检测到的人脸区域 # TODO: 是否需要？？
        self.p = 0.0                 # confidence(similarity)
        self.faceName = ''    # face name
        self.facePath = ''    # image path
        self.faceID = ''        # ID #TODO: 是否需要？？

    # 返回数据
    # def getData(self):
        # return self.faceExist, self.faces, self.p, self.faceName, self.facePath, self.faceID

    # 修改数据
    def setData(self, faceExist, faces, p, faceName, facePath, faceID):
        self.faceExist = faceExist
        self.faces = faces
        self.p = p
        self.faceName = faceName
        self.facePath = facePath
        self.faceID = faceID

