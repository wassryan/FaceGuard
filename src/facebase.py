import os,sys
import numpy as np

import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import dlib
import time

from scipy import spatial

from imutils import face_utils
from PyQt5.QtCore import QThread, pyqtSignal

from .utils import eye_aspect_ratio, CFPDataset, CaffeCrop
from lib import basemodel


class FRFace(QThread):
    recog_msg = pyqtSignal(object)

    def __init__(self):
        super(FRFace, self).__init__()
        self.root = 'data/'
        self.db_path = self.root + 'db/'
        self.batch_size = 32
        self.workers = 16
        self.gallery_file = self.root + 'search_gallery_1.csv'
        self.model_path = self.root + 'model_best.pth.tar'
        self.db_imgs = self.root + 'cropped_gallery/'

        self.min_thresh = 0.25

        def load_meta_data(meta_file):
            #read csv file
            meta_data = dict()
            with open(meta_file, 'r') as in_f:
                in_f.readline() # the first line is not data
                for idx, line in enumerate(in_f):
                    record = line.strip().split(',')
                    template, class_id = int(record[0]), int(record[1])
                    img_path = record[2]
                    if template not in meta_data:
                        meta_data[template] = ( class_id, [img_path,] )
                    else:
                        meta_data[template][1].append(img_path)
            return meta_data

        def load_feat():
            #load the feature of gallery set
            mid_feats = dict()
            feat_dim = 256
            feas = os.listdir(self.db_path)
            for fea in feas:
                fea_p = self.db_path +  fea
                temp = np.load(fea_p)
                
                idd = int(fea.split('.')[0])
                mid_feats[idd] = temp
                    
            return mid_feats, feat_dim

        def update_meta_data(meta_data, feats, feat_dim):
            new_meta_data = dict()
            for template in meta_data.keys():
                class_id, img_names = meta_data[template]
                feat = feats[template]
                new_meta_data[template] = (class_id, feat, img_names[0])
            return new_meta_data

        def init_model():
            class_num = 500

            model = basemodel.resnet18(pretrained=False, num_classes=class_num, \
                        extract_feature=True, end2end=True)
            model = torch.nn.DataParallel(model)
            model.eval()
            
            #load model
            checkpoint = torch.load(self.model_path,map_location='cpu')
            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = model.state_dict()
            for key in pretrained_state_dict:
                if key in model_state_dict:
                    model_state_dict[key] = pretrained_state_dict[key]
            model.load_state_dict(model_state_dict)

            return model

        feat_dim = 0
        #load csv file
        self.gallery_data = load_meta_data(self.gallery_file)

        #load the gallery set
        mid_feats, feat_dim = load_feat()
        self.gallery_data = update_meta_data(self.gallery_data, mid_feats, feat_dim)
        print("=> Load Database Feature into CPU Memory...")
        self.model = init_model()
        print("=> Model Load & Init...")

        self.result_face = RtFace() # save result

    def get_top(self,probe, gallery_data):
        #cal the cosine dist and return the top 5 results
        score_info = list()
        probe_feat = probe[0]
        for template in gallery_data.keys():
            gallery_id, gallery_feat, gallery_img_path = gallery_data[template]
            score = 1-spatial.distance.cosine(probe_feat, gallery_feat)
            #score = np.sqrt(np.sum(np.square(probe_feat - gallery_feat)))
            score_info.append((gallery_id,gallery_img_path,score))
        
        score_info = sorted(score_info, key=lambda a:a[2], reverse=True)
        #score_info = sorted(score_info, key=lambda a:a[1])
        top1 = [score_info[0][0], score_info[0][1], score_info[0][2]] # [id, path, score]
        return top1
        
    def extract_feat(self, image):
        """
        image: H,W,C
        """
        image = np.expand_dims(image.transpose((2,0,1)), axis=0) # (N,C,H,W)
        # print("input shape: ", image.shape) # (1,3,281,310)
        inp_tensor = torch.from_numpy(image).to(torch.float)
        inp_tensor = inp_tensor / 255. # scale to [0,1]

        # FIXME: set yaw to 0.0???
        yaw = torch.zeros((1,), dtype=torch.float)

        # print(inp_tensor.dtype)
        output = self.model(inp_tensor, yaw)
        output_data = output.cpu().data.numpy()

        return output_data

    def run(self):
        face_conf = 0.0
        face_name = 'Unknown'

        image = self.image # ndarray
        time_start=time.time()
        # TODO: make sure image's size > (300, 300) or the program would crash in CNN forward
        if image.shape[0] < 200 or image.shape[1] < 200:
            print(" Discard face({})".format(image.shape))
            self.recog_msg.emit("invalid face")

        print("=> ", image.shape)
        probe_feat = self.extract_feat(image)
        top1 = self.get_top(probe_feat, self.gallery_data)
        # gt_name = img_path.split('/')[-1].split('.')[0]
        # match_imgpath = top1[1]
        pred_name = top1[1].split('/')[-1].split('.')[0] # database path -> name
        match_imgpath = os.path.join(self.db_imgs, pred_name+'.jpg')
        conf = top1[2]
        time_end=time.time()
        # print("Time cost: {:.3f}s".format(time_end-time_start))

        # print("confidence: {}, face_name: {}, path: {}".format(conf, pred_name, match_imgpath))

        if conf >= self.min_thresh:
            face_conf = conf
            face_name = pred_name
            face_path = match_imgpath
            
            # TODO: 把识别数据反馈到FaceResult类中，让main类里面定义的槽函数出发getresult类获取数据到UI
            self.result_face.setData(True, None, face_conf, face_name, face_path, '')
            print("[Valid] confidence: {}, face_name: {}, path: {}".format(conf, pred_name, match_imgpath))
            self.recog_msg.emit("valid face")
        else:
            print("[InValid] confidence: {}".format(conf))
            self.recog_msg.emit("invalid face")


    # 执行人脸识别
    def run_debug(self):
        image = self.image # ndarray
        face_max_p = 0.0
        face_max_name = 'Unknown'
        face_max_ID = 'Unknown'

        try:
            # faces, i = doFaceDetection(self.hFDEngine, image)# SDK人脸检测

            if len(faces) > 0:
                face_exist = True
                # print(face_exist)
                # 获得特征库数据
                faceFeatureBase = self.facebase.getFaceData()
                # 提取人脸的 特征
                featureA = extractFRFeature(self.hFREngine, image, faces[i])
                if featureA == None:
                    print(u'extract face feature in Image faile')
                # 进行人脸比对
                face_max_p, face_max_name, face_max_ID = comparefaces(self.hFREngine, featureA, faceFeatureBase)
            else:
                face_exist = False

            # facefrdata：存放人脸识别后返回的数据类:人脸是否存在、检测到的人脸区域、置信度、ID
            self.facefrdata.setData(face_exist, faces, face_max_p, face_max_name, face_max_ID)      # 数据存放在facefrdata对象中
            # print(face_max_name+'   '+str(face_max_p))
            # 是否太过频繁？更新一次人脸识别就更新一次数据库？
            '''
            if self.updatefacebase:#　如果需要更新数据库的指令为True，则开始更新
                self.doUpdate()# 根据updatelist中的op操作进行增删数据
                self.facebase.loadfacedata()# 更新数据库到pkl中
                self.updatefacebase = False
            '''
        except Exception as e:
            print(e.message)
        finally:
            pass

    # def transform_frame(self, frame):
    #     # opencv->PIL
    #     self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    # def run(self):
    #     try:
    #         pass
    #     except Exception as e:
    #         print(e.message)


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
        # blink_counter = 0  # 眨眼计数
        count = 0
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
        frame_output = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # resize frame
        # h, w = frame_output.shape[0:2]
        # frame_output = cv2.resize(frame_output, (int(w * self.frameratio), int(h * self.frameratio)))

        # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        if ear < self.EYE_AR_THRESH_LOW:
            self.frame_counter += 1
        else:# 只有闭眼到一定帧数，再睁开眼，才会检测到人脸
            if self.frame_counter >= self.EYE_AR_CONSEC_FRAMES:
                self.blink_counter += 1
                count += 1
                self.Trueface = True
            self.frame_counter = 0
        
        # crop det true box
        maxh, maxw = frame.shape[0], frame.shape[1]
        oh, ow = bottom - top, right - left
        omax = max(oh, ow)
        # expand small one
        hpad = (omax - ow) / 2
        vpad = (omax - oh) / 2
        newleft, newright, newtop, newbot = int(left-hpad), int(right+hpad), int(top-vpad), int(bottom-vpad)

        det_box = frame[newtop:newbot, newleft:newright] # H, W

        # for rect in rects:
        #     self.faceDetected = True
        #     left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
        #     shape = self.predictor(frame_gray, rect)  # 检测特征点

        #     points = face_utils.shape_to_np(shape)  # convert the facial landmark (x, y)-coordinates to a NumPy array
        #     leftEye = points[self.LEFT_EYE_START:self.LEFT_EYE_END + 1]  # 取出左眼对应的特征点
        #     rightEye = points[self.RIGHT_EYE_START:self.RIGHT_EYE_END + 1]  # 取出右眼对应的特征点
        #     leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR
        #     rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼EAR
        #     # print('leftEAR = {0}'.format(leftEAR))
        #     # print('rightEAR = {0}'.format(rightEAR))
        #     ear = (leftEAR + rightEAR) / 2.0  # 求左右眼EAR的均值
        #     frame_output = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        #     # resize frame
        #     # h, w = frame_output.shape[0:2]
        #     # frame_output = cv2.resize(frame_output, (int(w * self.frameratio), int(h * self.frameratio)))

        #     # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
        #     if ear < self.EYE_AR_THRESH_LOW:
        #         self.frame_counter += 1
        #     else:# 只有闭眼到一定帧数，再睁开眼，才会检测到人脸
        #         if self.frame_counter >= self.EYE_AR_CONSEC_FRAMES:
        #             self.blink_counter += 1
        #             count += 1
        #             self.Trueface = True
        #         self.frame_counter = 0
        #     continue

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