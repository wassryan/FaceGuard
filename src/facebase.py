
import cv2
import dlib
from imutils import face_utils
from PyQt5.QtCore import QThread, pyqtSignal

from .eye_blink import eye_aspect_ratio

class FRFace(QThread):

    def __init__(self):
        super(FRFace, self).__init__()
    
    def run(self):
        try:
            pass
        except Exception as e:
            print(e.message)


class FDFace():

    def __init__(self, fr):
        self.frameratio = fr

        shape_detector_path = './src/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()  # 人脸检测器
        self.predictor = dlib.shape_predictor(shape_detector_path)  # 人脸特征点检测器

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
    # 只检测该帧是否有人脸，不识别
    def detectFace(self, frame):
        frame_output = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(frame_gray, 0)  # 调用dlib人脸检测

        # frame_counter = 0  # 连续帧计数
        # blink_counter = 0  # 眨眼计数
        count = 0
        name = 'Unknown'
        confidence = 0

        if len(rects) < 1:
            self.blink_counter = 0  # 眨眼计数清零
            self.frame_counter = 0  # 连续帧计数清零
            self.Trueface = False
            self.faceDetected = False

        for rect in rects:
            self.faceDetected = True
            left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()
            shape = self.predictor(frame_gray, rect)  # 检测特征点

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
            continue

        return frame_output, self.faceDetected, self.Trueface