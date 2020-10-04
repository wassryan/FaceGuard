import sys
import cv2
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from src.facebase import FRFace, FDFace

# _fromUtf8 = QtCore.QString.fromUtf8

class ARCFaceUI(QWidget):
    def __init__(self):
        super(ARCFaceUI, self).__init__()
        self.true_face = False
        self.face_detected = False
        self.freezeLogo = False # True-> freeze, False-> allowing changing logo
        self.frameratio = 0.6
        self.framesize = (800, 450)
        self.crop_size = (224, 224)

        self.initUI()

        self.recog_face = FRFace() # face recognition thread class
        self.detect_face = FDFace(self.frameratio, self.crop_size)# face detection class
        
        self.camera = cv2.VideoCapture(0) # init camera

        # self.through = Through()# 判断置信度，决定是否通过，并记录通过者的信息到list中

        # 定时器
        self.frametimer = QTimer(self)
        self.frametimer.timeout.connect(self.updateFrame)
        self.frametimer.start(20)

        self.logotimer = QTimer(self)
        self.logotimer.timeout.connect(self.unfreezeLogo)
        self.logotimer.start(1000)

        # 线程之间同步操作
        # arcface.finished即当arcface线程的run()函数执行完后，线程则会发出finished信号
        # face recogntion finished-> return result back to GUI

        # self.recog_face.finished.connect(self.getResult)  # 当人脸识别线程执行完,执行getresult() 获取结果
        self.recog_face.recog_msg.connect(self.getResult)
        self.frametimer.timeout.connect(self.updateFrame)
        # self.connect(self.timer, SIGNAL('timeout()'), self.updateFrame)   # connect连接 定时器timer 每到20ms 就触发更新画面的函数
        
        # self.through.pass_sign.connect(self.updateLogo)# 根据pass_sign信号(为一个str参数)决定是否需要更新UI界面上的logo

        '''
        # 与服务器交换数据
        self.connect_sever = ConnectSever()# 连接到服务器，进行相应的图片同步操作
        self.connect_sever.finished.connect(self.connect_sever.start)   # 触发服务器与本地数据的同步,ConnectSever类里run一次sleep(60)再执行服务器同步
        self.connect_sever.change.connect(self.arcface.getUpdateData)# 根据change信号(为一个list参数)决定是否需要更新数据到本地
        self.connect_sever.start()# 程序一开始即执行服务器与本地数据同步
        self.through.finished.connect(self.toSever)# through线程判断置信度之后，触发向服务器传图片的toSever函数
        '''

    def unfreezeLogo(self):
        self.freezeLogo = False

    def initUI(self):
        
        # self.resize(600, 480)
        # self.move(50, 50)
        self.setGeometry(50,50, self.framesize[0], self.framesize[1]) # ax, ay, w, h
        self.setWindowTitle('Intelligent Guard System')
        # print(self.width(), self.height())
        self.exit_button = QPushButton('Exit')
        self.exit_button.clicked.connect(self.exit_ARC)

        # 摄像机画面的标签
        self.camera_label = QLabel(self)
        self.camera_label.setText('')
        self.camera_label.setObjectName('Camera')
        self.camera_label.setScaledContents(True) # adaptively scale
        # self.camera_label.adjustSize()
        # self.camera_label.setGeometry(0, 0, 800, 600)

        # 检测结果标签
        self.detect_label = QLabel(self)
        self.detect_label.setObjectName('detect_label')
        self.detect_label.setText('Identity Name:')
        self.name_text = QLineEdit()
        self.name_text.setText(' ')

        # 检测 结果的置信度
        self.confidence_label = QLabel(self)
        self.confidence_label.setObjectName('confidence')
        self.confidence_label.setText('Confidence:')
        self.confid_text = QLineEdit()
        self.confid_text.setText(' ')

        # 图标标签
        self.icon_label = QLabel(self)
        self.logo_png = QtGui.QPixmap('./src/logo.png')
        # self.pass_png = QtGui.QPixmap('./src/pass.png')
        self.icon_label.setPixmap(self.logo_png)
        self.icon_label.resize(300, 400)
        self.setlayout()
        QtCore.QMetaObject.connectSlotsByName(self) # automatic connect signal to slot on widget

    def startFR(self, det_box):
        # ret, frame = self.camera.read()# read a frame from camera
        # self.recog_face.transform_frame(det_box) # opencv->PIL
        self.recog_face.image = det_box
        self.recog_face.start() # start QThread run()

    def exit_ARC(self):
        self.recog_face.exit() # 中止人脸识别的线程
        QCoreApplication.instance().quit() # 关闭窗口
        print("=> Happy Ending...")

    def getResult(self, msg):
        self.result = self.recog_face.result_face
        self.updateResult(msg)

    # update result to GUI
    def updateResult(self, msg):
        ##### Recog Result ######
        if self.freezeLogo: # in freeze time, do not update any result
            return
        elif msg == "valid face" and self.true_face:
            p = self.result.p # confidence
            faceName = self.result.faceName # result face name
            self.confid_text.setText(str(p))
            self.name_text.setText(faceName)

            self.updateLogo(passthrough='yes')
        else:
            p = 0.0
            faceName = 'Unknown'
            self.confid_text.setText(str(p))
            self.name_text.setText(faceName)

            self.updateLogo(passthrough='no')
            
        # self.confid_text.setText(_fromUtf8(str(p)))
        # self.name_text.setText(_fromUtf8(faceName))

    def updateLogo(self, passthrough='no'):
        ##### update logo with database corresponding face #####
        if passthrough == 'yes':
            qtimg = QtGui.QPixmap(self.result.facePath)
            qtimg = qtimg.scaled(300, 400, aspectRatioMode=Qt.KeepAspectRatio)
            # print(qtimg.height(), qtimg.width())
            self.icon_label.setPixmap(qtimg)
            self.freezeLogo = True
            self.logotimer.start(1000) # reset timer
            # self.icon_label.setScaledContents(True) # image adapt to label
        else:
            self.icon_label.setPixmap(self.logo_png)

        # self.icon_label.resize(300, 400)

    def setlayout(self):

        # hbox_button = QHBoxLayout()
        # hbox_button.addStretch(1)
        # hbox_button.addWidget(self.init_face_button)
        # hbox_button.addWidget(self.exit_button)

        # 创建水平布局
        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox_left = QHBoxLayout()
        self.hbox_left.addStretch(1)
        self.vbox_right = QVBoxLayout()
        self.vbox_right.addStretch(1)
        # self.vbox_ = QVBoxLayout()
        # self.vbox_.addStretch(1)

        # 创建网格布局
        self.grid_right = QGridLayout()
        self.grid_right.addWidget(self.detect_label, 0, 0)   # 添加姓名标签
        self.grid_right.addWidget(self.name_text, 0, 1)

        self.grid_right.addWidget(self.confidence_label, 1, 0)   # 置信度标签
        self.grid_right.addWidget(self.confid_text, 1, 1)

        # self.grid_right.addWidget(self.exit_button, 8, 0)

        self.hbox_left.addWidget(self.camera_label)   #  摄像头画面标签
        self.vbox_right.addWidget(self.icon_label)
        self.vbox_right.addLayout(self.grid_right)
        self.vbox_right.addWidget(self.exit_button)

        # self.hbox_left.setScaledContents()

        self.hbox.addLayout(self.hbox_left)
        self.hbox.addLayout(self.vbox_right)

        # self.vbox_.addLayout(self.hbox)

        # 应用布局
        self.setLayout(self.hbox)

    '''
    # 把通过的识别到的人脸图片上传到服务器
    def toSever(self):
        self.updateLogo()#如果人脸通过，则更换logo图标
        if len(self.through.through_lists) > 0:
            self.connect_sever.inputUploadData(self.through.through_lists.pop())
    '''

    # 格式转化
    def fromFrame2QImage(self, frame):
        height, width, bytesPerComponent = frame.shape
        bytesPerLine = bytesPerComponent * width
        # 变换彩色空间顺序
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
        # 转为QImage对象
        image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return image

    # 更新摄像头画面
    def updateFrame(self):
        ret, frame = self.camera.read()
        frame_output, det_box, face_detected, true_face = self.detect_face.detectFace(frame)
        self.true_face = true_face
        self.face_detected = face_detected
        if self.true_face: # 是真的人脸 才运行人脸识别
            self.startFR(det_box)

        else:
            p = 0
            faceName = 'Unknown'
            self.confid_text.setText(str(p))
            self.name_text.setText(faceName)
            self.updateLogo(passthrough='no')

        frame_Qimage = self.fromFrame2QImage(frame_output)
        frame_Qimage = QtGui.QPixmap.fromImage(frame_Qimage)
        frame_Qimage = frame_Qimage.scaled(self.framesize[0], self.framesize[1], aspectRatioMode=Qt.KeepAspectRatio)
        # print(frame_Qimage.height(), frame_Qimage.width())
        self.camera_label.setPixmap(frame_Qimage)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ARCFaceUI()
    print(window.size())
    
    window.show()
    sys.exit(app.exec_())
    print("=> Happy Ending...")