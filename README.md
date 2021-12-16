## FaceGuard

A new face recognition system (blow is our system's data flow)

![image](https://github.com/wassryan/FaceGuard/blob/dev/src/model.png)

### Requirement

This system is tested successfully under MacOS(havn't test on Ubuntu).
1. Environment: MacOS, *python3.6*
2. Dependence
- PyQt5(pip install pyqt5)
- imutils
- CMake(pip install CMake)
- Boost(pip install Boost)
- dlib(pip install dlib)

### Installation

```
pip install -r requirements.txt
```

*Attention: version of opencv-python should prior to 4.2, or pyqt5 will crash*

### How to use

#### Registration
1. put `shape_predictor_68_face_landmarks.dat` under `src/` folder
2. put face database under the `/data/cropped_gallery/`, run following command, this will produce feature data(`db_features.pkl`) extracted from the model.

```
python regist.py
```

#### Start System
```
python main.py
```