# FaceGuard

A new face recognition system

## Demo

[![FaceGuard Demo](https://res.cloudinary.com/marcomontalbano/image/upload/v1604060266/video_to_markdown/images/google-drive--1OkJSvIPGsrYHZg0AfAwWwyWSPtkPtGY2-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://drive.google.com/file/d/1OkJSvIPGsrYHZg0AfAwWwyWSPtkPtGY2/view "FaceGuard Demo")

## Requirement

This system is tested successfully under MacOS, Windows 10(havn't test on Ubuntu).

1. Environment: MacOS/Windows 10, *python3.6*
2. Dependence

- PyQt5(pip install pyqt5)
- imutils
- CMake(pip install CMake)
- Boost(pip install Boost)
- dlib(pip install dlib)
- Django(pip install django)

## Installation

```
pip install -r requirements.txt
```

*Attention: version of opencv-python should prior to 4.2, or pyqt5 will crash*





## How to use

### - Materials need to be downloaded first

- dlib landmark detection model

  - download [landmark detector](https://drive.google.com/file/d/1NuqK16wRZWFGF-eFXJ_zuaCRqLemSrJc/view?usp=sharing) from google drive, put it into `./src/`folder
- gallery dataset

  - download [gallery dataset](https://drive.google.com/file/d/1oHXjBcKwKr-BnquD-AGtMBPTBG1y5pQi/view?usp=sharing) from google drive, unzip this file, put this folder into current path

### - User Registration (Web)

####   1. Front-end

- Enter `./Registration_front/faceguard/` folder

- Start web application : **click index.html** 



####  2. Back-end

- Enter folder `./Registration_back/proj2Django/` 

- Database delopyment

  - build dependencies

    ```shell
    pip3 install sqlclient
    sql.server start
    pip3 install django-cors-headers
    ```

  - Modify `/proj2Django/setting.py `according to your environment

    ```python
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'HOST': 'localhost',  
            'PORT': 3306,  
            'USER': 'root',  # database user
            'PASSWORD': '',  # passwd
            'NAME': 'prj2',  # database name
            'OPTIONS': {'charset':'utf8mb4'}
        }
    }
    ```

  - create tables

    ```shell
    python manage.py makemigrations
    python manage.py migrate
    ```

  - Change path according to your environment

    *somehow the relative paths don't work, so absolute paths are needed*

    1. Enter `faceguard/` folder

    2. Open `views.py`, change the following to the absolute path of `src/shape_predictor_68_face_landmarks.dat` and `data/cropped_gallery/`

       ```python
       dlib_path = '/Users/liuchenxi/Desktop/faceguard_registration/src/shape_predictor_68_face_landmarks.dat'
       gallery_path = '/Users/liuchenxi/Desktop/faceguard_registration/data/cropped_gallery/'
       ```

    3. enter `../register/` folder

    4. open `regist.py`, change the following to the absolute path of `data/` and `data/cropped_gallery/`

       ```python
       save_path = '/Users/liuchenxi/Desktop/faceguard_registration/data/'
       gallery_path = '/Users/liuchenxi/Desktop/faceguard_registration/data/cropped_gallery/'
       ```

  - Run server

    ```shell
    python manage.py runserver
    ```

    

  Once a new user is registered successfully into our system, the feature extractor will extract his/her face feature and store this feature into `./data/db_features.pkl`

  `noted:` It is quite slow to generate features of our pre-defined  gallery set(around 1000 persons' images from CASIA-FaceV5 / LFW dataset and our classmates) . If you don't want to generate by yourself, you can [download](https://drive.google.com/file/d/1FvW05pH672BWRblYZofy05prhmbHJ4Yz/view?usp=sharing) it from here and then put the pkl file into the `./data/` folder.



### - Faceguard System (Local terminal)

To recognize user by matching features in `./data/db_features.pkl`


####  Run the System

```
python main.py
```



## How to train the feature extractor

### requirements

I was trained the model under Ubuntu.

1. Environment: Ubuntu, Python 3.7.9

### Installation

```
pip install -r requirements.txt
```

### Pre-trained model

All the model can be downloaded from [google drive](https://drive.google.com/file/d/1eoDZPg_8Yv-Z3Yw2t7hBNg5JIVrvZ1AZ/view?usp=sharing),  unzip the file and move it to the `train/` folder

We use `model/best/068.ckpt` from this [GitHub](https://github.com/Xiaoccer/MobileFaceNet_Pytorch) as our pre-trained model

### Dataset

All the pre-processed dataset that we used for training can be download from [google drive](https://drive.google.com/file/d/122sgDpPbYqkYG4mUBey_TWbqr0FlhHyG/view?usp=sharing) , unzip the file, move the `Asian/` folder under the `train/` folder

### Data pre-processing

detect faces and crop faces as $96*112$

1. enter `train/` folder

2. Change the following in `cropFaces.py` into the image path that you want to crop

   ```python
   img_path = '/home/lin/Desktop/caowen.jpg'
   ```

3. run the code

   ```shell
   python cropFaces.py
   ```

   

### first round finetune

**Objective:** enable the model to capture more features of the Asian face. After all, the pre-trained model is trained on a dataset with the majority of Europeans and Americans. **Dataset:** CASIA-FaceV5. There is a total of 500 different subjects in this dataset, and we keep the images of 20 subjects as the next round of finetune dataset. The remaining 480 subjects were divided into training set and test set according to the ratio of 8:2. 

 

enter `train/` folder

#### 1. Generate data needed

```shell
python pos_neg_pairs.py ./Asian/test/
python label_pairs.py ./Asian/train/
```

#### 2. Change the resumed model

change `config.py`

```python
RESUME = './model/best/068.ckpt'
```

#### 3. Run the training code

```shell
python train_finetune.py
```

the best model are store in `./model/`, you can found our trained model in this step in `./model/Asian/070.ckpt`

#### 3. evaluation

- change the resume model in `evaluate_Asian.py`

  ```python
  resume = './model/Asian/070.ckpt'
  ```

- run the code

  ```shell
  python evaluate_Asian.py
  ```

testing result of CASIA-FaceV5:

| Fold | 1      | 2      | 3      | 4     | 5      | 6      | 7      | AVE   |
| ---- | ------ | ------ | ------ | ----- | ------ | ------ | ------ | ----- |
| ACC  | 100.00 | 100.00 | 100.00 | 98.00 | 100.00 | 100.00 | 100.00 | 99.71 |

testing result of second-round-finetune testing dataset:

| Fold | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | AVE   |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| ACC  | 85.00 | 95.00 | 75.00 | 75.00 | 85.00 | 95.00 | 95.00 | 75.00 | 85.00 |



### second round finetune

**Objective:** Given that the CASIA-FaceV5 data set was taken under controlled conditions, we want to make our model more robust.

**Dataset:** we used our own data set plus 20 subjects that were not used in the previous round as dataset and perform finetune on the basis of the model obtained in the last round of finetune. In this round, there are a total of 40 individuals, and each individual has 5-7 pictures. We divided the data set into training set and test set according to the ratio of 3:1. 

enter `train/` folder

#### 1. Generate data needed

```shell
python pos_neg_pairs.py ./Asian/left_20/test/
python label_pairs.py ./Asian/left_20/train/
```

#### 2. Change the resumed model

change `config.py`

```python
RESUME = './model/Asian/070.ckpt'
```

#### 3. Run the training code

```shell
python train_finetune.py
```

the best model are store in `./model/`, you can found our trained model in this step in `./model/Asian+friends/070.ckpt`

#### 3. evaluation

- change the resume model in `evaluate_Asian.py`

  ```python
  resume = './model/Asian/070.ckpt'
  ```

- run the code

  ```shell
  python evaluate_Asian.py
  ```

testing result:

| Fold | 1     | 2     | 3     | 4     | 5     | 6      | 7     | 8     | AVE   |
| ---- | ----- | ----- | ----- | ----- | ----- | ------ | ----- | ----- | ----- |
| ACC  | 80.00 | 75.00 | 90.00 | 95.00 | 90.00 | 100.00 | 95.00 | 95.00 | 90.00 |

