# FaceGuard
## run the model
### register part
`python register.py`

input : `new_img`refer to the img path that the people who are newly registered
### recog part
`python recog_new.py`

input : `cur_img` refer to the img that we want to recog

output: `[cur_img,score,matched_img]`




## How we finetune the model
### 0. evaluate data on mobileFaceNet model
***20-subjects Asian dataset***
|  fold   | acc  |
|  ----  | ----  |
| 1  | 70.00 |
| 2  | 86.00 |
| 3  | 76.00 |
| 4  | 94.00 |
| 5  | 74.00 |
| 6  | 94.00 |
| 7  | 80.00 |
| AVE  | 82.00 |

***20 friends dataset***
|  fold   | acc  |
|  ----  | ----  |
| 1  | 85.00 |
| 2  | 95.00 |
| 3  | 75.00 |
| 4  | 75.00 |
| 5  | 85.00 |
| 6  | 95.00 |
| 7  | 95.00 |
| 8  | 75.00 |
| AVE  | 85.00 |

### 1. use Asian dataset(`CASIA-FACEV5`) to finetune the mobileFaceNet model
20 subjects for next-step finetune
20 subjects for test
460 subjects for train
- pre-processing

  crop the img into 112x96

- finetune the model

  lr = 0.1, 
  epoch = 70 (start from 0)

***use 20 subjects for test to evaluate the model:***
|  fold   | acc  |
|  ----  | ----  |
| 1  | 100.00 |
| 2  | 100.00 |
| 3  | 100.00 |
| 4  | 98.00 |
| 5  | 100.00 |
| 6  | 100.00 |
| 7  | 100.00 |
| AVE  | 99.71 |

### 2. use our classmates' photos and 20 subjects left from last step
we collected 20 classmates photos, each had 5 photos.
in this step , we used 18 subjects of Asian dataset and 12 subjects of our friends to finetune the model from last step
And we used 2 subjects of Asian dataset and 8 subjects of our friends as our test dataset.

***evaluate***
|  fold   | acc  |
|  ----  | ----  |
| 1  | 80.00 |
| 2  | 75.00 |
| 3  | 90.00 |
| 4  | 95.00 |
| 5  | 90.00 |
| 6  | 100.00 |
| 7  | 95.00 |
| 8  | 95.00 |
| AVE  | 90.00 |
