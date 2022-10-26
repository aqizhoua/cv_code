import torch
import torchvision.utils

from model import myLenet5
from PIL import Image
from torchvision import transforms

import numpy as np

import matplotlib.pyplot as plt



net = torch.load("model.plt",map_location=torch.device('cpu')) #把模型放到cpu上，因为是用mps训练出的模型，所以默认是在mps处理

img_path = "/Users/aqizhou/PycharmProjects/pytorch_study/21_model_train/imgs/cat.png"

img =Image.open(img_path)

img = img.convert("RGB") #4通道转为3通道 还有一个通道是透明度
trans = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
img = trans(img)







# print(img.shape)
img = torch.reshape(img,(1,3,32,32))

# print(img)

class_to_idx = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

label_pre_index = torch.argmax(net(img))

label_pre = class_to_idx[label_pre_index]

print(label_pre)
