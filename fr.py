# -*- coding: utf-8 -*-

#author joshwoo
import os
import sys
import time
import numpy
import PIL
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

olive_pic_path = "olivettifaces.gif"

def load_pic_data(path):
    img = Image.open(path)
    img_tensor = transforms.ToTensor()(img)
    faces = torch.zeros(400,2679)
    for i in range(20):
        for j in range(20):
            q = 0
            for x in range(57):
                for y in range(47):
                    faces[i*20 + j][q] = img_tensor[0][i*57 + x][j*47 + y]
                    q = q + 1
    labels = torch.zeros(400)
    for i in range(40):
        labels[i*10:i*10+10] = i
    return faces,labels

def split_data_set(faces,labels):
    tr_data=torch.zeros(320,2679)
    tr_label=torch.zeros(320)
    va_data=torch.zeros(80,2679)
    va_label=torch.zeros(80)
    for i in range(40):
        tr_data[i*8:i*8+8] = faces[i*10:i*10+8]
        tr_label[i*8:i*8+8] = labels[i*10:i*10+8]
        va_data[i*2:i*2+2] = faces[i*10+8:i*10+10]
        va_label[i*2:i*2+2]=labels[i*10+8:i*10+10]
    return [(tr_data,tr_label),(va_data,va_label)]

class fr_cnn_net(nn.Module):
    def __init__(self):
        input = 2679

if __name__=="__main__":
    load_pic_data(olive_pic_path)
