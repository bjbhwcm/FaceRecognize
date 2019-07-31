# -*- coding: utf-8 -*-

#author joshwoo

# the first one of face recognize

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
    faces = torch.zeros(400,57,47)
    for i in range(20):
        for j in range(20):
            for x in range(57):
                for y in range(47):
                    faces[i*20 + j][x][y] = img_tensor[0][i*57+x][j*47+y]
    labels = torch.zeros(400)
    for i in range(40):
        labels[i*10:i*10+10] = i
    return faces,labels

def split_data_set(faces,labels):
    tr_data=torch.zeros(320,57,47)
    tr_label=torch.zeros(320)
    va_data=torch.zeros(80,57,47)
    va_label=torch.zeros(80)
    for i in range(40):
        tr_data[i*8:i*8+8] = faces[i*10:i*10+8]
        tr_label[i*8:i*8+8] = labels[i*10:i*10+8]
        va_data[i*2:i*2+2] = faces[i*10+8:i*10+10]
        va_label[i*2:i*2+2]=labels[i*10+8:i*10+10]
    return [(tr_data,tr_label),(va_data,va_label)]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class fr_cnn_net(nn.Module):
    def __init__(self):
        super(fr_cnn_net, self).__init__()
        input = 2679
        Alayer = 512
        output = 40
        self.fowarding = nn.Sequential(
            nn.Conv2d(1,8,(5,5),stride=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(8, 16, (5,5), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            Flatten(),
            nn.Linear(16*11*8,512),
            nn.ReLU(True),
            # nn.Dropout(0.4),
            nn.Linear(512,40),
            #nn.Sigmoid()
        )
    def forward(self, input):
        out = self.fowarding(input)
        return out


if __name__=="__main__":
    faces,labels = load_pic_data(olive_pic_path)
    [(tr_data,tr_label),(va_data,va_label)] = split_data_set(faces,labels)
    tr_data = Variable(tr_data)
    tr_label = Variable(tr_label)
    va_data = Variable(va_data)
    va_label = Variable(va_label)
    fr_net = fr_cnn_net()
    lr = 0.001
    loss_func = nn.MSELoss()
    epoch = 50
    #batch = 40
    optimer = optim.Adam(fr_net.parameters(),lr)
    label = torch.zeros(8, 40)
    for i in range(8):
        label[i] = tr_label[i*40:i*40+40]
    input = torch.zeros(40,)
    for i in range(40):
        input = tr_data[i * 8:i * 8 + 8]
    for j in range(epoch):
        out = fr_net(input)
        loss = loss_func(out, label)
        optimer.zero_grad()
        loss.backward()
        optimer.step()
    fr_net.eval()
    out = fr_net(va_data[0:40])
    print(out)
    print(va_label[0:40])