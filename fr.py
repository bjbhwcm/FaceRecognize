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
    faces = torch.zeros(400,1,57,47)
    for i in range(20):
        for j in range(20):
            for x in range(57):
                for y in range(47):
                    faces[i*20 + j][0][x][y] = img_tensor[0][i*57+x][j*47+y]
    labels = torch.zeros(400)
    for i in range(40):
        labels[i*10:i*10+10] = i
    return faces,labels

def split_data_set(faces,labels):
    tr_data=torch.zeros(320,1,57,47)
    tr_label=torch.zeros(320)
    va_data=torch.zeros(80,1,57,47)
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
            nn.Conv2d(1,20,(5,5),stride=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(20, 50, (5,5), stride=1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            Flatten(),
            nn.Linear(50*11*8,2000),
            nn.ReLU(True),
            #nn.Dropout(0.4),
            nn.Linear(2000,40)
        )
    def forward(self, input):
        out = self.fowarding(input)
        return out

def training_nn():
    faces, labels = load_pic_data(olive_pic_path)
    [(tr_data, tr_label), (va_data, va_label)] = split_data_set(faces, labels)
    tr_data = Variable(tr_data)
    tr_label = Variable(tr_label)
    va_data = Variable(va_data)
    va_label = Variable(va_label)
    fr_net = fr_cnn_net()
    fr_net = fr_net.cuda()
    tr_data = tr_data.cuda()
    tr_label = tr_label.cuda()
    va_data = va_data.cuda()
    va_label = va_label.cuda()
    lr = 0.001
    loss_func = nn.MSELoss()
    # loss_func = nn.CrossEntropyLoss()
    epoch = 56
    optimer = optim.Adam(fr_net.parameters(), lr)  # ,weight_decay=1e-6
    for j in range(epoch):
        for i in range(8):
            inputx = tr_data[i * 40:i * 40 + 40]
            label = torch.LongTensor(40, 1)
            for x in range(40):
                label[x][0] = int(tr_label[i * 40 + x])
            one_hot = torch.zeros(40, 40).scatter_(1, label, 1)
            one_hot = Variable(one_hot)
            inputx = inputx.cuda()
            one_hot = one_hot.cuda()
            # print(one_hot)
            # print(label)
            out = fr_net(inputx)
            loss = loss_func(out, one_hot)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
    for i in range(40):
        fr_net.eval()
        out = fr_net(va_data[i:i + 1])
        print(out)
        print(torch.topk(out, 1)[1].squeeze(1))
        print(va_label[i])
    s = input("s:")
    if s == 's':
        torch.save(fr_net, "E:\\FR.NN")

def test(fr_net):
    faces, labels = load_pic_data(olive_pic_path)
    [(tr_data, tr_label), (va_data, va_label)] = split_data_set(faces, labels)
    tr_data = Variable(tr_data)
    tr_label = Variable(tr_label)
    va_data = Variable(va_data)
    va_label = Variable(va_label)
    tr_data = tr_data.cuda()
    tr_label = tr_label.cuda()
    va_data = va_data.cuda()
    va_label = va_label.cuda()
    fr_net.eval()
    count = 0
    for i in range(80):
        fr_net.eval()
        out = fr_net(va_data[i:i + 1])
        #print(out)
        #print(torch.topk(out, 1)[1].squeeze(1).item())
        #print(va_label[i].item())
        if torch.topk(out, 1)[1].squeeze(1).item() == va_label[i].item():
            count = count + 1
    print("correct:"+ str(count/80 * 100) + "%")


if __name__=="__main__":
    fr_net = torch.load("E:\\FR.NN")
    test(fr_net)