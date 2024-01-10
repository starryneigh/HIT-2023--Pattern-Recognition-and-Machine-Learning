# --*-- coding:utf-8 --*--
"""
@Filename: LeNet.py
@Author: Keyan Xu
@Time: 2023-10-30
"""
import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, activate='relu', p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.pool = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(4 * 4 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        print(f'dropout参数p为{p}')
        self.dropout = nn.Dropout(p=p)

        if activate == 'relu':
            print('使用relu函数作为激活函数')
            self.activate = nn.ReLU()
        elif activate == 'sigmoid':
            print('使用sigmoid函数作为激活函数')
            self.activate = nn.Sigmoid()
        elif activate == 'tanh':
            print('使用tanh函数作为激活函数')
            self.activate = nn.Tanh()
        elif activate == 'leaky_relu':
            print('使用leaky_relu函数作为激活函数')
            self.activate = nn.LeakyReLU()
        else:
            print('使用relu函数作为激活函数')
            self.activate = nn.ReLU()

    def forward(self, x):
        x = self.activate(self.conv1(x))
        x = self.pool(x)
        x = self.activate(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.activate(self.linear1(x))
        x = self.dropout(x)
        x = self.activate(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x


