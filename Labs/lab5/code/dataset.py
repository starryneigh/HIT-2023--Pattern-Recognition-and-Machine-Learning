# --*-- coding:utf-8 --*--
"""
@Filename: dataset.py
@Author: Keyan Xu
@Time: 2023-11-03
"""
import gzip
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms


class Mnist(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, data_name, label_name, transform=None):
        # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        (train_set, train_labels) = self.load_data(folder, data_name, label_name)
        self.train_set = torch.from_numpy(train_set.copy())
        self.train_labels = torch.from_numpy(train_labels.copy())
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

    def load_data(self, data_folder, data_name, label_name):
        with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
            y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

        with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
            x_train = np.frombuffer(
                imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
        return x_train, y_train


if __name__ == '__main__':
    transform = transforms.ToTensor()
    folder = '../data/mnist/MNIST/raw'
    train_data_name = "train-images-idx3-ubyte.gz"
    train_label_name = "train-labels-idx1-ubyte.gz"
    test_data_name = "t10k-images-idx3-ubyte.gz"
    test_label_name = "t10k-labels-idx1-ubyte.gz"
    train_data = Mnist(folder, train_data_name, train_label_name, transform=transform)
    test_data = Mnist(folder, train_data_name, train_label_name, transform=transform)
    print(train_data.train_set.size())



