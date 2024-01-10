# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-21
"""
import numpy as np

from SGD_BGD import BGD_mlp, SGD_mlp
from process_data import gen_data, train_test_split


if __name__ == '__main__':
    X, y, dic, ni, no = gen_data(num=800, plot=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    np.random.seed(20)
    BGD_mlp(x_train, x_test, y_train, y_test, dic, ni, no, batch=32)
    np.random.seed(20)
    SGD_mlp(x_train, x_test, y_train, y_test, dic, ni, no)
