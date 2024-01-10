# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-19
"""
# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-19
"""
import numpy as np
from multilayer_perceptron import softmax, loss


def SGD(x_train, y_train, alpha=0.03, epoch=1000, threshold=1e-6):
    num = x_train.shape[0]
    ni = x_train.shape[1]
    no = y_train.shape[1]
    w = np.random.randn(ni, no)
    pre_err = 0
    for i in range(epoch):
        err = 0
        for j in range(num):
            x = x_train[j].reshape(1, -1)
            y = y_train[j].reshape(1, -1)
            z = np.dot(x, w)
            h = softmax(z)
            err += loss(h, y)
            w_grad = np.dot(x.T, h - y)
            w -= alpha * w_grad
        err = err / num
        if abs(pre_err - err) <= threshold:
            print(f'iter = {i},\terr = {err}, diff = {err - pre_err}')
            return w
        if i % 100 == 0:
            print(f'iter = {i},\terr = {err}, diff = {err - pre_err}')
        pre_err = err
    return w


def lin_test(w, x_test, y_test):
    cnt = 0
    num = x_test.shape[0]
    predicts = []
    for i in range(num):
        x = x_test[i].reshape(1, -1)
        y = y_test[i].reshape(1, -1)
        z = np.dot(x, w)
        h = softmax(z)
        predict = np.argmax(h)
        predicts.append(predict)
        if y_test[i][predict] == 1:
            cnt += 1
    predicts = np.array(predicts)
    return cnt/num, predicts
