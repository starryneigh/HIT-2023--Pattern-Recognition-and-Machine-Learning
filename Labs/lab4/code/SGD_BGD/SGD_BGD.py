# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-21
"""
import time
import SGD
import BGD
from functions import predict_show


def SGD_mlp(x_train, x_test, y_train, y_test, dic, ni, no, nh=10, epoch=10, threshold=1e-3, alpha=0.03):
    start = time.time()
    n = SGD.NN(ni, nh, no)
    iter = n.train(x_train, y_train, epoch=epoch, threshold=threshold, alpha=alpha, use_thr=False)
    end = time.time()
    cost = round(end - start, 3)
    acc, predicts = n.test(x_test, y_test)
    print(f'准确率为：{round(acc, 3)}, 用时为：{cost}s')
    predict_show(predicts, x_test, y_test, dic)
    print()
    return acc, cost, iter


def BGD_mlp(x_train, x_test, y_train, y_test, dic, ni, no, nh=10, epoch=10, threshold=1e-3, alpha=0.03, batch=16):
    start = time.time()
    n = BGD.NN(ni, nh, no, batch=batch)
    iter = n.train(x_train, y_train, epoch=epoch, threshold=threshold, alpha=alpha, use_thr=False)
    end = time.time()
    cost = round(end - start, 3)
    acc, predicts = n.test(x_test, y_test)
    print(f'准确率为：{round(acc, 3)}, 用时为：{cost}s')
    predict_show(predicts, x_test, y_test, dic)
    print()
    return acc, cost, iter