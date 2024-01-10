"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-18
"""
import time
from multilayer_perceptron import *
from linear_classfication import *
from process_data import *
import random
import numpy as np
from ucimlrepo import fetch_ucirepo


def mlp(x_train, x_test, y_train, y_test, dic, ni, no, nh=10, epoch=1000, threshold=1e-3, alpha=0.03):
    start = time.time()
    n = NN(ni, nh, no)
    print(n)
    iter = n.train(x_train, y_train, epoch=epoch, threshold=threshold, alpha=alpha)
    end = time.time()
    cost = round(end - start, 3)
    acc, predicts = n.test(x_test, y_test)
    print(f'准确率为：{round(acc, 3)}, 用时为：{cost}s')
    predict_show(predicts, x_test, y_test, dic)
    print()
    return acc, cost, iter


def linear(x_train, x_test, y_train, y_test, dic):
    start = time.time()
    w = SGD(x_train, y_train, epoch=epoch, threshold=threshold, alpha=alpha)
    end = time.time()
    acc, predicts = lin_test(w, x_test, y_test)
    print(f'准确率为：{round(acc, 3)}, 用时为：{round(end - start, 3)}s')
    predict_show(predicts, x_test, y_test, dic)


if __name__ == '__main__':
    epoch = 1000
    threshold = 1e-3
    alpha = 0.03
    id = 53
    np.random.seed(20)

    X, y, dic, ni, no = gen_data(num=160)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    mlp(x_train, x_test, y_train, y_test, dic, ni, no)
    linear(x_train, x_test, y_train, y_test, dic)

    print()

    X, y, dic, ni, no = load_iris(id=id)
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    linear(x_train, x_test, y_train, y_test, dic)
    mlp(x_train, x_test, y_train, y_test, dic, ni, no)
