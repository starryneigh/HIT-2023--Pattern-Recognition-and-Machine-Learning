# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-20
"""
from multilayer_perceptron import *
from linear_classfication import *
from process_data import *
from main import mlp
import random
import numpy as np


def test_nh():
    for nh in nh_list:
        acc_i, cost_i, iter_i = mlp(x_train, x_test, y_train, y_test, dic, ni, no, nh)
        acc.append(acc_i)
        cost.append(cost_i)
        iter.append(iter_i)
    for i in range(len(acc)):
        print(f'隐藏层节点个数为：{nh_list[i]}, 迭代次数为：{iter[i]}, '
              f'准确率为：{acc[i]}, 用时为：{cost[i]}s')


def test_alpha():
    for alpha in alpha_list:
        acc_i, cost_i, iter_i = mlp(x_train, x_test, y_train, y_test, dic, ni, no, alpha=alpha)
        acc.append(acc_i)
        cost.append(cost_i)
        iter.append(iter_i)
    for i in range(len(acc)):
        print(f'学习率为：{round(alpha_list[i], 2)}, 迭代次数为：{iter[i]}, '
              f'准确率为：{acc[i]}, 用时为：{cost[i]}s')


def test_threshold():
    for threshold in thre_list:
        acc_i, cost_i, iter_i = mlp(x_train, x_test, y_train, y_test, dic, ni, no, threshold=threshold)
        acc.append(acc_i)
        cost.append(cost_i)
        iter.append(iter_i)
    for i in range(len(acc)):
        print(f'收敛域为：{thre_list[i]}, 迭代次数为：{iter[i]}, '
              f'准确率为：{acc[i]}, 用时为：{cost[i]}s')


if __name__ == '__main__':
    np.random.seed()
    X, y, dic, ni, no = gen_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    nh_list = np.arange(3, 20, 3)
    alpha_list = np.arange(0.01, 0.09, 0.01)
    thre_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    acc = []
    cost = []
    iter = []
    # test_nh()
    # test_alpha()
    test_threshold()
