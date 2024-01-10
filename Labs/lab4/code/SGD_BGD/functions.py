# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-21
"""
import numpy as np
from matplotlib import pyplot as plt


def softmax(z):
    # 计算总和
    sum_exp = np.sum(np.exp(z), axis=1, keepdims=True)
    softmax_z = np.exp(z) / sum_exp
    return softmax_z


def loss(h, y):
    l = -np.sum(y * np.log(h), axis=1)
    return l


def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h


# sigmoid函数求导
def dsigmoid(h):
    return h * (1 - h)


def predict_show(predicts, x_test, y_test, dic):
    num = predicts.shape[0]
    k = y_test.shape[1]
    y = np.argmax(y_test, axis=1)
    # print(y)
    predict_label = []
    true_label = []
    clusters = []
    for i in range(k):
        clusters.append([])
    for i in range(num):
        for key, value in dic.items():
            if value == y[i]:
                true_label.append(key)
            if value == predicts[i]:
                predict_label.append(key)
                clusters[predicts[i]].append(x_test[i])
    for i in range(num):
        print(f'第{i}项的真实标签为：{true_label[i]}，预测标签为：{predict_label[i]}', end='\t')
    print()
    fig = plt.figure(figsize=(5, 5))
    for i in range(k):
        cluster = np.array(clusters[i])
        plt.scatter(cluster[:, 0], cluster[:, 1], label=i, marker='.')
    plt.legend()
    plt.show()
    return predict_label