# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-19
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
    return l[0]


def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h


# sigmoid函数求导
def dsigmoid(h):
    return h * (1 - h)


class NN:
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni + 1  # 增加一个偏差节点
        self.nh = nh
        self.no = no

        # 激活神经网络的所有节点
        self.x = np.ones((1, self.ni))
        # print(self.x)
        self.h1 = np.ones((1, self.nh))
        self.h2 = np.ones((1, self.no))

        # 建立权重（矩阵）, 设为随机值
        self.w1 = np.random.randn(self.ni, self.nh)
        self.w2 = np.random.randn(self.nh, self.no)

    def __str__(self):
        return (f'输入层节点个数为（包括偏差节点）：{self.ni}，'
                f'隐藏层节点个数为：{self.nh}，'
                f'输出层节点个数为：{self.no}。')

    def update(self, input_x):
        # print(input_x)
        self.x = np.hstack((input_x, np.array([[1.0]])))
        z1 = np.dot(self.x, self.w1)
        self.h1 = sigmoid(z1)
        z2 = np.dot(self.h1, self.w2)
        self.h2 = softmax(z2)
        return self.h2

    def back_propagation(self, input_y, alpha):
        z2_grad = self.h2 - input_y
        w2_grad = np.dot(self.h1.T, z2_grad)

        h1_grad = np.dot(z2_grad, self.w2.T)
        z1_grad = h1_grad * dsigmoid(self.h1)
        w1_grad = np.dot(self.x.T, z1_grad)

        self.w2 -= alpha * w2_grad
        self.w1 -= alpha * w1_grad

        err = loss(self.h2, input_y)
        return err

    def train(self, x_train, y_train, epoch=30000, alpha=0.03, threshold=1e-6):
        num = x_train.shape[0]
        pre_err = 0
        for i in range(epoch):
            err = 0
            for j in range(num):
                self.update(x_train[j].reshape(1, -1))
                err += self.back_propagation(y_train[j].reshape(1, -1), alpha)
            err = err / num
            if abs(pre_err - err) <= threshold:
                print(f'iter = {i},\terr = {err}, diff = {err - pre_err}')
                return i
            if i % 100 == 0:
                print(f'iter = {i},\terr = {err}, diff = {err - pre_err}')
            pre_err = err
        return epoch

    def test(self, x_test, y_test):
        cnt = 0
        num = x_test.shape[0]
        predicts = []
        for i in range(num):
            predict = self.update(x_test[i].reshape(1, -1))
            predict = np.argmax(predict)
            predicts.append(predict)
            if y_test[i][predict] == 1:
                cnt += 1
        predicts = np.array(predicts)
        return round(cnt/num, 3), predicts


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


