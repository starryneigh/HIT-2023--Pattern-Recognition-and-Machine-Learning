# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-20
"""
import numpy as np
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo


def gen_data(num=200, plot=True, k=3):
    mu = np.array([[2, -2], [-2, 2]])
    cov = np.array([[[2, 0], [0, 2]], [[2, 0], [0, 2]]])

    X1 = gen_circle(num, 6, 0, 0)
    X2 = np.zeros(((k - 1) * num, 2))
    for i in range(k - 1):
        X2[i * num:(i + 1) * num, :] = np.random.multivariate_normal(mu[i], cov[i, :, :], num)
    X = np.vstack((X2, X1))
    dic = {}
    y = np.zeros((k * num, k))
    for i in range(k):
        y[i * num:(i + 1) * num, i] = 1
        dic[i] = i

    if plot:
        fig = plt.figure(figsize=(5, 5))
        for i in range(k):
            plt.scatter(X[i * num:(i + 1) * num, 0], X[i * num:(i + 1) * num, 1], marker='.', label=i)
        plt.legend()
        plt.show()

    X, y = shuffle_index(X, y)
    ni = 2
    no = k

    return X, y, dic, ni, no


def load_iris(id=53, plot=True):
    # 从uci获取iris数据集
    iris = fetch_ucirepo(id=id)
    # 数据（pd.dataframe格式）
    X = np.array(iris.data.features)
    y = np.array(iris.data.targets)
    X, y = shuffle_index(X, y)
    y, dic = relabel_y(y)
    ni = X.shape[1]
    no = y.shape[1]
    return X, y, dic, ni, no


def gen_circle(num=200, r=7, a=1, b=0, mu=0, sigma=0.2):
    mu = np.array([mu, mu])
    cov = np.array([[sigma, 0], [0, sigma]])
    X = np.random.multivariate_normal(mu, cov, num)
    dot = np.linspace(0, 360, num)
    X[:, 0] += r * np.sin(dot / 360 * 2 * np.pi) + a
    X[:, 1] += r * np.cos(dot / 360 * 2 * np.pi) + b
    return X


def relabel_y(y):
    num = y.shape[0]
    label_dic = {}
    cnt = 0

    # 将字符串标签编码成数字
    for i in range(num):
        label = str(y[i, 0])
        if label not in label_dic:
            label_dic[label] = cnt
            cnt += 1
    # print(label_dic)

    m = len(label_dic)
    index_y = np.zeros((num, m))
    for i in range(num):
        label_index = label_dic[str(y[i, 0])]
        index_y[i, label_index] = 1
    # print(index_y)
    return index_y, label_dic


def train_test_split(X, y, k=0.8):
    num = y.shape[0]
    split = int(k * num)
    x_train = X[:split]
    y_train = y[:split]
    x_test = X[split:]
    y_test = y[split:]
    return x_train, x_test, y_train, y_test


def shuffle_index(X, y):
    num = X.shape[0]
    shuffled_index = np.random.permutation(num)
    X = X[shuffled_index]
    y = y[shuffled_index]
    return X, y
