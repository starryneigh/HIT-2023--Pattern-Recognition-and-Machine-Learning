# --*-- coding:utf-8 --*--
"""
@Filename: .py
@Author: Keyan Xu
@Time: 2023-10-21
"""
import numpy as np
from matplotlib import pyplot as plt

# a = np.linspace(0, 360, 361)
# print(a)
# b = np.sin(a / 360 * 2 * np.pi)
# c = np.cos(a / 360 * 2 * np.pi)
# # plt.scatter(a, b)
# # plt.scatter(a, c)
# plt.scatter(b, c)
# plt.show()


def gen_circle(num=200, r=7, a=1, b=0, mu=0, sigma=0.2):
    mu = np.array([mu, mu])
    cov = np.array([[sigma, 0], [0, sigma]])
    X = np.random.multivariate_normal(mu, cov, num)
    # print(X.shape)
    dot = np.linspace(0, 360, num)
    X[:, 0] += r * np.sin(dot / 360 * 2 * np.pi) + a
    X[:, 1] += r * np.cos(dot / 360 * 2 * np.pi) + b
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.show()
    return X


gen_circle()
k = 3
num = 2
y = np.zeros((num*k, k))
for i in range(k):
    y[i * num:(i+1)*num, i] = 1
print(y)