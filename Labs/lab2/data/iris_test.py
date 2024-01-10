# --*-- coding:utf-8 --*--
"""
作者：徐柯炎
日期：2023年10月07日
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from logistic_regression import *
from ucimlrepo import fetch_ucirepo

# 从uci获取iris数据集
iris = fetch_ucirepo(id=53)

# 数据（pd.dataframe格式）
X = iris.data.features
y = iris.data.targets
# print(X.info())
# print(y.iloc[0])
# print(y.iloc[55])
# 这里进行前两种花的二分类
y = y[0:100]
num = y.shape[0]
# print(num)
# print(y)
# print(X[y['class'] == 'Iris-setosa'])
# print(X[50:100])
label = np.zeros((1, num))
# 数据标注
for i in range(100):
	label[0][i] = 0 if y.loc[i, 'class'] == 'Iris-setosa' else 1
# print(label)
X = np.array(X).T


# 分割训练集和测试集
def train_test_split(X, y):
	X = np.vstack((X, np.ones((1, X.shape[1]))))
	x_train = np.hstack((X[:, 0:40], X[:, 50:90]))
	y_train = np.hstack((y[:, 0:40], y[:, 50:90]))
	x_test = np.hstack((X[:, 40:50], X[:, 90:100]))
	y_test = np.hstack((y[:, 40:50], y[:, 90:100]))
	# print(x_test, y_test)
	return x_train, x_test, y_train, y_test


# 画出直观的分类图，x轴为sepal length，y轴sepal width
def plt_iris(X, y):
	x1 = []
	x2 = []
	for i in range(X.shape[1]):
		if y[i] == 0:
			x1.append(X[:, i])
		else:
			x2.append(X[:, i])
	x1 = np.array(x1)
	x2 = np.array(x2)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_xlabel('sepal length')
	ax.set_ylabel('sepal width')
	ax.scatter(x1[:, 0], x1[:, 1], marker='.', label='Iris-setosa')
	ax.scatter(x2[:, 0], x2[:, 1], marker='.', label='Iris-versicolor')


# 测试准确率，并返回准确率和测试标签
def test_acc(x_test, y_test, theta):
	test_label = sigmoid(theta, x_test)[0]
	num = test_label.shape[0]
	# print(test_label)
	label = np.zeros(num)
	cnt = 0
	for i in range(num):
		label[i] = 1 if test_label[i] >= 0.5 else 0
		if label[i] == y_test[0][i]:
			cnt += 1
	# print(label)
	return cnt / num, label


if __name__ == '__main__':
	x_train, x_test, y_train, y_test = train_test_split(X, label)

	# 参数如下
	m = x_train.shape[0] - 1
	num = x_train.shape[1]
	alpha = 0.01
	iter = 1000
	lamuda = 1
	threshold = 1e-3

	# 梯度下降
	theta = gradient_descent(x_train, y_train, alpha=alpha, iter=iter, lamuda=lamuda, threshold=threshold, m=m)

	# 在测试集上测试
	# acc, label = test_acc(x_test, y_test, theta)
	X = np.hstack((x_train, x_test))
	y = np.hstack((y_train, y_test))
	acc, label = test_acc(X, y, theta)
	print(f'准确率为：{round(acc, 3)}')

	# 画图
	plt_iris(X[0:2, :], label)
	plt.legend()
	plt.show()
