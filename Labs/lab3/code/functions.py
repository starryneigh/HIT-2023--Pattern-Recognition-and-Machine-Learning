# --*-- coding:utf-8 --*--
"""
作者：徐柯炎
日期：2023年10月13日
"""
import numpy as np
import random
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo


def load_iris():
	# 从uci获取iris数据集
	iris = fetch_ucirepo(id=53)

	# 数据（pd.dataframe格式）
	X = iris.data.features
	y = iris.data.targets
	num = y.shape[0]
	labels = []
	# 数据标注
	for i in range(num):
		label = y.loc[i, 'class']
		if label in labels:
			continue
		else:
			labels.append(label)
	X = np.array(X)
	# y = np.array(y)
	for i in range(len(labels)):
		plt.scatter(X[50 * i:50 * (i + 1), 0], X[50 * i:50 * (i + 1), 1], label=labels[i])
	plt.legend()
	plt.show()
	# print(X)
	# print(y)
	# print(labels)
	return X


# 生成数据（高斯分布）
def generate_data(num=1000, plot=True, k=2):
	mu = np.array([[1, 2], [-1, -2], [3, -2]])
	X = np.zeros((k * num, 2))
	# 协方差矩阵
	cov = np.array([[[1, 0], [0, 2]], [[2, 0], [0, 1]], [[1, 0], [0, 1]]])
	for i in range(k):
		X[i * num:(i+1)*num, :] = np.random.multivariate_normal(mu[i], cov[i, :, :], num)
	if plot:
		for i in range(k):
			plt.scatter(X[i * num:(i + 1) * num, 0], X[i * num:(i + 1) * num, 1], marker='.', label=i)
		plt.legend()
		plt.show()
	# print(X.T.shape, Y.T.shape)
	return X


def find_label(max_index, left, right, k):
	cnt = np.zeros(k)
	for i in range(left, right):
		if max_index[i] == 0:
			cnt[0] += 1
		elif max_index[i] == 1:
			cnt[1] += 1
		elif max_index[i] == 2:
			cnt[2] += 1
	return np.argmax(cnt)


def test_acc(max_index, k):
	num = int(len(max_index) / k)
	# print(num)
	index = np.zeros(k)
	cnt = 0
	for i in range(k):
		index[i] = find_label(max_index, i * num, (i+1) * num, k)
		for j in range(num):
			if index[i] == max_index[i * num + j]:
				cnt += 1
	return cnt / len(max_index)
