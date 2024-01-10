# --*-- coding:utf-8 --*--
"""
作者：徐柯炎
日期：2023年10月12日
"""
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
from functions import *


# 初始化参数
def init_GMM(dataset, k):
	num = dataset.shape[0]
	mu = dataset[random.sample(range(num), k)]
	var = np.var(dataset, axis=0)
	sigma = np.tile(var, (k, 1))
	alpha = np.ones((k, 1)) / k
	# print(f'init mu{mu}')
	# print(f'init sigma{sigma}')
	# print(f'init alpha{alpha}')
	return alpha, mu, sigma


# 定义高斯密度计算函数
def gauss_density_probability(data, mu, sigma):
	"""
	计算高斯概率密度。
	:param data: shape(1, m)
	:param mu: shape(k, m)
	:param sigma: shape(k, m)
	:return: p: shape(1, k)
	"""
	# 高斯混合函数
	k = mu.shape[0]
	m = mu.shape[1]
	p = np.zeros((1, k))
	for i in range(k):
		B_det = 1
		x_aB = 0
		for j in range(m):
			B_det *= sigma[i, j] * 2 * np.pi
			x_aB += (data[j]-mu[i, j]) ** 2 / sigma[i, j]
		part1 = math.sqrt(B_det)
		part2 = -0.5 * x_aB
		fx_k = 1 / part1 * np.exp(part2)
		p[0][i] = fx_k
	return p


# 更新gama
def e_gama(dataset, mu, alpha, sigma):
	num = dataset.shape[0]
	k = mu.shape[0]
	gama = np.zeros((num, k))

	for i in range(num):
		phi = gauss_density_probability(dataset[i], mu, sigma)
		sum = np.dot(phi, alpha)
		for j in range(k):
			gama[i, j] = alpha[j, 0] * phi[0, j] / sum
	return gama


# 更新参数
def m_theta(dataset, gama, mu):
	m = dataset.shape[1]
	k = gama.shape[1]
	num = dataset.shape[0]
	cal_alpha = np.zeros((k, 1))
	cal_mu = np.zeros((k, m))
	cal_sigma = np.zeros((k, m))
	sum_gama = np.sum(gama, axis=0)

	for i in range(k):
		sum_mu = np.zeros((1, m))
		sum_sigma = np.zeros((1, m))
		for j in range(num):
			data = dataset[j]
			sum_mu += gama[j, i] * data
			sum_sigma += gama[j, i] * (data - mu[i]) ** 2
		cal_mu[i] = sum_mu / sum_gama[i]
		cal_sigma[i] = sum_sigma / sum_gama[i]
		cal_alpha[i] = sum_gama[i] / num
	return cal_alpha, cal_mu, cal_sigma


# 分簇，返回簇和索引
def classify(dataset, mu, gama):
	num = dataset.shape[0]
	k = mu.shape[0]
	cluster = []
	for i in range(k):
		cluster.append([])
	max_index = np.argmax(gama, axis=1)
	for i in range(num):
		cluster[max_index[i]].append(dataset[i])
	return cluster, max_index


# EM算法
def em_algorithm(dataset, k, iteration=1000, threshold=1e-10):
	# alpha shape(k, 1)
	# mu shape(k, m)
	# sigma shape(k, m)
	# gama shape(num, k)
	alpha, mu, sigma = init_GMM(dataset, k)
	num = dataset.shape[0]
	gama = np.zeros((num, k))

	for iter in range(iteration):
		prev_mu = mu
		gama = e_gama(dataset, mu, alpha, sigma)
		alpha, mu, sigma = m_theta(dataset, gama, mu)
		if np.sum(abs(prev_mu - mu)) < threshold:  # 均值基本不变，结束迭代
			print(f'iter: {iter},\nmu={mu},\nsigma={sigma},\nalpha={alpha}')
			break
		if iter % (int(iteration/10)) == 0:
			print(f'iter={iter},\nmu={mu},\nsigma={sigma},\nalpha={alpha}')

	cluster, max_index = classify(dataset, mu, gama)
	return cluster, max_index


# 测试GMM，画图
def test_GMM(cluster, k):
	for i in range(k):
		temp = np.array(cluster[i])
		plt.scatter(temp[:, 0], temp[:, 1], alpha=0.5, label=i)


def test_main():
	k = 3
	dataset = generate_data(plot=True, k=k)
	cluster, max_index = em_algorithm(dataset, k)
	for i in range(k):
		clu = np.array(cluster[i])
		print(f'第{i + 1}个簇有{clu.shape[0]}个样本，分别为：')
		for j in range(clu.shape[0]):
			print(f'{j}: {clu[j]}', end='\t')
		print()
	acc = test_acc(max_index, k)
	print(f'准确率为{round(acc, 3)}')
	test_GMM(cluster, k)
	plt.legend()
	plt.show()


def test_iris():
	k = 3
	dataset = load_iris()
	cluster, max_index = em_algorithm(dataset, k)
	for i in range(k):
		clu = np.array(cluster[i])
		print(f'第{i+1}个簇有{clu.shape[0]}个样本，分别为：')
		for j in range(clu.shape[0]):
			print(f'{j}: {clu[j]}', end='\t')
		print()
	acc = test_acc(max_index, k)
	print(f'准确率为{round(acc, 3)}')
	test_GMM(cluster, k)
	plt.legend()
	plt.show()


if __name__ == '__main__':
	test_main()
	test_iris()
