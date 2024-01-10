# --*-- coding:utf-8 --*--
"""
作者：徐柯炎
日期：2023年10月12日
"""
import numpy as np
import random
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo
from functions import *


def eu_distance(dataset, centroids):
	"""
	计算数据集中每一个节点到每一个中心的欧拉距离
	:param dataset: shape(num, m)
	:param centroids: shape(k, m)
	:return: distance array
	"""
	dislist = []
	for data in dataset:
		# print((data - centroids)**2)
		# print(np.sum((data - centroids)**2, axis=1))
		dis = np.sum((data - centroids) ** 2, axis=1) ** 0.5
		dislist.append(dis)
	dislist = np.array(dislist)
	return dislist


# 进行聚类，并计算新的中心
def cal_cens(dataset, dislist):
	k = dislist.shape[1]
	num = dislist.shape[0]
	cluster = []
	# 初始化簇
	for i in range(k):
		cluster.append([])

	# 计算每一个点到中心的最小距离
	indexs = np.argmin(dislist, axis=1)
	# print(indexs)
	# 将每一个点加入相关的簇
	for i in range(num):
		j = indexs[i]
		cluster[j].append(dataset[i].tolist())
	# print(cluster)

	# 更新中心点
	newcentroids = []
	for i in range(k):
		clu = np.array(cluster[i])
		centroid = np.mean(clu, axis=0)
		newcentroids.append(centroid)
	# print(newcentroids)

	newcentroids = np.array(newcentroids)
	return newcentroids, cluster, indexs


def kmeans(dataset, k):
	"""
	kmeans 算法
	:param dataset: 数据集 shape(num, m)
	:param k: 中心点个数
	:return: 中心点，聚类结果
	"""
	num = dataset.shape[0]
	# 随机化初始点
	init = random.sample(range(num), k)
	centroids = dataset[init]
	# print(centroids)
	flag = 1
	cluster = []
	indexs = np.zeros(num)
	while flag:
		dislist = eu_distance(dataset, centroids)
		newcentroids, cluster, indexs = cal_cens(dataset, dislist)
		change = newcentroids - centroids
		# 迭代结束条件
		if not np.any(change):
			flag = 0
		centroids = newcentroids
	return centroids, cluster, indexs


def test_kmeans(centroids, cluster, k):
	for i in range(k):
		temp = np.array(cluster[i])
		plt.scatter(temp[:, 0], temp[:, 1], alpha=0.5, label=i)
		plt.scatter(centroids[i, 0], centroids[i, 1], c='r')


def test_main():
	k = 3
	num = 1000
	dataset = generate_data(num=num, k=k)
	centroids, cluster, indexs = kmeans(dataset, k)
	test_kmeans(centroids, cluster, k)
	# print(centroids)
	# print(cluster)
	acc = test_acc(indexs, k)
	print(f'准确率为{round(acc, 3)}')
	plt.legend()
	plt.show()


def test_iris():
	dataset = load_iris()
	k = 3
	centroids, cluster, indexs = kmeans(dataset, k)
	test_kmeans(centroids, cluster, k)
	acc = test_acc(indexs, k)
	print(f'准确率为{round(acc, 3)}')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	test_main()
	test_iris()
