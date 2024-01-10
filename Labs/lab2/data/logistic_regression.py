# --*-- coding:utf-8 --*--
"""
作者：徐柯炎
日期：2023年10月07日
"""
import math
import numpy as np
import matplotlib.pyplot as plt


# sigmoid函数
def sigmoid(theta, x):
	z = np.dot(theta, x)
	sig = 1 / (1 + np.exp(-z))
	return sig


# 生成数据（符合贝叶斯假设、不符合贝叶斯假设）
def generate_data(num=1000, m=2, plot=True):
	pos_mean = [1, 2]  # 正例的两维度均值
	neg_mean = [-1, -2]  # 反例的两维度均值
	X = np.zeros((2 * num, 2))
	Y = np.zeros((2 * num, 1))
	# 生成符合贝叶斯假设的数据
	cov = np.mat([[1, 0], [0, 1.5]])
	# 生成不符合贝叶斯假设的数据
	# cov = np.mat([[1, 0.5], [0.5, 1.5]])
	X[:num, :] = np.random.multivariate_normal(pos_mean, cov, num)
	X[num:, :] = np.random.multivariate_normal(neg_mean, cov, num)
	Y[:num] = 1
	Y[num:] = 0
	if plot:
		plt.scatter(X[:num, 0], X[:num, 1], c='b', marker='.', label='pos')
		plt.scatter(X[num:, 0], X[num:, 1], c='r', marker='.', label='neg')
	b = np.ones((2 * num, 1))
	X = np.hstack((X, b))
	# print(X.T.shape, Y.T.shape)
	return X.T, Y.T


# 损失函数，带正则项(极大似然）,并对loss做归一化处理
def loss(X, Y, theta, lamuda=0):
	"""
	:param theta: shape(1, m+1)
	:param X: shape(m+1, num)
	:param Y: shape(1, num)
	:param lamuda: penalty term
	:return: loss
	"""
	num = X.shape[1]
	theta_x = np.dot(theta, X)  # shape(1, num)
	part1 = np.dot(Y, theta_x.T)
	temp = np.log(1 + np.exp(theta_x))
	part2 = np.sum(temp)
	Loss = part1 - part2 - lamuda * np.dot(theta, theta.T) / 2
	return -Loss[0][0] / num


# 求解梯度
def gradient_theta(x_train, y_train, theta, lamuda):
	gradient = np.dot(sigmoid(theta, x_train) - y_train, x_train.T) + lamuda / x_train.shape[1] * np.sum(theta)
	return gradient


# 梯度下降
def gradient_descent(x_train, y_train, alpha=0.01, m=2, iter=1000, lamuda=1, threshold=1e-5):
	"""
	:param threshold:
	:param m: degree
	:param x_train: shape(m+1, num)
	:param y_train: shape(1, num)
	:param alpha: learning rate
	:param iter: iteration
	:param lamuda: penalty term
	:return: theta
	"""
	# 初始化theta
	# theta: shape(1, m+1)
	theta = np.random.rand(m + 1).reshape(1, m + 1)
	cost = 1e10
	for i in range(iter):
		# 当前梯度
		gradient = gradient_theta(x_train, y_train, theta, lamuda)
		# 更新参数
		theta = theta - alpha * gradient
		# 终止迭代的条件
		if abs(cost - loss(x_train, y_train, theta, lamuda)) < threshold:
			cost = loss(x_train, y_train, theta, lamuda)
			print(f'iter: {i},\tcost: {cost}')
			break
		# 计算损失
		cost = loss(x_train, y_train, theta, lamuda)
		if i % 10 == 0:
			print(f'iter: {i},\tcost: {cost}')
	print(f'拟合的参数theta分别为：', end='')
	for t in theta[0]:
		t = round(t, 5)
		print(t, end=' ')
	print()
	return theta


def test_main():
	# 参数如下
	m = 2
	num = 1000
	alpha = 0.01
	iter = 1000
	threshold = 1e-7
	lamuda = 1
	dit = {'m': 2, 'num': 1000, 'alpha': 0.01, 'iter': 1000, 'lamuda': 1, 'threshold': 1e-5}
	x_train, y_train = generate_data(num, m, plot=False)
	theta = gradient_descent(x_train, y_train, alpha=alpha, iter=iter, lamuda=lamuda, threshold=threshold)
	test(theta, num=num)
	x_test, y_test = generate_data(num, m)
	acc, _ = test_acc(x_test, y_test, theta)
	print(f'准确率为：{round(acc, 3)}')
	plt.legend()
	plt.show()


# 测试拟合效果，画图
def test(theta, num=1000, label="test"):
	de = theta.shape[1]
	theta = theta[0]
	b = theta[2]
	x = np.linspace(-4, 4, num).reshape(1, num)
	y = -theta[0] / theta[1] * x - b / theta[1]
	plt.plot(x[0], y[0], label=label)


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
	test_main()
