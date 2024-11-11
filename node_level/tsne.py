# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  10/18/2024 
# version： Python 3.7.8
# @File : tsne.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import datasets
from sklearn.manifold import TSNE
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D


# Visualize the abnormal prompt/pattern

# 对样本进行预处理并画图
def plot_embedding(data, label, title):
    """
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
    colors = np.array(['r', 'g'])
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  # 对数据进行归一化处理
    fig = plt.figure()  # 创建图形实例
    ax = plt.subplot(111)  # 创建子图
    # ax = Axes3D(fig)
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2],
    # 		   color=colors[label])

    # 遍历所有样本
    for i in range(data.shape[0]):
        # for i in range(500):
        print(i)
        # 在图中为每个数据点画出标签
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
        # 		 fontdict={'weight': 'bold', 'size': 7})
        plt.scatter(data[i, 0], data[i, 1], s=20, color=colors[int(label[i])],
                    )
    plt.yticks()
    plt.title(title, fontsize=14)
    # 返回值
    return fig


# def main():
# 	data, label = get_data()		# 调用函数，获取数据集信息
# 	print('Starting compute t-SNE Embedding...')
# 	ts = TSNE(n_components=2, init='pca', random_state=0)
#
# 	reslut = ts.fit_transform(data)
# 	fig = plot_embedding(reslut, label, 't-SNE Embedding of digits')
# 	plt.show()


# main
if __name__ == '__main__':
    dataset = 'amazon'
    data = sio.loadmat('tsne/{}_svd_1400.mat'.format(dataset))
    emb_residual_test = data['emb_residual_test']
    emb_residual_train = data['emb_residual_train']

    label_train = np.squeeze(data['label_train'])
    label_test = np.squeeze(data['label_test'])

    emb_residual_test_normal = emb_residual_test[label_test == 0]

    emb_residual_train_normal = emb_residual_train[label_train == 0]

    emb_residual_test_abnormal = emb_residual_test[label_test == 1]

    emb_residual_train_abnormal = emb_residual_train[label_train == 1]

    emb_residual = np.concatenate((emb_residual_test_abnormal, emb_residual_train_abnormal), 0)
    label = np.concatenate((np.zeros(emb_residual_test_abnormal.shape[0]), np.ones(emb_residual_train_abnormal.shape[0])))

    print('Starting compute t-SNE Embedding...')
    ts = TSNE(n_components=2, init='pca', random_state=1)
    emb = ts.fit_transform(emb_residual)

    fig = plot_embedding(emb, label, 't-SNE Embedding of digits')
    plt.show()
