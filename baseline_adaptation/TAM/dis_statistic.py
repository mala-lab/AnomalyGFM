# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  4/25/2023 
# version： Python 3.7.8
# @File : dis_statistic.py
# @Software: PyCharm

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.mlab as mlab


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


dataset_name = 'elliptic'
''''BlogCatalog , Amazon,  YelpChi'''
path = 'data'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.5, 7.5)
data = sio.loadmat('./data/{}/{}.mat'.format(path, dataset_name))

label = data['Label'] if ('Label' in data) else data['gnd']
attr = data['Attributes'] if ('Attributes' in data) else data['X']
network = data['Network'] if ('Network' in data) else data['A']

label = label[0, :]

adj_matrix = network.toarray()
attr = attr.toarray()
normal_node_index = np.where(label == 0)[0]
abnormal_node_index = np.where(label == 1)[0]


def draw_homo(message_all):
    with PdfPages('pdf/{}.pdf'.format(dataset_name)) as pdf:
        mu_0 = np.mean(message_all[0])  # 计算均值
        print('The mean dis of normal-normal is {}'.format(mu_0))
        sigma_0 = np.std(message_all[0])
        mu_1 = np.mean(message_all[1])  # 计算均值
        print('The mean dis of normal-abnormal is {}'.format(mu_1))
        sigma_1 = np.std(message_all[1])
        mu_2 = np.mean(message_all[2])  # 计算均值
        print('The mean dis of abnormal - abnormal is {}'.format(mu_2))
        sigma_2 = np.std(message_all[2])

        message_all2 = [message_all[0], message_all[1]]
        n, bins, patches = plt.hist(message_all2, bins=100, normed=1, label=['N-N', 'N-A'])
        # sns.histplot(message_all, kde=True, stat='probability', color='blue', label=['N-N', 'N-A', 'A-A'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
        y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
        # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
        # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线

        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=8.5)  # 绘制y的曲线
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=8.5)  # 绘制y的曲线
        # plt.plot(bins, y_2, color='green', linestyle='--', linewidth=3.5)  # 绘制y的曲线
        # plt.ylim(0, 100)
        plt.xlabel('Distance', fontsize=50)
        plt.ylabel('Density', size=50)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.legend(loc='upper right', fontsize=30)
        ax = plt.gca()  # 获取边框
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_linewidth('3.0')
        ax.spines['right'].set_linewidth('3.0')
        ax.spines['top'].set_linewidth('3.0')
        ax.spines['left'].set_linewidth('3.0')
        plt.title('{}'.format(dataset_name), fontsize=25)
        plt.show()
        # plt.savefig('{}_dis.pdf'.format(dataset_name), dpi=500, bbox_inches='tight')
        # pdf.savefig()

def calc_dis_edge(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = np.zeros((row, col))
    n_n_dis = []
    n_a_dis = []
    a_a_dis = []
    all_dis = []
    for i in range(row):
        node_index = np.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            dis = np.sqrt(np.sum((attr_matrix[i] - attr_matrix[j]) * (attr_matrix[i] - attr_matrix[j])))
            dis_array[i][j] = dis
            all_dis.append(dis)
            if i in normal_node_index and j in normal_node_index:
                n_n_dis.append(dis)
            elif i in abnormal_node_index and j in abnormal_node_index:
                a_a_dis.append(dis)
            else:
                n_a_dis.append(dis)
    n_n_dis = (n_n_dis - np.min(all_dis)) / (np.max(all_dis) - np.min(all_dis))
    n_a_dis = (n_a_dis - np.min(all_dis)) / (np.max(all_dis) - np.min(all_dis))
    a_a_dis = (a_a_dis - np.min(all_dis)) / (np.max(all_dis) - np.min(all_dis))
    return dis_array, n_n_dis, n_a_dis, a_a_dis


dis_array, n_n_dis, n_a_dis, a_a_dis = calc_dis_edge(adj_matrix, attr)
draw_homo([n_n_dis, n_a_dis, a_a_dis])
