# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  8/31/2022 
# version： Python 3.7.8
# @File : one_class_homo.py
# @Software: PyCharm
# @Institution: SMU

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.5, 7.5)
dataset_name = 'Facebook'

data = sio.loadmat('./data/{}.mat'.format(dataset_name))

label = data['Label'] if ('Label' in data) else data['gnd']

attr = data['Attributes'] if ('Attributes' in data) else data['X']
network = data['Network'] if ('Network' in data) else data['A']

attr = attr.toarray()
adj_matrix = network.toarray()

label = label[:, 0]
print(Counter(label))

plt.rcParams['figure.figsize'] = (8.5, 7.5)
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


def draw_homo(message_all, i):
    with PdfPages('pdf/BlogCatalog.pdf') as pdf:
        mu_0 = np.mean(message_all[0])  # 计算均值
        print('The mean of normal node is {}'.format(mu_0))
        sigma_0 = np.std(message_all[0])
        print('The std of normal node is {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])  # 计算均值
        print('The mean of abnormal node is {}'.format(mu_1))
        sigma_1 = np.std(message_all[1])
        print('The std of abnormal node is {}'.format(sigma_1))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
        # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
        # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
        plt.ylim(0, 100)
        plt.xlabel('Homophily', fontsize=25)
        plt.ylabel('Number of Samples', size=25)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.legend(loc='upper left', fontsize=30)
        plt.title('{}_{}'.format(dataset_name, i), fontsize=25)
        plt.title('Facebook', fontsize=25)
        plt.show()
        # plt.savefig('fb_homo.png', dpi=500, bbox_inches='tight')


def cmpute_home(adj_matrix, anomaly_ids):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    num_ano_list = []

    for i in range(row):
        node_index = np.argwhere(adj_matrix[i, :] == 1)[:, 0]
        num_ano = len(set(node_index) & set(anomaly_ids))
        if len(node_index) != 0:
            num_ano_list.append(num_ano / len(node_index))
        else:
            num_ano_list.append(-1)
    return np.array(num_ano_list)




normal_node_index = np.where(label == 0)[0]
abnormal_node_index = np.where(label == 1)[0]

homo1 = cmpute_home(adj_matrix, normal_node_index)
homo2 = cmpute_home(adj_matrix, abnormal_node_index)
normal_node_degree = homo1[normal_node_index]
abnormal_node_degree = homo2[abnormal_node_index]

data = [normal_node_degree, abnormal_node_degree]
fig, ax = plt.subplots()
ax.boxplot(data)
# plt.ylim(0, 1)
ax.set_xticklabels(["normal", "abnormal"])  # 设置x轴刻度标签
plt.title('Node Homophily')
plt.show()

draw_homo(data, 0)
