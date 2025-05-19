# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  8/31/2022 
# version： Python 3.7.8
# @File : degree_nsgt.py
# @Software: PyCharm
# @Institution: SMU
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from collections import Counter
from sklearn.metrics import roc_auc_score
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


np.random.seed(1)
dataset_name = 'Facebook'
data = sio.loadmat('./data/{}.mat'.format(dataset_name))

label = data['Label'] if ('Label' in data) else data['gnd']

attr = data['Attributes'] if ('Attributes' in data) else data['X']
network = data['Network'] if ('Network' in data) else data['A']

# label = data['label'] if ('label' in data) else data['gnd']
# attr = data['features'] if ('features' in data) else data['X']
# network = data['net_upu'] if ('net_upu' in data) else data['A']

if dataset_name in ['Amazon', 'YelpChi', 'Amazon-all', 'YelpChi-all']:
    attr, _ = preprocess_features(attr)
attr_matrix = attr.toarray()

adj_matrix = network.toarray()
label = np.squeeze(np.array(label))
print(Counter(label))

def calc_degree(adj_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = np.zeros((row, col))
    min_dis = 100
    max_dis = 0
    degree = np.sum(adj_matrix, axis=1)
    for i in range(row):
        print(i)
        node_index = np.argwhere(adj_matrix[i, :] == 1)[:, 0]
        for j in node_index:
            dis = np.abs(degree[i] - degree[j])
            dis_array[i][j] = dis
            if dis > max_dis:
                max_dis = dis_array[i][j]
            if dis < min_dis:
                min_dis = dis_array[i][j]
    return dis_array, max_dis, min_dis


def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def calc_sim(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = np.zeros((row, col))
    for i in range(row):
        # print(i)
        node_index = np.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            dis = get_cos_similar(attr_matrix[i].tolist(), attr_matrix[j].tolist())
            dis_array[i][j] = dis

    return dis_array


def calc_dis(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = np.zeros((row, col))
    min_dis = 100
    max_dis = 0
    for i in range(row):
        print(i)
        node_index = np.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            dis = np.sqrt(np.sum((attr_matrix[i] - attr_matrix[j]) * (attr_matrix[i] - attr_matrix[j])))
            dis_array[i][j] = dis
            if dis > max_dis:
                max_dis = dis_array[i][j]
            if dis < min_dis:
                min_dis = dis_array[i][j]
    return dis_array, max_dis, min_dis


def graph_nsgt(adj_matrix, dis_array):
    row = adj_matrix.shape[0]
    new_adj_matrix = adj_matrix.copy()
    # max_dis = dis_array.max()
    # min_dis = dis_array[dis_array != 0].min()
    dis_array_u = dis_array * adj_matrix
    mean_dis = dis_array_u[dis_array_u != 0].mean()
    for i in range(row):
        node_index = np.argwhere(new_adj_matrix[i, :] > 0)
        node_index = node_index.reshape(node_index.shape[0])

        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            # min_dis = dis_array[i, node_index].min()
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = np.argwhere(dis_array[i, node_index] > random_value)
                cutting_edge = cutting_edge.reshape(cutting_edge.shape[0])
                if cutting_edge.shape[0] != 0:
                    new_adj_matrix[i, node_index[cutting_edge]] = 0


    new_adj_matrix = new_adj_matrix + new_adj_matrix.T
    new_adj_matrix[new_adj_matrix > 1] = 1

    return new_adj_matrix

# raw feature nsgt
dis_array, max_dis, min_dis = calc_dis(adj_matrix, attr_matrix)
origin_adj = adj_matrix
N_t = 10
for i in range(N_t):
    new_adj_matrix = graph_nsgt(adj_matrix, dis_array)
    # score1 = np.sum(origin_adj, 0)
    # print(score1.tolist())
    # print(np.sum(origin_adj, 0))
    # print(np.sum(new_adj_matrix, 0))
    score1 = (np.sum(origin_adj, 0) - np.sum(new_adj_matrix, 0)) / (np.sum(origin_adj, 0))
    adj_matrix = new_adj_matrix
    score1[np.isinf(score1)] = 0.
    score1[np.isnan(score1)] = 0.
    score = score1
    auc = roc_auc_score(label, score)
    # print(score.tolist())
    print('AUC:{:.4f}'.format(auc))
    AP = average_precision_score(label, score, average='macro', pos_label=1, sample_weight=None)
    print('AP:', AP)
    normal_node_index = np.where(label == 0)[0]
    abnormal_node_index = np.where(label == 1)[0]
    normal_node_degree = score[normal_node_index]
    abnormal_node_degree = score[abnormal_node_index]

    data = [normal_node_degree, abnormal_node_degree]
    fig, ax = plt.subplots()
    ax.boxplot(data)
    # plt.ylim(0, 10)
    ax.set_xticklabels(["normal", "abnormal"])  # 设置x轴刻度标签

    plt.title('Node Degree')
    plt.show()


