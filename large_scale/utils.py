import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
from collections import Counter


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def preprocess_features_tensor(features):
    rowsum = torch.sum(features)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    features = features * r_inv
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset_train, dataset_test, train_rate=0.3, val_rate=0.1):
    # Load training and test dataset
    """Load .mat dataset."""
    # data_train = sio.loadmat("data/data/{}.mat".format(dataset_train))
    # data_test = sio.loadmat("data/data/{}.mat".format(dataset_test))
    data_train = sio.loadmat("/GADPrompt/node_level/data/data/{}.mat".format(dataset_train))
    data_test = sio.loadmat("/GADPrompt/node_level/data/data/{}.mat".format(dataset_test))

    label_train = data_train['Label'] if ('Label' in data_train) else data_train['gnd']
    attr_train = data_train['Attributes'] if ('Attributes' in data_train) else data_train['X']
    network_train = data_train['Network'] if ('Network' in data_train) else data_train['A']

    label_test = data_test['Label'] if ('Label' in data_test) else data_test['gnd']
    attr_test = data_test['Attributes'] if ('Attributes' in data_test) else data_test['X']
    network_test = data_test['Network'] if ('Network' in data_test) else data_test['A']

    adj_train = sp.csr_matrix(network_train)
    feat_train = sp.lil_matrix(attr_train)

    adj_test = sp.csr_matrix(network_test)
    feat_test = sp.lil_matrix(attr_test)

    # labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    # num_classes = np.max(labels) + 1
    # labels = dense_to_one_hot(labels, num_classes)

    all_labels = np.squeeze(np.array(label_train))
    ano_labels_test = np.squeeze(np.array(label_test))
    # import collections
    # data_count2 = collections.Counter(ano_labels)
    # print(data_count2)

    num_node_train = adj_train.shape[0]

    num_train = int(num_node_train * train_rate)
    num_val = int(num_node_train * val_rate)
    all_idx_train = list(range(num_node_train))
    random.shuffle(all_idx_train)

    idx_train = all_idx_train[: num_train]
    idx_val = all_idx_train[num_train: num_train + num_val]

    labels_train = all_labels[idx_train]
    labels_val = all_labels[idx_val]

    # print('Training', Counter(np.squeeze(idx_train)))
    # print('Test', Counter(np.squeeze(ano_labels_test)))

    return adj_train, adj_test, feat_train, feat_test, labels_train, labels_val, ano_labels_test, idx_train, idx_val


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1,
                                                           max_nodes_per_seed=subgraph_size * 3)
    subv = []

    for i, trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace), sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9,
                                                                      max_nodes_per_seed=subgraph_size * 5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]), sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time > 10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size * 3]
        subv[i].append(i)

    return subv


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

matplotlib.use('Agg')
plt.rcParams['figure.dpi'] = 300  # 图片像素
plt.rcParams['figure.figsize'] = (8.5, 7.5)
# plt.rcParams['figure.figsize'] = (10.5, 9.5)
from matplotlib.backends.backend_pdf import PdfPages


def draw_pdf(message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))
    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    # plt.ylim(0, 20)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    # from matplotlib.pyplot import MultipleLocator
    # x_major_locator = MultipleLocator(0.02)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}_{}.svg'.format(dataset, dataset, epoch))
    plt.close()


def draw_pdf_methods(method, message_normal, message_abnormal, message_real_abnormal, dataset, epoch):
    message_all = [np.squeeze(message_normal), np.squeeze(message_abnormal), np.squeeze(message_real_abnormal)]
    mu_0 = np.mean(message_all[0])  # 计算均值
    sigma_0 = np.std(message_all[0])
    # print('The mean of normal {}'.format(mu_0))
    # print('The std of normal {}'.format(sigma_0))
    mu_1 = np.mean(message_all[1])  # 计算均值
    sigma_1 = np.std(message_all[1])
    # print('The mean of abnormal {}'.format(mu_1))
    # print('The std of abnormal {}'.format(sigma_1))
    mu_2 = np.mean(message_all[2])  # 计算均值
    sigma_2 = np.std(message_all[2])
    # print('The mean of abnormal {}'.format(mu_2))
    # print('The std of abnormal {}'.format(sigma_2))

    n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Outlier', 'Abnormal'])
    y_0 = mlab.normpdf(bins, mu_0, sigma_0)  # 拟合一条最佳正态分布曲线y
    y_1 = mlab.normpdf(bins, mu_1, sigma_1)  # 拟合一条最佳正态分布曲线y
    y_2 = mlab.normpdf(bins, mu_2, sigma_2)  # 拟合一条最佳正态分布曲线y
    # plt.plot(bins, y_0, 'g--', linewidth=3.5)  # 绘制y的曲线
    # plt.plot(bins, y_1, 'r--', linewidth=3.5)  # 绘制y的曲线
    plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.plot(bins, y_2, color='green', linestyle='--', linewidth=7.5)  # 绘制y的曲线
    plt.ylim(0, 8)

    # plt.xlabel('RAW-based Affinity', fontsize=25)
    # plt.xlabel('TAM-based Affinity', fontsize=25)
    # plt.ylabel('Number of Samples', size=25)

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=25)
    # plt.legend(loc='upper left', fontsize=30)
    # plt.title('Amazon'.format(dataset), fontsize=25)
    # plt.title('BlogCatalog', fontsize=50)
    plt.savefig('fig/{}/{}2/{}_{}.svg'.format(method, dataset, dataset, epoch))
    plt.close()
