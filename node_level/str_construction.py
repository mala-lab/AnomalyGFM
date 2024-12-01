# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  11/28/2024 
# version： Python 3.7.8
# @File : str_construction.py
# @Software: PyCharm
import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp


def dense_to_sparse(dense_matrix):
    shape = dense_matrix.shape
    row = []
    col = []
    data = []
    for i, r in enumerate(dense_matrix):
        for j in np.where(r > 0)[0]:
            row.append(i)
            col.append(j)
            data.append(dense_matrix[i, j])

    sparse_matrix = sp.coo_matrix((data, (row, col)), shape=shape).tocsc()
    return sparse_matrix


def build_graph_from_embeddings(embeddings, threshold):
    # extract the mat info
    print()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    # 计算余弦相似性
    cosine_similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
    distances = 1 - cosine_similarities  # 转换为距离

    adjacency_matrix = (distances < threshold).astype(np.float32)
    np.fill_diagonal(adjacency_matrix, 0)  # 对角线不连边

    return adjacency_matrix


if __name__ == "__main__":
    """Load .mat dataset."""
    dataset = 'PROTEINS'
    data_train = sio.loadmat("data/graph_emb/{}.mat".format(dataset))

    label_train = data_train['Label']
    embeddings = data_train['emb']

    threshold = 0.5  # 距离阈值

    # 构建图
    adjacency_matrix = build_graph_from_embeddings(embeddings, threshold)

    # Pack & save them into .mat
    print('Saving mat file...')
    attribute = dense_to_sparse(embeddings)
    adj = dense_to_sparse(adjacency_matrix)

    sio.savemat('data/graph_str/{}.mat'.format(dataset), {'Network': adj, 'Label': label_train, 'Attributes': attribute})
