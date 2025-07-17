# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  1/6/2025 
# version： Python 3.7.8
# @File : mat_dgl.py
# @Software: PyCharm
import networkx as nx
import scipy.sparse as sp
from dgl.data import FraudDataset
from dgl.data.utils import load_graphs
import dgl
import scipy.io as sio
import numpy as np
import torch

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.from_networkx(nx_graph)
    return dgl_graph


def load_mat(dataset):
    """Load .mat dataset."""
    data = sio.loadmat("../data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    return adj, feat, ano_labels


dataset = ''
adj, features, ano_label = load_mat(dataset)
ano_label = torch.FloatTensor(ano_label)
features = features.todense()
graph = adj_to_dgl_graph(adj)
graph = dgl.add_self_loop(graph)
graph.ndata['feature'] = features
graph.ndata['label'] = ano_label.long().squeeze(-1)


dgl.save_graphs('graph_data', graph)
