# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  14/9/2022
# version： Python 3.7.8
# @File : raw_affinity.py
# @Software: PyCharm
# @Institution: SMU


import torch.nn as nn
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import pandas as pd
from utils import *
import argparse
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='Truncated Affinity Maximization for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str,
                    default='Amazon')  # 'BlogCatalog'  'ACM'  'Amazon' 'Facebook'  'Reddit'  'YelpChi' 'Amazon-all' 'YelpChi-all'
args = parser.parse_args()
# Load and preprocess data
adj, features,  ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

if args.dataset in ['Amazon', 'YelpChi']:
    features, _ = preprocess_features(features)
    raw_features = features

else:
    raw_features = features.todense()
    features = raw_features


dgl_graph = adj_to_dgl_graph(adj)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
raw_adj = (raw_adj + sp.eye(adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()
raw_features = torch.FloatTensor(raw_features[np.newaxis])
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])


def raw_affinity(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix

    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0
    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    message = torch.sum(sim_matrix, 1)

    message = message * r_inv
    # message = (message - torch.min(message)) / (torch.max(message) - torch.min(message))
    # message[torch.isinf(message)] = 0.
    # message[torch.isnan(message)] = 0.
    return message


message_sum = raw_affinity(features[0, :, :], raw_adj[0, :, :])
message = np.array(message_sum)
message = 1 - normalize_score(message)
draw_pdf(1 - message, ano_label, args.dataset)

