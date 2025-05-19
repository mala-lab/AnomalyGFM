from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
import scipy.sparse as sp
import networkx as nx
import scipy.io as sio
import torch.nn as nn
from sklearn.random_projection import GaussianRandomProjection


class Dataset:
    def __init__(self, name='tfinance', homo=True, anomaly_alpha=None, anomaly_std=None):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('dataset/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_std:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = graph.ndata['label'][:,1].nonzero().squeeze(1)
                feat = (feat-np.average(feat,0)) / np.std(feat,0)
                feat[anomaly_id] = anomaly_std * feat[anomaly_id]
                graph.ndata['feature'] = torch.tensor(feat)
                graph.ndata['label'] = graph.ndata['label'].argmax(1)

            if anomaly_alpha:
                graph, label_dict = load_graphs('dataset/tfinance')
                graph = graph[0]
                feat = graph.ndata['feature'].numpy()
                anomaly_id = list(graph.ndata['label'][:, 1].nonzero().squeeze(1))
                normal_id = list(graph.ndata['label'][:, 0].nonzero().squeeze(1))
                label = graph.ndata['label'].argmax(1)
                diff = anomaly_alpha * len(label) - len(anomaly_id)
                import random
                new_id = random.sample(normal_id, int(diff))
                # new_id = random.sample(anomaly_id, int(diff))
                for idx in new_id:
                    aid = random.choice(anomaly_id)
                    # aid = random.choice(normal_id)
                    feat[idx] = feat[aid]
                    label[idx] = 1  # 0

        elif name == 'tsocial':
            graph, label_dict = load_graphs('dataset/tsocial')
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
        graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph


def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim
    U, S, _ = torch.linalg.svd(data)
    newdata= torch.mm(U[:, :out_dim], torch.diag(S[:out_dim]))
    return newdata


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (np.max(ano_score) - np.min(ano_score)))
    return ano_score


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

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

def loaddata(dataset, args):
    if dataset == "tsocial":
        g = dgl.load_graphs('../data/tsocial')[0][0]  # tsocial dgraphfin
        g = dgl.to_homogeneous(g, ['feature', 'label'])
        g = dgl.add_self_loop(g)  # 添加自环以便采样时包含节点本身
        features = g.ndata['feature']
        features = features.float()
        # features = x_svd(features, args.unifeat)
        # bn = nn.BatchNorm1d(features.shape[1], affine=False)
        # features = bn(features)
        g.ndata['feature'] = features
        return g
    else:
        adj, features, ano_label = load_mat(dataset)
        ano_label = torch.FloatTensor(ano_label)
        features = features.todense()
        if args.dimreduction == 'svd':
            features = torch.FloatTensor(features)
            features = x_svd(features, args.unifeat)
            bn = nn.BatchNorm1d(features.shape[1], affine=False)
            features = bn(features)
        else:
            gaussian_rp = GaussianRandomProjection(n_components=args.unifeat)
            features = gaussian_rp.fit_transform(np.asarray(features))
        graph = adj_to_dgl_graph(adj)
        graph.ndata['feature'] = features
        graph.ndata['label'] = ano_label.long().squeeze(-1)
        graph = dgl.add_self_loop(graph)

        return graph


def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.from_networkx(nx_graph)
    return dgl_graph