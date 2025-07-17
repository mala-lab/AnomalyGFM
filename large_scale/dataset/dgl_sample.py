# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  12/17/2024 
# version： Python 3.7.8
# @File : dgl_sample.py
# @Software: PyCharm
import dgl
import numpy as np
import torch
from dgl.dataloading import NeighborSampler
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    # traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)
    traces, _ = dgl.sampling.random_walk(dgl_graph, all_idx, restart_prob=0.9, length=subgraph_size*3)
    subv = []
    # ipdb.set_trace()
    for i,trace in enumerate(traces):
        subv.append(torch.unique(trace, sorted=False).tolist())
        retry_time = 0
        print(i)
        while len(subv[i]) < reduced_size:
            # cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            cur_trace = dgl.sampling.random_walk(dgl_graph, [i], restart_prob=0.5, length=subgraph_size*5)
            subv[i] = torch.unique(cur_trace[0],sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv

def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim
    U, S, _ = torch.linalg.svd(data)
    newdata= torch.mm(U[:, :out_dim], torch.diag(S[:out_dim]))
    return newdata
# def assign_edge():

# 创建一个示例图
# g = dgl.graph((torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3, 0])))
# g = dgl.add_self_loop(g)  # 添加自环以便采样时包含节点本身
# 节点特征（可选）
# g.ndata['feat'] = torch.arange(g.num_nodes()).float().view(-1, 1)

# Load DGraph and T-social
dataset ='tfinance'  # tfinance
g = dgl.load_graphs(dataset)[0][0]  #tsocial dgraphfin
g = dgl.to_homogeneous(g, ['feature', 'label'])
g = dgl.add_self_loop(g)  # 添加自环以便采样时包含节点本身
features = g.ndata['feature'].float()
bn = nn.BatchNorm1d(features.shape[1], affine=False)
features = bn(features)
g.ndata['feature'] = features

# src, dst = g.edges()
# edge_similarity = F.cosine_similarity(features[src], features[dst], dim=-1)
# print("\n边上的余弦相似度：")
# print(torch.mean(edge_similarity))

# features = x_svd(features, 10)
# # bn = nn.BatchNorm1d(features.shape[1], affine=False)
# # features = bn(features)
# g.ndata['feature'] = features

ba = []
bf = []
subgraph_size = 15
subgraphs = generate_rwr_subgraph(g, subgraph_size)
adj = g.adjacency_matrix(scipy_fmt='coo')
labels = g.ndata['label']
for i in range(g.num_nodes()):
# for i in range(100):
    traces = subgraphs[i][1:]
    # traces[traces == -1] = traces[:, 0].unsqueeze(1)
    print(i, traces)
    # unique_nodes = torch.unique(traces[traces >= 0])
    # graph_adj = g.subgraph(unique_nodes)
    # adj = graph_adj.edges()
    # cur_adj = torch.eye(subgraph_size, subgraph_size)
    cur_feat = features[subgraphs[i][1:], :]
    ba.append(cur_feat)

# save feat listd
sample_feature = np.stack(ba, 0)
labels = np.array(labels)
np.save('{}_feature_15_1.npy'.format(dataset), sample_feature)
np.save('{}_label_15_1.npy'.format(dataset), labels)

# # 定义采样器
# sampler = NeighborSampler([2])  # 每层;采样2个邻居
#
# # 遍历每个节点并采样子图
# for nid in range(g.num_nodes()):
#     # 对单个节点进行采样
#     subgraph = dgl.sampling.sample_neighbors(g, [nid], fanout=2)
#     adj_matrix = subgraph.adjacency_matrix(scipy_fmt='coo').toarray()  # 返回稀疏矩阵格式 (COO)
#
#     # 将 DGL 图转换为边列表
#     edge_list = torch.stack(subgraph.edges()).numpy()  # 边列表为 [2, num_edges] 的 numpy 数组

#     # 保存为 .mat 文件
#     sio.savemat('graph.mat', {
#         'adj_matrix': adj_matrix,  # 邻接矩阵
#         'edge_list': edge_list.T  # 边列表
#     })
#
#     # 打印采样到的子图
#     print(f"采样到的子图（以节点 {nid} 为中心）:")
#     print("子图中的边:", subgraph.edges())
#     print("子图中的节点特征:", subgraph.ndata['feat'])
#
# # 或者使用采样器在所有节点上进行批量采样
# dataloader = dgl.dataloading.NodeDataLoader(
#     g, torch.arange(g.num_nodes()), sampler,
#     batch_size=1, shuffle=True, drop_last=False)
#
# for input_nodes, output_nodes, blocks in dataloader:
#     print(f"批量采样 - 输入节点: {input_nodes}, 输出节点: {output_nodes}")
#     print("子图结构:", blocks[0])
