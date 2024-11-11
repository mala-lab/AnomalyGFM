import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
import ipdb
from torch.utils.data import ConcatDataset

def init_structural_encoding(gs, rw_dim=16, dg_dim=16):
    for g in gs:
        A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(rw_dim-1):
            M_power = M_power * M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE,dim=-1)

        g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
        DGSE = torch.zeros([g.num_nodes, dg_dim])
        for i in range(len(g_dg)):
            DGSE[i, int(g_dg[i])] = 1

        g['x_s'] = torch.cat([RWSE, DGSE], dim=1)

    return gs



def x_svd(data, out_dim):
    assert data.shape[-1] >= out_dim
    U, S, _ = torch.linalg.svd(data)
    newdata= torch.mm(U[:, :out_dim], torch.diag(S[:out_dim]))
    return newdata

def unifeature(gs, unidim=8):
    # ipdb.set_trace()
    feature_dim = gs[0].x.shape[1]
    proj_matrix = torch.randn(feature_dim, unidim)
    for g in gs:
        feature = g.x
        random_feature = feature @ proj_matrix
        # svd_feature = x_svd(feature, unidim)
        # if svd_feature.shape[1] !=8:
        #     ipdb.set_trace()
        g['x'] = random_feature
    return gs

def get_ad_split_TU(args, fold=10):
    path_now = "/data/chniu/phase6_24_10/GLADdata"
    print(path_now)
    path = osp.join(path_now, args.DS)
    dataset = TUDataset(path, name=args.DS)
    data_list = []
    label_list = []
    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())
    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)
    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))
    return splits


#10 TU dataset
def get_ad_dataset_TU(args, target_dataset=None):
    path_now =  "/data/chniu/phase6_24_10/GLADdata"
    path = osp.join(path_now, args.DS)
    if target_dataset is not None:
        dataset_name = target_dataset
    else:
        dataset_name = args.DS
        
    if dataset_name in ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(path, name=dataset_name, transform=(Constant(1, cat=False)))
    else:
        dataset = TUDataset(path, name=dataset_name)
    # ipdb.set_trace()
    min_nodes_num = min([_.num_nodes for _ in dataset])
    # print(min_nodes_num)

    data_list = []
    label_list = []
    for data in dataset:
        data.edge_attr = None
        data_list.append(data)
        label_list.append(data.y.item())
    
    data_list = unifeature(data_list, args.unifeat)
    feat_dims = np.mean([_.x.shape[1] for _ in data_list])
    train_data_list = []
    for data in data_list:
        if data.y != 0:
            data.y = 0
            if target_dataset is None:
                train_data_list.append(data)
        else:
            data.y = 1

    # ipdb.set_trace()
    if target_dataset is None:
        dataloader = DataLoader(train_data_list, batch_size=args.batch_size, shuffle=True)
        meta = {'num_train':len(train_data_list), 'min_nodes_num':min_nodes_num,'num_edge_feat':0}
    else:
        dataloader = DataLoader(data_list, batch_size=args.batch_size, shuffle=True)
        meta = {'num_train':len(data_list), 'min_nodes_num':min_nodes_num,'num_edge_feat':0}

    return dataset, dataloader, meta