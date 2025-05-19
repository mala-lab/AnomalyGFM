
# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  7/5/2023
# version： Python 3.7.8
# @File : ocgnn.py
# @Software: PyCharm
import numpy as np
import scipy.sparse as sp
import torch

from model_AnomalyDAE import Model
from utils import *
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import  average_precision_score
import argparse
from tqdm import tqdm
import time
import torch.nn as nn
import  os

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='Amazon')  # ' tolokers_no_isolated 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--dataset_test', type=str,  default='tolokers')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--unifeat', type=int, default=8)
parser.add_argument('--dimreduction', type=str, default='svd')

args = parser.parse_args()

args.lr = 1e-4
args.num_epoch = 600

batch_size = args.batch_size
subgraph_size = args.subgraph_size

print('Dataset Train: ', args.dataset)
print('Dataset Test: ', args.dataset_test)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load and preprocess data

# Load and preprocess data
def loaddata(dataset, args):
    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)

    if dataset in ['Amazon', 'YelpChi', 'Amazon-all', 'YelpChi-all']:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()
    numnode = features.shape[0]
    if args.dimreduction == 'svd':
        features = torch.FloatTensor(features)
        features = x_svd(features, args.unifeat)
        # features = features[np.newaxis]
    else:
        gaussian_rp = GaussianRandomProjection(n_components=args.unifeat)
        features = gaussian_rp.fit_transform(np.asarray(features))
        features = torch.FloatTensor(features[np.newaxis])
    bn = nn.BatchNorm1d(features.shape[1], affine=False)
    features = bn(features)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    # adj = torch.FloatTensor(adj[np.newaxis])
    adj = torch.FloatTensor(adj)
    adj_norm = normalize_adj_tensor(adj)
    ano_label = torch.FloatTensor(ano_label)
    ano_label = ano_label.reshape(numnode,1)
    # if torch.cuda.is_available():
    #     features = features.cuda()
    #     adj_norm = adj_norm.cuda()
    #     ano_label = ano_label.cuda()
    return adj_norm, features, ano_label, str_ano_label, attr_ano_label

traindatasets = [args.dataset]
# target_datasets = ['Amazon', 'Reddit', 'weibo', 'YelpChi', 'Amazon-all', 'YelpChi-all']
target_datasets = ['tf_finace', 'elliptic', 'tolokers', 'question', 'Disney']
# target_datasets = ['book', 'Disney']
# target_datasets = ['BZR_svd', 'DD_svd', 'AIDS_svd', 'DHFR_svd', 'PROTEINS_svd', 'MUTAG_svd']
# target_datasets = [ 'COX2_svd', 'NCI1_svd']
adj_norm_train = []
feature_train = []
ano_label_train = []
for dataset in traindatasets:
    adj_norm, features, ano_label, str_ano_label, attr_ano_label = loaddata(dataset, args)
    adj_norm_train.append(adj_norm)
    feature_train.append(features)
    ano_label_train.append(ano_label)




ft_size = args.unifeat
input_size = 600

all_aucs = []
all_aps = []

for _ in range(5):

    # Initialize model and optimiser
    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # model = model.cuda()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    start_time = time.time()
    for epoch in range(args.num_epoch):
        for dataset in range(len(traindatasets)):
            model.train()
            optimiser.zero_grad()

            # Train model
            feat_train = feature_train[dataset]
            adj_train = adj_norm_train[dataset]
            # ipdb.set_trace()
            loss, score = model(feat_train, adj_train)
            loss.backward()
            optimiser.step()

            # if epoch % 2 == 0:
            #     print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Training Execution time: {elapsed_time} seconds")
    model.eval()
    aucs = []
    aps = []
    start_time = time.time()
    for dataset in target_datasets:
        adj_norm, features, ano_label, str_ano_label, attr_ano_label = loaddata(dataset, args)
        output, score = model(features, adj_norm)
        score = np.array(score.detach().cpu())
        test_auc = roc_auc_score(ano_label.squeeze().cpu(), score.squeeze())
        test_ap = average_precision_score(ano_label.squeeze().cpu(), score.squeeze(), average='macro', pos_label=1, sample_weight=None)
        aucs.append(test_auc)
        aps.append(test_ap)
        print('{} -> {} GC AUC:{:.4f} AP{:.4f}'.format(args.dataset, dataset, test_auc, test_ap))
        with open(f'results/{args.dataset}.txt','a') as f:
            f.write('\n{} -> {} AUC:{:.4f} AP{:.4f}\n'.format(args.dataset, dataset, test_auc, test_ap))
    all_aucs.append(aucs)
    all_aps.append(aps)
    # end_time = time.time()  # End the timer
    # elapsed_time = end_time - start_time
    # print(f"Testing Execution time: {elapsed_time} seconds")
all_aucs, all_aps = np.array(all_aucs), np.array(all_aps)
mean_auc, std_auc = np.mean(all_aucs, 0), np.std(all_aucs, 0)
mean_ap, std_ap = np.mean(all_aps, 0), np.std(all_aps, 0)

for i, dataset in enumerate(target_datasets):
    with open(f'results/{args.dataset}.txt','a') as f:
        f.write('\n Averaged {} -> {} AUC:{:.4f}$_{{\\pm {:.3f}}}$ AP:{:.4f}$_{{\\pm {:.3f}}}$\n'.format(args.dataset, dataset, mean_auc[i], std_auc[i], mean_ap[i], std_ap[i]))
