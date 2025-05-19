# -*- coding: utf-8 -*-
import torch.nn as nn
from model import Model
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.random_projection import GaussianRandomProjection
import os
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='GCN')
parser.add_argument('--dataset', type=str, default='Facebook') 
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_epoch', type=int, default=200)
parser.add_argument('--unifeat', type=int, default=8)
parser.add_argument('--dimreduction', type=str, default='svd')
args = parser.parse_args()


print('Dataset: ', args.dataset)
# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    if torch.cuda.is_available():
        features = features.cuda()
        adj_norm = adj_norm.cuda()
        ano_label = ano_label.cuda()
    return adj_norm, features, ano_label, str_ano_label, attr_ano_label

traindatasets = [args.dataset]
adj_norm_train = []
feature_train = []
ano_label_train = []
for dataset in traindatasets:
    adj_norm, features, ano_label, str_ano_label, attr_ano_label = loaddata(dataset, args)
    adj_norm_train.append(adj_norm)
    feature_train.append(features)
    ano_label_train.append(ano_label)


all_aucs = []
all_aps = []

for _ in range(5):
    model = Model(args.unifeat, args.embedding_dim, 'prelu')
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()
    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(args.num_epoch):
        for dataset in range(len(traindatasets)):
            model.train()
            optimiser.zero_grad()
            features = feature_train[dataset]
            adj_norm = adj_norm_train[dataset]
            ano_label = ano_label_train[dataset]
            output = model(features, adj_norm)
            loss = criterion(output, ano_label)
            loss.backward()
            optimiser.step()
            print(epoch)

        # if (epoch + 1) % int(args.num_epoch/10) == 0:
        #     for dataset in range(2):
        #         features = feature_train[dataset]
        #         adj_norm = adj_norm_train[dataset]
        #         ano_label = ano_label_train[dataset]
        #         output = model(features, adj_norm)
        #         score = output.detach().cpu().numpy()
        #         # ipdb.set_trace()
        #         auc = roc_auc_score(ano_label.squeeze().cpu(), score.squeeze())
        #         AP = average_precision_score(ano_label.squeeze().cpu(), score.squeeze(), average='macro', pos_label=1, sample_weight=None)
        #         print('{} AUC:{:.4f} AP:{:.4f}'.format(traindatasets[dataset], auc, AP))
            


    ##### Test on Target Datasets
    aucs = []
    aps = []
    # target_datasets = ['Amazon', 'Reddit', 'weibo', 'YelpChi', 'Amazon-all', 'YelpChi-all']
    # target_datasets = ['tf_finace']
    # target_datasets = ['tf_finace', 'elliptic']
    # target_datasets = ['elliptic', 'photo', 'tolokers']
    target_datasets = ['book', 'Disney']
    for dataset in target_datasets:

        adj_norm, features, ano_label, str_ano_label, attr_ano_label = loaddata(dataset, args)
        output = model(features, adj_norm)
        score = output.detach().cpu().numpy()
        test_auc = roc_auc_score(ano_label.squeeze().cpu(), score.squeeze())
        test_ap = average_precision_score(ano_label.squeeze().cpu(), score.squeeze(), average='macro', pos_label=1, sample_weight=None)
        aucs.append(test_auc)
        aps.append(test_ap)
        print('{} -> {} GC AUC:{:.4f} AP{:.4f}'.format(args.dataset, dataset, test_auc, test_ap))
        with open(f'results/{args.dataset}.txt','a') as f:
            f.write('\n{} -> {} AUC:{:.4f} AP{:.4f}\n'.format(args.dataset, dataset, test_auc, test_ap))
    all_aucs.append(aucs)
    all_aps.append(aps)

all_aucs, all_aps = np.array(all_aucs), np.array(all_aps)
mean_auc, std_auc = np.mean(all_aucs, 0), np.std(all_aucs, 0)
mean_ap, std_ap = np.mean(all_aps, 0), np.std(all_aps, 0)

for i, dataset in enumerate(target_datasets):
    with open(f'results/{args.dataset}.txt','a') as f:
        f.write('\n Averaged {} -> {} AUC {:.4f} {:.3f} AP:{:.4f} {:.3f}\n'.format(args.dataset, dataset, mean_auc[i], std_auc[i], mean_ap[i], std_ap[i]))
 