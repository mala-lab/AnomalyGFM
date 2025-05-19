# -*- coding: utf-8 -*-
import torch.nn as nn
from model import GAT
from gatutils import *
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import argparse
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
device = torch.device("cpu")
# Set argument
parser = argparse.ArgumentParser(description='GCN')
parser.add_argument('--dataset', type=str, default='Facebook') 
# parser.add_argument('--targetdatasets', type=str, default=['Amazon', 'Facebook', 'Reddit', 'weibo'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=64)
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
torch.cuda.manual_seed_all(args.seed).random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
sigmoid = nn.Sigmoid()

traindatasets = [args.dataset]
graphs = []
for dataset in traindatasets:
    graph = loaddata(dataset, args)
    # graph = graph.to(device)
    graphs.append(graph)

all_aucs = []
all_aps = []

for _ in range(5):
    model = GAT(args)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    model = model.to(device)
    start_time = time.time() 
    for epoch in range(args.num_epoch):
        for dataset in range(len(traindatasets)):
            model.train()
            optimiser.zero_grad()
            graph = graphs[dataset]
            feat = graph.ndata['feat']
            ano_label = graph.ndata['label']
            output = model(graph, feat, device)
            output = sigmoid(output)
            loss = criterion(output, ano_label)
            loss.backward()
            optimiser.step()
            print(epoch)
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time} seconds")
    ##### Test on Target Datasets
    aucs = []
    aps = []
    # target_datasets = ['Amazon', 'Reddit', 'weibo', 'YelpChi', 'Amazon-all', 'YelpChi-all']
    # target_datasets = ['tf_finace']
    # target_datasets = ['tf_finace', 'elliptic']
    target_datasets = ['book', 'Disney']
    start_time = time.time()
    for dataset in target_datasets:

        graph = loaddata(dataset, args)
        graph = graph.to(device)
        feat = graph.ndata['feat']
        ano_label = graph.ndata['label']
        output = model(graph, feat, device)
        output = sigmoid(output)
        score = output.detach().cpu().numpy()
        test_auc = roc_auc_score(ano_label.squeeze().cpu(), score.squeeze())
        test_ap = average_precision_score(ano_label.squeeze().cpu(), score.squeeze(), average='macro', pos_label=1, sample_weight=None)
        aucs.append(test_auc)
        aps.append(test_ap)
        print('{} -> {} GC AUC:{:.4f} AP{:.4f}'.format(args.dataset, dataset, test_auc, test_ap))

        with open(f'results/{args.dataset}.txt','a') as f:
            f.write('\n GAT {} -> {} AUC:{:.4f} AP{:.4f}\n'.format(args.dataset, dataset, test_auc, test_ap))
    all_aucs.append(aucs)
    all_aps.append(aps)
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Testing time: {elapsed_time} seconds")

all_aucs, all_aps = np.array(all_aucs), np.array(all_aps)
mean_auc, std_auc = np.mean(all_aucs, 0), np.std(all_aucs, 0)
mean_ap, std_ap = np.mean(all_aps, 0), np.std(all_aps, 0)

for i, dataset in enumerate(target_datasets):
    with open(f'results/{args.dataset}.txt','a') as f:
        f.write('\n GAT Averaged {} -> {} AUC {:.4f} {:.3f} AP:{:.4f} {:.3f}\n'.format(args.dataset, dataset, mean_auc[i], std_auc[i], mean_ap[i], std_ap[i]))
 