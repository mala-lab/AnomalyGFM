# -*- coding: utf-8 -*-
import torch.nn as nn
from model import Model
from utils import *
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.random_projection import GaussianRandomProjection
import os
import argparse
from tqdm import tqdm
import time


os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Set argument
parser = argparse.ArgumentParser(description='Truncated Affinity Maximization for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str, default='Facebook')  # 'BlogCatalog'  'ACM'  'Amazon' 'Facebook'  'Reddit'  'YelpChi'
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--subgraph_size', type=int, default=15)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--margin', type=int, default=2)
parser.add_argument('--negsamp_ratio', type=int, default=2)
parser.add_argument('--unifeat', type=int, default=8)
parser.add_argument('--dimreduction', type=str, default='svd')
parser.add_argument('--cutting', type=int, default=4)  # 3 5 8 10
parser.add_argument('--N_tree', type=int, default=3)  # 3 5 8 10
parser.add_argument('--lamda', type=int, default=1)  # 0  0.5  1
args = parser.parse_args()

args.lr = 1e-5
args.num_epoch = 200

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
def loaddata(dataset, args, device):
    adj, features,  ano_label, str_ano_label, attr_ano_label = load_mat(dataset)

    if dataset in ['Amazon', 'YelpChi', 'Amazon-all', 'YelpChi-all']:
        features, _ = preprocess_features(features)
        raw_features = features
    else:
        raw_features = features.todense()
        features = raw_features
    numnode = features.shape[0]
    if args.dimreduction == 'svd':
        features = torch.FloatTensor(features)
        features = x_svd(features, args.unifeat)
        bn = nn.BatchNorm1d(features.shape[1], affine=False)
        # ipdb.set_trace()
        features = bn(features)
        features = features[np.newaxis]
    else:
        gaussian_rp = GaussianRandomProjection(n_components=args.unifeat)
        features = gaussian_rp.fit_transform(np.asarray(features))
        features = torch.FloatTensor(features[np.newaxis])

    dgl_graph = adj_to_dgl_graph(adj)
    raw_adj = adj
    raw_adj = (raw_adj + sp.eye(adj.shape[0])).todense()
    adj = (adj + sp.eye(adj.shape[0])).todense()
    raw_features = features
    adj = torch.FloatTensor(adj[np.newaxis])
    raw_adj = torch.FloatTensor(raw_adj[np.newaxis])
    # ano_label = torch.FloatTensor(ano_label)
    # ano_label = ano_label.reshape(numnode,1)

    features = features.to(device)
    adj = adj.to(device)
    raw_adj = raw_adj.to(device)
    raw_features = raw_features.to(device)
    # ano_label = ano_label.to(device)
    return adj, raw_adj, features, raw_features, ano_label


traindatasets = [args.dataset]
adj_train = []
raw_adj_train = []
feature_train = []
raw_feature_train = []
ano_label_train = []
dis_array_train = []
for dataset in traindatasets:
    adj, raw_adj, features, raw_features, ano_label = loaddata(dataset, args, device)
    dis_array = calc_distance(raw_adj[0, :, :], raw_features[0, :, :])
    dis_array = dis_array.to(device)
    adj_train.append(adj)
    raw_adj_train.append(raw_adj)
    feature_train.append(features)
    raw_feature_train.append(raw_features)
    ano_label_train.append(ano_label)
    dis_array_train.append(dis_array)



def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    sim_u_u = torch.mm(emb, emb.T)
    adj_inverse = (1 - adj)
    sim_u_u = sim_u_u * adj_inverse
    sim_u_u_no_diag = torch.sum(sim_u_u, 1)
    row_sum = torch.sum(adj_inverse, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    sim_u_u_no_diag = sim_u_u_no_diag * r_inv
    loss_reg = torch.sum(sim_u_u_no_diag)
    return loss_reg


def max_message(feature, adj_matrix):
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
    return - torch.sum(message), message


def inference(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    message = torch.sum(sim_matrix, 1)
    message = message * r_inv
    return message


all_aucs = []
all_aps = []

for _ in range(1):
# Initialize model and optimiser
    optimiser_list = []
    model_list = []
    for i in range(args.cutting * args.N_tree):
        model = Model(args.unifeat, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        model = model.to(device)
        optimiser_list.append(optimiser)
        model_list.append(model)

    start_time = time.time()
    for epoch in range(args.num_epoch):
        for dataset in range(len(traindatasets)):
            raw_adj = raw_adj_train[dataset]
            features = feature_train[dataset]
            raw_features = raw_feature_train[dataset]

            score_list = []
            new_adj_list = []
            for n_t in range(args.N_tree):
                new_adj_list.append(raw_adj)
            all_cut_adj = torch.cat(new_adj_list)
            origin_degree = torch.sum(torch.squeeze(raw_adj), 0)
            print('<<<<<<Start to calculate distance<<<<<')
            dis_array = dis_array_train[dataset]

            index = 0
            message_mean_list = []
            for n_cut in range(args.cutting):
                for n_t in range(args.N_tree):
                    cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t, :, :])
                    cut_adj = cut_adj.unsqueeze(0)
                    optimiser_list[index].zero_grad()
                    model_list[index].train()
                    print("<<<< cutting num .{}<<<<<<".format(n_cut))
                    adj_norm = normalize_adj_tensor(cut_adj)
    
                    node_emb, feat1, feat2 = model_list[index].forward(features, adj_norm)
                    loss, message_sum1 = max_message(node_emb[0, :, :], raw_adj[0, :, :])

                    reg_loss = reg_edge(feat1[0, :, :], raw_adj[0, :, :])
                    loss = loss + args.lamda * reg_loss
                    loss.backward()
                    optimiser_list[index].step()

                    all_cut_adj[n_t, :, :] = torch.squeeze(cut_adj)
                    index += 1
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Training Execution time: {elapsed_time} seconds")
    ##### Test on Target Datasets
    aucs = []
    aps = []
    # target_datasets = ['Amazon', 'Reddit', 'weibo', 'YelpChi', 'Amazon-all', 'YelpChi-all']
    # target_datasets = ['tf_finace']
    # target_datasets = ['tf_finace', 'elliptic']
    target_datasets = ['book', 'Disney']
    start_time = time.time()
    for dataset in target_datasets:
        adj, raw_adj, features, raw_features, ano_label = loaddata(dataset, args, device)
        index = 0
        adj_norm = normalize_adj_tensor(adj)
        message_mean_list = []
        for n_cut in range(args.cutting):
            message_list = []
            for n_t in range(args.N_tree):
                node_emb, feat1, feat2 = model_list[index].forward(features, adj_norm)
                message_sum = inference(node_emb[0, :, :], raw_adj[0, :, :])
                message_list.append(torch.unsqueeze(message_sum, 0))
            
            message_list = torch.mean(torch.cat(message_list), 0)
            message_mean_list.append(torch.unsqueeze(message_list, 0))

        message_mean_cut = torch.mean(torch.cat(message_mean_list), 0)
        message_mean = np.array(message_mean_cut.cpu().detach())
        message_mean = 1 - normalize_score(message_mean)
        score = message_mean
        
        test_auc = roc_auc_score(ano_label, score)
        test_ap = average_precision_score(ano_label, score, average='macro', pos_label=1, sample_weight=None)
        # np.savez('TAMScore%s.npz'%dataset, comp=1-score, label=ano_label)
        aucs.append(test_auc)
        aps.append(test_ap)
        print('{} -> {} AUC:{:.4f} AP{:.4f}'.format(args.dataset, dataset, test_auc, test_ap))

        with open(f'results/{args.dataset}.txt','a') as f:
            f.write('\n{} -> {} AUC:{:.4f} AP{:.4f}\n'.format(args.dataset, dataset, test_auc, test_ap))
    all_aucs.append(aucs)
    all_aps.append(aps)
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    print(f"Testing Execution time: {elapsed_time} seconds")

all_aucs, all_aps = np.array(all_aucs), np.array(all_aps)
mean_auc, std_auc = np.mean(all_aucs, 0), np.std(all_aucs, 0)
mean_ap, std_ap = np.mean(all_aps, 0), np.std(all_aps, 0)

for i, dataset in enumerate(target_datasets):
    with open(f'results/{args.dataset}.txt','a') as f:
        f.write('\n Averaged {} -> {} AUC:{:.4f}$_{{\\pm {:.3f}}}$ AP:{:.4f}$_{{\\pm {:.3f}}}$\n'.format(args.dataset, dataset, mean_auc[i], std_auc[i], mean_ap[i], std_ap[i]))
 