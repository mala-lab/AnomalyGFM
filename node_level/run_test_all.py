import torch
import torch.nn as nn

from model import Model
from model import Model_DGI
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import torch.nn.functional as F

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_train', type=str,
                    default='amazon')  # Flickr_random  amazon  questions_random
parser.add_argument('--dataset_test', type=str,
                    default='reddit')  # BlogCatalog_random  yelp  weibo_random reddit_random
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
parser.add_argument('--mean', type=float, default=0)
parser.add_argument('--var', type=float, default=0)

args = parser.parse_args()

args.lr = 5e-4
args.num_epoch = 1000

print('Dataset Train: ', args.dataset_train)
print('Dataset Test: ', args.dataset_test)

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
# os.environ['PYTHONHASHSEED'] = str(args.seed)
# os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and preprocess data
adj_train, adj_test, feat_train, feat_test, ano_labels_train, ano_labels_val, ano_labels_test, idx_train, idx_val = load_mat(
    args.dataset_train,
    args.dataset_test)

if args.dataset_train in ['amazon', 'yelp', 'reddit_random', 'weibo_random', 'questions_random']:
    feat_train, _ = preprocess_features(feat_train)
    feat_test, _ = preprocess_features(feat_test)
else:
    feat_train = feat_train.todense()
    feat_test = feat_test.todense()

# dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = feat_train.shape[0]
ft_size = feat_train.shape[1]
input_size = 300

raw_adj_train = adj_train
raw_adj_test = adj_test

print(adj_train.sum())
adj_train = normalize_adj(adj_train)
adj_test = normalize_adj(adj_test)

if args.dataset_train in ['questions_no_isolated', 'tolokers_no_isolated']:
    adj_train = adj_train.todense()
else:
    adj_train = (adj_train + sp.eye(adj_train.shape[0])).todense()
    adj_test = (adj_test + sp.eye(adj_test.shape[0])).todense()
    raw_adj_train = raw_adj_train.todense()
    raw_adj_test = raw_adj_test.todense()

feat_train = torch.FloatTensor(feat_train[np.newaxis])
feat_train = torch.FloatTensor(feat_train)

feat_test = torch.FloatTensor(feat_test[np.newaxis])
feat_test = torch.FloatTensor(feat_test)

raw_adj_train = torch.FloatTensor(raw_adj_train)
raw_adj_test = torch.FloatTensor(raw_adj_test)

adj_train = torch.FloatTensor(adj_train[np.newaxis])
adj_test = torch.FloatTensor(adj_test[np.newaxis])

ano_labels_train = torch.FloatTensor(ano_labels_train)
ano_labels_test = torch.FloatTensor(ano_labels_test)

# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

# Initialize model and optimiser
model = Model(ft_size, input_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_dgi = Model_DGI(ft_size, input_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)

optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
# if torch.cuda.is_available():
#     print('Using CUDA')
#     model.cuda()
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     raw_adj = raw_adj.cuda()

# idx_train = idx_train.cuda()
# idx_val = idx_val.cuda()
# idx_test = idx_test.cuda()
#
# if torch.cuda.is_available():
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
# else:
#     b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
# xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    total_time = 0
    for epoch in range(args.num_epoch):
        start_time = time.time()
        model.train()
        optimiser.zero_grad()

        normal_prompt = torch.randn(args.embedding_dim)
        abnormal_prompt = torch.randn(args.embedding_dim)

        # normal_prompt = preprocess_features_tensor(normal_prompt)
        # abnormal_prompt = preprocess_features_tensor(abnormal_prompt)

        logits, logit_residual, emb, emb_residual, normal_prompt, abnormal_prompt = model(feat_train, adj_train,
                                                                                          raw_adj_train, normal_prompt,
                                                                                          abnormal_prompt)

        loss_bce = torch.mean(b_xent(torch.squeeze(logits[:, idx_train]), ano_labels_train))

        loss_bce_residual = torch.mean(b_xent(torch.squeeze(logit_residual[:, idx_train]), ano_labels_train))

        # col_normalized = raw_adj_train.sum(0, keepdim=True).sqrt()
        # raw_adj_train_normalized = raw_adj_train.div(col_normalized)
        # adj_train_ref = adj_train[:, idx_train, :][:, :, idx_train]
        # emb_residual = emb[:, idx_train, :] - torch.bmm(adj_train_ref, emb[:, idx_train, :])

        # Extract normal and abnormal prototype
        emb_residual_train = emb_residual[:, idx_train, :]

        normal_proto = emb_residual_train[:, ano_labels_train == 0, :]
        abnormal_proto = emb_residual_train[:, ano_labels_train == 1, :]

        # Compute the dis at node level
        dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2, dim=2))
        # dif_normal = dif_normal.detach().numpy()
        dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=2))
        # dif_abnormal = dif_abnormal.detach().numpy()

        # loss_alignment = torch.mean(dif_normal)
        loss_alignment = torch.mean(dif_abnormal)

        # if epoch < 200:
        #     loss = loss_bce
        # else:
        #     # loss = loss_bce + loss_bce_residual
        #     loss = loss_bce + loss_alignment

        loss = loss_bce + loss_alignment
        loss.backward()
        optimiser.step()
        end_time = time.time()
        total_time += end_time - start_time
        # print('Total time is', total_time)
        print("Epoch:", '%04d' % (epoch), "loss_bce =", "{:.5f}".format(loss_bce.item()))
        print("Epoch:", '%04d' % (epoch), "loss_alignment =", "{:.5f}".format(loss_alignment.item()))

        if epoch % 10 == 0:
            logits = np.squeeze(logits.cpu().detach().numpy())
            auc = roc_auc_score(ano_labels_val, logits[idx_val])
            AP = average_precision_score(ano_labels_val, logits[idx_val], average='macro', pos_label=1,
                                         sample_weight=None)
            print('Traininig {} AUC:{:.4f}'.format(args.dataset_train, auc))
            print('Traininig AP:', AP)

            # logit_residual = np.squeeze(logit_residual.cpu().detach().numpy())
            # auc_residual = roc_auc_score(ano_labels_val, logit_residual[idx_val])
            # AP_residual = average_precision_score(ano_labels_val, logit_residual[idx_val], average='macro',
            #                                       pos_label=1,
            #                                       sample_weight=None)
            # print('Traininig {} AUC residual:{:.4f}'.format(args.dataset_train, auc_residual))
            # print('Traininig AP_residual:', AP_residual)

            emb_residual_val = emb_residual[:, idx_val, :]

            # Obtain the anomaly score on the test data using distance
            # score_normal = torch.squeeze(torch.sqrt(torch.sum((emb_residual_val - normal_prompt) ** 2, dim=2)))
            # score_abnormal = torch.squeeze(torch.sqrt(torch.sum((emb_residual_val - abnormal_prompt) ** 2, dim=2)))
            #

            # Obtain the anomaly score on the test data Using Corr
            abnormal_prompt = F.normalize(abnormal_prompt, p=2, dim=0)
            normal_prompt = F.normalize(normal_prompt, p=2, dim=0)

            emb_residual_val = F.normalize(torch.squeeze(emb_residual_val), p=2, dim=1)
            score_normal = torch.mm(emb_residual_val, torch.unsqueeze(normal_prompt, 1))
            score_abnormal = torch.mm(emb_residual_val,torch.unsqueeze(abnormal_prompt, 1))

            ano_score = (score_abnormal).detach().numpy()
            # ano_score = -(score_normal).detach().numpy()
            auc_measure = roc_auc_score(ano_labels_val, ano_score)
            AP_measure = average_precision_score(ano_labels_val, ano_score)

            print('Traininig {} AUC measure:{:.4f}'.format(args.dataset_train, auc_measure))
            print('Traininig AP measure:', AP_measure)

        if epoch % 10 == 0:
            print("<<<<<<<<<<<<<<<<<<<<<< Test begin>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            model.eval()
            # Evaluate on other dataset
            normal_prompt = torch.randn(args.embedding_dim)
            abnormal_prompt = torch.randn(args.embedding_dim)
            logits_test, logits_test_residual, emb_test, emb_residual_test, normal_prompt_test, abnormal_prompt_test = model(
                feat_test,
                adj_test, raw_adj_test,
                normal_prompt,
                abnormal_prompt)

            # Evaluation on the validation and test node
            # logits_test = np.squeeze(logits_test.cpu().detach().numpy())
            # auc = roc_auc_score(ano_labels_test, logits_test)
            # AP = average_precision_score(ano_labels_test, logits_test, average='macro', pos_label=1, sample_weight=None)
            # print('Testing {} AUC:{:.4f}'.format(args.dataset_test, auc))
            # print('Testing AP:', AP)

            # logits_test_residual = np.squeeze(logits_test_residual.cpu().detach().numpy())
            # auc_residual = roc_auc_score(ano_labels_test, logits_test_residual)
            # AP_residual = average_precision_score(ano_labels_test, logits_test_residual, average='macro', pos_label=1,
            #                                       sample_weight=None)
            # print('Testing {} AUC residual:{:.4f}'.format(args.dataset_test, auc_residual))
            # print('Testing AP_residual:', AP_residual)


            # Obtain the anomaly score on the test data Using Corr
            # abnormal_prompt_test = abnormal_prompt_test / torch.norm(abnormal_prompt_test, dim=-1, keepdim=True)
            # normal_prompt_test = normal_prompt_test / torch.norm(normal_prompt_test, dim=-1, keepdim=True)
            abnormal_prompt_test = F.normalize(abnormal_prompt_test, p=2, dim=0)
            normal_prompt_test = F.normalize(normal_prompt_test, p=2, dim=0)

            emb_residual_test = F.normalize(torch.squeeze(emb_residual_test), p=2, dim=1)
            score_normal = torch.mm(emb_residual_test, torch.unsqueeze(normal_prompt_test, 1))
            score_abnormal = torch.mm(emb_residual_test, torch.unsqueeze(abnormal_prompt_test, 1))

            # Obtain the anomaly score on the test data Using distance
            # score_normal = torch.squeeze(torch.sqrt(torch.sum((emb_residual_test - normal_prompt_test) ** 2, dim=2)))
            # score_abnormal = torch.squeeze(torch.sqrt(torch.sum((emb_residual_test - abnormal_prompt_test) ** 2, dim=2)))

            ano_score = (score_abnormal).detach().numpy()
            # ano_score = -(score_normal).detach().numpy()
            auc_measure = roc_auc_score(ano_labels_test, ano_score)
            AP_measure = average_precision_score(ano_labels_test, ano_score, average='macro', pos_label=1,
                                                 sample_weight=None)

            print('Testing {} AUC_measure:{:.4f}'.format(args.dataset_test, auc_measure))
            print('Testing AP_measure:', AP_measure)
            print("<<<<<<<<<<<<<<<<<<<<<< Test end >>>>>>>>>>>>>>>>>>>>>>>>>>>")
