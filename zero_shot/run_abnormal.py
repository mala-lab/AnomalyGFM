import torch
import torch.nn as nn

from model import Model
from utils import *

from sklearn.metrics import roc_auc_score
import random
import dgl
from sklearn.metrics import average_precision_score
import argparse
from tqdm import tqdm
import time
import torch.nn.functional as F
import scipy.io as scio

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [3]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
# Set argument
parser = argparse.ArgumentParser(description='')

parser.add_argument('--dataset_train', type=str,
                    default='Facebook_svd')  # Flickr_random  amazon  questions_random
# add the graph level dataset
parser.add_argument('--dataset_test', type=str,
                        default='yelp_svd')  # BlogCatalog_random  yelp  weibo_random reddit_random  tolokers_random tf_finace_random, elliptic_random
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=400)    # 200  #300  400 500
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0)
parser.add_argument('--var', type=float, default=0)
parser.add_argument('--input_size', type=int, default=8)

args = parser.parse_args()


# dimension 6
# args.lr = 1e-4
# args.num_epoch =500  # Facebook


# dimension 8
args.lr = 1e-4
args.num_epoch =301  # Facebook


# args.lr = 5e-4
# args.num_epoch = 500  # Amazon


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

feat_train, _ = preprocess_features(feat_train)
feat_test, _ = preprocess_features(feat_test)
# feat_test = feat_test.todense()

# if args.dataset_train in ['amazon', 'yelp', 'reddit_random', 'weibo_random', 'questions_random',
#                           'tolokers_random', 'elliptic_random', 'photo_random']:
#     feat_train, _ = preprocess_features(feat_train)
#     feat_test, _ = preprocess_features(feat_test)
# else:
#     feat_train = feat_train.todense()
#     feat_test = feat_test.todense()

# dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = feat_train.shape[0]
ft_size = feat_train.shape[1]
input_size = args.input_size

raw_adj_train = adj_train
raw_adj_test = adj_test

print(adj_train.sum())
adj_train = normalize_adj(adj_train)
adj_test = normalize_adj(adj_test)


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

        # mean_n, std_n = 0, 1
        normal_prompt_raw = torch.randn(args.embedding_dim)
        abnormal_prompt_raw = torch.randn(args.embedding_dim)

        # mean_n, std_n = 0, 0.5
        # mean_a, std_a = 0, 0.5
        # normal_prompt_raw = torch.normal(mean_n, std_n, size=(args.embedding_dim,))
        # abnormal_prompt_raw = torch.normal(mean_a, std_a, size=(args.embedding_dim,))

        # mean_n, std_n = 0.1, 0.5
        # mean_a, std_a = 0.1, 0.5
        # normal_prompt_raw = torch.normal(mean_n, std_n, size=(args.embedding_dim,))
        # abnormal_prompt_raw = torch.normal(mean_a, std_a, size=(args.embedding_dim,))
        #
        # mean_n, std_n = 0.2, 1
        # mean_a, std_a = 0.2, 1
        # normal_prompt_raw = torch.normal(mean_n, std_n, size=(args.embedding_dim,))
        # abnormal_prompt_raw = torch.normal(mean_a, std_a, size=(args.embedding_dim,))


        # noise_std = 0.5
        # noise = torch.normal(0, noise_std, size=(args.embedding_dim, ))
        # abnormal_prompt_raw = abnormal_prompt_raw + noise

        # abnormal_prompt_raw = preprocess_features_tensor(abnormal_prompt_raw)
        # normal_prompt_raw = preprocess_features_tensor(normal_prompt_raw)

        logits, logit_residual, emb, emb_residual, normal_prompt, abnormal_prompt, emb_neighbors = model(feat_train, adj_train,
                                                                                          raw_adj_train, normal_prompt_raw,
                                                                                          abnormal_prompt_raw)

        loss_bce = torch.mean(b_xent(torch.squeeze(logits[:, idx_train]), ano_labels_train))

        loss_bce_residual = torch.mean(b_xent(torch.squeeze(logit_residual[:, idx_train]), ano_labels_train))

        # col_normalized = raw_adj_train.sum(0, keepdim=True).sqrt()
        # raw_adj_train_normalized = raw_adj_train.div(col_normalized)
        # adj_train_ref = adj_train[:, idx_train, :][:, :, idx_train]
        # emb_residual = emb[:, idx_train, :] - torch.bmm(adj_train_ref, emb[:, idx_train, :])

        # Extract normal and abnormal prototype
        emb_residual_train = emb_residual[:, idx_train, :]

        normal_proto = emb_residual_train[:, ano_labels_train == 0, :]
        # Add contraints on normal_proto  to zero

        abnormal_proto = emb_residual_train[:, ano_labels_train == 1, :]

        # Compute the dis at node level
        dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2, dim=2))
        # dif_normal = dif_normal.detach().numpy()
        dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=2))
        # dif_abnormal = dif_abnormal.detach().numpy()

        loss_alignment = torch.mean(dif_abnormal) + 0.1*torch.mean(dif_normal)
        # loss_alignment = torch.mean(dif_abnormal) + torch.mean(dif_normal)

        # if epoch < 200:
        #     loss = loss_bce
        # else:
        #     # loss = loss_bce + loss_bce_residual
        #     loss = loss_bce + loss_alignment

        # Conduct BCE on the anomaly score
        # abnormal_prompt_train = F.normalize(abnormal_prompt, p=2, dim=0)
        # normal_prompt_train = F.normalize(normal_prompt, p=2, dim=0)
        # emb_residual_train = F.normalize(torch.squeeze(emb_residual_train), p=2, dim=1)

        # score_normal_train = torch.mm(emb_residual_train, torch.unsqueeze(normal_prompt_train, 1))
        # score_abnormal_train = torch.mm(emb_residual_train, torch.unsqueeze(abnormal_prompt_train, 1))
        # score_normal_train = torch.squeeze(score_normal_train)
        # score_abnormal_train = torch.squeeze(score_abnormal_train)

        # Conduct softmax
        # score_softmax = torch.exp(score_abnormal_train) / (
        #             torch.exp(score_normal_train) + torch.exp(score_abnormal_train))
        # # score_softmax = torch.exp(score_abnormal_train) / (torch.exp(score_normal_train))
        # loss_bce_score = torch.mean(b_xent(score_softmax, ano_labels_train))

        loss = loss_bce + 1*loss_alignment
        loss.backward()
        optimiser.step()
        end_time = time.time()
        total_time += end_time - start_time
        # print('Total time is', total_time)
        # print("Epoch:", '%04d' % (epoch), "loss_bce =", "{:.5f}".format(loss_bce.item()))
        # print("Epoch:", '%04d' % (epoch), "loss_alignment =", "{:.5f}".format(loss_alignment.item()))

        if epoch % 50 == 0:
            logits = np.squeeze(logits.cpu().detach().numpy())
            auc = roc_auc_score(ano_labels_val, logits[idx_val])
            AP = average_precision_score(ano_labels_val, logits[idx_val], average='macro', pos_label=1,
                                         sample_weight=None)
            print("Epoch:", '%04d' % (epoch), "loss =", "{:.5f}".format(loss.item()))
            print('Val Traininig {} AUC:{:.4f}'.format(args.dataset_train, auc))
            print('Val Traininig AP:', AP)

            # logit_residual = np.squeeze(logit_residual.cpu().detach().numpy())
            # auc_residual = roc_auc_score(ano_labels_val, logit_residual[idx_val])
            # AP_residual = average_precision_score(ano_labels_val, logit_residual[idx_val], average='macro',
            #                                       pos_label=1,
            #                                       sample_weight=None)
            # print('Traininig {} AUC residual:{:.4f}'.format(args.dataset_train, auc_residual))
            # print('Traininig AP_residual:', AP_residual)

            emb_residual_val = emb_residual[:, idx_val, :]

            # Obtain the anomaly score on the test data Using Corr
            abnormal_prompt_val = F.normalize(abnormal_prompt, p=2, dim=0)
            normal_prompt_val = F.normalize(normal_prompt, p=2, dim=0)
            #
            emb_residual_val = F.normalize(torch.squeeze(emb_residual_val), p=2, dim=1)
            score_normal = torch.mm(emb_residual_val, torch.unsqueeze(normal_prompt_val, 1))
            score_abnormal = torch.mm(emb_residual_val, torch.unsqueeze(abnormal_prompt_val, 1))
            #
            # ano_score = torch.exp(score_abnormal)

            # ano_score_n = torch.exp(-score_normal) / ((torch.exp(score_abnormal) + torch.exp(-score_normal)))
            # ano_score_n = ano_score_n.detach().numpy()
            # ano_score_ab = torch.exp(score_abnormal) / ((torch.exp(score_abnormal) + torch.exp(-score_normal)))
            # ano_score_ab = ano_score_ab.detach().numpy()

            ano_score_n = -torch.exp(score_normal)
            ano_score_n = ano_score_n.detach().numpy()

            ano_score_ab = torch.exp(score_abnormal)
            ano_score_ab = ano_score_ab.detach().numpy()

            auc_measure_abnormal = roc_auc_score(ano_labels_val, ano_score_ab)
            AP_measure_abnormal = average_precision_score(ano_labels_val, ano_score_ab)

            print('Val Abnormal  Traininig {} AUC measure:{:.4f}'.format(args.dataset_train, auc_measure_abnormal))
            print('Val Abnormal Traininig AP measure:', AP_measure_abnormal)

            auc_measure_normal = roc_auc_score(ano_labels_val, ano_score_n)
            AP_measure_normal = average_precision_score(ano_labels_val, ano_score_n)

            print('Val Normal Traininig {} AUC measure:{:.4f}'.format(args.dataset_train, auc_measure_normal))
            print('Val Normal Traininig AP measure:', AP_measure_normal)

        if epoch % 50 == 0:
            print("<<<<<<<<<<<<<<<<<<<<<< Test begin>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            model.eval()
            # start_time = time.time()
            # Evaluate on other dataset
            # mean_n, std_n = 0, 1
            normal_prompt_raw = torch.randn(args.embedding_dim)
            abnormal_prompt_raw = torch.randn(args.embedding_dim)

            # mean_n, std_n = 0, 0.5
            # mean_a, std_a = 0, 0.5
            # normal_prompt_raw = torch.normal(mean_n, std_n, size=(args.embedding_dim,))
            # abnormal_prompt_raw = torch.normal(mean_a, std_a, size=(args.embedding_dim,))

            # mean_n, std_n = 0.1, 0.5
            # mean_a, std_a = 0.1, 0.5
            # normal_prompt_raw = torch.normal(mean_n, std_n, size=(args.embedding_dim,))
            # abnormal_prompt_raw = torch.normal(mean_a, std_a, size=(args.embedding_dim,))

            # mean_n, std_n = 0.2, 1
            # mean_a, std_a = 0.2, 1
            # normal_prompt_raw = torch.normal(mean_n, std_n, size=(args.embedding_dim,))
            # abnormal_prompt_raw = torch.normal(mean_a, std_a, size=(args.embedding_dim,))

            # normal_prompt = preprocess_features_tensor(normal_prompt)
            # abnormal_prompt = preprocess_features_tensor(abnormal_prompt)

            logits_test, logits_test_residual, emb_test, emb_residual_test, normal_prompt_test, abnormal_prompt_test, emb_neighbors = model(
                feat_test,
                adj_test, raw_adj_test,
                normal_prompt_raw,
                abnormal_prompt_raw)

            # Evaluation on the validation and test node
            logits_test = np.squeeze(logits_test.cpu().detach().numpy())
            auc = roc_auc_score(ano_labels_test, logits_test)
            AP = average_precision_score(ano_labels_test, logits_test, average='macro', pos_label=1, sample_weight=None)
            print('Epoch {}'.format(epoch))
            print('Testing {} AUC:{:.4f}'.format(args.dataset_test, auc))
            print('Testing AP:', AP)
            print("<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>")
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

            # score_normal = torch.cosine_similarity(torch.squeeze(emb_residual_test), normal_prompt_test)
            # score_abnormal = torch.cosine_similarity(torch.squeeze(emb_residual_test), abnormal_prompt_test)

            # Obtain the anomaly score on the test data Using distance
            # score_normal = torch.squeeze(
            #     torch.sqrt(torch.sum((emb_residual_test - torch.unsqueeze(normal_prompt, 0)) ** 2, dim=1)))
            # score_abnormal = torch.squeeze(
            #     torch.sqrt(torch.sum((emb_residual_test - torch.unsqueeze(abnormal_prompt, 0)) ** 2, dim=1)))

            # Conduct softmax
            # ano_score = torch.exp(score_abnormal)/(torch.exp(score_abnormal)+torch.exp(score_normal))

            # Store the embedding for Visualization
            # data_dict = dict([
            #     ('emb_test', emb_test.detach().numpy()),
            #     ('emb_train', emb.detach().numpy()),
            #     ('emb_residual_test', emb_residual_test.detach().numpy()),
            #     ('emb_residual_train', emb_residual_train.detach().numpy()),
            #     ('label_train', ano_labels_train.detach().numpy()),
            #     ('label_test', ano_labels_test.detach().numpy()),
            # ])
            # scio.savemat('tsne2/{}_{}.mat'.format(args.dataset_test, epoch), data_dict)

            # # 保存模型权重到文件
            # torch.save(model.state_dict(), 'pretrain/model_weights_abnormal300.pth')


            ano_score_n = torch.exp(-score_normal)
            ano_score_n = ano_score_n.detach().numpy()

            ano_score_ab = torch.exp(score_abnormal)
            ano_score_ab = ano_score_ab.detach().numpy()


            # ano_score_n = torch.exp(score_normal) / (torch.exp(score_abnormal) + torch.exp(score_normal))
            # ano_score_n = ano_score_n.detach().numpy()
            #
            # ano_score_ab = torch.exp(score_abnormal) / (torch.exp(score_abnormal) + torch.exp(score_normal))
            # ano_score_ab = ano_score_ab.detach().numpy()

            # mean_normal = np.mean(ano_score_n)
            # variance_normal = np.var(ano_score_n)
            # print('Normal mean {} and var {}:'.format(mean_normal, variance_normal))
            #
            # mean_abnormal = np.mean(ano_score_ab)
            # variance_abnormal = np.var(ano_score_ab)
            # print('Abnormal mean {} and var {}:'.format(mean_abnormal, variance_abnormal))

            # ano_score = (-score_normal + score_abnormal).detach().numpy()
            auc_measure_normal = roc_auc_score(ano_labels_test, ano_score_n)
            AP_measure_normal = average_precision_score(ano_labels_test, ano_score_n, average='macro', pos_label=1,
                                                 sample_weight=None)

            print('Normal Testing {} AUC_measure:{:.4f}'.format(args.dataset_test, auc_measure_normal))
            print('Normal Testing AP_measure:', AP_measure_normal)


            print("<<<<<<<<<<<<<<<<<<<")
            auc_measure_abnormal = roc_auc_score(ano_labels_test, ano_score_ab)
            AP_measure_abnormal = average_precision_score(ano_labels_test, ano_score_ab, average='macro', pos_label=1,
                                                 sample_weight=None)

            print('Abnormal Testing {} AUC_measure:{:.4f}'.format(args.dataset_test, auc_measure_abnormal))
            print('Abnormal Testing AP_measure:', AP_measure_abnormal)

            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            ano_score_mid = ano_score_ab + 6 * ano_score_n

            auc_measure_mid = roc_auc_score(ano_labels_test, ano_score_mid )
            AP_measure_mid = average_precision_score(ano_labels_test, ano_score_mid, average='macro', pos_label=1,
                                                 sample_weight=None)

            print('Mid Testing {} AUC_measure:{:.4f}'.format(args.dataset_test, auc_measure_mid))
            print('Mid Testing AP_measure:', AP_measure_mid)
            print("<<<<<<<<<<<<<<<<<<<<<< Test end >>>>>>>>>>>>>>>>>>>>>>>>>>>")
            # end_time = time.time()
            # total_time = end_time - start_time
            # print('Total time is', total_time)