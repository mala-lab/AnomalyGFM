
from sklearn.metrics import roc_auc_score

from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
import numpy as np
from utils import *
import argparse
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(description='')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--beta', type=float, default=4)
args = parser.parse_args()

dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)



subgraph_size = 7
embedding_dim = 300
# load the dataset
dataset = 'tfinance'   # tsocial
sample_feature = np.load('/data/{}_feature_{}_1.npy'.format(dataset, subgraph_size+1))
sample_labels = np.load('/data/{}_label_{}_1.npy'.format(dataset, subgraph_size+1))


nodes_num = sample_labels.shape[0]

sample_feature = torch.tensor(sample_feature).float()


# load the pre-trained model
model = torch.load("model_residual2.pth")
model.eval()

adj = torch.eye(subgraph_size, subgraph_size)
# adj = torch.ones(subgraph_size, subgraph_size)
# adj[-1, :] = 0
# adj[:, -1] = 0
# adj[-1, -1] = 1
adj_norm = torch.tensor(normalize_adj(adj).todense())
# Evaluate on the large scale graph
score_abnormal_all = []
normal_prompt = torch.randn(embedding_dim)
abnormal_prompt = torch.randn(embedding_dim)

for i in range(nodes_num):
    print(i)
    # Generate the adjacent matrix
    # adj = torch.eye(subgraph_size)
    feat = sample_feature[i, :, :]
    # similarity_matrix = cosine_similarity(np.array(feat), np.array(feat))
    # print(np.average(similarity_matrix))

    # feat = preprocess_features_tensor(feat)
    logits_test, logits_test_residual, emb_test, emb_residual_test, normal_prompt_test, abnormal_prompt_test = model(
        feat.unsqueeze(0), adj_norm.unsqueeze(0), adj, normal_prompt, abnormal_prompt)

    # abnormal_prompt_test = F.normalize(abnormal_prompt_test, p=2, dim=0)
    # normal_prompt_test = F.normalize(normal_prompt_test, p=2, dim=0)

    # emb_test = torch.squeeze(emb_test)
    # residual_test = emb_test[-1, :] - torch.mean(emb_test[:-1, :], 0)
    # residual_test = F.normalize(torch.squeeze(residual_test), p=2, dim=0)

    residual_test = torch.squeeze(emb_residual_test[:, -1, :])
    # residual_test = F.normalize(torch.squeeze(residual_test), p=2, dim=0)


    score_abnormal = F.cosine_similarity(residual_test.unsqueeze(0), abnormal_prompt_test.unsqueeze(0))
    score_normal =  F.cosine_similarity(residual_test.unsqueeze(0), normal_prompt_test.unsqueeze(0))

    score = torch.exp(score_abnormal) + args.beta*torch.exp(-score_normal)
    # # #
    # score_abnormal = -F.cosine_similarity(residual_test.unsqueeze(0), normal_prompt_test.unsqueeze(0))

    # score_abnormal = F.cosine_similarity(residual_test.unsqueeze(0), abnormal_prompt_test.unsqueeze(0))\


    score_abnormal_all.append(score.detach().numpy())
    # score_abnormal_all.append(logits_test.detach().numpy())

ano_score1 = np.squeeze(np.array(score_abnormal_all))
ano_score2 = np.exp(np.squeeze(np.array(score_abnormal_all)))

auc_measure1 = roc_auc_score(sample_labels,  ano_score1)
AP_measure1 = average_precision_score(sample_labels, ano_score1, average='macro', pos_label=1,
                                     sample_weight=None)
auc_measure2 = roc_auc_score(sample_labels,  ano_score2)
AP_measure2 = average_precision_score(sample_labels, ano_score2, average='macro', pos_label=1,
                                     sample_weight=None)

print('Testing AUC:{:.4f}'.format(auc_measure1))
print('Testing AP:', AP_measure1)

print('Testing AUC:{:.4f}'.format(auc_measure2))
print('Testing AP:', AP_measure2)
