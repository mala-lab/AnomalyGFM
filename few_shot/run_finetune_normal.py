import torch
import torch.nn as nn

from model import Model_fine_tuning

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
                    default='Facebook_svd')  # Facebook_svd
# add the graph level dataset
parser.add_argument('--dataset_test', type=str,
                            default='elliptic_svd')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--mean', type=float, default=0)
parser.add_argument('--var', type=float, default=0)
parser.add_argument('--num_few_train_shot', type=int, default=100)
parser.add_argument('--tuning', type=bool, default=True)
parser.add_argument('--num_few_shot', type=int, default=100)
parser.add_argument('--beta', type=float, default=0.5)

args = parser.parse_args()

# args.lr = 1e-5
# args.num_epoch = 6
args.lr = 1e-4
args.num_epoch = 15   # large datasets  including elliptic and tfinance    


if args.dataset_test in ['amazon_svd',  'Amazon_upu_svd', 'Disney_svd']:  # small datasets
    args.num_epoch = 5
if args.dataset_test in ['tolokers_svd',  'yelp_svd']:  # middle datasets
    args.num_epoch = 10

if args.dataset_test in ['elliptic_svd',  'tf_finace_svd']:
    args.beta = 4.0

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


feat_test, _ = preprocess_features(feat_test)

# For few-shot setting
all_idx_test = list(range(ano_labels_test.shape[0]))
all_idx_test_normal = list(np.where(ano_labels_test != 1)[0])
random.shuffle(all_idx_test)
random.shuffle(all_idx_test_normal)
# few-shot id for training  [#top n]
few_shot_train_id = all_idx_test_normal[:args.num_few_train_shot]
few_shot_id = few_shot_train_id[:args.num_few_shot]
# few-shot id for testing
few_shot_eval_id = list(set(all_idx_test) - set(few_shot_train_id))

ft_size = feat_test.shape[1]
input_size = 8


raw_adj_test = adj_test

adj_test = normalize_adj(adj_test)

adj_test = (adj_test + sp.eye(adj_test.shape[0])).todense()
raw_adj_test = raw_adj_test.todense()

feat_test = torch.FloatTensor(feat_test[np.newaxis])
feat_test = torch.FloatTensor(feat_test)

raw_adj_test = torch.FloatTensor(raw_adj_test)

adj_test = torch.FloatTensor(adj_test[np.newaxis])

ano_labels_test = torch.FloatTensor(ano_labels_test)

# idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
# idx_test = torch.LongTensor(idx_test)

# Initialize model and optimiser
model = Model_fine_tuning(ft_size, input_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
# optimiser = torch.optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
# xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0


checkpoint = torch.load('pretrain/model_weights_abnormal300.pth')  # 加载模型权重文件
model.load_state_dict(checkpoint, strict=False)  # 加载权重到模型中

for param in model.parameters():
    param.requires_grad = False
#
model.prompt.a.weight.requires_grad = True
model.prompt.a.bias.requires_grad = True
model.prompt.global_emb.requires_grad = True

for name, param in model.named_parameters():
    print(f"{name}: requires_grad = {param.requires_grad}")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    total_time = 0
    if args.tuning is True:
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            normal_prompt_raw = torch.randn(args.embedding_dim, requires_grad=True)
            abnormal_prompt_raw = torch.randn(args.embedding_dim, requires_grad=True)


            logits, logit_residual, emb, emb_residual, normal_prompt, abnormal_prompt, emb_neighbors = model(feat_test,
                                                                                                             adj_test,
                                                                                                             raw_adj_test,
                                                                                                             normal_prompt_raw,
                                                                                                             abnormal_prompt_raw)

            # loss_bce = torch.mean(b_xent(torch.squeeze(logits[:, few_shot_id]), ano_labels_train))

            # Extract normal and abnormal prototype
            normal_proto = emb_residual[:, few_shot_id, :]
            abnormal_proto = emb_residual[:, few_shot_id, :]

            # dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2))
            # dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - normal_proto) ** 2))
            # Compute the dis at node level
            dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2, dim=2))

            dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - normal_proto) ** 2, dim=2))

            # loss_alignment = -torch.mean(dif_abnormal)
            # loss_alignment = -torch.mean(dif_abnormal) + torch.mean(dif_normal)
            loss_alignment = torch.mean(dif_normal)

            loss = loss_alignment
            loss.backward()
            optimizer.step()
            # optimiser_prompt.step()

            print("Epoch:", '%04d' % (epoch), "loss =", "{:.5f}".format(loss.item()))

            if epoch % 2 == 0:
                print("<<<<<<<<<<<<<<<<<<<<<< Test begin>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                model.eval()

                # Evaluate on other dataset under few-shot setting
                normal_prompt_raw = torch.randn(args.embedding_dim)
                abnormal_prompt_raw = torch.randn(args.embedding_dim)

                logits_test, logits_test_residual, emb_test, emb_residual_test, normal_prompt_test, abnormal_prompt_test, emb_neighbors = model(
                    feat_test,
                    adj_test, raw_adj_test,
                    normal_prompt_raw,
                    abnormal_prompt_raw)

                abnormal_prompt_test = F.normalize(abnormal_prompt_test, p=2, dim=0)
                normal_prompt_test = F.normalize(normal_prompt_test, p=2, dim=0)

                emb_residual_test = F.normalize(torch.squeeze(emb_residual_test), p=2, dim=1)

                score_normal = torch.mm(emb_residual_test, torch.unsqueeze(normal_prompt_test, 1))
                score_abnormal = torch.mm(emb_residual_test, torch.unsqueeze(abnormal_prompt_test, 1))

                emb_residual_test = torch.squeeze(emb_residual_test)

                ano_score_n = torch.exp(-score_normal)
                ano_score_n = ano_score_n.detach().numpy()

                ano_score_ab = torch.exp(score_abnormal)
                ano_score_ab = ano_score_ab.detach().numpy()

                # ano_score = ano_score_ab + 4 * ano_score_n
                # ano_score = ano_score_ab + 0.5 * ano_score_n
                ano_score = ano_score_ab + args.beta * ano_score_n
                # ano_score = ano_score_ab
                auc_measure = roc_auc_score(ano_labels_test[few_shot_eval_id], ano_score[few_shot_eval_id])
                AP_measure = average_precision_score(ano_labels_test[few_shot_eval_id], ano_score[few_shot_eval_id],
                                                     average='macro', pos_label=1,
                                                     sample_weight=None)

                print('Testing {} AUC_measure:{:.4f}'.format(args.dataset_test, auc_measure))
                print('Testing AP_measure:', AP_measure)
                print("<<<<<<<<<<<<<<<<<<<<<< Test end >>>>>>>>>>>>>>>>>>>>>>>>>>>")



