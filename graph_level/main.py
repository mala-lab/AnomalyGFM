import torch
import argparse
import sys, os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from metric import *
from model import GIN
from dataloader import *
import torch.nn as nn
import torch.nn.functional as F
import random
import ipdb


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(args):
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    auc, ap, rec = [], [], []
    seed = 3407
    set_seed(seed)

    dataset, dataloader, meta = get_ad_dataset_TU(args)

    args.min_nodes_num = meta['min_nodes_num']
    args.n_train = meta['num_train']
    args.n_edge_feat = meta['num_edge_feat']

    model = GIN(args.unifeat, args).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio])).to(device)
    # ipdb.set_trace()
    #### Train on source graph datase ####
    for epoch in range(args.num_epoch):
        model.train()
        optimiser.zero_grad()

        normal_prompt = torch.randn(args.hidden_dim * args.num_layer).to(device)
        abnormal_prompt = torch.randn(args.hidden_dim * args.num_layer).to(device)

        for data in dataloader:
            data = data.to(device)
            logits, emb_residual, normal_prompt, abnormal_prompt = model(data, normal_prompt, abnormal_prompt)
            ipdb.set_trace()
            loss_bce = torch.mean(b_xent(logits, torch.unsqueeze(data.y.float(), 1)))
            normal_proto = emb_residual[data.y == 0]
            abnormal_proto = emb_residual[data.y == 1]

            dif_normal = torch.sqrt(torch.sum((normal_prompt - normal_proto) ** 2, dim=1))
            dif_abnormal = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=1))
            loss_alignment = torch.mean(dif_abnormal) + torch.mean(dif_normal)

            abnormal_prompt = F.normalize(abnormal_prompt, p=2, dim=0)
            normal_prompt = F.normalize(normal_prompt, p=2, dim=0)
            emb_residual = F.normalize(emb_residual, p=2, dim=1)

            score_normal = torch.mm(emb_residual, torch.unsqueeze(normal_prompt, 1))
            score_abnormal = torch.mm(emb_residual, torch.unsqueeze(abnormal_prompt, 1))
            score_normal = torch.squeeze(score_normal)
            score_abnormal = torch.squeeze(score_abnormal)

            loss = loss_bce + loss_alignment
            loss.backward()
            optimiser.step()

            print("Epoch:", '%04d' % (epoch), "loss_bce =", "{:.5f}".format(loss_bce.item()))
            print("Epoch:", '%04d' % (epoch), "loss_alignment =", "{:.5f}".format(loss_alignment.item()))

        if epoch % 10 == 0:
            print("<<<<<<<<<<<<<<<<<<<<<< Test begin>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            model.eval()
            test_datasets = ['DD', 'BZR', 'AIDS', 'COX2', 'NCI1', 'DHFR']
            for target_datset in test_datasets:

                target_dataset, target_dataloader, meta = get_ad_dataset_TU(args, target_datset)
                normal_prompt = torch.randn(args.embedding_dim * args.num_layer)
                abnormal_prompt = torch.randn(args.embedding_dim * args.num_layer)
                ano_labels_test = []

                for data in target_dataloader:
                    logits_test, emb_residual_test, normal_prompt_test, abnormal_prompt_test = model(target_dataloader,
                                                                                                     normal_prompt,
                                                                                                     abnormal_prompt)
                    ano_labels_test.append(data.y)

                logits_test = np.squeeze(logits_test.cpu().detach().numpy())
                auc = roc_auc_score(ano_labels_test, logits_test)
                ap = average_precision_score(ano_labels_test, logits_test, average='macro', pos_label=1,
                                             sample_weight=None)
                print(f"{args.DS}-->{target_datset} Testing AUROC:{auc:.4f}, Testing AUPRC:{ap:.4f}")

                abnormal_prompt_test = F.normalize(abnormal_prompt_test, p=2, dim=0)
                normal_prompt_test = F.normalize(normal_prompt_test, p=2, dim=0)

                emb_residual_test = F.normalize(torch.squeeze(emb_residual_test), p=2, dim=1)
                score_normal = torch.mm(emb_residual_test, normal_prompt_test, 1)
                score_abnormal = torch.mm(emb_residual_test, abnormal_prompt_test, 1)

                ano_score = torch.exp(score_abnormal)
                ano_score = ano_score.detach().numpy()
                auc_measure = roc_auc_score(ano_labels_test, ano_score)
                AP_measure = average_precision_score(ano_labels_test, ano_score, average='macro', pos_label=1,
                                                     sample_weight=None)
                print(
                    f"{args.DS}-->{target_datset} Testing AUC_measure:{auc_measure:.4f}, Testing AUC_measure:{AP_measure:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=int, default=0, help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("-data_root", default='data', type=str)
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad', 'ood'])
    parser.add_argument('-DS', help='Dataset', default='AIDS')
    parser.add_argument('-DS_ood', help='Dataset', default='ogbg-molsider')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-unifeat', type=int, default=8)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=100)
    parser.add_argument('-eval_freq', type=int, default=4)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-n_train', type=int, default=10)
    parser.add_argument('-dropout', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('-aggregation', default='add')
    parser.add_argument('--negsamp_ratio', type=int, default=1)
    parser.add_argument('-bias', default=False)
    args = parser.parse_args()

    main(args)
