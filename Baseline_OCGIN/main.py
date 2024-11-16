import tqdm
import torch
import argparse
import warnings
import sys, os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from metric import *
from utils import init_model
from dataloader import *
import pandas as pd
import random



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
    auc, ap, rec = [], [], []
    seed = 3407
    set_seed(seed)
    if args.exp_type == 'ad':
        print("-------")
        dataset, dataloader, meta = get_ad_dataset_TU(args)


    args.min_nodes_num = meta['min_nodes_num']
    args.n_train = meta['num_train']
    args.n_edge_feat = meta['num_edge_feat']

    model = init_model(args)

    if args.model == "bce":
        print(args.model)
        # train on dataset
        model.fit(dataset=dataset, args=args, label=None, dataloader=dataloader)

        # zero test on other datasets
        test_datasets = ['DD', 'BZR', 'AIDS', 'COX2', 'NCI1', 'DHFR']
        for target_datset in test_datasets:
            # ipdb.set_trace()
            target_dataset, target_dataloader, meta = get_ad_dataset_TU(args, target_datset)
            score, y_all = model.predict(dataset=target_dataset, dataloader=target_dataloader, args=args, return_score=False)
            rec.append(fpr95(y_all, score))
            auc.append(ood_auc(y_all, score))
            ap.append(ood_aupr(y_all, score))
            print(f"{args.DS}-->{target_datset} AUROC:{auc[-1]:.4f}, AUPRC:{ap[-1]:.4f}, FPR95:{rec[-1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="bce", help="supported model: [GLocalKD, GLADC, SIGNET, GOOD-D, GraphDE, CVTGAD].")
    parser.add_argument("-gpu", type=int, default=0, help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("-data_root", default='data', type=str)
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad', 'ood'])
    parser.add_argument('-DS', help='Dataset', default='AIDS')
    parser.add_argument('-DS_ood', help='Dataset', default='ogbg-molsider')
    parser.add_argument('--negsamp_ratio', type=int, default=1)
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
    parser.add_argument('-num_epoch', type=int, default=20)
    parser.add_argument('-eval_freq', type=int, default=4)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-n_train', type=int, default=10)
    parser.add_argument('-dropout', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('-aggregation', default='add')
    parser.add_argument('-bias', default=False)


    args = parser.parse_args()

    main(args)
