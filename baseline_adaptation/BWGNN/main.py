import dgl
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from dataset import loaddata
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from BWGNN import *
from sklearn.model_selection import train_test_split

import time

def train(model, graphs, args):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    thress = np.zeros(2)
    for e in range(args.epoch):
        for dataset in range(len(graphs)):
            g = graphs[dataset]
            features = g.ndata['feature']
            labels = g.ndata['label']
            weight = (1-labels).sum().item() / labels.sum().item()
            model.train()
            logits = model(g, features)
            loss = F.cross_entropy(logits, labels, weight=torch.tensor([1., weight]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.eval()
            probs = logits.softmax(1)
            f1, thres = get_best_f1(labels, probs)
            thress[dataset] = thres
    return thress

def inference(model, graph, args, thress):
    thres = np.mean(thress)
    features = graph.ndata['feature']
    labels = graph.ndata['label']
    model.eval()
    logits = model(graph, features)
    probs = logits.softmax(1)
    tauc = roc_auc_score(labels, probs[:, 1].detach().numpy())
    tap = average_precision_score(labels, probs[:, 1].detach().numpy(), average='macro', pos_label=1, sample_weight=None)
    return tauc, tap, probs, labels

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="Facebook", help="Dataset for this model (yelp/amazon/tfinance/tsocial)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    parser.add_argument('--unifeat', type=int, default=10)
    parser.add_argument('--dimreduction', type=str, default='svd')

    args = parser.parse_args()
    print(args)
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    in_feats = args.unifeat
    num_classes = 2


    traindatasets = [args.dataset]
    graphs = []
    for dataset in traindatasets:
        graph = loaddata(dataset, args)
        graphs.append(graph)

    all_aucs = []
    all_aps = []

    for _ in range(5):
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        start_time = time.time() 
        thr = train(model, graphs, args)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time
        print(f"Training time: {elapsed_time} seconds")
        aucs = []
        aps = []
        # target_datasets = ['Amazon', 'Reddit', 'weibo', 'YelpChi', 'Amazon-all', 'YelpChi-all']
        # target_datasets = ['tf_finace', 'elliptic']
        # target_datasets = ['book', 'Disney']
        target_datasets = ['tsocial']

        start_time = time.time()
        for dataset in target_datasets:
            graph = loaddata(dataset, args)
            test_auc, test_ap, probs, labels = inference(model, graph, args, thr)
            aucs.append(test_auc)
            aps.append(test_ap)
            print('{} -> {} GC AUC:{:.4f} AP{:.4f}'.format(args.dataset, dataset, test_auc, test_ap))
            np.savez('BWGNNScore%s.npz'%dataset, comp=probs.detach().numpy(), label=labels)
            with open(f'results/{args.dataset}.txt','a') as f:
                f.write('\n{} -> {} AUC:{:.4f} AP{:.4f}\n'.format(args.dataset, dataset, test_auc, test_ap))
        all_aucs.append(aucs)
        all_aps.append(aps)
        # end_time = time.time()  # End the timer
        # elapsed_time = end_time - start_time
        # print(f"Testing time: {elapsed_time} seconds")
    
    all_aucs, all_aps = np.array(all_aucs), np.array(all_aps)
    mean_auc, std_auc = np.mean(all_aucs, 0), np.std(all_aucs, 0)
    mean_ap, std_ap = np.mean(all_aps, 0), np.std(all_aps, 0)

    for i, dataset in enumerate(target_datasets):
        with open(f'results/{args.dataset}.txt','a') as f:
            f.write('\n Averaged {} -> {} AUC:{:.4f}$_{{\\pm {:.3f}}}$ AP:{:.4f}$_{{\\pm {:.3f}}}$\n'.format(args.dataset, dataset, mean_auc[i], std_auc[i], mean_ap[i], std_ap[i]))
 
        

    # else:
    #     final_mf1s, final_aucs = [], []
    #     for tt in range(args.run):
    #         if homo:
    #             model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
    #         else:
    #             model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
    #         mf1, auc = train(model, graph, args)
    #         final_mf1s.append(mf1)
    #         final_aucs.append(auc)
    #     final_mf1s = np.array(final_mf1s)
    #     final_aucs = np.array(final_aucs)
    #     print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s), 100 * np.std(final_mf1s), 100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
