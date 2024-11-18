import torch
from torch_geometric.nn import GCN
import ocgin
import residual
import bce
import numpy as np
from metric import *
import os
import torch.nn.functional as F

class OCGIN(torch.nn.Module):
    def __init__(self,
                 in_dim=None,
                 hid_dim=64,
                 num_layers=2,
                 str_dim=64,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 args=None,
                 **kwargs):
        super(OCGIN, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.beta = beta
        self.warmup = warmup
        self.eps = eps
        self.args = args

    def process_graph(self, data):
        pass

    def init_model(self):
        # return ocgin.OCGIN(dim_features=self.in_dim, args = self.args).to(self.device)
        return residual.Residual(dim_features=self.in_dim, args=self.args).to(self.device)

    def fit(self, dataset, args=None, label=None, dataloader=None):
        print("this is ocgin")
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.model.train()
        self.decision_score_ = None
        self.train_dataloader = dataloader

        stop_counter = 0
        N = 30

        for epoch in range(1, args.num_epoch + 1):
            all_loss, n_bw = 0, 0
            for data in dataloader:
                n_bw += 1
                data = data.to(self.device)
                loss_epoch = self.forward_model(data, dataloader, args, False)
                loss_mean = loss_epoch.mean()
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                all_loss += loss_epoch.sum()
            mean_loss = all_loss.item() / args.n_train
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, mean_loss))

            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.model.eval()
                y_val = []
                score_val = []
                for data in dataloader:
                    data = data.to(self.device)
                    score_epoch = self.forward_model(data, dataloader, args, True)
                    score_val = score_val + score_epoch.detach().cpu().tolist()
                    y_true = data.y
                    y_val = y_val + y_true.detach().cpu().tolist()

                val_auc = ood_auc(y_val, score_val)
                print('Epoch:{:03d} | val_auc:{:.4f}'.format(epoch, val_auc))
        return self

    def is_directory_empty(self, directory):
        files_and_dirs = os.listdir(directory)
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        self.model.eval()

        y_score_all = []
        y_true_all = []
        for data in dataloader:
            data = data.to(self.device)
            y_score = self.forward_model(data, dataloader, args, True)
            # outlier_score[node_idx[:batch_size]] = y_score
            y_score_all = y_score_all + y_score.detach().cpu().tolist()
            y_true = data.y
            y_true_all = y_true_all + y_true.detach().cpu().tolist()
        return y_score_all, y_true_all

    def forward_model(self, data, ano_label_train, abnormal_prompt, abnormal_proto):
        emb = self.model(data)
        loss = self.model.loss_func(emb, ano_label_train,  abnormal_prompt, abnormal_proto)
        return loss

    def predict(self,
                dataset=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False,
                return_emb=False,
                dataloader=None,
                args=None):

        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score, y_all = self.decision_function(dataset, label, dataloader, args)
            output = (score, y_all)
            return output


class BCE(torch.nn.Module):
    def __init__(self,
                 in_dim=None,
                 hid_dim=64,
                 num_layers=2,
                 str_dim=64,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 args=None,
                 **kwargs):
        super(BCE, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.beta = beta
        self.warmup = warmup
        self.eps = eps
        self.args = args

    def process_graph(self, data):
        pass

    def init_model(self):
        return bce.BCE(dim_features=self.in_dim, args=self.args).to(self.device)

    def fit(self, dataset, args=None, label=None, dataloader=None):
        print("this is bce")
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.model.train()
        self.decision_score_ = None
        self.train_dataloader = dataloader

        stop_counter = 0
        N = 30

        for epoch in range(1, args.num_epoch + 1):
            all_loss, n_bw = 0, 0
            for data in dataloader:
                n_bw += 1
                data = data.to(self.device)
                ano_label_train = data.y
                loss_epoch, y_predict = self.forward_model(data, ano_label_train)
                loss_mean = loss_epoch.mean()
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                all_loss += loss_epoch.sum()
            mean_loss = all_loss.item() / args.n_train
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, mean_loss))

            # if (epoch) % args.eval_freq == 0 and epoch > 0:
            #     self.model.eval()

            #     y_val = []
            #     score_val = []
            #     for data in dataloader:
            #         data = data.to(self.device)
            #         score_epoch = self.forward_model(data, dataloader, args,True)
            #         score_val = score_val + score_epoch.detach().cpu().tolist()
            #         y_true = data.y
            #         y_val = y_val + y_true.detach().cpu().tolist()

            #     val_auc = ood_auc(y_val, score_val)
            #     print('Epoch:{:03d} | val_auc:{:.4f}'.format(epoch, val_auc))
        return self

    def is_directory_empty(self, directory):
        files_and_dirs = os.listdir(directory)
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, dataloader=None, args=None):
        self.model.eval()

        y_score_all = []
        y_true_all = []
        for data in dataloader:
            data = data.to(self.device)
            loss, y_score = self.forward_model(data, data.y)
            # outlier_score[node_idx[:batch_size]] = y_score
            y_score_all += y_score.detach().cpu().tolist()
            y_true = data.y
            y_true_all += y_true.detach().cpu().tolist()
        return y_score_all, y_true_all

    def forward_model(self, data, ano_label_train):
        y_predict = self.model(data)
        loss = self.model.loss_func(torch.squeeze(y_predict), ano_label_train)
        return loss, y_predict

    def predict(self,
                dataset=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False,
                return_emb=False,
                dataloader=None,
                args=None):

        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score, y_all = self.decision_function(dataset, dataloader, args)
            output = (score, y_all)
            return output


class Residual(torch.nn.Module):
    def __init__(self,
                 in_dim=None,
                 hid_dim=64,
                 num_layers=2,
                 str_dim=64,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 args=None,
                 **kwargs):
        super(Residual, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.beta = beta
        self.warmup = warmup
        self.eps = eps
        self.args = args
        self.abnormal_prompt = torch.randn(hid_dim)

    def process_graph(self, data):
        pass

    def init_model(self):
        return residual.Residual(dim_features=self.in_dim, args=self.args).to(self.device)

    def fit(self, dataset, args=None, label=None, dataloader=None):
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.model.train()
        self.decision_score_ = None
        self.train_dataloader = dataloader

        stop_counter = 0
        N = 30

        for epoch in range(1, args.num_epoch + 1):
            all_loss, n_bw = 0, 0
            for data in dataloader:
                n_bw += 1
                data = data.to(self.device)
                ano_label_train = data.y
                loss_epoch, y_predict, abnormal_prompt, abnormal_residual, all_residual = self.forward_model(data, ano_label_train)
                loss_mean = loss_epoch.mean()
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()
                all_loss += loss_epoch.sum()
            mean_loss = all_loss.item() / args.n_train
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, mean_loss))

            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.model.eval()

                y_val = []
                score_val = []
                for data in dataloader:
                    data = data.to(self.device)
                    ano_label_val = data.y
                    score_epoch, y_predict, abnormal_prompt, abnormal_residual, all_residual = self.forward_model(data, ano_label_val)

                    abnormal_prompt = F.normalize(abnormal_prompt, p=2, dim=0)
                    all_residual = F.normalize(torch.squeeze(all_residual), p=2, dim=1)

                    score_abnormal = torch.mm(all_residual, torch.unsqueeze(abnormal_prompt, 1))
                    y_score = torch.exp(score_abnormal)

                    score_val += y_score.detach().cpu().tolist()
                    y_true = data.y
                    y_val += y_true.detach().cpu().tolist()

                val_auc = ood_auc(y_val, score_val)
                print('Epoch:{:03d} | val_auc:{:.4f}'.format(epoch, val_auc))
        return self

    def is_directory_empty(self, directory):
        files_and_dirs = os.listdir(directory)
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        self.model.eval()
        # using the similarity measurement with abnormal prompt
        y_score_all = []
        y_true_all = []
        for data in dataloader:
            data = data.to(self.device)
            ano_label_test = data.y
            y_score,  y_predict, abnormal_prompt, abnormal_residual, all_residual = self.forward_model(data, ano_label_test)
            # outlier_score[node_idx[:batch_size]] = y_score

            abnormal_prompt = F.normalize(abnormal_prompt, p=2, dim=0)
            all_residual = F.normalize(torch.squeeze(all_residual), p=2, dim=1)

            score_abnormal = torch.mm(all_residual, torch.unsqueeze(abnormal_prompt, 1))
            y_score = torch.exp(score_abnormal)

            y_score_all += y_score.detach().cpu().tolist()
            y_true = data.y
            y_true_all += y_true.detach().cpu().tolist()

        return y_score_all, y_true_all

    def forward_model(self, data, ano_label_train):
        abnormal_prompt = torch.randn(self.hid_dim).cuda()
        emb, y_predict, abnormal_prompt, abnormal_residual, all_residual = self.model(data, ano_label_train, abnormal_prompt)
        loss = self.model.loss_func(torch.squeeze(y_predict), ano_label_train, abnormal_prompt, abnormal_residual)
        return loss, y_predict, abnormal_prompt, abnormal_residual,  all_residual

    def predict(self,
                dataset=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False,
                return_emb=False,
                dataloader=None,
                args=None):

        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score, y_all = self.decision_function(dataset, label, dataloader, args)
            output = (score, y_all)
            return output
