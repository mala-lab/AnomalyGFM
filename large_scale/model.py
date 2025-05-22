import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias

        return self.act(out)


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq, 1).values


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0, 2, 1)
        sim = torch.matmul(seq, query)
        sim = F.softmax(sim, dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq, sim)
        out = torch.sum(out, 1)
        return out


# class Discriminator(nn.Module):
#     def __init__(self, n_h, negsamp_round):
#         super(Discriminator, self).__init__()
#         self.f_k = nn.Bilinear(n_h, n_h, 1)
#
#         for m in self.modules():
#             self.weights_init(m)
#
#         self.negsamp_round = negsamp_round
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, c, h_pl):
#         scs = []
#         # positive
#         scs.append(self.f_k(h_pl, c))
#
#         # negative
#         c_mi = c
#         for _ in range(self.negsamp_round):
#             c_mi = torch.cat((c_mi[-2:-1, :], c_mi[:-1, :]), 0)
#             scs.append(self.f_k(h_pl, c_mi))
#
#         logits = torch.cat(tuple(scs))
#
#         return logits

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Model(nn.Module):
    def __init__(self, n_in_1, n_in_2, n_h, activation, negsamp_round, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.fc_map = nn.Linear(n_in_1, n_in_2, bias=False)

        self.gcn1 = GCN(n_in_2, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, 1, bias=False)
        self.fc2 = nn.Linear(n_h, 1, bias=False)
        self.act = nn.PReLU()

        self.fc_normal_prompt = nn.Linear(n_h, n_h, bias=False)
        self.fc_abnormal_prompt = nn.Linear(n_h, n_h, bias=False)
        # Generate the prompt embedding of normal and abnormal node
        # self.normal_prompt = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.abnormal_prompt = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)n

        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

    def forward(self, seq1, adj, raw_adj, normal_prompt, abnormal_prompt, sparse=False):
        # seq1 = self.fc_map(seq1)
        h_1 = self.gcn1(seq1, adj, sparse)
        emb = self.gcn2(h_1, adj, sparse)

        normal_prompt = self.act(self.fc_normal_prompt(normal_prompt))
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))

        # residual feature
        raw_adj = raw_adj * (1 - torch.eye(raw_adj.size(0)))
        col_normalized = raw_adj.sum(1, keepdim=True)
        adj_normalized = raw_adj.div(col_normalized)
        adj_normalized[torch.isinf(adj_normalized)] = 0
        adj_normalized[torch.isnan(adj_normalized)] = 0

        emb_residual = emb - torch.bmm(torch.unsqueeze(adj_normalized, 0), emb)
        # emb = emb - torch.mean(emb, 1)
        # emb = torch.cat((emb_residual, emb), 1)

        logit = self.fc1(emb)
        logit_residual = self.fc2(emb_residual)

        return logit, logit_residual, emb, emb_residual, normal_prompt, abnormal_prompt


class Model_DGI(nn.Module):
    def __init__(self, n_in_1, n_in_2, n_h, activation, negsamp_round, readout):
        super(Model_DGI, self).__init__()
        self.read_mode = readout
        self.fc_map = nn.Linear(n_in_1, n_in_2, bias=False)

        self.gcn1 = GCN(n_in_2, n_h, activation)
        self.gcn2 = GCN(n_h, n_h, activation)
        self.gcn3 = GCN(n_h, n_h, activation)
        self.fc1 = nn.Linear(n_h, 1, bias=False)
        self.fc2 = nn.Linear(n_h, 1, bias=False)

        self.fc_normal_prompt = nn.Linear(n_h, n_h, bias=False)
        self.fc_abnormal_prompt = nn.Linear(n_h, n_h, bias=False)
        # Generate the prompt embedding of normal and abnormal node
        # self.normal_prompt = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.abnormal_prompt = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)n

        self.act = nn.ReLU()
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(n_h)
        self.sigm = nn.Sigmoid()

    def forward(self, seq1, seq2, adj, raw_adj, normal_prompt, abnormal_prompt, sparse=False):
        # seq1 = self.fc_map(seq1)
        h_1 = self.gcn1(seq1, adj, sparse)
        emb = self.gcn2(h_1, adj, sparse)

        c = self.read(h_1)
        c = self.sigm(c)

        h_2 = self.gcn1(seq2, adj, sparse)
        h_2 = self.gcn2(h_2, adj, sparse)

        ret = self.disc(c, h_1, h_2)

        normal_prompt = self.fc_normal_prompt(normal_prompt)
        abnormal_prompt = self.fc_abnormal_prompt(abnormal_prompt)

        # residual feature
        raw_adj = raw_adj * (1 - torch.eye(raw_adj.size(0)))
        col_normalized = raw_adj.sum(1, keepdim=True)
        adj_normalized = raw_adj.div(col_normalized)
        adj_normalized[torch.isinf(adj_normalized)] = 0
        adj_normalized[torch.isnan(adj_normalized)] = 0

        emb_residual = emb - torch.bmm(torch.unsqueeze(adj_normalized, 0), emb)
        # emb = emb - torch.mean(emb, 1)
        # emb = torch.cat((emb_residual, emb), 1)

        # logit = self.fc1(emb)
        logit_residual = self.fc2(emb_residual)

        return ret, logit_residual, emb, emb_residual, normal_prompt, abnormal_prompt
