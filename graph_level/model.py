import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool,global_max_pool
from torch.nn import Linear, ReLU, ModuleList, Sequential, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGPooling, TopKPooling, BatchNorm, global_add_pool, global_mean_pool
from torch_geometric.utils import batched_negative_sampling, dropout_adj
from torch_scatter import scatter
import ipdb

class GraphNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super(GraphNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim),requires_grad=affine)
        self.bias = nn.Parameter(torch.zeros(dim),requires_grad=False)
        self.scale = nn.Parameter(torch.ones(dim),requires_grad=affine)
    def forward(self,node_emb,graph):
        try:
            num_nodes_list = torch.tensor(graph.__num_nodes_list__).long().to(node_emb.device)
        except:
            num_nodes_list = graph.ptr[1:]-graph.ptr[:-1]

        graph_batch_size = graph.batch.max().item() + 1
        num_nodes_list = num_nodes_list.long().to(node_emb.device)
        node_mean = scatter(node_emb, graph.batch, dim=0, dim_size=graph_batch_size, reduce='mean')
        node_mean = node_mean.repeat_interleave(num_nodes_list, 0)

        sub = node_emb - node_mean*self.scale
        node_std = scatter(sub.pow(2), graph.batch, dim=0, dim_size=graph_batch_size, reduce='mean')
        node_std = torch.sqrt(node_std + 1e-8)
        node_std = node_std.repeat_interleave(num_nodes_list, 0)
        norm_node = self.weight * sub / node_std + self.bias
        return norm_node
    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)
        init.ones_(self.scale)

class myGIN(torch.nn.Module):
    def __init__(self, dim_features, dim_targets, args):
        super(myGIN, self).__init__()
        hidden_dim = args.hidden_dim
        self.num_layers = args.num_layer
        self.nns = []
        self.convs = []
        self.norms = []
        self.projs = []
        self.use_norm = 'gn'
        bias = args.bias

        if args.aggregation == 'add':
            self.pooling = global_add_pool
        elif args.aggregation == 'mean':
            self.pooling = global_mean_pool

        for layer in range(self.num_layers):
            if layer == 0:
                input_emb_dim = dim_features
            else:
                input_emb_dim = hidden_dim
            self.nns.append(Sequential(Linear(input_emb_dim, hidden_dim, bias=bias), ReLU(), Linear(hidden_dim, hidden_dim, bias=bias)))
            self.convs.append(GINConv(self.nns[-1], train_eps=bias))  # Eq. 4.2
            if self.use_norm == 'gn':
                self.norms.append(GraphNorm(hidden_dim, True))
            self.projs.append(myMLP(hidden_dim, hidden_dim, dim_targets, bias))

        self.nns = nn.ModuleList(self.nns)
        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.projs = nn.ModuleList(self.projs)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        z_cat = []
        for layer in range(self.num_layers):
            x = self.convs[layer](x, edge_index)
            if self.use_norm == 'gn':
                x = self.norms[layer](x, graph)
            x = F.relu(x)
            z = self.projs[layer](x)
            z = self.pooling(z, batch)
            z_cat.append(z)
        z_cat = torch.cat(z_cat, -1)
        return z_cat

    def reset_parameters(self):
        for norm in self.norms:
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for proj in self.projs:
            proj.reset_parameters()

class myMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, bias=True):
        super(myMLP, self).__init__()
        self.lin1 = Linear(in_dim, hidden, bias=bias)
        self.lin2 = Linear(hidden, out_dim, bias=bias)

    def forward(self, z):
        z = self.lin2(F.relu(self.lin1(z)))
        return z
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


class GIN(nn.Module):
    def __init__(self, dim_features, args):
        super(GIN, self).__init__()

        self.dim_targets = args.hidden_dim
        self.num_layers = args.num_layer
        self.net = myGIN(dim_features, self.dim_targets, args)
        self.fc1 = nn.Linear(self.dim_targets*args.num_layer, 1, bias=False)
        self.act = nn.PReLU()

        self.fc_normal_prompt = nn.Linear(self.dim_targets*args.num_layer, self.dim_targets*args.num_layer, bias=False)
        self.fc_abnormal_prompt = nn.Linear(self.dim_targets*args.num_layer, self.dim_targets*args.num_layer, bias=False)
        self.act = nn.ReLU()

        self.reset_parameters()

    def forward(self, data, normal_prompt, abnormal_prompt):
        # ipdb.set_trace()
        Z = self.net(data)
        normal_prompt = self.act(self.fc_normal_prompt(normal_prompt))
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))
        adj = self.constructA(Z)
        emb_residual = self.residual(Z, adj)
        logit = self.fc1(Z)

        return logit, emb_residual, normal_prompt, abnormal_prompt
    
    def residual(self, Z, adj):
        adj = adj.fill_diagonal_(0)
        col_normalized = adj.sum(1, keepdim=True)
        adj_normalized = adj.div(col_normalized)
        adj_normalized[torch.isinf(adj_normalized)] = 0
        adj_normalized[torch.isnan(adj_normalized)] = 0
        emb_residual = Z - torch.matmul(adj_normalized, Z)
        return emb_residual
    
    def constructA(self, Z, threshold=0.9):
        Z = F.normalize(Z, p=2, dim=1)
        adj = torch.matmul(Z, Z.T)
        adj[adj>=threshold] = 1
        adj[adj<threshold] = 0
        adj = adj.detach()
        return adj

    def reset_parameters(self):
        self.net.reset_parameters()

