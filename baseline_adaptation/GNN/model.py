import torch.nn as nn
import torch.nn.functional as F
from utils import *
from dgl.nn.pytorch import edge_softmax
from torch.nn.modules.module import Module
import dgl.function as fn

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True, Nor=False):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.bn = None
        if Nor:
            self.bn = nn.BatchNorm1d(out_ft)
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
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        # ipdb.set_trace()
        if self.bn is not None:
            out = self.bn(out)
        return self.act(out)


class Model(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, 2 * n_h, activation)
        self.gcn2 = GCN(2 * n_h, 1, activation)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, adj):
        feat = self.gcn1(seq, adj)
        feat = self.gcn2(feat, adj)
        output = self.sigmoid(feat)
        return output


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.2,
                 attn_drop=0.2,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 k = 1):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
            
        self.attn_l1 = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r1 = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.lrelu = nn.Sigmoid()#nn.LeakyReLU()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l1, gain=gain)
        nn.init.xavier_normal_(self.attn_r1, gain=gain)        
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

        
    def forward(self, graph, feat, device):
        r"""Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        elist = []
        graph = graph.local_var().to(device)
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l1).sum(dim=-1).unsqueeze(-1) 
        er = (feat * self.attn_r1).sum(dim=-1).unsqueeze(-1) 
        graph.ndata.update({'ft': feat, 'el': el, 'er': er}) 
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))      
        e = self.leaky_relu(graph.edata.pop('e'))  
        e_soft = edge_softmax(graph, e)
        elist.append(e_soft)
        graph.edata['a'] = self.attn_drop(e_soft)       
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft')) 
        rst = graph.ndata['ft'] 
        if self.activation:
            rst = self.activation(rst)
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        return rst, elist

    def forward_batch(self, block, feat):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        elist = []
        block = block.local_var().to('cuda:{}'.format(feat.get_device()))
        h_src = h_dst = self.feat_drop(feat)
        feat_src = self.fc(h_src).view(
            -1, self._num_heads, self._out_feats)
        feat_dst = feat_src[:block.number_of_dst_nodes()] # the first few nodes are dst nodes, as explained in https://docs.dgl.ai/tutorials/large/L1_large_node_classification.html
        el = (feat_src * self.attn_l1).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r1).sum(dim=-1).unsqueeze(-1)
        #block.srcdata.update({'ft': feat, 'el': el, 'er': er})
        block.srcdata.update({'ft': feat_src, 'el': el})
        block.dstdata.update({'er': er})
        block.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(block.edata.pop('e'))
        e_soft = edge_softmax(block, e)
        elist.append(e_soft)
        block.edata['a'] = self.attn_drop(e_soft)
        block.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = block.dstdata['ft']
        if self.activation:
            rst = self.activation(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        return rst, elist

class PairNorm(nn.Module):
    def __init__(self, mode='PN-SCS', scale=1.0):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version
           
            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
        
    def forward(self, x):
        if self.mode == 'None':
            return x
        
        col_mean = x.mean(dim=0)      
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x




class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        #self.g = g
        heads = 8
        self.num_layers = 2
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.activation = F.elu
        self.gat_layers.append(GATConv(args.unifeat, args.embedding_dim, 8))
        self.norm_layers.append(PairNorm())
        
        # hidden layers
        for l in range(1, 2):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(args.embedding_dim * heads, args.embedding_dim, heads, activation=self.activation))
            # self.norm_layers.append(nn.BatchNorm1d(num_hidden*heads[l]))
            self.norm_layers.append(PairNorm())
        # output projection

        self.gat_layers.append(GATConv(args.embedding_dim * heads, 1, 1))

    def forward(self, g, inputs, device, save_logit_name = None):
        h = inputs
        e_list = []
        for l in range(self.num_layers):
            h, e = self.gat_layers[l](g, h, device)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        # store for ergnn
        self.second_last_h = h
        # output projection
        logits, e = self.gat_layers[-1](g, h, device)
        #self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        return logits

    def forward_batch(self, blocks, features):
        e_list = []
        h = features
        for i,layer in enumerate(self.gat_layers[:-1]):
            h, e = layer.forward_batch(blocks[i], h)
            h = h.flatten(1)
            h = self.activation(h)
            e_list = e_list + e
        logits, e = self.gat_layers[-1].forward_batch(blocks[-1], h)
        self.second_last_h = logits if len(self.gat_layers) == 1 else h
        logits = logits.mean(1)
        e_list = e_list + e
        return logits, e_list


    def reset_params(self):
        for layer in self.gat_layers:
            layer.reset_parameters()