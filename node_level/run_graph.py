
import load_data

import torch.nn as nn

from model import Model
from model import Model_DGI
from utils import *



args = arg_parse()
DS = args.DS
graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)
datanum = len(graphs)
if args.max_nodes == 0:
    node_num_list = [G.number_of_nodes() for G in graphs]
    max_nodes_num = max(node_num_list)
    avg_nodes_num = int(np.ceil(np.mean(node_num_list)))
else:
    max_nodes_num = args.max_nodes

print(datanum, max_nodes_num)
graphs_label = [graph.graph['label'] for graph in graphs]



train_graph =

test_graph =


# Initialize model and optimiser
model = Model(ft_size, input_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
model_dgi = Model_DGI(ft_size, input_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)

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


