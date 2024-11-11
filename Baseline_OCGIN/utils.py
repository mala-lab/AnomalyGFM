from random import choice
from detector import *

from torch_geometric.nn import MLP
from sklearn.ensemble import IsolationForest


def init_model(args):
    weight_decay = 0.01
    model_name = args.model
   
    if model_name == "OCGIN":
        return OCGIN(in_dim=args.unifeat,
                     hid_dim=args.hidden_dim,
                     num_layers=args.num_layer,
                     str_dim=args.dg_dim + args.rw_dim,
                     weight_decay=weight_decay,
                     dropout=args.dropout,
                     lr=args.lr,
                     n_train_data=args.n_train,
                     epoch=args.num_epoch,
                     gpu=args.gpu,
                     batch_size=args.batch_size,
                     grand=False,
                     args=args)
    