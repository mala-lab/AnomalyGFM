import torch.nn as nn
from encoder import myGIN
import torch.nn.init as init
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch


class BCE(nn.Module):
    def __init__(self, dim_features, args):
        super(BCE, self).__init__()

        self.dim_targets = args.hidden_dim
        self.num_layers = args.num_layer
        self.device = args.gpu
        self.myGIN = myGIN(dim_features, self.dim_targets, args)
        self.center = torch.zeros(1, self.dim_targets * self.num_layers, requires_grad=False).to('cuda')
        self.b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
        self.ord = 2
        self.reset_parameters()

        self.fc_normal_prompt = nn.Linear(self.dim_targets, self.dim_targets, bias=False)
        self.fc_abnormal_prompt = nn.Linear(self.dim_targets, self.dim_targets, bias=False)

    def forward(self, data, ano_label):
        data = data.to(self.device)
        z, y_predict = self.myGIN(data)  # modifiy GIN

        y_predict = self.fc1(z)

        return  y_predict

    def init_center(self, train_loader):
        with torch.no_grad():
            for data in train_loader:
                data = data.to('cuda')
                z = self.forward(data)
                self.center += torch.sum(z[0], 0, keepdim=True)
            self.center = self.center / len(train_loader.dataset)

    def reset_parameters(self):
        self.net.reset_parameters()

    def loss_func(self, y_predict, ano_label_train):

        loss_bce_score = torch.mean(self.b_xent(y_predict, ano_label_train))
        return  loss_bce_score
