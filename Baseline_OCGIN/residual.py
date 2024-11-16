import torch.nn as nn
from encoder import myGIN
import torch.nn.init as init
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import torch


class Residual(nn.Module):
    def __init__(self, dim_features, args):
        super(Residual, self).__init__()

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

    def forward(self, data, ano_label, abnormal_prompt):
        data = data.to(self.device)
        z, y_predict = self.myGIN(data)  # modifiy GIN
        # mean z_mean
        z_mean = torch.mean(z)
        # normal
        # z_normal_residual = z[ano_label==0] - z_mean
        # abnormal
        z_abnormal_residual = z[ano_label == 1] - z_mean

        # residual feature
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))

        # contrastive learning
        y_predict = self.fc1(z)

        return z_abnormal_residual, y_predict, abnormal_prompt

    def init_center(self, train_loader):
        with torch.no_grad():
            for data in train_loader:
                data = data.to('cuda')
                z = self.forward(data)
                self.center += torch.sum(z[0], 0, keepdim=True)
            self.center = self.center / len(train_loader.dataset)

    def reset_parameters(self):
        self.net.reset_parameters()

    def loss_func(self, z_c, ano_labels_train):
        # loss_residual + BCE loss

        loss_bce_score = torch.mean(self.b_xent(z_c, ano_labels_train))

        loss_alignment = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=2))

        # contrastive learning

        return loss_alignment + loss_bce_score
