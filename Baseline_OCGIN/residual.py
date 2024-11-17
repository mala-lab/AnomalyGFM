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
        self.reset_parameters()
        self.fc1 =  nn.Linear(args.hidden_dim * self.num_layers, 1, bias=False)

        self.fc_normal_prompt = nn.Linear(self.dim_targets, self.dim_targets, bias=False)
        self.fc_abnormal_prompt = nn.Linear(self.dim_targets, self.dim_targets, bias=False)

    def forward(self, data, ano_label, abnormal_prompt):
        data = data.to(self.device)
        z = self.myGIN(data)  # modifiy GIN
        # mean z_mean
        z_mean = torch.mean(z)
        # normal
        # z_normal_residual = z[ano_label==0] - z_mean
        # abnormal
        abnormal_proto = z[ano_label == 1] - z_mean

        # residual feature
        abnormal_prompt = self.act(self.fc_abnormal_prompt(abnormal_prompt))

        y_predict = self.fc1(z)

        return z, y_predict, abnormal_prompt, abnormal_proto


    def reset_parameters(self):
        self.net.reset_parameters()

    def loss_func(self, y_predict, ano_label_train, abnormal_prompt, abnormal_proto):

        loss_bce_score = torch.mean(self.b_xent(y_predict, ano_label_train.float()))

        loss_alignment = torch.sqrt(torch.sum((abnormal_prompt - abnormal_proto) ** 2, dim=2))

        return loss_alignment + loss_bce_score
