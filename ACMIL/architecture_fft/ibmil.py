import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from architecture.network import Classifier_1fc, DimReduction

class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N


class IBMIL_fft(nn.Module):
    def __init__(self, conf, confounder_dim=128, confounder_merge='cat'):
        super(IBMIL_fft, self).__init__()
        self.confounder_merge = confounder_merge
        assert confounder_merge in ['cat', 'add', 'sub']
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, 128, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0)
        self.confounder_path = None
        if conf.c_path:
            print('deconfounding')
            self.confounder_path = conf.c_path
            conf_list = []
            for i in conf.c_path:
                conf_list.append(torch.from_numpy(np.load(i)).view(-1, conf.D_inner).float())
            conf_tensor = torch.cat(conf_list, 0)
            conf_tensor_dim = conf_tensor.shape[-1]
            if conf.c_learn:
                self.confounder_feat = nn.Parameter(conf_tensor, requires_grad=True)
            else:
                self.register_buffer("confounder_feat", conf_tensor)
            joint_space_dim = confounder_dim
            dropout_v = 0.5
            self.W_q = nn.Linear(conf.D_inner, joint_space_dim)
            self.W_k = nn.Linear(conf_tensor_dim, joint_space_dim)
            if confounder_merge == 'cat':
                self.classifier = nn.Linear(conf.D_inner + conf_tensor_dim, conf.n_class)
            elif confounder_merge == 'add' or 'sub':
                self.classifier = nn.Linear(conf.D_inner, conf.n_class)
            self.dropout = nn.Dropout(dropout_v)

        # FFTMIL
        self.conv_fft = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )
        self.fc_fft = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

    def forward(self, x, fft):
        x = x[0]
        x = self.dimreduction(x)

        # FFTMIL
        x2 = fft
        x2 = self.conv_fft(x2)
        min_vals = x2.amin(dim=(-2, -1), keepdim=True)
        max_vals = x2.amax(dim=(-2, -1), keepdim=True)
        x2 = (x2 - min_vals) / (max_vals - min_vals + 1e-8)
        x2 = x2.view(x2.size(0), -1)
        x2 = self.fc_fft(x2)        

        x = x + x2
        A = self.attention(x)  ## K x N
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, x) ## K x L
        # x = x.squeeze(0)

        # M = torch.mm(A, x)  # KxL
        if self.confounder_path:
            device = M.device
            # bag_q = self.confounder_W_q(M)
            # conf_k = self.confounder_W_k(self.confounder_feat)
            bag_q = self.W_q(M)
            conf_k = self.W_k(self.confounder_feat)
            deconf_A = torch.mm(conf_k, bag_q.transpose(0, 1))
            deconf_A = F.softmax(
                deconf_A / torch.sqrt(torch.tensor(conf_k.shape[1], dtype=torch.float32, device=device)),
                0)  # normalize attention scores, A in shape N x C,
            conf_feats = torch.mm(deconf_A.transpose(0, 1),
                                  self.confounder_feat)  # compute bag representation, B in shape C x V
            if self.confounder_merge == 'cat':
                M = torch.cat((M, conf_feats), dim=1)
            elif self.confounder_merge == 'add':
                M = M + conf_feats
            elif self.confounder_merge == 'sub':
                M = M - conf_feats
        Y_prob = self.classifier(M)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        if self.confounder_path:
            return Y_prob, M, deconf_A
        else:
            return Y_prob, M, A
