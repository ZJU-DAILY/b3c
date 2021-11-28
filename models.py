import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter
from torch.nn import Module
import math


class B3C(Module):
    def __init__(self, in_features, hidden, out_features, n_views, n_cluster, dropout=0., alpha=0.5,
                 act=lambda x: x):
        super(B3C, self).__init__()

        self.is_clu = False
        self.n_views = n_views

        self.gdec = nn.ModuleList()
        for v in range(n_views):
            self.gdec.append(InnerProductDecoder(in_features=hidden[-1], act=act))

        self.enc = nn.Sequential(
            nn.Linear(in_features, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1])
        )

        self.dec = nn.Sequential(
            nn.Linear(hidden[-1], hidden[-2]),
            nn.ReLU(),
            nn.Linear(hidden[-2], in_features)
        )

        self.cluster_layer = Parameter(torch.Tensor(n_cluster, hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, features, adjs, input_view, s_mat=None):
        self.fz = self.enc(features)
        self.embeddings = torch.mm(adjs[input_view], self.fz)
        self.s_embeddings = torch.mm(s_mat, self.fz)

        self.embeddings = 0.5 * self.embeddings + 0.5 * self.s_embeddings

        dist = torch.sum(torch.pow(self.embeddings.unsqueeze(1) - self.cluster_layer, 2), 2)
        q = 1.0 / (1.0 + dist)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        topk = torch.topk(dist, 2, -1, False)
        top1 = topk.indices[:, 0]
        top2 = topk.indices[:, 1]

        reconstructed_views = []
        for v in range(self.n_views):
            reconstructed_view = self.gdec[v](self.embeddings)
            reconstructed_views.append(reconstructed_view)

        x_bar = self.dec(self.embeddings)

        return self.embeddings, reconstructed_views, x_bar, q, top1, top2, self.cluster_layer


# co-clustering
class CC(Module):

    def __init__(self, trans_dim=32, act=lambda x: x):
        super().__init__()

        self.act = act
        self.proj = Parameter(torch.FloatTensor(trans_dim, trans_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.proj)

    def forward(self, emb1, emb2):
        tmp = torch.mm(emb1, self.proj)
        recon = torch.mm(tmp, emb2.t())
        return self.act(recon)
