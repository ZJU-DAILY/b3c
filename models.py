import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn.parameter import Parameter
from torch.nn import Module
import math


class GraphEncoder(Module):
    def __init__(self, in_features, hidden, order, dropout=0., act=F.relu):
        super(GraphEncoder, self).__init__()
        self.order = order
        self.dropout = dropout
        self.act = act
        self.n_layers = len(hidden)
        self.encoders = nn.ModuleList()
        hidden.insert(0, in_features)

        for i in range(len(hidden) - 1):
            self.encoders.append(GraphConvolution(hidden[i], hidden[i + 1], dropout))

    def forward(self, input, adjs):
        x = input
        for i in range(len(self.order)):  # todo: no shortcut here
            x = self.encoders[i](x, adjs[self.order[i]])

        return x


class AE(Module):
    def __init__(self, in_features, hidden):
        super(AE, self).__init__()

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

    def forward(self, input):
        z = self.enc(input)
        x_bar = self.dec(z)

        return z, x_bar


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class B3C(Module):
    def __init__(self, in_features, hidden, out_features, n_views, n_cluster, dropout=0., alpha=0.5,
                 act=lambda x: x):
        super(B3C, self).__init__()

        self.is_clu = False
        self.n_views = n_views
        # self.alpha = alpha

        # self.s_layer = GraphConvolution(hidden[-1], hidden[-1], dropout)
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

        # self.fc = nn.Linear(hidden[-1], hidden[-1])
        # self.fc1 = nn.Linear(hidden[-1], hidden[-1])

        # self.gamma = Parameter(torch.zeros(1))
        # self.attn = Attention(hidden[-1])

        # self.clu_layer = ClusteringLayer(out_features, n_cluster)
        self.cluster_layer = Parameter(torch.Tensor(n_cluster, hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # self.dcs = SampleDecoder(act=lambda x: x)

    def forward(self, features, adjs, input_view, s_mat=None):
        # print('input view:', input_view)

        # self.gz = self.genc(features, adjs)\
        # encoding features
        self.fz = self.enc(features)

        # self.embeddings=self.s_gcn1(features,adjs[input_view])
        # self.embeddings=self.s_gcn2(self.embeddings,adjs[input_view])
        # self.embeddings = (1 - self.alpha) * self.embeddings + self.alpha * self.fz   #1
        # self.embeddings=torch.mm(adjs[input_view],self.embeddings)
        #
        #
        # self.s_embeddings=self.f_gcn1(features,s_mat)
        # self.s_embeddings=self.f_gcn2(self.s_embeddings,s_mat)
        # self.s_embeddings = (1 - self.alpha) * self.s_embeddings + self.alpha * self.fz   #1
        # self.s_embeddings=torch.mm(s_mat,self.s_embeddings)
        #
        # emb = torch.stack([self.embeddings, self.s_embeddings], dim=1)
        # self.embeddings, att = self.attn(emb)

        # self.mix_z = (1 - self.alpha) * self.gz + self.alpha * self.fz   #1
        self.embeddings = torch.mm(adjs[input_view], self.fz)
        self.s_embeddings = torch.mm(s_mat, self.fz)

        self.embeddings = 0.5 * self.embeddings + 0.5 * self.s_embeddings

        # self.embeddings=1.0/3*self.embeddings+1.0/3*self.s_embeddings+1.0/3*self.fz
        # emb = torch.stack([self.embeddings, self.s_embeddings,self.fz], dim=1)
        # self.embeddings, att = self.attn(emb)

        #
        # s=torch.mm(self.embeddings,self.embeddings.t())
        # s=F.softmax(s,dim=1)
        # g_z=torch.mm(s,0.5*self.fz+0.5*self.embeddings)
        # self.embeddings=self.embeddings+0.1*g_z

        # print(att)

        # pooling
        # self.affi_emb=torch.mm(adjs[1],self.fz)
        # self.affi_emb=self.fc(self.affi_emb)

        # self.s_embeddings=torch.mm(s_mat,self.fz)
        # self.s_embeddings = self.fc1(self.s_embeddings)
        # self.embeddings = torch.mm(adjs[input_view], self.embeddings)  # 2
        # self.embeddings = self.mix_layer(self.fz, adjs[input_view], active=False)    #3
        # self.embeddings = self.s_layer(self.embeddings,adjs[input_view],active=False)
        # self.embeddings_1 = torch.mm(adjs[1], self.fz)  # 2
        # self.embeddings_1=self.fc1(self.embeddings_1)
        # emb = torch.stack([self.embeddings, self.embeddings_1], dim=1)
        # self.embeddings, att = self.attn(emb)
        # self.embeddings_1=torch.mm(adjs[1],self.fz)
        # self.embeddings=self.embeddings+self.gamma*self.embeddings_1
        # print(self.gamma)
        # self.embeddings = self.mix_layer(self.fz, adjs[input_view], active=True)    #3
        # z_s=self.s_layer(self.fz,sm,active=True)
        # self.embeddings=0.5*self.fz+0.5*self.embeddings    #4

        # self.embeddings=0.5*self.fz+0.5*self.embeddings+self.gamma*z_s    #4
        # emb = torch.stack([self.embeddings, self.fz], dim=1)
        # self.embeddings, att = self.attn(emb)
        # print(att)

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


class Model_gcn_encoder(Module):
    def __init__(self, in_features, hidden, out_features, n_views, order, n_cluster, alpha=0.5,
                 dropout=0., act=lambda x: x):  # out_features=hidden(len(hidden)-1)
        super(Model_gcn_encoder, self).__init__()

        self.is_clu = False
        self.n_views = n_views
        self.alpha = alpha

        self.s_gcn1 = GraphConvolution(in_features, hidden[0], dropout)
        self.s_gcn2 = GraphConvolution(hidden[0], hidden[1], dropout)

        self.f_gcn1 = GraphConvolution(in_features, hidden[0], dropout)
        self.f_gcn2 = GraphConvolution(hidden[0], hidden[1], dropout)

        self.genc = GraphEncoder(in_features, hidden.copy(), order, dropout=dropout)
        self.mix_layer = GraphConvolution(hidden[-1], hidden[-1], dropout)
        self.s_layer = GraphConvolution(hidden[-1], hidden[-1], dropout)
        # self.encoder=GraphConvolution(in_features,hidden[-1],bias=False)
        # self.decoder = MultiDecoder(out_features, n_view)
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

        self.fc = nn.Linear(hidden[-1], hidden[-1])
        self.fc1 = nn.Linear(hidden[-1], hidden[-1])

        self.gamma = Parameter(torch.zeros(1))
        self.attn = Attention(hidden[-1])

        # self.clu_layer = ClusteringLayer(out_features, n_cluster)
        self.cluster_layer = Parameter(torch.Tensor(n_cluster, hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def scale(self, z):
        pass

        # zmax = z.max(dim=1, keepdim=True)[0]
        # zmin = z.min(dim=1, keepdim=True)[0]
        # z_std = (z - zmin) / (zmax - zmin)
        # z_scaled = z_std
        #
        # return z_scaled

    def forward(self, features, adjs, input_view, s_mat=None):
        print('input view:', input_view)

        # self.gz = self.genc(features, adjs)
        h1 = self.s_gcn1(features, adjs[input_view])
        h2 = self.s_gcn2(h1, adjs[input_view])
        self.embeddings = h2

        # dist = torch.sum(torch.pow(self.embeddings.unsqueeze(1) - self.cluster_layer, 2), 2)
        # q = 1.0 / (1.0 + dist)
        # q = q.pow((1.0 + 1.0) / 2.0)
        # q = (q.t() / torch.sum(q, 1)).t()
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


class Model_abl_sm(Module):
    def __init__(self, in_features, hidden, out_features, n_views, order, n_cluster, alpha=0.5,
                 dropout=0., act=lambda x: x):  # out_features=hidden(len(hidden)-1)
        super(Model_abl_sm, self).__init__()

        self.is_clu = False
        self.n_views = n_views
        self.alpha = alpha

        self.s_gcn1 = GraphConvolution(in_features, hidden[0], dropout)
        self.s_gcn2 = GraphConvolution(hidden[0], hidden[1], dropout)

        self.f_gcn1 = GraphConvolution(in_features, hidden[0], dropout)
        self.f_gcn2 = GraphConvolution(hidden[0], hidden[1], dropout)

        self.genc = GraphEncoder(in_features, hidden.copy(), order, dropout=dropout)
        self.mix_layer = GraphConvolution(hidden[-1], hidden[-1], dropout)
        self.s_layer = GraphConvolution(hidden[-1], hidden[-1], dropout)
        # self.encoder=GraphConvolution(in_features,hidden[-1],bias=False)
        # self.decoder = MultiDecoder(out_features, n_view)
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

        self.fc = nn.Linear(hidden[-1], hidden[-1])
        self.fc1 = nn.Linear(hidden[-1], hidden[-1])

        self.gamma = Parameter(torch.zeros(1))
        self.attn = Attention(hidden[-1])

        # self.clu_layer = ClusteringLayer(out_features, n_cluster)
        self.cluster_layer = Parameter(torch.Tensor(n_cluster, hidden[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, features, adjs, input_view, s_mat=None):
        print('input view:', input_view)

        self.fz = self.enc(features)

        # self.embeddings = torch.mm(adjs[input_view], self.fz)
        self.embeddings = torch.mm(s_mat, self.fz)

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


class GraphEncoderSingle(GraphEncoder):

    def forward(self, input, adj):
        x = input
        for i in range(self.n_layers):
            x = self.encoders[i](x, adj)

        return x


class GraphEncoder_1(Module):
    def __init__(self, in_features, hidden, order, n_views, dropout=0., act=F.relu):
        super(GraphEncoder_1, self).__init__()
        self.n_views = n_views
        self.order = order
        self.dropout = dropout
        self.act = act
        self.n_layers = len(hidden)
        self.encoders = nn.ModuleList()
        hidden.insert(0, in_features)

        for i in range(n_views):
            self.encoders.append(GraphConvolution(hidden[0], hidden[1], dropout))

        self.encoders.append(GraphConvolution(hidden[1], hidden[2], dropout))

    def forward(self, input, adjs):
        self.fst_hidden = []
        for v in range(self.n_views):
            self.fst_hidden.append(self.encoders[v](input, adjs[v]).unsqueeze(0))
        self.fst_hidden = torch.cat(self.fst_hidden, 0)
        mean = self.fst_hidden.mean(0)
        ret = self.encoders[-1](mean, adjs[0])

        return ret


class Model_1(Module):
    def __init__(self, in_features, hidden, n_views, order, n_clusters,
                 dropout=0., act=lambda x: x):
        super(Model_1, self).__init__()
        self.is_clu = False
        self.n_views = n_views

        self.encoder = GraphEncoder_1(in_features, hidden, order, n_views)
        self.decoder = nn.ModuleList()
        for v in range(n_views):
            self.decoder.append(InnerProductDecoder(in_features=hidden[-1], act=act))

        self.clu_layer = ClusteringLayer(hidden[-1], n_clusters)

    def forward(self, input, adjs):

        # fs=torch.mm(input,input.t())
        # fs=F.softmax(fs,dim=1)
        # f_embeddings=torch.mm(fs,)

        self.embeddings = self.encoder(input, adjs)
        reconstructed_views = []
        for v in range(self.n_views):
            reconstructed_view = self.decoder[v](self.embeddings)
            reconstructed_views.append(reconstructed_view)
        # s=torch.mm(z_i,z_i.t())
        # s=F.softmax(s,dim=1)
        # z_g = torch.mm(s,z_i)

        if self.is_clu:
            q = self.clu_layer(self.embeddings)
        else:
            q = None
        return self.embeddings, reconstructed_views, q


class Model_2(Module):
    def __init__(self, in_features, hidden, n_views, order, n_clusters,
                 dropout=0., act=lambda x: x):
        super(Model_2, self).__init__()
        self.is_clu = False
        self.n_views = n_views
        self.encoder = nn.ModuleList()
        hidden.insert(0, in_features)

        for i in range(len(hidden) - 1):
            self.encoder.append(GraphConvolution(hidden[i], hidden[i + 1], dropout))
        self.decoder = nn.ModuleList()
        for v in range(n_views):
            self.decoder.append(InnerProductDecoder(in_features=hidden[-1], act=act))

        self.clu_layer = ClusteringLayer(hidden[-1], n_clusters)

    def forward(self, input, adjs):
        x = input
        for l in range(2):
            embeddings = []
            for adj in adjs:
                embeddings.append(self.encoder[l](x, adjs[0]).unsqueeze(0))
            x = torch.cat(embeddings, 0).mean(0)
        self.embeddings = x
        reconstructed_views = []
        for v in range(self.n_views):
            reconstructed_view = self.decoder[v](self.embeddings)
            reconstructed_views.append(reconstructed_view)
        if self.is_clu:
            q = self.clu_layer(self.embeddings)
        else:
            q = None
        return self.embeddings, reconstructed_views, q


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
