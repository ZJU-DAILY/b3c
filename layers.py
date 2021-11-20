import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        #
        # if not self.active:
        #     self.act = lambda x: x

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        # if self.bias is not None:
        #     torch.nn.init.xavier_uniform_(self.bias)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, active=True):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias

        if active:
            return self.act(output)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(Module):

    def __init__(self, in_features, dropout=0., act=F.sigmoid):  # todo:use sigmoid or not
        super(InnerProductDecoder, self).__init__()

        self.in_features = in_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.data)

    def forward(self, input):
        input = F.dropout(input, self.dropout, self.training)
        tmp = torch.mm(input, self.weight)
        recon = torch.mm(tmp, input.t())

        return self.act(recon)


class ClusteringLayer(Module):

    def __init__(self, in_features, n_clusters=3, weights=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.in_features = in_features
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.cluster_centroids = Parameter(torch.FloatTensor(self.n_clusters, self.in_features))
        torch.nn.init.xavier_normal_(self.cluster_centroids.data)

    def forward(self, input):
        q = 1.0 / (1.0 + torch.sum(torch.pow(input.unsqueeze(1) - self.cluster_centroids, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q
