from __future__ import print_function, division
import torch
import argparse
from sklearn.decomposition import PCA
from constructor import format_data
from utils import *
from models import *
from train import train
import scipy.io as sio
from tqdm import tqdm
from evaluation import eva, plotClusters
from train_cc import train_cc


class load_data(Dataset):
    def __init__(self, features, labels):
        self.x = features  # features
        self.y = labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))


def to_torch_sparse_tensor(matrices):
    ret = []
    for mat in matrices:
        ret.append(sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(mat)))
    return ret


def load_bipart(dataset, key):
    return sio.loadmat('gtn/acm_w_alabels.mat')[key]


def load_type(dataset, ntype, input_view, cpr=True):
    all_data = format_data(dataset, ntype)
    views = to_torch_sparse_tensor(all_data['adjs_norm'])
    s_mat = np.matmul(all_data['features'], all_data['features'].T)
    # for stability
    if cpr:
        pca = PCA(n_components=100)
        all_data['features'] = pca.fit_transform(all_data['features'])

    data = load_data(all_data['features'], all_data['true_labels'])
    model = B3C(in_features=data.x.shape[1], hidden=args.hidden, out_features=args.out,
                n_views=all_data['numView'], n_cluster=args.n_clusters, dropout=args.dropout).to(device)

    if ntype == 'P':
        model.load_state_dict(torch.load('pretrain_acmp'))
    elif ntype == 'A':
        model.load_state_dict(torch.load('pretrain_acma'))
    model.to(device)

    ret = {'model': model, 'dataset': data, 'views': views, 'adj_labels': all_data['adjs_label'],
           'pos_weight': all_data['pos_weights'], 'input_view': input_view, 's_mat': s_mat}
    return ret


def preest():
    paper = load_type(args.dataset, 'P', 0)
    author = load_type(args.dataset, 'A', 0)
    pa = load_bipart(args.dataset, 'PA')
    cc = CC().to(device)

    return paper, author, pa, cc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ACM')
    parser.add_argument('--order', nargs='+', type=int, default=[0, 0])
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--kl_epochs', type=int, default=500)
    parser.add_argument('--kl_factor', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--hidden', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--input_view', type=int, default=0)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    args.dataset = 'ACM'

    #
    # for l in lamd:
    #     for b in beta:
    #         for g in gamma:
    #             for i in range(5):
    # for i in range(5):
    #     paper, author, pa, cc = preest()
    #     train_cc(paper, author, cc, kl_epochs=args.kl_epochs, lr=args.lr, n_clusters=args.n_clusters, device=device,
    #              bipart=pa, lamb=10, beta=0, gamma=0)
    #
    # for i in range(5):
    #     paper, author, pa, cc = preest()
    #     train_cc(paper, author, cc, kl_epochs=args.kl_epochs, lr=args.lr, n_clusters=args.n_clusters, device=device,
    #              bipart=pa, lamb=10, beta=1, gamma=0)
    #
    # for i in range(5):
    #     paper, author, pa, cc = preest()
    #     train_cc(paper, author, cc, kl_epochs=args.kl_epochs, lr=args.lr, n_clusters=args.n_clusters, device=device,
    #              bipart=pa, lamb=10, beta=10, gamma=0)
    #
    # for b in [1, 10]:
    #     for g in [0.1, 1, 10]:
    #         for i in range(5):
    #             paper, author, pa, cc = preest()
    #             train_cc(paper, author, cc, kl_epochs=args.kl_epochs, lr=args.lr, n_clusters=args.n_clusters,
    #                      device=device,
    #                      bipart=pa, lamb=10, beta=b, gamma=g)

    lamd = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  # 10
    beta = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]  #
    gamma = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    for l in lamd:
        for i in range(5):
            paper, author, pa, cc = preest()
            train_cc(paper, author, cc, kl_epochs=args.kl_epochs, lr=args.lr, n_clusters=args.n_clusters,
                     device=device, bipart=pa, lamb=l, beta=0, gamma=0)

    for b in beta:
        for g in [0.1, 1, 10]:
            for i in range(5):
                paper, author, pa, cc = preest()
                train_cc(paper, author, cc, kl_epochs=args.kl_epochs, lr=args.lr, n_clusters=args.n_clusters,
                         device=device, bipart=pa, lamb=10, beta=b, gamma=g)

    for g in gamma:
        for b in [1, 10]:
            for i in range(5):
                paper, author, pa, cc = preest()
                train_cc(paper, author, cc, kl_epochs=args.kl_epochs, lr=args.lr, n_clusters=args.n_clusters,
                         device=device, bipart=pa, lamb=10, beta=b, gamma=g)
