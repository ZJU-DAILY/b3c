import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from evaluation import eva
from pytorchtools import EarlyStopping


def scale(z):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / (zmax - zmin)
    z_scaled = z_std

    return z_scaled


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def process(data, device, n_clusters):
    data['s_mat'] = torch.Tensor(data['s_mat']).to(device)
    data['s_mat'] = F.softmax(data['s_mat'], dim=1)
    data['dataset'].x = torch.Tensor(data['dataset'].x).to(device)
    data['dataset'].y = np.argmax(data['dataset'].y, axis=1)
    data['views'] = torch.stack(data['views'], dim=0).to(device)
    adjs = []
    for al in data['adj_labels']:
        adjs.append(torch.Tensor(al))
    adjs = torch.stack(adjs, dim=0).to(device)
    data['adj_labels'] = adjs
    data['pos_weight'] = torch.Tensor(data['pos_weight']).unsqueeze(-1).to(device)

    model = data['model']
    model.train()
    embeddings, _, _, _, _, _, _ = model(data['dataset'].x, data['views'], data['input_view'], data['s_mat'])
    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # acc, nmi, ari, f1 = eva(data['dataset'].y, kmeans.labels_, 0)


def train_single(data, cur_epoch, n_clusters, lamb=10.0, beta=0.01):
    model = data['model']

    triplet_loss = nn.TripletMarginLoss(margin=1, p=2)
    if cur_epoch % 5 == 0:
        # update interval
        _, _, _, data['q'], _, _, _ = model(data['dataset'].x, data['views'], data['input_view'], data['s_mat'])
        data['p'] = target_distribution(data['q'].data)

    embeddings, reconstructed_views, x_bar, data['q'], top1, top2, centroids = model(data['dataset'].x,
                                                                                     data['views'],
                                                                                     data['input_view'],
                                                                                     data['s_mat'])

    kmeans = KMeans(n_clusters=n_clusters).fit(embeddings.data.cpu().numpy())
    acc, nmi, ari, f1 = eva(data['dataset'].y, kmeans.labels_, cur_epoch)

    tqdm.write("Epoch: {}, acc={:.5f}, nmi={:.5f}, ari={:.5f}, f1={:.5f}".format(cur_epoch + 1, acc, nmi, ari, f1))

    adj_loss = 0.
    for v in range(len(data['views'])):
        adj_loss += F.binary_cross_entropy_with_logits(reconstructed_views[v].reshape([-1]),
                                                       data['adj_labels'][v].reshape(([-1])),
                                                       pos_weight=data['pos_weight'][v])

    centroids_norm = scale(centroids)
    centroids_norm = F.normalize(centroids_norm)
    embeddings_norm = scale(embeddings)
    embeddings_norm = F.normalize(embeddings_norm)
    tc_loss = triplet_loss(embeddings_norm, centroids_norm[top1], centroids[top2])

    fea_loss = F.mse_loss(x_bar, data['dataset'].x)
    kl_loss = F.kl_div(data['q'].log(), data['p'], reduction='batchmean')
    loss = adj_loss + fea_loss + lamb * kl_loss + beta * tc_loss

    return loss, embeddings, acc, f1, nmi, ari


def train_cc(data1, data2, cc, kl_epochs, lr, n_clusters, device, bipart=None, lamb=10.0, beta=1, gamma=0.1):

    acc_1 = nmi_1 = ari_1 = f1_1 = 0.0
    best_acc_1 = best_nmi_1 = best_ari_1 = best_f1_1 = 0.0
    acc_2 = nmi_2 = ari_2 = f1_2 = 0.0
    best_acc_2 = best_nmi_2 = best_ari_2 = best_f1_2 = 0.0
    best_epoch1=best_epoch2=0

    process(data1, device, n_clusters)
    process(data2, device, n_clusters)
    model1 = data1['model']
    model2 = data2['model']
    bipart = torch.Tensor(bipart).to(device)

    optimizer = Adam([
        {'params': model1.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4, },
        {'params': model2.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4, },
        {'params': cc.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4, },
    ])

    for e in range(kl_epochs):
        optimizer.zero_grad()
        loss1, emb1, acc_1, f1_1, nmi_1, ari_1 = train_single(data1, e, n_clusters, lamb, beta)
        loss2, emb2, acc_2, f1_2, nmi_2, ari_2 = train_single(data2, e, n_clusters, lamb, beta)

        if acc_1 > best_acc_1:
            best_acc_1 = acc_1
            best_f1_1 = f1_1
            best_nmi_1 = nmi_1
            best_ari_1 = ari_1
            best_epoch1=e

        if acc_2 > best_acc_2:
            best_acc_2 = acc_2
            best_f1_2 = f1_2
            best_nmi_2 = nmi_2
            best_ari_2 = ari_2
            best_epoch2=e

        re_bipart = cc(emb1, emb2)
        cc_loss = F.binary_cross_entropy_with_logits(re_bipart.reshape([-1]),
                                                     bipart.reshape(([-1])))

        loss = loss1 + loss2 + gamma * cc_loss
        loss.backward()
        optimizer.step()

