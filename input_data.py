import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
from sklearn.decomposition import PCA


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def label_mask(labels):
    num = labels.shape[1]
    label_mask = np.dot(labels, np.array(range(num)).reshape((num, 1))).reshape((labels.shape[0]))
    return label_mask


def genSamples(data, numView):
    pos = np.where((data - numView * np.eye(data.shape[0])) > 0)
    pvalue = data[pos]

    neg = np.where(data == 0)
    nvalue = data[neg]

    return (pos, pvalue), (neg, nvalue)


def load_data(dataset, ntype):
    # load the data: x, tx, allx, graph
    # if dataset == "ACM":
    #     dataset_path = "data/ACM3025.mat"
    # elif dataset == "DBLP":
    #     dataset_path = "data/DBLP4057_GAT_with_idx.mat"
    # elif dataset == "IMDB":
    #     dataset_path="data/imdb5k.mat"

    if dataset == "ACM":
        dataset_path = "gtn/acm_w_alabels.mat"
        # dataset_path = "data/ACM3025.mat"
    elif dataset == "DBLP":
        dataset_path = "heco/dblp_w_plabels.mat"
    elif dataset == "IMDB":
        dataset_path = "magnn/imdb_w_dlabels.mat"
        # dataset_path="gtn/imdb.mat"

    # if dataset == "ACM":
    #     dataset_path = "/home/yau/hete_embedding/B3C/gtn/acm_w_alabels.mat"
    # elif dataset == "DBLP":
    #     dataset_path = "/home/yau/hete_embedding/B3C/heco/dblp_w_plabels.mat"
    # elif dataset == "IMDB":
    #     dataset_path="/home/yau/hete_embedding/B3C/magnn/imdb_w_dlabels.mat"

    data = sio.loadmat(dataset_path)

    if dataset == "ACM":  # P-A
        if ntype == 'P':
            truelabels, truefeatures = data['labels'], data['p_fea'].astype(float)
            N = truefeatures.shape[0]
            rownetworks = np.array([(data['PAP']).tolist(), (data['PSP']).tolist()])
            linkCount = data['PAP'] + data['PSP']
        elif ntype == 'A':
            truelabels, truefeatures = data['a_labels'], data['a_fea'].astype(float)
            N = truefeatures.shape[0]
            rownetworks = np.array([(data['APA']).tolist(), (data['APAPA']).tolist(), (data['APSPA']).tolist()])
            linkCount = data['APA'] + data['APAPA'] + data['APSPA']


    elif dataset == "DBLP":  # A-P

        if ntype == "A":
            truelabels, truefeatures = data['labels'], data['a_fea'].A.astype(float)
            rownetworks = np.array([(data['APA']).tolist(), (data['APCPA']).tolist(), (data['APTPA']).tolist()])
            linkCount = data['APA'] + data['APCPA'] + data['APTPA']
        elif ntype == 'P':
            truelabels, truefeatures = data['p_labels'], data['p_fea'].A.astype(float)
            rownetworks = np.array([(data['PAP']).tolist(), (data['PCP']).tolist(), (data['PTP']).tolist()])
            linkCount = data['PAP'] + data['PCP'] + data['PTP']

    elif dataset == "IMDB":  # M-D
        # truelabels, truefeatures = data['labels'], data['m_fea'].astype(float)
        # N = truefeatures.shape[0]
        # rownetworks = np.array([(data['MAM']).tolist(), (data['MDM']).tolist()])
        # linkCount = data['MAM'] + data['MDM']

        truelabels, truefeatures = data['d_labels'], data['d_fea'].astype(float)
        N = truefeatures.shape[0]
        rownetworks = np.array([(data['dmamd']).tolist()])
        linkCount = data['dmamd']

    numView = rownetworks.shape[0]

    return np.array(rownetworks), numView, truefeatures, truelabels
