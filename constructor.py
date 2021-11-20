import numpy as np
import scipy.sparse as sp
from input_data import load_data
from preprocessing import preprocess_graph, preprocess_graph_row, sparse_to_tuple, mask_test_edges, construct_feed_dict
from networkx.algorithms.community.quality import modularity
import networkx as nx


def format_data(data_name,ntype='P'):
    print('Loading data...')
    rownetworks, numView, features, truelabels = load_data(data_name,ntype)

    adjs_orig = []
    for v in range(numView):
        adj_orig = rownetworks[v]
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]),
                                            shape=adj_orig.shape)  # 去self loop
        # adj_orig.eliminate_zeros()
        adjs_orig.append(adj_orig)
    adjs_label = rownetworks

    adjs_orig = np.array(adjs_orig)
    adjs = adjs_orig
    # if FLAGS.features == 0:
    #     features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adjs_norm = preprocess_graph(adjs)  # D(-1/2) * A * D(-1/2)
    # adjs_norm = preprocess_graph_row(adjs)  # D(-1/2) * A * D(-1/2)

    # weighted_adj = np.zeros(adjs[0].shape)
    # for adj in adjs:
    #     weighted_adj += adj

    # weighted_adj_norm = preprocess_graph(weighted_adj[np.newaxis,:,:])
    # print('weighted adj norm:',weighted_adj_norm.shape)

    num_nodes = adjs[0].shape[0]

    features = features
    num_features = features.shape[1]
    # features_nonzero = features[1].shape[0]
    fea_pos_weights = float(features.shape[0] * features.shape[1] - features.sum()) / features.sum()
    pos_weights = []
    norms = []
    for v in range(numView):
        pos_weight = float(adjs[v].shape[0] * adjs[v].shape[0] - adjs[v].sum()) / adjs[v].sum()
        norm = adjs[v].shape[0] * adjs[v].shape[0] / float((adjs[v].shape[0] * adjs[v].shape[0] - adjs[v].sum()) * 2)
        pos_weights.append(pos_weight)
        norms.append(norm)
    true_labels = truelabels

    feas = {'adjs': adjs_norm, 'adjs_label': adjs_label, 'num_features': num_features, 'num_nodes': num_nodes,
            'true_labels': true_labels, 'pos_weights': pos_weights, 'norms': np.array(norms), 'adjs_norm': adjs_norm,
            'features': features, 'fea_pos_weights': fea_pos_weights, 'numView': numView}
    return feas
