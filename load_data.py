import scipy.sparse as sp
from scipy.io import loadmat
import numpy as np
import torch

IMBALANCE_THRESHOLD = 101

def load_data_cora(path="/Users/yutaoming/PycharmProjects/Rare-Category-Detection/data/cora/", dataset="cora"):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    set_labels = set(labels)
    classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}
    classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3,
                    'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}
    labels = np.array(list(map(classes_dict.get, labels)))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    print(adj)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)

    features = torch.from_numpy(np.array(features.todense()))
    labels = torch.from_numpy(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj:
    """
    tensor(indices=tensor([[   0,    0,    0,  ..., 2707, 2707, 2707],
                           [   8,   14,  258,  ...,  774, 1389, 2344]]),
    values=tensor([1., 1., 1.,  ..., 1., 1., 1.]),
    size=(2708, 2708), nnz=10556, layout=torch.sparse_coo)
    """
    return adj, features, labels


def load_data_blog():
    mat = loadmat('/Users/yutaoming/PycharmProjects/Rare-Category-Detection/data/BlogCatalog/blogcatalog.mat')
    adj = mat['network']
    label = mat['group']

    embedding = np.loadtxt('/Users/yutaoming/PycharmProjects/Rare-Category-Dectection/data/BlogCatalog/blogcatalog.embeddings_64')
    # 这里-1 是因为embedding中有一项表示节点id
    feature = np.zeros((embedding.shape[0], embedding.shape[1]-1))
    feature[embedding[:, 0].astype(int), :] = embedding[:, 1:]

    features = normalize(feature)
    # label 应该是一个稀疏矩阵的形式
    # todense 可以把稀疏矩阵转化成正常矩阵的形式
    # Reference: https://blog.csdn.net/weixin_42067234/article/details/80247194
    labels = np.array(label.todense().argmax(axis=1)).squeeze()
    # 因为labels中15直接跳到了17，所以从17开始统一 -1
    labels[labels > 16] = labels[labels > 16] - 1
    labels = refine_label_order(labels)

    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, labels


def normalize(mx):
    """对一个矩阵的行做归一化处理"""
    # 如果一行之和过小，则这一行都被视为0
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    # 如果倒数是无穷大，说明这个数本身很小，可以视作是0
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# 让稀有类的标签在多数类的后面 如果一个类的个数小于101，则被视为是稀有类
# 从0到23是多数类，从24到37是稀有类
def refine_label_order(labels):
    j = 0
    for i in range(labels.max(), 0, -1):
        if sum(labels == i) >= IMBALANCE_THRESHOLD and i > j:
            while sum(labels == j) >= IMBALANCE_THRESHOLD and i > j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    load_data_cora()
