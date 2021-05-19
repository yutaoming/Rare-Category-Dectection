import scipy.sparse as sp
from scipy.io import loadmat
import numpy as np
import torch

IMBALANCE_THRESHOLD = 101


def load_data_blog():
    mat = loadmat('data/BlogCatalog/blogcatalog.mat')
    adj = mat['network']
    label = mat['group']

    embedding = np.loadtxt('data/BlogCatalog/blogcatalog.embeddings_64')
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
    print("改变labels的顺序,稀有类别在最后")
    labels = refine_label_order(labels)

    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return features, labels, adj


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
    load_data_blog()
