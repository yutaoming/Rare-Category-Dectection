import scipy.sparse as sp
from scipy.io import loadmat
import numpy as np


def load_data_blog():
    mat = loadmat('data/BlogCatalog/blogcatalog.mat')
    adj = mat['network']
    label = mat['group']

    embedding = np.loadtxt('data/BlogCatalog/blogcatalog.embeddings_64')
    # 这里-1 是因为embedding中有一项表示节点id
    feature = np.zeros((embedding.shape[0], embedding.shape[1]-1))
    feature[embedding[:, 0].astype(int), :] = embedding[:, 1:]

    feature = normalize(feature)


def normalize(mx):
    """对一个矩阵的行做归一化处理"""
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
