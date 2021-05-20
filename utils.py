from load_data import load_data_blog, load_data_cora
import numpy as np
import torch
import random
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn.functional as F

# cora 用
# c_train_num是一个数组，用于记录每个类有多少个节点用于训练
def split_arti(labels, c_train_num):
    # labels: n-dim Longtensor, each element in [0,...,m-1].
    # cora: m=7
    num_classes = len(set(labels.tolist()))
    class_masks = []  # class-wise index
    train_mask = []
    val_mask = []
    test_mask = []
    c_num_mat = np.zeros((num_classes, 3)).astype(int)
    # 每个类有25个节点用于验证集
    c_num_mat[:, 1] = 25
    # 每个类用55个节点用于测试集
    c_num_mat[:, 2] = 55

    for i in range(num_classes):
        # 得到i的索引
        class_mask = (labels == i).nonzero()[:, -1].tolist()
        # print('{:d}-th class sample number: {:d}'.format(i,len(class_mask)))
        random.shuffle(class_mask)
        class_masks.append(class_mask)

        train_mask = train_mask + class_mask[:c_train_num[i]]
        c_num_mat[i, 0] = c_train_num[i]

        val_mask = val_mask + class_mask[c_train_num[i]:c_train_num[i]+25]
        test_mask = test_mask + class_mask[c_train_num[i]+25:c_train_num[i]+80]

    random.shuffle(train_mask)

    # list -> numpy -> tensor
    train_mask = np.array(train_mask)
    train_mask = torch.from_numpy(train_mask)

    val_mask = np.array(val_mask)
    val_mask = torch.from_numpy(val_mask)

    test_mask = np.array(test_mask)
    test_mask = torch.from_numpy(test_mask)

    c_num_mat = np.array(c_num_mat)
    c_num_mat = torch.from_numpy(c_num_mat)
    # c_num_mat = torch.LongTensor(c_num_mat)
    
    return train_mask, val_mask, test_mask, c_num_mat


# 如何随机生成 训练集 测试集？
# 对所有类做如下操作：
# 先获得当前类的
# 只要把数组随机排序，然后按比例切片即可
# for blog
def split_mask(labels):
    """用于生成trian_mask, val_mask, test_mask"""
    # labels: n-dim Longtensor, each element in [0,...,m-1].
    num_classes = len(set(labels.tolist()))
    # num_classes = 38
    # 对应class的索引
    class_masks = []
    train_mask = []
    val_mask = []
    test_mask = []
    # 生成一个num_classes * 3 的ndarray
    # 用来记录各个类 train val test的数量 0.25, 0.25, 0.5
    c_num_mat = np.zeros((num_classes, 3)).astype(int)
    # 0是false 0以外是true
    for i in range(num_classes):
        # class_mask是某个类的索引
        class_mask = (labels == i).nonzero()[:, -1].tolist()
        class_num = len(class_mask)
        # print('{:d}-th class sample number: {:d}'.format(i,len(class_mask)))
        # shuffle 把一个数组打乱重新排序 就像是洗牌
        random.shuffle(class_mask)
        class_masks.append(class_mask)

        if class_num < 4:
            if class_num < 3:
                print("too small class type")
                # 一般不会执行到这步，除非某个类的数量小于3
                ipdb.set_trace()
            c_num_mat[i, 0] = 1
            c_num_mat[i, 1] = 1
            c_num_mat[i, 2] = 1
        else:
            c_num_mat[i, 0] = int(class_num / 4)
            c_num_mat[i, 1] = int(class_num / 4)
            c_num_mat[i, 2] = int(class_num / 2)

        train_mask += class_mask[:c_num_mat[i, 0]]
        val_mask += class_mask[c_num_mat[i, 0]:c_num_mat[i, 0] + c_num_mat[i, 1]]
        test_mask += class_mask[c_num_mat[i, 0] + c_num_mat[i, 1]:c_num_mat[i, 0] + c_num_mat[i, 1] + c_num_mat[i, 2]]

    # 避免出现相同的类连在一起的情况
    random.shuffle(train_mask)
    # list -> numpy -> tensor
    train_mask = np.array(train_mask)
    train_mask = torch.from_numpy(train_mask)

    val_mask = np.array(val_mask)
    val_mask = torch.from_numpy(val_mask)

    test_mask = np.array(test_mask)
    test_mask = torch.from_numpy(test_mask)

    c_num_mat = np.array(c_num_mat)
    c_num_mat = torch.from_numpy(c_num_mat)
    # 只有train_mask顺序是乱的，val_mask，test_mask会按照类别的顺序
    return train_mask, val_mask, test_mask, c_num_mat


# evaluation function
# 一共三个指标
# 第一个是 ACC 计算测试集的accuracy 因为稀有类只是少数，所以不够准确
# 第二个是 AUC-ROC 对每个类都求 然后取平均值
# 第三个是 F1 综合了precision和recall 同样是对每个类都求，然后取平均
def print_evaluation_metrics(output, labels, pre='valid'):
    # class_num_list: 一个记录类数目的列表
    pre_num = 0
    # print class-wise performance
    # 如果添加以下代码，请把class_num_list添加到函数参数列表中
    # for i in range(labels.max()+1):
    #     # 如果labels[mask]，那么label则会按照索引的顺序来 而不是labels自带的顺序
    #     cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
    #     print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i, cur_tpr.item()))
    #
    #     index_negative = labels != i
    #     # 生成一个全是i的labels
    #     labels_negative = labels.new(labels.shape).fill_(i)
    #     # output[index_negative, :] 预测不是i类的
    #     cur_fpr = accuracy(output[index_negative, :], labels_negative[index_negative])
    #     print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i, cur_fpr.item()))
    #
    #     pre_num = pre_num + class_num_list[i]

    # ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1).detach(), average='macro',
                                  multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach(), F.softmax(output, dim=-1)[:, 1].detach(), average='macro')

    macro_F = f1_score(labels.detach(), torch.argmax(output, dim=-1).detach(), average='macro')
    print(str(pre) + ' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score, macro_F))

    return


def accuracy(output, labels):
    # max(1)是返回每一行最大值组成的一维数组
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def main():
    features, labels, adj = load_data_blog()
    train_mask, val_mask, test_mask, c_num_mat = split_mask(labels)


if __name__ == '__main__':
    main()
# 要做的事：1. 在cora上测试evaluation  1.5.在cora上人为制造稀有类，重复1  2.在blog上跑GCN  3.在blog上测试evaluation
# 4. 实现active learning