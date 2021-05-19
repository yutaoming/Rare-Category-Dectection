from load_data import load_data_blog
import numpy as np
import torch
import random

# 如何随机生成 训练集 测试集？
# 只要把数组随机排序，然后按比例切片即可
def split_mask(labels):
    """用于生成trian_mask, val_mask, test_mask"""
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    num_classes = len(set(labels.tolist()))
    class_masks = [] # class-wise index
    train_mask = []
    val_mask = []
    test_mask = []
    # 生成一个num_classes * 3 的ndarray
    # 用来记录各个类 train val test的数量 0.25, 0.25, 0.5
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    # 0是false 0以外是true
    for i in range(num_classes):
        class_mask = (labels==i).nonzero()[:,-1].tolist()
        class_num = len(class_mask)
        # print('{:d}-th class sample number: {:d}'.format(i,len(class_mask)))
        # shuffle 吧一个数组打乱重新排序 就像是洗牌
        random.shuffle(class_mask)
        class_masks.append(class_mask)

        if class_num <4:
            if class_num < 3:
                print("too small class type")
                # 一般不会执行到这步，除非某个类的数量小于3
                ipdb.set_trace()
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(class_num/4)
            c_num_mat[i,1] = int(class_num/4)
            c_num_mat[i,2] = int(class_num/2)


        train_mask = train_mask + class_mask[:c_num_mat[i,0]]
        val_mask = val_mask + class_mask[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_mask = test_mask + class_mask[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

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

    return train_mask, val_mask, test_mask, c_num_mat

if __name__ == '__main__':
    features, labels, adj = load_data_blog()
    split_mask(labels)