from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, split_mask, split_arti, print_evaluation_metrics
from GCN.pytorch.models import GCN
from load_data import load_data_cora, load_data_blog

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--im_ratio', type=float, default=0.5)
parser.add_argument('--setting', type=str, default='no',
                    choices=['no', 'active_learning', 'cost_sensitive', 'pseudo_active_learning'])
parser.add_argument('--up_scale', type=float, default=1)
# 主动学习的轮数
parser.add_argument('--rounds', type=int, default=10)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
if args.dataset == 'cora':
    adj, features, labels = load_data_cora()
    im_class_num = 3
    class_sample_num = 10
    # for artificial imbalanced setting: only the last im_class_num classes are imbalanced
    c_train_num = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - im_class_num:  # only imbalance the last classes
            c_train_num.append(int(class_sample_num * args.im_ratio))
        else:
            c_train_num.append(class_sample_num)
    train_mask, val_mask, test_mask, candidate_mask, c_num_mat, imbalance_ratio = split_arti(labels, c_train_num)
elif args.dataset == 'blog':
    im_class_num = 14
    adj, features, labels = load_data_blog()
    train_mask, val_mask, test_mask, candidate_mask, c_num_mat, imbalance_ratio = split_mask(labels)
else:
    print("no this dataset: {args.dataset}")
    exit()
t = time.time()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)


if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()
    candidate_mask = candidate_mask.cuda()


def train(epoch):
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # 使用cost sensitive的函数
    if args.setting == 'active_learning' or 'cost_sensitive' or 'pseudo_active_learning':
        weight = features.new((labels.max().item() + 1)).fill_(1)
        weight[-im_class_num:] = 1+args.up_scale
        loss_train = F.cross_entropy(output[train_mask], labels[train_mask], weight=weight.float())
    # 使用常规的损失函数
    else:
        loss_train = F.nll_loss(F.log_softmax(output[train_mask]), labels[train_mask])
    acc_train = accuracy(output[train_mask], labels[train_mask])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[val_mask], labels[val_mask])
    acc_val = accuracy(output[val_mask], labels[val_mask])
    if((epoch+1)%50 == 0):
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    # 以cora为例，output是2708*7的tensor
    output = model(features, adj)
    loss_test = F.nll_loss(output[test_mask], labels[test_mask])
    acc_test = accuracy(output[test_mask], labels[test_mask])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print_evaluation_metrics(output, labels, 'test')


# 训练GCN，args.epochs次
def train_epochs():
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    test()


# 一轮主动学习 主动学习本质上来说就是改变了训练集和候选集
def active_learning(round, train_mask, candidate_mask):
    print("第{:02d}轮主动学习".format(round+1))
    model.apply(weight_reset)
    train_epochs()
    # 此时有了一个训练好的model
    model.eval()
    # 以cora为例，output是2708*7的tensor
    output = model(features, adj)
    output = F.softmax(output, dim=1).tolist()
    for i in range(len(output)):
        output[i].append(i)
    output = np.array(output)

    # 把train_mask和candidate_mask转化成list，方便操作
    train_mask_temp = train_mask.tolist()
    candidate_mask_temp = candidate_mask.tolist()
    for i in range(c_num_mat.shape[0]):
        output_candidate = torch.from_numpy(output)[candidate_mask_temp]
        output_candidate = output_candidate.tolist()
        # temp用来储存排好序的output
        temp = sorted(output_candidate, key=(lambda x:x[i]), reverse=True)

        # print(len(output_candidate))
        # print(len(candidate_mask_temp))
        # 计算对该类取几个放入训练集
        # 注意！！ (1-imbalance_ratio)*5*rounds < len(candidate_mask)
        for j in range(int((1-imbalance_ratio[i])*5)):
            train_mask_temp.append(temp[j][-1])
            candidate_mask_temp.remove(temp[j][-1])
        # 如何从张量中找到某个元素并删除?
    train_mask_temp = np.array(train_mask_temp)
    train_mask_temp = torch.from_numpy(train_mask_temp).long()
    candidate_mask_temp = np.array(candidate_mask_temp)
    candidate_mask_temp = torch.from_numpy(candidate_mask_temp).long()
    print(len(train_mask_temp))
    return train_mask_temp, candidate_mask_temp


    # 得到预测结果
    # 对候选集中的结果排序



# 重置网络的参数
def weight_reset(m):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


if __name__ == '__main__':
    t_total = time.time()
    if args.setting == 'no' or args.setting == 'cost_sensitive':
        train_epochs()
    elif args.setting == 'active_learning':
        for i in range(args.rounds):
            train_mask, candidate_mask = active_learning(i, train_mask, candidate_mask)
    elif args.setting == 'pseudo_active_learning':
        for i in range(args.rounds):
            active_learning(i)
    else:
        print("no this setting: {args.setting}")
        exit()
