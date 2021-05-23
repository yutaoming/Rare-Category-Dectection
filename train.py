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
    class_sample_num = 20
    # for artificial imbalanced setting: only the last im_class_num classes are imbalanced
    c_train_num = []
    for i in range(labels.max().item() + 1):
        if i > labels.max().item() - im_class_num:  # only imbalance the last classes
            c_train_num.append(int(class_sample_num * args.im_ratio))
        else:
            c_train_num.append(class_sample_num)
    train_mask, val_mask, test_mask, c_num_mat = split_arti(labels, c_train_num)
elif args.dataset == 'blog':
    im_class_num = 14
    adj, features, labels = load_data_blog()
    train_mask, val_mask, test_mask, c_num_mat = split_mask(labels)
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


def train(epoch):
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    if args.setting == 'active_learning' or 'cost_sensitive' or 'pseudo_active_learning':
        weight = features.new((labels.max().item() + 1)).fill_(1)
        weight[-im_class_num:] = 1+args.up_scale
        loss_train = F.cross_entropy(output[train_mask], labels[train_mask], weight=weight.float())
    else:
        loss_train = F.nll_loss(output[train_mask], labels[train_mask])
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


if __name__ == '__main__':
    t_total = time.time()
    if args.setting == 'no' or 'cost_sensitive':
        # Train model
        for epoch in range(args.epochs):
            train(epoch)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        # Testing
        test()
    elif args.setting == 'active_learning':
        print()
    elif args.setting == 'pseudo_active_learning':
        print()
    else:
        print("no this setting: {args.setting}")
        exit()
