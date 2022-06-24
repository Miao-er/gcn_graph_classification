from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy,build_graph
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, #默认不使用cuda
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, #默认使用验证集
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #随机种子
parser.add_argument('--epochs', type=int, default=10,  #训练epoch
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, #学习率
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, #权重衰减
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, #隐藏层单元（维度）
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, 
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='batch size.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("using GPU" if args.cuda else "using CPU")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
train_loader,test_loader = load_data(batch_size = args.batch_size)
# Model and optimizer
model = GCN(nfeat=3,
            nhid=args.hidden,
            nclass=40,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()


def train(epoch):
    iter = 0
    all_loss = []
    all_acc = []
    print('Epoch: {:04d}'.format(epoch+1))
    for data,labels in train_loader:
        t = time.time()
        model.train()
        optimizer.zero_grad()
        adj_matrix = build_graph(data)
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
            adj_matrix = adj_matrix.cuda()
        output = model(data, adj_matrix)
        loss_train = F.nll_loss(output, labels)
        acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()

        all_acc.append(acc_train)
        all_loss.append(loss_train)
        iter += 1
        if iter % 50 == 49:
            print('iter:{:04d}'.format(iter + 1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train.item()),
                'time: {:.4f}s'.format(time.time() - t))
    print("this epoch:\n",
        "loss= {:.4f}".format(torch.tensor(all_loss).mean().item()),
        "accuracy= {:.4f}".format(torch.tensor(all_acc).mean().item()))


def test():
    model.eval()
    all_acc = []
    all_loss = []
    for data,labels in train_loader:
        t = time.time()
        adj_matrix = build_graph(data)
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
            adj_matrix = adj_matrix.cuda()
        output = model(data, adj_matrix) #batch_size * nclass
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        all_acc.append(acc_test)
        all_loss.append(loss_test)

    print("Test set results:",
          "loss= {:.4f}".format(torch.tensor(all_loss).mean().item()),
          "accuracy= {:.4f}".format(torch.tensor(all_acc).mean().item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
