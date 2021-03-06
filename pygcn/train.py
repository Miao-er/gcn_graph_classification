from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import GCN,PointNet,DGCNN
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, #默认不使用cuda
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=41, help='Random seed.') #随机种子
parser.add_argument('--weight_decay', type=float, default=5e-4, #权重衰减
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,#隐藏层单元（维度）
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, 
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--epochs', type=int, default=60,  #训练epoch
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default= 0.001, #学习率
                    help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size.')
parser.add_argument('--knn_param', type=int, default=50, 
                    help='top-k nearest neighbors.')
parser.add_argument('--lr_step', type=int, default=15, 
                    help='update learning rate every n steps')
parser.add_argument('--lr_decay', type=float, default=0.1, 
                    help='decay learning rate every n steps')
                    
parser.add_argument('--test_mode', action='store_true', default=False, 
                    help='test using saved model.')
parser.add_argument('--trained', action='store_true', default=False, 
                    help='trained using saved model.')
parser.add_argument('--last_epoch', type = int, default= -1, 
                    help='checkpoint.')
parser.add_argument('--logger', type = str, default='../log/train.log', 
                    help='the training log.')
parser.add_argument('--save_model', type = str, default='../train_model/model.pt', 
                    help='save trained model path.')
parser.add_argument('--best_model', type = str, default='../best_model/model.pt', 
                    help='save best model path.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("using GPU" if args.cuda else "using CPU")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
test_loader = load_data(batch_size = args.batch_size,tag = 'test')
if not args.test_mode:
    train_loader = load_data(batch_size = args.batch_size,tag = 'train')
# Model and optimizer
model = GCN(nfeat=3,
            nhid=args.hidden,
            nclass=40,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,args.lr_step, gamma = args.lr_decay)
scheduler.last_epoch = args.last_epoch

#load logger
logger = get_logger(args.logger)

if args.cuda:
    model.cuda()

def train(epoch,best_accuracy = 0):
    iter = 0
    all_loss = []
    all_corr = 0
    data_num = 0
    logger.info('Epoch: {:04d}'.format(epoch+1))
    for data,labels in train_loader:
        t = time.time()
        model.train()
        optimizer.zero_grad()
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        adj_matrix = build_graph(data, k = args.knn_param)
        output = model(data, adj_matrix)
        loss_train = F.nll_loss(output, labels)
        correct,num = accuracy(output, labels)
        loss_train.backward() 

        optimizer.step()
        data_num += num
        all_corr += correct
        all_loss.append(loss_train)
        iter += 1
        if iter % 20 == 19:
            logger.info('iter:{:04d} '.format(iter + 1) +
                'loss_train: {:.4f} '.format(loss_train.item()) +
                'acc_train: {:.4f} '.format(correct / num) +
                'time: {:.4f}s'.format(time.time() - t))
    
    mean_acc = all_corr / data_num
    if mean_acc > best_accuracy:
        torch.save(model.state_dict(), args.best_model)
        best_accuracy = mean_acc 

    scheduler.step()
    logger.info("this epoch:\n" + 
    "loss= {:.4f} ".format(torch.tensor(all_loss).mean().item()) +
    "accuracy= {:.4f}".format(mean_acc))
    
    return  best_accuracy


def test():
    model.eval()
    all_loss = []
    all_corr = 0
    data_num = 0
    class_map = {key:[0,0] for key in range(40)}
    for data,labels in test_loader:
        t = time.time()
        if args.cuda:
            data = data.cuda()
            labels = labels.cuda()
        adj_matrix = build_graph(data,k = args.knn_param)
        output = model(data, adj_matrix) #batch_size * nclass
        loss_test = F.nll_loss(output, labels)
        correct,num = accuracy(output, labels)
        class_accuracy(class_map,output,labels)

        data_num += num
        all_corr += correct
        all_loss.append(loss_test)

    logger.info("Test set results: "+
          "loss= {:.4f} ".format(torch.tensor(all_loss).mean().item()) + 
          "accuracy= {:.4f}".format(all_corr / data_num))

    class_map = class_accuracy(class_map)
    logger.info(f'mean class accuracy:{np.mean(list(class_map.values()))}')

# Train model
if not args.test_mode:
    logger.info(f'trained model saved at :{args.save_model}')
    logger.info(f'best model saved at :{args.best_model}')
    t_total = time.time()
    best_accuracy = 0
    if args.trained:
        model.load_state_dict(torch.load(args.save_model))

#恢复断点学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = scheduler.get_lr()[0]

    for epoch in range(args.epochs):
        best_accuracy = train(epoch,best_accuracy)
        torch.save(model.state_dict(), args.save_model)
    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logger.info('the best accuarcy is {:.4f}'.format(best_accuracy))
    with torch.no_grad():
        test()
# Testing
else:
    model.load_state_dict(torch.load(args.best_model))
    logger.info(f'test model saved at :{args.best_model}')
    with torch.no_grad():
        test()
