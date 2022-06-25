import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self,in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight) #batch_szie * 2048 * feature_dim
        output = torch.bmm(adj, support) #batch_size * 2048 * 2048
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphPooling(Module):
    def __init__(self,in_features,ratio):
        super(GraphPooling,self).__init__()
        self.ratio = ratio
        self.attn = GraphConvolution(in_features,1)
    
    def forward(self,x,adj): 
        '''
        x:batch_size * num_points * in_features
        adj:batch_szie * num_points * num_points
        '''
        num_points = x.shape[1]
        select_points = int(self.ratio * num_points)
        if self.ratio == 0:
            select_points = 1
        attn = F.softmax(self.attn(x,adj).squeeze(-1),dim = 1) #batch * num_points * 1 ->batch * num_points
        _,idx = attn.topk(select_points ,dim = 1)
        x,adj = self.coarsen(x,adj,idx) #batch_size * (ratio * num_points) * in_features
        return x,adj

    @staticmethod
    def coarsen(x,adj,idx):
        batch_size = x.shape[0]
        seq = torch.arange(batch_size)
        x = x[seq,idx.T].transpose(0,1)  #根据idx取子图节点
        adj = adj[seq,idx.T].transpose(0,1) #根据idx取子图邻接矩阵的行
        adj = adj[seq,:,idx.T].transpose(0,1) #根据idx取子图邻接矩阵的列
        return x.contiguous(),adj

class DiffPooling(Module):
    def __init__(self,in_feature,out_feature,in_num,ratio = 0.25):
        super(Attention,self).__init__()
        self.ratio = ratio
        self.in_num = in_num
        self.out_feature = out_feature
        self.out_num = int(ratio * in_num) if ratio > 0 else 1
        self.attn = GraphConvolution(in_feature,self.out_num) # in_num * in_feature -> in_num * out_num
        self.conv = GraphConvolution(in_feature,out_feature) #batch_size * in_num_points * out_feature
    
    def forward(x,adj):
        '''
        x:batch_size * num_points * in_features
        adj:batch_szie * num_points * num_points
        '''
        select_mtx = self.attn(x,adj) #batch_size * in_num_points * out_num_points
        x_transform = self.conv(x,adj) #batch_size * in_num_points * out_features

        x_pool = torch.bmm(select_mtx.transpose(2,1),x_transform)
        adj_pool = torch.bmm(torch.bmm(select_mtx.transpose(2,1),adj),select_mtx)

        return x_pool,adj_pool

class DiffModel(Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = DiffPooling(nfeat, nhid,in_num = 2048,ratio = 0.5)
        self.gc2 = DiffPooling(nhid, nhid,in_num = 1024,ratio = 0.5)
        self.gc3 = DiffPooling(nhid, nhid,in_num = 512,ratio = 0.25)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)

        self.fc1 = nn.Linear(nhid,128)
        self.fc2 = nn.Linear(128,nclass)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x, adj):
        x1,adj = self.gc1(x,adj)
        x1 = F.relu(self.bn1(x1.transpose(2,1))) # batch_size * 1024 * nhid
        x1 = x1.transpose(2,1)

        x2,adj = self.gc2(x1.adj)
        x2 = F.relu(self.bn2(x2.transpose(2,1))) #batch_size * 512 * nhid
        x2 = x2.transpose(2,1)

        x3,adj = self.gc3(x2,adj)
        x3 = F.relu(self.bn3(x3.transpose(2,1))) #batch_size * 128 * nhid
        x3 = x3.transpose(2,1)

        x = self.readout(x1) + self.read(x2) + self.readout(x3) #batch_size 

        x = F.relu(self.bn4(self.fc1(x))) #batch_size * 128
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    @staticmethod
    def readout(x):
        '''
        batch * num * features ->batch * (2*features)
        '''
        x_max = x.max(dim = 1)[0]
        x_mean = x.mean(dim = 1) 
        return torch.cat([x_mean,x_max],dim = 1)

