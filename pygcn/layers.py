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