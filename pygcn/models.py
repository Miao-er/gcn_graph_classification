import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(2 * nhid, 2 * nhid)
        self.gc3 = GraphConvolution(2 * nhid, 2 * nhid)
        self.att1 = GraphConvolution(nhid,1)
        self.att2 = GraphConvolution(2 * nhid,1)
        self.att3 = GraphConvolution(2 * nhid,1)
        self.fc1 = nn.Linear(4 * nhid,128)
        self.fc2 = nn.Linear(128,40)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) # batch_size * 2048 * nhid
        x1 = F.dropout(x, self.dropout, training=self.training) 
        x1_mean = x1.mean(dim = 1).squeeze(1).repeat(1,2048).reshape(x1.shape[0],x1.shape[1],-1)
        x1 = torch.cat([x1,x1_mean],dim = 2)
        # att1 = F.softmax(self.att1(x,adj).squeeze(-1),dim = 1) #batch * 2048
        # _,idx = att1.topk(128,dim = 1)
        # x,adj = self.coarsen(x,adj,idx) #batch_size * 128 * nhid

        # x2 = F.relu(self.gc2(x1, adj)) #batch_size * 2048 * 2nhid
        # x2_mean = x2.mean(dim = 1).squeeze(1).repeat(1,2048).reshape(x2.shape[0],x2.shape[1],-1)
        # x2 = torch.cat([x2,x2_mean],dim = 2)

        # att2 = F.softmax(self.att2(x,adj).squeeze(-1),dim = 1) #batch * 2048
        # _,idx = att2.topk(8,dim = 1)
        # x,adj = self.coarsen(x,adj,idx) #batch_size * 8 * 2nhid

        x3 = F.relu(self.gc3(x1,adj)) #batch_size * 2048 * 4nhid

        att3 = F.softmax(self.att3(x3,adj).squeeze(-1),dim = 1) #batch * 2048
        _,idx = att3.topk(1,dim = 1)
        x,adj = self.coarsen(x3,adj,idx) #batch_size * 1 * 4nhid
        x = x.squeeze(1)
        x3_mean = x3.mean(dim = 1).squeeze(1) #batch_size * 4nhid
        x = torch.cat([x,x3_mean],dim = 1)

        # x3_max = x3.max(dim = 1)[0]
        # x3_mean = x3.mean(dim = 1)
        # x = torch.cat([x3_mean,x3_max],dim = 1)

        x = F.relu(self.fc1(x)) #batch_size * 128
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    @staticmethod
    def coarsen(x,adj,idx):
        batch_size = x.shape[0]
        seq = torch.arange(batch_size)
        x = x[seq,idx.T].transpose(0,1)  #根据idx取子图节点
        adj = adj[seq,idx.T].transpose(0,1) #根据idx取子图邻接矩阵的行
        adj = adj[seq,:,idx.T].transpose(0,1) #根据idx取子图邻接矩阵的列
        return x.contiguous(),adj