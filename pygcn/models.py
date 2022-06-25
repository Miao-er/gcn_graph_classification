import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,GraphPooling


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid,2 * nhid)
        self.pool = GraphPooling(4 * nhid,ratio = 0.25)
        self.fc1 = nn.Linear(8 * nhid,128)
        self.fc2 = nn.Linear(128,40)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj)) # batch_size * 2048 * nhid
        #x1 = F.dropout(x, self.dropout, training=self.training) 
        #x1_mean = x1.mean(dim = 1).squeeze(1).repeat(1,2048).reshape(x1.shape[0],x1.shape[1],-1)
        #x1 = torch.cat([x1,x1_mean],dim = 2)
        # att1 = F.softmax(self.att1(x,adj).squeeze(-1),dim = 1) #batch * 2048
        # _,idx = att1.topk(128,dim = 1)
        # x,adj = self.coarsen(x,adj,idx) #batch_size * 128 * nhid

        x2 = F.relu(self.gc2(x1, adj)) #batch_size * 2048 * 2nhid
        #x2_mean = x2.mean(dim = 1).squeeze(1).repeat(1,2048).reshape(x2.shape[0],x2.shape[1],-1)
        #x2 = torch.cat([x2,x2_mean],dim = 2)
        # att2 = F.softmax(self.att2(x,adj).squeeze(-1),dim = 1) #batch * 2048
        # _,idx = att2.topk(8,dim = 1)
        # x,adj = self.coarsen(x,adj,idx) #batch_size * 8 * 2nhid

        x3 = F.relu(self.gc3(x2,adj)) 

        x = torch.cat([x1,x2,x3],dim = 2) #batch_size * 2048 * 4nhid
        #x,_ = self.pool(x,adj)
        x_mean = x.mean(dim = 1) #batch_size * 4nhid
        x_max = x.max(dim = 1)[0]
        x = torch.cat([x_mean,x_max],dim = 1)

        # x3_max = x3.max(dim = 1)[0]
        # x3_mean = x3.mean(dim = 1)
        # x = torch.cat([x3_mean,x3_max],dim = 1)

        x = F.relu(self.fc1(x)) #batch_size * 128
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)