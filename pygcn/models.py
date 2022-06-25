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
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(2 * nhid)

        self.pool = GraphPooling(4 * nhid,ratio = 0.25)
        self.fc1 = nn.Linear(8 * nhid,128)
        self.fc2 = nn.Linear(128,40)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x, adj):
        x1 = F.relu(self.bn1(self.gc1(x, adj).transpose(2,1))) # batch_size * 2048 * nhid
        x1 = x1.transpose(2,1)
        x2 = F.relu(self.bn2(self.gc2(x1, adj).transpose(2,1))) #batch_size * 2048 * 2nhid
        x2 = x2.transpose(2,1)
        x3 = F.relu(self.bn3(self.gc3(x2,adj).transpose(2,1)))
        x3 = x3.transpose(2,1)
        x = torch.cat([x1,x2,x3],dim = 2) #batch_size * 2048 * 4nhid
        #x,_ = self.pool(x,adj)
        x_mean = x.mean(dim = 1) #batch_size * 4nhid
        x_max = x.max(dim = 1)[0]
        x = torch.cat([x_mean,x_max],dim = 1)

        x = F.relu(self.bn4(self.fc1(x))) #batch_size * 128
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)