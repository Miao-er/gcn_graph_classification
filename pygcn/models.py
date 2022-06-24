import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 1)
        self.fc = nn.Linear(2048,nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)) # batch_size * 2048 * nhid
        x = F.dropout(x, self.dropout, training=self.training) 
        x = self.gc2(x, adj) #batch_size * 2048 * 1
        x = x.squeeze(-1)
        x = self.fc(x) #batch_size * nclass
        return F.log_softmax(x, dim=1)
