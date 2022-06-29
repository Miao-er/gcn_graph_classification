import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat,nhid)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gc2 = GraphConvolution(2 * nhid,2*nhid)
        self.bn2 = nn.BatchNorm1d(2*nhid)
        self.gc3 = GraphConvolution(4 * nhid,4*nhid)
        self.bn3 = nn.BatchNorm1d(4*nhid)

        self.fc1 = nn.Linear(8*nhid,4*nhid)
        self.bn4 = nn.BatchNorm1d(4*nhid)
        self.fc2 = nn.Linear(4*nhid,2*nhid)
        self.bn5 = nn.BatchNorm1d(2*nhid)
        self.fc3 = nn.Linear(2*nhid,nclass)
        
    def forward(self, x, adj):
        x1 = self.gc1(x, adj)
        x1 = F.relu(self.bn1(x1.transpose(2,1)))
        x1 = x1.transpose(2,1)
        x1_out = self.readout(x1)
        x1 = self.cat_feat(x1,self.readout(x1,mean = False)) # b*n* 2hid

        x2 = self.gc2(x1, adj)
        x2 = F.relu(self.bn2(x2.transpose(2,1))) 
        x2 = x2.transpose(2,1)
        x2_out = self.readout(x2)
        x2 = self.cat_feat(x2,self.readout(x2,mean = False)) # b*n* 4hid

        x3 = self.gc3(x2,adj)
        x3 = F.relu(self.bn3(x3.transpose(2,1))) 
        x3 = x3.transpose(2,1)

        x = self.readout(x3,mean = True)
        #x = torch.cat([x1_out,x2_out,x3_out],dim = 1)
        x = F.relu(self.bn4(self.fc1(x))) 
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    
    @staticmethod
    def readout(x,mean = True):
        '''
        batch * num * features ->batch * (2*features)
        '''
        x_max = x.max(dim = 1)[0]
        x_mean = x.mean(dim = 1) 
        if mean:
            return torch.cat([x_mean,x_max],dim = 1)
        return x_max
    @staticmethod
    def cat_feat(x,x_feature):
        '''
        x:batch_size * num_points * feat
        x_feature:batch_size * feat
        '''
        x_feature = x_feature.unsqueeze(1).repeat(1,x.shape[1],1)
        x = torch.cat([x,x_feature],dim = 2)
        return x

class PointNet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(nfeat, nhid, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(nhid, nhid, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(nhid, nhid, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(nhid, 2*nhid, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(2*nhid, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.bn3 = nn.BatchNorm1d(nhid)
        self.bn4 = nn.BatchNorm1d(2*nhid)
        self.bn5 = nn.BatchNorm1d(1024)
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, nclass)

    def forward(self, x,adj):
        x = x.transpose(1,2)  #batch_size * feat_dim * num_points
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


class DGCNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(DGCNN, self).__init__()
        self.k = 20
        self.bn1 = nn.BatchNorm2d(nhid)
        self.bn2 = nn.BatchNorm2d(nhid)
        self.bn3 = nn.BatchNorm2d(2*nhid)
        self.bn4 = nn.BatchNorm2d(4*nhid)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, nhid, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(2*nhid, nhid, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(2*nhid, 2*nhid, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(4*nhid, 4*nhid, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(8*nhid, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, nclass)

    def forward(self, x,adj):
        x = x.transpose(1,2)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature