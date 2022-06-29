from re import A
import numpy as np
import scipy.sparse as sp
from sklearn.utils import shuffle
import torch
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from yaml import load
import logging
 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s]%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def download():
    # gcn_graph_classification/data/...
    PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = PATH + '/data/modelnet40_ply_hdf5_2048'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
    return DATA_DIR

def load_raw_data(partition):
    '''
    train:9843
    test:2468
    label class:40
    '''
    DATA_DIR = download()
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32') #2048,2048,3
        label = f['label'][:].astype('int64') #2048,1
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label 

def translate_pointcloud(pointcloud):
    x_max = pointcloud.max(axis = 0)
    x_min = pointcloud.min(axis = 0)
    x_mid = (x_max + x_min)/2
    pointcloud = pointcloud - x_mid
    scale = pointcloud.max()
    translated_pointcloud = pointcloud / scale

    return translated_pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_raw_data(partition)
        self.label = self.label.squeeze(1)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        pointcloud = translate_pointcloud(pointcloud)
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def load_data(batch_size,tag):
    data = ModelNet40(2048,tag)
    if tag == 'train':
        loader = DataLoader(dataset=data,batch_size = batch_size,shuffle = True)
    else:
        loader = DataLoader(dataset=data,batch_size = batch_size)

    return loader

def knn(x, k):
    device = torch.device('cuda')
    inner = -2*torch.matmul(x,x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True) # batch_size * num_point * 1
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # (a-b)^2 = a^2 + b^2 -2ab
 
    val,idx = pairwise_distance.topk(k=k, dim=-1)   # (batch_size, num_points, k)
    #weight = torch.exp(pairwise_distance)
    batch_size,num_points,feat_dim = x.shape
    idx = idx.reshape(batch_size * num_points,-1)
    idx_base = torch.arange(0,batch_size * num_points).view(-1,1)*num_points
    idx = (idx + idx_base.to(device)).reshape(-1)
    # weight = weight.reshape(-1)
    # mask = torch.zeros_like(weight).bool()
    # mask[idx] = True
    # weight[~mask] = 0
    # weight = weight.reshape(batch_size,num_points,num_points)                                                                  
    mask = torch.zeros_like(pairwise_distance.reshape(-1),device = x.device)
    mask[idx] = 1.0
    mask = mask.reshape(batch_size,num_points,num_points)
    return mask

def build_graph(batch_data,k):
    '''
    返回邻接矩阵
    data:batch_size * 2048 * features
    '''
    adj = knn(batch_data,k = k)
    adj_T = adj.transpose(2,1)
    adj = adj + adj_T.mul((adj_T > adj).float()) - adj.mul((adj_T > adj).float())
    adj = normalize(adj)# +torch.eye(adj.shape[1]).to(torch.device('cuda')))
    return adj
    

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = mx.sum(dim = 2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    rr_inv = torch.pow(r_inv,0.5)
    mx = rr_inv.unsqueeze(-1).mul(mx).mul(rr_inv.unsqueeze(1))
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct.item() , len(labels)

def class_accuracy(class_map,output = None,labels = None):
    if labels is not None:
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        for corr,label in zip(correct,labels):
            class_map[label.item()][0] += corr.item()
            class_map[label.item()][1] += 1
    else:
        new_map = {}
        for label,count in class_map.items():
            try:
                new_map[label] = count[0]/count[1]
            except:
                new_map[label] = 0
            #finally:
                #print(f'label:{label},accuracy:{new_map[label]}')
        return new_map
