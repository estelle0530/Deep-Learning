import torch
import pandas as pd
import numpy as np

from scipy.spatial import distance
from scipy.sparse import vstack

from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import scanpy as sc
import anndata

from collections import OrderedDict
from collections import defaultdict
from collections import Counter

import os
import functools
import itertools as it
import time
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import tensorflow as tf
import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as F

import argparse
import torch.utils.data as data
import numpy as np
import torch

import argparse
import torch.utils.data as data
import numpy as np
import torch

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-pretrain_batch', '--pretrain_batch',
                        type=int,
                        help='Batch size for pretraining. Default: no batch',
                        default=None)
    
    parser.add_argument('-pretrain','--pretrain',
                        type = bool,
                        default = True,
                        help='Pretrain model with autoencoder; otherwise load existing')
    
    parser.add_argument('-nepoch', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=30)

    parser.add_argument('-nepoch_pretrain', '--epochs_pretrain',
                        type=int,
                        help='number of epochs to pretrain for',
                        default=25)

    parser.add_argument('-source_file','--model_file',
                        type = str,
                        default = 'trained_models/source.pt',
                        help='location for storing source model and data')

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20) 

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)
  
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=3)
    
    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda')
    
    return parser


'''
Class representing dataset for an single-cell experiment.
'''

IMG_CACHE = {}


class ExperimentDataset(data.Dataset):
    
    
    def __init__(self, x, cells, genes, metadata, y=[]):
        '''
        x: numpy array of gene expressions of cells (rows are cells)
        cells: cell IDs in the order of appearance
        genes: gene IDs in the order of appearance
        metadata: experiment identifier
        y: numeric labels of cells (empty list if unknown)
        '''
        super(ExperimentDataset, self).__init__()
        
        self.nitems = x.shape[0]
        if len(y)>0:
            print("== Dataset: Found %d items " % x.shape[0])
            print("== Dataset: Found %d classes" % len(np.unique(y)))
                
        if type(x)==torch.Tensor:
            self.x = x
        else:
            shape = x.shape[1]
            self.x = [torch.from_numpy(inst).view(shape).float() for inst in x]
        if len(y)==0:
            y = np.zeros(len(self.x), dtype=np.int64)
        
        self.y = tuple(y.tolist())
        self.xIDs = cells
        self.yIDs = genes
        self.metadata = metadata
            
    def __getitem__(self, idx):
        return self.x[idx].squeeze(), self.y[idx], self.xIDs[idx]
    #, self.yIDs[idx]

    def __len__(self):
        return self.nitems
    
    def get_dim(self):
        return self.x[0].shape[0]

class EpochSampler(object):
    '''
    EpochSampler: yield permuted indexes at each epoch.
   
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, indices):
        '''
        Initialize the EpochSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - iterations: number of epochs
        '''
        super(EpochSampler, self).__init__()
        
        self.indices = indices
        

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        
    
        while(True):
            shuffled_idx = self.indices[torch.randperm(len(self.indices))]
            
            yield shuffled_idx
            

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
    
import torch
from torch.utils.data import DataLoader ##prefetch by batch
#from model.epoch_sampler import EpochSampler


def init_labeled_loader(data, val_split = 0.8):
    """Initialize loaders for train and validation sets. 
    Class labels are used only
    for stratified sampling between train and validation set.
    
    Validation_split = % keeps in training set 
    
    """
    
    target = torch.tensor(list(data.y))
    uniq = torch.unique(target, sorted=True)
    
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
    class_idxs = [idx[torch.randperm(len(idx))] for idx in class_idxs]
    
    train_idx = torch.cat([idx[:int(val_split*len(idx))] for idx in class_idxs])
    val_idx = torch.cat([idx[int(val_split*len(idx)):] for idx in class_idxs])
    
    train_loader = DataLoader(data, 
                              batch_sampler=EpochSampler(train_idx),
                              pin_memory=True)
    
    val_loader = DataLoader(data, 
                            batch_sampler=EpochSampler(val_idx),
                            pin_memory=True)
    
    return train_loader, val_loader


def init_loader(datasets, val_split = 0.8):
    
    train_loader_all = []
    val_loader_all = []
    
    for data in datasets:
        
        curr_load_tr, curr_load_val = init_labeled_loader(data, val_split)
        train_loader_all.append(curr_load_tr)
        val_loader_all.append(curr_load_val)
    
    if val_split==1:
        val_loader_all = None
        
    return train_loader_all, val_loader_all


def init_data_loaders(labeled_data, unlabeled_data, 
                      pretrain_data, pretrain_batch, val_split):
    
    """Initialize loaders for pretraing, 
    training (labeled and unlabeled datasets) and validation. """
    
    train_loader, val_loader = init_loader(labeled_data, val_split)
    
    if not pretrain_data:
        pretrain_data = unlabeled_data
    
    pretrain_loader = torch.utils.data.DataLoader(dataset=pretrain_data,
                                                  shuffle=True,
                                                  batch_size=pretrain_batch if pretrain_batch!=None else len(unlabeled_data.x))        
    test_loader = DataLoader(unlabeled_data, 
                            batch_sampler=EpochSampler(torch.randperm(len(unlabeled_data.x))),
                            pin_memory=True) 
    
    #test_loader,_ = init_loader([unlabeled_data], 1.0) # to reproduce results in the paper
    #test_loader = test_loader[0]
    return train_loader, test_loader, pretrain_loader, val_loader
           
           
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import linear_sum_assignment 

def compute_scores(y_true, y_pred, scoring={'accuracy','precision','recall','nmi',
                                                'adj_rand','f1_score','adj_mi'}):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    scores = {}
    y_true, y_pred = hungarian_match(y_true, y_pred)
    set_scores(scores, y_true, y_pred, scoring)
        
    return scores


def set_scores(scores, y_true, y_pred, scoring):
    labels=list(set(y_true))
    
    for metric in scoring:
        if metric=='accuracy':
            scores[metric] = metrics.accuracy_score(y_true, y_pred)
        elif metric=='precision':
            scores[metric] = metrics.precision_score(y_true, y_pred, labels, average='macro')
        elif metric=='recall':
            scores[metric] = metrics.recall_score(y_true, y_pred, labels, average='macro')
        elif metric=='f1_score':
            scores[metric] = metrics.f1_score(y_true, y_pred, labels, average='macro')
        elif metric=='nmi':
            scores[metric] = metrics.normalized_mutual_info_score(y_true, y_pred)
        elif metric=='adj_mi':
            scores[metric] = metrics.adjusted_mutual_info_score(y_true, y_pred)
        elif metric=='adj_rand':
            scores[metric] = metrics.adjusted_rand_score(y_true, y_pred)
                
                
def hungarian_match(y_true, y_pred):
    """Matches predicted labels to original using hungarian algorithm."""
    
    y_true = adjust_range(y_true)
    y_pred = adjust_range(y_pred)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(-w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    d = {i:j for i, j in ind}
    y_pred = np.array([d[v] for v in y_pred])
    
    return y_true, y_pred


def adjust_range(y):
    """Assures that the range of indices if from 0 to n-1."""
    y = np.array(y, dtype=np.int64)
    val_set = set(y)
    mapping = {val:i for  i,val in enumerate(val_set)}
    y = np.array([mapping[val] for val in y], dtype=np.int64)
    return y


class VAE(nn.Module):
    '''
    '''
    def __init__(self, latent_dim , n_feature , network_architecture , 
                 lambda_reconstruct, lambda_kl ,
                 p_drop=0.2):
        
        super(VAE, self).__init__()
        
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_kl = lambda_kl
        
        self.hidden_encode1=network_architecture['n_hidden_recog_1']
        self.hidden_encode2=network_architecture['n_hidden_recog_2']
        self.hidden_encode3=network_architecture['n_hidden_recog_3']
        
        self.hidden_decode1=network_architecture['n_hidden_gener_1']
        self.hidden_decode2=network_architecture['n_hidden_gener_2']
        self.hidden_decode3=network_architecture['n_hidden_gener_3']
        
        self.latent_dim = latent_dim
        self.n_feature = n_feature
        
        self.encoder = nn.Sequential(
            
            nn.Linear(self.n_feature, self.hidden_encode1, bias=True),
            nn.ELU(alpha=0.2),
            nn.Dropout(p=p_drop),
            nn.Linear(self.hidden_encode1, self.hidden_encode2, bias=True),
            nn.ELU(alpha=0.2),
            nn.Linear(self.hidden_encode2, self.hidden_encode3, bias=True),
            nn.ELU(alpha=0.2)
        )
        
        self.mu = nn.Linear(self.hidden_encode3, self.latent_dim, bias=True)
        self.logvar = nn.Linear(self.hidden_encode3, self.latent_dim, bias=True)
        
        self.decoder = nn.Sequential(
            
            nn.Linear(self.latent_dim, self.hidden_decode1, bias=True),
            nn.ELU(alpha=0.2),
            nn.Dropout(p=p_drop),
            nn.Linear(self.hidden_decode1, self.hidden_decode2, bias=True),
            nn.ELU(alpha=0.2),
            nn.Linear(self.hidden_decode2, self.hidden_decode3, bias=True),
            nn.ELU(alpha=0.2),
            nn.Linear(self.hidden_decode3, self.n_feature, bias=True)
        )
    
    
    def encode(self, x):
        '''
        Return latent parameters
        '''
        encoded = self.encoder(x)
        
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        
        return [mu, logvar]
        
    
    def decode(self, z):
        '''
        Reconstruct
        '''
        decoded = self.decoder(z)
        return decoded
    
    
    def reparameterize(self, mu , logvar):
        ''' 
        Reparametraization sample N(mu, var) from N(0,1) noise
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) #std represents the size of the tensor
        
        return eps*std + mu                
      
    def forward(self, x):
        
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)
        
        decoded = self.decode(z)
        
        return decoded, mu, logvar
    
import torch
from torch.nn import functional as F

#from model.utils import euclidean_dist

##training set encoded distance to landmarks 
def loss_task(encoded, prototypes, target, criterion='dist'):
    """Calculate loss.
    criterion: NNLoss - assign to closest prototype and calculate NNLoss
    dist - loss is distance to prototype that example needs to be assigned to
                and -distance to prototypes from other class
    """
    
    uniq = torch.unique(target, sorted=True)
    
    ###index of samples for each class of labels
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
    
    # prepare targets so they start from 0,1
    for idx,v in enumerate(uniq):
        target[target==v]=idx
    
    dists = euclidean_dist(encoded, prototypes)
    
    if criterion=='NNLoss':
       
        loss = torch.nn.NLLLoss()
        log_p_y = F.log_softmax(-dists, dim=1)
        
        loss_val = loss(log_p_y, target)
        _, y_hat = log_p_y.max(1)
        
    
    elif criterion=='dist':
        
        loss_val = torch.stack([dists[idx_example, idx_proto].mean(0) for idx_proto,idx_example in enumerate(class_idxs)]).mean()
        #loss_val1 = loss_val1/len(embeddings) 
        y_hat = torch.max(-dists,1)[1]
        
    acc_val = y_hat.eq(target.squeeze()).float().mean()    
        
    return loss_val, acc_val

def loss_test_nn(encoded, prototypes):
    dists = euclidean_dist(encoded, prototypes)
    min_dist = torch.min(dists, 1)
    
    y_hat = min_dist[1]
    args_uniq = torch.unique(y_hat, sorted=True)
    args_count = torch.stack([(y_hat==x_u).sum() for x_u in args_uniq])
    print(args_count)
    
    loss = torch.nn.NLLLoss()
    log_p_y = F.log_softmax(-dists, dim=1)
    print(log_p_y.shape)
        
    loss_val = loss(log_p_y, y_hat)
    _, y_hat = log_p_y.max(1)
    
    return loss_val, args_count

###Intra cluster distance
def loss_test_basic(encoded, prototypes):
    
    dists = euclidean_dist(encoded, prototypes)
    min_dist = torch.min(dists, 1)
    
    y_hat = min_dist[1]
    args_uniq = torch.unique(y_hat, sorted=True)
    args_count = torch.stack([(y_hat==x_u).sum() for x_u in args_uniq])
    #print(args_count)
    
    min_dist = min_dist[0] # get_distances
    
    #thr = torch.stack([torch.sort(min_dist[y_hat==idx_class])[0][int(len(min_dist[y_hat==idx_class])*0.9)] for idx_class in args_uniq])
    #loss_val = torch.stack([min_dist[y_hat==idx_class][min_dist[y_hat==idx_class]>=thr[idx_class]].mean(0) for idx_class in args_uniq]).mean()
    
    loss_val = torch.stack([min_dist[y_hat==idx_class].mean(0) for idx_class in args_uniq]).mean()
    
    #loss_val,_ = loss_task(encoded, prototypes, y_hat, criterion='dist') # same
    
    return loss_val, args_count

##For unannotated set 
def loss_test(encoded, prototypes, tau):
    
    #prototypes = torch.stack(prototypes).squeeze() 
    loss_val_test, args_count = loss_test_basic(encoded, prototypes)
    
    ###inter cluster distance 
    if tau>0:
        dists = euclidean_dist(prototypes, prototypes)
        nproto = prototypes.shape[0]
        loss_val2 = - torch.sum(dists)/(nproto*nproto-nproto)
        loss_val_test += tau*loss_val2
        
    return loss_val_test, args_count

def reconstruction_loss(decoded, x):
    
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    #print('Reconstruction {}'.format(loss_rcn))
    
    return loss_rcn


def vae_loss(model, decoded, x, mu, logvar):
    
#     loss_func = torch.nn.MSELoss()
#     loss_rcn = loss_func(decoded, x)
    #KL =  torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    
    loss_rcn = torch.mean(torch.sum((x - decoded).pow(2),1))
    
    KL_loss = (-0.5)*(1 + logvar - mu.pow(2) - logvar.exp())
    KL = torch.mean(torch.sum(KL_loss, axis=1))
    
    #print(model.lambda_reconstruct, model.lambda_kl)
    total_loss = model.lambda_reconstruct*loss_rcn + model.lambda_kl * KL 
    
    
    #print(f"Reconstruction Loss: {loss_rcn} KL Loss: {KL}")
    
    return total_loss 


import numpy as np
import torch
#from sklearn.cluster import k_means_
from sklearn.cluster import KMeans

## Training set (annotated landmarks)
def compute_landmarks_tr(embeddings, target, prev_landmarks=None, tau=0.2):
    
    """Computing landmarks of each class in the labeled meta-dataset. 
    
    Landmark is a closed form solution of 
    minimizing distance to the mean and maximizing distance to other landmarks. 
    
    If tau=0, landmarks are just mean of data points.
    
    embeddings: embeddings of the labeled dataset
    target: labels in the labeled dataset
    prev_landmarks: landmarks from previous iteration
    tau: regularizer for inter- and intra-cluster distance
    """
    
    uniq = torch.unique(target, sorted=True)
    class_idxs = list(map(lambda c: target.eq(c).nonzero(), uniq))
    
    landmarks_mean = torch.stack([embeddings[idx_class].mean(0) for idx_class in class_idxs]).squeeze()
    
    if prev_landmarks is None or tau==0:
        return landmarks_mean
    
    suma = prev_landmarks.sum(0)
    nlndmk = prev_landmarks.shape[0]
    lndmk_dist_part = (tau/(nlndmk-1))*torch.stack([suma-p for p in prev_landmarks])
    landmarks = 1/(1-tau)*(landmarks_mean-lndmk_dist_part)
    
    return landmarks



def init_landmarks(n_clusters, tr_load, test_load, model, device, mode='kmeans', pretrain=True):
    """Initialization of landmarks of the labeled and unlabeled meta-dataset.
    nclusters: number of expected clusters in the unlabeled meta-dataset
    tr_load: data loader for labeled meta-dataset
    test_load: data loader for unlabeled meta-dataset
    """
    lndmk_tr = [torch.zeros(size=(len(np.unique(dl.dataset.y)), 
                                  model.latent_dim), 
                            requires_grad=True, device=device) for dl in tr_load]
    
    lndmk_test = [torch.zeros(size=(1, model.latent_dim), 
                              requires_grad=True, device=device) 
                       for _ in range(n_clusters)]
    
    kmeans_init_tr = [init_step(dl.dataset, model, device, pretrained=pretrain, mode=mode) 
                      for dl in tr_load]
    
    kmeans_init_test = init_step(test_load.dataset, model, device, 
                                 pretrained=pretrain, mode=mode, 
                                 n_clusters=n_clusters)
    
    ##No gradient calculation
    with torch.no_grad():
        [lndmk.copy_(kmeans_init_tr[idx])  for idx,lndmk in enumerate(lndmk_tr)]
        [lndmk_test[i].copy_(kmeans_init_test[i,:]) for i in range(kmeans_init_test.shape[0])]
        
    return lndmk_tr, lndmk_test


def init_step(dataset, model, device, pretrained, mode='kmeans',n_clusters=None):
    """Initialization of landmarks with k-means or k-means++ given dataset."""
    
    if n_clusters==None:
        n_clusters = len(np.unique(dataset.y))
    
    nexamples = len(dataset.x)
        
    X =  torch.stack([dataset.x[i] for i in range(nexamples)])
    
    if mode=='kmeans++':
        if not pretrained: # find centroids in original space
            landmarks = k_means_._init_centroids(X.cpu().numpy(), n_clusters, 'k-means++')
            landmarks = torch.tensor(landmarks, device=device)
            landmarks = landmarks.to(device)
            lndmk_encoded,_ = model(landmarks)
            
        else:
            X = X.to(device)
            encoded,_ = model(X)
            landmarks = k_means_._init_centroids(encoded.data.cpu().numpy(), n_clusters, 'k-means++')
            lndmk_encoded = torch.tensor(landmarks, device=device)
    
    elif mode=='kmeans': # run kmeans clustering
        if not pretrained: 
            kmeans = KMeans(n_clusters, random_state=0).fit(X.cpu().numpy())
            landmarks = torch.tensor(kmeans.cluster_centers_, device=device)
            landmarks = landmarks.to(device)
            ##Feed forward net on landmarks  (k means cluster)
            ##landmarks are k means cluster centers = coordinates of cluster center
            #lndmk_encoded,_ = model(landmarks)
            _ , lndmk_encoded, _ = model(landmarks)
        
        ##cluster on lower embedding space
        else:
            X = X.to(device)
            #encoded,_ = model(X)
            decode , encoded, logvar = model(X)
            kmeans = KMeans(n_clusters, random_state=0).fit(encoded.data.cpu().numpy())
            lndmk_encoded = torch.tensor(kmeans.cluster_centers_, device=device)
    
    return lndmk_encoded


class VAE_MARS:
    def __init__(self, n_clusters, params, 
                 labeled_data, unlabeled_data, 
                 pretrain_data=None, 
                 
                 val_split=1.0, hid_dim_1=1000, hid_dim_2=100, p_drop=0.0, tau=0.2,
                 
                 latent_dim = 50, 
                 n_feature = 10, 
                 
                 network_architecture={"n_hidden_recog_1": 250, "n_hidden_recog_2": 100,
                                       "n_hidden_recog_3": 100, "n_hidden_gener_1": 100,
                                       "n_hidden_gener_2": 100, "n_hidden_gener_3": 100},
                 
                 lambda_reconstruct = 1e-4, lambda_kl = 1e-3
                ):
        
        """Initialization of MARS.
        n_clusters: number of clusters in the unlabeled meta-dataset
        params: parameters of the MARS model
        labeled_data: list of labeled datasets. Each dataset needs to be instance of CellDataset.
        unlabeled_data: unlabeled dataset. Instance of CellDataset.
        pretrain_data: dataset for pretraining MARS. Instance of CellDataset. If not specified, unlabeled_data
                        will be used.
        val_split: percentage of data to use for train/val split (default: 1, meaning no validation set)
        hid_dim_1: dimension in the first layer of the network (default: 1000)
        hid_dim_2: dimension in the second layer of the network (default: 100)
        p_drop: dropout probability (default: 0)
        tau: regularizer for inter-cluster distance
        """
        train_load, test_load, pretrain_load, val_load = init_data_loaders(labeled_data, unlabeled_data, 
                                                                           pretrain_data, params.pretrain_batch, 
                                                                           val_split)
        self.train_loader = train_load
        self.test_loader = test_load
        self.pretrain_loader = pretrain_load
        self.val_loader = val_load
        
        ##data file type (string name)
        self.labeled_metadata = [data.metadata for data in labeled_data]
        self.unlabeled_metadata = unlabeled_data.metadata
        
        self.genes = unlabeled_data.yIDs
        
        ##number of genes 
        x_dim = self.test_loader.dataset.get_dim()
        
        ##KNN clusters
        self.n_clusters = n_clusters
        
        ##General Parameters
        self.device = params.device
        self.epochs = params.epochs
        self.epochs_pretrain = params.epochs_pretrain
        self.pretrain_flag = params.pretrain
        self.model_file = params.model_file
        self.lr = params.learning_rate
        self.lr_gamma = params.lr_scheduler_gamma
        self.step_size = params.lr_scheduler_step
        
        self.tau = tau
        
        ##VAE parameter
        self.lambda_reconstruct = lambda_reconstruct
        self.lambda_kl = lambda_kl
        
#         self.hidden_encode1=network_architecture['n_hidden_recog_1']
#         self.hidden_encode2=network_architecture['n_hidden_recog_2']
#         self.hidden_encode3=network_architecture['n_hidden_recog_3']
        
#         self.hidden_decode1=network_architecture['n_hidden_gener_1']
#         self.hidden_decode2=network_architecture['n_hidden_gener_2']
#         self.hidden_decode3=network_architecture['n_hidden_gener_3']
        
        self.latent_dim = latent_dim
        self.n_feature = n_feature
        
        self.init_model(self.latent_dim, self.n_feature, network_architecture, 
                        self.lambda_reconstruct, self.lambda_kl, params.device)
        
        
    ###################################################################
    ###################################################################    
    ###### With the fine tuned hyper parameter     
    ###### Change the implementation of AE to VAE     
    def init_model(self, 
                   latent_dim , 
                   n_feature, 
                   network_architecture, 
                   lambda_reconstruct, lambda_kl,
                   device
                  ):
        """
        Initialize the model.
        """
        #self.model = FullNet(x_dim, hid_dim, z_dim, p_drop).to(device)
        self.model = VAE(latent_dim = latent_dim, 
                         n_feature = n_feature, 
                         network_architecture= network_architecture,
                         lambda_reconstruct = lambda_reconstruct, 
                         lambda_kl = lambda_kl).to(device)
        
    ###################################################################
    ###################################################################
    
    def init_optim(self, param1, param2, learning_rate):
        """Initializing optimizers."""
        
        optim = torch.optim.Adam(params=param1, lr=learning_rate)
        optim_landmk_test = torch.optim.Adam(params=param2, lr=learning_rate)
        
        return optim, optim_landmk_test
    
    ###################################################################
    ###################################################################
    ########VAE (KL + reconstruction loss + regularized)
    def pretrain(self, optim):
        """
        Pretraining model with variational autoencoder.
        only on unannotated dataset 
        optim: optimizer
        """
        print('Pretraining..')
        
        pretrain_loss =[]
        
        for e in range(self.epochs_pretrain):
            
            start = time.time()
            
            
            for _, batch in enumerate(self.pretrain_loader):
                
                x,_,_ = batch
                x = x.to(self.device)
                
                #_, decoded = self.model(x)
                decoded, mu, logvar = self.model(x)
                
                #loss = reconstruction_loss(decoded, x) 
                loss = vae_loss(self.model, decoded, x, mu, logvar)
                
                optim.zero_grad()              
                loss.backward()                    
                optim.step() 
                
                
            pretrain_loss.append(loss) 
            
            print(f"Pretraining Epoch {e}, Loss: {loss}")
            print("Time: ", time.time()-start)
        
        print("Pretraining done")
        return mu, pretrain_loss
    ###################################################################
    ###################################################################
    
    
    def train(self, evaluation_mode=True, save_all_embeddings=True):
        """Train model.
        evaluation_mode: if True, validates model on the unlabeled dataset. 
        In the evaluation mode, ground truth labels of the unlabeled dataset must be 
        provided to validate model
        
        save_all_embeddings: if True, MARS embeddings for annotated and unannotated 
        experiments will be saved in an anndata object,
        otherwise only unnanotated will be saved. 
        If naming is called after, all embeddings need to be saved
        
        return: adata: anndata object containing labeled and unlabeled meta-dataset 
        with MARS embeddings and estimated labels on the unlabeled dataset
                landmk_all: landmarks of the labeled and unlabeled meta-dataset in the 
                order given for training. Landmarks on the unlabeled
                            dataset are provided last
                metrics: clustering metrics if evaluation_mode is True
                
        """
        tr_iter = [iter(dl) for dl in self.train_loader]
        
        if self.val_loader is not None:
            val_iter = [iter(dl) for dl in self.val_loader]
        
        ##############################
        ####Pre train step 
        ##############################
        optim_pretrain = torch.optim.Adam(params=list(self.model.parameters()), lr=self.lr)
        
        if self.pretrain_flag:
            pretrain_latent , pretrain_loss = self.pretrain(optim_pretrain)
        else:
            self.model.load_state_dict(torch.load(self.MODEL_FILE))    
        ##############################
        
        
        test_iter = iter(self.test_loader)
        
        
        ##initialize training (annotated landmark) and testing (unannotated landmark)
        landmk_tr, landmk_test = init_landmarks(self.n_clusters, 
                                                self.train_loader, 
                                                self.test_loader, 
                                                self.model, self.device)
        
        optim, optim_landmk_test = self.init_optim(list(self.model.encoder.parameters()), 
                                                   landmk_test, self.lr)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                       gamma=self.lr_gamma,
                                                       step_size=self.step_size)
        
        latent_tracker = []
        training_history = {'Loss':[],'Accuracy':[],'Loss_tracker':[]}
        best_acc = 0
        
        for epoch in range(1, self.epochs+1):
            
            start = time.time()
            
            #########################
            #####   Model training 
            #########################
            self.model.train()
            
            ##Do_epoch train over each minibatch return
            ##training loss, accuracy, updated landmarks (annoated and unannotated)
            ##latent_history returns latents space for 1) training 2) testing 3) training landmark 4) testing landmark
            loss_tr, acc_tr, landmk_tr, landmk_test, latent_history, loss_tracker = self.do_epoch(tr_iter, test_iter,
                                                                                    optim, 
                                                                                    optim_landmk_test,
                                                                                    landmk_tr, 
                                                                                    landmk_test)
            
            ##Loss training includes 1 - VAE loss 2- embedding and landmark distance 
            print(f'Epoch {epoch} Loss training: {loss_tr}, Accuracy training: {acc_tr}')
            
            print("Time: ", time.time()-start)
        
            ###Track model training 
            training_history['Loss'].append(loss_tr)
            training_history['Accuracy'].append(acc_tr)
            training_history['Loss_tracker'].append(loss_tracker)
            latent_tracker.append(latent_history)
            
            ##only print out the last epoch result indicating end of training
            if epoch==self.epochs: 
                print('\n=== Epoch: {} ==='.format(epoch))
                print('Train acc: {}'.format(acc_tr))
            
            if self.val_loader is None:
                continue
            
            #########################
            #####   Model evaluation 
            #########################
            self.model.eval()
            
            with torch.no_grad():
                
                loss_val, acc_val = self.do_val_epoch(val_iter, landmk_tr)
                
                if acc_val > best_acc:
                    print('Saving model...')
                    best_acc = acc_val
                    best_state = self.model.state_dict()
                    #torch.save(model.state_dict(), self.model_file)
                postfix = ' (Best)' if acc_val >= best_acc else ' (Best: {})'.format(best_acc)
                print('Val loss: {}, acc: {}{}'.format(loss_val, acc_val, postfix))
            lr_scheduler.step()
            
           
            
            
        if self.val_loader is None:
            best_state = self.model.state_dict() # best is last
        
        landmk_all = landmk_tr + [torch.stack(landmk_test).squeeze()]
        
        ##Test time (assign labels to unlabaled data)
        adata_test, eval_results = self.assign_labels(landmk_all[-1], evaluation_mode)
        
        adata = self.save_result(tr_iter, adata_test, save_all_embeddings)
        
        if evaluation_mode:
            return adata, landmk_all, eval_results, training_history, latent_tracker, pretrain_latent , pretrain_loss
        
        return adata, landmk_all, training_history, latent_tracker, pretrain_latent , pretrain_loss
    
    def save_result(self, tr_iter, adata_test, save_all_embeddings):
        """Saving embeddings from labeled and unlabeled dataset, ground truth labels and 
        predictions to joint anndata object."""
        
        adata_all = []

        if save_all_embeddings:
            for task in range(len(tr_iter)): # saving embeddings from labeled dataset
                
                task = int(task)
                x, y, cells = next(tr_iter[task])
                x, y = x.to(self.device), y.to(self.device)
                
                #encoded,_ = self.model(x)
                decoded, mu, logvar = self.model(x)
                #adata_all.append(self.pack_anndata(x, cells, encoded, gtruth=y))
                adata_all.append(self.pack_anndata(x, cells, mu, gtruth=y))
            
        adata_all.append(adata_test)    
        
        if save_all_embeddings:
            adata = adata_all[0].concatenate(adata_all[1:], 
                                             batch_key='experiment',
                                             batch_categories=self.labeled_metadata+[self.unlabeled_metadata])
        else:
            adata = adata_all[0]

        adata.obsm['MARS_embedding'] = np.concatenate([a.uns['MARS_embedding'] for a in adata_all])
        #adata.write('result_adata.h5ad')
        
        return adata
    
    def assign_labels(self, landmk_test, evaluation_mode):
        """Assigning cluster labels to the unlabeled meta-dataset.
        test_iter: iterator over unlabeled dataset
        landmk_test: landmarks in the unlabeled dataset
        evaluation mode: computes clustering metrics if True
        """
          
        torch.no_grad()
        self.model.eval() # eval mode
        
        test_iter = iter(self.test_loader)
        
        x_test, y_true, cells = next(test_iter) # cells are needed because dataset is in random order
        x_test = x_test.to(self.device)
        
        #encoded_test,_ = self.model(x_test)
        decoded, encoded_test, logvar = self.model(x_test)
        
        ###Embedding space eucledian distance 
        dists = euclidean_dist(encoded_test, landmk_test)
        
        ###Prediction based on the minimal distance to learned landmark
        y_pred = torch.min(dists, 1)[1]
        
        adata = self.pack_anndata(x_test, cells, encoded_test, y_true, y_pred)
        
        eval_results = None
        if evaluation_mode:
            eval_results = compute_scores(y_true, y_pred)
            
        return adata, eval_results
    
    
    def pack_anndata(self, x_input, cells, embedding, gtruth=[], estimated=[]):
        """Pack results in anndata object.
        x_input: gene expressions in the input space
        cells: cell identifiers
        embedding: resulting embedding of x_test using MARS
        landmk: MARS estimated landmarks
        gtruth: ground truth labels if available (default: empty list)
        estimated: MARS estimated clusters if available (default: empty list)
        """
        adata = anndata.AnnData(x_input.data.cpu().numpy())
        adata.obs_names = cells
        adata.var_names = self.genes
        if len(estimated)!=0:
            adata.obs['MARS_labels'] = pd.Categorical(values=estimated.cpu().numpy())
        if len(gtruth)!=0:
            adata.obs['truth_labels'] = pd.Categorical(values=gtruth.cpu().numpy())
        adata.uns['MARS_embedding'] = embedding.data.cpu().numpy()
        
        return adata
    
    
    ####Originally, MARS does not care about reconstruction loss, it only minimizes embedding to landmark distances
    ####We can alter this by adding reconstruction and KL term to the total loss being regularized 
    
    def do_epoch(self, tr_iter, test_iter, optim, optim_landmk_test, landmk_tr, landmk_test):
        """
        One training epoch.
        tr_iter: iterator over labeled meta-data
        test_iter: iterator over unlabeled meta-data
        
        optim: optimizer for embedding
        optim_landmk_test: optimizer for test landmarks
        
        landmk_tr: landmarks of labeled meta-data from previous epoch
        landmk_test: landmarks of unlabeled meta-data from previous epoch
        """
        
        latent_history = {'Train_latent':[], 'Train_label':[],
                          'Test_latent':[], 'Test_label':[], 
                          'Train_landmark':[], 'Test_landmark':[]
                         }
        loss_tracker = {'Test_anno':0,
                        'Train_latent_loss':0, 'Train_reconstr_loss': 0, 
                        'Test_latent_loss':0, 'Test_reconstr_loss': 0}
        
        self.set_requires_grad(False)
        
        ##Partially freeze landmark_test
        for landmk in landmk_test:
            landmk.requires_grad=False
            
        ##Initialize gradient of landmakr_test
        optim_landmk_test.zero_grad()
        
        ##########################################################################
        ##########################################################################
        # Update centroids    
        ##########################################################################
        ##########################################################################
        
        task_idx = torch.randperm(len(tr_iter)) ##shuffle per epoch
        ###Annotated landmark (No gradient descent needed)
        for task in task_idx:
            
            ##Training set through the VAE to generate latent space
            task = int(task)
            x, y, _ = next(tr_iter[task])
            x, y = x.to(self.device), y.to(self.device)
            #encoded,_ = self.model(x)
            decoded, encoded, logvar = self.model(x)
            
            curr_landmk_tr = compute_landmarks_tr(encoded, y, landmk_tr[task], tau=self.tau)
            landmk_tr[task] = curr_landmk_tr.data # save landmarks
            
            latent_history['Train_landmark'].append(encoded)
            latent_history['Train_label'].append(y)
        
        ###Unannotated landmark (Use gradient descent to minimize)
        
        ##Unfreeze landmark_test --> autograd update centroids
        for landmk in landmk_test:
            landmk.requires_grad=True
            
        x, y_test,_ = next(test_iter)
        x = x.to(self.device)
        #encoded,_ = self.model(x)
        decoded, encoded, logvar = self.model(x)
        
        ###minimize intra cluster difference and maximize inter cluster difference 
        loss, args_count = loss_test(encoded, 
                                     torch.stack(landmk_test).squeeze(), 
                                     self.tau)
        loss.backward()
        optim_landmk_test.step()
        
        latent_history['Test_landmark'].append(encoded)
        latent_history['Test_label'].append(y_test)
        loss_tracker['Test_anno']=loss
        
        ##########################################################################
        ##########################################################################
        # Update embedding
        ##########################################################################
        ##########################################################################
        
        self.set_requires_grad(True)
        for landmk in landmk_test:
            landmk.requires_grad=False
            
        optim.zero_grad()
        total_accuracy = 0
        total_loss = 0
        ntasks = 0
        mean_accuracy = 0
        
        ###Annotated set
        task_idx = torch.randperm(len(tr_iter))
        
        for task in task_idx:
            task = int(task)
            x, y, _ = next(tr_iter[task])
            x, y = x.to(self.device), y.to(self.device)
            #encoded,_ = self.model(x)
            decoded, encoded, logvar = self.model(x)
            
            ###Eucledian distance between embedding and landmark
            loss, acc = loss_task(encoded, landmk_tr[task], y, criterion='dist')
            
            ##Add VAE reconstruction loss to the model 
            annotated_vae_loss = vae_loss(self.model, decoded, x, encoded, logvar)
            
            ##Record latent space
            latent_history['Train_latent'].append(encoded)
            
            loss_tracker['Train_latent_loss']=loss
            loss_tracker['Train_reconstr_loss']=annotated_vae_loss
            
            total_loss += loss
            #total_loss += annotated_vae_loss
            
            total_accuracy += acc.item()
            
            ntasks += 1
        
        if ntasks>0:
            mean_accuracy = total_accuracy / ntasks
        
        ##Un-Annotated
        # test part
        x,y,_ = next(test_iter)
        x = x.to(self.device)
        #encoded,_ = self.model(x)
        decoded, encoded, logvar = self.model(x)
        
        loss,_ = loss_test(encoded, torch.stack(landmk_test).squeeze(), self.tau)
        
        ##Add VAE reconstruction loss to the model 
        unannotated_vae_loss = vae_loss(self.model, decoded, x, encoded, logvar)
        latent_history['Test_latent'].append(encoded)
        
        loss_tracker['Test_latent_loss']=(loss)
        loss_tracker['Test_reconstr_loss']=(unannotated_vae_loss)
            
        total_loss += loss
        #total_loss += unannotated_vae_loss
        ntasks += 1
    
        mean_loss = total_loss / ntasks
        
        mean_loss.backward()
        optim.step()
        
        return mean_loss, mean_accuracy, landmk_tr, landmk_test, latent_history, loss_tracker
    
    def do_val_epoch(self, val_iter, prev_landmk):
        """One epoch of validation.
        val_iter: iterator over validation set
        prev_landmk: landmarks from previous epoch
        """
        ntasks = len(val_iter)
        task_idx = torch.randperm(ntasks)
        
        total_loss = 0
        total_accuracy = 0
        
        for task in task_idx:
            x, y, _ = next(val_iter[task])
            x, y = x.to(self.device), y.to(self.device)
            #encoded = self.model(x)
            decoded, encoded, logvar = self.model(x)
            
            ###Eucledian distance between embedding and landmark
            loss, acc = loss_task(encoded, prev_landmk[task], y, criterion='dist')
            val_vae_loss = vae_loss(self.model, decoded, x, encoded, logvar)
            
            #total_loss += val_vae_loss
            total_loss += loss
            total_accuracy += acc.item()
        
        mean_accuracy = total_accuracy / ntasks
        mean_loss = total_loss / ntasks
        
        return mean_loss, mean_accuracy
    
    
    def set_requires_grad(self, requires_grad):
        for param in self.model.parameters():
            param.requires_grad = requires_grad
    
    def name_cell_types(self, adata, landmk_all, cell_name_mappings, 
                        top_match=5, umap_reduce_dim=True, ndim=10):
        """For each test cluster, estimate sigma and mean. 
        Fit Gaussian distribution with that mean and sigma
        and calculate the probability of each of the train landmarks 
        to be the neighbor to the mean data point.
        Normalization is performed with regards to all other landmarks in train."""
        
        experiments = list(OrderedDict.fromkeys(list(adata.obs['experiment'])))
        
        ###only get labels from labeled data
        encoded_tr = []
        landmk_tr = []
        landmk_tr_labels = []
        for idx, exp in enumerate(experiments[:-1]):
            tiss = adata[adata.obs['experiment'] == exp,:]
            
            if exp==self.unlabeled_metadata: 
                raise ValueError("Error: Unlabeled dataset needs to be last one in the input anndata object.")
                
            encoded_tr.append(tiss.obsm['MARS_embedding'])
            landmk_tr.append(landmk_all[idx])
            landmk_tr_labels.append(np.unique(tiss.obs['truth_labels']))
            
        tiss = adata[adata.obs['experiment'] == self.unlabeled_metadata,:]
        ypred_test = tiss.obs['MARS_labels']
        uniq_ytest = np.unique(ypred_test)
        encoded_test = tiss.obsm['MARS_embedding']
        
        landmk_tr_labels = np.concatenate(landmk_tr_labels)
        encoded_tr = np.concatenate(encoded_tr)
        landmk_tr = np.concatenate([p.cpu() for p in landmk_tr])
        
        if  umap_reduce_dim:
            encoded_extend = np.concatenate((encoded_tr, encoded_test, landmk_tr))
            adata = anndata.AnnData(encoded_extend)
            sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
            sc.tl.umap(adata, n_components=ndim)
            encoded_extend = adata.obsm['X_umap']
            n1 = len(encoded_tr)
            n2 = n1 + len(encoded_test)
            
            ##UMAP embedding space
            encoded_tr = encoded_extend[:n1,:]
            encoded_test = encoded_extend[n1:n2,:]
            landmk_tr = encoded_extend[n2:,:]
        
        interp_names = defaultdict(list)
        for ytest in uniq_ytest:
            print('\nCluster label: {}'.format(str(ytest)))
            idx = np.where(ypred_test==ytest)
            subset_encoded = encoded_test[idx[0],:]
            
            mean = np.expand_dims(np.mean(subset_encoded, axis=0),0)
            
            sigma  = self.estimate_sigma(subset_encoded)
            
            prob = np.exp(-np.power(distance.cdist(mean, landmk_tr, metric='euclidean'),2)/(2*sigma*sigma))
            prob = np.squeeze(prob, 0)
            normalizat = np.sum(prob)
            
            if normalizat==0:
                print('Unassigned')
                interp_names[ytest].append("unassigned")
                continue
            
            prob = np.divide(prob, normalizat)
            
            uniq_tr = np.unique(landmk_tr_labels)
            prob_unique = []
            
            for cell_type in uniq_tr: # sum probabilities of same landmarks
                prob_unique.append(np.sum(prob[np.where(landmk_tr_labels==cell_type)]))
            
            sorted = np.argsort(prob_unique, axis=0)
            best = uniq_tr[sorted[-top_match:]]
            sortedv = np.sort(prob_unique, axis=0)
            sortedv = sortedv[-top_match:]
            for idx, b in enumerate(best):
                interp_names[ytest].append((cell_name_mappings[b], sortedv[idx]))
                print('{}: {}'.format(cell_name_mappings[b], sortedv[idx]))
                
        return interp_names
    
    
    def estimate_sigma(self, dataset):
        nex = dataset.shape[0]
        dst = []
        for i in range(nex):
            for j in range(i+1, nex):
                dst.append(distance.euclidean(dataset[i,:],dataset[j,:]))
        return np.std(dst)
    