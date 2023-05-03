import numpy as np

import pandas as pd

import os.path as osp

import torch

import torch.nn.functional as func

from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils import resample

from pathlib import Path

from torch_geometric.data import Data

from torch_geometric.utils import dense_to_sparse

from scipy.sparse import coo_matrix

from utils import *

import torch.nn.functional as func

from sklearn.ensemble import AdaBoostClassifier

from torch_geometric.nn import ChebConv, global_mean_pool

def get_data(dataset = "Training",feature = True):

    healthy = []

    patient = []

    if dataset == "Training" and feature == True:

        for i in range(1,766):

            healthy.append([np.genfromtxt('./newdataset/Health/sub'+str(i)+'/h1160.txt'),np.genfromtxt('./newdataset/Health/sub'+str(i)+'/h1160.txt'),0])

        for j in range(1,207):

            patient.append([np.genfromtxt('./Finaldatasetwithnodefeatures/Patient/REC/sub'+str(j)+'/p1160.txt'),np.genfromtxt('./Finaldatasetwithnodefeatures/Patient/REC/sub'+str(j)+'/p1160.txt'), 1])             

    data = []

    data1 = []

    for i in range(len(healthy)):

        data.append(healthy[i])

    for i in range(len(patient)):

        data1.append(patient[i])

    return data,data1

def create_dataset(data, features):

    dataset_list = []

    label=[]

    for i in range(len(data)):

        y = torch.tensor([data[i][2]])

        #print(y)

        connectivity=data[i][0]

        connectivity1 = features[i][1]

        np.fill_diagonal(connectivity1, 0)

        x = torch.from_numpy(connectivity1).float()

        #print(x)

        #print(connectivity)

        np.fill_diagonal(connectivity, 0)

        adj = compute_KNN_graph(connectivity)

        #np.fill_diagonal(adj, 1)

        #print(adj)

        adj = torch.from_numpy(adj).float()

        edge_index, edge_attr = dense_to_sparse(adj)

        graph_data = Data(x, edge_index=edge_index, edge_attr=edge_attr, y = y)

            #graph_data=Data(x=x, edge_index=edge_index,y=y)

        y=data[i][2]

        print(graph_data)

        dataset_list.append(graph_data)

        label.append(y)

    return dataset_list,label

trainh_data,trainp_data  = get_data("Training", feature=True)

print(len(trainh_data))

print(len(trainp_data))

train_data1 = resample(trainp_data,

           replace=True,

             n_samples=len(trainh_data), random_state=42)

print(len(train_data1))

print(len(trainp_data))

train_data = train_data1 + trainh_data

print("train")

full_dataset,label = create_dataset(train_data, train_data)

print(label)

from torch.nn import Linear

import torch.nn.functional as F

from torch_geometric.nn import ChebConv

from torch_geometric.nn import global_mean_pool, global_max_pool 

import torch.nn.functional as func

class GCN1(torch.nn.Module):

    def __init__(self,

                 num_features =160,

                 num_classes =2,

                 k_order=3,

                 dropout=.3):

        super(GCN1, self).__init__()

        self.p = dropout

        self.conv1 = ChebConv(int(num_features), 128, K=k_order)

        self.conv2 = ChebConv(128, 64, K=k_order)

        self.conv3 = ChebConv(64, 32, K=k_order)

        self.lin1 = torch.nn.Linear(32, int(num_classes))

    def forward(self,data):

        x,edge_index,edge_attr,batch=data.x,data.edge_index,data.edge_attr,data.batch

        x = func.relu(self.conv1(x, edge_index,edge_attr))

        x = func.dropout(x, p=self.p, training=self.training)

        x = func.relu(self.conv2(x, edge_index, edge_attr))

        x = func.dropout(x, p=self.p, training=self.training)

        x = func.relu(self.conv3(x, edge_index,edge_attr))

        x = global_mean_pool(x, batch)

        x = self.lin1(x)

        return x
    
num_node_features =  160
num_classes = 2
class SAGPool(torch.nn.Module):
    def __init__(self, num_layers=6, hidden=64, ratio=0.11):
        super(SAGPool, self).__init__()
        self.conv1 = GraphConv(num_node_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [SAGPooling(hidden, ratio) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, _, _ = pool(x, edge_index,
                                                     batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool 
import torch.nn.functional as func
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.hid = 8
        self.head = 8
        self.conv1 = GATConv(160, self.hid, heads=self.head, dropout=0.3)
        self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head,dropout=0.3)
        self.conv3 = GATConv(self.hid * self.head, self.hid, heads=self.head,dropout=0.3)
        self.lin = nn.Linear(self.hid * self.head, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.cat([x], dim=-1)
        #x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

model = GAT()
print(model)

from sklearn.metrics import roc_auc_score

def GCN_train(loader):

    model.train()

    loss_all = 0

    for data in loader:

        data = data.to(device)

        optimizer.zero_grad()

        output = model(data)

       # print(data.y)

        loss = func.cross_entropy(output, data.y)

        loss.backward()

        loss_all += data.num_graphs * loss.item()

        optimizer.step()

    return loss_all / len(train_dataset)

def GCN_test(loader):

    model.eval()

    pred = []

    label1 = []

    loss_all = 0

    correct=0

    for data in loader:

        data = data.to(device)

        output = model(data)

        loss = func.cross_entropy(output, data.y)

        loss_all += data.num_graphs * loss.item()

       # pred.append(func.softmax(output, dim=1).max(dim=1)[1]) 

        pred.append(output.max(dim=1)[1])

        #correct += pred.eq(data.y).sum().item()

        label1.append(data.y)

    y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()

 #   print(y_pred)

    y_true = torch.cat(label1, dim=0).cpu().detach().numpy()

 #   print(y_true)

    roc_auc = roc_auc_score(y_true, y_pred)

    #print(roc_auc)

    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()

    epoch_sen = tp / (tp + fn)

    epoch_spe = tn / (tn + fp)

    epoch_acc = (tn + tp) / (tn + tp + fn + fp)

  #  print(epoch_sen)

   # print(epoch_spe)

    return epoch_sen, epoch_spe, epoch_acc, loss_all / len(val_dataset),roc_auc

device = torch.device('cpu')

labels =np.array(label)

print(labels)

eval_metrics = np.zeros((5,5))

print(labels)

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (GraphConv, SAGPooling, global_mean_pool,
                                JumpingKnowledge)



class MyEnsemble(nn.Module):
    def __init__(self, GCN1, SAGPool,nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.modelA = GCN1
        self.modelB = SAGPool
        
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()

        # Create new classifier
        self.classifier = nn.Linear(4, 2)
        
    def forward(self, x):
        x1 = self.modelA(x)  # clone to make sure x is not changed by inplace methods
        #print(x1)
        x1 = x1.view(x1.size(0), -1)  
        #print(x1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)  

        #print(x2)
        x = torch.cat((x1, x2), dim=1)
        #print(x)
        #x = self.classifier(F.relu(x))
        #print(x)
        x= (F.softmax(x1, 1) +F.softmax(x2, 1)  )
        
        
        return x

for n_fold in range(5):

    train_val_index1 = np.arange(len(full_dataset))

    train_val,test,_,_=train_test_split(train_val_index1, labels, test_size=0.11, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    #print(train_val)

    #print(test)

    np.array([full_dataset],dtype=object)

    train_val_dataset =[]

    test_dataset = []

    for i in range(len(train_val)):

       # print(train_val[i])

       # print(full_dataset[train_val[i]])

        train_val_dataset.append(full_dataset[train_val[i]])

   # print(train_val_dataset)

    for j in range(len(test)):

        #print(test[j])

        #print(full_dataset[test[j]])

        test_dataset.append(full_dataset[test[j]])

   # print(test_dataset)

    train_val_labels = labels[train_val]

    #print(train_val_labels)

    train_val_index = np.arange(len(train_val_dataset))

    #print(train_val_index)

    train, val, _, _ = train_test_split(train_val_index, train_val_labels, test_size=0.11, shuffle=True)

    train_dataset =[]

    val_dataset = []

    for i in range(len(train)):

       # print(train[i])

       # print(train_val_dataset[train[i]])

        train_dataset.append(train_val_dataset[train[i]])

   # print(train_dataset)

    for j in range(len(val)):

    #    print(val[j])

    #    print(train_val_dataset[val[j]])

        val_dataset.append(train_val_dataset[val[j]])

    #  print(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    min_v_loss = np.inf

    for epoch in range(50):

        t_loss = GCN_train(train_loader)

        val_sen, val_spe, val_acc, v_loss, roc_auc = GCN_test(val_loader)

        test_sen, test_spe, test_acc, _, roc_auc = GCN_test(test_loader)

        if min_v_loss > v_loss:

            min_v_loss = v_loss

            best_val_acc = val_acc

            best_test_sen, best_test_spe, best_test_acc = test_sen, test_spe, test_acc

           # torch.save(model.state_dict(), 'best_model921_%02i.pth' % (n_fold + 1))

            print('CV: {:03d}, Epoch: {:03d}, Val Loss: {:.5f}, Val BAC: {:.5f}, Test BAC: {:.5f}, TEST SEN: {:.5f}, '

                  'TEST SPE: {:.5f}, TEST AUC: {:.5f}'.format(n_fold + 1, epoch + 1, min_v_loss, best_val_acc, best_test_acc,

                                            best_test_sen, best_test_spe, roc_auc))

    eval_metrics[n_fold, 0] = best_test_sen

    eval_metrics[n_fold, 1] = best_test_spe

    eval_metrics[n_fold, 2] = best_test_acc

    eval_metrics[n_fold, 3] = best_val_acc

    eval_metrics[n_fold, 4] = roc_auc

eval_df = pd.DataFrame(eval_metrics)

eval_df.columns = ['SEN', 'SPE', 'TEST ACC','VAL ACC','AUC']

eval_df.index = ['Fold_%02i' % (i + 1) for i in range(5)]

print(eval_df)

print('Average Sensitivity: %.4f±%.4f' % (eval_metrics[:, 0].mean(), eval_metrics[:, 0].std()))

print('Average Specificity: %.4f±%.4f' % (eval_metrics[:, 1].mean(), eval_metrics[:, 1].std()))

print('Average Test Accuracy: %.4f±%.4f' % (eval_metrics[:, 2].mean(), eval_metrics[:, 2].std()))

print('Average Val Accuracy: %.4f±%.4f' % (eval_metrics[:, 3].mean(), eval_metrics[:, 3].std()))

print('Average AUC: %.4f±%.4f' % (eval_metrics[:, 4].mean(), eval_metrics[:, 4].std()))