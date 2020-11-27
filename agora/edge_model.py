from typing import Tuple, Dict, List

import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score
#from pudb import set_trace as st

class EdgeModel(torch.nn.Module):
    def __init__(self, data, zDim=4, hiddenDim=20, device='cpu'):
        super().__init__()

        self.device = device

        self.data = train_test_split_edges(data, val_ratio=0.25, test_ratio=0.0,)

        self.conv1 = SGConv(self.data.num_features, hiddenDim)
        self.conv2 = SGConv(hiddenDim, zDim)

        optimizer = torch.optim.Adam(params=self.parameters(), lr=0.01,weight_decay=0.001)
        optimizer.zero_grad() 
        self.optimizer = optimizer

    def forward(self, data, edge_index):
        x = self.conv1(data.x, edge_index)
        x = self.conv2(x, edge_index)
        return x

    @staticmethod
    def _decode(z, edge_index):
        # Get features of the source nodes of the edge list (|E| X |F|)
        z_j = z[edge_index[0],:]

        # Get features of the target nodes of the edge list (|E| X |F|)
        z_i = z[edge_index[1],:]

        # get dot product by element wise multiplication followed by the sum
        # (|E| X |E| -> |E| X |F| -> |E|)
        dot_product = torch.sum(z_i*z_j, axis=1)
        return dot_product

    def _get_edge_loss(self, z, neg_edge_index):
        pos_edge_index = self.data.train_pos_edge_index
        
        # Inner product of node representations for positive edges
        logits_pos = self._decode(z,pos_edge_index)

        # Inner product of node representations for negative edges
        logits_neg = self._decode(z,neg_edge_index)

        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones(pos_edge_index.shape[1])).to(self.device)
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros(neg_edge_index.shape[1])).to(self.device)

        loss = (loss_pos + loss_neg)/2.
    
        return loss

    def train_one_epoch(self):
        # Set the model.training attribute to True
        self.train() 

        # Reset the gradients of all the variables in a model
        self.optimizer.zero_grad() 

        # One forward pass on the positive edges (learning node embeddings)
        z = self(self.data, self.data.train_pos_edge_index)

        # Create a negative sample
        neg_edge_index = negative_sampling(edge_index=self.data.train_pos_edge_index,
                                            num_nodes=self.data.x.size(0))

        # Calculate the prediction error (how many of the positive and negative edges have been predicted) 
        loss = self._get_edge_loss(z, neg_edge_index)
        loss.backward()
        self.optimizer.step()

        acc_val = self.compute_accuracy('val')
        return loss.item(), acc_val


    @torch.no_grad() # Decorator to deactivate autograd functionality  
    def compute_accuracy(self, testdata='val', return_prediction=False):
        self.eval()
        z = self(self.data, self.data.train_pos_edge_index)

        if testdata == "val":
            pos_edge_index = self.data.val_pos_edge_index
            neg_edge_index = self.data.val_neg_edge_index
        elif testdata =="test":
            pos_edge_index = self.data.test_pos_edge_index
            neg_edge_index = self.data.test_neg_edge_index
        else:
            raise ValueError(f'Invalid selection for testdata={testdata}, can be either val or test')

        # Inner product of node representations for positive edges
        logits_pos = self._decode(z,pos_edge_index)

        # Inner product of node representations for negative edges
        logits_neg = self._decode(z,neg_edge_index)

        edge_pos_pred = torch.sigmoid(logits_pos)
        edge_neg_pred = torch.sigmoid(logits_neg)
        edge_pred = torch.cat([edge_pos_pred, edge_neg_pred])
        edge_true = torch.cat([torch.ones_like(edge_pos_pred), torch.zeros_like(edge_neg_pred)])
        edge_true = edge_true.cpu().numpy()
        edge_pred = edge_pred.cpu().numpy()
        score = roc_auc_score(edge_true, edge_pred)
        if return_prediction:
            return score, logits_pos
        else:
            return score 

    
    @torch.no_grad() # Decorator to deactivate autograd functionality  
    def test(self):
        acc_val, pred_val = self.compute_accuracy('val', return_prediction=True)
        return acc_val, pred_val
        #acc_test, pred_test = self.compute_accuracy('test', return_prediction=True)
        #return acc_val, pred_val, acc_test, pred_test

