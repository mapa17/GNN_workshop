from typing import Tuple, Optional, Dict, List
from pathlib import Path
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
import torch_geometric.transforms as T

from graphdataset import GraphDataset

from pudb import set_trace as st

class NodeModel(torch.nn.Module):
    def __init__(self, num_features, nClasses, K=1):
        super().__init__()

        self.num_features = num_features
        self.nClasses = nClasses
        self.K = K

        # Create a Simple convolutional layer with K neighbourhood 
        # "averaging" steps
        self.conv = SGConv(in_channels=num_features,
                            out_channels=num_features, 
                           K=K, cached=True, add_self_loops=False)
        fcInputSize = num_features + num_features 
        self.hidden_size = num_features 

        self.fc1 = torch.nn.Linear(fcInputSize, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, nClasses)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad() 
        self.optimizer = optimizer

    def forward(self, data: GraphDataset):
        # Apply convolution to node features
        avg_neighbourhood = self.conv(data.nodes, data.edges)

        # Combine local node information and aggregated neighbourhood information
        local_and_neighbourhood = torch.cat([data.nodes, avg_neighbourhood], axis=1)

        # Run it through an MLP
        z = self.fc1(local_and_neighbourhood)
        z = self.relu(z)
        x = self.fc2(z)

        # Compute log softmax.
        # Note: Negative log likelihood loss expects a log probability
        return F.log_softmax(x, dim=1) 
    
    def train_one_epoch(self, data: GraphDataset):
        # Set the model.training attribute to True
        self.train() 

        # Reset the gradients of all the variables in a model
        self.optimizer.zero_grad() 

        # Get the output of the network. The output is a log probability of each
        prediction_log_softmax = self(data) 

        # Use only the nodes specified by the train_mask to compute the loss.
        nll_loss = F.nll_loss(prediction_log_softmax[data.trn_mask], data.labels[data.trn_mask], weight=torch.tensor([1, 5], dtype=torch.float))
        
        #Computes the gradients of all model parameters used to compute the nll_loss
        #Note: These can be listed by looking at model.parameters()
        nll_loss.backward()

        # Finally, the optimizer looks at the gradients of the parameters 
        # and updates the parameters with the goal of minimizing the loss.
        self.optimizer.step()

        trn_loss = float(nll_loss.detach())
        with torch.no_grad():
            val_loss = float(F.nll_loss(prediction_log_softmax[data.val_mask], data.labels[data.val_mask]))
        
        return trn_loss, val_loss

    def compute_accuracy(self, data: GraphDataset, mask):
        # Set the model.training attribute to False
        self.eval()

        logprob = self(data)
        _, y_pred = logprob[mask].max(dim=1)
        y_true=data.labels[mask]
        acc = y_pred.eq(y_true).sum() / mask.sum()

        return acc.item()

    @torch.no_grad()
    def predict(self, data: GraphDataset):
        self.eval()
        logprob = self(data)
        pred_value, pred_class = logprob.max(dim=1)
        return pred_class
 
    
    @torch.no_grad() # Decorator to deactivate autograd functionality  
    def test(self, data: GraphDataset):
        acc_train = self.compute_accuracy(data, data.trn_mask)
        acc_val = self.compute_accuracy(data, data.val_mask)
        return acc_train, acc_val


    @torch.no_grad()
    def save(self, path: Path):
        torch.save({'node_model_kwargs': {'num_features': self.num_features,
                'nClasses': self.nClasses,
                'K': self.K,
                },
            'node_model': self.state_dict(),
            'node_optimizer': self.optimizer.state_dict(),
        }, path)

    @classmethod
    @torch.no_grad()
    def load(cls, path: Path):
        """
        Create a new model, and load its state from the provided path
        """
        ckp = torch.load(path)
        model_kwargs = ckp['node_model_kwargs']
        model_state = ckp['node_model']
        optimizer_state = ckp['node_optimizer']

        model = cls(**model_kwargs)
        model.load_state_dict(model_state)
        model.optimizer.load_state_dict(optimizer_state)
    
        return model
    


