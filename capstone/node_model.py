import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import SGConv
import torch_geometric.transforms as T

class NodeModel(torch.nn.Module):
    def __init__(self, data, K=1):
        super().__init__()

        self.dataset = data
        # Until we have not target class, select 10% random positive nodes
        self.y = torch.tensor(
                np.random.choice([0, 1], p=[0.9, 0.1], size=data.x.shape[0]),
                dtype=torch.long)

        self.train_mask = np.random.choice([True, False], p=[0.8, 0.2], replace=True, size=data.x.shape[0])
        self.validation_mask = np.invert(self.train_mask)

        # Probability for node beeing normal/abnormal
        num_classes = 2 

        # Create a Simple convolutional layer with K neighbourhood 
        # "averaging" steps
        self.conv = SGConv(in_channels=data.num_features,
                            out_channels=num_classes, 
                           K=K, cached=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.2)
        optimizer.zero_grad() 
        self.optimizer = optimizer

    def forward(self, data):
        # Apply convolution to node features
        x = self.conv(data.x, data.edge_index)

        # Compute log softmax.
        # Note: Negative log likelihood loss expects a log probability
        return F.log_softmax(x, dim=1) 
    
    def train_one_epoch(self):
        # Set the model.training attribute to True
        self.train() 

        # Reset the gradients of all the variables in a model
        self.optimizer.zero_grad() 

        # Get the output of the network. The output is a log probability of each
        prediction_log_softmax = self(self.dataset) 

        labels = self.y # Labels of each node

        # Use only the nodes specified by the train_mask to compute the loss.
        nll_loss = F.nll_loss(prediction_log_softmax[self.train_mask], labels[self.train_mask])
        
        #Computes the gradients of all model parameters used to compute the nll_loss
        #Note: These can be listed by looking at model.parameters()
        nll_loss.backward()

        # Finally, the optimizer looks at the gradients of the parameters 
        # and updates the parameters with the goal of minimizing the loss.
        self.optimizer.step()


    def compute_accuracy(self, mask):
        # Set the model.training attribute to False
        self.eval()

        logprob = self(self.dataset)
        _, y_pred = logprob[mask].max(dim=1)
        y_true=self.y[mask]
        acc = y_pred.eq(y_true).sum()/ mask.sum()

        return acc.item()
    
    @torch.no_grad() # Decorator to deactivate autograd functionality  
    def test(self):
        acc_train = self.compute_accuracy(self.train_mask)
        acc_val = self.compute_accuracy(self.validation_mask)

        return acc_train, acc_val

