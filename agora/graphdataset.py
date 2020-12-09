from typing import Tuple, Optional, Dict, List
import numpy as np

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.data import Data


class GraphDataset():
    def __init__(self,
        data: torch_geometric.data.Data,
        training: bool,
        labels: Optional[List] = None,
        training_split = 0.8):

        self.data = data
        if labels is None: 
            self.training_labels = torch.tensor([])
        else:
            self.training_labels = torch.tensor(labels, dtype=torch.long) # Labels of each node

        if training:
            self.training_mask = np.random.choice([True, False], p=[training_split, 1.0-training_split], replace=True, size=data.x.shape[0])
            self.validation_mask = np.invert(self.training_mask)
        else:
            self.training_mask = [True]*data.x.shape[0]
            self.validation_mask = [True]*data.x.shape[0]

    @property 
    def num_features(self):
        return self.data.num_features

    @property
    def nodes(self):
        return self.data.x
    
    @property
    def edges(self):
        return self.data.edge_index

    @property
    def edge_attributes(self):
        return self.data.edge_attr
    
    @property
    def labels(self) -> Tensor:
        return self.training_labels

    @property
    def trn_mask(self) -> List[bool]:
        return self.training_mask
    
    @property
    def val_mask(self) -> List[bool]:
        return self.validation_mask

