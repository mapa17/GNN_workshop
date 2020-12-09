from pathlib import Path

import torch
import torch.nn.functional as F

from edge_gcn import Edge_GCNConv
from graphdataset import GraphDataset


class ModelMessage(torch.nn.Module):
    def __init__(self, num_features_nodes, num_features_edges):
        super().__init__()

        self.num_features_nodes = num_features_nodes
        self.num_features_edges = num_features_edges

        # Create a model with edge information via message passing
        self.edge_conv = Edge_GCNConv(
                            n_features_nodes=num_features_nodes,
                            n_features_edges=num_features_edges,
                            out_channels_node_features=2,
                            out_channels_message=2)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer.zero_grad()
        self.optimizer = optimizer

    def forward(self, data: GraphDataset):
        # Apply convolution to node features
        node_with_message = self.edge_conv(x=data.nodes, edge_attr=data.edge_attributes, edge_index=data.edges)

        # Compute log softmax.
        # Note: Negative log likelihood loss expects a log probability
        return F.log_softmax(node_with_message, dim=1)

    def train_one_epoch(self, data: GraphDataset):
        # Set the model.training attribute to True
        self.train()

        # Reset the gradients of all the variables in a model
        self.optimizer.zero_grad()

        # Get the output of the network. The output is a log probability of each
        prediction_log_softmax = self(data)

        # Use only the nodes specified by the train_mask to compute the loss.
        nll_loss = F.nll_loss(prediction_log_softmax[data.trn_mask], data.labels[data.trn_mask],
                              weight=torch.tensor([1, 5], dtype=torch.float))

        # Computes the gradients of all model parameters used to compute the nll_loss
        # Note: These can be listed by looking at model.parameters()
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
        y_true = data.labels[mask]
        acc = y_pred.eq(y_true).sum() / mask.sum()

        return acc.item()

    @torch.no_grad()
    def predict(self, data: GraphDataset):
        self.eval()
        logprob = self(data)
        pred_value, pred_class = logprob.max(dim=1)
        return pred_class

    @torch.no_grad()  # Decorator to deactivate autograd functionality
    def test(self, data: GraphDataset):
        acc_train = self.compute_accuracy(data, data.trn_mask)
        acc_val = self.compute_accuracy(data, data.val_mask)
        return acc_train, acc_val


    @torch.no_grad()
    def save(self, path: Path):
        torch.save({'edge_model_kwargs': {'num_features_nodes': self.num_features_nodes,
                                          'num_features_edges': self.num_features_edges,
                                          },
                    'edge_model': self.state_dict(),
                    'edge_optimizer': self.optimizer.state_dict(),
                    }, path)

    @classmethod
    @torch.no_grad()
    def load(cls, path: Path):
        """
        Create a new model, and load its state from the provided path
        """
        ckp = torch.load(path)
        model_kwargs = ckp['edge_model_kwargs']
        model_state = ckp['edge_model']
        optimizer_state = ckp['edge_optimizer']

        model = cls(**model_kwargs)
        model.load_state_dict(model_state)
        model.optimizer.load_state_dict(optimizer_state)

        return model