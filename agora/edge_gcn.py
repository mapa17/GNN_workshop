import torch
from torch_geometric.nn import MessagePassing

class Edge_GCNConv(MessagePassing):
    def __init__(self,
                 n_features_nodes,
                 n_features_edges,
                 out_channels_node_features,
                 out_channels_message):
        super().__init__(aggr='mean')

        # node update
        self.lin_x = torch.nn.Linear(n_features_nodes,
                                     out_channels_node_features,
                                     bias=False)

        # edge update
        self.lin_e = torch.nn.Linear(n_features_edges + 2 * out_channels_node_features,
                                     out_channels_message,
                                     bias=False)

    def forward(self, x, edge_attr, edge_index):
        x = self.lin_x(x)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)),
                              x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_j:features of the source node for each edge. Shape: [E, out_channels_node_features]
        # x_i:features of the destination node for each edge. Shape: [E, out_channels_node_features]
        # edge_attr: Features of the edges. Shape: [E, n_features_edges]]
        message = torch.cat([x_i, x_j, edge_attr], dim=1)
        message = self.lin_e(message)
        return message

    # The update function. Takes the output of the aggregate function.
    # The arguments in the propagate function can also be accessed here
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels_message]
        x_new = aggr_out
        return x_new