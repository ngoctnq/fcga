from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool


class GNNBase(nn.Module):
    def __init__(
        self,
        input_dim,
        conv_layer_cls,
        output_dim=5,
        hidden_dim=64,
        num_layers=5,
        dropout=0.5,
        **kwargs,
    ):
        '''
        Base GNN model that can handle different types of convolutional layers.

        :param input_dim: Dimension of input node features
        :param conv_layer_cls: Convolutional layer class (e.g., GINConv, GCNConv, etc.)
        :param output_dim: Dimension of output (number of classes for classification)
        :param hidden_dim: Dimension of hidden layers
        :param num_layers: Number of convolutional layers
        :param dropout: Dropout rate
        '''
        super().__init__()

        # ----- pre_mp -----
        self.pre_mp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # ----- GIN layers -----
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(conv_layer_cls(hidden_dim, **kwargs))

        # ----- post_mp -----
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.pre_mp:
            if not isinstance(layer, nn.ReLU):
                layer.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.post_mp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def freeze_base_model(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.post_mp.parameters():
            param.requires_grad = True

    def forward(self, x, edge_index, batch=None):
        # pre-message-passing
        x = self.pre_mp(x)

        # message passing layers
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # graph-level pooling (for graph classification)
        x = global_max_pool(x, batch)

        # post-message-passing head
        x = self.post_mp(x)
        return x
    