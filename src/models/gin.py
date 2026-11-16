from torch_geometric.nn import GINConv
from torch.nn import Module, Linear, Sequential, ReLU
from .base_gnn import GNNBase


class _GINConv(Module):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINConv(mlp, **kwargs)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
    
    def reset_parameters(self):
        self.conv.reset_parameters()


class GIN(GNNBase):
    def __init__(
        self,
        input_dim,
        output_dim=5,
        hidden_dim=64,
        num_layers=5,
        dropout=0.5,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            conv_layer_cls=_GINConv,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
