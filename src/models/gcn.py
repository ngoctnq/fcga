from torch_geometric.nn import GCNConv
from torch.nn import Module
from .base_gnn import GNNBase


class _GCNConv(Module):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        self.conv = GCNConv(
            hidden_dim,
            hidden_dim,
            cached=False,
            **kwargs,
        )
    
    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GCN(GNNBase):
    def __init__(
        self,
        input_dim,
        output_dim=5,
        hidden_dim=64,
        num_layers=5,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            conv_layer_cls=_GCNConv,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
