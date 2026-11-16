from torch_geometric.nn import GATConv
from torch.nn import Module
from .base_gnn import GNNBase


class _GATConv(Module):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        num_heads = kwargs.get("num_heads", 8)
        self.conv = GATConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=kwargs.get("dropout", 0.5),
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

    def reset_parameters(self):
        self.conv.reset_parameters()


class GAT(GNNBase):
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
            conv_layer_cls=_GATConv,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
