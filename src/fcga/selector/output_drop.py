import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph as pyg_subgraph
from typing import List


def _build_dropped_graph(data: Data, drop_nodes: frozenset) -> Data | None:
    keep_set = set(range(data.num_nodes)) - drop_nodes
    if not keep_set:
        return None
    keep_idx = torch.tensor(sorted(keep_set), dtype=torch.long)
    edge_attr = getattr(data, 'edge_attr', None)
    edge_index, edge_attr = pyg_subgraph(
        keep_idx, data.edge_index, edge_attr,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    sub = Data(x=data.x[keep_idx], edge_index=edge_index)
    if edge_attr is not None:
        sub.edge_attr = edge_attr
    return sub


@torch.inference_mode()
def get_output_drop_scores(
    dataset,
    graph_idx: int,
    node_subsets: List[frozenset],
    cfg,
) -> List[float]:
    """
    Output-drop reward for MCTS.

    reward = baseline_loss - loss_after_drop

    where both losses use the model's predicted class on the full graph as the
    target. Maximizing this reward favors subsets whose removal causes the
    smallest loss increase.
    """
    model = cfg.injected.model
    device = cfg.device

    graph_data = dataset[graph_idx]

    model.eval()
    batch_idx_full = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
    logits_full = model(
        graph_data.x.to(device),
        graph_data.edge_index.to(device),
        batch_idx_full,
    )
    predicted_class = int(logits_full.argmax(dim=1).item())
    target = torch.tensor([predicted_class], device=device)
    baseline_loss = F.cross_entropy(logits_full, target).item()

    flat_data: list[Data | None] = [_build_dropped_graph(graph_data, s) for s in node_subsets]

    # default: empty graph -> reward = 0 (baseline - baseline)
    drop_losses = [baseline_loss] * len(flat_data)
    non_empty_idx = [i for i, d in enumerate(flat_data) if d is not None]

    if non_empty_idx:
        batch = Batch.from_data_list([flat_data[i] for i in non_empty_idx]).to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        targets_batch = torch.full((len(non_empty_idx),), predicted_class, dtype=torch.long, device=device)
        per_loss = F.cross_entropy(logits, targets_batch, reduction='none').tolist()
        for orig_i, loss in zip(non_empty_idx, per_loss):
            drop_losses[orig_i] = loss

    return [baseline_loss - dl for dl in drop_losses]
