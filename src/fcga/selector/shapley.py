import random
from typing import List

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph


def _build_subgraph(data: Data, node_subset: frozenset) -> Data:
    keep_idx = torch.tensor(sorted(node_subset), dtype=torch.long)
    edge_attr = getattr(data, 'edge_attr', None)
    edge_index, edge_attr = subgraph(
        keep_idx, data.edge_index, edge_attr,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    sub = Data(x=data.x[keep_idx], edge_index=edge_index)
    if edge_attr is not None:
        sub.edge_attr = edge_attr
    return sub


@torch.inference_mode()
def get_shapley_scores(
    dataset,
    graph_idx: int,
    node_subsets: List[frozenset],
    cfg,
    n_samples: int,
) -> List[float]:
    """
    Monte Carlo Shapley value for each candidate coalition S:

        sigma(S) = (1/K) sum_k [ v(S \\cup T_k) - v(T_k) ]

    where T_k is a random subset of V\\S (each node included independently with
    probability 0.5), and v(nodes) is the model's softmax probability for the
    class predicted on the full graph.

    Subgraphs are built and evaluated in chunks of cfg.train.batch_size to
    bound peak GPU memory.

    Args:
        dataset: PyG dataset
        graph_idx: index of the graph to explain
        node_subsets: candidate coalitions from MCTS simulation
        cfg: config object with cfg.injected.model and cfg.device
        n_samples: K, number of Monte Carlo samples per coalition
    """
    model = cfg.injected.model
    device = cfg.device

    graph_data = dataset[graph_idx]
    all_nodes = set(range(graph_data.num_nodes))

    # Predicted class on the full graph defines the target value function.
    model.eval()
    batch_idx = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
    logits_full = model(
        graph_data.x.to(device),
        graph_data.edge_index.to(device),
        batch_idx,
    )
    predicted_class = int(logits_full.argmax(dim=1).item())

    # build all subgraphs: for each (S, k) we need S union T_k and T_k
    # flat list of Data objects; None for empty subgraphs (v=0)
    flat_data: List[Data | None] = []
    # for each subset: list of (flat_idx_with_S, flat_idx_T_k) per sample
    index_pairs: List[List[tuple]] = []

    for subset in node_subsets:
        complement = list(all_nodes - subset)
        pairs = []
        for _ in range(n_samples):
            t_k = frozenset(n for n in complement if random.random() < 0.5)
            s_union_t = subset | t_k

            idx_with_s = len(flat_data)
            flat_data.append(_build_subgraph(graph_data, s_union_t) if s_union_t else None)

            idx_t_k = len(flat_data)
            flat_data.append(_build_subgraph(graph_data, t_k) if t_k else None)

            pairs.append((idx_with_s, idx_t_k))
        index_pairs.append(pairs)

    # chunked forward passes to bound peak GPU memory
    scores = [0.0] * len(flat_data)
    non_empty_idx = [i for i, d in enumerate(flat_data) if d is not None]
    chunk_size = cfg.train.batch_size
    for chunk_start in range(0, len(non_empty_idx), chunk_size):
        chunk_idx = non_empty_idx[chunk_start:chunk_start + chunk_size]
        batch = Batch.from_data_list([flat_data[i] for i in chunk_idx]).to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        probs = F.softmax(logits, dim=1)[:, predicted_class].tolist()
        for orig_i, prob in zip(chunk_idx, probs):
            scores[orig_i] = prob

    # average marginal contributions -> Shapley value per coalition
    return [
        sum(scores[a] - scores[b] for a, b in pairs) / len(pairs)
        for pairs in index_pairs
    ]
