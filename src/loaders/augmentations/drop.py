import json
import torch
import networkx as nx
from pathlib import Path
from typing import Optional
from torch_geometric.utils import subgraph, to_networkx
from src.fcga.selector.greedy import load_node_values_from_cache, save_node_values_to_cache


def drop_nodes(data, drop_rate=0.1):
    num_nodes = data.num_nodes
    keep_mask = torch.rand(num_nodes) >= drop_rate
    if keep_mask.sum() == 0:
        keep_mask[torch.randint(num_nodes, (1,))] = True

    keep_idx = keep_mask.nonzero(as_tuple=False).view(-1)
    edge_index, edge_attr = subgraph(
        keep_idx, data.edge_index, data.edge_attr,
        relabel_nodes=True, num_nodes=num_nodes,
    )
    new_data = data.clone()
    new_data.x = data.x[keep_idx]
    new_data.edge_index = edge_index
    if edge_attr is not None:
        new_data.edge_attr = edge_attr
    return new_data


def _load_edge_betweenness(
    cache_dir: Path,
    graph_idxs: list[int],
) -> Optional[dict[int, dict[tuple, float]]]:
    result = {}
    for graph_idx in graph_idxs:
        cache_path = cache_dir / f"{graph_idx}.json"
        if not cache_path.exists():
            return None
        with open(cache_path) as f:
            raw = json.load(f)
        result[graph_idx] = {
            tuple(int(x) for x in k.split(",")): v for k, v in raw.items()
        }
    return result


def _save_edge_betweenness(data: dict[int, dict[tuple, float]], cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for graph_idx, betweenness in data.items():
        cache_path = cache_dir / f"{graph_idx}.json"
        with open(cache_path, "w") as f:
            json.dump({f"{u},{v}": b for (u, v), b in betweenness.items()}, f)


def _compute_edge_betweenness(
    dataset,
    graph_idxs: list[int],
) -> dict[int, dict[tuple, float]]:
    result = {}
    for graph_idx in graph_idxs:
        graph_nx = to_networkx(dataset[graph_idx])
        result[graph_idx] = nx.edge_betweenness_centrality(graph_nx)
    return result


def load_or_compute_edge_betweenness(
    dataset,
    graph_idxs: list[int],
    cache_dir: Path,
) -> dict[int, dict[tuple, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [idx for idx in graph_idxs if not (cache_dir / f"{idx}.json").exists()]
    if missing:
        new_values = _compute_edge_betweenness(dataset, missing)
        _save_edge_betweenness(new_values, cache_dir)
    return _load_edge_betweenness(cache_dir, graph_idxs) # type: ignore


_SCORED_CRITERIA = {"degree", "pagerank", "in_degree", "out_degree", "betweenness", "closeness", "ppr"}


class NodeDrop:
    def __init__(self, cfg):
        self.cfg = cfg
        self.drop_rate = cfg.augmentation.params.node_drop_rate
        self.protected_drop_rate = cfg.augmentation.params.node_protected_drop_rate
        self.criteria = cfg.augmentation.params.node_criteria
        self.epoch = None
        self.node_scores = None
        self.node_scores_tensors = None

        if self.criteria is not None and self.criteria != "degree":
            raise NotImplementedError(f"NodeDrop only supports node_criteria='degree', got {self.criteria!r}")

        if not cfg.dataset.name.startswith("MalNetTinyFeatures"):
            raise NotImplementedError(f"NodeDrop not supported for dataset: {cfg.dataset.name}")

        self.cache_dir = (
            Path(cfg.injected.base_dir)
            / "datasets" / "augmented" / "mntf"
            / cfg.dataset.name
            / cfg.dataset.params.variant
            / "selection" / self.criteria
        ) if self.criteria in _SCORED_CRITERIA else None

    def propagate_information_from_dataset(self, dataset):
        if self.cache_dir is None:
            return
        assert self.criteria is not None
        graph_idxs = list(dataset.indices())
        cached = load_node_values_from_cache(self.cache_dir, graph_idxs)
        if cached is not None:
            self.node_scores = cached
        else:
            from src.fcga.selector.degree import get_node_values
            self.node_scores = get_node_values(self.cfg.injected.dataset, graph_idxs=graph_idxs, criteria=self.criteria)
            save_node_values_to_cache(self.node_scores, self.cache_dir)

        self.node_scores_tensors = {}
        for graph_idx, scores_dict in self.node_scores.items():
            if not scores_dict:
                continue
            arr = torch.zeros(max(scores_dict) + 1, dtype=torch.float32)
            for node_idx, score in scores_dict.items():
                arr[node_idx] = score
            m = arr.max().item()
            if m > 0:
                arr /= m
            self.node_scores_tensors[graph_idx] = arr

    def set_epoch(self, epoch):
        self.epoch = epoch
        (self.cfg.injected.temp_dir / "epoch.txt").write_text(str(epoch))

    def __call__(self, batch):
        if self.epoch is None:
            self.epoch = int((self.cfg.injected.temp_dir / "epoch.txt").read_text())

        if self.epoch < min(self.cfg.train.augmentation_start_epoch):
            return batch

        num_nodes = batch.x.size(0)
        drop_probs = torch.full((num_nodes,), self.drop_rate)
        ptr = batch.ptr.tolist()

        if self.node_scores_tensors is not None:
            for i, global_idx in enumerate(batch.global_idx.tolist()):
                scores = self.node_scores_tensors.get(global_idx)
                if scores is None:
                    continue
                x_start, x_end = ptr[i], ptr[i + 1]
                n = min(len(scores), x_end - x_start)
                drop_probs[x_start:x_start + n] = (
                    self.protected_drop_rate
                    + (self.drop_rate - self.protected_drop_rate) * (1 - scores[:n])
                )

        drop_mask = torch.rand(num_nodes) < drop_probs

        for graph_i in range(batch.num_graphs):
            start, end = ptr[graph_i], ptr[graph_i + 1]
            if drop_mask[start:end].all():
                drop_mask[torch.randint(end - start, (1,)).item() + start] = False

        keep_idx = (~drop_mask).nonzero(as_tuple=False).view(-1)
        edge_index, edge_attr = subgraph(
            keep_idx, batch.edge_index, batch.edge_attr,
            relabel_nodes=True, num_nodes=num_nodes,
        )

        batch.x = batch.x[keep_idx]
        batch.edge_index = edge_index
        if edge_attr is not None:
            batch.edge_attr = edge_attr
        batch.batch = batch.batch[keep_idx]
        batch.ptr = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.bincount(batch.batch, minlength=batch.num_graphs).cumsum(0),
        ])

        return batch


class EdgeDrop:
    def __init__(self, cfg):
        self.cfg = cfg
        self.drop_rate = cfg.augmentation.params.edge_drop_rate
        self.protected_drop_rate = cfg.augmentation.params.edge_protected_drop_rate
        self.use_betweenness = cfg.augmentation.params.edge_criteria == "betweenness"
        self.epoch = None
        self.edge_betweenness = None

        edge_criteria = cfg.augmentation.params.edge_criteria
        if edge_criteria is not None and edge_criteria != "betweenness":
            raise NotImplementedError(f"EdgeDrop only supports edge_criteria='betweenness', got {edge_criteria!r}")

        if not cfg.dataset.name.startswith("MalNetTinyFeatures"):
            raise NotImplementedError(f"EdgeDrop not supported for dataset: {cfg.dataset.name}")

        self.cache_dir = (
            Path(cfg.injected.base_dir)
            / "datasets" / "augmented" / "mntf"
            / cfg.dataset.name
            / cfg.dataset.params.variant
            / "selection" / "edge-betweenness"
        )

    def propagate_information_from_dataset(self, dataset):
        if not self.use_betweenness:
            return
        graph_idxs = list(dataset.indices())
        self.edge_betweenness = load_or_compute_edge_betweenness(
            self.cfg.injected.dataset, graph_idxs, self.cache_dir,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch
        (self.cfg.injected.temp_dir / "epoch.txt").write_text(str(epoch))

    def __call__(self, batch):
        if self.epoch is None:
            self.epoch = int((self.cfg.injected.temp_dir / "epoch.txt").read_text())

        if self.epoch < min(self.cfg.train.augmentation_start_epoch):
            return batch

        num_edges = batch.edge_index.size(1)
        drop_probs = torch.full((num_edges,), self.drop_rate)

        if self.use_betweenness:
            assert self.edge_betweenness is not None, \
                "Edge betweenness not loaded. Call propagate_information_from_dataset() first."

            edge_graph = batch.batch[batch.edge_index[0]]
            edge_ptr = torch.cat([
                torch.zeros(1, dtype=torch.long),
                torch.bincount(edge_graph, minlength=batch.num_graphs).cumsum(0),
            ]).tolist()
            ptr = batch.ptr.tolist()

            for graph_i in range(batch.num_graphs):
                e_start, e_end = edge_ptr[graph_i], edge_ptr[graph_i + 1]
                if e_start == e_end:
                    continue

                global_idx = batch.global_idx[graph_i].item()
                b_dict = self.edge_betweenness[global_idx]
                node_offset = ptr[graph_i]
                local_edges = (batch.edge_index[:, e_start:e_end] - node_offset).T.tolist()
                b_values = [b_dict.get((u, v), 0.0) for u, v in local_edges]

                b_tensor = torch.tensor(b_values, dtype=torch.float32)
                b_max = b_tensor.max()
                b_norm = b_tensor / b_max if b_max > 0 else torch.zeros_like(b_tensor)

                drop_probs[e_start:e_end] = (
                    self.protected_drop_rate
                    + (self.drop_rate - self.protected_drop_rate) * (1 - b_norm)
                )

        keep_mask = torch.rand(num_edges) >= drop_probs
        batch.edge_index = batch.edge_index[:, keep_mask]
        if batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr[keep_mask]

        return batch


class CombinedDrop:
    """Edge drop followed by node drop. Edge first so node pruning doesn't orphan kept edges."""
    def __init__(self, edge_drop: EdgeDrop, node_drop: NodeDrop):
        self.edge_drop = edge_drop
        self.node_drop = node_drop

    def propagate_information_from_dataset(self, dataset):
        self.edge_drop.propagate_information_from_dataset(dataset)
        self.node_drop.propagate_information_from_dataset(dataset)

    def set_epoch(self, epoch):
        self.edge_drop.set_epoch(epoch)
        self.node_drop.set_epoch(epoch)

    def __call__(self, batch):
        batch = self.edge_drop(batch)
        batch = self.node_drop(batch)
        return batch
