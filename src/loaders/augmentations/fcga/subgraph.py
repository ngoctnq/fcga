import json
import torch
from torch_geometric.data import Data, Batch

from . import BaseFunctionAugmenter


class Augmenter(BaseFunctionAugmenter):
    def __init__(self, cfg):
        super().__init__(cfg)

        if self.cfg.dataset.name.startswith("MalNetTinyFeatures"):
            self.refactor_dir = (
                self.data_root_dir / "refactor" / cfg.augmentation.params.rewriter
            )

    def propagate_information_from_dataset(self, dataset):
        super().propagate_information_from_dataset(dataset)

        if self.node_budget == -1:
            raise NotImplementedError("Node budget must be specified for subgraph FCGA.")
        if any(dataset.dup_idxs):
            raise NotImplementedError("Duplication is not supported for subgraph FCGA yet.")

        self.ldp_transform = dataset.transform
        self.augmented_cache = {}  # global_idx -> {'x': tensor, 'edge_index': tensor}

    def regenerate_augmented_embeddings(self):
        super().regenerate_augmented_embeddings()

        needs_pipeline = any(
            not (
                self.refactor_dir
                / str(data_idx)
                / "_".join(str(n) for n in sorted(self.topk_nodes[data_idx]))  # type: ignore
                / "graph.json"
            ).exists()
            for data_idx in self.graph_idxs
        )
        if needs_pipeline:
            from src.fcga.rewriter.mntf.subgraph.augment import augment_dataset
            augment_dataset(cfg=self.cfg, dataset=self.cfg.injected.dataset,
                            cache_dir=self._get_persistent_cache_dir(self.epoch))

        for data_idx in self.graph_idxs:
            graph_topk = self.topk_nodes[data_idx]  # type: ignore
            node_set_str = "_".join(str(n) for n in sorted(graph_topk))
            out_dir = self.refactor_dir / str(data_idx) / node_set_str

            if not (out_dir / "graph.json").exists():
                continue  # skipped during augmentation pipeline
            with open(out_dir / "graph.json") as f:
                graph_data = json.load(f)

            original_x = self.cfg.injected.dataset.get(data_idx).x  # pre-LDP
            num_nodes = len(graph_data["nodes"])
            x = torch.zeros(num_nodes, self.feature_dim, dtype=original_x.dtype)

            for node in graph_data["nodes"]:
                fn_idx = node["id"]
                prev = node.get("prev_node_idx", -1)
                if prev >= 0:
                    x[fn_idx] = original_x[prev]
                else:
                    embedding = torch.load(
                        out_dir / "embeddings" / f"{fn_idx}.pth",
                        map_location="cpu",
                    )
                    assert embedding.numel() == self.feature_dim, (
                        f"Embedding dim mismatch for node {fn_idx} in graph {data_idx}: "
                        f"expected {self.feature_dim}, got {embedding.numel()}"
                    )
                    x[fn_idx] = embedding.reshape(self.feature_dim)

            raw_edges = graph_data.get("edges", [])
            if raw_edges:
                src = [e["source"] for e in raw_edges]
                dst = [e["target"] for e in raw_edges]
                edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)

            aug_data = Data(x=x, edge_index=edge_index)
            if self.ldp_transform is not None:
                aug_data = self.ldp_transform(aug_data)

            new_x = aug_data.x
            new_ei = aug_data.edge_index
            assert new_x is not None and new_ei is not None, f"Augmented data for graph {data_idx} is missing x or edge_index."

            if data_idx in self.augmented_cache:
                cached = self.augmented_cache[data_idx]
                if cached["x"].shape == new_x.shape and cached["edge_index"].shape == new_ei.shape:
                    cached["x"].copy_(new_x)
                    cached["edge_index"].copy_(new_ei)
                else:
                    self.augmented_cache[data_idx] = {
                        "x": new_x.share_memory_(),
                        "edge_index": new_ei.share_memory_(),
                    }
            else:
                self.augmented_cache[data_idx] = {
                    "x": new_x.share_memory_(),
                    "edge_index": new_ei.share_memory_(),
                }

    def __call__(self, batch):
        super().__call__(batch)

        if self.epoch < min(self.cfg.train.augmentation_start_epoch):
            return batch

        if self.topk_nodes is None:
            self._lazy_load_topk()

        global_idxs = batch.global_idx.tolist()
        augment_mask = torch.rand(len(global_idxs)) < self.ratio

        if not augment_mask.any():
            return batch

        data_list = batch.to_data_list()
        for i, (gidx, do_augment) in enumerate(zip(global_idxs, augment_mask.tolist())):
            if not do_augment:
                continue
            if gidx not in self.augmented_cache:
                continue  # no augmented version available (skipped during pipeline)
            cached = self.augmented_cache[gidx]
            aug = Data(
                x=cached["x"],
                edge_index=cached["edge_index"],
                y=data_list[i].y,
                num_nodes=cached["x"].shape[0],
            )
            aug.global_idx = data_list[i].global_idx
            aug.local_idx = data_list[i].local_idx
            aug.dup_idx = data_list[i].dup_idx
            data_list[i] = aug

        return Batch.from_data_list(data_list)
