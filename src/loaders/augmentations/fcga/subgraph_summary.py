from pathlib import Path

import torch

from . import BaseFunctionAugmenter


class Augmenter(BaseFunctionAugmenter):
    """Replace an MCTS-selected subgraph's node features with cached summaries.

    Unlike full FCGA-S, this augmentation retains the original graph topology.
    All selected nodes in a sampled graph are replaced together.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.summary_dir = self.data_root_dir / "summaries"
        self.embedding_dir = self.data_root_dir / "summary_embeddings"

    def propagate_information_from_dataset(self, dataset):
        super().propagate_information_from_dataset(dataset)

        if self.node_budget == -1:
            raise NotImplementedError(
                "Node budget must be specified for summary-only FCGA-S."
            )
        if any(dataset.dup_idxs):
            raise NotImplementedError(
                "Duplication is not supported for summary-only FCGA-S yet."
            )

        self.augmented_cache = torch.empty(
            (len(self.graph_idxs), self.node_budget, self.feature_dim),
            dtype=dataset._data.x.dtype,
        )
        self.augmented_cache.share_memory_()

    def regenerate_augmented_embeddings(self):
        super().regenerate_augmented_embeddings()
        assert self.topk_nodes is not None

        embedding_paths = {
            (data_idx, node_idx): (self.embedding_dir / str(data_idx) / f"{node_idx}.pth")
            for data_idx in self.graph_idxs
            for node_idx in self.topk_nodes[data_idx]
        }

        missing = [path for path in embedding_paths.values() if not path.exists()]

        if missing:
            from src.fcga.rewriter.mntf.subgraph_summary.augment import (
                augment_dataset,
            )

            augment_dataset(
                cfg=self.cfg,
                dataset=self.cfg.injected.dataset,
                topk_nodes=self.topk_nodes,
            )

            still_missing = [path for path in missing if not path.exists()]
            if still_missing:
                raise RuntimeError(
                    "On-demand summary embedding did not populate "
                    f"{len(still_missing)} expected cache file(s)."
                )

        for local_idx, data_idx in enumerate(self.graph_idxs):
            for selected_idx, node_idx in enumerate(self.topk_nodes[data_idx]):
                embedding = torch.load(
                    embedding_paths[(data_idx, node_idx)],
                    map_location="cpu",
                    weights_only=True,
                )
                if embedding.numel() != self.feature_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for graph {data_idx}, "
                        f"node {node_idx}: expected {self.feature_dim}, "
                        f"got {embedding.numel()}"
                    )
                self.augmented_cache[local_idx, selected_idx] = embedding.reshape(
                    self.feature_dim
                )

    def __call__(self, batch):
        super().__call__(batch)

        if (
            self.epoch < min(self.cfg.train.augmentation_start_epoch)
            or getattr(self, "_selecting", False)
        ):
            return batch

        if self.topk_nodes is None:
            self._lazy_load_topk()
            assert self.topk_nodes is not None

        original_idxs: list[int] = []
        topk_counts: list[int] = []
        for global_idx, x_offset in zip(
            batch.global_idx.tolist(), batch.ptr[:-1].tolist()
        ):
            selected = self.topk_nodes[global_idx]
            original_idxs.extend(x_offset + node_idx for node_idx in selected)
            topk_counts.append(len(selected))

        counts = torch.tensor(topk_counts, dtype=torch.long)
        expected_counts = torch.clamp(torch.diff(batch.ptr).cpu(), max=self.node_budget)
        if not torch.equal(counts, expected_counts):
            raise ValueError(
                "Selected-node counts do not match the configured node budget "
                "and graph sizes."
            )

        cached_mask = (
            torch.arange(self.node_budget)[None, :] < counts[:, None]
        ).flatten()

        # FCGA-S samples once per graph. Every selected node in an augmented
        # graph receives its cached summary embedding together.
        graph_mask = torch.rand(len(topk_counts)) < self.ratio
        node_mask = torch.repeat_interleave(graph_mask, counts)
        if not node_mask.any():
            return batch

        new_x = batch.x.clone()
        augment_idxs = torch.tensor(original_idxs, dtype=torch.long)[node_mask]
        replacement = (
            self.augmented_cache[batch.local_idx]
            .reshape(len(cached_mask), -1)[cached_mask][node_mask]
        )
        new_x[augment_idxs, : self.feature_dim] = replacement
        batch.x = new_x
        return batch
