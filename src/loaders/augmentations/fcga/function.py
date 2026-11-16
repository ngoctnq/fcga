import torch
from . import BaseFunctionAugmenter


class Augmenter(BaseFunctionAugmenter):
    def __init__(self, cfg):
        super().__init__(cfg)

        # populate embedding path and selector for function-level augmentation
        if self.cfg.dataset.name.startswith("MalNetTinyFeatures"):
            self.embedding_dir = (self.data_root_dir
                / "rewrites"
                / self.cfg.augmentation.params.rewriter
                / "embeddings"
            )

    
    def propagate_information_from_dataset(self, dataset):
        super().propagate_information_from_dataset(dataset)

        if self.node_budget == -1:
            raise NotImplementedError("Node budget must be specified for FCGA.")
        if any(dataset.dup_idxs):
            raise NotImplementedError("Duplication is not supported for FCGA yet.")

        # preallocate augmented cache for function-level augmentation
        self.augmented_cache = torch.empty(
            (len(self.graph_idxs), self.node_budget, self.feature_dim),
            dtype=dataset._data.x.dtype,
        ) # CUDA .to(self.cfg.device)
        self.augmented_cache.share_memory_()


    def regenerate_augmented_embeddings(self):
        super().regenerate_augmented_embeddings()
        
        for i, data_idx in enumerate(self.graph_idxs):
            for j, fn_idx in enumerate(self.topk_nodes[data_idx]): # type: ignore
                f = self.embedding_dir / str(data_idx) / f"{fn_idx}.pth"

                # if feature doesn't exist, run the embedder. it should only be run once by design.
                # NOTE: we recommend running the embedding separately, as to maximize utilization.
                # In this script, the GPU will be hogged with the model and dataloader, and will underutilize GPU for the embedding process as a result.
                if not f.exists():
                    from src.fcga.rewriter.mntf.function.augment import augment_dataset
                    augment_dataset(cfg=self.cfg, dataset=self.cfg.injected.dataset, topk_nodes=self.topk_nodes)
                    
                embedding = torch.load(f, map_location='cpu')
                assert self.feature_dim == embedding.numel(), f"Inconsistent feature dimensions in augmented embeddings: expected {self.feature_dim}, got {embedding.numel()}"
                self.augmented_cache[i, j] = embedding


    def __call__(self, batch):
        super().__call__(batch)

        # CUDA: batch = batch.to(self.cfg.device)

        if self.epoch < min(self.cfg.train.augmentation_start_epoch) or getattr(self, '_selecting', False):
            return batch

        if self.topk_nodes is None:
            self._lazy_load_topk()
            assert self.topk_nodes is not None, "Top-k nodes should be populated after lazy loading."
        new_x = batch.x.clone()

        original_idxs = []
        topk_counts = []
        # calculate the indices of the batch
        for global_idx, x_offset in zip(
            batch.global_idx.tolist(),
            batch.ptr.tolist()
        ):
            for node_idx in self.topk_nodes[global_idx]:
                original_idxs.append(x_offset + node_idx)
            topk_counts.append(len(self.topk_nodes[global_idx]))
        
        topk_counts = torch.tensor(topk_counts)
        assert (topk_counts == torch.clamp(torch.diff(batch.ptr), max=self.node_budget)).all(), \
            f"Top-k counts do not match expected counts based on node count and budget {self.node_budget}"
            
        cached_mask = (torch.arange(self.node_budget)[None, :] < topk_counts[:, None]).flatten()

        is_augment = (torch.rand(len(original_idxs)) < self.ratio).to(new_x.device)
        augment_idxs = torch.tensor(original_idxs)[is_augment].long().to(new_x.device)

        new_x[augment_idxs, :self.feature_dim] = (
            self.augmented_cache[batch.local_idx]
            .reshape(len(cached_mask), -1)[cached_mask][is_augment]
        )

        batch.x = new_x
        return batch
