import torch
from .fcga import BaseFunctionAugmenter


@torch.no_grad()
def augment_cosine(target_vector, min_cosine=0.5, random_vector=None):
    '''
    Augment the target batch of vectors by adding a scaled random batch of vectors such that the result has a cosine similarity above the given threshold.
    
    :param target_vector: The original vector to be augmented
    :type target_vector: torch.Tensor of shape (batch_size, feature_dim)
    :param min_cosine: The minimum cosine similarity threshold for the augmented vector, between -1 and 1
    :type min_cosine: float
    :param random_vector: Optional pre-generated random vector to be added to the target vector. If not provided, a random vector will be generated internally.
    :type random_vector: torch.Tensor of shape (batch_size, feature_dim), optional
    :return: The additive noise vector that after summing up will have a cosine similarity above the given threshold with the target vector
    :rtype: torch.Tensor of shape (batch_size, feature_dim)
    '''
    if random_vector is None:
        random_vector = torch.randn_like(target_vector)

    # flip random vector if it has negative cosine similarity with target vector
    sign_vector = (
        # use comparison instead of .sign() to avoid division by 0 when target_vector and random_vector are orthogonal
        (target_vector * random_vector).sum(dim=1, keepdim=True) >= 0
    ).to(random_vector.dtype) * 2 - 1
    random_vector = random_vector * sign_vector

    # after this, coefficient is always positive and decreasing with cosine similarity
    # first random vector should be able to overshoot 1
    normalized_random_vector = (
        random_vector
        / random_vector.norm(dim=1, keepdim=True)
        * target_vector.norm(dim=1, keepdim=True)
    )

    cosine_sim = (target_vector * normalized_random_vector).sum(
        dim=1, keepdim=True
    ) / (target_vector.norm(dim=1, keepdim=True) ** 2)

    satisfied = cosine_sim >= min_cosine
    # if additive noise is within the cone, the sum will even be closer
    if torch.all(satisfied):
        return normalized_random_vector
    
    # calculate the required coefficient to satisfy the cosine similarity
    coeff = (
        (1 - min_cosine ** 2) * cosine_sim + 
        min_cosine * torch.sqrt((1 - min_cosine ** 2) * (1 - cosine_sim ** 2))
    ) / (min_cosine ** 2 - cosine_sim ** 2 + 1e-8)  # for stability

    # both random_vector and normalized works when satisfied=1, so pick your poison
    new_random_vector = torch.where(
        satisfied,
        normalized_random_vector,
        normalized_random_vector * coeff
    )

    return new_random_vector


class Augmenter(BaseFunctionAugmenter):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.budget = cfg.augmentation.params.cosine_threshold
        self.reaugment = cfg.augmentation.params.reaugment
        self.seed = cfg.seed

    
    def propagate_information_from_dataset(self, dataset):
        super().propagate_information_from_dataset(dataset)
        
        # preaugment allocation
        if self.reaugment:
            self.augmented_cache = None
        else:
            if self.node_budget == -1:
                # populate number of nodes for the case of whole-graph augmentation -- based on local_idx
                all_num_nodes = torch.diff(dataset.slices['x'])
                offsets = all_num_nodes[self.graph_idxs].cumsum(dim=0)
                # tack a zero at the beginning for easier indexing
                self.offsets = torch.cat((torch.tensor([0]), offsets))
                dimension = (int(self.offsets[-1]), self.feature_dim)
            else:
                dimension = (len(self.graph_idxs), self.node_budget, self.feature_dim)

            self.augmented_cache = torch.empty(dimension, dtype=dataset._data.x.dtype)

            # only move to device if we aren't augmenting everything
            if self.node_budget != -1:
                self.augmented_cache = self.augmented_cache # CUDA: .to(self.cfg.device)

            self.augmented_cache.share_memory_()


    def regenerate_augmented_embeddings(self):
        super().regenerate_augmented_embeddings()

        if not self.reaugment:
            if self.node_budget == -1:
                for local_idx, (graph_idx, dup_idx) in enumerate(zip(self.graph_idxs, self.dup_idxs)):
                    generator = torch.Generator().manual_seed(self.seed ^ self.epoch ^ graph_idx ^ dup_idx * 127)
                    self.augmented_cache[self.offsets[local_idx] : self.offsets[local_idx + 1]].normal_(generator=generator,) # type: ignore
            else:
                for i, (graph_idx, dup_idx) in enumerate(zip(self.graph_idxs, self.dup_idxs)):
                    for j, node_idx in enumerate(self.topk_nodes[graph_idx]): # type: ignore
                        generator = torch.Generator().manual_seed(
                            # dup_idx does not affect if 0
                            self.seed ^ graph_idx ^ self.epoch ^ node_idx * 32_767 ^ dup_idx * 127,
                        )
                        self.augmented_cache[i, j].normal_(generator=generator,) # type: ignore


    def __call__(self, batch):
        super().__call__(batch)

        # CUDA:
        # if self.node_budget > -1:
        #     batch = batch.to(self.cfg.device)

        if self.epoch < min(self.cfg.train.augmentation_start_epoch) or getattr(self, '_selecting', False):
            return batch

        if self.node_budget != -1 and self.topk_nodes is None:
            self._lazy_load_topk()
            assert self.topk_nodes is not None, "Top-k nodes should be populated after lazy loading."
        new_x = batch.x.clone()
        
        # determine the indices of the batch to be augmented
        if self.node_budget == -1:
            original_idxs = torch.arange(new_x.size(0))

            if not self.reaugment:
                cached_mask = []
                start_idxs = self.offsets[batch.local_idx]
                end_idxs = self.offsets[batch.local_idx + 1]
                for start, end in zip(start_idxs, end_idxs):
                    cached_mask.extend(range(int(start), int(end)))
        else:
            original_idxs = []
            topk_counts = []
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

        if self.reaugment:
            noise_node = None
        else:
            if self.node_budget == -1:
                noise_node = self.augmented_cache[cached_mask][is_augment] # type: ignore
            else:
                noise_node = (
                    self.augmented_cache[batch.local_idx] # type: ignore
                    .reshape(len(cached_mask), -1)[cached_mask][is_augment] # type: ignore
                )

        # NOTE TODO: remember to multiply noise by ratio if using the perturbation directly
        # (1-r) * x + r * (x + perturbation) = x + r * perturbation
        new_x[augment_idxs, :self.feature_dim] = augment_cosine(
            new_x[augment_idxs, :self.feature_dim],
            min_cosine=self.budget,
            random_vector=noise_node
        ) + new_x[augment_idxs, :self.feature_dim]

        batch.x = new_x
        return batch
