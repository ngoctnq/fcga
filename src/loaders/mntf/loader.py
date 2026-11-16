from torch_geometric.loader import DataLoader
from multiprocessing import cpu_count
import torch


class AugmentedLoader(DataLoader):
    def __init__(self, *args, augmenter=None, loader_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_collate = self.collate_fn
        del self.collate_fn  # expose class method below
        self.augmenter = augmenter
        self.loader_name = loader_name # for logging
        self._persistent_iter = None # for persistent workers
        self.reloaded = False

    def collate_fn(self, data_list):
        # worker_id = torch.utils.data.get_worker_info()
        # print(f"[{self.loader_name or 'worker'} {worker_id.id if worker_id else 'main'}] collating {len(data_list)} samples", flush=True)
        batch = self._base_collate(data_list)
        # print(f"[{self.loader_name or 'worker'} {worker_id.id if worker_id else 'main'}] finished collating, augmenting", flush=True)
        if self.augmenter is not None:
            batch = self.augmenter(batch)
        # print(f"[{self.loader_name or 'worker'} {worker_id.id if worker_id else 'main'}] finished augmenting", flush=True)
        return batch

    def reload(self):
        # worker_id = torch.utils.data.get_worker_info()
        # print(f"[{self.loader_name or 'worker'} {worker_id.id if worker_id else 'main'}] reloading", flush=True)
        self._persistent_iter = super().__iter__()
        self.reloaded = True

    def hard_reload(self):
        """Shut down existing workers and fork fresh ones so they inherit current shared state."""
        self._persistent_iter = None  # drop our reference
        self._iterator = None         # drop DataLoader's internal reference -> __del__ -> shutdown
        self._persistent_iter = super().__iter__()  # fork fresh workers
        self.reloaded = True

    def __iter__(self):  # type: ignore
        if not self.reloaded:
            self.reload()
        self.reloaded = False
        return self._persistent_iter


def create_loader(
    dataset,
    batch_size,
    split,
    shuffle,
    num_workers=None,
    pin_memory=True,
    truncate_ratio=None,
    augmenter=None,
):
    '''
    Create a DataLoader for the given dataset and split, with optional truncation and augmentation.
    :param dataset: The full dataset object containing all splits.
    :param batch_size: Batch size for the DataLoader.
    :param split: Which split to load ('train', 'valid', 'test', or 'all').
    :param shuffle: Whether to shuffle the data (typically True for training, False for validation/test).
    :param num_workers: Number of worker processes for data loading. If None, it will be set to min(4, cpu_count()).
    :param pin_memory: Whether to pin memory in the DataLoader (typically True for GPU training).
    :param truncate_ratio: If set, keep only the first `truncate_ratio` fraction of samples per class, in (0, 1].
    :param augmenter: Optional augmenter object to patch the dataset for later augmentation in the DataLoader.
    :return: A DataLoader object for the specified dataset and split, with augmentations applied if provided.
    '''
    if truncate_ratio is not None:
        idx_per_class = {
            c: [i for i in dataset.split_idxs[split] if dataset.y[i] == c]
            for c in range(dataset.num_classes)
        }
        keep = [i for idxs in idx_per_class.values() for i in idxs[:int(len(idxs) * truncate_ratio)]]
        dataset = dataset[keep]
    elif split == "all":
        dataset = dataset[:]
    elif split in dataset.split_idxs:
        dataset = dataset[dataset.split_idxs[split]]
    else:
        raise ValueError(f"Split {split} not found in dataset splits: {dataset.split_idxs.keys()}")

    if augmenter is not None:
        augmenter.propagate_information_from_dataset(dataset)

    if num_workers is None:
        num_workers = min(4, cpu_count())
    assert num_workers > 0, "Num workers should be positive"
    pw = True if num_workers > 0 else False

    return AugmentedLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        # prefetch_factor=1 if pw else 0,
        # force fork: workers inherit shared memory and mmap tensors without pickling the dataset
        # otherwise 'spawn' would trigger _share_fd_cpu_ per tensor -> EMFILE
        multiprocessing_context='fork',
        augmenter=augmenter,
        loader_name=f"{split}",
    )
