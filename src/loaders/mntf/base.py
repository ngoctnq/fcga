import numpy as np
import torch
import os.path as osp
from torch_geometric.data.dataset import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
from glob import glob
import os
import pickle
from tqdm import tqdm


class MalNetDataset(InMemoryDataset):
    '''
    PyG Dataset class for MalNet-Tiny-Features.
    Contains auxilliary information for reconstructions from original raw APKs.
    `dup_idxs` is a list of the same length as the dataset, where each element
        indicates the duplicate index for that data point, i.e. the i-th time
        this data point appears in the dataset.
    '''

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        # PyTorch >= 2.6 requires weights_only=False to load full objects
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False, mmap=True)
        self.split_idxs = torch.load(self.processed_paths[1], weights_only=False)
        self.keep_idxs_list = torch.load(self.processed_paths[2], weights_only=False)
        self.mask_list = torch.load(self.processed_paths[3], weights_only=False)
        self.hash_list = torch.load(self.processed_paths[4], weights_only=False)

        # original dataset has no duplicates
        self.dup_idxs = [0] * len(self)


    def __getitem__(self, idx):
        '''
        Inject local_idx, global_idx, and dup_idx into the data object
        '''
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            global_idx = int(self.indices()[idx])  # type: ignore
            data = self.get(global_idx)
            data.global_idx = global_idx
            data.local_idx = idx
            data.dup_idx = self.dup_idxs[idx] # type: ignore
            # attach index to the Data object
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)
        

    def index_select(self, idx) -> Dataset:
        new_dataset = super().index_select(idx)

        # update dup_idxs for the new dataset
        counter = {}
        # new list to avoid shallow copy
        new_dup_idxs = []
        for global_idx in new_dataset.indices():
            dup_idx = counter.get(global_idx, 0)
            new_dup_idxs.append(dup_idx)
            counter[global_idx] = dup_idx + 1

        new_dataset.dup_idxs = new_dup_idxs # type: ignore
        return new_dataset


    @property
    # always use fast path
    def num_classes(self) -> int:
        return self._infer_num_classes(self._data.y) # type: ignore


    @property
    def raw_file_names(self):
        return []


    @property
    def processed_file_names(self):
        return ['data.pt', 'split_dict.pt', 'keep_idxs_list.pt', 'mask_list.pt', 'hash_list.pt']
    

    @property
    def processed_dir(self) -> str:
        return self.root


    def download(self):
        pass


    def process(self):
        pass


    def get_source_codes(self, topk_nodes, raw_dir=None):
        '''
        topk_nodes: dict of global indices to list of node idxs to get source codes for
        '''
        all_source_codes = {}
        for data_idx in tqdm(topk_nodes, desc="Getting source codes"):
            file_hash = self.hash_list[data_idx]
            keep_idxs = self.keep_idxs_list[data_idx] # for prune only
            mask_list = self.mask_list[data_idx] # for when remove isolated nodes

            if raw_dir is None:
                real_path = glob(os.path.join(self.raw_dir, "../../../raw/malnet-graphs-tiny/*/*/{}.pkl".format(file_hash)))[0]
            else:
                real_path = glob(os.path.join(raw_dir, "{}.pkl".format(file_hash)))[0]
            with open(real_path, "rb") as f:
                data_pkl = pickle.load(f)

            # first deal with keep_idxs in case of prune
            if keep_idxs is not None:
                # map the nodes again
                source_codes = [data_pkl['source_code'][node_idx] for node_idx in keep_idxs]
            else:
                source_codes = data_pkl['source_code']

            # then deal with node masks
            new_node_idxs = [node_idx for node_idx, mask_value in enumerate(mask_list) if mask_value]
            source_codes = [source_codes[node_idx] for node_idx in new_node_idxs]

            # finally get the source codes for the selected nodes
            source_codes = [source_codes[node_idx] for node_idx in topk_nodes[data_idx]]
            all_source_codes[data_idx] = source_codes
        return all_source_codes


def load_dataset(name, variant, base_path, check_iso=True):
    '''
    Load MalNet dataset with specified variant.
    Args:
        name (str): Name of the dataset
            (e.g., 'MalNetTinyFeaturesCommon+qwen+keepisolated')
        variant (str): Variant of the dataset
            (e.g., 'prune+llm+ldp', 'zero+all+none')
        base_path (str): Base path to the dataset directory
        check_iso (bool): Whether to check and enforce isolated node retention
            for prune variants. Defaults to True.
    Returns:
        MalNetDataset: Loaded dataset object
    '''

    if check_iso:
        if 'prune' in variant and not name.endswith('+keepisolated'):
            print('Prune variant without isolated nodes, adding suffix +keepisolated...')
            name += '+keepisolated'

    names = name.split('+')
    path = osp.join(base_path, names[0], 'processed')

    if len(names) == 3:
        llm_variant = '+' + names[1]
        keepiso = '_keepisolated'
    elif len(names) == 2:
        if names[1] == 'keepisolated':
            llm_variant = ''
            keepiso = '_keepisolated'
        else:
            llm_variant = '+' + names[1]
            keepiso = ''
    else:
        assert len(names) == 1
        llm_variant = ''
        keepiso = ''

    variant, ldp = variant.rsplit('+', 1)
    path = osp.join(path, variant + llm_variant + keepiso)
    transform = T.LocalDegreeProfile() if ldp == 'ldp' else None
    dataset = MalNetDataset(path, transform=transform)
    
    return dataset
