import json
import random
import os.path as osp
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import Dataset

from src.fcga.rewriter.ollama import call_ollama, parse_backtick
from src.fcga.embedder.qwen3 import embed_batch
from src.loaders.mntf.base import load_dataset
from src.utils.syntax import is_valid_source


_SAMPLES_PER_CLASS = 20
_BASE_NAME = "MalNetTinyFeatures+qwen+keepisolated"
_BASE_VARIANT = "prune+llm+ldp"


class MalNetTinyFeatures100(InMemoryDataset):
    """
    100-sample-per-split subset of MalNetTinyFeatures with LLM-rewritten embeddings
    baked into node features. Selects the first 20 samples per class from train and
    test splits; no val split.

    Shares the rewrite/embed cache with the fcga-f pipeline at:
      datasets/augmented/mntf/MalNetTinyFeatures+qwen+keepisolated/prune+llm+ldp/rewrites/{rewriter}/

    process_cfg (dict) is only required on the first build and ignored once cached:
      ollama_urls, context_length, num_workers, embedder_bsz, device, seed, raw_dir
    """

    def __init__(
        self,
        root: str,
        rewriter: str,
        base_dir: str,
        raw_dir: str = None,
        process_cfg: dict = None,
        transform=None,
        pre_transform=None,
    ):
        # must be set before super().__init__ so process() can access them
        self.rewriter = rewriter
        self.base_dir = base_dir
        self.apk_raw_dir = raw_dir
        self.process_cfg = process_cfg
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False, mmap=True)
        self.split_idxs = torch.load(self.processed_paths[1], weights_only=False)
        self.dup_idxs = [0] * len(self)

    @property
    def num_classes(self) -> int:
        return self._infer_num_classes(self._data.y)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt', 'split_dict.pt']

    @property
    def processed_dir(self) -> str:
        return self.root

    def download(self):
        pass

    def __getitem__(self, idx):
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):
            global_idx = int(self.indices()[idx])
            data = self.get(global_idx)
            data.global_idx = global_idx
            data.local_idx = idx
            data.dup_idx = self.dup_idxs[idx]
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)

    def index_select(self, idx) -> Dataset:
        new_dataset = super().index_select(idx)
        counter = {}
        new_dup_idxs = []
        for global_idx in new_dataset.indices():
            dup_idx = counter.get(global_idx, 0)
            new_dup_idxs.append(dup_idx)
            counter[global_idx] = dup_idx + 1
        new_dataset.dup_idxs = new_dup_idxs
        return new_dataset

    def process(self):
        cfg = self.process_cfg
        assert cfg is not None, "process_cfg is required to build the dataset for the first time"

        if self.rewriter == "qwen2.5-coder":
            from src.fcga.rewriter.mntf.function.qwen25_coder import tokenizer, build_prompt, ollama_kwargs
        elif self.rewriter == "phi4-mini":
            from src.fcga.rewriter.mntf.function.phi4_mini import tokenizer, build_prompt, ollama_kwargs
        elif self.rewriter == "gemma4":
            from src.fcga.rewriter.mntf.function.gemma4 import tokenizer, build_prompt, ollama_kwargs
        else:
            raise NotImplementedError(f"Rewriter not supported: {self.rewriter}")

        ollama_urls = cfg["ollama_urls"]
        context_length = cfg["context_length"]
        num_workers = cfg["num_workers"]
        embedder_bsz = cfg["embedder_bsz"]
        device = cfg["device"]
        seed = cfg["seed"]

        base_dataset = load_dataset(
            _BASE_NAME, _BASE_VARIANT,
            osp.join(self.base_dir, "datasets", "original", "mntf"),
        )

        # first _SAMPLES_PER_CLASS per class from train and test; mirrors create_loader truncate logic
        sampled: dict[str, list[int]] = {}
        for split in ("train", "test"):
            idx_per_class = {
                c: [i for i in base_dataset.split_idxs[split] if base_dataset.y[i] == c]
                for c in range(base_dataset.num_classes)
            }
            sampled[split] = [i for idxs in idx_per_class.values() for i in idxs[:_SAMPLES_PER_CLASS]]

        all_global_idxs = sampled["train"] + sampled["test"]
        all_node_idxs: dict[int, list[int]] = {
            data_idx: list(range(base_dataset[data_idx].num_nodes))
            for data_idx in all_global_idxs
        }

        shared_dir = (
            Path(self.base_dir)
            / "datasets" / "augmented" / "mntf"
            / _BASE_NAME / _BASE_VARIANT
            / "rewrites" / self.rewriter
        )
        stats_dir = shared_dir / "statistics"
        embed_dir = shared_dir / "embeddings"
        syntax_dir = shared_dir / "syntax"
        stats_dir.mkdir(parents=True, exist_ok=True)

        # rewrite functions
        # only fetch source codes for graphs that have at least one missing stats file
        need_rewrite: dict[int, list[int]] = {
            data_idx: [
                fn_idx for fn_idx in node_list
                if not (stats_dir / str(data_idx) / f"{fn_idx}.json").exists()
            ]
            for data_idx, node_list in all_node_idxs.items()
        }
        need_rewrite = {k: v for k, v in need_rewrite.items() if v}

        if need_rewrite:
            source_codes = base_dataset.get_source_codes(need_rewrite, raw_dir=self.apk_raw_dir)

            def rewrite_graph(data_idx: int) -> None:
                node_list = need_rewrite[data_idx]
                srcs = source_codes[data_idx]
                cache_base = stats_dir / str(data_idx)
                cache_base.mkdir(exist_ok=True)
                for pos, fn_idx in enumerate(node_list):
                    cache_path = cache_base / f"{fn_idx}.json"
                    if cache_path.exists():
                        continue
                    url = random.choice(ollama_urls)
                    response = call_ollama(
                        build_prompt(srcs[pos]),
                        url,
                        model_id=self.rewriter,
                        tokenizer=tokenizer,
                        context_length=context_length,
                        min_output_tokens=context_length // 2,
                        seed=seed,
                        **ollama_kwargs,
                    )
                    with open(cache_path, "w") as f:
                        json.dump(response, f)

            with tqdm(total=len(need_rewrite), desc="Rewriting functions") as pbar:
                with ThreadPoolExecutor(num_workers) as executor:
                    futures = [executor.submit(rewrite_graph, idx) for idx in need_rewrite]
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)
                        pbar.refresh()

        # embed rewrites
        unembeddeds: list[tuple[str, Path]] = []
        # check rewrite validity to filter out
        bad_rewrites = set()

        for data_idx, node_list in all_node_idxs.items():
            for fn_idx in node_list:
                syntax_file = syntax_dir / str(data_idx) / f"{fn_idx}.json"
                stats_path = stats_dir / str(data_idx) / f"{fn_idx}.json"
                syntax_result = None
                
                if syntax_file.exists():
                    with open(syntax_file) as f:
                        syntax_result = json.load(f)
                    if syntax_result["error"]:
                        bad_rewrites.add((data_idx, fn_idx))
                        continue

                with open(stats_path) as f:
                    response = json.load(f)

                if syntax_result is None:
                    syntax_result = is_valid_source(response)
                    if syntax_result["error"]:
                        bad_rewrites.add((data_idx, fn_idx))
                        continue
                
                dest = embed_dir / str(data_idx) / f"{fn_idx}.pth"
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                source = parse_backtick(response["raw_output"]["message"]["content"])
                unembeddeds.append((source, dest))

        if unembeddeds:
            all_sources, all_destinations = zip(*unembeddeds)

            def on_batch_done(start_idx: int, end_idx: int, batch_embeddings):
                for dest, emb in zip(all_destinations[start_idx:end_idx], batch_embeddings):
                    if not dest.exists():
                        torch.save(emb, dest)

            embed_batch(all_sources, bsz=embedder_bsz, device=device, on_batch_done=on_batch_done)

        # and build dataset
        data_list = []
        count = total = 0
        for data_idx in tqdm(all_global_idxs, desc="Building dataset"):
            data = base_dataset[data_idx]
            data_x = []
            for fn_idx in all_node_idxs[data_idx]:
                total += 1
                if (data_idx, fn_idx) in bad_rewrites:
                    data_x.append(data.x[fn_idx][:1024])
                else:
                    embed_file = embed_dir / str(data_idx) / f"{fn_idx}.pth"
                    data_x.append(torch.load(embed_file, weights_only=True))
                    count += 1
            data.x = torch.stack(data_x)
            data_list.append(data)
        print(f"{count}/{total} nodes embedded ({count/total:.2%})")

        collated, slices = self.collate(data_list)
        torch.save((collated, slices), self.processed_paths[0])

        n_train = len(sampled["train"])
        torch.save(
            {
                "train": list(range(n_train)),
                "valid": [],
                "test": list(range(n_train, n_train + len(sampled["test"]))),
            },
            self.processed_paths[1],
        )


def load_mntf100_dataset(rewriter: str, base_dir: str, apk_raw_dir: str = None,
                          process_cfg: dict = None, transform=None):
    path = osp.join(
        base_dir, "datasets", "original", "mntf",
        "MalNetTinyFeatures-100", "processed", rewriter,
    )
    return MalNetTinyFeatures100(
        path, rewriter=rewriter, base_dir=base_dir,
        raw_dir=apk_raw_dir, process_cfg=process_cfg, transform=transform,
    )
