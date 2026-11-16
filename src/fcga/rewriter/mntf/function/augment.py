import torch
import json
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import List, Optional

from src.fcga.rewriter.ollama import call_ollama, parse_backtick
from src.fcga.selector import get_selector
from src.loaders.mntf.base import MalNetDataset
from src.fcga.embedder.qwen3 import embed_batch


def augment_dataset(
        cfg,
        dataset: MalNetDataset,
        topk_nodes: Optional[dict[int, List[int]]] = None
    ):
    if cfg.augmentation.params.rewriter == "qwen2.5-coder":
        from .qwen25_coder import tokenizer, build_prompt, ollama_kwargs
    elif cfg.augmentation.params.rewriter == "phi4-mini":
        from .phi4_mini import tokenizer, build_prompt, ollama_kwargs
    elif cfg.augmentation.params.rewriter == "gemma4":
        from .gemma4 import tokenizer, build_prompt, ollama_kwargs
    else:
        raise NotImplementedError(f"Rewriter not supported for augmentation: {cfg.augmentation.params.rewriter}")

    root_dir = (
        Path(cfg.injected.base_dir)
        / "datasets"
        / "augmented"
        / "mntf"
        / cfg.dataset.name
        / cfg.dataset.params.variant
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    model_id = cfg.augmentation.params.rewriter

    # note to whoever's reading this: you can consider cleaning up intermediate
    # files after augmentation to save space
    stats_dir = root_dir / "rewrites" / model_id / "statistics"
    stats_dir.mkdir(exist_ok=True, parents=True)

    raw_dir = getattr(cfg.dataset.params, "raw_dir", None)

    # run node selection outside because we need we can cache it for later
    if topk_nodes is None:
        target_set = dataset.split_idxs["train"]
        # target_set = dataset.split_idxs["train"] + dataset.split_idxs["test"]
        # target_set = dataset.split_idxs["train"] + dataset.split_idxs["valid"] + dataset.split_idxs["test"]
        node_selector = get_selector(cfg)
        topk_nodes = node_selector(
            dataset, graph_idxs=target_set, cache_dir=root_dir / "selection"
        )
        assert topk_nodes is not None, "Error happened during node selection. Please check the selection cache and logic."

    source_codes = dataset.get_source_codes(topk_nodes, raw_dir=raw_dir)


    def func(cache_dir: Path, topk_nodes: List[int], source_codes: List[str], cfg, **ollama_kwargs):
        """
        Individual function augmentation logic for a single graph.
        Used for parallelization.

        :param cache_dir: Path to cache the augmented code for each node
        :type cache_dir: Path
        :param topk_nodes: List of selected node indices for the graph
        :type topk_nodes: List[int]
        :param source_codes: List of source code strings corresponding to the selected nodes
        :type source_codes: List[str]
        :param cfg: Config object containing augmentation parameters
        :type cfg: Config
        """
        context_length = cfg.augmentation.params.context_length

        cache_dir.mkdir(exist_ok=True)

        for node_idx_subgraph, node_idx in enumerate(topk_nodes):
            cache_path = cache_dir / f"{node_idx}.json"
            if cache_path.exists():
                continue
            # balance load across multiple Ollama instances if provided
            ollama_urls = cfg.augmentation.params.ollama_urls
            ollama_url = random.choice(ollama_urls)

            response = call_ollama(
                build_prompt(source_codes[node_idx_subgraph]),
                ollama_url,
                model_id=model_id,
                tokenizer=tokenizer,
                context_length=context_length,
                min_output_tokens=context_length // 2, # use half of the context for input to leave room for output
                seed=cfg.seed,
                **ollama_kwargs
            )

            # the input and output are both contained in the response for easy logging and debugging
            # we'll fetch them later in the embedder stage to construct augmented graphs
            # new_code = parse_backtick(response["raw_output"]["message"]["content"])
            with open(cache_path, "w") as f:
                json.dump(response, f)

    with tqdm(total=len(topk_nodes), desc="Augmenting functions") as pbar:
        with ThreadPoolExecutor(cfg.augmentation.params.num_workers) as executor:
            futures = [
                executor.submit(
                    func,
                    stats_dir / f"{data_idx}",
                    topk_nodes[data_idx],
                    source_codes[data_idx],
                    cfg,
                    **ollama_kwargs
                )
                for data_idx in topk_nodes.keys()
            ]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
                pbar.refresh()

    # embed everything after augmentation is done
    all_fns = stats_dir.glob("*/*.json")
    all_destinations = []
    all_sources = []
    for fn in all_fns:
        # change the second last token (folder)
        destination = fn.parent.parent.parent / "embeddings" / fn.parent.name / f"{fn.stem}.pth"
        if destination.exists():
            continue
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        all_destinations.append(destination)
        with open(fn, "r") as f:
            response = json.load(f)
        source = response["raw_output"]["message"]["content"]
        all_sources.append(parse_backtick(source))
    if all_sources:
        embeddings = embed_batch(
            all_sources,
            bsz=cfg.augmentation.params.embedder_bsz,
            device=cfg.device
        )
        for destination, embedding in tqdm(zip(all_destinations, embeddings), total=len(all_destinations), desc="Saving embeddings"):
            torch.save(embedding, destination)
