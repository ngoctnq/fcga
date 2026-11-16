from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from src.fcga.embedder.qwen3 import embed_batch
from src.fcga.selector import get_selector
from src.loaders.mntf.base import MalNetDataset


def augment_dataset(
    cfg,
    dataset: MalNetDataset,
    topk_nodes: Optional[dict[int, List[int]]] = None,
) -> None:
    """Embed cached pre-refactoring summaries for the selected original nodes.

    Summary text is expected at:
        summaries/<graph_idx>/<node_idx>.txt

    Embeddings are written separately at:
        summary_embeddings/<graph_idx>/<node_idx>.pth

    This function does not generate missing summary text.
    """
    root_dir = (
        Path(cfg.injected.base_dir)
        / "datasets"
        / "augmented"
        / "mntf"
        / cfg.dataset.name
        / cfg.dataset.params.variant
    )
    summary_dir = root_dir / "summaries"
    embedding_dir = root_dir / "summary_embeddings"

    if topk_nodes is None:
        selector = get_selector(cfg)
        topk_nodes = selector(
            dataset,
            graph_idxs=dataset.split_idxs["train"],
            cache_dir=root_dir / "selection",
        )
        assert topk_nodes is not None, "Node selection failed."

    sources: list[str] = []
    destinations: list[Path] = []
    missing_summaries: list[Path] = []

    for data_idx in sorted(topk_nodes):
        graph_summary_dir = summary_dir / str(data_idx)
        graph_embedding_dir = embedding_dir / str(data_idx)
        for node_idx in sorted(topk_nodes[data_idx]):
            destination = graph_embedding_dir / f"{node_idx}.pth"
            if destination.exists():
                continue

            source = graph_summary_dir / f"{node_idx}.txt"
            if not source.exists():
                missing_summaries.append(source)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            sources.append(source.read_text())
            destinations.append(destination)

    if missing_summaries:
        preview = "\n".join(f"  - {path}" for path in missing_summaries[:10])
        remainder = len(missing_summaries) - min(len(missing_summaries), 10)
        suffix = f"\n  ... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            f"Missing {len(missing_summaries)} cached summary text file(s):\n"
            f"{preview}{suffix}"
        )

    if not sources:
        return

    def save_batch(start_idx: int, end_idx: int, batch_embeddings) -> None:
        for destination, embedding in zip(
            destinations[start_idx:end_idx], batch_embeddings
        ):
            if not destination.exists():
                torch.save(embedding, destination)

    embed_batch(
        sources,
        bsz=cfg.augmentation.params.embedder_bsz,
        device=cfg.device,
        on_batch_done=save_batch,
    )

    missing_embeddings = [path for path in destinations if not path.exists()]
    if missing_embeddings:
        raise RuntimeError(
            f"Embedding completed without creating {len(missing_embeddings)} "
            "expected cache file(s)."
        )

    tqdm.write(f"Embedded {len(destinations)} cached function summaries.")
