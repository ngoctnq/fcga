from src.loaders.mntf import load_dataset
from pathlib import Path


def get_dataset(cfg):
    if cfg.dataset.name == "MalNetTinyFeatures-100":
        import torch_geometric.transforms as T
        from src.loaders.mntf.tiny100 import load_mntf100_dataset
        p = cfg.dataset.params
        process_cfg = {
            "ollama_urls": p.ollama_urls,
            "context_length": p.context_length,
            "num_workers": p.num_workers,
            "embedder_bsz": p.embedder_bsz,
            "device": cfg.device,
            "seed": cfg.seed,
        }
        transform = T.LocalDegreeProfile() if getattr(p, "ldp", True) else None
        return load_mntf100_dataset(
            rewriter=p.rewriter,
            base_dir=str(Path(cfg.injected.base_dir)),
            process_cfg=process_cfg,
            transform=transform,
        )
    if cfg.dataset.name.startswith("MalNetTinyFeatures"):
        dataset = load_dataset(
            cfg.dataset.name,
            cfg.dataset.params.variant,
            Path(cfg.injected.base_dir) / "datasets" / "original" / "mntf",
        )
        return dataset
    raise NotImplementedError(f"Dataset not implemented: {cfg.dataset.name}")
