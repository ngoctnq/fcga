import importlib


def get_augmenter(cfg, **kwargs):
    if cfg.augmentation.name == "none":
        return IdentityAugmenter()

    if cfg.augmentation.name == "fcga":
        aug_module = importlib.import_module(f"src.loaders.augmentations.fcga.{cfg.augmentation.params.granularity}")
        return aug_module.Augmenter(cfg, **kwargs)

    if cfg.augmentation.name == "cosine":
        from src.loaders.augmentations.cosine import Augmenter
        return Augmenter(cfg)

    if cfg.augmentation.name.lower() == "drop":
        from src.loaders.augmentations.drop import NodeDrop, EdgeDrop, CombinedDrop
        mode = cfg.augmentation.params.mode
        if mode == "node":
            return NodeDrop(cfg)
        if mode == "edge":
            return EdgeDrop(cfg)
        if mode == "both":
            return CombinedDrop(EdgeDrop(cfg), NodeDrop(cfg))
        raise NotImplementedError(f"Unknown drop mode: {mode!r}")

    raise NotImplementedError(f"Dataset not supported for augmentation: {cfg.dataset.name}")


class IdentityAugmenter:
    def __call__(self, batch):
        return batch

    def propagate_information_from_dataset(self, dataset):
        pass

    def set_epoch(self, epoch):
        pass
    