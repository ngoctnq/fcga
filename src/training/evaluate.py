import json
import torch

from src.loaders.augmentations import get_augmenter
from src.utils import get_model
from src.evaluation import evaluate, lite_eval_for_print

def train(cfg, dataset):
    # get dataloader, augmentation, and evaluation
    truncate_ratio = getattr(cfg.dataset.params, "truncate_ratio", None)
    if cfg.dataset.name.startswith("MalNetTinyFeatures"):
        # get training set
        from src.loaders.mntf import create_loader
        # get augmentation
        augmenter = get_augmenter(cfg)

        trainloader = create_loader(
            dataset,
            cfg.train.batch_size,
            split="train",
            shuffle=False,
            truncate_ratio=truncate_ratio,
            augmenter=augmenter,
        )

        valloader = create_loader(
            dataset,
            cfg.train.batch_size,
            split="valid",
            shuffle=False,
            truncate_ratio=truncate_ratio,
            augmenter=augmenter,
        )

        testloader = create_loader(
            dataset,
            cfg.train.batch_size,
            split="test",
            shuffle=False,
            truncate_ratio=truncate_ratio,
            augmenter=augmenter,
        )
    else:
        raise NotImplementedError(f"Dataset not supported for training: {cfg.dataset.name}")

    # has to access idx 0 to get num_node_features because LDP changes dimensions
    cfg.model.input_dim = dataset[0].num_node_features
    cfg.model.output_dim = dataset.num_classes
    model = get_model(cfg)

    # checkpoints for evaluation
    save_dir = cfg.injected.save_dir

    # epoch to activate augmentation at. set_epoch() also initializes/refreshes
    # topk_nodes, augmented_cache, and epoch.txt -- none of which exist yet
    # since the training loop's set_epoch() was never called in this process.
    aug_epoch = min(cfg.train.augmentation_start_epoch)

    def evaluate_split(model, loader):
        # propagate_information_from_dataset + set_epoch must run for this
        # split's graph_idxs right before evaluating it, since the augmenter's
        # selection/cache state (unique_graph_idxs, augmented_cache, topk_nodes)
        # is shared/mutable and gets overwritten by whichever split last called
        # propagate_information_from_dataset (i.e. testloader, by construction order).
        augmenter.propagate_information_from_dataset(loader.dataset)
        augmenter.set_epoch(aug_epoch)
        return evaluate(model, loader, cfg.device)

    all_results = {}
    if not (save_dir / "best.pt").exists():
        print("[!] No best checkpoint to evaluate")
    else:
        model.load_state_dict(torch.load(save_dir / "best.pt", map_location=cfg.device))
        train_best_stats = evaluate_split(model, trainloader)
        print(f"[Best] Train: {lite_eval_for_print(train_best_stats)}")
        val_best_stats = evaluate_split(model, valloader)
        print(f"[Best] Validation: {lite_eval_for_print(val_best_stats)}")
        test_best_stats = evaluate_split(model, testloader)
        print(f"[Best] Test: {lite_eval_for_print(test_best_stats)}")

        all_results['best'] = {
            'train': train_best_stats,
            'val': val_best_stats,
            'test': test_best_stats
        }

    if not (save_dir / "last.pt").exists():
        print("[!] No last checkpoint to evaluate")
    else:
        model.load_state_dict(torch.load(save_dir / "last.pt", map_location=cfg.device))
        train_last_stats = evaluate_split(model, trainloader)
        print(f"[Last] Train: {lite_eval_for_print(train_last_stats)}")
        val_last_stats = evaluate_split(model, valloader)
        print(f"[Last] Validation: {lite_eval_for_print(val_last_stats)}")
        test_last_stats = evaluate_split(model, testloader)
        print(f"[Last] Test: {lite_eval_for_print(test_last_stats)}")

        all_results["last"] = {
            'train': train_last_stats,
            'val': val_last_stats,
            'test': test_last_stats
        }

    if not all_results:
        print("No checkpoints found to evaluate, exiting...")
    else:
        (save_dir / "evaluation").mkdir(exist_ok=True)
        with open(save_dir / "evaluation" / f"{cfg.injected.evaluation_name}.json", "w") as f:
            json.dump(all_results, f, indent=4)
