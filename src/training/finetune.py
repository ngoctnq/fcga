import json
import torch
import torch.nn.functional as F
import yaml

from src.loaders.augmentations import get_augmenter
from src.utils import get_optimizer, get_model
from src.evaluation import evaluate, lite_eval_for_print

def train(cfg, dataset):
    # some dataset will not have some splits or custom evaluation
    valloader = testloader = None

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
            shuffle=True,
            truncate_ratio=truncate_ratio,
            augmenter=augmenter,
        )
        valloader = create_loader(
            dataset,
            cfg.train.batch_size,
            split="valid",
            shuffle=False,
        )
        testloader = create_loader(
            dataset,
            cfg.train.batch_size,
            split="test",
            shuffle=False,
        )
    else:
        raise NotImplementedError(f"Dataset not supported for training: {cfg.dataset.name}")

    # has to access idx 0 to get num_node_features because LDP changes dimensions
    cfg.model.input_dim = dataset[0].num_node_features
    cfg.model.output_dim = dataset.num_classes
    model = get_model(cfg)
    optimizer = get_optimizer(cfg, model)

    # for finetune mode, base checkpoint has to exist
    base_ckpt_path = cfg.train.params.base_checkpoint # if not exists, will raise error
    if base_ckpt_path is None:
        raise ValueError("Base checkpoint path must be provided for finetuning.")
    
    
    # HACK: for augmentations that need the special injections
    try:
        # HACK: quick access possibly-nonexistent attributes
        _model_dependent = {"gnnexplainer", "pgmexplainer", "subgraphx", "shapley", "output_drop"}
        if cfg.augmentation.params.criteria in _model_dependent:
            cfg.injected.model = model
            if cfg.augmentation.params.criteria in {"gnnexplainer", "pgmexplainer"}:
                cfg.injected.trainloader = trainloader
    except AttributeError:
        pass

    # default set of trackers
    starting_epoch = 0
    # keep track of best validation checkpoint
    best_val_acc = 0
    best_val_epoch = 0
    train_at_best_val = 0
    val_stats_at_best_val = None
    test_stats_at_best_val = None

    validation_criterion = "accuracy"

    # check for resuming
    save_dir = cfg.injected.save_dir
    if (save_dir / "last.pt").exists():
        with open(save_dir / "log.jsonl", "r") as f:
            logs = [json.loads(line) for line in f]
            # get best validation epoch and accuracy so far
            if 'val' in logs[-1]:
                for log in logs:
                    if log['val'][validation_criterion] > best_val_acc:
                        best_val_acc = log['val'][validation_criterion]
                        best_val_epoch = log['epoch']
                        train_at_best_val = log['train']['accuracy']
                        val_stats_at_best_val = log['val']
                        test_stats_at_best_val = log.get('test', None)
        starting_epoch = len(logs)
        print(f"Resuming from epoch {starting_epoch}")
        model.load_state_dict(torch.load(save_dir / "last.pt"))
        optimizer.load_state_dict(torch.load(save_dir / "optim.pt"))
    else:
        model.load_state_dict(torch.load(base_ckpt_path, map_location=cfg.device))
        # save the config for reproducibility
        with open(save_dir / "config.yaml", "w") as f:
            yaml.dump(cfg.model_dump(exclude=("injected")), f)

    if cfg.train.params.freeze_base_model:
        model.freeze_base_model()

    # training loop
    for epoch in range(starting_epoch, cfg.train.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        if epoch in cfg.train.augmentation_start_epoch:
            augmenter._selecting = True   # workers forked below inherit _selecting=True
            trainloader.hard_reload()     # pre-fork so GNNExplainer iteration sees early-return
        augmenter.set_epoch(epoch)
        if epoch in cfg.train.augmentation_start_epoch:
            trainloader.hard_reload()     # re-fork with _selecting=False, topk_nodes set

        for batch in trainloader:
            batch = batch.to(cfg.device, non_blocking=True)
            optimizer.zero_grad()
            
            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(outputs, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch.y).sum().item()
            total_samples += batch.num_graphs

            loss.backward()
            optimizer.step()
        
        if epoch < cfg.train.epochs - 1:
            trainloader.reload()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        log_entry = {
            "epoch": epoch + 1,
            "train": {
                'loss': avg_loss,
            },
        }
        print(f"Epoch {epoch+1}/{cfg.train.epochs}: Loss: {avg_loss:.4f} Train Acc: {accuracy:.4f}")

        # evaluate on train if we are about to switch augmentation strategy
        if epoch + 1 in cfg.train.augmentation_start_epoch or epoch + 1 == cfg.train.epochs:
            train_stats = evaluate(model, trainloader, cfg.device)
            log_entry["train"].update(train_stats)
        else:
            log_entry["train"]["accuracy"] = accuracy

        is_best_epoch = False
        if valloader is not None:
            val_stats = evaluate(model, valloader, cfg.device)
            print(f"  Val: {lite_eval_for_print(val_stats)}")
            log_entry["val"] = val_stats

            if epoch < cfg.train.epochs - 1:
                valloader.reload()

            # save best validation checkpoint
            if val_stats[validation_criterion] > best_val_acc:
                print(f"    New best validation {validation_criterion}: {val_stats[validation_criterion]:.4f} (previous best: {best_val_acc:.4f}), saving checkpoint")
                is_best_epoch = True
                best_val_acc = val_stats[validation_criterion]
                best_val_epoch = epoch + 1
                train_at_best_val = accuracy
                val_stats_at_best_val = val_stats
                torch.save(model.state_dict(), save_dir / "best.pt")

        if testloader is not None:
            test_stats = evaluate(model, testloader, cfg.device)
            print(f"  Test: {lite_eval_for_print(test_stats)}")
            log_entry["test"] = test_stats
            if is_best_epoch:
                test_stats_at_best_val = test_stats

            if epoch < cfg.train.epochs - 1:
                testloader.reload()

        torch.save(model.state_dict(), save_dir / "last.pt")
        torch.save(optimizer.state_dict(), save_dir / "optim.pt")

        with open(save_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(
        f"Best checkpoint at epoch {best_val_epoch}: "
        f"Train Acc: {train_at_best_val:.4f}\n"
        f"  Val Stats: {lite_eval_for_print(val_stats_at_best_val)}\n"
        f"  Test Stats: {lite_eval_for_print(test_stats_at_best_val)}"
    )
