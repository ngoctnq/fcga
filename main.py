# if __name__ == '__main__':
#     import multiprocessing as mp
#     print('Setting multiprocessing to spawn')
#     mp.set_start_method('spawn', force=True)


import argparse
import importlib
import sys
import pprint
from datetime import datetime
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from src.utils import get_config, get_dataset, setup_logging, remove_logging, seed_everything


sys.stdout.flush() # ensure all print statements are flushed immediately


def augment(cfg, dataset):
    print(f"[augment] name={cfg.augmentation.name}")
    if cfg.augmentation.name.lower() in ["none", "cosine", "drop"]:
        print("[augment] no augmentation needed")

    elif cfg.augmentation.name == "fcga":
        if cfg.dataset.name.startswith("MalNetTinyFeatures"):
            data_sublevel = 'mntf'
        else:
            raise NotImplementedError(f"Dataset not supported for augmentation: {cfg.dataset.name}")
        
        if cfg.augmentation.params.granularity == "function":
            granularity_sublevel = "function"
        elif cfg.augmentation.params.granularity in {"subgraph", "subgraph_summary"}:
            granularity_sublevel = cfg.augmentation.params.granularity
        else:
            raise NotImplementedError(f"Granularity not supported for augmentation: {cfg.augmentation.params.granularity}")

        _model_dependent = {"gnnexplainer", "pgmexplainer", "subgraphx", "shapley", "output_drop"}
        if cfg.augmentation.params.criteria in _model_dependent:
            print(f"[augment] {cfg.augmentation.params.criteria} selection deferred to training.")
            return

        module_path = f"src.fcga.rewriter.{data_sublevel}.{granularity_sublevel}.augment"
        module = importlib.import_module(module_path)
        module.augment_dataset(cfg, dataset)

    else:
        raise NotImplementedError(f"Augmentation method not implemented: {cfg.augmentation.name}")


def train(cfg, dataset):
    print(f"[train] mode={cfg.train.mode}")
    if cfg.train.mode == "none":
        return
    else:
        train_mode = cfg.train.mode.replace('-', '_').lower()
        module_path = f"src.training.{train_mode}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError:
            raise NotImplementedError(f"Training mode not implemented: {cfg.train.mode}")
        module.train(cfg, dataset)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cpu' or 'cuda')")
    parser.add_argument("--dataset-cfg", type=str, required=True, help="Path to the dataset config file")
    parser.add_argument("--augment-cfg", type=str, required=True, help="Path to the augmentation config file")
    parser.add_argument("--model-cfg", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--train-cfg", type=str, required=True, help="Path to the training config file")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Path to a pretrained model checkpoint (optional)")
    args = parser.parse_args()
    return args


def get_cfg(args):
    _root = Path(__file__).resolve().parent

    dataset_cfg_path = Path(args.dataset_cfg)
    augment_cfg_path = Path(args.augment_cfg)
    model_cfg_path = Path(args.model_cfg)
    train_cfg_path = Path(args.train_cfg)

    cfg = get_config(
        dataset_cfg_path,
        augment_cfg_path,
        model_cfg_path,
        train_cfg_path,
    )

    log_path = _root / "logs" / (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{dataset_cfg_path.stem}"
        f"_{augment_cfg_path.stem}"
        f"_{model_cfg_path.stem}"
        f"_{train_cfg_path.stem}.log"
    )

    # if augmentation is "none", training config should not have aug
    # run this first so that we can inject savedir into cfg
    train_cfg_path_stem_processed = train_cfg_path.stem
    if cfg.augmentation.name == "none":
        matches = re.findall(r"_aug\d\.\d", train_cfg_path_stem_processed)
        if len(matches) > 0:
            assert len(matches) == 1, f"Expected exactly one augmentation ratio in train config filename, found {len(matches)}: {matches}"
            train_cfg_path_stem_processed = train_cfg_path_stem_processed.replace(matches[0], "")
    save_dir = _root / "results" / (
        f"{dataset_cfg_path.stem}"
        f"+{augment_cfg_path.stem}"
        f"+{model_cfg_path.stem}"
        f"+{train_cfg_path_stem_processed}"
    ) / str(args.seed)

    # live patch if pretrained_path is provided
    if args.pretrained_path is not None:
        print(f"Pretrained path provided: {args.pretrained_path}, patching configuration for downstream training.")
        pretrained_path = Path(args.pretrained_path)
        if cfg.train.mode == 'finetune':
            if not pretrained_path.is_file():
                raise ValueError(f"Provided pretrained path does not exist or is not a file: {pretrained_path}")
            save_dir = (
                pretrained_path.parent
                / (pretrained_path.stem + "_finetune")
                / save_dir.parent.name
                / str(args.seed)
            )
            cfg.train.params.base_checkpoint = args.pretrained_path # type: ignore # string for easy dumping
        elif cfg.train.mode == 'evaluate':
            save_dir = pretrained_path
            cfg.injected.evaluation_name = f'{dataset_cfg_path.stem}_{augment_cfg_path.stem}' # type: ignore

    # inject fields that are not from config file
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.injected.base_dir = _root # type: ignore
    cfg.injected.log_path = log_path # type: ignore
    cfg.injected.save_dir = save_dir # type: ignore

    return cfg


def run(cfg, logging_setup_prefix=""):
    seed_everything(cfg.seed)

    log_path = cfg.injected.log_path
    setup_logging(log_path, prefix=logging_setup_prefix)

    # create temporary directory for augmentation artifacts, which will be cleaned up automatically
    temp_dir = TemporaryDirectory()
    cfg.injected.temp_dir = Path(temp_dir.name) # type: ignore

    print(f"[main] Configuration:\n{pprint.pformat(cfg)}")

    if getattr(cfg.injected, 'dataset', None):
        # dataset already injected (e.g., from augmentation stage), reuse it
        dataset = cfg.injected.dataset
    else:
        # load dataset once and pass to all stages
        dataset = get_dataset(cfg)
        # inject augmented dataset back to cfg for reuse
        cfg.injected.dataset = dataset # type: ignore

    # run training and evaluation
    cfg.injected.save_dir.mkdir(parents=True, exist_ok=True)

    # real main code
    augment(cfg, dataset)
    train(cfg, dataset)

    # cleanup
    temp_dir.cleanup()
    remove_logging(log_path)


def main():
    args = get_args()
    cfg = get_cfg(args)
    run(cfg, logging_setup_prefix=f"$ python {' '.join(sys.argv)}")


if __name__ == "__main__":
    main()
