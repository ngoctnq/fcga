from pathlib import Path
import torch
from src.loaders.mntf import load_dataset, create_loader
from .standard import evaluate


# lazy loading
mntf_common_test_loader = None


@torch.inference_mode()
def evaluate_custom(cfg, model, **kwargs):
    '''
    Evaluate the model on MalNet-Tiny-Common and compute metrics.
    Returns a dictionary of metrics: accuracy and confusion matrix.

    :param cfg: Configuration object containing evaluation settings.
    :param model: The trained model to evaluate.
    :param kwargs: Additional keyword arguments for compatibility
    :return: Dictionary containing 'accuracy' and 'confusion_matrix'.
    '''

    assert cfg.dataset.name.startswith("MalNetTinyFeatures"), \
        "This custom evaluation only supports MalNetTinyFeatures dataset."
    dataset_name_tokens = cfg.dataset.name.split("+")
    dataset_name_tokens[0] = "MalNetTinyFeaturesCommon"
    common_dataset_name = "+".join(dataset_name_tokens)

    global mntf_common_test_loader
    if mntf_common_test_loader is None:
        mntf_common = load_dataset(
            common_dataset_name,
            cfg.dataset.params.variant,
            Path(cfg.injected.base_dir) / "datasets" / "original" / "mntf",
        )
        mntf_common_test_loader = create_loader(
            mntf_common,
            cfg.train.batch_size,
            split="test",
            shuffle=False,
        )
        
    return evaluate(model, mntf_common_test_loader, cfg.device)
