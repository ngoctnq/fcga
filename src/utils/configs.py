from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseParams(BaseModel):
    # generic parameter set that can be defined arbitrarily
    model_config = ConfigDict(extra="allow")


# Augmentation
class FCGAParams(BaseModel):
    """FCGA-specific augmentation parameters."""
    granularity: str = "function"  # "function", "subgraph", or "subgraph_summary"

    # selector
    mcts_iterations: int = 100
    mcts_exploration: float = 1.41  # UCB exploration constant
    mcts_expand_atoms: int = 5      # children expanded per rollout (breadth)
    mcts_samples: int = 4           # Monte Carlo samples per coalition in Shapley reward
    criteria: str = "degree"  # node selection criteria for graph-level augmentation
    num_workers: int = 4  # number of parallel workers for augmentation
    node_budget: int = 5  # number of nodes to select per augmentation

    # rewriter
    rewriter: str = "qwen2.5-coder"
    context_length: int = 32_768  # maximum context length for the rewriter
    ollama_urls: list[str] = ["http://localhost:11434"]

    # embedder
    embedder: str = "qwen3"
    embedder_bsz: int = 32


class DropParams(BaseModel):
    """Parameters for node/edge/combined drop augmentation."""
    mode: str = "node"  # "node", "edge", or "both"

    node_drop_rate: float = 0.1
    node_protected_drop_rate: float = 0.0
    node_criteria: str | None = None  # None = uniform; currently only "degree" supported

    edge_drop_rate: float = 0.1
    edge_protected_drop_rate: float = 0.0
    edge_criteria: str | None = None  # None = uniform; currently only "betweenness" supported


class AugmentationConfig(BaseModel):
    name: str = "none"
    params: FCGAParams | DropParams | BaseParams = Field(default_factory=BaseParams)

    @model_validator(mode='before')
    @classmethod
    def select_params_type(cls, data):
        if isinstance(data, dict) and 'params' in data:
            name = data.get('name', '').lower()
            if name == 'fcga':
                data['params'] = FCGAParams(**data['params'])
            elif name == 'drop':
                data['params'] = DropParams(**data['params'])
            else:
                data['params'] = BaseParams(**data['params'])
        return data


# Model
class ModelConfig(BaseModel):
    name: str = "GIN"
    num_layers: int = 5
    hidden_dim: int = 64
    dropout: float = 0.5
    # injected at runtime, not from config file
    input_dim: int = Field(default=0)
    output_dim: int = Field(default=0)
    params: BaseParams = Field(default_factory=BaseParams)


# Dataset
class DatasetConfig(BaseModel):
    name: str
    params: BaseParams = Field(default_factory=BaseParams)


# Training
class TrainParams(BaseModel):
    # generic training parameters that can be defined arbitrarily
    model_config = ConfigDict(extra="allow")


class TrainConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mode: str = "standard"
    epochs: int = 150
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "Adam"
    augmentation_ratio: float = 0.2
    # epoch(s) to restart applying augmentation, used for regenerating augmentations.
    # some arbitrary notation: can specify a single epoch (e.g., 50), a list of epochs (e.g., [50, 100]), or a range (e.g., "50:100:2")
    augmentation_start_epoch: int | list[int] | tuple[int, ...] | str = 0

    params: TrainParams = Field(default_factory=TrainParams)

    # feel free to implement additional features such as learning rate scheduler, early stopping, etc.
    # scheduler: Optional[str] = None
    # scheduler_params: dict[str, Any] = Field(default_factory=dict)

    # define custom parser for augmentation_start_epoch to convert it into a list of epochs
    # 'after' mode so that it runs after the default validators and we can access the total number of epochs for validation
    @model_validator(mode='after')
    @classmethod
    def parse_augmentation_start_epoch(cls, data):
        epoch_spec = data.augmentation_start_epoch
        if isinstance(epoch_spec, int):
            assert 0 <= epoch_spec < data.epochs, "augmentation_start_epoch must be between 0 and total epochs"
            data.augmentation_start_epoch = [epoch_spec]
        elif isinstance(epoch_spec, str):
            # parse range notation
            tokens = epoch_spec.split(':')
            try:
                if len(tokens) == 2:
                    tokens.append('')  # default step
                if len(tokens) != 3:
                    raise ValueError # the error message will be caught by the except block below
                start = 0 if tokens[0] == '' else int(tokens[0])
                end = data.epochs if tokens[1] == '' else int(tokens[1])
                step = 1 if len(tokens) == 2 else int(tokens[2])
                data.augmentation_start_epoch = list(range(start, end, step))
            except (ValueError, IndexError):
                raise ValueError(f"Invalid augmentation_start_epoch format: {epoch_spec}")
        elif isinstance(epoch_spec, (list, tuple)):
            for epoch in epoch_spec:
                assert isinstance(epoch, int) and 0 <= epoch < data.epochs, "Each epoch in augmentation_start_epoch list must be between 0 and total epochs"
        else:
            raise RuntimeError("Catastrophic error: This should never happen given the type hints and the validator.")
        return data


# Top-level

class Config(BaseModel):
    seed: int = 42
    device: str = "cuda"
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig  # required
    train: TrainConfig = Field(default_factory=TrainConfig)
    # injected at runtime, will not save to config file
    injected: BaseParams = Field(default_factory=BaseParams)


def get_config(
    dataset_cfg_path: str | Path,
    augment_cfg_path: str | Path,
    model_cfg_path: str | Path,
    train_cfg_path: str | Path,
) -> Config:
    with open(dataset_cfg_path) as f:
        raw_dataset_cfg = yaml.safe_load(f)
    with open(augment_cfg_path) as f:
        raw_augment_cfg = yaml.safe_load(f)
    with open(model_cfg_path) as f:
        raw_model_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        raw_train_cfg = yaml.safe_load(f)

    return Config(
        dataset=raw_dataset_cfg,
        augmentation=raw_augment_cfg,
        model=raw_model_cfg,
        train=raw_train_cfg,
    )
