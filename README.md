# Defending FCG-Based Malware Detectors Against Metamorphic Android Variants via LLM-Powered Code Refactoring

## Overview

Function Call Graph Augmentation (FCGA) augments training data for Android malware detectors based on graph neural networks (GNNs). Each application is represented as a function call graph: nodes represent functions, node features encode their content, and edges represent calls between them. FCGA augments these representations through large language model (LLM) rewriting and refactoring, or direct perturbation of node embeddings.

We provide three augmentation variants:

- **FCGA-F (function-level)** rewrites selected functions and replaces their node features with embeddings of the rewritten source, keeping the graph topology unchanged.
- **FCGA-P (perturbation-based)** adds analytically scaled Gaussian noise to selected node embeddings under a cosine-similarity constraint. It preserves graph topology and requires no LLM rewriting or re-embedding. This variant is named `cosine` in the code and configuration files.
- **FCGA-S (subgraph-level)** refactors selected subgraphs using function summaries and surrounding call relationships. It updates the graph topology and uses embeddings of generated summaries for new or refactored nodes.

The repository includes training, fine-tuning, and evaluation workflows, with unaugmented training and node/edge dropping as comparison methods.

## Installation

### Python environment

The commands below assume Linux, Python 3.10 or newer, and an NVIDIA GPU with CUDA support.

```bash
git clone https://github.com/ngoctnq/fcga.git
cd fcga
python -m venv .venv
source .venv/bin/activate
```

Install a CUDA-enabled build of the PyTorch version listed in [requirements.txt](requirements.txt), using the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/). Choose the CUDA build appropriate for your system, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt --no-build-isolation
```

The embedding pipeline uses FlashAttention 2. PyTorch must be installed before building FlashAttention; see its [installation instructions](https://github.com/Dao-AILab/flash-attention/tree/v2.8.3#installation-and-features) for supported GPUs and CUDA build requirements.

Check that the command-line entry point loads:

```bash
python main.py --help
```

### Ollama and models

FCGA-F and FCGA-S use [Ollama](https://docs.ollama.com/linux) for function rewriting and subgraph refactoring. FCGA-P and node/edge-drop augmentation do not require an Ollama server.

After installing Ollama, start a server in a separate terminal:

```bash
OLLAMA_HOST=127.0.0.1:8080 ollama serve
```

In another terminal, download the [Qwen2.5-Coder](https://ollama.com/library/qwen2.5-coder) model used by the default configurations:

```bash
OLLAMA_HOST=127.0.0.1:8080 ollama pull qwen2.5-coder
```

The supplied augmentation configurations use `http://localhost:8080`. If your server runs elsewhere, update `params.ollama_urls` in the augmentation YAML to match. Keep the Ollama server running while generating FCGA-F or FCGA-S augmentations.

## Dataset preparation

FCGA uses the Tiny and Common splits of [MalNet-Tiny-Features](https://huggingface.co/datasets/ngoctnq/malnet-features). Prepare both datasets before running standard training, which also evaluates on Common.

With the supplied dataset configurations, place the processed files in these directories relative to the repository root:

```text
datasets/original/mntf/
  MalNetTinyFeatures/processed/prune+llm+qwen_keepisolated/
  MalNetTinyFeaturesCommon/processed/prune+llm+qwen_keepisolated/
```

Each directory must contain `data.pt`, `split_dict.pt`, `keep_idxs_list.pt`, `mask_list.pt`, and `hash_list.pt`.

**Update:** The current HuggingFace release uses a newer format from the FCGA loader. Please construct the dataset using the [instructions in the dataset's GitHub repository](https://github.com/ngoctnq/malnet-features#feature-construction-dataset-construction) instead of downloading the precomputed data directly from Hugging Face. A compatibility patch is already in progress.

Generating FCGA-F or FCGA-S augmentations also requires the corresponding decompiled function-source files (`<hash>.pkl`). Set `params.raw_dir` in the dataset YAML to the directory containing these files.

## Quick start

After completing [installation](#installation) and [dataset preparation](#dataset-preparation), run the following commands from the repository root. Both examples train a three-layer GCN for 150 epochs.

Train a baseline without augmentation:

```bash
python main.py --seed 0 --device cuda:0 \
    --dataset-cfg configs/datasets/mntf.yaml \
    --augment-cfg configs/augmentations/none.yaml \
    --model-cfg configs/models/gcn_3.yaml \
    --train-cfg configs/training/standard_aug0.2.yaml
```

To train with FCGA-F, keep the Ollama server running and configure `params.raw_dir` as described above:

```bash
python main.py --seed 0 --device cuda:0 \
    --dataset-cfg configs/datasets/mntf.yaml \
    --augment-cfg configs/augmentations/mntf/fcga-f_degree.yaml \
    --model-cfg configs/models/gcn_3.yaml \
    --train-cfg configs/training/standard_aug0.2.yaml
```

## Running experiments

### Augmentation methods

To run another augmentation method, replace `--augment-cfg` in the quick-start command with one of these presets:

| Method | Augmentation configuration |
| --- | --- |
| No augmentation | `configs/augmentations/none.yaml` |
| FCGA-F | `configs/augmentations/mntf/fcga-f_degree.yaml` |
| FCGA-P | `configs/augmentations/cosine/cosine_degree_0.8_static.yaml` |
| FCGA-S | `configs/augmentations/mntf/fcga-s_subgraphx.yaml` |
| Node drop | `configs/augmentations/drop/drop_node0.1.yaml` |
| Edge drop | `configs/augmentations/drop/drop_edge0.1.yaml` |
| Node and edge drop | `configs/augmentations/drop/drop_node0.1_edge0.3.yaml` |

For model-based selection, the `start75` training preset delays augmentation until after 75 training epochs. For example, run FCGA-S with SubgraphX selection:

```bash
python main.py --seed 0 --device cuda:0 \
    --dataset-cfg configs/datasets/mntf.yaml \
    --augment-cfg configs/augmentations/mntf/fcga-s_subgraphx.yaml \
    --model-cfg configs/models/gcn_3.yaml \
    --train-cfg configs/training/standard_aug0.2_start75.yaml
```

Additional selectors and augmentation settings are available under `configs/augmentations/`.

### Fine-tuning

Use `--pretrained-path` to pass a checkpoint file. This example fine-tunes the quick-start baseline on Common using FCGA-F:

```bash
python main.py --seed 0 --device cuda:0 \
    --dataset-cfg configs/datasets/mntf-cmn.yaml \
    --augment-cfg configs/augmentations/mntf/fcga-f_degree.yaml \
    --model-cfg configs/models/gcn_3.yaml \
    --train-cfg configs/training/finetune_freeze_aug0.2.yaml \
    --pretrained-path results/mntf+none+gcn_3+standard/0/best.pt
```

### Evaluation

For evaluation, `--pretrained-path` takes a run directory containing `best.pt` and/or `last.pt`. The following command evaluates the baseline checkpoints on Tiny without augmentation:

```bash
python main.py --seed 0 --device cuda:0 \
    --dataset-cfg configs/datasets/mntf.yaml \
    --augment-cfg configs/augmentations/none.yaml \
    --model-cfg configs/models/gcn_3.yaml \
    --train-cfg configs/training/evaluate.yaml \
    --pretrained-path results/mntf+none+gcn_3+standard/0
```

## Configuration guide

Each run reads four YAML files. Start from an existing preset, copy it to a new filename, and pass the edited file through the corresponding argument.

| Argument | Presets | Main settings |
| --- | --- | --- |
| `--dataset-cfg` | [Dataset configs](configs/datasets/) | Dataset `name`, preprocessing `params.variant`, and source path `params.raw_dir` |
| `--augment-cfg` | [Augmentation configs](configs/augmentations/) | Augmentation `name` and method-specific options under `params` |
| `--model-cfg` | [Model configs](configs/models/) | Model `name`, `num_layers`, `hidden_dim`, and `dropout` |
| `--train-cfg` | [Training configs](configs/training/) | `mode`, `epochs`, `batch_size`, `lr`, `weight_decay`, and augmentation schedule |

Set the random seed and device with `--seed` and `--device`, and `--pretrained-path` for specifying the checkpoint for fine-tuning or evaluation. The easiest way to run new experiments is to adapt the provided configurations in the `configs` folder.

## Citation

```
TODO: add citation when available
```
