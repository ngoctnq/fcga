import torch
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from src.utils import clear_gpu_cache


def _create_perturbations(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    indices_to_perturb: torch.Tensor,
    num_samples: int,
    percentage: float = 50.0,
):
    """
    Create num_samples perturbations on x.device.

    Return CPU seed records of shape (num_samples, num_nodes) and a list of
    perturbed Data objects.
    """
    num_nodes = x.shape[0]
    n_cand = len(indices_to_perturb)

    perturb_mask = torch.randint(0, 100, (num_samples, n_cand), device=x.device) < int(percentage)

    seed_records = torch.zeros(num_samples, num_nodes, device=x.device)
    seed_records[:, indices_to_perturb] = perturb_mask.float()

    random_feats = torch.randint(
        0, 2, (num_samples, num_nodes, x.shape[1]), dtype=x.dtype, device=x.device
    )
    x_perturbed = torch.where(
        seed_records.unsqueeze(-1).bool(),
        random_feats,
        x.unsqueeze(0).expand(num_samples, -1, -1),
    )  # [S, N, F]

    perturbed_list = [Data(x=x_perturbed[s], edge_index=edge_index) for s in range(num_samples)]
    return seed_records.cpu(), perturbed_list


def _apply_binarize(samples: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Binarize the prediction-change column in-place: top 1/8 -> 1, rest -> 0."""
    top = max(1, samples.shape[0] // 8)
    top_idx = torch.argsort(samples[:, num_nodes])[-top:]
    samples[:, num_nodes] = 0
    samples[top_idx, num_nodes] = 1
    return samples


def _chi_square_pvalues(samples: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Vectorized Pearson chi-square independence test for all nodes vs the
    prediction-change column, matching scipy.stats.chi2_contingency with
    correction=True (Yates) and lambda_='pearson'.
    Returns a [num_nodes] CPU tensor of p-values.
    """
    X = samples[:, :num_nodes].float()
    Y = samples[:, num_nodes].float()

    n11 = (X * Y.unsqueeze(1)).sum(0)
    n10 = (X * (1 - Y).unsqueeze(1)).sum(0)
    n01 = ((1 - X) * Y.unsqueeze(1)).sum(0)
    n00 = ((1 - X) * (1 - Y).unsqueeze(1)).sum(0)

    N = float(samples.shape[0])
    R0, R1 = n00 + n01, n10 + n11
    C0, C1 = n00 + n10, n01 + n11

    obs = torch.stack([n11, n10, n01, n00])
    exp = torch.stack([R1 * C1, R1 * C0, R0 * C1, R0 * C0]) / N

    chi2 = (torch.clamp((obs - exp).abs() - 0.5, min=0.0) ** 2 / (exp + 1e-10)).sum(0)
    chi2[(R0 == 0) | (R1 == 0) | (C0 == 0) | (C1 == 0)] = 0.0

    return torch.igammac(torch.full_like(chi2, 0.5), chi2 / 2)


def get_node_values(dataset, cfg, graph_idxs=None) -> dict[int, dict[int, float]]:
    """
    Return PGMExplainer node scores for graphs in cfg.injected.trainloader.

    Each round uses NUM_SAMPLES forward passes, with one perturbation per
    nontrivial graph in each pass. Importance is 1 - p_value.
    dataset and graph_idxs are unused and retained for the selector interface.
    """
    NUM_SAMPLES = 50

    model = cfg.injected.model
    model.eval()
    new_node_values = {}

    for batch in tqdm(cfg.injected.trainloader, desc="Running PGMExplainer"):
        batch = batch.to(cfg.device)

        # Original predictions for all graphs in one forward pass
        with torch.no_grad():
            orig_probs = torch.softmax(
                model(batch.x, batch.edge_index, batch.batch), dim=1
            )  # [B, C]
        pred_labels = orig_probs.argmax(dim=1)  # [B]
        num_classes = orig_probs.shape[1]

        # Split batch into per-graph tensors; handle trivial graphs immediately
        valid_graphs = []
        for i, graph_idx in enumerate(batch.global_idx.tolist()):
            node_start = batch.ptr[i].item()
            node_end   = batch.ptr[i + 1].item()
            num_nodes  = node_end - node_start
            if num_nodes <= 1:
                new_node_values[graph_idx] = {j: 1.0 for j in range(num_nodes)}
                continue
            x_i = batch.x[node_start:node_end]
            edge_mask    = batch.batch[batch.edge_index[0]] == i
            edge_index_i = batch.edge_index[:, edge_mask] - node_start
            valid_graphs.append((i, graph_idx, x_i, edge_index_i, num_nodes))

        if not valid_graphs:
            continue

        G = len(valid_graphs)

        # Round 1: perturb all nodes
        seed_records_r1 = [torch.zeros(NUM_SAMPLES, num_nodes) for *_, num_nodes in valid_graphs]
        preds_r1 = torch.zeros(G, NUM_SAMPLES, num_classes, device=cfg.device)

        with torch.no_grad():
            for s in range(NUM_SAMPLES):
                sample_graphs = []
                for g, (batch_i, graph_idx, x_i, edge_index_i, num_nodes) in enumerate(valid_graphs):
                    all_nodes = torch.arange(num_nodes, device=cfg.device)
                    seed_record, perturbed = _create_perturbations(x_i, edge_index_i, all_nodes, 1)
                    seed_records_r1[g][s] = seed_record[0]
                    sample_graphs.append(perturbed[0])
                batch_s = Batch.from_data_list(sample_graphs).to(cfg.device)
                preds_r1[:, s, :] = torch.softmax(
                    model(batch_s.x, batch_s.edge_index, batch_s.batch), dim=1
                )
                del batch_s, sample_graphs
        clear_gpu_cache()

        # use the first round's p-values to select candidate nodes
        r2_targets = []
        for idx, (batch_i, graph_idx, x_i, edge_index_i, num_nodes) in enumerate(valid_graphs):
            pred_label  = pred_labels[batch_i].item()
            pred_change = (orig_probs[batch_i].max() - preds_r1[idx, :, pred_label]).cpu()

            samples_r1 = _apply_binarize(
                torch.cat([seed_records_r1[idx], pred_change.unsqueeze(1)], dim=1), num_nodes
            )
            p_values_r1 = _chi_square_pvalues(samples_r1, num_nodes)

            max_subgraph_size = max(1, min(
                max(cfg.augmentation.params.node_budget * 2, num_nodes // 20),
                num_nodes - 1,
            ))
            candidate_nodes = torch.topk(p_values_r1, max_subgraph_size, largest=False).indices.to(cfg.device)
            r2_targets.append((batch_i, graph_idx, x_i, edge_index_i, num_nodes, candidate_nodes))

        del preds_r1, seed_records_r1
        clear_gpu_cache()

        # Round 2: perturb candidate nodes
        seed_records_r2 = [torch.zeros(NUM_SAMPLES, num_nodes) for *_, num_nodes, _ in r2_targets]
        preds_r2 = torch.zeros(G, NUM_SAMPLES, num_classes, device=cfg.device)

        with torch.no_grad():
            for s in range(NUM_SAMPLES):
                sample_graphs = []
                for g, (batch_i, graph_idx, x_i, edge_index_i, num_nodes, candidate_nodes) in enumerate(r2_targets):
                    seed_record, perturbed = _create_perturbations(x_i, edge_index_i, candidate_nodes, 1)
                    seed_records_r2[g][s] = seed_record[0]
                    sample_graphs.append(perturbed[0])
                batch_s = Batch.from_data_list(sample_graphs).to(cfg.device)
                preds_r2[:, s, :] = torch.softmax(
                    model(batch_s.x, batch_s.edge_index, batch_s.batch), dim=1
                )
                del batch_s, sample_graphs
        clear_gpu_cache()

        for idx, (batch_i, graph_idx, x_i, edge_index_i, num_nodes, candidate_nodes) in enumerate(r2_targets):
            pred_label  = pred_labels[batch_i].item()
            pred_change = (orig_probs[batch_i].max() - preds_r2[idx, :, pred_label]).cpu()

            samples_r2 = _apply_binarize(
                torch.cat([seed_records_r2[idx], pred_change.unsqueeze(1)], dim=1), num_nodes
            )
            p_values_r2 = _chi_square_pvalues(samples_r2, num_nodes)

            new_node_values[graph_idx] = {
                node_idx: 1.0 - p_values_r2[node_idx].item()
                for node_idx in range(num_nodes)
            }

        del preds_r2, seed_records_r2
        clear_gpu_cache()

    clear_gpu_cache()
    return new_node_values
