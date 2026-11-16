import torch


def get_f1_score(confusion_matrix):
    '''
    Compute precision, recall, and F1-score from the confusion matrix.
    '''
    tp = confusion_matrix.diag()
    fp = confusion_matrix.sum(dim=0) - tp
    fn = confusion_matrix.sum(dim=1) - tp
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1


@torch.inference_mode()
def evaluate(model, dataloader, device):
    '''
    Evaluate the model on the given dataloader and compute metrics.
    Returns a dictionary of metrics: accuracy and confusion matrix.

    :param model: The trained model to evaluate.
    :param dataloader: DataLoader for the evaluation dataset.
    :param device: Device to run the evaluation on (e.g., 'cpu' or 'cuda').
    :return: Dictionary containing 'accuracy' and 'confusion_matrix'.
    '''
    model.eval()
    num_classes = dataloader.dataset.num_classes
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int).to(device)
    correct = 0
    total = 0

    # get the normalization factor for distance to boundary
    normalization_factor = get_normalization_factor(model)

    # accumulate distances to decision boundary towards the true class, or the second best class if the prediction is correct
    # keep as a dictionary of indices for ease of merging with other splits later
    gaps_to_boundary = {}

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        outputs = model(batch.x, batch.edge_index, batch.batch)
        preds = outputs.argmax(dim=1)
        confusion_matrix += torch.bincount(batch.y * num_classes + preds, minlength=num_classes**2).reshape(num_classes, num_classes)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)
        gap_to_boundary = get_margin(outputs, batch.y, normalization_factor)
        
        # check if the shape is identical
        assert batch.global_idx.numel() == gap_to_boundary.numel(), f"Expected batch.global_idx and gap_to_boundary to have the same number of elements, but got {batch.global_idx.numel()} and {gap_to_boundary.numel()} respectively."
        for data_idx, dup_idx, gap in zip(batch.global_idx.tolist(), batch.dup_idx.tolist(), gap_to_boundary.tolist()):
            if data_idx not in gaps_to_boundary:
                gaps_to_boundary[data_idx] = {}
            gaps_to_boundary[data_idx][dup_idx] = gap

    # f1-score
    precision, recall, f1_score = get_f1_score(confusion_matrix)

    eval_stats = {
        "accuracy": correct / total if total > 0 else 0,
        "confusion_matrix": confusion_matrix.cpu().numpy().tolist(),
        "precision": precision.cpu().numpy().tolist(),
        "recall": recall.cpu().numpy().tolist(),
        "macro_f1": f1_score.mean().item(),
        "margin": gaps_to_boundary
    }

    return eval_stats


def get_normalization_factor(model):
    # get the last Linear layer -- note you can probably iterate through .modules();
    # otherwise make sure any new architectures have a Sequential .post_mp with a Linear as the last layer
    last_linear_layer = model.post_mp[-1]
    assert isinstance(last_linear_layer, torch.nn.Linear), "Expected the last layer of post_mp to be a Linear layer for normalization factor calculation."
    biased_weights = last_linear_layer.weight.data.clone() # out x in
    if last_linear_layer.bias is not None:
        biased_weights = torch.cat([biased_weights, last_linear_layer.bias.data.unsqueeze(1)], dim=1) # out x (in + 1)
    diff = biased_weights.unsqueeze(1) - biased_weights.unsqueeze(0) # out x out x (in + 1)
    normalization_factor = torch.norm(diff, dim=2) # out x out
    normalization_factor[normalization_factor == 0] = 1e-8 # avoid division by zero for identical weight vectors
    return normalization_factor


def get_margin(outputs, labels, normalization_factor):
    '''
    Compute the margin (distance to decision boundary) for each sample in the batch.
    For correctly classified samples, compute the margin to the second best class.
    
    The normalization_factor is the L2-distance between the weight vectors of the classes.
    '''
    masked_outputs = outputs - 1e6 * torch.eye(outputs.size(1), device=outputs.device)[labels] # arbitrary large number
    idxs_2nd_best = masked_outputs.argmax(dim=1)
    return (
        (outputs[torch.arange(labels.size(0)), labels] - masked_outputs[torch.arange(labels.size(0)), idxs_2nd_best])
        / normalization_factor[labels, idxs_2nd_best]
    )


def lite_eval_for_print(eval_stats) -> str:
    '''
    Extracts and formats key metrics from eval_stats for easy printing.
    '''
    return f"Accuracy: {eval_stats['accuracy']:.4f}, Macro F1: {eval_stats['macro_f1']:.4f}"
