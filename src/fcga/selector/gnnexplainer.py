from torch_geometric.explain import GNNExplainer, Explainer
from src.utils import clear_gpu_cache

from tqdm import tqdm


def get_node_values(
    dataset,
    cfg,
    graph_idxs=None,
) -> dict[int, dict[int, float]]:
    '''
    Return GNNExplainer node scores for every graph in cfg.injected.trainloader.

    dataset and graph_idxs are unused and retained for the selector interface.
    Results map graph indices to dictionaries of node indices and scores.
    '''
    new_node_values = {}
    model = cfg.injected.model

    algorithm = GNNExplainer()
    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type='model', # wrt model behavior and not label
        node_mask_type='object', # one scalar importance score per node
        edge_mask_type=None,
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='log_probs',
        ),
    )

    model.eval()
    for batch in tqdm(cfg.injected.trainloader, desc="Running GNNExplainer"):
        batch = batch.to(cfg.device)
        explanation = explainer(
            x=batch.x,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        # node_mask: [num_nodes, 1] -> squeeze to [num_nodes]
        node_mask = explanation.node_mask.squeeze(-1)
        for i, graph_idx in enumerate(batch.global_idx):
            new_node_values[graph_idx] = {}
            for node_idx in range(batch.ptr[i + 1] - batch.ptr[i]):
                new_node_values[graph_idx][node_idx] = node_mask[batch.ptr[i] + node_idx].item()

    del explainer
    del algorithm
    clear_gpu_cache()

    return new_node_values
