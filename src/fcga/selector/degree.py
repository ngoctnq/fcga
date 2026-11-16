import networkx as nx
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from src.fcga.rewriter.mntf.utils import get_function_signature
import re


def _compute_node_values_single(args):
    graph_nx, graph_idx, criteria, personalization = args
    if criteria == "degree":
        node_values = dict(nx.degree_centrality(graph_nx))
    elif criteria == "pagerank":
        node_values = dict(nx.pagerank(graph_nx))
    elif criteria == "in_degree":
        node_values = dict(graph_nx.in_degree())
    elif criteria == "out_degree":
        node_values = dict(graph_nx.out_degree())
    elif criteria == "betweenness":
        node_values = nx.betweenness_centrality(graph_nx)
    elif criteria == "closeness":
        node_values = nx.closeness_centrality(graph_nx)
    elif criteria == "ppr":
        node_values = dict(nx.pagerank(graph_nx, personalization=personalization))
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
    return graph_idx, node_values


def get_node_values(
    dataset,
    graph_idxs: list[int],
    criteria: str,
) -> dict[int, dict[int, float]]:
    graphs = [to_networkx(dataset[idx]) for idx in graph_idxs]
    if criteria == "ppr":
        personalizations = [
            get_restart_nodes(dataset, idx, g.number_of_nodes())
            for idx, g in zip(graph_idxs, graphs)
        ]
    else:
        personalizations = [None] * len(graph_idxs)
    args = list(zip(graphs, graph_idxs, [criteria] * len(graph_idxs), personalizations))
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return dict(tqdm(map(_compute_node_values_single, args), total=len(args)))
    with ProcessPoolExecutor(max_workers=32) as executor:
        return dict(tqdm(executor.map(_compute_node_values_single, args), total=len(args)))


def get_restart_nodes(dataset, data_idx, num_nodes):
    source_codes = dataset.get_source_codes({data_idx: list(range(num_nodes))})
    signatures = [re.search(r'(\S*)\(([^)]*)\)', get_function_signature(src)) for src in source_codes[data_idx]]
    is_entrypoint = {}
    for node_idx, match in enumerate(signatures):
        assert match is not None, "Signature not found!"
        if match.groups()[0] in [
            'onStartCommand', 'onBind', 'onReceive', 'onCreate', 'onResume',
            'onNewIntent', 'onHandleIntent', 'onStartJob', 'onAccessibilityEvent'
        ]:
            is_entrypoint[node_idx] = 1
    found_count = len(is_entrypoint)
    for node_idx in range(num_nodes):
        if node_idx not in is_entrypoint:
            is_entrypoint[node_idx] = 1 / (num_nodes - found_count)
    return is_entrypoint
