import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import networkx as nx
from torch_geometric.utils import to_networkx


def _collapse_and_score(args):
    """Collapse a coalition into a virtual node and return its centrality score."""
    graph_nx, coalition, criteria, personalization = args
    coalition = set(coalition)

    virtual = max(graph_nx.nodes()) + 1
    collapsed = nx.DiGraph()

    for node in graph_nx.nodes():
        if node not in coalition:
            collapsed.add_node(node)
    collapsed.add_node(virtual)

    for u, v in graph_nx.edges():
        u_c = virtual if u in coalition else u
        v_c = virtual if v in coalition else v
        if u_c != v_c:
            collapsed.add_edge(u_c, v_c)

    if criteria == "degree":
        scores = nx.degree_centrality(collapsed)
    elif criteria == "pagerank":
        try:
            scores = nx.pagerank(collapsed)
        except nx.PowerIterationFailedConvergence:
            n = collapsed.number_of_nodes()
            scores = {node: 1.0 / n for node in collapsed.nodes()}
    elif criteria == "in_degree":
        scores = dict(collapsed.in_degree())
    elif criteria == "out_degree":
        scores = dict(collapsed.out_degree())
    elif criteria == "betweenness":
        scores = nx.betweenness_centrality(collapsed)
    elif criteria == "closeness":
        scores = nx.closeness_centrality(collapsed)
    elif criteria == "ppr":
        # Redistribute coalition nodes' personalization weights onto the virtual node.
        # Non-coalition nodes keep their original weights; nx.pagerank normalizes internally.
        new_pers = {node: personalization.get(node, 0.0)
                    for node in graph_nx.nodes() if node not in coalition}
        new_pers[virtual] = sum(personalization.get(n, 0.0) for n in coalition)
        try:
            scores = nx.pagerank(collapsed, personalization=new_pers)
        except nx.PowerIterationFailedConvergence:
            n = collapsed.number_of_nodes()
            scores = {node: 1.0 / n for node in collapsed.nodes()}
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    return float(scores.get(virtual, 0.0))


def get_degree_subgraph_scores(
    dataset,
    graph_idx: int,
    node_subsets: List[frozenset],
    criteria: str = "degree",
) -> List[float]:
    """
    For each coalition, collapse all coalition nodes into a single virtual node
    and return the virtual node's centrality as the MCTS reward score.

    For ppr, the personalization weights of coalition nodes are summed onto the
    virtual node, preserving the entrypoint-biased restart distribution.

    Parallelised over coalitions with ProcessPoolExecutor.
    """
    if not node_subsets:
        return []

    graph_nx = to_networkx(dataset[graph_idx])

    if criteria == "ppr":
        from .degree import get_restart_nodes
        personalization = get_restart_nodes(dataset, graph_idx, dataset[graph_idx].num_nodes)
    else:
        personalization = None

    args = [(graph_nx, coalition, criteria, personalization) for coalition in node_subsets]

    if os.environ.get("CUDA_VISIBLE_DEVICES") or len(args) == 1:
        return [_collapse_and_score(a) for a in args]

    with ProcessPoolExecutor(max_workers=len(args)) as executor:
        return list(executor.map(_collapse_and_score, args))
