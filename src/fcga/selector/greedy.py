import json
from pathlib import Path
from typing import Callable, Optional, List, Dict
from random import shuffle


def load_node_values_from_cache(
    cache_dir: Path,
    graph_idxs: List[int]
) -> Optional[Dict[int, Dict[int, float]]]:
    '''
    Load cached node values, or return None if any graph is missing.
    '''
    return_dict = {}
    for graph_idx in graph_idxs:
        cache_path = cache_dir / f"{graph_idx}.json"
        if cache_path.exists():
            with open(cache_path, "r") as f:
                node_values: dict[int, float] = json.load(f)
            # force all keys to be int (json converts them to str)
            node_values = {int(k): v for k, v in node_values.items()}
            return_dict[graph_idx] = node_values
        else:
            return None
    return return_dict


def save_node_values_to_cache(
    node_values: Dict[int, Dict[int, float]],
    cache_dir: Path,
):
    '''
    Write one JSON file per graph, replacing existing cached values.
    '''
    for graph_idx, node_value_dict in node_values.items():
        cache_path = cache_dir / f"{graph_idx}.json"
        with open(cache_path, "w") as f:
            json.dump(node_value_dict, f)
        

def greedy_selector(
    dataset,
    graph_idxs: List[int],
    cache_dir: Optional[str]=None, # type: ignore
    suffix_dir: Optional[str]=None,
    node_budget: int = -1,
    get_node_values_fn: Optional[Callable] = None,
    check_cache_only: bool = False,
) -> Optional[Dict[int, List[int]]]:
    """
    Select up to node_budget nodes per graph by descending score.

    A budget of -1 selects every node; 0 selects none. If get_node_values_fn is
    None, use a random node ordering. Otherwise, call it with dataset and
    graph_idxs to obtain a mapping from graph indices to per-node scores.

    Store scores or random orderings under cache_dir/suffix_dir when provided.
    Return a mapping from graph indices to selected node lists, or None when
    check_cache_only is True and the required cache is incomplete.
    """
    assert node_budget >= -1, "Node budget cannot be less than -1"

    selected_nodes = {}
    if node_budget == -1:
        for graph_idx in graph_idxs:
            selected_nodes[graph_idx] = list(range(dataset[graph_idx].num_nodes))
        return selected_nodes

    if cache_dir is not None:
        cache_dir: Path = Path(cache_dir)
        if suffix_dir is not None:
            cache_dir = cache_dir / suffix_dir

        if not cache_dir.exists():
            if check_cache_only:
                return None
            cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        if check_cache_only:
            return None

    if get_node_values_fn is None:
        # Reuse cached orderings across runs and node budgets.
        if cache_dir is not None:
            cache_path = Path(cache_dir) / "random.json"
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    selected_nodes = json.load(f)
                # convert keys back to int
                selected_nodes = {
                    int(k): v[:node_budget] for k, v in selected_nodes.items()
                    if int(k) in graph_idxs
                }
                if len(selected_nodes) == len(graph_idxs):
                    print("Loaded selected nodes from cache.")
                    return selected_nodes

        if check_cache_only:
            return None

        print("Warning: running random selection from scratch.")
        for graph_idx in graph_idxs:
            num_nodes = dataset[graph_idx].num_nodes
            node_list = list(range(num_nodes))
            shuffle(node_list)
            selected_nodes[graph_idx] = node_list

        # Cache the full ordering before applying the node budget.
        if cache_dir is not None:
            cache_path = Path(cache_dir) / "random.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(selected_nodes, f)

        for graph_idx in selected_nodes:
            selected_nodes[graph_idx] = selected_nodes[graph_idx][:node_budget]
        return selected_nodes

    # Graphs within the budget do not need scoring.
    large_idxs = []
    for graph_idx in graph_idxs:
        num_nodes = dataset[graph_idx].num_nodes
        if num_nodes <= node_budget:
            selected_nodes[graph_idx] = list(range(num_nodes))
        else:
            large_idxs.append(graph_idx)
    graph_idxs = large_idxs
    if not graph_idxs:
        return selected_nodes

    # Score graphs missing from the cache.
    if cache_dir is not None:
        missing = [idx for idx in graph_idxs if not (cache_dir / f"{idx}.json").exists()]
    else:
        missing = list(graph_idxs)

    if missing:
        if check_cache_only:
            return None
        new_values = get_node_values_fn(dataset, graph_idxs=missing)
        if cache_dir is not None:
            save_node_values_to_cache(new_values, cache_dir)

    if cache_dir is not None:
        node_values = load_node_values_from_cache(cache_dir, graph_idxs)
        if node_values is None:
            return None
    else:
        node_values = get_node_values_fn(dataset, graph_idxs=graph_idxs)
    
    for k, v in node_values.items():
        top_node_values = sorted(v.items(), key=lambda x: x[1], reverse=True)[:node_budget]
        selected_nodes[k] = [node_idx for node_idx, _ in top_node_values]
    return selected_nodes
