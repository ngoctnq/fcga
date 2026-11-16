from functools import partial


def get_selector(cfg):
    '''
    Build a selector with signature:
        selector(dataset, graph_idxs, cache_dir=None, check_cache_only=False)
    '''
    criteria = cfg.augmentation.params.criteria
    granularity = cfg.augmentation.params.granularity
    is_subgraph = granularity in {"subgraph", "subgraph_summary"}

    if criteria in ["degree", "pagerank", "in_degree", "out_degree", "betweenness", "closeness", "ppr"]:
        if is_subgraph:
            from .degree_subgraph import get_degree_subgraph_scores
            get_node_value_fn = partial(get_degree_subgraph_scores, criteria=criteria)
            suffix_dir = f"{criteria}_subgraph"
        else:
            from .degree import get_node_values
            get_node_value_fn = partial(get_node_values, criteria=criteria)
            suffix_dir = criteria

    elif criteria in ["random", "all"]:
        get_node_value_fn = None
        if criteria == "all":
            assert cfg.augmentation.params.node_budget == -1, "Node budget must be -1 for 'all' criteria"
        suffix_dir = None

    else:
        suffix_dir = criteria

        if criteria in ["gnnexplainer"]:
            # cache_dir is patched later
            from .gnnexplainer import get_node_values
            get_node_value_fn = partial(get_node_values, cfg=cfg)

        elif criteria in ["pgmexplainer"]:
            from .pgmexplainer import get_node_values
            get_node_value_fn = partial(get_node_values, cfg=cfg)

        elif criteria in ["subgraphx", "shapley"]:
            from .shapley import get_shapley_scores
            suffix_dir = "shapley"  # Both criteria use the same Shapley scores.
            get_node_value_fn = partial(get_shapley_scores, cfg=cfg, n_samples=cfg.augmentation.params.mcts_samples)

        elif criteria in ["output_drop"]:
            from .output_drop import get_output_drop_scores
            get_node_value_fn = partial(get_output_drop_scores, cfg=cfg)

        else:
            raise NotImplementedError(f"Unsupported criteria: {criteria}")

    if granularity == "function":
        from .greedy import greedy_selector as selector
    elif is_subgraph:
        from .mcts import mcts_selector as selector
    else:
        raise NotImplementedError(f"Unsupported granularity: {granularity}")

    additional_kwargs = {
        "node_budget": cfg.augmentation.params.node_budget,
        "get_node_values_fn": get_node_value_fn,
    }

    if suffix_dir is not None:
        additional_kwargs["suffix_dir"] = suffix_dir

    if is_subgraph:
        additional_kwargs["n_iterations"] = cfg.augmentation.params.mcts_iterations
        additional_kwargs["exploration_weight"] = cfg.augmentation.params.mcts_exploration
        additional_kwargs["expand_atoms"] = cfg.augmentation.params.mcts_expand_atoms

    return partial(selector, **additional_kwargs)
