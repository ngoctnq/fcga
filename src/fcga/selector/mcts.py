import json
import math
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple


class MCTSNode:
    __slots__ = ('nodes', 'parent', 'children', 'visit_count', 'total_reward')

    def __init__(self, nodes: frozenset, parent: Optional['MCTSNode'] = None):
        self.nodes = nodes
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visit_count: int = 0
        self.total_reward: float = 0.0

    def ucb_score(self, exploration_weight: float) -> float:
        if self.visit_count == 0:
            return float('inf')
        return (self.total_reward / self.visit_count
                + exploration_weight * math.sqrt(math.log(self.parent.visit_count) / self.visit_count))

    def is_fully_expanded(self, all_nodes: List[int], graph_adj: Optional[Dict[int, Set[int]]]) -> bool:
        added = {next(iter(child.nodes - self.nodes)) for child in self.children}
        return all(c in added for c in _get_candidates(self.nodes, all_nodes, graph_adj))


def _get_candidates(
    current: frozenset,
    all_nodes: List[int],
    graph_adj: Optional[Dict[int, Set[int]]],
) -> List[int]:
    """Connected neighbors of current set; falls back to all unselected if none or no adjacency."""
    if graph_adj is not None and current:
        connected = set()
        for n in current:
            connected.update(graph_adj.get(n, set()))
        candidates = [n for n in connected if n not in current]
        if candidates:
            return candidates
    return [n for n in all_nodes if n not in current]


def _select(root: MCTSNode, exploration_weight: float, all_nodes: List[int], graph_adj: Optional[Dict[int, Set[int]]]) -> MCTSNode:
    node = root
    while node.children:
        if not node.is_fully_expanded(all_nodes, graph_adj):
            return node
        node = max(node.children, key=lambda c: c.ucb_score(exploration_weight))
    return node


def _expand(node: MCTSNode, all_nodes: List[int], graph_adj: Optional[Dict[int, Set[int]]]) -> MCTSNode:
    added = {next(iter(child.nodes - node.nodes)) for child in node.children}
    candidates = [n for n in _get_candidates(node.nodes, all_nodes, graph_adj) if n not in added]
    child = MCTSNode(node.nodes | frozenset({random.choice(candidates)}), parent=node)
    node.children.append(child)
    return child


def _simulate(node: MCTSNode, target_size: int, all_nodes: List[int], graph_adj: Optional[Dict[int, Set[int]]]) -> frozenset:
    nodes = set(node.nodes)
    while len(nodes) < target_size:
        candidates = _get_candidates(frozenset(nodes), all_nodes, graph_adj)
        if not candidates:
            break
        nodes.add(random.choice(candidates))
    return frozenset(nodes)


def _backpropagate(node: Optional[MCTSNode], reward: float) -> None:
    while node is not None:
        node.visit_count += 1
        node.total_reward += reward
        node = node.parent


def run_mcts(
    all_nodes: List[int],
    target_size: int,
    reward_fn: Callable[[List[frozenset]], List[float]],
    n_iterations: int = 100,
    exploration_weight: float = 1.0,
    expand_atoms: int = 1,
    graph_adj: Optional[Dict[int, Set[int]]] = None,
) -> List[Tuple[frozenset, float]]:
    """
    Growing MCTS over node subsets to find the highest-scoring subgraph of size `target_size`.

    Starts from the empty set and adds one node per step. When graph_adj is provided,
    candidate nodes at each step are restricted to those adjacent to the current subgraph;
    if none exist (isolated or complete subgraph), the restriction is lifted.

    Returns all (terminal, score) pairs evaluated during the search. The caller selects
    the best (argmax) and may cache the full list for later inspection or reuse.

    Args:
        all_nodes: all node indices available for selection
        target_size: number of nodes to select
        reward_fn: batched evaluator that takes node subsets (frozensets) and returns a
            score per subset. All coalitions from one rollout are passed together.
        n_iterations: number of search iterations
        exploration_weight: UCB1 exploration constant (higher = more exploration)
        expand_atoms: maximum children expanded per iteration
        graph_adj: undirected adjacency dict for connectivity-constrained expansion
    """
    root = MCTSNode(frozenset())
    all_evaluations: List[Tuple[frozenset, float]] = []

    pending_mcts_nodes: List[MCTSNode] = []
    pending_terminals: List[frozenset] = []

    def flush() -> None:
        rewards = reward_fn(pending_terminals)
        for mcts_node, terminal, reward in zip(pending_mcts_nodes, pending_terminals, rewards):
            all_evaluations.append((terminal, reward))
            _backpropagate(mcts_node, reward)
        pending_mcts_nodes.clear()
        pending_terminals.clear()

    for _ in range(n_iterations):
        node = _select(root, exploration_weight, all_nodes, graph_adj)

        if len(node.nodes) >= target_size or node.is_fully_expanded(all_nodes, graph_adj):
            # evaluate terminal nodes directly; otherwise complete the rollout
            terminal = node.nodes if len(node.nodes) >= target_size else _simulate(node, target_size, all_nodes, graph_adj)
            pending_mcts_nodes.append(node)
            pending_terminals.append(terminal)
        else:
            for _ in range(expand_atoms):
                if node.is_fully_expanded(all_nodes, graph_adj):
                    break
                child = _expand(node, all_nodes, graph_adj)
                terminal = child.nodes if len(child.nodes) >= target_size else _simulate(child, target_size, all_nodes, graph_adj)
                pending_mcts_nodes.append(child)
                pending_terminals.append(terminal)

        flush()

    if pending_mcts_nodes:
        flush()

    return all_evaluations


def mcts_selector(
    dataset,
    graph_idxs: List[int],
    cache_dir: Optional[str] = None,  # type: ignore
    suffix_dir: Optional[str] = None,
    node_budget: int = -1,
    get_node_values_fn: Optional[Callable] = None,
    check_cache_only: bool = False,
    n_iterations: int = 100,
    exploration_weight: float = 1.0,
    expand_atoms: int = 1,
) -> Optional[Dict[int, List[int]]]:
    """
    Select the highest-scoring subgraph found by MCTS for each graph.

    get_node_values_fn evaluates subgraphs with signature:
        fn(dataset, graph_idx: int, node_subsets: List[frozenset]) -> List[float]

    Args:
        dataset: PyG graph dataset
        graph_idxs: list of graph indices to run MCTS on
        cache_dir: directory to cache evaluated node subsets and their scores
        suffix_dir: subdirectory under cache_dir
        node_budget: number of nodes to select per graph (-1 = all)
        get_node_values_fn: batched subgraph evaluator
        check_cache_only: return None if any graph is missing from cache
        n_iterations: number of search iterations per graph
        exploration_weight: UCB1 exploration constant
        expand_atoms: maximum children expanded per iteration
    """
    assert node_budget >= -1

    if node_budget == -1:
        return {idx: list(range(dataset[idx].num_nodes)) for idx in graph_idxs}

    if cache_dir is not None:
        cache_dir: Path = Path(cache_dir)
        if suffix_dir is not None:
            cache_dir = cache_dir / suffix_dir
        if not cache_dir.exists():
            if check_cache_only:
                return None
            cache_dir.mkdir(parents=True, exist_ok=True)
    elif check_cache_only:
        return None

    selected_nodes: Dict[int, List[int]] = {}

    if cache_dir is not None:
        for graph_idx in graph_idxs:
            cache_path = cache_dir / f"{graph_idx}.json"
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    evaluations: List[Tuple[List[int], float]] = json.load(f)
                best_nodes, _ = max(evaluations, key=lambda x: x[1])
                selected_nodes[graph_idx] = best_nodes

    missing = [idx for idx in graph_idxs if idx not in selected_nodes]

    if missing:
        if check_cache_only:
            return None
        if get_node_values_fn is None:
            raise ValueError("get_node_values_fn must be provided when cache is incomplete")

        for graph_idx in missing:
            graph_data = dataset[graph_idx]
            all_nodes = list(range(graph_data.num_nodes))

            if node_budget >= len(all_nodes):
                selected_nodes[graph_idx] = all_nodes
                continue

            ei = graph_data.edge_index
            graph_adj: Dict[int, Set[int]] = {}
            for u, v in zip(ei[0].tolist(), ei[1].tolist()):
                graph_adj.setdefault(u, set()).add(v)
                graph_adj.setdefault(v, set()).add(u)

            def reward_fn(subsets: List[frozenset], _idx=graph_idx) -> List[float]:
                return get_node_values_fn(dataset, _idx, subsets)

            evaluations = run_mcts(
                all_nodes=all_nodes, # type: ignore
                target_size=node_budget,
                reward_fn=reward_fn,
                n_iterations=n_iterations,
                exploration_weight=exploration_weight,
                expand_atoms=expand_atoms,
                graph_adj=graph_adj,
            )
            best_nodes, _ = max(evaluations, key=lambda x: x[1])
            selected_nodes[graph_idx] = sorted(best_nodes)

            if cache_dir is not None:
                cache_path = cache_dir / f"{graph_idx}.json"
                with open(cache_path, "w") as f:
                    json.dump([[sorted(t), s] for t, s in evaluations], f)

    return selected_nodes
