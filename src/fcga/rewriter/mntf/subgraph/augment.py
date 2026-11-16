import json
import re
import random
import torch
import networkx as nx
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from torch_geometric.utils import to_networkx

from src.fcga.rewriter.ollama import call_ollama
from src.fcga.selector import get_selector
from src.loaders.mntf.base import MalNetDataset
from src.fcga.embedder.qwen3 import embed_batch
from src.fcga.rewriter.mntf.utils import get_function_signature


print_ = tqdm.write


def parse_graph(s: str) -> nx.DiGraph:
    """
    Extract a graph JSON from raw LLM output and return it as a NetworkX DiGraph.
    Expected schema: {nodes: [{id, signature, summary}], edges: [[src, tgt]]}

    Works by iteratively replacing innermost {} blocks with same-length string
    placeholders, preserving character offsets into the original string. Once all
    nested objects are flattened, the outer graph object becomes the innermost
    {} and can be extracted and parsed against the original string.
    """
    original = s
    while True:
        match = re.search(r'\{[^\{\}]*\}', s)
        if not match:
            raise ValueError("No valid graph JSON found in LLM output")
        try:
            content = json.loads(match.group(0))
            if "nodes" in content and "edges" in content:
                data = json.loads(original[match.start():match.end()])
                G = nx.DiGraph()
                for node in data["nodes"]:
                    G.add_node(node["id"], **node)
                for edge in data["edges"]:
                    if len(edge) >= 2:
                        G.add_edge(edge[0], edge[1])
                return G
            # replace with same-length JSON string so char offsets into `original` are preserved
            s = (
                s[: match.start()]
                + '"' + "a" * (match.end() - match.start() - 2) + '"'
                + s[match.end():]
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            s = s[: match.start()] + "a" * (match.end() - match.start()) + s[match.end():]


def _reconstruct_graph(
    new_subgraph: nx.DiGraph | None,
    graph_nx: nx.DiGraph,
    summaries: dict[int, str],
    topk_nodes: list[int],
    reverse_mapping: dict[str, int],
) -> tuple[nx.DiGraph | None, dict[int, str] | None]:
    """
    Merge the LLM-produced new_subgraph back into the complete original graph.

    Returns:
        new_graph: full reconstructed DiGraph. Each node carries:
            - signature (str)
            - prev_node_idx (int): original node index if unchanged, -1 if new/refactored
        new_node_summaries: {fn_idx: summary} for new/refactored nodes only.
            Context and non-expanded nodes retain their original summaries from cache.
    """
    if new_subgraph is None:
        return None, None

    new_graph = nx.DiGraph()
    new_node_summaries: dict[int, str] = {}
    fn_map: dict = {}                        # new_subgraph str ID or original int idx -> fn_idx
    seen_context_nodes: dict[int, str] = {}  # original_idx -> new_subgraph str ID

    # add nodes produced by the LLM
    for node in new_subgraph.nodes:
        fn_sig: str = new_subgraph.nodes[node].get("signature", "")
        fn_summary: str = new_subgraph.nodes[node].get("summary", "")
        fn_idx = len(fn_map)
        prev_node_idx = -1

        if node in reverse_mapping:  # context node (O* prefix)
            original_idx = reverse_mapping[node]
            seen_context_nodes[original_idx] = node
            prev_node_idx = original_idx

            if not fn_sig:
                fn_sig = graph_nx.nodes[original_idx]["signature"]
            if not fn_summary:
                fn_summary = summaries[original_idx]
            if fn_sig != graph_nx.nodes[original_idx]["signature"]:
                print_(
                    f"[!] Context node signature changed: "
                    f"{graph_nx.nodes[original_idx]['signature']!r} -> {fn_sig!r}, reverting"
                )
                fn_sig = graph_nx.nodes[original_idx]["signature"]
        else:
            # new or refactored node (R* prefix or any other non-O prefix)
            if not fn_sig or not fn_summary:
                print_(f"[!] Refactored node {node!r} missing fields, skipping LLM output -- falling back to original graph")
                return None, None
            new_node_summaries[fn_idx] = fn_summary

        new_graph.add_node(fn_idx, signature=fn_sig, prev_node_idx=prev_node_idx)
        fn_map[node] = fn_idx

    for edge_from, edge_to in new_subgraph.edges:
        new_graph.add_edge(fn_map[edge_from], fn_map[edge_to])

    # restore dropped context nodes and nodes outside the expanded subgraph
    dropped = set(reverse_mapping.values()) - seen_context_nodes.keys() - set(topk_nodes)
    if dropped:
        print_(f"[+] Re-adding {len(dropped)} context node(s) dropped by LLM")

    for node in graph_nx.nodes:
        if node in summaries and node not in dropped:
            continue  # already added from the LLM output
        fn_idx = len(fn_map)
        new_graph.add_node(
            fn_idx,
            signature=graph_nx.nodes[node]["signature"],
            prev_node_idx=node,
        )
        fn_map[node] = fn_idx

    # copy original edges outside the refactored set
    topk_set = set(topk_nodes)
    for edge_from, edge_to in graph_nx.edges:
        # skip edges touching the refactored set -- those come from new_subgraph
        if edge_from in topk_set or edge_to in topk_set:
            continue
        # edges between two successfully-processed context nodes are already in new_subgraph
        if edge_from in seen_context_nodes and edge_to in seen_context_nodes:
            continue
        # context nodes that the LLM kept are in fn_map only under their O* key;
        # redirect via seen_context_nodes to get the right fn_map key
        mapped_from = seen_context_nodes[edge_from] if edge_from not in fn_map else edge_from
        mapped_to = seen_context_nodes[edge_to] if edge_to not in fn_map else edge_to
        if mapped_from not in fn_map or mapped_to not in fn_map:
            continue
        new_graph.add_edge(fn_map[mapped_from], fn_map[mapped_to])

    return new_graph, new_node_summaries


def augment_dataset(cfg, dataset: MalNetDataset, cache_dir=None):
    if cfg.augmentation.params.rewriter == "qwen2.5-coder":
        from .qwen25_coder import tokenizer, build_summarize_prompt, build_refactor_prompt, ollama_kwargs
    elif cfg.augmentation.params.rewriter == "phi4-mini":
        from .phi4_mini import tokenizer, build_summarize_prompt, build_refactor_prompt, ollama_kwargs
    elif cfg.augmentation.params.rewriter == "gemma4":
        from .gemma4 import tokenizer, build_summarize_prompt, build_refactor_prompt, ollama_kwargs
    else:
        raise NotImplementedError(
            f"Rewriter not supported for subgraph augmentation: {cfg.augmentation.params.rewriter}"
        )

    model_id = cfg.augmentation.params.rewriter
    train_split = dataset.split_idxs["train"]
    raw_dir = getattr(cfg.dataset.params, "raw_dir", None)
    context_length = cfg.augmentation.params.context_length
    num_workers = cfg.augmentation.params.num_workers
    ollama_urls = cfg.augmentation.params.ollama_urls

    root_dir = (
        Path(cfg.injected.base_dir)
        / "datasets" / "augmented" / "mntf"
        / cfg.dataset.name
        / cfg.dataset.params.variant
    )
    summary_dir = root_dir / "summaries"
    refactor_dir = root_dir / "refactor" / model_id

    node_selector = get_selector(cfg)
    _selection_cache = Path(cache_dir) if cache_dir is not None else root_dir / "selection"
    topk_nodes = node_selector(dataset, graph_idxs=train_split, cache_dir=_selection_cache)
    assert topk_nodes is not None, "Node selection failed."

    # fetch source codes for every node in each training graph
    all_node_idxs = {
        data_idx: list(range(dataset[data_idx].num_nodes)) # type: ignore[union-attr]
        for data_idx in train_split
    }
    all_source_codes = dataset.get_source_codes(all_node_idxs, raw_dir=raw_dir)

    # summarize the selected nodes and their neighbors
    def func_summarize(data_idx: int) -> None:
        graph_topk = topk_nodes[data_idx]
        source_codes = all_source_codes[data_idx]
        graph_nx = to_networkx(dataset[data_idx])

        graph_topk = [n for n in graph_topk if n in graph_nx]
        if not graph_topk:
            return
        expanded: set[int] = set(graph_topk)
        for n in graph_topk:
            expanded.update(graph_nx.predecessors(n))
            expanded.update(graph_nx.successors(n))

        node_summary_dir = summary_dir / str(data_idx)
        node_summary_dir.mkdir(parents=True, exist_ok=True)

        for node_idx in expanded:
            cache = node_summary_dir / f"{node_idx}.txt"
            if cache.exists():
                continue
            ollama_url = random.choice(ollama_urls)
            response = call_ollama(
                build_summarize_prompt(source_codes[node_idx]),
                ollama_url,
                model_id=model_id,
                tokenizer=tokenizer,
                context_length=context_length,
                min_output_tokens=256,
                seed=cfg.seed,
                **ollama_kwargs,
            )
            cache.write_text(response["raw_output"]["message"]["content"].strip())

    with tqdm(total=len(train_split), desc="Summarizing nodes") as pbar:
        with ThreadPoolExecutor(num_workers) as executor:
            futures = [executor.submit(func_summarize, idx) for idx in train_split]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
                pbar.refresh()

    # refactor each selected subgraph using the cached summaries
    def func_refactor(data_idx: int) -> None:
        graph_topk = topk_nodes[data_idx]
        source_codes = all_source_codes[data_idx]
        node_set_str = "_".join(str(n) for n in sorted(graph_topk))
        out_dir = refactor_dir / str(data_idx) / node_set_str

        if (out_dir / "graph.json").exists():
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summaries").mkdir(exist_ok=True)

        graph_nx = to_networkx(dataset[data_idx])
        nx.set_node_attributes(
            graph_nx,
            {n: get_function_signature(source_codes[n]) for n in graph_nx.nodes},
            "signature",
        )

        graph_topk = [n for n in graph_topk if n in graph_nx]
        if not graph_topk:
            return
        expanded: set[int] = set(graph_topk)
        for n in graph_topk:
            expanded.update(graph_nx.predecessors(n))
            expanded.update(graph_nx.successors(n))

        summaries = {
            n: (summary_dir / str(data_idx) / f"{n}.txt").read_text()
            for n in expanded
        }

        messages, reverse_mapping = build_refactor_prompt(
            graph_nx, summaries, list(graph_topk)
        )
        min_output_tokens = context_length // 2

        max_retries = 3
        response = None
        new_subgraph = None
        attempt = 0
        while True:
            seed = cfg.seed + attempt
            ollama_url = random.choice(ollama_urls)
            try:
                response = call_ollama(
                    messages,
                    ollama_url,
                    model_id=model_id,
                    tokenizer=tokenizer,
                    context_length=context_length,
                    min_output_tokens=min_output_tokens,
                    seed=seed,
                    **ollama_kwargs,
                )
                new_subgraph = parse_graph(response["raw_output"]["message"]["content"])
                break
            except (ValueError, TypeError):
                if attempt == max_retries - 1:
                    print_(f"[!] Failed to parse LLM output after {max_retries} attempts, skipping")
                    break
                attempt += 1

        new_graph, new_node_summaries = _reconstruct_graph(
            new_subgraph, graph_nx, summaries, list(graph_topk), reverse_mapping  # type: ignore[arg-type]
        )
        if new_graph is None:
            # irrecoverable LLM output, fall back to original graph (no refactoring applied)
            new_graph = nx.DiGraph()
            for node in graph_nx.nodes:
                new_graph.add_node(node, signature=graph_nx.nodes[node]["signature"], prev_node_idx=node)
            for edge_from, edge_to in graph_nx.edges:
                new_graph.add_edge(edge_from, edge_to)
            new_node_summaries = {}

        with open(out_dir / "response.json", "w") as f:
            json.dump(response, f)
        with open(out_dir / "graph.json", "w") as f:
            json.dump(nx.node_link_data(new_graph, edges="edges"), f)
        for fn_idx, summary in new_node_summaries.items():
            (out_dir / "summaries" / f"{fn_idx}.txt").write_text(summary)

    with tqdm(total=len(train_split), desc="Refactoring subgraphs") as pbar:
        with ThreadPoolExecutor(num_workers) as executor:
            futures = [executor.submit(func_refactor, idx) for idx in train_split]
            for future in as_completed(futures):
                future.result()
                pbar.update(1)
                pbar.refresh()

    # Only new/refactored nodes (prev_node_idx == -1) need fresh embeddings.
    # Nodes with prev_node_idx >= 0 map back to an original graph node whose
    # embedding can be looked up from the base dataset by the training loader.

    all_summaries: list[str] = []
    all_destinations: list[Path] = []

    for data_idx in train_split:
        graph_topk = topk_nodes[data_idx]
        node_set_str = "_".join(str(n) for n in sorted(graph_topk))
        out_dir = refactor_dir / str(data_idx) / node_set_str
        if not (out_dir / "graph.json").exists():
            continue
        embed_dir = out_dir / "embeddings"
        embed_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "graph.json") as f:
            graph_data = json.load(f)

        for node in graph_data["nodes"]:
            if node.get("prev_node_idx", -1) != -1:
                continue  # original node embedding provided by base dataset
            fn_idx = node["id"]
            dest = embed_dir / f"{fn_idx}.pth"
            if dest.exists():
                continue
            all_summaries.append((out_dir / "summaries" / f"{fn_idx}.txt").read_text())
            all_destinations.append(dest)

    if all_summaries:
        embeddings = embed_batch(
            all_summaries,
            bsz=cfg.augmentation.params.embedder_bsz,
            device=cfg.device,
        )
        for dest, embedding in tqdm(
            zip(all_destinations, embeddings),
            total=len(all_destinations),
            desc="Saving embeddings",
        ):
            torch.save(embedding, dest)
