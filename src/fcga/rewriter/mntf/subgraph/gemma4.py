import json
from transformers import AutoTokenizer  # type: ignore
import networkx as nx


tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e4b-it")
ollama_kwargs = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def build_summarize_prompt(source_code: str) -> list[dict]:
    return [
        {"role": "system", "content":
            "You are a Java code analyst. "
            "From the user's source code, summarize it in 2-4 sentences. "
            "Describe what it does, its key side effects, and what it returns. "
            "Respond with only the summary, no code or markdown."
        },
        {"role": "user", "content": f"Summarize this function:\n\n```java\n{source_code.strip()}\n```"},
    ]


def build_refactor_prompt(
    graph_nx: nx.DiGraph,
    summaries: dict[int, str],
    topk_nodes: list[int],
) -> tuple[list[dict], dict[str, int]]:
    node_mapping: dict[int, str] = {
        node_idx: f"O{counter}"
        for counter, node_idx in enumerate(summaries.keys())
    }

    subg = graph_nx.subgraph(summaries.keys())
    graph_dict_temp = nx.node_link_data(subg, edges="edges")

    graph_dict: dict = {"nodes": [], "edges": []}
    for node in graph_dict_temp["nodes"]:
        nid = node["id"]
        graph_dict["nodes"].append({
            "id": node_mapping[nid],
            "signature": graph_nx.nodes[nid].get("signature", ""),
            "summary": summaries[nid],
        })
    for edge in graph_dict_temp["edges"]:
        graph_dict["edges"].append([
            node_mapping[edge["source"]],
            node_mapping[edge["target"]],
        ])

    mapped_topk = [node_mapping[n] for n in topk_nodes if n in node_mapping]
    user_content = (
        f"\nRefactor this set of functions: {mapped_topk}. "
        "Here is the FCG in JSON format:\n"
        + json.dumps(graph_dict)
    ).strip()

    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": user_content},
    ]
    reverse_mapping: dict[str, int] = {v: k for k, v in node_mapping.items()}
    return messages, reverse_mapping


def _build_system_prompt() -> str:
    return (
        "You are a code refactoring engine specializing in Android applications. "
        "Your task is to accept a partial Function Call Graph (subgraph) and refactor specific target nodes "
        "while STRICTLY preserving the integrity of their adjacent nodes.\n\n"
        "### INSTRUCTIONS ###\n"
        "1. **Refactor**: Rewrite the logic of the whole set of nodes you are asked to refactor. YOU HAVE CREATIVE FREEDOM here.\n"
        "   - **Example Strategies**:\n"
        "     * **Splitting**: Break a complex function into multiple smaller helper nodes.\n"
        "     * **Merging**: Combine a simple group of functions into a single node.\n"
        "     * **Obfuscation**: Add dummy control flow (e.g., new functions that do meaningful nothing), or rename operations.\n"
        "     * **Logic Hiding**: Move sensitive string construction into a new dedicated helper node.\n"
        "     * **Other**: Any other strategy that you think is appropriate.\n"
        "   - You can combine strategies, or even rewrite the entire set of functions.\n"
        "2. **Preserve Functionalities**: All nodes that are not in the refactoring set MUST keep their original functionality and signature "
        "even if they are changed to accommodate the refactoring. They MUST be present in the final output graph.\n"
        "3. **Reconstruct**: Generate the correct JSON of the new refactored program.\n\n"
        "### CONSTRAINTS & FORMATTING ###\n"
        "- **Output Format**: ONLY provide the final refactored graph in JSON. Do NOT include any other text, introduction, thinking, or explanation.\n"
        "- **JSON Schema**: {nodes: [{id: str, signature: str, summary: str}], edges: [[source, target]]}.\n"
        "  - **Node IDs**: Every node in the output graph MUST have an `id`.\n"
        "    - For refactored or new nodes: Give a new unique ID starting with the letter R.\n"
        "    - For context nodes: Copy the original node ID verbatim.\n"
        "  - **Signatures**: Every node in the output graph MUST have a `signature`.\n"
        "    - Function signatures MUST match exactly for non-refactored nodes.\n"
        "  - **Summary**: Every node in the output graph MUST have a `summary`.\n"
        "    - For updated nodes: Write a new summary of the new implementation.\n"
        "    - For unchanged nodes: Copy the original summary verbatim.\n"
        "- **No Code**: Do not output Java/Kotlin source code.\n"
        "- **No Curly Braces**: Do NOT use curly braces `{}` in your summaries to simplify parsing.\n"
    )
