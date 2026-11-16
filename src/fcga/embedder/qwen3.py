from typing import Callable, Optional

import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel # type: ignore
from tqdm import tqdm
from src.utils import clear_gpu_cache


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_embedding(model, tokenizer, texts: list[str], max_length: int = 8192) -> Tensor:
    # Tokenize the input texts
    batch_dict = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    del batch_dict, outputs
    return embeddings


def _embed_chunk(model, tokenizer, contents: list[str]) -> Tensor:
    """Embed a chunk, halving on OOM until it fits (down to single items)."""
    try:
        with torch.no_grad():
            return get_embedding(model, tokenizer, contents).detach().cpu().float()
    except torch.OutOfMemoryError:
        clear_gpu_cache()
        if len(contents) == 1:
            raise
        mid = len(contents) // 2
        first = _embed_chunk(model, tokenizer, contents[:mid])
        second = _embed_chunk(model, tokenizer, contents[mid:])
        return torch.cat([first, second], dim=0)


@torch.inference_mode()
def embed_batch(
    texts: list[str],
    bsz=32,
    device: str = 'cuda',
    on_batch_done: Optional[Callable[[int, int, tuple[Tensor, ...]], None]] = None,
) -> tuple[torch.Tensor, ...]:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')
    # We recommend enabling flash_attention_2 for better acceleration and memory saving.
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', attn_implementation="flash_attention_2", dtype=torch.float16).to(device)
    model.eval()

    ret = tuple()

    for start_idx in tqdm(range(0, len(texts), bsz)):
        end_idx = min(start_idx + bsz, len(texts))
        # get the content of the files
        contents = texts[start_idx:end_idx]
        func_embeddings = _embed_chunk(model, tokenizer, contents)

        assert func_embeddings.shape == (end_idx - start_idx, 1024)
        batch_embeddings = func_embeddings.unbind(0)
        ret += batch_embeddings

        if on_batch_done is not None:
            on_batch_done(start_idx, end_idx, batch_embeddings)

        clear_gpu_cache()

    # free up memory
    del model
    del tokenizer
    clear_gpu_cache()

    return ret
