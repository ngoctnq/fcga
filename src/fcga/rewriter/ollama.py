import json
import datetime
import requests
from typing import Optional
from copy import deepcopy
import re


def _call_ollama_helper(
    messages: list[dict],
    model_id: str,
    ollama_url: str,
    seed: Optional[int] = None,
    num_predict: Optional[int] = None,
    log_dir: Optional[str] = None,
    **ollama_kwargs
) -> dict:
    '''
    Helper function to call ollama with the given conversation and parameters.

    :param messages: The conversation messages to send to the model.
    :type messages: list[dict]
    :param model_id: The ID of the model to use for generation.
    :type model_id: str
    :param ollama_url: The base URL of the Ollama server.
    :type ollama_url: str
    :param seed: Optional random seed for generation, must be an integer if provided.
    :type seed: Optional[int]
    :param num_predict: Optional number of tokens to predict, must be a positive integer if provided.
    :type num_predict: Optional[int]
    :param log_dir: Optional directory to log the input and output of the model calls.
    :type log_dir: Optional[str]
    :param ollama_kwargs: Additional keyword arguments to pass to the Ollama API.
    :return: A dictionary containing the model input and raw output.
    :rtype: dict
    '''
    # craft payload to log to file later
    input_json = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": {
            **ollama_kwargs
        }
    }

    if seed is not None:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            raise ValueError(f"Seed must be an integer: {seed} ({type(seed)})")
        input_json["options"]["seed"] = seed

    if num_predict is not None:
        if not isinstance(num_predict, int) or num_predict <= 0:
            raise ValueError(f"num_predict must be a positive integer: {num_predict} ({type(num_predict)})")
        input_json["options"]["num_predict"] = num_predict

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        headers={"Content-Type": "application/json"},
        json=input_json,
    )
    response.raise_for_status()

    response_data = response.json()
    response.close()  # Explicitly close the connection

    result = {
        "model_input": input_json,
        "raw_output": response_data,
    }

    if log_dir is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{log_dir}/{timestamp}_{model_id}.json", "w") as f:
            json.dump(result, f)
            f.write("\n")

    return result


def truncate_messages(messages, max_tokens, tokenizer) -> tuple[list[dict], int]:
    """
    Truncate the list of messages to fit within the max_tokens limit.
    Delete/truncate the oldest messages first (system message are preserved).

    Args:
        messages (list[dict]): The list of messages to truncate.
        max_tokens (int): The maximum number of tokens allowed.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.

    Returns:
        list[dict]: The truncated list of messages
        int: The number of tokens used
    """

    # check if truncation is needed
    num_tokens = len(
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
    )
    if num_tokens <= max_tokens:
        return messages, num_tokens

    # if first message is not system message, we can simply truncate from the start
    if messages[0]["role"] != "system":
        truncate_index_start = 0
    # otherwise, do not truncate the system instruction message
    else:
        truncate_index_start = 1
    truncated_messages = deepcopy(messages)
    truncated_messages.insert(truncate_index_start, {"role": "user", "content": "\n[TRUNCATED]"})

    # find the oldest message that can be truncated to fit within the token limit,
    # deleting everything before it except the system message
    while True:
        # remove the oldest user message
        temp_message = truncated_messages.pop(truncate_index_start + 1)
        # make sure to set the role of the [TRUNCATED] message to the removed message role
        truncated_messages[truncate_index_start]["role"] = temp_message["role"]
        num_tokens = len(
            tokenizer.apply_chat_template(
                truncated_messages, add_generation_prompt=True, return_dict=False
            )
        )
        if num_tokens <= max_tokens:
            break

    # binary search the temp_message content length to match max_tokens
    target_token_count = max_tokens - num_tokens
    low, high = 0, len(temp_message["content"])
    while low < high:
        mid = (low + high) // 2
        test_content = temp_message["content"][:mid] + "\n[TRUNCATED]"
        token_ids = tokenizer.encode(test_content, add_special_tokens=False)
        current_token_count = len(token_ids)
        if current_token_count > target_token_count:
            high = mid
        else:
            low = mid + 1
    # set the truncated content
    temp_message["content"] = temp_message["content"][:high]
    # reduce to the first whitespace/tab/newline before high
    match = re.search(r"\s+", temp_message["content"][high - 1 :: -1])
    if match:
        # truncate all whitespaces if it comes as a block
        truncate_index = high - match.end()
        temp_message["content"] = temp_message["content"][:truncate_index]
        # note that we intentionally leave [TRUNCATED] even if the truncated last message is empty
        truncated_messages[truncate_index_start]["content"] = temp_message["content"] + "\n[TRUNCATED]"

    num_tokens = len(
        tokenizer.apply_chat_template(truncated_messages, add_generation_prompt=True, return_dict=False)
    )
    assert num_tokens <= max_tokens, "Truncation failed to reduce tokens within limit"
    return truncated_messages, num_tokens


def parse_backtick(output: str, verbose: bool = False) -> str:
    """
    Parse the model output to extract the code block content.
    Args:
        output (str): The raw output from the model.
        verbose (bool): Whether to print verbose output.
    Returns:
        str: The extracted code content.
    """
    if not verbose:
        print = lambda *args, **kwargs: None  # no-op

    # backup the original output to print if an error occurs
    original_output = output

    # most of the time, the code is in code blocks, and sometimes there are miscellaneous texts
    if "```" in output:
        # finds the first code block
        splits = output.split("```")
        # note that output length cannot be 1 or less because we checked for the presence of ```
        if len(splits) < 3:
            print(
                "Warning: hanging code block found, returning the second part."
            )
            if len(splits[1]) == 0:
                print(
                    "Warning: returned the first part because the second part is empty!"
                )
                return splits[0]

        # first line is an optional language specifier, e.g., ```java
        new_splits = splits[1].split("\n", 1)
        if len(new_splits) > 1:
            output = new_splits[1]
        else:
            print("Warning: code block has no content after language specifier, returning the first line.")
            output = new_splits[0]

        if not output:
            # last resort: return the original output
            print("Warning: Empty post-processing output. Returning the original output instead.")
            output = original_output
    else:
        print("Warning: No code block found in the output. Returning the full output.")

    return output.rstrip()  # remove trailing whitespace/newlines


def call_ollama(
    messages: list[dict],
    ollama_url: str,
    model_id: str,
    tokenizer,
    context_length: int,
    min_output_tokens: int = 0,
    seed: Optional[int] = None,
    log_dir: Optional[str] = None,
    **ollama_kwargs
) -> dict:
    '''
    Call Ollama with the given conversation and parameters, with automatic truncation to fit the model's context window.
    Context length is passed to the Ollama API here.

    :param messages: The conversation messages to send to the model.
    :type messages: list[dict]
    :param ollama_url: The base URL of the Ollama server.
    :type ollama_url: str
    :param model_id: The ID of the model to use for generation.
    :type model_id: str
    :param tokenizer: The tokenizer to use for tokenization.
    :type tokenizer: Tokenizer
    :param context_length: The maximum context length (in tokens) for the model. The function will automatically truncate the input messages to fit within this limit, reserving space for the output.
    :type context_length: int
    :param min_output_tokens: The minimum number of tokens expected in the model's output, used to reserve space in the context window. Defaults to 0.
    :type min_output_tokens: int
    :param seed: Optional random seed for generation, must be an integer if provided.
    :type seed: Optional[int]
    :param log_dir: Optional directory to log the input and output of the model calls.
    :type log_dir: Optional[str]
    :param ollama_kwargs: Additional keyword arguments to pass to the Ollama API.
    :type ollama_kwargs: dict
    :return: A dictionary containing the model input and raw output.
    :rtype: dict
    '''

    min_output_tokens = min(min_output_tokens, context_length)

    messages, token_count = truncate_messages(
        messages,
        max_tokens=context_length - min_output_tokens,
        tokenizer=tokenizer,
    )

    ollama_kwargs = ollama_kwargs.copy()
    # set num_ctx to fit input + expected output; use min_output_tokens directly
    # so thinking-heavy models (e.g. gemma4) don't get starved by the input-size cap
    target_num_ctx = token_count + min_output_tokens
    # get the next power of 2 for target_num_ctx for efficiency
    target_num_ctx = 2 ** (target_num_ctx - 1).bit_length()
    # and do not exceed the context length
    target_num_ctx = min(target_num_ctx, context_length)
    # and a minimum of 8192 -- arbitrary, based on a typical Llama 3 config
    target_num_ctx = max(target_num_ctx, 8192)
    ollama_kwargs["num_ctx"] = target_num_ctx


    response = _call_ollama_helper(
        messages,
        model_id,
        ollama_url,
        seed,
        context_length - token_count,
        log_dir,
        **ollama_kwargs
    )

    return response
