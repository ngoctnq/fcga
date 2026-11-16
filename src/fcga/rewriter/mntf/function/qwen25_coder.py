from transformers import AutoTokenizer  # type: ignore
from src.fcga.rewriter.mntf.utils import get_function_signature


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
ollama_kwargs = {}


def build_prompt(source_code: str) -> list[dict]:
    """
    Create a prompt for rewriting Java function with semantic preservation.
    Qwen does not perform well with a custom system prompt, so we include all instructions in the user prompt.

    :param source_code: the original Java function code to be rewritten
    :type source_code: str
    :return: a list of messages to be sent to the model
    :rtype: list[dict]
    """
    signature = get_function_signature(source_code)

    user_prompt = f"""You are a Java code rewriter. Your task is to rewrite the following Java function to be as different as possible while STRICTLY preserving its functionality.

CRITICAL REQUIREMENTS:
1. The function signature MUST remain EXACTLY the same: {signature}
2. The function MUST produce identical outputs for all possible inputs
3. The function MUST have identical side effects (exceptions, state changes, etc.)
4. The rewritten code MUST be syntactically valid Java code

FOCUS ON: Maximum transformation across all dimensions
- Apply control flow restructuring (loops, conditionals, branches)
- Rename variables and reorganize computations
- Transform the algorithmic approach while preserving semantics
- Use different Java language features and APIs
- Restructure the entire function logic flow
- Make the code as different as possible from the original

IMPORTANT: Do NOT add unnecessary code just for the sake of change. Every transformation must:
- Serve a legitimate structural purpose
- Maintain readability
- Preserve exact functionality

Original function:
```java
{source_code.strip()}
```

Respond with ONLY the rewritten Java function code, no explanations or markdown. The function signature must be: {signature}. Keep the response within 8192 tokens."""

    return [{"role": "user", "content": user_prompt}]
