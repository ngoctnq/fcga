from transformers import AutoTokenizer  # type: ignore
from src.fcga.rewriter.mntf.utils import get_function_signature


tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
ollama_kwargs = {"temperature": 0.0}


def build_prompt(source_code: str) -> list[dict]:
    signature = get_function_signature(source_code)

    return [
        {
            "role": "system",
            "content": "You are a Java code rewriter. Rewrite functions to be as structurally different as possible while STRICTLY preserving functionality. Respond with ONLY the rewritten Java function code, no explanations or markdown.",
        },
        {
            "role": "user",
            "content": f"""Rewrite the following Java function to be as different as possible while preserving exact functionality.

REQUIREMENTS:
1. The function signature MUST remain EXACTLY the same: {signature}
2. The function MUST produce identical outputs for all possible inputs
3. The function MUST have identical side effects (exceptions, state changes, etc.)
4. The rewritten code MUST be syntactically valid Java

FOCUS ON: Maximum transformation across all dimensions
- Apply control flow restructuring (loops, conditionals, branches)
- Rename variables and reorganize computations
- Transform the algorithmic approach while preserving semantics
- Use different Java language features and APIs
- Restructure the entire function logic flow

Original function:
```java
{source_code.strip()}
```

The function signature must be: {signature}. Keep the response within 8192 tokens.""",
        },
    ]
