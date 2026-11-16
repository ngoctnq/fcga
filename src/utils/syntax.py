import json
from pathlib import Path

import tqdm
import tree_sitter_java as tsjava
import tree_sitter_kotlin as tskotlin
from tree_sitter import Language, Node, Parser

java_parser = Parser(language=Language(tsjava.language()))
kotlin_parser = Parser(language=Language(tskotlin.language()))

def find_errors(node: Node):
    # use a queue to avoid recursion overflow
    pending = [node]
    while pending:
        node = pending.pop()
        if node.type == "ERROR" or node.is_missing:
            if node.text is None:
                content = None
            else:
                content = node.text
                try:
                    content = content.decode("utf8")
                except UnicodeDecodeError:
                    pass
            return ("MISSING" if node.is_missing else node.type, content)

        pending.extend(reversed(node.children))

    return (None, None)


def is_valid_language(src, parser):
    root_node = parser.parse(src).root_node
    if root_node.has_error:
        return find_errors(root_node)
    return (None, None)


def preprocess_java(src):
    src = src.replace('\n', ';\n')
    return src


def preprocess_kotlin(src):
    src = src.replace(';', '')
    return src


def class_wrap(src):
    src = 'class FCGAAudit{\n' + src + '\n}\n'
    return src


def _check_with_wrapper(src, parser):
    error = is_valid_language(src.encode(), parser)
    if error[0] is None:
        return error
    wrapped_error = is_valid_language(class_wrap(src).encode(), parser)
    if wrapped_error[0] is None:
        return wrapped_error
    return error


def is_valid(src):
    java_error = _check_with_wrapper(src, java_parser)
    kotlin_error = _check_with_wrapper(src, kotlin_parser)
    if java_error[0] is None or kotlin_error[0] is None:
        return (None, None)

    for parser, preprocess in (
        (java_parser, preprocess_java),
        (kotlin_parser, preprocess_kotlin),
    ):
        if _check_with_wrapper(preprocess(src), parser)[0] is None:
            return (None, None)

    return (java_error, kotlin_error)


def parse_backtick(output: str) -> str:
    # backup the original output to print if an error occurs
    original_output = output

    # most of the time, the code is in code blocks, and sometimes there are miscellaneous texts
    if "```" in output:
        # finds the first code block
        splits = output.split("```")
        # note that output length cannot be 1 or less because we checked for the presence of ```
        if len(splits) < 3 and len(splits[1]) == 0:
                return splits[0]

        # first line is an optional language specifier, e.g., ```java
        new_splits = splits[1].split("\n", 1)
        if len(new_splits) > 1:
            output = new_splits[1]
        else:
            output = new_splits[0]

        if not output:
            # last resort: return the original output
            output = original_output

    return output


def is_limit_hit(response):
    raw = response["raw_output"]

    output_truncated = raw.get("done_reason") == "length"
    output_finished = raw.get("done") is True

    generated = raw.get("eval_count")
    limit = response["model_input"]["options"].get("num_predict")

    hit_limit = (
        generated is not None
        and limit is not None
        and generated >= limit
    )

    return output_truncated or not output_finished or hit_limit


def is_valid_source(response):
    '''Wrapped over the whole process. Use in import statements.'''
    limit_hit = is_limit_hit(response)
    source = response["raw_output"]["message"]["content"]
    source = parse_backtick(source)
    java, kotlin = is_valid(source)
    error = java is not None
    missing = error and (java[0] == "MISSING" or kotlin[0] == "MISSING")
    return {
        "source": source,
        "error": error,
        "limit_hit": limit_hit,
        "missing": missing
    }


if __name__ == "__main__":
    path = Path("datasets/augmented/mntf/MalNetTinyFeatures+qwen+keepisolated/prune+llm+ldp/rewrites/phi4-mini/statistics")
    fnames = sorted(path.glob("*/*.json"))
    bad_count = 0
    missing_count = 0
    limit_hit_count = 0

    for idx, fn in tqdm.tqdm(enumerate(fnames), total=len(fnames)):
        result_file = path.parent / "syntax" / fn.relative_to(path)
        if result_file.exists():
            result = json.loads(result_file.read_text())
        else:
            result = is_valid_source(json.loads(fn.read_text()))
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(result))
        if result["error"]:
            bad_count += 1
            if result["missing"]:
                missing_count += 1
            if result["limit_hit"]:
                limit_hit_count += 1

    total = len(fnames)
    print(
        f"bad_count: {bad_count} ({bad_count/total*100:.2f}%), "
        f"missing_count: {missing_count} ({missing_count/total*100:.2f}%), "
        f"limit_hit_count: {limit_hit_count} ({limit_hit_count/total*100:.2f}%), "
        f"total: {total}"
    )
