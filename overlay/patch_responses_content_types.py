#!/usr/bin/env python3
"""Accept OpenAI Responses API text content in the DeepSeek V4.1 tokenizer.

The Responses API represents message text as ``type: input_text`` (and prior
assistant output as ``output_text``).  The model-specific tokenizer currently
accepts only the Chat Completions spelling, ``type: text``, even though vLLM's
generic Responses adapter deliberately preserves the Responses content type.
"""
from __future__ import annotations

from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41.py"
)
OLD = '                if part_type == "text":\n'
NEW = '                if part_type in ("text", "input_text", "output_text"):\n'


def patch_text(source: str) -> tuple[str, str]:
    if NEW in source:
        return source, "already applied"
    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(f"expected one DeepSeek V4.1 text-content anchor, found {count}")
    return source.replace(OLD, NEW, 1), "applied"


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    patched, status = patch_text(source)
    if patched != source:
        TARGET.write_text(patched, encoding="utf-8")
    print(f"DeepSeek V4.1 Responses content-types patch: {status}")


if __name__ == "__main__":
    main()
