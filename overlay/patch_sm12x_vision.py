#!/usr/bin/env python3
"""Opt-in vision window for MiaAI's already-applied SM12x block64 patch.

This does not install kernels or enable the vision tower. The candidate image
must supply the separately qualified FlashInfer build, and launch must set both
DSV41_SM12X_VISION=1 and LANGUAGE_MODEL_ONLY=0. Defaults remain text-only.
"""
from __future__ import annotations

import ast
from pathlib import Path

MARK = "dsv41-sm12x-vision-opt-in"
FILES = ("models/deepseek_v4_1/attention.py", "v1/attention/backends/mla/sparse_swa.py")
OLD = "    return bool(current_platform.is_device_capability_family(120))"
NEW = '''    # [dsv41-sm12x-vision-opt-in] The image must supply compatible kernels.
    import os

    return bool(current_platform.is_device_capability_family(120)) and (
        os.environ.get("DSV41_SM12X_VISION", "0") != "1"
    )'''


def apply(text: str) -> str:
    tree = ast.parse(text)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_dsv41_text_only_swa"]
    if len(matches) != 1:
        raise ValueError("expected exactly one pre-existing text-only SWA helper")
    node = matches[0]
    lines = text.splitlines(keepends=True)
    source = "".join(lines[node.lineno - 1:node.end_lineno])
    if MARK in source:
        if NEW not in source:
            raise ValueError("vision helper marker exists but body drifted")
        return text
    if source.count(OLD) != 1:
        raise ValueError("text-only SWA helper drifted")
    source = source.replace(OLD, NEW)
    result = "".join(lines[:node.lineno - 1]) + source + "".join(lines[node.end_lineno:])
    compile(result, "patched-backend", "exec")
    return result


def main() -> None:
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("vLLM installation not found")
    root = Path(spec.origin).parent
    pending = [(root / rel, apply((root / rel).read_text())) for rel in FILES]
    for path, result in pending:
        path.write_text(result)
        print("vision opt-in helper verified:", path)


def check_configuration(env, receipt: Path) -> None:
    if env.get("LANGUAGE_MODEL_ONLY", "0") != "0":
        raise ValueError("vision opt-in requires LANGUAGE_MODEL_ONLY=0")
    if int(env.get("MAX_NUM_BATCHED_TOKENS", "0")) < 1025:
        raise ValueError("vision requires MAX_NUM_BATCHED_TOKENS >= 1025; tested value: 2048")
    import json
    if not receipt.is_file() or json.loads(receipt.read_text()).get("flashinfer_revision") != "453aa7c7296e9ec711fd4c1f3aa6ee061a6b69dc":
        raise ValueError("use the pinned experimental Dockerfile.vision image")


if __name__ == "__main__":
    import os
    import sys
    if sys.argv[1:] == ["--check"]:
        check_configuration(os.environ, Path("/opt/dsv41/vision-build.json"))
        from flashinfer.mla._sparse_mla_sm120_plan import _DECODE_DSV4_DISPATCH, prefill_mg_eligible
        if (32, 1152) not in _DECODE_DSV4_DISPATCH or not prefill_mg_eligible(1, 32, 1152, 64, False):
            raise RuntimeError("FlashInfer does not support the required vision shape")
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("usage: patch_sm12x_vision.py [--check]")
