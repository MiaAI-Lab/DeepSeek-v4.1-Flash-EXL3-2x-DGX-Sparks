#!/usr/bin/env python3
"""Host checks for the DeepSeek V4.1 Responses content-type patch."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCH_PATH = ROOT / "overlay" / "patch_responses_content_types.py"
spec = importlib.util.spec_from_file_location("patch_responses_content_types", PATCH_PATH)
assert spec and spec.loader
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

FIXTURE = '''def normalize(content):
    result = []
    for message in ({"content": content},):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                part_type = block.get("type")
                if part_type == "text":
                    parts.append(block.get("text", ""))
                elif part_type in ("image_url", "input_image", "image_pil"):
                    parts.append("<image>")
                else:
                    raise ValueError(f"got {part_type!r}")
            result.extend(parts)
    return result
'''


def test_supported_types_and_idempotence() -> None:
    out, status = patch.patch_text(FIXTURE)
    assert status == "applied"
    namespace: dict[str, object] = {}
    exec(compile(out, "fixture.py", "exec"), namespace)
    normalize = namespace["normalize"]
    assert callable(normalize)
    assert normalize(  # type: ignore[operator]
        [
            {"type": "text", "text": "chat"},
            {"type": "input_text", "text": "input"},
            {"type": "output_text", "text": "output"},
            {"type": "input_image"},
        ]
    ) == ["chat", "input", "output", "<image>"]
    again, repeated = patch.patch_text(out)
    assert again == out
    assert repeated == "already applied"


def test_drift_fails_closed() -> None:
    for source in (FIXTURE.replace(patch.OLD, ""), FIXTURE + FIXTURE):
        try:
            patch.patch_text(source)
        except RuntimeError:
            pass
        else:
            raise AssertionError("drifted source was accepted")


def test_recipe_wiring() -> None:
    start = (ROOT / "start.sh").read_text()
    assert str(PATCH_PATH.relative_to(ROOT)) in start
    assert start.count("/opt/dsv41/patch_responses_content_types.py \\") == 2
    assert start.count("/opt/dsv41/patch_responses_content_types.py:ro") == 2
    assert start.count("--enable-prompt-tokens-details") == 2


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"Responses content-types patch OK ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
