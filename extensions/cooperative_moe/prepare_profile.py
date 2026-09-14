"""Create an exclusive opt-in overlay from the pinned source and native artifacts.

Does not download, install, change defaults, or restart anything. Rebuilt binaries
with a different hash require review and numerical validation before repinning.
"""

import argparse
import hashlib
from pathlib import Path, PurePosixPath

STOCK_SHA = "ccdc69bfa04bff4870c3e555736a990fde6448ddb329441c4e0a27d6fc41078d"
BINARY_SHA = "a09a589cbdcecb5372991c7b091d732236d58bc5f5aea14ab91e38e426f08d78"
ADAPTER_SHA = "9f1d10ffc39ac4433828a000c4932a4a773b00acadd80b46c7568f494a77b2fb"


def checked(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unvalidated source/binary hash: {path}")
    return data


def make_profile(stock, artifacts, runtime_directory, output):
    base = checked(Path(stock), STOCK_SHA)
    artifacts = Path(artifacts)
    checked(artifacts / "cooperative_moe.so", BINARY_SHA)
    checked(artifacts / "runtime.py", ADAPTER_SHA)
    runtime_root = PurePosixPath(runtime_directory)
    if not runtime_root.is_absolute() or ".." in runtime_root.parts:
        raise ValueError(
            "Use an absolute container runtime directory without parent traversal"
        )
    footer = (
        "\n# Explicit fixed cooperative MoE opt-in; unsupported calls stay stock.\n"
        "import runpy as _coop_runpy\nimport sys as _coop_sys\n"
        f'_coop_setup = _coop_runpy.run_path({str(runtime_root / "runtime.py")!r})\n'
        f'_coop_setup["install"](_coop_sys.modules[__name__], library_root={str(runtime_root)!r}, enabled=True)\n'
    )
    with Path(output).open("xb") as handle:
        handle.write(base + footer.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--runtime-directory",
        required=True,
        help="Container path containing the verified binary and adapter on BOTH ranks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_profile(args.stock, args.artifacts, args.runtime_directory, args.output)
    print(f"Wrote opt-in overlay: {args.output}; no service changes made")
