#!/usr/bin/env python3
"""Install the pinned SM121 vision-kernel candidate inside an image build.

No GPU is needed. This intentionally keeps the base image's torch/vLLM/TVM
versions and builds only the sparse-MLA SM120/121 module.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

REVISION = "453aa7c7296e9ec711fd4c1f3aa6ee061a6b69dc"
SOURCES = (
    ("flashinfer-ai/flashinfer", REVISION, "eefc68a51dad5029eadbdcfa007b778b6fb6520b0197244645fc8f160a6f36fd", "flashinfer"),
    ("NVIDIA/cutlass", "b46b16d003484063bca4ed365e44095c4c6ed633", "eb32d2231412541cdaefedf82827903360b07762a0c8d554ca10a59951b321d0", "flashinfer/3rdparty/cutlass"),
    ("gabime/spdlog", "c3aed4b68373955e1cc94307683d44dca1515d2b", "206184e0d3be5b4ea56343dca519d38d1fdf567b59dd9913e3e959ed4c6b0f4d", "flashinfer/3rdparty/spdlog"),
    ("NVIDIA/cccl", "16bd510c9b712e82b0ab6cbb630d8e29ba1f7116", "8c4a1bebc5dfefaebbd40f21460951c62403f9c1f7c52b151b99cae82a02c844", "flashinfer/3rdparty/cccl"),
)


def unpack(archive: Path, destination: Path, expected: str) -> None:
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"source archive hash mismatch: {archive.name}")
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        with tarfile.open(archive) as bundle:
            bundle.extractall(directory, filter="data")
        roots = list(Path(directory).iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("expected one source root")
        if destination.exists():
            destination.rmdir()  # Fail on unexpected nonempty content.
        shutil.move(str(roots[0]), destination)


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    if Path("/dev/nvidia0").exists():
        raise RuntimeError("build in a no-GPU container, not in the serving container")
    pinned = {name: importlib.metadata.version(name) for name in ("torch", "vllm", "apache-tvm-ffi")}
    os.environ.update(FLASHINFER_CUDA_ARCH_LIST="12.1a", MAX_JOBS="1", BUILD_NVEP="0", BUILD_NIXL_EP="0", BUILD_NCCL_EP="0", FLASHINFER_BUILD_NO_PIP="1")
    with tempfile.TemporaryDirectory(prefix="dsv41-vision-") as directory:
        root = Path(directory)
        for index, (repo, revision, digest, relative) in enumerate(SOURCES):
            archive = root / f"source-{index}.tar.gz"
            with urllib.request.urlopen(f"https://codeload.github.com/{repo}/tar.gz/{revision}", timeout=60) as response, archive.open("wb") as output:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > 512 * 1024**2:
                        raise ValueError("source download exceeds 512 MiB")
                    output.write(block)
            unpack(archive, root / relative, digest)
        run("python3", "-m", "pip", "install", "--no-deps", "--no-build-isolation", str(root / "flashinfer"))
    # These packages can otherwise select a stale kernel with the same version.
    run("python3", "-m", "pip", "uninstall", "-y", "flashinfer-jit-cache", "flashinfer-cubin")
    from flashinfer.mla._sparse_mla_sm120_plan import _DECODE_DSV4_DISPATCH, prefill_mg_eligible
    if (32, 1152) not in _DECODE_DSV4_DISPATCH or not prefill_mg_eligible(1, 32, 1152, 64, False):
        raise RuntimeError("vision attention shape is not supported")
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module
    spec = gen_sparse_mla_sm120_module()
    spec.build(verbose=True)
    library = spec.jit_library_path
    if not library.is_file():
        raise RuntimeError("sparse-MLA build did not produce a library")
    if any(importlib.metadata.version(name) != version for name, version in pinned.items()):
        raise RuntimeError("base runtime version changed")
    receipt = {"flashinfer_revision": REVISION, "base_versions": pinned, "kernel": str(library), "gpu_tested_by_build": False}
    Path("/opt/dsv41/vision-build.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
