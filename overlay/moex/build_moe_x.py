#!/usr/bin/env python3
"""Build exl3_moe_x_ext (kernel-shape variants of exl3_moe) against the installed exllamav3_ext source tree."""
import argparse, os, shutil, sys, time
from pathlib import Path
EXT_DEFAULT = "/usr/local/lib/python3.12/dist-packages/exllamav3/exllamav3_ext"
MODULE = "exl3_moe_x_ext"
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--ext", default=EXT_DEFAULT); ap.add_argument("--arch", default="121a"); ap.add_argument("--verbose", action="store_true")
ap.add_argument("--install", default="", help="copy the built module into this site-packages dir")
a = ap.parse_args()
import torch  # noqa
from torch.utils.cpp_extension import load
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); tree = out / "ext"
if tree.is_dir(): shutil.rmtree(tree)
shutil.copytree(a.ext, tree, ignore=shutil.ignore_patterns("*.so", "__pycache__"))
for n in ("exl3_moe_x.cu", "exl3_moe_x.cuh"): shutil.copyfile(Path(a.src) / n, tree / "quant" / n)
cu13 = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"
for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
    os.environ[var] = cu13 + (":" + os.environ[var] if os.environ.get(var) else "")
t0 = time.time()
mod = load(name=MODULE, sources=[str(tree / "quant" / "exl3_moe_x.cu")],
           extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{a.arch},code=sm_{a.arch}", "-Xcudafe", "--diag_suppress=177", "-Xcudafe", "--diag_suppress=20012", "--use_fast_math"],
           extra_include_paths=[str(tree)], build_directory=str(out), verbose=a.verbose)
print(f"built {MODULE} in {time.time()-t0:.0f}s: {mod.__file__}", flush=True)
for v in range(mod.moex_num_variants()):
    print(v, mod.moex_name(v), "regs/smem/maxblk/block/sms/local/maxthreads =", mod.moex_info(v), flush=True)
if a.install:
    so = Path(mod.__file__); dest = Path(a.install) / so.name; shutil.copyfile(so, dest); print(f"installed {dest}", flush=True)
