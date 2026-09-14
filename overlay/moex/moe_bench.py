#!/usr/bin/env python3
"""Standalone exl3_moe microbenchmark: real layer-3 experts (TP rank-0 shard), the overlay's
exact launch (same args as _exl3_moe_launch), M tokens x top-6 routing to distinct experts.
Usage: moe_bench.py [--m 4] [--iters 200] [--experts 48] [--ncu]  (ncu mode: few launches, no timing loop)"""
import argparse, json, time, sys, importlib.util, torch
ap = argparse.ArgumentParser(); ap.add_argument("--m", type=int, default=4); ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--experts", type=int, default=48); ap.add_argument("--ncu", action="store_true"); ap.add_argument("--topk", type=int, default=6)
ap.add_argument("--file", default="/experts.safetensors"); ap.add_argument("--layer", default="3"); ap.add_argument("--prefix", default="", help="override tensor name prefix pattern with {e} and {w}, e.g. mtp.layers.0.ffn.experts.{e}.{w}."); ap.add_argument("--all-m", action="store_true")
ap.add_argument("--variants", default="", help="comma list of exl3_moe_x variants (e.g. 0,1,5); empty = stock kernel only")
ap.add_argument("--groups", type=int, default=0); ap.add_argument("--group-size", type=int, default=8); ap.add_argument("--xdir", default="/moex-build")
ap.add_argument("--gshapes", default="", help="comma list of GxS group shapes to sweep for the variants, e.g. 12x8,24x4")
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("ov", "/patched_exl3.py"); ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
import exllamav3_ext
from safetensors import safe_open
E = a.experts; TP = 2; RANK = 0
inners = []
with safe_open(a.file, "pt", device="cuda") as fh:
    for e in range(E):
        pack = {}
        for role, w, shard in (("gate", "w1", ov.shard_exl3_col), ("up", "w3", ov.shard_exl3_col), ("down", "w2", ov.shard_exl3_row)):
            p = a.prefix.format(e=e, w=w) if a.prefix else f"layers.{a.layer}.ffn.experts.{e}.{w}."
            t = {s: shard(fh.get_tensor(p + s), s, RANK, TP).contiguous() for s in ("trellis", "suh", "svh")}
            mul1 = fh.get_tensor(p + "mul1").reshape(()).to(torch.int32)
            pack[role] = ov.make_linear_exl3(t["trellis"], t["suh"], t["svh"], mul1, codebook="mul1")
        inners.append(pack)
g0 = inners[0]["gate"]; d0 = inners[0]["down"]
hidden, inter = g0.in_features, g0.out_features
print(json.dumps({"experts": E, "hidden": hidden, "intermediate_local": inter, "K": (g0.K, inners[0]["up"].K, d0.K),
                  "bytes_per_expert_MB": round(sum(inners[0][r].trellis.numel() * 2 for r in ("gate", "up", "down")) / 1e6, 2)}), flush=True)
dev = torch.device("cuda:0")
def ptrs(role, attr): return torch.tensor([int(getattr(p[role], attr).data_ptr()) for p in inners], dtype=torch.int64, device=dev)
P = {f"{r}_{s}": ptrs(r, s) for r in ("gate", "up", "down") for s in ("trellis", "suh", "svh")}
conc = max(1, int(exllamav3_ext.exl3_moe_max_concurrency(0)))
rows = ov.temp_rows_fused()
temps = tuple(torch.empty((conc, rows, n), dtype=torch.float16, device=dev) for n in (hidden, hidden, inter, inter))
fn = exllamav3_ext.exl3_moe
n_active = -1 if ov._exl3_moe_accepts_num_active(fn) else None
print(json.dumps({"concurrency": conc, "temp_rows": rows, "num_active_arg": n_active is not None}), flush=True)
variants = [int(v) for v in a.variants.split(",") if v != ""]
if variants:
    sys.path.insert(0, a.xdir); import exl3_moe_x_ext as mx
    locks = torch.zeros(mx.moex_locks_ints() + 4096, dtype=torch.int32, device=dev)
    xgroups = max([a.groups or conc] + [int(gs.split("x")[0]) for gs in a.gshapes.split(",") if gs])
    xtemps = tuple(torch.empty((max(xgroups, conc), rows, n), dtype=torch.float16, device=dev) for n in (hidden, hidden, inter, inter))
    for v in variants:
        vi = mx.moex_variant_index(v, int(g0.K))
        print(json.dumps({"shape": v, "K": int(g0.K), "instance": vi, "name": mx.moex_name(vi) if vi >= 0 else None, "regs_smem_maxblk_block_sms_local_maxthr": mx.moex_info(vi) if vi >= 0 else None}), flush=True)
def launch_x(v, x, out, count, ts, ws, groups, gsize):
    mx.exl3_moe_x(x, out, count, ts, ws, xtemps[0], xtemps[1], xtemps[2], xtemps[3], ov.MOE_ACT_SILU, int(g0.K), int(inners[0]["up"].K), int(d0.K),
                  P["gate_trellis"], P["gate_suh"], P["gate_svh"], P["up_trellis"], P["up_suh"], P["up_svh"], P["down_trellis"], P["down_suh"], P["down_svh"],
                  bool(g0.mcg), bool(g0.mul1), bool(inners[0]["up"].mcg), bool(inners[0]["up"].mul1), bool(d0.mcg), bool(d0.mul1), float(ov.SWIGLU_LIMIT_DEFAULT),
                  locks, v, groups, gsize)
def make_batch(M, topk):
    # M tokens, each routed to topk distinct experts, all (token,expert) pairs distinct experts when possible
    ids = torch.tensor([[(t * topk + j) % E for j in range(topk)] for t in range(M)], device=dev, dtype=torch.long)
    w = torch.full((M, topk), 1.0 / topk, device=dev, dtype=torch.float16)
    flat_token = torch.arange(M, device=dev, dtype=torch.long).repeat_interleave(topk)
    local = ids.reshape(-1); order = local.argsort()
    token_sorted = flat_token[order]; weight_sorted = w.reshape(-1)[order]
    count = torch.zeros(E + 1, dtype=torch.long, device=dev); count.scatter_add_(0, local, torch.ones_like(local))
    x = (torch.randn(M, hidden, device=dev) * 0.5).half()
    out = torch.zeros(M, hidden, dtype=torch.float32, device=dev)
    distinct = int(torch.unique(local).numel())
    return x, out, count, token_sorted, weight_sorted, distinct
def launch(x, out, count, ts, ws):
    args = (x, out, count, ts, ws, temps[0], temps[1], temps[2], temps[3], ov.MOE_ACT_SILU, int(g0.K), int(inners[0]["up"].K), int(d0.K),
            P["gate_trellis"], P["gate_suh"], P["gate_svh"], P["up_trellis"], P["up_suh"], P["up_svh"], P["down_trellis"], P["down_suh"], P["down_svh"],
            bool(g0.mcg), bool(g0.mul1), bool(inners[0]["up"].mcg), bool(inners[0]["up"].mul1), bool(d0.mcg), bool(d0.mul1), float(ov.SWIGLU_LIMIT_DEFAULT))
    fn(*args, n_active) if n_active is not None else fn(*args)
Ms = (1, 2, 4, 8, 16) if a.all_m else (a.m,)
for M in Ms:
    x, out, count, ts, ws, distinct = make_batch(M, a.topk)
    bytes_read = distinct * sum(inners[0][r].trellis.numel() * 2 for r in ("gate", "up", "down"))
    if a.ncu:
        if variants:
            for _ in range(a.iters): out.zero_(); launch_x(variants[0], x, out, count, ts, ws, a.groups or conc, a.group_size)
        else:
            for _ in range(a.iters): out.zero_(); launch(x, out, count, ts, ws)
        torch.cuda.synchronize(); print(json.dumps({"M": M, "distinct_experts": distinct, "launches": a.iters, "variant": variants[:1]})); continue
    for _ in range(20): launch(x, out, count, ts, ws)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(a.iters): launch(x, out, count, ts, ws)
    e.record(); torch.cuda.synchronize(); us = s.elapsed_time(e) * 1e3 / a.iters
    print(json.dumps({"M": M, "topk": a.topk, "distinct_experts": distinct, "us_per_launch": round(us, 1), "expert_MB_read": round(bytes_read / 1e6, 1),
                      "GB_s": round(bytes_read / us / 1e3, 1), "out_finite": bool(torch.isfinite(out).all()), "out_absmax": round(out.abs().max().item(), 3)}), flush=True)
    if variants:
        out.zero_(); launch(x, out, count, ts, ws); torch.cuda.synchronize(); ref = out.clone()
        shapes = [(int(g), int(s_)) for g, s_ in (gs.split("x") for gs in a.gshapes.split(",") if gs)] or [(a.groups or conc, a.group_size)]
        for v in variants:
          for groups, gsize in shapes:
            a.group_size = gsize
            try:
                outx = torch.zeros_like(out); launch_x(v, x, outx, count, ts, ws, groups, a.group_size); torch.cuda.synchronize()
            except Exception as ex:
                print(json.dumps({"M": M, "variant": v, "error": str(ex)[:160]}), flush=True); continue
            rel = ((outx - ref).norm() / ref.norm()).item()
            for _ in range(20): launch_x(v, x, outx, count, ts, ws, groups, a.group_size)
            torch.cuda.synchronize(); s_ = torch.cuda.Event(enable_timing=True); e_ = torch.cuda.Event(enable_timing=True); s_.record()
            for _ in range(a.iters): launch_x(v, x, outx, count, ts, ws, groups, a.group_size)
            e_.record(); torch.cuda.synchronize(); usx = s_.elapsed_time(e_) * 1e3 / a.iters
            print(json.dumps({"M": M, "variant": v, "name": mx.moex_name(mx.moex_variant_index(v, int(g0.K))), "groups": groups, "group_size": a.group_size, "us_per_launch": round(usx, 1),
                              "GB_s": round(bytes_read / usx / 1e3, 1), "speedup_vs_stock": round(us / usx, 3), "rel_diff_vs_stock": rel}), flush=True)
