"""Prepared integration gate; run only in a confirmed maintenance window.

Exercises the actual overlay + Torch tensors/CUDA graphs, not CPU dispatch stubs.
Covers K2/K3 decode, K4 fallback, and oversized DSpark capture shapes.
Numerical screening is peak-normalized plus row-peak and relative-L2 checks;
strict discrepancies remain recorded. Run without Python optimization.
"""

import functools
import json
import os
import sys

if os.environ.get("DSV41_COOP_MAINTENANCE_TEST") != "1":
    raise RuntimeError("requires an explicit maintenance test")
if sys.flags.optimize:
    raise RuntimeError("run the GPU gate without -O: numerical assertions are required")
os.environ.update(
    EXL3_FUSED_MOE="1",
    EXL3_FAT_GROUPED="1",
    EXL3_FAT_KERNEL="0",
    EXL3_TEMP_ROWS_FUSED="8",
    EXL3_FAT_EXPERT_LOG="0",
    DSV41_EXL3_SERIAL_STREAMS="1",
    VLLM_DISABLE_SHARED_EXPERTS_STREAM="1",
)
sys.path.insert(0, "/opt/dsv41")
import torch
import test_exl3_overlay as tests
from vllm.model_executor.layers.quantization import exl3

assert exl3._dsv41_coop_installed
wrapper = exl3.apply_exl3_fused_moe
stock = exl3._dsv41_coop_original_apply
checks = 0
strict_raw_failures = 0
strict_bf16_failures = 0
max_peak_rel = max_row_peak_rel = max_rel_l2 = 0.0


def compare(actual, reference):
    global strict_raw_failures, strict_bf16_failures
    global max_peak_rel, max_row_peak_rel, max_rel_l2
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    delta = (actual - reference).abs()
    peak = reference.abs().max().clamp_min(1e-30)
    peak_rel = float(delta.max() / peak)
    row_peak_rel = float(
        (delta.amax(dim=1) / reference.abs().amax(dim=1).clamp_min(1e-30)).max()
    )
    rel_l2 = float(
        (actual.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-30)
    )
    max_peak_rel = max(max_peak_rel, peak_rel)
    max_row_peak_rel = max(max_row_peak_rel, row_peak_rel)
    max_rel_l2 = max(max_rel_l2, rel_l2)
    assert peak_rel <= 0.003, "failed established peak-normalized numerical screen"
    assert row_peak_rel <= 0.05, "failed per-row numerical screen"
    assert rel_l2 <= 0.05, "failed relative-L2 numerical screen"
    strict_raw_failures += int((delta > 1e-3 + 1e-3 * reference.abs()).sum())
    a, r = actual.bfloat16().float(), reference.bfloat16().float()
    strict_bf16_failures += int(((a - r).abs() > 1e-3 + 1e-3 * r.abs()).sum())


for prefix in (
    "model.layers.0.mlp.experts",
    "model.layers.18.mlp.experts",
    "mtp.0.ffn.experts",
):
    torch.manual_seed(20260918)
    original_method = exl3.Exl3MoEMethod
    exl3.Exl3MoEMethod = functools.partial(original_method, prefix=prefix)
    try:
        _, owner = tests._tiny_layer(
            torch.device("cuda"), n_exp=32, hidden=5120, inter=1152
        )
    finally:
        exl3.Exl3MoEMethod = original_method
    native = getattr(owner, "_dsv41_coop_native", None)
    assert (native is not None) == (owner._exl3_k in (2, 3))
    for rows in (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 18, 24):
        for concentrated in (False, True):
            torch.manual_seed(13000 + 100 * owner._exl3_k + rows)
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16) * 0.1
            ids = (
                torch.randperm(32, device="cuda")[:6].repeat(rows, 1)
                if concentrated
                else torch.stack(
                    [torch.randperm(32, device="cuda")[:6] for _ in range(rows)]
                )
            )
            weights = torch.rand(rows, 6, device="cuda").softmax(-1)

            def run():
                return exl3.apply_exl3_experts(
                    x.float(), ids, weights, owner, fused=True
                )

            exl3.apply_exl3_fused_moe = stock
            baseline = run()
            calls = []
            if native is not None:
                original_native = native.launch

                def counted(*args):
                    calls.append(True)
                    return original_native(*args)

                native.launch = counted
            exl3.apply_exl3_fused_moe = wrapper
            actual = run()
            torch.cuda.synchronize()
            selected = native is not None and rows <= 8
            assert bool(calls) == selected
            compare(actual, baseline)
            if rows <= 8:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                for _ in range(5):
                    graph.replay()
                    torch.cuda.synchronize()
                    compare(captured, baseline)
                    if selected:
                        assert torch.equal(captured, actual)
                for scale in (0.03, 0.1, 0.6):
                    x.copy_(torch.randn_like(x) * scale)
                    ids.copy_(torch.randperm(32, device="cuda")[:6].repeat(rows, 1))
                    weights.copy_(torch.rand_like(weights).softmax(-1))
                    exl3.apply_exl3_fused_moe = stock
                    changed = run()
                    exl3.apply_exl3_fused_moe = wrapper
                    changed_candidate = run()
                    for _ in range(3):
                        graph.replay()
                        torch.cuda.synchronize()
                        compare(captured, changed)
                        if selected:
                            assert torch.equal(captured, changed_candidate)
                # Invalid local routes and zero weights must clear prior output.
                ids.fill_(-1)
                graph.replay()
                torch.cuda.synchronize()
                assert not torch.any(captured)
                ids.copy_(torch.randperm(32, device="cuda")[:6].repeat(rows, 1))
                weights.zero_()
                graph.replay()
                torch.cuda.synchronize()
                assert not torch.any(captured)
                del graph, captured
            if native is not None:
                native.launch = original_native
            checks += 1
            print(
                json.dumps(
                    {
                        "stage": "fixture",
                        "prefix": prefix,
                        "bits": owner._exl3_k,
                        "rows": rows,
                        "concentrated": concentrated,
                        "candidate_selected": selected,
                        "status": "pass",
                    }
                ),
                flush=True,
            )
print(
    json.dumps(
        {
            "stage": "complete",
            "checks": checks,
            "status": "pass",
            "strict_raw_failed_elements_retained": strict_raw_failures,
            "strict_post_bf16_failed_elements_retained": strict_bf16_failures,
            "max_peak_rel": max_peak_rel,
            "max_row_peak_rel": max_row_peak_rel,
            "max_rel_l2": max_rel_l2,
            "numerical_screen": "peak<=0.003; row peak<=0.05; rel L2<=0.05; strict differences retained",
            "distributed_serving_verified": False,
        }
    ),
    flush=True,
)
