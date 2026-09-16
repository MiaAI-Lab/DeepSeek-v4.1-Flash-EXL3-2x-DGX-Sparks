# GLM PR #202: DS4.1 runtime safety port

This adapts the host-side safeguards and diagnostics from
[GLM-5.3-Flash PR #202](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/pull/202),
tested at `f4a505c`. It is **not a new DS4.1 kernel optimization**.

## What transfers

- Required native initialization and validation must not live inside `assert`.
  Python `-O` previously removed the **calls** to `goal50_coop_abi` and
  `goal50_coop_info`, including the latter's CUDA preparation side effects.
  Explicit exceptions now retain capability, capture, hash, ABI, layout,
  occupancy, launch-precondition and pointer-table checks under optimization.
- Reject known ubatching/DBO environment flags and `EXTRA_ARGS` that violate the
  shared per-device scratch contract. Keep both DS4.1 serialized-stream guards.
  This is configuration screening, not a runtime cross-stream lock; custom
  integrations must still serialize launches.
- Separate wrapper-install, native-preparation, layer-eligibility and eager/
  capture selection diagnostics. Python does not count CUDA graph replays.
- Expand DS4.1 GPU fallback coverage to rows 9/12/18/24, retaining the original
  0.3% peak numerical screen and adding row-peak/relative-L2 checks at 5%.
  Refuse `-O` for the GPU gate itself, whose assertions must remain active.

## What does not transfer

GLM has H=4096, local I=1024, top-k=8, K4 MCG, up to 32 rows and a separate
`glm53_coop_*` ABI. DS4.1 stays H=5120, local I=1152, top-k=6, K2/K3 mul1,
1–8 rows, `goal50_coop_*` ABI v1, geometry 1. K4 MTP remains stock.

DS4.1 already uses wide/wide geometry 1; GLM's geometry improvement cannot be
claimed again here. Native sources and release binary digest are unchanged.
Only the adapter source pin changes. Rebuilt binaries still need explicit local
repinning and GPU validation; this PR does not publish the missing release binary.

## Validation on two Sparks, 2026-09-16

Machine-readable results and all measured rates:
[`pr202-runtime-validation.json`](../extensions/cooperative_moe/benchmarks/pr202-runtime-validation.json).

### GLM test

The clean build produced `dfccdea3…`, not the PR's `aa3fe5e9…` binary pin.
A separate locally repinned test profile passed the packaged **48-case gate on
each node**, geometry 1, before serving. That gate records two allowed clipping/
mutation peak exceptions (about 0.36% and 0.33%); these are not strict-parity
passes and remain in the receipt. Both ranks prepared 42 eligible layers.

Matched serving configuration: InstantTensor image, TP2, DFlash2 k=3,
probabilistic draft sampling, adaptive-k and dense FP8 off, E3 grouped on,
850k configured context, batch budget 7168, GPU utilization 0.88. These are not
the PR author's k=7/adaptive/dense-FP8 measurements.

| Workload, three-run median | Stock | Cooperative |
|---|---:|---:|
| Hash-map prose, 400 tokens, ×1 | 28.71 tok/s | 32.19 tok/s |
| Counting, 200 tokens, ×1 | 40.33 tok/s | 43.96 tok/s |

Used `tests/bench_decode.py`, temperature 0, thinking off. Initial accidentally
overlapping benchmark attempts were discarded and replaced by isolated runs.
The ×2 test intermittently serialized requests (TTFT up to ~8 seconds); its raw
results are retained but do not establish a concurrency gain.

### DS4.1 test

Native test artifact `cf6ff1e2…` is the previously validated local DS4.1 rebuild,
not GLM's binary. Adapter source `a5e953e3…` differs in the staged copy only by
this local binary pin. The expanded GPU gate passed **72/72 cases per node**:

| Metric | Head | Worker |
|---|---:|---:|
| Max error / reference peak | 0.26604% | 0.26605% |
| Max per-row peak ratio | 0.26915% | 0.26915% |
| Max relative L2 | 0.17953% | 0.17953% |

Strict raw/post-BF16 discrepancies remain recorded; this is not bit-exact with
stock. Both ranks then reported 40 eligible and 3 K4-ineligible layers, native
K2/K3 occupancy 2/2/3, and completed startup warmups. A smoke request returned
`323` with `finish_reason: stop`.

A same-day old-adapter/new-adapter restart comparison used PR #12's launcher,
packed rank-local Engram, DSpark k=3, 1536 batch budget, 96 IO threads, 600k
configured context and two-request capacity. sparkDash's exact 400-token prose
protocol ran three times per profile, sequentially, at concurrency 1 then 2.

| Aggregate decode, three-run median | Old adapter | Hardened adapter |
|---|---:|---:|
| ×1 | 40.54 tok/s | 36.05 tok/s |
| ×2 | 61.37 tok/s | 62.03 tok/s |

All requests completed. New-adapter ×1 ranged 31.17–40.45 tok/s, with variable
speculative acceptance visible in service logs. This is **not** a demonstrated
speedup or a performance non-regression proof. Matched-output/acceptance-controlled
retesting is needed to attribute the ×1 difference; no device arithmetic changed.

Host checks: 123 dispatch checks, eight profile tests, three build tests and seven
native-init/stream-guard tests under both normal and optimized Python. No 600k
prompt test, new sanitizer run, real-weight activation comparison, or prolonged
quality/burn-in evaluation was performed for this runtime port.
