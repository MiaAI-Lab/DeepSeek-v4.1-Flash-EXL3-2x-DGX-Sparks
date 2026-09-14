# Cooperative MoE: benchmark and validation report

To reproduce the setup on an existing installation, follow the
[two-node opt-in and rollback guide](cooperative-moe-quickstart.md). It includes
the immutable build-image reference and the actual artifact deployment and
launcher commands; the adapter enablement variable alone is not sufficient.

## Configuration and provenance

The September 14, 2026 comparison used recipe `979e68a62c90b24d928f5638596e0ceed90e9f34`
on two SM121a Sparks. Both profiles used the same checkpoint, quantization and
image, TP2, 600,000 maximum context, two-request capacity, fixed DSpark k=3, native
FP4 KV, 3072 batched tokens, 2816 long-prefill threshold and an eight-row fused
threshold. The shared KV pool was approximately 749K tokens, not two separate
600K allocations. No clock, power, network or memory-limit changes were made for
the paired comparison.

The stock extension was ExLlamaV3 1.4.5 at
`e648f1a131365aae15920073e761a3fa5a527654`. Cooperative source is pinned to
`02aef45cd681b960a00afcd0749a4ab99e6c1bfe`, with the separate fixed-shape
specialization included in this PR. The tested recipe image ID was
`sha256:4cdba4e946da2d19bf5b5a20c6d3a1a4bf421fa4d6db5082f271a986168176cb`.

## Results

| Workload | Stock | Cooperative | Change |
|---|---:|---:|---:|
| Poetry decode, three-seed median | 23.62 tok/s | 29.26 tok/s | +23.9% |
| Coding decode, three-seed median | 38.76 tok/s | 42.96 tok/s | +10.8% |
| Incident reasoning, three-seed median | 31.92 tok/s | 41.26 tok/s | +29.3% |
| Reference C1, one paired pass | 31.45 tok/s | 40.23 tok/s | +27.9% |
| Reference C2 combined, one paired pass | 45.87 tok/s | 61.06 tok/s | +33.1% |
| Fully uncached 32K prefill, one paired pass | 1137.76 tok/s | 1135.35 tok/s | -0.2%, effectively unchanged |
| Two simultaneous ~32K coding requests, slower completion | 74.59 s | 69.18 s | 7.3% less elapsed time |

Two additional cooperative reference repetitions measured C1 39.22–39.81 tok/s,
C2 combined 58.92–60.00 tok/s, and uncached prefill 1138.90–1142.77 tok/s. Across
all three cooperative repetitions, medians were 39.81 / 60.00 / 1138.90 tok/s.
The paired stock reference had **one repetition**, not three. These bounded
measurements do not establish a universal or statistically guaranteed speedup.

Cooperative poetry ran at 28.70–29.59 tok/s with actual draft-token acceptance of
23.26–24.41%. Acceptance is workload-dependent. C2 values are aggregate
throughput, not per-stream speed. The [per-seed results](../extensions/cooperative_moe/benchmarks/results.json)
retain both configurations' token rates, acceptance counters and response hashes.
All nine paired full responses differed, so the comparison is not an
identical-output timing control or proof of unchanged answer quality.

## Measurement protocol

The sampled poetry, coding and incident prompts are included verbatim in
[`workloads.json`](../extensions/cooperative_moe/benchmarks/workloads.json).
Each workload ran separately with seeds 11, 23 and 47, a 512-output-token cap,
temperature 1 and top_p .95. Thinking was enabled only for incident analysis.
Chat-template flags `enable_thinking`, `thinking`, and `thinking_mode` were set
consistently. A 32-token warmup preceded each workload; all measured samples
completed 512 tokens with finish_reason `length`.

Reference C1/C2 used temperature 0, top_p 1 and thinking off, with 400 outputs per
request. The prompt was: “Write a detailed step-by-step explanation of how a hash
map works, including collision handling, resizing, and time complexity. Be
thorough.” C2 appended ` (stream 1/2)` and ` (stream 2/2)` and used a start barrier.

The 32K prefill probe used a random identifier, repeated ` the` filler and an
instruction to reply `OK`. Server counters confirmed fully uncached processing;
the returned answer was `OK`. Concurrent long-context validation used two
approximately 32,768-token synthetic Python cache-service review fixtures, each
with a unique identifier and 512 output tokens. The fixture contained diff-like
`+` prefixes and is a performance/stability workload, not a code-quality score.

Rates are derived from streamed API timings and request-boundary server counters:

- C1 decode: `(completion_tokens - 1) / (last_content_time - first_content_time)`.
- C2 decode: sum of `(completion_tokens - 1)` over the window from the earliest
  first content to the latest last content.
- Prefill: locally computed prompt tokens divided by time to first content
  (latest first-content time for concurrent requests).
- Acceptance: `accepted_draft_tokens / proposed_draft_tokens`. Fixed-k checks
  verified three proposed drafts per cycle and the output/counter relationship.

These are serving-level measurements, not direct GPU cycle latencies. A streamed
chunk may contain multiple speculative tokens. Only the benchmark requests were
active during each measured wave. Decode, prefill and acceptance must not be
treated as interchangeable measures.

## Numerical and safety validation

The fixed specialization matched upstream wide/wide exactly on 96 synthetic,
24 actual-input and six dense fixtures, including graph repeats and 48
empty/invalid/recovery checks. Actual captures were subset/tiled for other row
counts; they were not new live C2 captures.

The cooperative arithmetic is **not bit-exact with the installed stock kernel**.
Actual-input comparisons retained 35 strict `rtol=atol=1e-3` failures out of
645,120 output elements, with maximum error/stock peak 0.1123%. Six independent
dense checks passed the 0.3%-of-reference-peak criterion (maximum fixed peak error
0.0917%, normalized RMS 0.0794%). These are numerical screens, not quality scores.

The actual Torch/vLLM integration gate completed 54 cases: 32 cooperative-path
and 22 stock-fallback cases across K2/K3 main layers, K4 MTP, rows 1–8 and 12,
routing patterns, graph mutation, invalid routes and output casts. With
unnormalized synthetic scales, accumulated strict failed-element counts were
6,124,455 raw / 3,912,190 post-BF16 across repeated comparisons. These passed only
the documented peak-normalized criterion, not strict parity; counts are retained
rather than presented as independent-fixture failures or hidden as strict passes.

Four bounded sanitizer runs covered K2/three-row and K3/eight-row memcheck and
racecheck, the first 18 matching launches per run. They reported zero errors or
hazards; not every later graph mutation was instrumented.

Distributed startup completed all 14 warmup requests. Both long concurrent
requests completed their output caps. Initial reference-suite minimum sampled
MemAvailable was 5.18 GiB on the head and 5.65 GiB on the worker; no stream or
memory-monitor errors, preemptions, or OOM kills were observed. The existing
nonfatal p-only sampler-cache coverage warning remains.

## Current package validation and remaining work

The public naming/layout cleanup preserves native source byte-for-byte and keeps
the validated ABI v1 symbols and binary digest. Python dispatch state, artifact
filenames and enablement names were updated consistently, with CPU tests and
source-equivalence checks. No running service was switched to these renamed
artifacts during PR preparation.

Completed host-side checks include 113 dispatch assertions, seven profile
integrity/creation tests, shell syntax and Python formatting checks. The original
implementation has the GPU/full-serving evidence above. The final package still
needs a clean rebuild and GPU integration retest before merge, plus review of
numerical tolerances and broader task-quality coverage. Near-600K prefill and a
prolonged production burn-in were not repeated in this comparison. Upstream
build/distribution integration and support beyond the documented configuration
remain explicit review decisions; default serving behavior is unchanged.
