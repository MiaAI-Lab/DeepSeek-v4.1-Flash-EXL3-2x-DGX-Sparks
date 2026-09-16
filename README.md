<h1 align="center">DeepSeek v4.1 Flash EXL3 2.9 bpw for 2x DGX Sparks</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

OpenAI-compatible vLLM serve of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
as a local **EXL3 2.9 bpw / mul1** checkpoint (`model/`, 39 shards, 196 GiB)
on a **2× NVIDIA GB10** kit: tensor-parallel size 2 over CX7, native `sm_121a`
cubins, API on `:8888`. Served model id: **`DeepSeek-v4.1-Flash-EXL3`**.

Speculation is **DSpark**, and the draft experts live in the checkpoint
(`mtp.*`, `dspark_block_size=5`, 128 draft experts / top-3, target layers
37–39) — there is no separate drafter to download.

The image overlays EXL3 onto `vllm/vllm-openai:deepseekv41-flash-0909`
(linux/arm64, vLLM `0.1.dev20904+g179dd0fa9`), the only base carrying the
`DeepseekV41` architecture.

## What this checkpoint is

| | |
|---|---|
| Arch | CED 20+20 (20-layer causal encoder + 20-layer decoder), CSA2, Engram at layers 1 and 14, vision tower, native DSpark |
| Quant | EXL3 codebook **mul1** (not `mcg`), average **2.9 bpw**, `head_bits=6`, `mtp_bits=4`. Quantizer `version 1.4.2`; runtime is ExLlamaV3 **v1.4.5** (`e648f1a1`) |
| K | **Per tensor**, from `files/exl3_k_map.json`: routed experts **3** except layers 18–22 (**2**); shared experts **5** on layers 0–10 and 30–39, **4** on 11–29 (layer 29 is mixed, inferred from the trellis); attention **5**; `lm_head` **6**; indexer `wk` **8**; Engram wkv L1=**5** / L14=**4**; MTP **4**. Do not `int(2.9)` → 2 |
| Packed | `trellis` / `suh` / `svh` / `mul1` — marker int32 **`-2082680531`** (unsigned `2212286765` = `0x83DCD12D`). 47,900 packed matrices, 852 native tensors |
| Native | `embed.weight`, router gate weight+bias, all norms, `attn_sink`, the `hc_*` coefficients, the whole vision tower, Engram `k_weight`/`q_weight`, indexer `weights_proj` |
| KV | vLLM picks DeepSeek's **`fp8_ds_mla`** layout itself (`Using DeepSeek's fp8_ds_mla KV cache format` in the log). Do not pass `--kv-cache-dtype`. Measured pool cost here: 2.5 GiB per rank = **774,400 tokens** at `MAX_MODEL_LEN=614400`, i.e. ~3.4 KiB/token. (Upstream's "890 B/token" is the model's native FP4 main-KV design, not what this build allocates.) |
| Sampling | Official: `temperature=1.0`, `top_p=0.95`. Thinking defaults **on**; the template's `reasoning_effort` defaults to `"high"` (= 75; `"low"`=50, `"max"`=100, or an int 1–100). Smokes should send `chat_template_kwargs.enable_thinking=false` |

## Speculation

| | |
|---|---|
| Method | **DSpark** — the MTP/draft experts already in the checkpoint |
| Flag | `--speculative-config '{"method":"dspark","num_speculative_tokens":3}'` |
| k | **3** (`DSPARK_TOKENS`). `dspark_block_size=5` is the checkpoint's ceiling, not the setting: k=3 measured faster on prose |
| Capture sizes | `1 2 3 4 6 8 12 18 24` — 6 is included so a k=3 step (2 seqs × 3 tokens) is captured |
| Parsers | `--tokenizer-mode deepseek_v41` `--tool-call-parser deepseek_v41` `--reasoning-parser deepseek_v41` |
| Off | `SPEC_METHOD=none` frees ~3.5 GiB and is faster once the batch is wide (see Measured) |

## Memory

Host RAM *is* GPU memory on a Spark, and every `cudaMalloc` is committed the moment it is made.
Per node, from `scripts/weight_budget.py --tp 2`:

| Per node (GiB) | |
|---|---|
| EXL3 weights per rank | **99.5** |
| KV pool, pinned with `KV_CACHE_MEMORY_BYTES` | **2.5** |
| Context workspaces, CUDA context + NCCL, CUDA graphs | ~5–7 |
| vLLM processes, OS, docker, desktop | ~9 |

Shipped defaults: `MAX_MODEL_LEN=600000`, `MAX_NUM_SEQS=2`, `MAX_NUM_BATCHED_TOKENS=1024`, a
2.5 GiB KV pool (774,400 tokens at 614400 ctx). That leaves the head **4.07–4.21 GiB**
`MemAvailable` after warm-up; its real floor comes during a long *prefill*, not at boot —
**2.1 GiB** at the end of a 601k prompt.

`start.sh` enforces the rest: a boot-margin preflight, per-prefill allocator release, and a
post-load page-cache drop. Engram tables are never pinned — the row store replaces vLLM's
`cpu_offload`, which would `cudaHostAlloc` ~47 GiB per layer.

Raise one knob at a time and check `MemAvailable` **after a long prompt**, not after a boot. Every extra 0.5 GiB of KV pool costs ~1 GiB of head prefill margin: a 3 GiB pool boots and
passes the smoke test but dies at ~470k of a 600k prompt.

### Memory guard — off by default, and not needed in practice

`scripts/memguard.sh` is a watchdog that samples `/proc/meminfo` `MemAvailable` once a second and
`docker kill`s the local container after two consecutive samples below `DSV41_MEM_GUARD_GIB`
(1.5). It ships **disabled** (`DSV41_MEM_GUARD=0`); set `DSV41_MEM_GUARD=1` to arm it.

It was written for a real failure: on GB10 the ~113 GiB of weights lives in GPU driver
allocations that are not charged to any process's RSS, so the kernel OOM killer — which scores by
RSS — cannot see the actual consumer. On 2026-09-11 it picked small desktop daemons instead, the
memory was never released, and both nodes wedged until a hard reboot. The guard's job was to make
that kill deliberate and early.

**Sustained use since then says it is not needed.** Across normal serving the head sits at
**3.7–4.7 GiB** `MemAvailable`, never approaching 1.5 GiB — the floor is a long prefill, and even
a 601k prompt only reaches 2.1 GiB. The one time the guard ever fired, vLLM was **idle**: an
unrelated host process (a 5 GB-per-shard Hugging Face upload holding ~7 GiB of anonymous memory)
took the margin in 41 seconds, and the guard killed the 25-minute-boot server instead of the
process that caused it. It has no notion of which process is growing; it only ever kills its own
container. That is the trade it loses: it protects the host from a wedge, at the cost of the
service, even when the service is blameless.

What still protects the box with the guard off:

- `--oom-score-adj 1000` on both containers, so if the kernel OOM killer does fire it takes vLLM
  and not the desktop — which is precisely the 2026-09-11 failure;
- the 12 GiB boot-margin preflight, the per-prefill allocator release, and the post-load
  page-cache drop, none of which depend on the guard.

The residual risk is timing: the kernel fires later than the guard did, and on GB10 driver
allocations can begin failing before it engages. If you run heavy host-side jobs on a serving
node, cap them rather than re-arming the guard —
`systemd-run --scope -p MemoryMax=2G <job>` makes the kernel refuse *their* allocations, so
`MemAvailable` never moves.

## Quick start

On the **head** Spark (`10.0.0.1`), from this repo:

```bash
cp .env.example .env   # first run also does this
# edit WORKER_USER / GID if needed
./start.sh             # fetch weights, pull image, NFS-share, serve :8888
```

First boot downloads what is missing, then pulls
`ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw` (public, no
login) and ships it to the worker.

A clean checkout hashes to the same recipe stamp the published image carries,
so nothing is compiled: you pull ~9 GiB and serve. `start.sh` builds from this
Dockerfile only when that stamp moves — i.e. when you edit
`Dockerfile`/`overlay`/`files`/`tests` — or when the registry cannot be
reached, in which case it warns and builds instead of failing.

| Knob | Effect |
|---|---|
| `BUILD=1` | always compile, ignore the registry |
| `SKIP_BUILD=1` | never compile; keep the pulled image even if the stamp differs |
| `SKIP_PULL=1` | never reach the registry; use the local image or build |
| `PULL=1` | re-pull even when the stamp already matches |
| `SKIP_SHIP=1` | do not copy the image to the worker |

The worker pulls the image itself when it can; otherwise the head ships it, by
default as a staged tar over `rsync` that resumes where it left off if the link
drops. `IMAGE_SHIP=stream` pipes `docker save` straight into the worker's
`docker load` instead — no scratch disk, but an interrupted transfer restarts
from zero — and `IMAGE_SHIP=auto` prefers rsync and falls back to the pipe when
the worker is short of disk.

**Weights** (~387 GiB total, resumable — re-run to continue):

| Source | Into | Size |
|---|---|---:|
| [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) — 39 EXL3 shards | `MODEL_HOST` (`./model`) | ~197 GiB |
| [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) — shards **47+48 and the index only** | `ENGRAM_DIR` (`./engram-src`) | ~190 GiB |

Engram tables are never quantized and never copied into the EXL3 tree, which is
why they come from the original checkpoint; the other 46 shards are never read.
Point `ENGRAM_DIR` at an existing `DeepSeek-V4.1-Flash` tree to skip that half.
`./start.sh` fetches both automatically (`AUTO_DOWNLOAD=0` disables); run
`./download.sh` to stage them without booting the server. Needs the Hugging
Face CLI: `pip install -U 'huggingface_hub[hf_transfer]'`.

### Optional: Keys abliterated overlay (third party)

Not a fork of this kit. A ~650 MB sidecar replaces `layers.10–35.attn.wo_b`
(EXL3 mul1 K=5) on a **copy** of the 2.9 bpw pack. L0–9 / L36–39 / MTP /
Engram / experts stay stock. Keep this repo’s `start.sh`, image, and native
Engram shards 47+48.

1. Overlay (gated, automatic approval):
   [`drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed`](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed)
   — file **`mia_exl3_wo_b_l10_35.safetensors`**. Do **not** use
   `wo_b_l10_35.safetensors` (that is FP8 for other packs).
2. Apply helper:
   [`drowzeys/keys-DeepSeek-V4.1-Flash-Abliterated-Mia-2x-Spark-EXL3`](https://github.com/drowzeys/keys-DeepSeek-V4.1-Flash-Abliterated-Mia-2x-Spark-EXL3)
3. Set `MODEL_HOST` in `.env` to the applied dest. Leave `ENGRAM_DIR` as native
   shards 47+48.

### Optional: cooperative MoE (decode, two Sparks)

Off by default. Stock serving, the published image, and `./start.sh` without an
overlay override do not load this path. `DSV41_COOPERATIVE_MOE=1` alone does
nothing.

It is a **selected overlay** plus a native library copied to **both** ranks.
Prefill stays on the stock kernel; decode on eligible K2/K3 mul1 experts (hidden
5120, local 1152, top-k 6, 1–8 physical rows) can use Turboderp’s cooperative
MoE specialization. Requires `DSV41_EXL3_SERIAL_STREAMS=1` and
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` (already the shipped stream contract).

Do **not** compile inside the recipe image. That image has no `git`, and two
clean `nvcc` runs of the same sources are not bit-identical (`-lineinfo` and the
GNU build-id). `prepare_profile.py` will reject a rebuilt `.so` that is not the
pin below.

**Binary pin** (sha256 of `cooperative_moe.so`):
`a09a589cbdcecb5372991c7b091d732236d58bc5f5aea14ab91e38e426f08d78`

Put that file at
`extensions/cooperative_moe/artifacts/cooperative_moe.so` (root `.gitignore` skips
`*.so` except this path) or keep a matching GitHub Release asset.
Then:

```bash
printf '%s  cooperative_moe.so\n' \
  'a09a589cbdcecb5372991c7b091d732236d58bc5f5aea14ab91e38e426f08d78' |
  sha256sum -c
```

The full two-node procedure — drain, stage both ranks, 54-case GPU gate, `.env`
`EXL3_OVERLAY_HOST`, activation grep, rollback — is
[docs/cooperative-moe-quickstart.md](docs/cooperative-moe-quickstart.md).
Extension notes, rebuild-for-lab only:
[extensions/cooperative_moe/README.md](extensions/cooperative_moe/README.md).
Paired measurements (recipe `979e68a`, same checkpoint/image/k=3/600K/2 seqs):

| Workload | Stock | Cooperative |
|---|---:|---:|
| Poetry decode (3-seed median) | 23.62 tok/s | 29.26 tok/s |
| Coding decode (3-seed median) | 38.76 tok/s | 42.96 tok/s |
| C1 decode (temp 0) | 31.45 tok/s | 40.23 tok/s |
| C2 aggregate (temp 0) | 45.87 tok/s | 61.06 tok/s |
| Uncached 32K prefill | 1138 tok/s | ~unchanged |

Post-merge revalidation on the same pair saw C1 **+25.3%** and C2 **+35.2%**
after a locally rebuilt binary was **repinned**; that rebuild is not a substitute
for `a09a589c…`. Arithmetic is not bit-exact with stock. Vision and near-limit
context were not the validation target.

```bash
./start.sh status
./start.sh logs
./start.sh logs worker
./start.sh stop
./start.sh pack                   # optional: Engram rows onto each node's NVMe (+25–50 % prefill)
SKIP_SYNC=1 ./start.sh restart    # weights already on the worker
BUILD=1 ./start.sh                # force overlay rebuild
```

Smoke (thinking off so the answer is not buried in a long CoT):

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-v4.1-Flash-EXL3","messages":[{"role":"user","content":"What is 17*19? Reply with the integer only."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Official sampling for real work: `temperature=1.0`, `top_p=0.95`, and leave thinking on (or set `reasoning_effort` 1–100).

## Images

On by default. The checkpoint carries the full vision tower — 259 `vision.*` tensors plus
`aligner.*` and the `image_start` / `image_end` / `image_newline` embeddings, all kept native
rather than quantized — and the NVIDIA entry point is the multimodal one either way
(`vl_model.py`), with the tower stubbed out when the image limit is 0. Standard OpenAI
`image_url` parts, `data:` URIs included:

```bash
python3 - <<'PY'
import base64, json, urllib.request
b = base64.b64encode(open("files/ds.png", "rb").read()).decode()
req = {"model": "DeepSeek-v4.1-Flash-EXL3", "max_tokens": 300, "temperature": 0,
       "chat_template_kwargs": {"enable_thinking": False},
       "messages": [{"role": "user", "content": [
           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}},
           {"type": "text", "text": "Describe this image, and transcribe any text in it exactly."}]}]}
r = urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8888/v1/chat/completions", data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"}), timeout=600)
print(json.loads(r.read())["choices"][0]["message"]["content"])
PY
```

Two settings matter. `LANGUAGE_MODEL_ONLY=0` (the default; set it to `1` for a text-only
server) and **`MAX_NUM_BATCHED_TOKENS >= 1536`**. The chunk floor is not advisory: with the
tower on, vLLM forces `--disable_chunked_mm_input` for multimodal-bidirectional attention and
then refuses to start unless a whole image item fits in one prefill chunk —
`max_tokens_per_mm_item` (1025 = 1024 image tokens + 1) must not exceed
`max_num_batched_tokens`. At 1024 the boot dies with a `ValueError` out of
`compute_mm_encoder_budget`, several minutes into weight load.

Verified 2026-09-13 on 2× GB10, temp 0: a 1672×941 logo described correctly with its caption
transcribed exactly; a synthetic 10-row text image transcribed 10/10 exactly including
unguessable codes; three distinct cards in one prompt attributed to the right image with all
nine values exact; one image behind a 200k-token text prefix at 1,017 tok/s. DSpark keeps
drafting throughout (~60 % acceptance) despite `vl_model.py` dropping `mtp.*` from the main
model's weight map — the drafter loads those itself.

What is *not* established: any quality parity probe against the native checkpoint. The 128-wide
window clamp (see [SM12x runtime notes](#sm12x-runtime-notes)) means in-image bidirectional
visibility is genuinely off. It cost nothing measurable on OCR or multi-image attribution, but
these were pass/fail probes — a subtle deficit on dense document work or fine spatial reasoning
would not have shown up. Check before relying on it there.

Existing `.env` files are not rewritten on upgrade. If image requests return HTTP 400
`At most 0 image(s)`, set `LANGUAGE_MODEL_ONLY=0` and `MAX_NUM_BATCHED_TOKENS=1536` (or
higher) and restart.

## Layout

| Path | Role |
|---|---|
| `model/` | EXL3 2.9 bpw checkpoint (this workspace) |
| `ENGRAM_DIR` | Engram tables: shards 47+48 + index of the original `DeepSeek-V4.1-Flash` (`./engram-src`, auto-fetched) |
| `start.sh` | 2-node launcher (`start` / `share` / `pack` / `stop` / `restart` / `status` / `logs`) |
| `Dockerfile` | `vllm-openai:deepseekv41-flash-0909` + SM121 EXL3 ext + the overlay |
| `overlay/exl3.py` | Packed mul1 loader + apply: routed MoE, attn/shared/engram wkv linears, pinned H2D staging, pre-tune |
| `overlay/e3v2/` | v2 grouped fat-expert kernels, templated on (bits, codebook) |
| `overlay/patch_sm120_block64.py` | SM12x kernel envelope: 64-token KV blocks, indexer workspace, no `persistent_topk`, 128-wide image window |
| `overlay/patch_h2d_stage.py` | vLLM stock weight loaders copy through pinned memory (no copy-on-write of the shard mmaps) |
| `overlay/patch_memory_log.py` | `[dsv41-mem]` phase lines, page-cache drop, adaptive prefill release, EXL3 pre-tune hook |
| `overlay/patch_exl3_lm_head.py` | packed `wo_a` through `quant_method.apply` in the CUDA o-proj and the DSpark draft loader |
| `overlay/engram_file_backend.py`, `overlay/row_store.cpp` | file-backed `ParallelEngramEmbedding` |
| `extensions/cooperative_moe/` | Optional decode MoE; not loaded unless `EXL3_OVERLAY_HOST` selects its overlay |
| `files/exl3_k_map.json` | Per-tensor K |
| `files/chat_template.jinja` | Port of DeepSeek `encoding.py` |
| `scripts/nfs-share.sh` | NFSv4 export + worker docker volumes (reuses `vllm-fn-nfs`) |
| `scripts/zfs-share.sh` | Optional `zfs snapshot` + `send \| recv` fallback |
| `scripts/prepare_engram_src.py` | Hardlink shards 47+48 + embed-only index |
| `scripts/pack_engram.py` | Per-rank hash-head Engram shards for local NVMe |
| `scripts/memguard.sh` | per-node MemAvailable watchdog, **off by default** (`DSV41_MEM_GUARD=1` arms it) |
| `scripts/weight_budget.py` | per-rank resident weight bytes from the safetensors index (preflight headroom check) |

## Engram

The n-gram tables were not quantized. They live in native shards **47 and 48**
only (`layers.{1,14}.engram.embed.{weight,scale}`, ~95 GiB per shard), not the
other 46 shards and not the 476 GiB tree.

vLLM looks up rows via `--hf-overrides '{"engram_table_dir":"/engram-src"}'`
and a **file-backed** `ParallelEngramEmbedding` (not pinned UVA tables).
The rows are FP8 (`fp8_e4m3fn`) and must be handed to the dequant kernel as
such — reading them as `uint8` produces fluent garbage
(`tests/test_engram_dequant.py` guards it).

**NFS (default).** Reuses the live `vllm-fn-nfs` exporter (HF cache as NFSv4
root), hardlinks EXL3 into `dsv41-exl3` and a slim Engram dir (47+48 + embed-only
index) into `dsv41-engram`. spark2 docker volumes:

| Volume | NFS device | In-container |
|---|---|---|
| `dsv41-exl3-weights` | `10.0.22.1:/dsv41-exl3` | `/model` |
| `dsv41-exl3-engram` | `10.0.22.1:/dsv41-engram` | `/engram-src` |

```bash
./start.sh share    # once; start.sh also does this
./start.sh          # WEIGHT_SYNC=nfs is the default in .env
```

Without a live exporter the recipe starts its own `dsv41-exl3-nfs`. It binds
one real host directory (`NFS_EXPORT_ROOT`, default `CACHE_ROOT/nfs-export`)
at `/export` holding hardlinks of both trees, because the kernel nfsd refuses
to export the container's overlayfs `/export` root (the first activation on a
stock overlay2 docker root failed with exactly that). Trees on another
filesystem cannot be hardlinked; the recipe then falls back to the nested
per-tree binds.

**ZFS (optional, `WEIGHT_SYNC=zfs`).** Keeps a node-local replica on each Spark
instead of serving the worker over the network, and updates it with
`zfs send | recv` snapshot deltas rather than re-walking the tree.

What it buys, against the NFS default:

- the worker's reads — every Engram row miss, the whole shard load at boot —
  come off its own NVMe instead of the head's export over CX7, so the head
  stops being a single point of failure mid-serve;
- after the first replica, later syncs ship only changed blocks, so swapping in
  a re-quantised checkpoint is minutes rather than another 385 GiB;
- `recordsize=1M` + `lz4` suit 5–6 GiB shards, and snapshots let you roll back
  a bad checkpoint without re-downloading it.

What it costs: a pool on **both** nodes with ~385 GiB free each. That is why it
is off by default — spark2 has ~41 GiB spare, so it physically cannot hold a
replica, and NFS is the path that fits this kit.

One-time setup on each node, then flip the knob:

```bash
sudo apt-get install -y zfsutils-linux
sudo zpool create -f models <vdev>            # or import an existing pool
sudo zfs create -o recordsize=1M -o atime=off -o compression=lz4 \
     -o mountpoint=/path/to/model  models/dsv41-exl3
sudo zfs create -o recordsize=1M -o atime=off -o compression=lz4 \
     -o mountpoint=/path/to/engram models/dsv41-engram
sudo zfs allow -u "$WORKER_USER" create,destroy,mount,receive,snapshot,rollback models

```

Then set `WEIGHT_SYNC=zfs` **in `.env`** and run `./start.sh`.

> `start.sh` sources `.env` over its own environment, so a key that `.env`
> names cannot be overridden on the command line — `WEIGHT_SYNC=zfs ./start.sh`
> is silently ignored while `.env` says `nfs`. Edit `.env` for anything listed
> in it. Knobs absent from `.env` (`BUILD`, `PULL`, `SKIP_BUILD`, `SKIP_PULL`,
> `SKIP_SHIP`, `SKIP_SYNC`, `FORCE_SYNC`) do work as one-shot prefixes.

Populate `models/dsv41-engram` with shards 47+48 and the index only — never
send the 476 GiB native tree. Override `ZFS_POOL` and the four dataset names in
`.env`; `start.sh` falls back to the streaming path if the pool or the worker's
`zfs recv` is not reachable. `WEIGHT_SYNC=rsync` is the third option: a plain
node-local copy with no pool, and no incremental updates.

`./start.sh pack` writes vLLM hash-head `engram-l{1,14}-r<rank>of2.bin` shards
onto each node's local NVMe (47.2 GiB per layer per rank at TP=2, so ~94 GiB per
node; packing took 8.5 min). Decode and prefill then miss to local O_DIRECT
instead of NFS — worth 25–50 % of prefill. Not required to boot.

## CX7

Pins on this pair: spark1 `enp1s0f1np1`/`rocep1s0f1` ↔
spark2 `enp1s0f0np0`/`rocep1s0f0`. NCCL cannot use the `10.0.0.x` loopback
aliases. GID index is per-NIC — an all-zero entry dies ~60 s in with
`ibv_modify_qp` errno 61. Preflight checks each rank.

## Tests

Host-side (pure source/JSON checks — no torch, no vLLM):

```bash
python3 tests/test_numeric_config.py
python3 tests/test_engram_src.py
python3 tests/test_engram_secondary.py
python3 tests/test_k_map.py
python3 tests/test_memory_log.py
python3 tests/test_exl3_lm_head.py
python3 tests/test_sm120_block64.py
python3 tests/test_h2d_stage.py
python3 scripts/weight_budget.py --tp 2
python3 extensions/cooperative_moe/test_dispatch.py
python3 extensions/cooperative_moe/test_profile.py
python3 extensions/cooperative_moe/test_build.py
```

The chat-template parity harness needs `transformers` **and** the original
checkpoint's `encoding/encoding.py`; run it in the image if the host has no
`transformers`:

```bash
python3 tests/test_chat_template.py --src /home/mia/NewModels/DeepSeek-V4.1-Flash
```

After `./start.sh` is healthy:

```bash
bash tests/test_smoke.sh          # 17*19 -> 323
```

The image build runs thirteen in-image tests (EXL3 overlay with
`EXL3_SELFCHECK_GPU=0`, suppress-stops, scheduler decode floor, spinwait, Engram
secondary, EXL3 lm_head, Engram layout, row store, memory log, H2D staging,
SM120 block64, EXL3 pre-tune, Engram dequant), plus xgrammar termination in a
layer of its own.
`./start.sh` then runs the **GPU** self-check — `tests/test_exl3_overlay.py`
GEMM parity plus `tests/test_engram_dequant.py` — in a throwaway container
before launching (`logs/overlay-verify.log`).

## SM12x runtime notes

Four things this vLLM build needs on GB10:

- **E3 v2 grouped fat-expert kernels** (`overlay/e3v2/exl3_fat_moe.cu`, built into the image as
  `exl3_fat_moe_ext`): the original E3 kernels were K=4/MCG-only and needed one shared gate/up
  sign vector, so they were dead code here; v2 is templated on (bits, codebook) and gathers gate
  and up separately, which is what lets this K=3/K=2 mul1 tree use them. Every expert with more than `EXL3_TEMP_ROWS_FUSED` (16) rows in a
  prefill chunk goes through three grouped launches per layer instead of the per-16-row fused
  kernel (17.0 vs 34.1 ms per layer at a 1536-token chunk; `scripts/quality/moe_e3_probe.py`).
- **Kernel envelope** (`overlay/patch_sm120_block64.py`): 64-token KV blocks (DeepGEMM paged MQA
  logits and the FlashInfer SM120 sparse-MLA decode page), compressed-KV blocks scaled by the
  compress ratio, and a **128-wide image window**: FlashInfer has no 1152-wide (1024 image
  tokens + 128 window) sparse-MLA kernel on SM120, so the patch pins `max_image_tokens` to 0
  on SM12x. That clamp keys off device capability, not `LANGUAGE_MODEL_ONLY`, so it holds with
  images on — and it is what makes vision *runnable* here rather than what blocks it: every
  widened path (`prefill_index_width`, the in-image visibility buffers, `_build_image_visibility`,
  the widened Triton variants) is a guarded branch, so the missing kernel is never requested.
  See [Images](#images).
- **One lock buffer per device in exllamav3**: two EXL3 kernels on different CUDA streams deadlock,
  so the model's aux streams (`DSV41_EXL3_SERIAL_STREAMS=1`) and the shared-experts stream
  (`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`) are off. Every EXL3 GEMM shape is autotuned before
  CUDA-graph capture — 26 distinct shapes × 5 row buckets = 130 launches in ~3 s
  (`[dsv41-mem] EXL3 pretune` in the log); an escapee raises
  `operation not permitted when stream is capturing`.
- **Hang detector**: `start.sh` dumps py-spy stacks of both ranks (`logs/hang-*-pyspy.txt`) when the
  head log goes quiet for 420 s during boot.

## Performance

| Streams | Aggregate | Per stream | TTFT |
|---|---:|---:|---:|
| ×1 | **31.6 tok/s** | 31.6 tok/s | 221 ms |
| ×2 | **42.5 tok/s** | 21.6 tok/s | 347 ms |

**Prefill**, single request, by prompt length:

| Prompt | Tokens | tok/s | TTFT |
|---:|---:|---:|---:|
| 8k | 8,211 | **970.8** | 8.46 s |
| 16k | 16,403 | **986.9** | 16.62 s |
| 32k | 32,789 | **1,054.7** | 31.09 s |
| 64k | 65,555 | **1,022.7** | 64.10 s |
| 128k | 131,092 | **961.3** | 136.37 s |
| 256k | 262,158 | **872.6** | 300.44 s |

Benchmarks run via [sparkDash](https://github.com/MiaAI-Lab/sparkDash).

## Measured

TP=2, DSpark k=3, single request unless noted. Figures below predate the image default and were
taken with a text-only server; the shipped 1536-token chunk was rechecked with images on.

**Decode** (400-token prose, temp 0, thinking off): **31.6 tok/s** ×1, **42.5** aggregate ×2,
**42.8** aggregate ×4; structured count 1→200: 40 tok/s. With `SPEC_METHOD=none` a single stream
drops to 23 tok/s but ×4 reaches **53.7 aggregate** — batch serving wants speculation off.

**Prefill, short prompts**: **1,041 / 1,008 / 965 tok/s** at 10k / 28k / 57k with the packed
Engram shards (`./start.sh pack`, `DSV41_IO_THREADS=96`) and the E3 v2 grouped kernels
(`EXL3_FAT_GROUPED=1`, `EXL3_TEMP_ROWS_FUSED=16`), at a 1536-token chunk. Run-to-run spread is
about 10 %.

**Prefill, long prompts** (`expandable_segments:True`, **2048**-token chunks,
`LONG_PREFILL_TOKEN_THRESHOLD=1792`, single request):

| Prompt | TTFT | tok/s | Steady-state decode at that context |
|---:|---:|---:|---:|
| 50k | ~49 s | 1,029 | — |
| 100k | 102 s | 979 | 19.0–19.4 |
| 181k | 191 s | 948 | — |
| 455k | 567 s | 802 | 24.0 |
| 601k | 742 s | 810 | 22.3 |

Two fresh 100k prompts at once: 973 tok/s aggregate. A 17-token chat sent into a running 181k
prefill answers in **3.6 s**. Head `MemAvailable` 4.07–4.21 GiB after warm-up, low-water
2.56 GiB at 455k and 2.1 GiB at 601k; worker 5.8 GiB.

At the **shipped 1536-token chunk** with images on, head low-water is *higher* than the
2048-chunk run above — a larger chunk finishes the prefill in fewer scheduler passes, which
more than pays for the bigger per-chunk activation.

**Optional cooperative MoE** (same pair, not the default overlay): see the table under
Quick start. Stock numbers in this section are the shipped fused-MoE path.

## License

Launcher/overlay: AGPL-3.0, plus MIT for files that carry `LICENSE.MIT`.
Model weights: MIT (DeepSeek).
