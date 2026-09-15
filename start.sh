#!/usr/bin/env bash
# ============================================================================
# start.sh — Spark runtime for DeepSeek-V4.1-Flash EXL3 (SM121 / GB10)
# ============================================================================
#
# Serve this repo's EXL3 2.9 bpw checkpoint (mul1, mixed K, native DSpark)
# on a 2× DGX Spark kit: vLLM TP=2 over CX7, OpenAI API on :8888.
# Base image: vllm/vllm-openai:deepseekv41-flash-0909 plus the EXL3 overlay.
#
# This is DSpark, not DFlash. Draft experts live in the checkpoint
# (mtp.*, dspark_block_size=5). There is no extra drafter repo to download.
#
#   head   : this machine (HEAD_IP, default 10.0.0.1) — vLLM rank 0 + API
#   worker : WORKER_USER@WORKER_IP (default: $USER@10.0.0.2) — rank 1, --headless
#   layout : --tensor-parallel-size 2, --nnodes 2, mp executor (not Ray)
#
# EXL3 mul1, not NVFP4, not GLM's K4/MCG. Do not pass --moe-backend marlin.
# E2/E3 fat kernels stay compiled but are ineligible (K4/MCG-only).
#
# What we do:
#   1. preflight  — docker/ssh/disk/GID on both nodes
#   2. image      — docker build IMAGE (local tag). Recipe-stamp rebuilds
#                   after overlay/Dockerfile edits. SKIP_BUILD=1 keeps the
#                   existing image. SKIP_SHIP=1 never copies to the worker.
#   3. weights    — NFSv4 share of EXL3 + slim Engram (shards 47+48 only),
#                   same as DeepSeek-v4.1-Flash-DGX-Sparks. No 385 GiB copy.
#                   ZFS send|recv / rsync remain optional fallbacks.
#   4. launch     — worker --headless, then head + `vllm serve`
#   5. wait       — poll /health, then a nonfatal DSpark/sampler warmup
#
# Usage:
#   ./start.sh                    start (share/launch) — default
#   ./start.sh share              NFSv4 export + worker docker volumes (no copy)
#   ./start.sh pack               pack Engram rows onto local NVMe (optional)
#   ./start.sh stop               stop both nodes
#   ./start.sh restart            stop + start
#   ./start.sh status             containers + API health
#   ./start.sh logs               follow head logs
#   ./start.sh logs worker        follow worker container logs
#
# Node IPs live in .env (copied from .env.example on first run).
# Handy overrides: SKIP_SYNC=1 SKIP_SHIP=1 SKIP_BUILD=1 BUILD=1 TAIL=1
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    [ -f "$SCRIPT_DIR/.env.example" ] || {
        echo "ERROR: missing .env.example" >&2
        exit 1
    }
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    printf '\033[1;36m[dsv41-exl3]\033[0m wrote .env from .env.example — edit HEAD_IP / WORKER_IP if needed\n'
fi
_cli_spec="${SPEC_METHOD-}"
_cli_eager="${ENFORCE_EAGER-}"
_cli_fused="${EXL3_FUSED_MOE-}"
_cli_row_tile="${EXL3_MOE_ROW_TILE-}"
_cli_temp_rows="${EXL3_TEMP_ROWS_FUSED-}"
_cli_fat_sorted="${EXL3_FAT_SORTED-}"
_cli_fat_batched="${EXL3_FAT_BATCHED-}"
_cli_fat_kernel="${EXL3_FAT_KERNEL-}"
_cli_fat_grouped="${EXL3_FAT_GROUPED-}"
_cli_mnbt="${MAX_NUM_BATCHED_TOKENS-}"
_cli_image="${IMAGE-}"
_cli_util="${GPU_MEM_UTIL-}"
_cli_lm="${LANGUAGE_MODEL_ONLY-}"
_cli_max_num_seqs="${MAX_NUM_SEQS-}"
_cli_max_model_len="${MAX_MODEL_LEN-}"
_cli_spinwait_ms_set="${GLM53_SPINWAIT_MS+1}"
_cli_spinwait_ms="${GLM53_SPINWAIT_MS-}"
_cli_dspark="${DSPARK_TOKENS-}"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a
[ -n "${_cli_spec}" ] && SPEC_METHOD="$_cli_spec"
[ -n "${_cli_eager}" ] && ENFORCE_EAGER="$_cli_eager"
[ -n "${_cli_fused}" ] && EXL3_FUSED_MOE="$_cli_fused"
[ -n "${_cli_row_tile}" ] && EXL3_MOE_ROW_TILE="$_cli_row_tile"
[ -n "${_cli_temp_rows}" ] && EXL3_TEMP_ROWS_FUSED="$_cli_temp_rows"
[ -n "${_cli_fat_sorted}" ] && EXL3_FAT_SORTED="$_cli_fat_sorted"
[ -n "${_cli_fat_batched}" ] && EXL3_FAT_BATCHED="$_cli_fat_batched"
[ -n "${_cli_fat_kernel}" ] && EXL3_FAT_KERNEL="$_cli_fat_kernel"
[ -n "${_cli_fat_grouped}" ] && EXL3_FAT_GROUPED="$_cli_fat_grouped"
[ -n "${_cli_mnbt}" ] && MAX_NUM_BATCHED_TOKENS="$_cli_mnbt"
[ -n "${_cli_image}" ] && IMAGE="$_cli_image"
[ -n "${_cli_util}" ] && GPU_MEM_UTIL="$_cli_util"
[ -n "${_cli_lm}" ] && LANGUAGE_MODEL_ONLY="$_cli_lm"
[ -n "${_cli_max_num_seqs}" ] && MAX_NUM_SEQS="$_cli_max_num_seqs"
[ -n "${_cli_max_model_len}" ] && MAX_MODEL_LEN="$_cli_max_model_len"
[ -n "${_cli_spinwait_ms_set}" ] && GLM53_SPINWAIT_MS="$_cli_spinwait_ms"
[ -n "${_cli_dspark}" ] && DSPARK_TOKENS="$_cli_dspark"

# ----------------------------- configuration -------------------------------
IMAGE="${IMAGE:-ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks:2.9bpw}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-v4.1-Flash-EXL3}"
GHCR_USER="${GHCR_USER:-MiaAI-Lab}"

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
WORKER_USER="${WORKER_USER:-$USER}"
if [ "$WORKER_USER" = "$USER" ]; then
    WORKER_HOME="${WORKER_HOME:-$HOME}"
else
    WORKER_HOME="${WORKER_HOME:-/home/${WORKER_USER}}"
fi
WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"

HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
HEAD_GID="${HEAD_GID:-$NCCL_IB_GID_INDEX}"
WORKER_GID="${WORKER_GID:-$NCCL_IB_GID_INDEX}"
CG_ESTIMATE="${CG_ESTIMATE:-1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-$WORKER_HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
USE_HOST_NCCL="${USE_HOST_NCCL:-0}"
# DS4.1 native finding: shrink NCCL pinned buffers on UMA (4.7 GiB -> ~0.14 GiB).
NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
NCCL_LL128_BUFFSIZE="${NCCL_LL128_BUFFSIZE:-262144}"
NCCL_PROTO="${NCCL_PROTO:-^LL128}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
# expandable_segments:True (the GLM recipe's setting): the per-chunk indexer/attention
# transients grow with the prefix, and without expandable segments the caching allocator
# cannot reuse the previous chunk's smaller blocks, so a long prefill fragments until
# cudaMalloc fails on MemFree and the allocator flushes — 244 tok/s at 455k (2026-09-13c).
# The native SGLang recipe saw NaN logits with it on ITS stack; on this vLLM stack the
# smoke/sanity/greedy-batch checks and the 100k-455k ladder were clean (13e).
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TP="${TP:-2}"
NNODES="${NNODES:-2}"
PORT="${PORT:-8888}"
MASTER_PORT="${MASTER_PORT:-29521}"

# dspark (default, in-checkpoint, k=5) | none
SPEC_METHOD="${SPEC_METHOD:-dspark}"
DSPARK_TOKENS="${DSPARK_TOKENS:-5}"
# Native CSA2 KV is ~890 B/token FP4. Do not force GLM's fp8_ds_mla.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
# 64-token KV blocks on GB10 (SM12x DeepGEMM paged indexer: 32/64 states per block).
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-600000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
# 1536: measured agent-workload sweet spot; the prefill chunk bounds the
# activation peak because the indexer scores every chunk row against the whole
# prefix. 2048 is the older long-context setting.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1536}"
CHAT_TEMPLATE_HOST="${CHAT_TEMPLATE_HOST:-$SCRIPT_DIR/files/chat_template.jinja}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/opt/dsv41/chat_template.jinja}"
STOP_PATCH_HOST="${STOP_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_suppress_stops_in_reasoning.py}"
SCHED_PATCH_HOST="${SCHED_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_scheduler_decode_floor.py}"
XGRAMMAR_PATCH_HOST="${XGRAMMAR_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_xgrammar_termination.py}"
SPINWAIT_PATCH_HOST="${SPINWAIT_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_spinwait.py}"
RESPONSES_PATCH_HOST="${RESPONSES_PATCH_HOST:-$SCRIPT_DIR/overlay/patch_responses_content_types.py}"
EXL3_OVERLAY_HOST="${EXL3_OVERLAY_HOST:-$SCRIPT_DIR/overlay/exl3.py}"
KMAP_HOST="${KMAP_HOST:-$SCRIPT_DIR/files/exl3_k_map.json}"
QUANTIZATION="${QUANTIZATION:-exl3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
if [ -z "${LIMIT_MM:-}" ]; then
    LIMIT_MM='{"image":100}'
fi
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
if [ "${ENFORCE_EAGER}" != "1" ]; then
    case " ${EXTRA_ARGS:-} " in
        *" --cudagraph-capture-sizes "*|*" cudagraph-capture-sizes "*) ;;
        *)
            # DSpark k=5 → 6 tokens/step including the target. Include 6 and
            # small multiples so graph capture covers 1..4 concurrent reqs.
            EXTRA_ARGS="${EXTRA_ARGS:+$EXTRA_ARGS }--cudagraph-capture-sizes 1 2 3 4 6 8 12 18 24"
            ;;
    esac
fi
EXL3_FUSED_MOE="${EXL3_FUSED_MOE:-1}"
EXL3_MOE_ROW_TILE="${EXL3_MOE_ROW_TILE:-0}"
# Fat kernels are K4/MCG-only — leave them compiled but off.
EXL3_FAT_GROUPED="${EXL3_FAT_GROUPED:-0}"
EXL3_FAT_KERNEL="${EXL3_FAT_KERNEL:-0}"
EXL3_FAT_SORTED="${EXL3_FAT_SORTED:-0}"
EXL3_FAT_BATCHED="${EXL3_FAT_BATCHED:-0}"
EXL3_TEMP_ROWS_FUSED="${EXL3_TEMP_ROWS_FUSED:-256}"

READY_TIMEOUT="${READY_TIMEOUT:-1500}"
# Hang detector while waiting for /health: no new head-log line for this many
# seconds -> py-spy dump of both ranks into logs/ and treat the boot as failed.
DSV41_HANG_SECONDS="${DSV41_HANG_SECONDS:-420}"
GLM53_SUPPRESS_STOPS_IN_REASONING="${GLM53_SUPPRESS_STOPS_IN_REASONING:-${DSV41_SUPPRESS_STOPS_IN_REASONING:-1}}"
GLM53_MIXED_PREFILL_CHUNK="${GLM53_MIXED_PREFILL_CHUNK:-${DSV41_MIXED_PREFILL_CHUNK:-0}}"
GLM53_INDEXER_WORKSPACE="${GLM53_INDEXER_WORKSPACE-stock}"
GLM53_SPINWAIT_MS="${GLM53_SPINWAIT_MS-${DSV41_SPINWAIT_MS-stock}}"
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
DSV41_BOOT_SHAPE_WARMUP="${DSV41_BOOT_SHAPE_WARMUP:-${GLM53_BOOT_SHAPE_WARMUP:-1}}"
GLM53_BOOT_SHAPE_WARMUP="$DSV41_BOOT_SHAPE_WARMUP"
DSV41_WARMUP_REQ_TIMEOUT="${DSV41_WARMUP_REQ_TIMEOUT:-${GLM53_WARMUP_REQ_TIMEOUT:-240}}"
GLM53_WARMUP_REQ_TIMEOUT="$DSV41_WARMUP_REQ_TIMEOUT"
VLLM_API_KEY="${VLLM_API_KEY:-}"

CONTAINER_HEAD="${CONTAINER_HEAD:-dsv41-exl3-head}"
CONTAINER_WORKER="${CONTAINER_WORKER:-dsv41-exl3-worker}"

MODEL_HOST="${MODEL_HOST:-$SCRIPT_DIR/model}"
ENGRAM_DIR="${ENGRAM_DIR:-$SCRIPT_DIR/engram-src}"
# Hub sources. The EXL3 weights are ours; the Engram tables are never
# quantized and never copied into the EXL3 tree, so they come from the
# original DeepSeek checkpoint (shards 47+48, the index, and config.json only).
HF_MODEL_REPO="${HF_MODEL_REPO:-Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
HF_ENGRAM_REPO="${HF_ENGRAM_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
# 1 = fetch whatever is missing before preflight checks it. 0 = only verify.
AUTO_DOWNLOAD="${AUTO_DOWNLOAD:-1}"
WORKER_MODEL_DIR="${WORKER_MODEL_DIR:-$WORKER_HOME/.cache/dsv41-flash-exl3/model}"
WORKER_ENGRAM_DIR="${WORKER_ENGRAM_DIR:-$WORKER_HOME/.cache/dsv41-flash-exl3/engram-src}"
# nfs (default) = share over CX7 like the native Spark recipe. auto prefers
# nfs, then ZFS datasets, then rsync. zfs/rsync still copy onto the worker.
WEIGHT_SYNC="${WEIGHT_SYNC:-nfs}"
NFS_SHARE="${NFS_SHARE:-1}"
NFS_VOLUME_MODEL="${NFS_VOLUME_MODEL:-dsv41-exl3-weights}"
NFS_VOLUME_ENGRAM="${NFS_VOLUME_ENGRAM:-dsv41-exl3-engram}"
NFS_EXPORT_MODEL="${NFS_EXPORT_MODEL:-dsv41-exl3}"
NFS_EXPORT_ENGRAM="${NFS_EXPORT_ENGRAM:-dsv41-engram}"
# Left empty on purpose: scripts/nfs-share.sh derives both from the route to
# WORKER_IP. Hard-coding this cluster's 10.0.22.1 here made nfs_server_ip's
# detection dead code and handed every other install the wrong export address.
NFS_SERVER_IP="${NFS_SERVER_IP:-}"
NFS_CLIENTS="${NFS_CLIENTS:-}"
ZFS_POOL="${ZFS_POOL:-models}"
ZFS_HEAD_MODEL_DS="${ZFS_HEAD_MODEL_DS:-${ZFS_POOL}/dsv41-exl3}"
ZFS_HEAD_ENGRAM_DS="${ZFS_HEAD_ENGRAM_DS:-${ZFS_POOL}/dsv41-engram}"
ZFS_WORKER_MODEL_DS="${ZFS_WORKER_MODEL_DS:-${ZFS_POOL}/dsv41-exl3}"
ZFS_WORKER_ENGRAM_DS="${ZFS_WORKER_ENGRAM_DS:-${ZFS_POOL}/dsv41-engram}"
ZFS_WORKER_IP="${ZFS_WORKER_IP:-10.0.22.2}"
ZFS_SSH="${ZFS_SSH:-${WORKER_USER}@${ZFS_WORKER_IP}}"
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/vllm-dsv41-flash-exl3}"
# Image ship to the worker: rsync a staged tar (resumable, needs scratch on both
# nodes), stream it through docker save | ssh docker load (no scratch, restarts
# from zero if the link drops), or auto = rsync with a stream fallback.
IMAGE_SHIP="${IMAGE_SHIP:-rsync}"
IMAGE_SHIP_DIR="${IMAGE_SHIP_DIR:-$CACHE_ROOT/image-ship}"
WORKER_IMAGE_SHIP_DIR="${WORKER_IMAGE_SHIP_DIR:-/tmp}"
IMAGE_SHIP_KEEP="${IMAGE_SHIP_KEEP:-0}"
ENGRAM_SRC="${ENGRAM_SRC:-$CACHE_ROOT/engram-src}"
WORKER_MODEL_BIND="${WORKER_MODEL_BIND:-$WORKER_MODEL_DIR}"
WORKER_ENGRAM_BIND="${WORKER_ENGRAM_BIND:-$WORKER_ENGRAM_DIR}"
HEAD_ENGRAM_BIND="${HEAD_ENGRAM_BIND:-}"
WORKER_VLLM_CACHE="${WORKER_VLLM_CACHE:-$WORKER_HOME/.cache/vllm-dsv41-flash-exl3}"
TRITON_HOST_CACHE="${TRITON_HOST_CACHE:-$CACHE_ROOT/triton}"
TILELANG_HOST_CACHE="${TILELANG_HOST_CACHE:-$CACHE_ROOT/tilelang}"
WORKER_TRITON_CACHE="${WORKER_TRITON_CACHE:-$WORKER_VLLM_CACHE/triton}"
WORKER_TILELANG_CACHE="${WORKER_TILELANG_CACHE:-$WORKER_VLLM_CACHE/tilelang}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/.triton/cache}"
TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/root/.tilelang/cache}"
MODEL_DIR="${MODEL_DIR:-/model}"
ENGRAM_MOUNT="${ENGRAM_MOUNT:-/engram-src}"
# File-backed Engram (not 47 GiB pin). Optional packed shards from ./start.sh pack.
# Host RAM is GPU memory on GB10: the row cache and the pinned scale shard
# (row_store defaults: 8 GiB cache + ~1.5 GiB/layer scales = ~11 GiB of
# anonymous RSS in the worker process) come straight out of the ~22 GiB left
# after ~99.5 GiB of weights. Both off by default, like the native DS4.1
# recipe; raise only after a clean boot shows headroom.
DSV41_CACHE_GIB="${DSV41_CACHE_GIB:-0}"
DSV41_RESIDENT_SCALES="${DSV41_RESIDENT_SCALES:-0}"
DSV41_IO_THREADS="${DSV41_IO_THREADS:-96}"
DSV41_CACHE_WAYS="${DSV41_CACHE_WAYS:-4}"
DSV41_STATS_SECONDS="${DSV41_STATS_SECONDS:-60}"
# Host memory guard: kill the local container when MemAvailable drops under
# this (GiB) so the kernel OOM killer never gets to wedge the node.
# 0 = do not arm the memguard watchdog (default). vLLM itself runs at ~4 GiB
# MemAvailable in normal use and never approaches the threshold; the only
# observed trips came from unrelated host processes, where the guard killed
# the server rather than the process that took the memory. Kernel fallback
# stays in place: the containers run with --oom-score-adj 1000, so if the box
# does run out the OOM killer takes vLLM and not the desktop.
DSV41_MEM_GUARD="${DSV41_MEM_GUARD:-0}"
DSV41_MEM_GUARD_GIB="${DSV41_MEM_GUARD_GIB:-1.5}"
# Pre-launch: each node must have (weights per rank + this margin) available.
DSV41_BOOT_MARGIN_GIB="${DSV41_BOOT_MARGIN_GIB:-12}"
MEMGUARD_HOST="${MEMGUARD_HOST:-$SCRIPT_DIR/scripts/memguard.sh}"
WEIGHT_BUDGET_HOST="${WEIGHT_BUDGET_HOST:-$SCRIPT_DIR/scripts/weight_budget.py}"
HEAD_PACKED_DIR="${HEAD_PACKED_DIR:-$HOME/dsv41-engram}"
WORKER_PACKED_DIR="${WORKER_PACKED_DIR:-$WORKER_HOME/dsv41-engram}"
PACKED_MOUNT="${PACKED_MOUNT:-/engram-packed}"

LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.dsv41-exl3-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.dsv41-exl3-worker.inner.sh"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-39}"
ENGRAM_SHARDS="${ENGRAM_SHARDS:-model-00047-of-00048.safetensors model-00048-of-00048.safetensors}"

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[dsv41-exl3]\033[0m %s\n' "$*"; }
mem_avail_gib() { awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo; }
worker_mem_avail_gib() {
    worker_ssh "awk '/^MemAvailable:/ { printf \"%.1f\", \$2 / 1048576 }' /proc/meminfo" 2>/dev/null || echo "?"
}
warn() { printf '\033[1;33m[dsv41-exl3]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[dsv41-exl3]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# GLM53 numeric config guard (begin)
_glm53_canonical_positive_int() {
    local name="$1" value="$2" maximum="$3" canonical
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$name must be a positive base-10 integer (got: $value)" >&2
        return 2
    fi
    canonical="$value"
    while [ "${canonical#0}" != "$canonical" ]; do canonical="${canonical#0}"; done
    [ -n "$canonical" ] || canonical=0
    if [ "$canonical" = 0 ] \
       || [ "${#canonical}" -gt "${#maximum}" ] \
       || [ "$canonical" -gt "$maximum" ]; then
        echo "$name must be between 1 and $maximum (got: $value)" >&2
        return 2
    fi
    printf -v "$name" '%s' "$canonical"
    # shellcheck disable=SC2163
    export "$name"
}

_glm53_validate_enum() {
    local name="$1" value="$2" allowed
    shift 2
    for allowed in "$@"; do
        [ "$value" = "$allowed" ] && return 0
    done
    echo "$name must be one of: $* (got: $value)" >&2
    return 2
}

_glm53_validate_spinwait_ms() {
    if [ "$GLM53_SPINWAIT_MS" = "stock" ]; then
        export GLM53_SPINWAIT_MS
        return 0
    fi
    _glm53_canonical_positive_int \
        GLM53_SPINWAIT_MS "$GLM53_SPINWAIT_MS" 1000
}

validate_numeric_config() {
    if ! [[ "$GPU_MEM_UTIL" =~ ^(0([.][0-9]+)?|[.][0-9]+|1([.]0+)?)$ ]] \
       || ! awk -v u="$GPU_MEM_UTIL" 'BEGIN { exit !(u > 0 && u <= 1) }'; then
        echo "GPU_MEM_UTIL must be greater than 0 and at most 1 (got: $GPU_MEM_UTIL)" >&2
        return 2
    fi
    _glm53_canonical_positive_int MAX_MODEL_LEN "$MAX_MODEL_LEN" 1048576 || return
    _glm53_canonical_positive_int MAX_NUM_SEQS "$MAX_NUM_SEQS" 4096 || return
    _glm53_canonical_positive_int MAX_NUM_BATCHED_TOKENS "$MAX_NUM_BATCHED_TOKENS" 8388608 || return
    _glm53_validate_enum GLM53_INDEXER_WORKSPACE "${GLM53_INDEXER_WORKSPACE-stock}" \
        stock rightsize || return
    _glm53_validate_spinwait_ms || return
}
# GLM53 numeric config guard (end)

banner() {
    local label="${1:-start.sh}"
    printf '\n'
    printf '  \033[1;36m┌────────────────────────────────────────────┐\033[0m\n'
    printf '  \033[1;36m│\033[0m  \033[1mDeepSeek-V4.1 Flash EXL3\033[0m  \033[2m%-10s\033[0m \033[1;36m│\033[0m\n' "$label"
    printf '  \033[1;36m└────────────────────────────────────────────┘\033[0m\n'
    printf '\n'
}

worker_ssh() { ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }

# shellcheck source=scripts/zfs-share.sh
source "$SCRIPT_DIR/scripts/zfs-share.sh"
# shellcheck source=scripts/nfs-share.sh
source "$SCRIPT_DIR/scripts/nfs-share.sh"

resolve_weight_backend() {
    case "$WEIGHT_SYNC" in
        nfs|share) printf 'nfs' ;;
        zfs) printf 'zfs' ;;
        rsync) printf 'rsync' ;;
        auto)
            if [ "${NFS_SHARE:-1}" != "0" ]; then
                printf 'nfs'
            elif zfs_should_use; then
                printf 'zfs'
            else
                printf 'rsync'
            fi
            ;;
        *) die "unknown WEIGHT_SYNC=$WEIGHT_SYNC (nfs|zfs|rsync|auto)" ;;
    esac
}

prepare_engram_src_dir() {
    python3 "$SCRIPT_DIR/scripts/prepare_engram_src.py" \
        --src "$ENGRAM_DIR" --dst "$ENGRAM_SRC" \
        || die "failed to build slim Engram src at $ENGRAM_SRC (need shards 47+48 under $ENGRAM_DIR)"
}

usage() { sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

count_model_shards() {
    find "$1" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]' || true
}

# ---------------------------- weight fetch ---------------------------------
# hf (huggingface_hub >= 0.34) or the older huggingface-cli. Resumable: both
# skip files already complete, so a killed download is restarted by re-running.
hf_cli() {
    if command -v hf >/dev/null 2>&1; then hf "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli "$@"
    else return 127
    fi
}

hf_fetch() {
    local repo="$1" dest="$2" label="$3"; shift 3
    hf_cli download "$repo" "$@" --local-dir "$dest" --max-workers "${HF_MAX_WORKERS:-8}" \
        || die "download of ${label} from ${repo} failed (re-run ./start.sh to resume, or fetch by hand into ${dest})"
}

# Engram needs only shards 47+48 of the 48-shard original (~95 GiB each), the
# index, and config.json; the other 46 shards are never read. Do not pull the
# whole 476 GiB repo.
ENGRAM_FILES=(
    "model-00047-of-00048.safetensors"
    "model-00048-of-00048.safetensors"
    "model.safetensors.index.json"
    "config.json"
)

engram_complete() {
    local f
    for f in "${ENGRAM_FILES[@]}"; do
        [ -f "$ENGRAM_DIR/$f" ] || return 1
    done
    return 0
}

fetch_weights() {
    local need_model=0 need_engram=0 have
    have="$(count_model_shards "$MODEL_HOST")"
    [ "${have:-0}" -ge "$EXPECTED_SHARDS" ] && [ -f "$MODEL_HOST/config.json" ] || need_model=1
    engram_complete || need_engram=1
    [ "$need_model" = "0" ] && [ "$need_engram" = "0" ] && return 0

    if [ "${AUTO_DOWNLOAD:-1}" != "1" ]; then
        warn "AUTO_DOWNLOAD=0 — not fetching; preflight will report what is missing"
        return 0
    fi
    command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1 \
        || die "need the Hugging Face CLI to fetch weights: pip install -U 'huggingface_hub[hf_transfer]' (or set AUTO_DOWNLOAD=0 and place them by hand)"
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

    if [ "$need_model" = "1" ]; then
        log "fetching EXL3 weights from ${HF_MODEL_REPO} into ${MODEL_HOST} (~197 GiB, resumable)"
        mkdir -p "$MODEL_HOST"
        hf_fetch "$HF_MODEL_REPO" "$MODEL_HOST" "EXL3 weights"
    fi
    if [ "$need_engram" = "1" ]; then
        log "fetching Engram shards 47+48 from ${HF_ENGRAM_REPO} into ${ENGRAM_DIR} (~190 GiB, resumable)"
        mkdir -p "$ENGRAM_DIR"
        local inc=() f
        for f in "${ENGRAM_FILES[@]}"; do inc+=(--include "$f"); done
        hf_fetch "$HF_ENGRAM_REPO" "$ENGRAM_DIR" "Engram tables" "${inc[@]}"
    fi
}

check_port_free() {
    local port="$1" name="$2"
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
        die "${name}=${port} is already bound on the head — stop the other server or pick a free port"
    fi
}

trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set it in .env"

    log "checking worker ${WORKER_SSH} ..."
    worker_ssh true 2>/dev/null \
        || die "cannot ssh (key-based) to ${WORKER_SSH} — set up passwordless ssh first"
    worker_ssh "docker info >/dev/null 2>&1" \
        || die "worker cannot talk to its docker daemon (docker group?)"
    worker_ssh "nvidia-smi -L 2>/dev/null | grep -q GB10" \
        || warn "no GB10 GPU visible on worker"

    local gid_head gid_worker gid_path
    gid_path="/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gids/${HEAD_GID}"
    gid_head=$(cat "$gid_path" 2>/dev/null | tr -d ':0' || true)
    gid_path="/sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gids/${WORKER_GID}"
    gid_worker=$(worker_ssh "cat '$gid_path' 2>/dev/null" | tr -d ':0' || true)
    if [ -z "$gid_head" ] || [ -z "$gid_worker" ]; then
        if [ -z "$gid_head" ]; then
            warn "head GID index ${HEAD_GID} is EMPTY on ${HEAD_CX7_IB}"
        fi
        if [ -z "$gid_worker" ]; then
            warn "worker GID index ${WORKER_GID} is EMPTY on ${WORKER_CX7_IB}"
        fi
        warn "GID tables — pick each node's ::ffff:<ip> entry whose type is RoCE v2:"
        for i in 0 1 2 3 4 5 6 7; do
            printf '    head   gid%s: %-40s %s\n' "$i" \
                "$(cat "/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gids/$i" 2>/dev/null)" \
                "$(cat "/sys/class/infiniband/${HEAD_CX7_IB}/ports/1/gid_attrs/types/$i" 2>/dev/null)" >&2
        done
        worker_ssh "for i in 0 1 2 3 4 5 6 7; do printf '    worker gid%s: %-40s %s\n' \"\$i\" \"\$(cat /sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gids/\$i 2>/dev/null)\" \"\$(cat /sys/class/infiniband/${WORKER_CX7_IB}/ports/1/gid_attrs/types/\$i 2>/dev/null)\"; done" >&2 || true
        die "set NCCL_IB_GID_INDEX (same index both ranks) or HEAD_GID/WORKER_GID (per rank) in .env to populated indices"
    fi

    [ "$TP" = "2" ] || warn "TP=${TP} on a 2×1-GPU cluster — expected TP=2"
    [ "$NNODES" = "2" ] || warn "NNODES=${NNODES} — expected 2"

    local others
    others=$(worker_ssh "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  ${CONTAINER_WORKER}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the worker:"
        echo "$others" >&2
        warn "this model needs most of each GB10 — stop GPU containers on the worker first"
    fi

    check_port_free "$PORT" PORT
    check_port_free "$MASTER_PORT" MASTER_PORT

    [ -f "$STOP_PATCH_HOST" ] || die "$STOP_PATCH_HOST missing"
    [ -f "$SCHED_PATCH_HOST" ] || die "$SCHED_PATCH_HOST missing"
    [ -f "$XGRAMMAR_PATCH_HOST" ] || die "$XGRAMMAR_PATCH_HOST missing"
    [ -f "$SPINWAIT_PATCH_HOST" ] || die "$SPINWAIT_PATCH_HOST missing"
    [ -f "$RESPONSES_PATCH_HOST" ] || die "$RESPONSES_PATCH_HOST missing"
    [ -f "$EXL3_OVERLAY_HOST" ] || die "$EXL3_OVERLAY_HOST missing"
    [ -f "$KMAP_HOST" ] || die "$KMAP_HOST missing"
    [ -f "$CHAT_TEMPLATE_HOST" ] || die "$CHAT_TEMPLATE_HOST missing"
    fetch_weights

    [ -f "$MODEL_HOST/config.json" ] || die "EXL3 checkpoint missing config.json at $MODEL_HOST"
    local have
    have="$(count_model_shards "$MODEL_HOST")"
    [ "${have:-0}" -ge "$EXPECTED_SHARDS" ] \
        || die "EXL3 checkpoint at $MODEL_HOST has ${have:-0}/$EXPECTED_SHARDS shards"
    [ -f "$ENGRAM_DIR/model-00047-of-00048.safetensors" ] \
        || die "Engram shard 47 missing under ENGRAM_DIR=$ENGRAM_DIR"
    [ -f "$ENGRAM_DIR/model-00048-of-00048.safetensors" ] \
        || die "Engram shard 48 missing under ENGRAM_DIR=$ENGRAM_DIR"
    [ -f "$ENGRAM_DIR/model.safetensors.index.json" ] \
        || die "Engram index missing under ENGRAM_DIR=$ENGRAM_DIR"

    check_memory_headroom

    WEIGHT_BACKEND="$(resolve_weight_backend)"
    log "weight backend: ${WEIGHT_BACKEND} (WEIGHT_SYNC=${WEIGHT_SYNC})"
    case "$WEIGHT_BACKEND" in
        nfs)
            HEAD_ENGRAM_BIND="$ENGRAM_SRC"
            WORKER_MODEL_BIND="$NFS_VOLUME_MODEL"
            WORKER_ENGRAM_BIND="$NFS_VOLUME_ENGRAM"
            ;;
        zfs)
            zfs_preflight
            HEAD_ENGRAM_BIND="$ENGRAM_SRC"
            WORKER_MODEL_BIND="$WORKER_MODEL_DIR"
            WORKER_ENGRAM_BIND="$WORKER_ENGRAM_DIR"
            ;;
        rsync)
            HEAD_ENGRAM_BIND="$ENGRAM_SRC"
            WORKER_MODEL_BIND="$WORKER_MODEL_DIR"
            WORKER_ENGRAM_BIND="$WORKER_ENGRAM_DIR"
            if ! worker_ssh "mkdir -p '$WORKER_MODEL_DIR' '$WORKER_ENGRAM_DIR' && test -w '$WORKER_MODEL_DIR' && test -w '$WORKER_ENGRAM_DIR'"; then
                die "worker cannot write $WORKER_MODEL_DIR / $WORKER_ENGRAM_DIR — fix ownership"
            fi
            local need_b avail_b
            need_b=$(( $(du -sb "$MODEL_HOST" | awk '{print $1}') + $(zfs_bytes_for_engram) ))
            avail_b="$(worker_ssh "df -PB1 '$WORKER_HOME' | awk 'NR==2{print \$4}'" || true)"
            if [ -n "${avail_b:-}" ] && [ "$avail_b" -lt "$need_b" ]; then
                die "worker has $((avail_b/1024/1024/1024)) GiB free under $WORKER_HOME, need ~$((need_b/1024/1024/1024)) GiB for a local EXL3+Engram copy. Use WEIGHT_SYNC=nfs (default; no local copy) instead of rsync."
            fi
            ;;
    esac

    log "preflight OK (head=$(hostname) ${HEAD_IP}, worker=${WORKER_SSH})"
}

# ------------------------------ image --------------------------------------
image_from_registry() {
    case "$IMAGE" in
        */*) return 0 ;;
        *) return 1 ;;
    esac
}

login_ghcr_if_token() {
    [ -n "${GHCR_TOKEN:-}" ] || return 0
    log "docker login ghcr.io as ${GHCR_USER} (GHCR_TOKEN)"
    echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin >/dev/null
}

login_ghcr_if_token_worker() {
    [ -n "${GHCR_TOKEN:-}" ] || return 0
    log "docker login ghcr.io on worker as ${GHCR_USER} (GHCR_TOKEN)"
    echo "$GHCR_TOKEN" | worker_ssh "docker login ghcr.io -u '$GHCR_USER' --password-stdin" >/dev/null
}

# Identity for "does the worker already have the head's image?". No single
# field survives every path: overlay2 and containerd disagree on .Id (config
# digest vs index digest, issue #8), and docker save | docker load drops
# RepoDigests, so a shipped image never matched the GHCR tag it came from and
# we re-shipped the whole image on every run. RootFS.Layers (diff IDs) is
# identical on both sides in both cases — fold it into a short digest (the
# full layer list does not belong in a log line) and keep RepoDigest/.Id only
# as fallbacks for the rare inspect that reports no layers.
_IMAGE_KEY_FMT='{{if .RootFS.Layers}}layers {{join .RootFS.Layers ","}}{{else if .RepoDigests}}other {{index .RepoDigests 0}}{{else}}other {{.Id}}{{end}}'

parse_image_key() {
    local raw
    raw="$(tr -d '\r' | sed -n 's/^GLM53KEY //p' | tail -n 1)"
    case "$raw" in
        "layers "*) printf 'layers:%s' "$(printf '%s' "${raw#layers }" | sha256sum | cut -c1-16)" ;;
        "other "*)  printf '%s' "${raw#other }" ;;
    esac
}

local_image_key() {
    docker image inspect -f "GLM53KEY ${_IMAGE_KEY_FMT}" "$IMAGE" 2>/dev/null | parse_image_key
}

worker_image_key() {
    worker_ssh "docker image inspect -f 'GLM53KEY ${_IMAGE_KEY_FMT}' '$IMAGE' 2>/dev/null" | parse_image_key
}

images_match() {
    [ -n "${1:-}" ] && [ -n "${2:-}" ] && [ "$1" = "$2" ]
}

image_platform() {
    if [ -n "${IMAGE_PLATFORM:-}" ]; then
        printf '%s' "$IMAGE_PLATFORM"
        return
    fi
    local p
    p="$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$IMAGE" 2>/dev/null || true)"
    printf '%s' "${p:-linux/arm64}"
}

# Hash of Dockerfile + overlay/tests/files/ablit inputs that docker COPY.
# Compared to LABEL dsv41.recipe.stamp so a git pull rebuilds once.
# Content hash of everything that goes into the image. sha256sum prints the
# path next to each digest and we hash that text, so the paths must be relative
# to the checkout: with absolute ones the stamp changed with the clone
# directory, no clone could ever match the published image's label, and every
# node rebuilt instead of pulling.
overlay_recipe_hash() {
    (
        cd "$SCRIPT_DIR" || exit 1
        {
            printf '%s\n' "Dockerfile"
            find overlay files tests \
                -type f \
                ! -path '*/__pycache__/*' \
                ! -name '*.pyc' \
                ! -name '*.so' \
                2>/dev/null
        } | LC_ALL=C sort | xargs -d '\n' -r sha256sum | sha256sum | awk '{print $1}'
    )
}

image_recipe_stamp() {
    local stamp
    stamp="$(docker image inspect -f '{{ index .Config.Labels "dsv41.recipe.stamp" }}' "$IMAGE" 2>/dev/null || true)"
    case "$stamp" in
        ""|"<no value>"|"<nil>") printf '' ;;
        *) printf '%s' "$stamp" ;;
    esac
}

build_image() {
    local stamp
    stamp="$(overlay_recipe_hash)"
    log "building ${IMAGE} (DeepSeek-V4.1 EXL3 + SM121 ext) stamp=${stamp:0:12} (log: $LOGDIR/build-sm121.log) ..."
    docker build --build-arg "DSV41_RECIPE_STAMP=$stamp" -t "$IMAGE" "$SCRIPT_DIR" \
        >"$LOGDIR/build-sm121.log" 2>&1 \
        || { tail -n 40 "$LOGDIR/build-sm121.log" >&2; die "docker build of $IMAGE failed"; }
}

# Non-fatal: ensure_image falls back to building from this checkout when the
# registry is unreachable, the tag is private, or there is no network at all.
pull_image() {
    login_ghcr_if_token
    log "pulling ${IMAGE} ..."
    docker pull "$IMAGE"
}

pull_image_on_worker() {
    login_ghcr_if_token_worker
    log "pulling ${IMAGE} on worker ..."
    worker_ssh "docker pull '$IMAGE'"
}

# docker save | ssh docker load. One stream, no staging disk on either node,
# but not resumable: a dropped link costs the whole 22 GB again.
ship_image_stream() {
    local platform
    platform="$(image_platform)"
    log "shipping ${IMAGE} (${platform}) to worker via docker save | ssh docker load ..."
    # A multi-arch OCI index references blobs docker save does not pack
    # (only the native platform is local). docker load then dies with:
    #   open /var/lib/docker/tmp/docker-import-*/blobs/sha256/<id>: no such file
    # (issue #8). --platform emits a complete single-manifest tar.
    if docker save --platform "$platform" "$IMAGE" | worker_ssh docker load; then
        return 0
    fi
    warn "docker save --platform ${platform} failed — retrying without --platform"
    docker save "$IMAGE" | worker_ssh docker load
}

# Stage the image as a tar on the head, rsync it (resumable), then load it on
# the worker. Costs tar-sized scratch on both nodes; buys a ship that survives
# an interrupted link, which the streaming pipe does not. docker save is
# deterministic for a given image id, so a half-sent tar resumes byte-for-byte.
ship_image_rsync() {
    local platform tar idfile remote_tar have_id avail_kb need_kb tar_kb
    platform="$(image_platform)"
    mkdir -p "$IMAGE_SHIP_DIR" || return 1
    tar="$IMAGE_SHIP_DIR/$(printf '%s' "$IMAGE" | tr '/:' '__').tar"
    idfile="${tar}.imageid"
    remote_tar="${WORKER_IMAGE_SHIP_DIR}/$(basename "$tar")"
    local want_id
    want_id="$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || true)"

    have_id="$(cat "$idfile" 2>/dev/null || true)"
    if [ -s "$tar" ] && [ "$have_id" = "$want_id" ]; then
        log "reusing staged image tar $(du -h "$tar" | cut -f1) ($(basename "$tar"))"
    else
        rm -f "$tar" "$idfile"
        log "staging ${IMAGE} (${platform}) to ${tar} ..."
        if ! docker save --platform "$platform" -o "$tar" "$IMAGE"; then
            warn "docker save --platform ${platform} failed — retrying without --platform"
            docker save -o "$tar" "$IMAGE" || { rm -f "$tar"; return 1; }
        fi
        printf '%s' "$want_id" > "$idfile"
    fi

    # docker save can exit 0 and still leave nothing usable behind (a full disk
    # truncates it). Without this the size arithmetic below runs on an empty
    # string and the ship dies on a shell syntax error instead of saying why.
    if [ ! -s "$tar" ]; then
        warn "staged image tar ${tar} is missing or empty after docker save"
        rm -f "$tar" "$idfile"
        return 1
    fi

    tar_kb=$(( $(stat -c %s "$tar") / 1024 ))
    # docker load rehydrates every layer the worker does not already have, so
    # the tar alone is only half the bill on a worker that has no copy yet.
    need_kb=$tar_kb
    if ! worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
        need_kb=$(( tar_kb * 2 ))
    fi
    need_kb=$(( need_kb + 5 * 1024 * 1024 ))   # 5 GiB margin
    avail_kb="$(worker_ssh "df -Pk '${WORKER_IMAGE_SHIP_DIR}' 2>/dev/null || df -Pk /tmp" | awk 'NR==2 {print $4}')"
    if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$need_kb" ]; then
        warn "worker has $(( avail_kb / 1024 / 1024 )) GiB free, rsync ship wants $(( need_kb / 1024 / 1024 )) GiB — streaming instead"
        return 1
    fi

    log "rsyncing $(du -h "$tar" | cut -f1) to ${WORKER_SSH}:${remote_tar} (resumable) ..."
    worker_ssh "mkdir -p '${WORKER_IMAGE_SHIP_DIR}'" || return 1
    rsync -a --partial --inplace --info=progress2 "$tar" "${WORKER_SSH}:${remote_tar}" || return 1
    log "loading ${IMAGE} on the worker ..."
    worker_ssh "docker load -i '${remote_tar}'" || return 1
    worker_ssh "rm -f '${remote_tar}'" || true
    [ "${IMAGE_SHIP_KEEP:-0}" = "1" ] || rm -f "$tar" "$idfile"
    return 0
}

ship_image_to_worker() {
    case "${IMAGE_SHIP:-rsync}" in
        stream) ship_image_stream ;;
        rsync)  ship_image_rsync || die "rsync image ship failed (IMAGE_SHIP=stream to pipe it instead)" ;;
        auto)   ship_image_rsync || { warn "falling back to the streaming ship"; ship_image_stream; } ;;
        *)      die "unknown IMAGE_SHIP=${IMAGE_SHIP} (rsync|stream|auto)" ;;
    esac
}

ensure_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 worker_ok=0 head_key="" worker_key=""
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        head_ok=1
        head_key="$(local_image_key)"
    fi
    if worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
        worker_key="$(worker_image_key)"
        if images_match "$head_key" "$worker_key"; then
            worker_ok=1
        else
            worker_ok=0
            log "worker image differs (head=${head_key:-none} worker=${worker_key:-none}) — will refresh worker"
        fi
    fi
    local skip_pull="${SKIP_PULL:-0}"
    [ "${PULL:-0}" = "1" ] && skip_pull=0
    local wanted_stamp have_stamp have_short
    wanted_stamp="$(overlay_recipe_hash)"
    have_stamp=""
    [ "$head_ok" = "1" ] && have_stamp="$(image_recipe_stamp)"
    have_short="${have_stamp:0:12}"
    # The published image is the fast path: a clean checkout of this repo hashes
    # to the same recipe stamp the image carries, so a new node pulls ~9 GiB and
    # serves it. Compiling exllamav3_ext for SM121 is the fallback, taken only
    # when a local edit to Dockerfile/overlay/files/tests moves the stamp, when
    # BUILD=1 asks for it, or when the registry cannot be reached.
    if [ "${BUILD:-0}" != "1" ] \
       && [ "$skip_pull" != "1" ] \
       && image_from_registry \
       && { [ "$have_stamp" != "$wanted_stamp" ] || [ "${PULL:-0}" = "1" ]; }; then
        local before_key="$head_key"
        if pull_image; then
            head_ok=1
            head_key="$(local_image_key)"
            have_stamp="$(image_recipe_stamp)"
            have_short="${have_stamp:0:12}"
            if [ "$head_key" != "$before_key" ]; then
                log "pulled ${IMAGE} (${before_key:-missing} -> ${head_key})"
            else
                log "${IMAGE} already current"
            fi
            if images_match "$worker_key" "$head_key"; then
                worker_ok=1
            else
                worker_ok=0
            fi
        else
            warn "docker pull ${IMAGE} failed — falling back to a local build"
        fi
    fi
    if [ "${BUILD:-0}" != "1" ] && [ "${SKIP_BUILD:-0}" != "1" ]; then
        if [ "$head_ok" = "0" ]; then
            log "no ${IMAGE} on the head — building from this checkout"
            BUILD=1
        elif [ "$have_stamp" != "$wanted_stamp" ]; then
            log "image recipe ${have_short:-none} != repo ${wanted_stamp:0:12} — rebuilding (SKIP_BUILD=1 keeps the pulled image)"
            BUILD=1
        fi
    elif [ "${SKIP_BUILD:-0}" = "1" ] && [ "$have_stamp" != "$wanted_stamp" ]; then
        warn "SKIP_BUILD=1 — not rebuilding; stamp ${have_short:-none} != repo ${wanted_stamp:0:12}"
    fi
    if [ "${BUILD:-0}" = "1" ]; then
        build_image
        head_key="$(local_image_key)"
        head_ok=1
        worker_ok=0
    elif [ "$head_ok" = "0" ]; then
        die "${IMAGE} is not on the head: the pull did not run or failed, and a build was not allowed (SKIP_PULL=${skip_pull} SKIP_BUILD=${SKIP_BUILD:-0})"
    fi
    if [ "${SKIP_SHIP:-0}" = "1" ]; then
        [ "$worker_ok" = "1" ] || warn "SKIP_SHIP=1 — not copying ${IMAGE} to the worker"
    elif [ "$worker_ok" = "0" ]; then
        if image_from_registry && [ "$skip_pull" != "1" ] && [ "${BUILD:-0}" != "1" ]; then
            if pull_image_on_worker; then
                worker_key="$(worker_image_key)"
                if images_match "$head_key" "$worker_key"; then
                    worker_ok=1
                    log "worker pulled ${IMAGE} — matches head"
                else
                    warn "worker pull left a different image (head=${head_key:-none} worker=${worker_key:-none}) — shipping"
                fi
            else
                warn "worker docker pull failed — shipping over SSH (worker does not need GHCR)"
            fi
        fi
        if [ "$worker_ok" = "0" ]; then
            ship_image_to_worker
            worker_key="$(worker_image_key)"
            if images_match "$head_key" "$worker_key"; then
                worker_ok=1
            elif worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
                warn "worker has ${IMAGE} after ship but keys still differ (head=${head_key:-none} worker=${worker_key:-none}) — continuing"
                worker_ok=1
            else
                die "worker still missing ${IMAGE} after ship"
            fi
        fi
    fi
    if [ "${SKIP_OVERLAY_VERIFY:-0}" != "1" ]; then
        log "GPU EXL3 self-check on ${IMAGE} (log: $LOGDIR/overlay-verify.log) ..."
        # GEMM parity plus the Engram fp8 row dequant (the kernel must decode
        # fp8 e4m3 bytes; as integers every boot up to 2026-09-12 spoke garbage).
        docker run --rm --gpus all \
            -e EXL3_SELFCHECK_GPU=1 \
            --entrypoint bash "$IMAGE" -c "python3 /opt/dsv41/test_exl3_overlay.py && python3 /opt/dsv41/test_engram_dequant.py" \
            >"$LOGDIR/overlay-verify.log" 2>&1 \
            || { tail -n 80 "$LOGDIR/overlay-verify.log" >&2; die "EXL3 overlay GPU self-check failed"; }
        log "overlay verify OK"
    fi
    log "image ready on both nodes"
}


# ---------------------------- local weights --------------------------------
check_weights() {
    local have
    have="$(count_model_shards "$MODEL_HOST")"
    log "EXL3 checkpoint: $MODEL_HOST ($have shards, expected $EXPECTED_SHARDS)"
    [ "${have:-0}" -ge "$EXPECTED_SHARDS" ] || die "incomplete EXL3 tree"
}

sync_dir_to_worker() {
    local src="$1" dest="$2" label="$3" marker="$4" rev
    rev="$(python3 - <<PY
from pathlib import Path
p = Path("$src")
parts = []
for f in sorted(p.glob("model-*.safetensors")):
    st = f.stat()
    parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
print("|".join(parts) or "empty")
PY
)"
    if [ "${FORCE_SYNC:-0}" != "1" ] \
       && [ "$(worker_ssh "cat '$marker' 2>/dev/null" || true)" = "$rev" ]; then
        log "worker ${label} already in sync — rsync skipped (FORCE_SYNC=1 to force)"
        return 0
    fi
    log "syncing ${label} to ${WORKER_SSH}:${dest} ..."
    worker_ssh "mkdir -p '$dest'"
    rsync -a --partial --info=progress2 "$src/" "${WORKER_SSH}:${dest}/"
    worker_ssh "printf '%s' '$rev' > '$marker'"
}

sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not sharing/syncing to worker"; return; }
    prepare_engram_src_dir
    local backend="${WEIGHT_BACKEND:-$(resolve_weight_backend)}"
    case "$backend" in
        nfs)
            nfs_share
            log "worker weights in sync (NFS, no local copy)"
            ;;
        zfs)
            zfs_sync_weights
            WORKER_MODEL_BIND="$WORKER_MODEL_DIR"
            WORKER_ENGRAM_BIND="$WORKER_ENGRAM_DIR"
            log "worker weights in sync (ZFS)"
            ;;
        rsync)
            log "rsync fallback (node-local copy; prefer WEIGHT_SYNC=nfs)"
            WORKER_MODEL_BIND="$WORKER_MODEL_DIR"
            WORKER_ENGRAM_BIND="$WORKER_ENGRAM_DIR"
            sync_dir_to_worker "$MODEL_HOST" "$WORKER_MODEL_DIR" "EXL3 weights" \
                "${WORKER_MODEL_DIR}/.dsv41-exl3-synced"
            sync_dir_to_worker "$ENGRAM_SRC" "$WORKER_ENGRAM_DIR" "slim Engram 47+48" \
                "${WORKER_ENGRAM_DIR}/.dsv41-engram-synced"
            log "worker weights in sync (rsync of EXL3 + slim Engram, not the 476 GiB native tree)"
            ;;
        *)
            die "unknown weight backend $backend"
            ;;
    esac
}

share_weights() {
    preflight
    WEIGHT_BACKEND=nfs
    WEIGHT_SYNC=nfs
    prepare_engram_src_dir
    nfs_share
}

pack_engram() {
    preflight
    ensure_image
    prepare_engram_src_dir
    local backend="${WEIGHT_BACKEND:-$(resolve_weight_backend)}"
    if [ "$backend" = "nfs" ]; then
        nfs_share
    else
        sync_weights
    fi
    mkdir -p "$HOME/dsv41-engram"
    log "packing head Engram rank 0 of TP=${TP} into $HOME/dsv41-engram ..."
    docker run --rm --network host \
        -v "$ENGRAM_SRC:/models:ro" \
        -v "$HOME/dsv41-engram:/engram" \
        --entrypoint python3 "$IMAGE" \
        /opt/dsv41/pack_engram.py --model /models --rank 0 --tp "$TP" --out /engram
    log "packing worker Engram rank 1 (from NFS/slim src, not a 190 GiB copy) ..."
    worker_ssh "mkdir -p '$WORKER_HOME/dsv41-engram'"
    local pack_src="$WORKER_ENGRAM_BIND"
    [ "$backend" = "nfs" ] && pack_src="$NFS_VOLUME_ENGRAM"
    worker_ssh "docker run --rm --network host \
        -v '$pack_src:/models:ro' \
        -v '$WORKER_HOME/dsv41-engram:/engram' \
        --entrypoint python3 '$IMAGE' \
        /opt/dsv41/pack_engram.py --model /models --rank 1 --tp '$TP' --out /engram"
    log "packed Engram shards written (vLLM hash-head ranges). File-backed lookup uses them when mounted at /engram-packed."
}


# ------------------------ inner container scripts --------------------------
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[dsv41-exl3-head] $*"; }

SITE="$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).resolve().parent)')"
export GLM53_DETOKENIZER_PY="$SITE/v1/engine/detokenizer.py"
export GLM53_SCHEDULER_PY="$SITE/v1/core/sched/scheduler.py"
export GLM53_SPINWAIT_TARGET="$SITE/distributed/device_communicators/shm_broadcast.py"
export GLM53_XGRAMMAR_BACKEND_PY="$SITE/v1/structured_output/backend_xgrammar.py"
export GLM53_XGRAMMAR_MANAGER_PY="$SITE/v1/structured_output/__init__.py"
if [ -f /opt/dsv41/exl3.py ]; then
    cp /opt/dsv41/exl3.py "$SITE/model_executor/layers/quantization/exl3.py"
    say "installed runtime EXL3 overlay"
fi
say "MemAvailable=$(awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo) GiB engram: cache=${DSV41_CACHE_GIB:-?}GiB resident_scales=${DSV41_RESIDENT_SCALES:-default} io_threads=${DSV41_IO_THREADS:-default}"

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 0
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41
    --reasoning-parser deepseek_v41
    --enable-auto-tool-choice
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-enable-flashinfer-autotune
    --hf-overrides "{\"engram_table_dir\":\"${ENGRAM_MOUNT}\"}"
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
# Pin the KV pool. Unpinned, vLLM sizes it from host MemAvailable on this UMA
# box (MemorySnapshot uses psutil on integrated GPUs) and the result swings
# with page cache; pinned, memory profiling is skipped entirely.
[ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
# 64-token KV blocks: the SM12x DeepGEMM paged indexer kernel accepts 32/64
# states per block only (patch_sm120_block64.py lets the backends take 64).
[ -n "${KV_BLOCK_SIZE:-}" ] && ARGS+=(--block-size "${KV_BLOCK_SIZE}")
if [ "${SPEC_METHOD:-dspark}" = "dspark" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
print(json.dumps({"method":"dspark","num_speculative_tokens":int(os.environ.get("DSPARK_TOKENS","5"))},separators=(",",":")))')")
elif [ "${SPEC_METHOD:-dspark}" = "none" ]; then
    :
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
    say "language-model-only: no vision tower"
else
    ARGS+=(--mm-encoder-tp-mode data)
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
    say "vision on: limit-mm=${LIMIT_MM:-} skip-mm-profiling=${SKIP_MM_PROFILING:-1}"
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
[ -d "${ENGRAM_MOUNT}" ] || { say "FATAL: ${ENGRAM_MOUNT} missing"; exit 1; }
for p in /opt/dsv41/patch_suppress_stops_in_reasoning.py \
         /opt/dsv41/patch_scheduler_decode_floor.py \
         /opt/dsv41/patch_xgrammar_termination.py \
         /opt/dsv41/patch_spinwait.py \
         /opt/dsv41/patch_responses_content_types.py \
         /opt/dsv41/patch_exl3_packed_names.py \
         /opt/dsv41/patch_exl3_lm_head.py \
         /opt/dsv41/patch_engram_secondary.py \
         /opt/dsv41/patch_engram_file.py \
         /opt/dsv41/patch_h2d_stage.py \
         /opt/dsv41/patch_sm120_block64.py \
         /opt/dsv41/patch_memory_log.py; do
    if [ -f "$p" ]; then
        python3 "$p" || say "WARN: $p returned $?"
    fi
done
say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]}"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF

    cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
set -euo pipefail
say() { echo "[dsv41-exl3-worker] $*"; }

SITE="$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).resolve().parent)')"
export GLM53_DETOKENIZER_PY="$SITE/v1/engine/detokenizer.py"
export GLM53_SCHEDULER_PY="$SITE/v1/core/sched/scheduler.py"
export GLM53_SPINWAIT_TARGET="$SITE/distributed/device_communicators/shm_broadcast.py"
export GLM53_XGRAMMAR_BACKEND_PY="$SITE/v1/structured_output/backend_xgrammar.py"
export GLM53_XGRAMMAR_MANAGER_PY="$SITE/v1/structured_output/__init__.py"
if [ -f /opt/dsv41/exl3.py ]; then
    cp /opt/dsv41/exl3.py "$SITE/model_executor/layers/quantization/exl3.py"
    say "installed runtime EXL3 overlay"
fi
say "MemAvailable=$(awk '/^MemAvailable:/ { printf "%.1f", $2 / 1048576 }' /proc/meminfo) GiB engram: cache=${DSV41_CACHE_GIB:-?}GiB resident_scales=${DSV41_RESIDENT_SCALES:-default} io_threads=${DSV41_IO_THREADS:-default}"

ARGS=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --nnodes "${NNODES}"
    --node-rank 1
    --master-addr "${HEAD_IP}"
    --master-port "${MASTER_PORT}"
    --distributed-executor-backend mp
    --headless
    --tokenizer-mode deepseek_v41
    --tool-call-parser deepseek_v41
    --reasoning-parser deepseek_v41
    --enable-auto-tool-choice
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --no-enable-flashinfer-autotune
    --hf-overrides "{\"engram_table_dir\":\"${ENGRAM_MOUNT}\"}"
)
[ "${ENFORCE_EAGER:-1}" = "1" ] && ARGS+=(--enforce-eager)
[ -n "${QUANTIZATION:-}" ] && [ "${QUANTIZATION}" != "none" ] && ARGS+=(--quantization "${QUANTIZATION}")
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${MAX_NUM_SEQS:-}" ] && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ] && ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ] && ARGS+=(--long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}")
[ -n "${KV_CACHE_DTYPE:-}" ] && ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
# Pin the KV pool. Unpinned, vLLM sizes it from host MemAvailable on this UMA
# box (MemorySnapshot uses psutil on integrated GPUs) and the result swings
# with page cache; pinned, memory profiling is skipped entirely.
[ -n "${KV_CACHE_MEMORY_BYTES:-}" ] && ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
# 64-token KV blocks: the SM12x DeepGEMM paged indexer kernel accepts 32/64
# states per block only (patch_sm120_block64.py lets the backends take 64).
[ -n "${KV_BLOCK_SIZE:-}" ] && ARGS+=(--block-size "${KV_BLOCK_SIZE}")
if [ "${SPEC_METHOD:-dspark}" = "dspark" ]; then
    ARGS+=(--speculative-config "$(python3 -S -c 'import json,os
print(json.dumps({"method":"dspark","num_speculative_tokens":int(os.environ.get("DSPARK_TOKENS","5"))},separators=(",",":")))')")
elif [ "${SPEC_METHOD:-dspark}" = "none" ]; then
    :
fi
if [ -n "${CHAT_TEMPLATE:-}" ] && [ -f "${CHAT_TEMPLATE}" ]; then
    ARGS+=(--chat-template "${CHAT_TEMPLATE}")
fi
if [ "${LANGUAGE_MODEL_ONLY:-0}" = "1" ]; then
    ARGS+=(--language-model-only)
else
    ARGS+=(--mm-encoder-tp-mode data)
    [ -n "${LIMIT_MM:-}" ] && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
    [ "${SKIP_MM_PROFILING:-1}" = "1" ] && ARGS+=(--skip-mm-profiling)
fi
if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

[ -f "${MODEL_DIR}/config.json" ] || { say "FATAL: ${MODEL_DIR}/config.json missing"; ls -la "${MODEL_DIR}" | head; exit 1; }
[ -d "${ENGRAM_MOUNT}" ] || { say "FATAL: ${ENGRAM_MOUNT} missing"; exit 1; }
for p in /opt/dsv41/patch_suppress_stops_in_reasoning.py \
         /opt/dsv41/patch_scheduler_decode_floor.py \
         /opt/dsv41/patch_xgrammar_termination.py \
         /opt/dsv41/patch_spinwait.py \
         /opt/dsv41/patch_responses_content_types.py \
         /opt/dsv41/patch_exl3_packed_names.py \
         /opt/dsv41/patch_exl3_lm_head.py \
         /opt/dsv41/patch_engram_secondary.py \
         /opt/dsv41/patch_engram_file.py \
         /opt/dsv41/patch_h2d_stage.py \
         /opt/dsv41/patch_sm120_block64.py \
         /opt/dsv41/patch_memory_log.py; do
    if [ -f "$p" ]; then
        python3 "$p" || say "WARN: $p returned $?"
    fi
done
say "joining TP2 at ${HEAD_IP}:${MASTER_PORT} as rank 1"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}"
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}


# --------------------------- memory headroom -------------------------------
# GB10 unified memory: the ~99.5 GiB of EXL3 weights per rank is committed at
# cudaMalloc time and never shows up in any process RSS, so the only reliable
# pre-flight signal is MemAvailable on each node. A stray GPU container (the
# native 3-Spark serve, an old worker) leaves no room and the boot ends in the
# kernel OOM killer taking the desktop apart (2026-09-11, HANDOFF.md).
check_memory_headroom() {
    local per_rank_bytes weights_gib need_gib head_gib worker_gib bad=0
    per_rank_bytes=$(python3 "$WEIGHT_BUDGET_HOST" --model "$MODEL_HOST" --tp "$TP" --json 2>/dev/null \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["per_rank_bytes"])' 2>/dev/null || echo 0)
    if [ "${per_rank_bytes:-0}" -le 0 ] 2>/dev/null; then
        warn "weight budget unavailable ($WEIGHT_BUDGET_HOST) — skipping headroom check"
        return 0
    fi
    weights_gib=$(awk -v b="$per_rank_bytes" 'BEGIN { printf "%.1f", b / 1073741824 }')
    need_gib=$(awk -v b="$per_rank_bytes" -v m="$DSV41_BOOT_MARGIN_GIB" 'BEGIN { printf "%.1f", b / 1073741824 + m }')
    head_gib=$(mem_avail_gib)
    worker_gib=$(worker_mem_avail_gib)
    log "memory headroom: weights ${weights_gib} GiB/rank + ${DSV41_BOOT_MARGIN_GIB} GiB margin = need ${need_gib} GiB MemAvailable; head=${head_gib} worker=${worker_gib}"
    if ! awk -v a="$head_gib" -v n="$need_gib" 'BEGIN { exit !(a + 0 >= n + 0) }'; then
        warn "head has only ${head_gib} GiB available — running containers:"
        docker ps --format '  {{.Names}}  ({{.Image}})' >&2 || true
        bad=1
    fi
    if ! awk -v a="$worker_gib" -v n="$need_gib" 'BEGIN { exit !(a + 0 >= n + 0) }'; then
        warn "worker has only ${worker_gib} GiB available — running containers:"
        worker_ssh "docker ps --format '  {{.Names}}  ({{.Image}})'" >&2 || true
        bad=1
    fi
    [ "$bad" = "0" ] || die "not enough free unified memory to boot safely (stop other GPU workloads; DSV41_BOOT_MARGIN_GIB=${DSV41_BOOT_MARGIN_GIB} is the margin)"
}

# ----------------------------- memory guards -------------------------------
start_memguards() {
    mkdir -p "$LOGDIR"
    stop_memguards
    if [ "${DSV41_MEM_GUARD:-0}" != "1" ]; then
        log "memory guards disabled (DSV41_MEM_GUARD=0) — no container is killed on low MemAvailable"
        return 0
    fi
    : >"$LOGDIR/memguard-head.log"
    worker_ssh ": >/tmp/dsv41-memguard-worker.log" 2>/dev/null || true
    nohup setsid bash "$MEMGUARD_HOST" "$CONTAINER_HEAD" "$DSV41_MEM_GUARD_GIB" \
        "$LOGDIR/memguard-head.log" "$LOGDIR/memguard-head.pid" >/dev/null 2>&1 &
    worker_ssh "nohup setsid bash /tmp/dsv41-memguard.sh '$CONTAINER_WORKER' '$DSV41_MEM_GUARD_GIB' \
        /tmp/dsv41-memguard-worker.log /tmp/dsv41-memguard-worker.pid >/dev/null 2>&1 &" \
        || warn "worker memory guard did not start"
    log "memory guards armed: a node's container is killed when its MemAvailable < ${DSV41_MEM_GUARD_GIB} GiB (logs: $LOGDIR/memguard-head.log, worker:/tmp/dsv41-memguard-worker.log)"
}

stop_memguards() {
    local pf="$LOGDIR/memguard-head.pid"
    if [ -f "$pf" ]; then
        kill "$(cat "$pf")" 2>/dev/null || true
        rm -f "$pf"
    fi
    worker_ssh "if [ -f /tmp/dsv41-memguard-worker.pid ]; then kill \$(cat /tmp/dsv41-memguard-worker.pid) 2>/dev/null; rm -f /tmp/dsv41-memguard-worker.pid; fi" 2>/dev/null || true
}

# ------------------------------- launch ------------------------------------
launch_cluster() {
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || true

    mkdir -p "$CACHE_ROOT" "$TRITON_HOST_CACHE" "$TILELANG_HOST_CACHE"
    worker_ssh "mkdir -p '$WORKER_VLLM_CACHE' '$WORKER_TRITON_CACHE' '$WORKER_TILELANG_CACHE'"
    scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${WORKER_SSH}:/tmp/${CONTAINER_WORKER}.sh"
    scp -q -o BatchMode=yes "$CHAT_TEMPLATE_HOST" "${WORKER_SSH}:/tmp/dsv41-chat_template.jinja"
    scp -q -o BatchMode=yes "$STOP_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_suppress_stops_in_reasoning.py"
    scp -q -o BatchMode=yes "$SCHED_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_scheduler_decode_floor.py"
    scp -q -o BatchMode=yes "$XGRAMMAR_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_xgrammar_termination.py"
    scp -q -o BatchMode=yes "$SPINWAIT_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_spinwait.py"
    scp -q -o BatchMode=yes "$RESPONSES_PATCH_HOST" "${WORKER_SSH}:/tmp/patch_responses_content_types.py"
    scp -q -o BatchMode=yes "$EXL3_OVERLAY_HOST" "${WORKER_SSH}:/tmp/dsv41-exl3.py"
    scp -q -o BatchMode=yes "$KMAP_HOST" "${WORKER_SSH}:/tmp/exl3_k_map.json"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_exl3_packed_names.py" "${WORKER_SSH}:/tmp/patch_exl3_packed_names.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_exl3_lm_head.py" "${WORKER_SSH}:/tmp/patch_exl3_lm_head.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_engram_secondary.py" "${WORKER_SSH}:/tmp/patch_engram_secondary.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_engram_file.py" "${WORKER_SSH}:/tmp/patch_engram_file.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/engram_file_backend.py" "${WORKER_SSH}:/tmp/engram_file_backend.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/engram_layout.py" "${WORKER_SSH}:/tmp/engram_layout.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_memory_log.py" "${WORKER_SSH}:/tmp/patch_memory_log.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_h2d_stage.py" "${WORKER_SSH}:/tmp/patch_h2d_stage.py"
    scp -q -o BatchMode=yes "$SCRIPT_DIR/overlay/patch_sm120_block64.py" "${WORKER_SSH}:/tmp/patch_sm120_block64.py"
    scp -q -o BatchMode=yes "$MEMGUARD_HOST" "${WORKER_SSH}:/tmp/dsv41-memguard.sh"

    local -a nccl_common=(
        -e NCCL_IB_DISABLE=0
        -e NCCL_IB_ROCE_VERSION_NUM=2
        -e NCCL_NET=IB
        -e NCCL_NET_PLUGIN=none
        -e NCCL_NVLS_ENABLE=0
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_MERGE_NICS=0
        -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
        -e NCCL_IGNORE_CPU_AFFINITY=1
        -e "NCCL_DEBUG=$NCCL_DEBUG"
        -e "NCCL_BUFFSIZE=$NCCL_BUFFSIZE"
        -e "NCCL_LL128_BUFFSIZE=$NCCL_LL128_BUFFSIZE"
        -e "NCCL_PROTO=$NCCL_PROTO"
        -e "NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS"
        # vLLM runs the shared experts on a side CUDA stream for batches up to
        # VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD (256) tokens, concurrently
        # with the routed experts. Here both are EXL3 and every exllamav3
        # trellis kernel (exl3_gemm split-K tile locks, exl3_moe group
        # barriers) shares ONE per-device lock buffer, so two concurrent EXL3
        # kernels corrupt each other's locks and spin forever: boots 11-13 on
        # 2026-09-12 hung in the first <=256-token forward (both GPUs 96 %,
        # ranks parked in cudaStreamSynchronize). Keep the shared experts on
        # the main stream.
        -e "VLLM_DISABLE_SHARED_EXPERTS_STREAM=${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}"
        # Diagnostics (default off): CUDA_LAUNCH_BLOCKING=1 makes every launch
        # synchronous so a hang's Python stack names the kernel; pair it with
        # ENFORCE_EAGER=1. NCCL_DEBUG_SUBSYS narrows NCCL_DEBUG=INFO output.
        -e "CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
        # 1 (default) = no aux CUDA streams in the DeepSeek V4.1 model: EXL3
        # kernels share one lock buffer per device (patch_sm120_block64.py).
        -e "DSV41_EXL3_SERIAL_STREAMS=${DSV41_EXL3_SERIAL_STREAMS:-1}"
        -e "DSV41_ENGRAM_DISABLE=${DSV41_ENGRAM_DISABLE:-0}"
        # Sparse-indexer prefill gather workspace = max_model_len x this (default:
        # MAX_NUM_SEQS; stock vLLM uses 40 = 2.6 GB at 500k). patch_sm120_block64.py.
        -e "DSV41_INDEXER_PREFILL_FACTOR=${DSV41_INDEXER_PREFILL_FACTOR:-}"
        # Peak bytes of one indexer logits launch (vLLM default 512): smaller = less live memory
        # per chunk at long prefixes, more launches (117 TFLOPS at 294 rows vs 151 at 1536).
        -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
        # torch.cuda.empty_cache() after each prefill chunk of sequences this long
        # (patch_memory_log.py): reserved memory otherwise grows with the square of
        # the prompt on this UMA box. 0 disables.
        -e "DSV41_PREFILL_EMPTY_CACHE_TOKENS=${DSV41_PREFILL_EMPTY_CACHE_TOKENS:-8192}"
        # ... but only while the node's MemAvailable after the chunk is below this
        # (GiB); with headroom the allocator keeps its blocks for the next chunk
        # (re-committing them every chunk cost ~20 % of prefill). 0 = always.
        -e "DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=${DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB:-2.5}"
        # One release when a long prefill hands over to decode (page cache for the Engram rows).
        -e "DSV41_PREFILL_END_EMPTY_CACHE=${DSV41_PREFILL_END_EMPTY_CACHE:-0}"
        -e "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-}"
        -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1
        -e VLLM_CACHE_ROOT=/root/.cache/vllm
        -e "GLM53_SUPPRESS_STOPS_IN_REASONING=$GLM53_SUPPRESS_STOPS_IN_REASONING"
        -e "GLM53_MIXED_PREFILL_CHUNK=$GLM53_MIXED_PREFILL_CHUNK"
        -e "GLM53_INDEXER_WORKSPACE=$GLM53_INDEXER_WORKSPACE"
        -e "GLM53_SPINWAIT_MS=$GLM53_SPINWAIT_MS"
        -e "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
        -e "TILELANG_CACHE_DIR=$TILELANG_CACHE_DIR"
        -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
        -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        -e "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        -e FLASHINFER_DISABLE_VERSION_CHECK=1
        -e "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
        -e "VLLM_ENGINE_READY_TIMEOUT_S=$READY_TIMEOUT"
        -e VLLM_NO_USAGE_STATS=1
        -e DO_NOT_TRACK=1
        -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$CG_ESTIMATE"
        -e "DSV41_EXL3_K_MAP=/opt/dsv41/exl3_k_map.json"
        -e GLM53_DENSE_FP8=off
    )
    local worker_nccl="" e
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        worker_nccl+=" -e $e"
    done

    local -a head_preload=() worker_preload=""
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
        if worker_ssh "test -f '$WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME'"; then
            worker_preload="-v '$WORKER_NCCL_HOST_DIR:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
            log "worker: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "worker: $WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    local serve_env=""
    local v
    for v in SERVED_MODEL_NAME PORT TP NNODES HEAD_IP MASTER_PORT QUANTIZATION \
             MAX_MODEL_LEN GPU_MEM_UTIL MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
             KV_CACHE_DTYPE SPEC_METHOD DSPARK_TOKENS \
             LANGUAGE_MODEL_ONLY SKIP_MM_PROFILING \
             LIMIT_MM CHAT_TEMPLATE ENFORCE_EAGER EXL3_FUSED_MOE EXL3_MOE_ROW_TILE \
             EXL3_TEMP_ROWS_FUSED EXL3_FAT_SORTED EXL3_FAT_BATCHED EXL3_FAT_KERNEL \
             EXL3_FAT_GROUPED MODEL_DIR ENGRAM_MOUNT EXTRA_ARGS \
             DSV41_CACHE_GIB DSV41_RESIDENT_SCALES DSV41_IO_THREADS \
             DSV41_CACHE_WAYS DSV41_STATS_SECONDS KV_CACHE_MEMORY_BYTES KV_BLOCK_SIZE \
             DSV41_SKIP_MIXED_WARMUP DSV41_DROP_PAGE_CACHE \
             DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB DSV41_PREFILL_END_EMPTY_CACHE VLLM_SPARSE_INDEXER_MAX_LOGITS_MB \
             LONG_PREFILL_TOKEN_THRESHOLD; do
        serve_env+=" -e $v='${!v:-}'"
    done
    serve_env+=" -e VLLM_API_KEY='${VLLM_API_KEY:-}'"
    serve_env+=" -e DSV41_SOURCE='${ENGRAM_MOUNT}'"

    local worker_packed_args=""
    if worker_ssh "[ -f '$WORKER_PACKED_DIR/engram-l1-r1of${TP}.bin' ]"; then
        worker_packed_args="-v '$WORKER_PACKED_DIR:$PACKED_MOUNT:ro' -e DSV41_PACKED_DIR='$PACKED_MOUNT'"
        log "worker packed Engram: $WORKER_PACKED_DIR -> $PACKED_MOUNT"
    fi
    local -a head_packed_args=()
    if [ -f "$HEAD_PACKED_DIR/engram-l1-r0of${TP}.bin" ]; then
        head_packed_args=(-v "$HEAD_PACKED_DIR:$PACKED_MOUNT:ro" -e "DSV41_PACKED_DIR=$PACKED_MOUNT")
        log "head packed Engram: $HEAD_PACKED_DIR -> $PACKED_MOUNT"
    else
        log "no TP=${TP} packed Engram at $HEAD_PACKED_DIR (optional ./start.sh pack)"
    fi

    log "starting worker on ${WORKER_SSH} (NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
    worker_ssh "docker run -d --name '$CONTAINER_WORKER' \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --oom-score-adj 1000 --cap-add SYS_PTRACE \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v '$WORKER_MODEL_BIND:/model:ro' \
        -v '$WORKER_ENGRAM_BIND:/engram-src:ro' \
        -v '$WORKER_VLLM_CACHE:/root/.cache/vllm' \
        -v '$WORKER_TRITON_CACHE:/root/.triton/cache' \
        -v '$WORKER_TILELANG_CACHE:/root/.tilelang/cache' \
        -v '/tmp/${CONTAINER_WORKER}.sh:/start.sh:ro' \
        -v '/tmp/dsv41-chat_template.jinja:${CHAT_TEMPLATE}:ro' \
        -v '/tmp/patch_suppress_stops_in_reasoning.py:/opt/dsv41/patch_suppress_stops_in_reasoning.py:ro' \
        -v '/tmp/patch_scheduler_decode_floor.py:/opt/dsv41/patch_scheduler_decode_floor.py:ro' \
        -v '/tmp/patch_xgrammar_termination.py:/opt/dsv41/patch_xgrammar_termination.py:ro' \
        -v '/tmp/patch_spinwait.py:/opt/dsv41/patch_spinwait.py:ro' \
        -v '/tmp/patch_responses_content_types.py:/opt/dsv41/patch_responses_content_types.py:ro' \
        -v '/tmp/dsv41-exl3.py:/opt/dsv41/exl3.py:ro' \
        -v '/tmp/exl3_k_map.json:/opt/dsv41/exl3_k_map.json:ro' \
        -v '/tmp/patch_exl3_packed_names.py:/opt/dsv41/patch_exl3_packed_names.py:ro' \
        -v '/tmp/patch_exl3_lm_head.py:/opt/dsv41/patch_exl3_lm_head.py:ro' \
        -v '/tmp/patch_engram_secondary.py:/opt/dsv41/patch_engram_secondary.py:ro' \
        -v '/tmp/patch_engram_file.py:/opt/dsv41/patch_engram_file.py:ro' \
        -v '/tmp/engram_file_backend.py:/opt/dsv41/engram_file_backend.py:ro' \
        -v '/tmp/engram_layout.py:/opt/dsv41/engram_layout.py:ro' \
        -v '/tmp/patch_memory_log.py:/opt/dsv41/patch_memory_log.py:ro' \
        -v '/tmp/patch_h2d_stage.py:/opt/dsv41/patch_h2d_stage.py:ro' \
        -v '/tmp/patch_sm120_block64.py:/opt/dsv41/patch_sm120_block64.py:ro' \
        ${worker_packed_args} \
        ${worker_preload} \
        ${worker_nccl} \
        -e NCCL_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e GLOO_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e NCCL_IB_HCA='$WORKER_CX7_IB' \
        -e NCCL_IB_GID_INDEX='$WORKER_GID' \
        -e VLLM_HOST_IP='$WORKER_IP' \
        ${serve_env} \
        --entrypoint bash '$IMAGE' /start.sh" >/dev/null

    log "starting head (vLLM API :${PORT}; NCCL if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
    docker run -d --name "$CONTAINER_HEAD" \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --oom-score-adj 1000 --cap-add SYS_PTRACE \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$MODEL_HOST:/model:ro" \
        -v "${HEAD_ENGRAM_BIND:-$ENGRAM_SRC}:/engram-src:ro" \
        -v "$CACHE_ROOT:/root/.cache/vllm" \
        -v "$TRITON_HOST_CACHE:/root/.triton/cache" \
        -v "$TILELANG_HOST_CACHE:/root/.tilelang/cache" \
        -v "$HEAD_SCRIPT:/start.sh:ro" \
        -v "$CHAT_TEMPLATE_HOST:$CHAT_TEMPLATE:ro" \
        -v "$STOP_PATCH_HOST:/opt/dsv41/patch_suppress_stops_in_reasoning.py:ro" \
        -v "$SCHED_PATCH_HOST:/opt/dsv41/patch_scheduler_decode_floor.py:ro" \
        -v "$XGRAMMAR_PATCH_HOST:/opt/dsv41/patch_xgrammar_termination.py:ro" \
        -v "$SPINWAIT_PATCH_HOST:/opt/dsv41/patch_spinwait.py:ro" \
        -v "$RESPONSES_PATCH_HOST:/opt/dsv41/patch_responses_content_types.py:ro" \
        -v "$EXL3_OVERLAY_HOST:/opt/dsv41/exl3.py:ro" \
        -v "$KMAP_HOST:/opt/dsv41/exl3_k_map.json:ro" \
        -v "$SCRIPT_DIR/overlay/patch_exl3_packed_names.py:/opt/dsv41/patch_exl3_packed_names.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_exl3_lm_head.py:/opt/dsv41/patch_exl3_lm_head.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_engram_secondary.py:/opt/dsv41/patch_engram_secondary.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_engram_file.py:/opt/dsv41/patch_engram_file.py:ro" \
        -v "$SCRIPT_DIR/overlay/engram_file_backend.py:/opt/dsv41/engram_file_backend.py:ro" \
        -v "$SCRIPT_DIR/overlay/engram_layout.py:/opt/dsv41/engram_layout.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_memory_log.py:/opt/dsv41/patch_memory_log.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_h2d_stage.py:/opt/dsv41/patch_h2d_stage.py:ro" \
        -v "$SCRIPT_DIR/overlay/patch_sm120_block64.py:/opt/dsv41/patch_sm120_block64.py:ro" \
        "${head_packed_args[@]}" \
        "${head_preload[@]}" \
        "${nccl_common[@]}" \
        -e NCCL_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e GLOO_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e NCCL_IB_HCA="$HEAD_CX7_IB" \
        -e NCCL_IB_GID_INDEX="$HEAD_GID" \
        -e VLLM_HOST_IP="$HEAD_IP" \
        -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
        -e PORT="$PORT" -e TP="$TP" -e NNODES="$NNODES" \
        -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="$MASTER_PORT" \
        -e QUANTIZATION="$QUANTIZATION" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        -e MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
        -e LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-}" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" \
        -e SPEC_METHOD="$SPEC_METHOD" \
        -e DSPARK_TOKENS="${DSPARK_TOKENS:-5}" \
        -e LANGUAGE_MODEL_ONLY="$LANGUAGE_MODEL_ONLY" \
        -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e LIMIT_MM="$LIMIT_MM" \
        -e CHAT_TEMPLATE="$CHAT_TEMPLATE" \
        -e ENFORCE_EAGER="$ENFORCE_EAGER" \
        -e EXL3_FUSED_MOE="$EXL3_FUSED_MOE" \
        -e EXL3_MOE_ROW_TILE="$EXL3_MOE_ROW_TILE" \
        -e EXL3_TEMP_ROWS_FUSED="$EXL3_TEMP_ROWS_FUSED" \
        -e EXL3_FAT_SORTED="$EXL3_FAT_SORTED" \
        -e EXL3_FAT_BATCHED="$EXL3_FAT_BATCHED" \
        -e EXL3_FAT_KERNEL="$EXL3_FAT_KERNEL" \
        -e EXL3_FAT_GROUPED="$EXL3_FAT_GROUPED" \
        -e MODEL_DIR="$MODEL_DIR" \
        -e ENGRAM_MOUNT="$ENGRAM_MOUNT" \
        -e DSV41_SOURCE="$ENGRAM_MOUNT" \
        -e DSV41_CACHE_GIB="$DSV41_CACHE_GIB" \
        -e DSV41_RESIDENT_SCALES="$DSV41_RESIDENT_SCALES" \
        -e DSV41_IO_THREADS="$DSV41_IO_THREADS" \
        -e DSV41_CACHE_WAYS="$DSV41_CACHE_WAYS" \
        -e DSV41_STATS_SECONDS="$DSV41_STATS_SECONDS" \
        -e KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}" \
        -e KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-}" \
        -e DSV41_SKIP_MIXED_WARMUP="${DSV41_SKIP_MIXED_WARMUP:-0}" \
        -e DSV41_DROP_PAGE_CACHE="${DSV41_DROP_PAGE_CACHE:-1}" \
        -e VLLM_API_KEY="$VLLM_API_KEY" \
        -e EXTRA_ARGS="${EXTRA_ARGS:-}" \
        --entrypoint bash "$IMAGE" /start.sh >/dev/null

    log "containers up — head=${CONTAINER_HEAD}, worker=${CONTAINER_WORKER}"
    start_memguards
}

# ------------------------------ hang dump ----------------------------------
dump_hang_stacks() {
    mkdir -p "$LOGDIR"
    docker exec "$CONTAINER_HEAD" bash /opt/dsv41/pyspy_dump.sh >"$LOGDIR/hang-head-pyspy.txt" 2>&1 || true
    worker_ssh "docker exec '$CONTAINER_WORKER' bash /opt/dsv41/pyspy_dump.sh" >"$LOGDIR/hang-worker-pyspy.txt" 2>&1 || true
    warn "stacks: $LOGDIR/hang-head-pyspy.txt, $LOGDIR/hang-worker-pyspy.txt"
}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (weight load + warmup on V4.1-Flash EXL3 + Engram is slow; timeout ${READY_TIMEOUT}s) ..."
    log "streaming head logs live — Ctrl-C detaches, the server keeps running"

    local logpid=""
    _stop_logtail() {
        [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true
        wait "$logpid" 2>/dev/null || true
        logpid=""
    }
    trap '_stop_logtail; warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT
    docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 &
    logpid=$!

    local elapsed=0 healthy=0 exited=0 dead_side="" worker_fail=0
    local last_lines=0 quiet=0 cur_lines
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        # Hang detector: a rank spinning on a hung kernel logs nothing.
        # (the EngineCore prints a "No available shared memory broadcast block"
        # line every 60 s while a rank is stuck, so those do not count)
        cur_lines=$(docker logs "$CONTAINER_HEAD" 2>&1 | grep -avc "shm_broadcast.py" || true)
        if [ "$cur_lines" -eq "$last_lines" ]; then
            quiet=$((quiet + 10))
        else
            quiet=0; last_lines=$cur_lines
        fi
        if [ "$quiet" -ge "$DSV41_HANG_SECONDS" ]; then
            warn "no head log line for ${quiet}s — dumping Python stacks (py-spy) and giving up"
            dump_hang_stacks
            exited=1; dead_side="hang"; DSV41_LAST_DEAD_SIDE=hang; break
        fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            log "head container exited during startup"
            exited=1; dead_side="head"; break
        fi
        # A dead worker rank can never make the head healthy — fail fast with
        # the log dump instead of polling for the full READY_TIMEOUT (issue
        # #22, item 4). Transient ssh/docker hiccups are tolerated; only
        # three consecutive non-running answers (~30 s) count as a dead
        # worker.
        if worker_ssh "docker inspect -f '{{.State.Running}}' '$CONTAINER_WORKER' 2>/dev/null" | grep -q true; then
            worker_fail=0
        else
            worker_fail=$((worker_fail + 1))
            if [ "$worker_fail" -ge 3 ]; then
                log "worker container '$CONTAINER_WORKER' not running on ${WORKER_SSH} (3 consecutive checks)"
                exited=1; dead_side="worker"; break
            fi
        fi
        if [ $((elapsed % 60)) -eq 0 ]; then
            log "t+${elapsed}s MemAvailable head=$(mem_avail_gib) GiB worker=$(worker_mem_avail_gib) GiB"
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    _stop_logtail
    trap 'warn "interrupted — containers keep running ('"'"'./start.sh logs'"'"' / '"'"'./start.sh stop'"'"')"; exit 130' INT

    if [ "$healthy" = "1" ]; then
        log "health check passed after ${elapsed}s — server is up"
    elif [ "$exited" = "1" ]; then
        warn "${dead_side:-head} container exited/stopped after ${elapsed}s"
    else
        warn "timed out after ${elapsed}s without becoming healthy"
    fi
    [ "$healthy" = "1" ]
}

collect_failure_logs() {
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head.log" 2>&1 || true
    worker_ssh "docker logs '$CONTAINER_WORKER' 2>&1" >"$LOGDIR/worker.log" 2>&1 || true
    worker_ssh "cat /tmp/dsv41-memguard-worker.log 2>/dev/null" >"$LOGDIR/memguard-worker.log" 2>/dev/null || true
    if [ -s "$LOGDIR/hang-head-pyspy.txt" ] && [ "${DSV41_LAST_DEAD_SIDE:-}" = "hang" ]; then
        warn "the ranks stopped making progress (hang): Python/native stacks in $LOGDIR/hang-head-pyspy.txt and hang-worker-pyspy.txt"
        grep -m1 -A3 '"MainThread"' "$LOGDIR/hang-head-pyspy.txt" >&2 || true
    elif grep -q KILLING "$LOGDIR/memguard-head.log" "$LOGDIR/memguard-worker.log" 2>/dev/null; then
        warn "a memory guard killed a container — the node ran out of unified memory:"
        grep -h -E 'LOW|KILLING' "$LOGDIR/memguard-head.log" "$LOGDIR/memguard-worker.log" 2>/dev/null | tail -n 6 >&2 || true
    fi
}

# ------------------------------- stop --------------------------------------
stop() {
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    log "stopping worker container on ${WORKER_SSH} ..."
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 \
        || log "  (no worker container was running)"
    stop_memguards
    log "stopped."
}

# ------------------------------ status -------------------------------------
status() {
    log "head (${CONTAINER_HEAD} on $(hostname)):"
    docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "  API: not responding"
    fi
    log "worker (${CONTAINER_WORKER} on ${WORKER_SSH}):"
    worker_ssh "docker ps -a --filter name=${CONTAINER_WORKER} --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
        || log "  (worker unreachable)"
}

# ------------------------------- logs --------------------------------------
logs() {
    case "${1:-head}" in
        worker)
            log "following worker container logs on ${WORKER_SSH} ..."
            trap '' INT
            worker_ssh "docker logs -f --tail 100 '$CONTAINER_WORKER'" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
        head|*)
            log "following head logs (driver + API server) ..."
            trap '' INT
            docker logs -f --tail 100 "$CONTAINER_HEAD" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
    esac
}

post_ready_warmup() {
    if [ "${DSV41_BOOT_SHAPE_WARMUP:-1}" = "0" ]; then
        log "boot shape warmup skipped (DSV41_BOOT_SHAPE_WARMUP=0)"
        return 0
    fi
    [ -f "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" ] \
        || { warn "boot-shape-warmup.sh missing — skipping"; return 0; }
    log "post-ready DSpark/sampler warmup (nonfatal; timeout ${DSV41_WARMUP_REQ_TIMEOUT}s/req) ..."
    GLM53_WARMUP_MAX_CONCURRENCY="$MAX_NUM_SEQS" \
    GLM53_WARMUP_REQ_TIMEOUT="$DSV41_WARMUP_REQ_TIMEOUT" \
    GLM53_WARMUP_DSPARK_K="${DSPARK_TOKENS:-5}" \
    GLM53_WARMUP_TRITON_CACHE_DIR="$TRITON_HOST_CACHE" \
    GLM53_WARMUP_BEARER="${VLLM_API_KEY:-}" \
        bash "$SCRIPT_DIR/scripts/boot-shape-warmup.sh" \
            "http://127.0.0.1:${PORT}" "$SERVED_MODEL_NAME" \
        || warn "boot shape warmup incomplete — uncovered shapes may JIT mid-serve on TP=2"
}

on_ready() {
    log "======================================================================"
    log "DeepSeek-V4.1-Flash EXL3 is UP (TP=${TP}, nnodes=${NNODES})"
    log "  endpoints  : http://127.0.0.1:${PORT}/v1   (LAN: ${HEAD_IP}:${PORT})"
    log "  model name : ${SERVED_MODEL_NAME}"
    log "  weights    : ${MODEL_HOST}  quant=${QUANTIZATION}  kv=${KV_CACHE_DTYPE:-auto (fp8_ds_mla)}  sync=${WEIGHT_SYNC}/${WEIGHT_BACKEND:-}"
    log "  engram     : file-backed ${ENGRAM_SRC} (no 47GiB pin; packed=${HEAD_PACKED_DIR}) worker=${WORKER_ENGRAM_BIND}"
    local vision=on
    [ "${LANGUAGE_MODEL_ONLY}" = "1" ] && vision=off
    local spec="DSpark k=${DSPARK_TOKENS} (in-checkpoint)"
    [ "$SPEC_METHOD" = "none" ] && spec=off
    log "  features   : tokenizer=deepseek_v41 tools=deepseek_v41 reasoning=deepseek_v41 spec=${spec} vision=${vision}"
    local auth_line="none (VLLM_API_KEY empty)"
    if [ -n "${VLLM_API_KEY:-}" ]; then
        auth_line="bearer token set (VLLM_API_KEY) — send Authorization: Bearer <key> on /v1 requests"
    fi
    log "  auth       : ${auth_line}"
    log "  sampling   : temperature=1.0 top_p=0.95 (official). Thinking defaults ON."
    log "  quick test :"
    log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
    if [ -n "${VLLM_API_KEY:-}" ]; then
        log "      -H 'Authorization: Bearer <KEY>' \\"
    fi
    log "      -H 'Content-Type: application/json' \\"
    log "      -d '{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"17*19=\"}],\"max_tokens\":32,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}'"
    if [ "${DSV41_MEM_GUARD:-0}" = "1" ]; then
        log "  memory     : head MemAvailable=$(mem_avail_gib) GiB worker=$(worker_mem_avail_gib) GiB; guards kill a node's container under ${DSV41_MEM_GUARD_GIB} GiB ($LOGDIR/memguard-head.log)"
    else
        log "  memory     : head MemAvailable=$(mem_avail_gib) GiB worker=$(worker_mem_avail_gib) GiB; guards disabled"
    fi
    log "  manage     : ./start.sh status | ./start.sh logs | ./start.sh logs worker | ./start.sh stop"
    log "======================================================================"
    if [ "${TAIL:-0}" = "1" ]; then
        log "tailing head logs — Ctrl-C just detaches, the server keeps running"
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER_HEAD" || true
        trap 'warn "interrupted — containers keep running"; exit 130' INT
    fi
}

start() {
    preflight
    ensure_image
    check_weights
    sync_weights
    write_inner_scripts

    log "model load path (in-container): ${MODEL_DIR}"
    log "engram path (in-container): ${ENGRAM_MOUNT}  host=${HEAD_ENGRAM_BIND:-$ENGRAM_SRC}  worker=${WORKER_ENGRAM_BIND}"
    log "config: image=${IMAGE} tp=${TP} nnodes=${NNODES} quant=${QUANTIZATION} spec=${SPEC_METHOD} dspark_k=${DSPARK_TOKENS} max-len=${MAX_MODEL_LEN} mnbt=${MAX_NUM_BATCHED_TOKENS} gpu-util=${GPU_MEM_UTIL} kv=${KV_CACHE_DTYPE:-native} kv-pool=${KV_CACHE_MEMORY_BYTES:-profiled} kv-block=${KV_BLOCK_SIZE:-auto} lm-only=${LANGUAGE_MODEL_ONLY} port=${PORT}"
    log "memory: engram cache=${DSV41_CACHE_GIB}GiB resident_scales=${DSV41_RESIDENT_SCALES} io_threads=${DSV41_IO_THREADS} guard=$([ "${DSV41_MEM_GUARD:-0}" = "1" ] && echo "<${DSV41_MEM_GUARD_GIB}GiB" || echo off) boot-margin=${DSV41_BOOT_MARGIN_GIB}GiB"
    log "exl3: fused=${EXL3_FUSED_MOE} fat_kernel=${EXL3_FAT_KERNEL} fat_grouped=${EXL3_FAT_GROUPED} temp_rows=${EXL3_TEMP_ROWS_FUSED} (E3 v2 kernels: K2/K3/K4 mul1 + K4 mcg; E2 stays K4/MCG-only)"

    launch_cluster
    if wait_for_health; then
        post_ready_warmup
        on_ready
        return
    fi
    collect_failure_logs
    echo "---- last 60 lines of head log ($LOGDIR/head.log) ----"
    tail -n 60 "$LOGDIR/head.log" || true
    echo "---- last 40 lines of worker log ($LOGDIR/worker.log) ----"
    tail -n 40 "$LOGDIR/worker.log" || true
    die "server did not become healthy — full logs in $LOGDIR/"
}

main() {
    local cmd="${1:-start}"
    case "$cmd" in
        start|restart) validate_numeric_config ;;
    esac
    case "$cmd" in
        stop)     banner stop.sh ;;
        pack)     banner pack.sh ;;
        share)    banner share.sh ;;
        *)        banner start.sh ;;
    esac
    case "$cmd" in
        start)    shift || true; start ;;
        share)    share_weights ;;
        pack)     pack_engram ;;
        stop)     stop ;;
        restart)  stop; start ;;
        status)   status ;;
        logs)     shift || true; logs "$@" ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
