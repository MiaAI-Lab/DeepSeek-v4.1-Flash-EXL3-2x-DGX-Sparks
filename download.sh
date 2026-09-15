#!/usr/bin/env bash
# download.sh — fetch the weights this recipe serves, without booting it.
#
# EXL3 weights (39 shards, ~197 GiB) come from HF_MODEL_REPO into ./model.
# Engram tables are never quantized and never copied into the EXL3 tree: only
# shards 47+48 of the original 48-shard checkpoint (~95 GiB each), the index,
# and config.json are pulled from HF_ENGRAM_REPO into ENGRAM_DIR.
#
# Resumable — re-run after an interruption. ./start.sh runs the same fetch
# automatically, so this script is only for staging the download separately.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
[ -f "$SCRIPT_DIR/.env" ] || cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a

MODEL_HOST="${MODEL_HOST:-$SCRIPT_DIR/model}"
ENGRAM_DIR="${ENGRAM_DIR:-$SCRIPT_DIR/engram-src}"
HF_MODEL_REPO="${HF_MODEL_REPO:-Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw}"
HF_ENGRAM_REPO="${HF_ENGRAM_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-39}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# With `set -euo pipefail`, probing a missing destination with find(1) exits
# before the first download. A clean checkout has neither directory yet.
mkdir -p "$MODEL_HOST" "$ENGRAM_DIR"

hf_cli() {
    if command -v hf >/dev/null 2>&1; then hf "$@"
    elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli "$@"
    else
        echo "need the Hugging Face CLI: pip install -U 'huggingface_hub[hf_transfer]'" >&2
        exit 1
    fi
}

have=$(find "$MODEL_HOST" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')
echo "EXL3    $MODEL_HOST  ${have:-0}/$EXPECTED_SHARDS shards"
if [ "${have:-0}" -lt "$EXPECTED_SHARDS" ] || [ ! -f "$MODEL_HOST/config.json" ]; then
    echo "fetching $HF_MODEL_REPO -> $MODEL_HOST"
    mkdir -p "$MODEL_HOST"
    hf_cli download "$HF_MODEL_REPO" --local-dir "$MODEL_HOST" --max-workers "${HF_MAX_WORKERS:-8}"
fi

ENGRAM_FILES=(
    "model-00047-of-00048.safetensors"
    "model-00048-of-00048.safetensors"
    "model.safetensors.index.json"
    "config.json"
)
missing=()
for f in "${ENGRAM_FILES[@]}"; do
    [ -f "$ENGRAM_DIR/$f" ] || missing+=("$f")
done
echo "Engram  $ENGRAM_DIR  $(( ${#ENGRAM_FILES[@]} - ${#missing[@]} ))/${#ENGRAM_FILES[@]} files"
if [ "${#missing[@]}" -gt 0 ]; then
    echo "fetching $HF_ENGRAM_REPO (shards 47+48 + index/config only) -> $ENGRAM_DIR"
    mkdir -p "$ENGRAM_DIR"
    inc=()
    for f in "${ENGRAM_FILES[@]}"; do inc+=(--include "$f"); done
    hf_cli download "$HF_ENGRAM_REPO" "${inc[@]}" --local-dir "$ENGRAM_DIR" --max-workers "${HF_MAX_WORKERS:-8}"
fi

have=$(find "$MODEL_HOST" -maxdepth 1 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d '[:space:]')
[ "${have:-0}" -ge "$EXPECTED_SHARDS" ] || { echo "still ${have:-0}/$EXPECTED_SHARDS EXL3 shards" >&2; exit 1; }
for f in "${ENGRAM_FILES[@]}"; do
    [ -f "$ENGRAM_DIR/$f" ] || { echo "missing $ENGRAM_DIR/$f" >&2; exit 1; }
done
echo "complete. ./start.sh will NFS-share both trees to the worker (WEIGHT_SYNC=nfs)."
