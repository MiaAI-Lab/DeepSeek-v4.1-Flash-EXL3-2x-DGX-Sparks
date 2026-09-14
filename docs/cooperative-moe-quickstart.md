# Test the cooperative MoE extension on two Sparks

This is the complete manual opt-in procedure for an **already working DS4.1 TP2
installation**. Use a dedicated Bash shell on the head Spark, with this PR checked
out, the checkpoint already installed, and passwordless SSH/Docker access to the
configured worker. The measured configuration is text-only, 600K maximum context,
two requests, and DSpark k=3. Vision is outside this validation scope.

**Activation is a selected overlay, not a standalone environment toggle.**
`DSV41_COOPERATIVE_MOE=1 ./start.sh start` by itself does not import the adapter or
deploy its binary. The sequence below builds and verifies artifacts, copies them
to both nodes, then selects the generated overlay in `.env`.

Read the rollback section before starting. Steps 2–7 require a maintenance window;
the stop command cancels any remaining DS4.1 requests. Do not run these GPU tests
alongside another model or active user workloads. A failed check is a reason to
roll back, not to disable the integrity checks.

## 1. Read the existing configuration and save a rollback copy

Run from the recipe root. Only source your own trusted `.env`; do not print or
commit its contents. These assignments follow the launcher's cache/SSH defaults.
They do not change the configuration or live service.

```bash
set -euo pipefail
COOP_REPO="$(pwd -P)"
test -f "$COOP_REPO/start.sh"
test -f "$COOP_REPO/.env"
test -f "$COOP_REPO/extensions/cooperative_moe/build.sh"
SCRIPT_DIR="$COOP_REPO"
source "$COOP_REPO/.env"

COOP_WORKER_USER="${WORKER_USER:-$USER}"
COOP_WORKER="${WORKER_SSH:-${COOP_WORKER_USER}@${WORKER_IP:-10.0.0.2}}"
if [ "$COOP_WORKER_USER" = "$USER" ]; then
  COOP_WORKER_HOME="${WORKER_HOME:-$HOME}"
else
  COOP_WORKER_HOME="${WORKER_HOME:-/home/${COOP_WORKER_USER}}"
fi
COOP_HEAD_CACHE="${CACHE_ROOT:-$HOME/.cache/vllm-dsv41-flash-exl3}"
COOP_WORKER_CACHE="${WORKER_VLLM_CACHE:-$COOP_WORKER_HOME/.cache/vllm-dsv41-flash-exl3}"
COOP_HEAD_CONTAINER="${CONTAINER_HEAD:-dsv41-exl3-head}"
COOP_WORKER_CONTAINER="${CONTAINER_WORKER:-dsv41-exl3-worker}"
COOP_PORT="${PORT:-8888}"

mkdir -p "$COOP_REPO/logs"
COOP_RUN="$(mktemp -d "$COOP_REPO/logs/cooperative-moe.XXXXXX")"
COOP_SLOT="${COOP_RUN##*/}"
COOP_HEAD_STAGE="$COOP_HEAD_CACHE/$COOP_SLOT"
COOP_WORKER_STAGE="$COOP_WORKER_CACHE/$COOP_SLOT"
COOP_RUNTIME="/root/.cache/vllm/$COOP_SLOT"
install -m 600 "$COOP_REPO/.env" "$COOP_RUN/original.env"
printf 'Rollback configuration: %s\n' "$COOP_RUN/original.env"
```

The versioned staging directory avoids overwriting artifacts from an existing
profile. Both cache roots must be absolute paths matching `CACHE_ROOT` and
`WORKER_VLLM_CACHE` in the launcher. The SSH commands below assume normal Unix
cache paths without embedded single quotes.

## 2. Drain requests, stop DS4.1, and obtain the pinned build image

Stop clients from submitting new requests before continuing. These commands are
shown for an operator to run deliberately; generating a profile does not run them.

```bash
./start.sh stop
COOP_IMAGE='ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce'
docker pull "$COOP_IMAGE"
ssh -o BatchMode=yes "$COOP_WORKER" "docker pull '$COOP_IMAGE'"
docker image inspect --format '{{.Id}}' "$COOP_IMAGE"
ssh -o BatchMode=yes "$COOP_WORKER" "docker image inspect --format '{{.Id}}' '$COOP_IMAGE'"
```

Both image IDs must be
`sha256:4cdba4e946da2d19bf5b5a20c6d3a1a4bf421fa4d6db5082f271a986168176cb`.
The registry digest above resolves to this image; unlike a local image ID it can
be pulled by another operator. Do not substitute a mutable `latest` tag and
assume that it reproduces the validated artifact.

## 3. Build in the pinned image

The build is CPU-only and bounded to two CPUs/2 GiB. It uses `/work` as its output
mount to preserve the original compiler input paths. No GPU is assigned to it.

```bash
git clone --filter=blob:none --no-checkout https://github.com/turboderp-org/exllamav3.git "$COOP_RUN/exllamav3"
git -C "$COOP_RUN/exllamav3" checkout --detach 02aef45cd681b960a00afcd0749a4ab99e6c1bfe
mkdir "$COOP_RUN/build"
docker run --rm --network none --cpus 2 --memory 2g --memory-swap 2g \
  --user "$(id -u):$(id -g)" \
  -v "$COOP_REPO/extensions/cooperative_moe:/src:ro" \
  -v "$COOP_RUN/exllamav3:/upstream:ro" \
  -v "$COOP_RUN/build:/work" \
  --entrypoint bash "$COOP_IMAGE" /src/build.sh /upstream /work
```

The expected `cooperative_moe.so` digest is
`a09a589cbdcecb5372991c7b091d732236d58bc5f5aea14ab91e38e426f08d78`.
`prepare_profile.py` enforces this and the stock/runtime source digests. If a
clean rebuild differs, preserve the build log and stop: the current release gate
is intentionally closed to unvalidated artifacts. Do not change the digest just
to make the helper accept a build.

## 4. Generate the overlay and deploy identical artifacts to both nodes

The launcher copies the selected overlay to the worker, but **does not copy the
native library or adapter**. The explicit copying here is required. The launcher
mounts each host's cache at `/root/.cache/vllm`, so the generated container path is
the same on both ranks even if their host cache paths differ.

```bash
mkdir -p "$COOP_HEAD_STAGE"
install -m 644 "$COOP_RUN/build/cooperative_moe.so" "$COOP_RUN/build/runtime.py" "$COOP_HEAD_STAGE/"
python3 extensions/cooperative_moe/prepare_profile.py \
  --stock overlay/exl3.py \
  --artifacts "$COOP_HEAD_STAGE" \
  --runtime-directory "$COOP_RUNTIME" \
  --output "$COOP_HEAD_STAGE/exl3-cooperative.py"
install -m 644 extensions/cooperative_moe/test_cuda_integration.py tests/test_exl3_overlay.py "$COOP_HEAD_STAGE/"

ssh -o BatchMode=yes "$COOP_WORKER" "mkdir -p '$COOP_WORKER_STAGE'"
scp "$COOP_HEAD_STAGE/cooperative_moe.so" "$COOP_HEAD_STAGE/runtime.py" \
  "$COOP_HEAD_STAGE/exl3-cooperative.py" "$COOP_HEAD_STAGE/test_cuda_integration.py" \
  "$COOP_HEAD_STAGE/test_exl3_overlay.py" "$COOP_WORKER:$COOP_WORKER_STAGE/"
(cd "$COOP_HEAD_STAGE" && sha256sum cooperative_moe.so runtime.py exl3-cooperative.py test_cuda_integration.py test_exl3_overlay.py) > "$COOP_RUN/SHA256SUMS"
scp "$COOP_RUN/SHA256SUMS" "$COOP_WORKER:$COOP_WORKER_STAGE/SHA256SUMS"
ssh -o BatchMode=yes "$COOP_WORKER" "cd '$COOP_WORKER_STAGE' && sha256sum -c SHA256SUMS"
```

## 5. Run the packaged GPU gate on each node

DS4.1 and any other GPU workloads must still be stopped. These containers do not
load the full checkpoint; they run the synthetic integration fixtures. Require
the final 54-case `status: pass` record and a zero process exit code on each node.
The gate reports its peak-normalized numerical criterion and retained strict
differences; a pass is not a claim of bitwise equality to stock.

```bash
docker run --rm --network none --gpus all --cpus 2 --memory 6g --memory-swap 6g \
  -e DSV41_COOP_MAINTENANCE_TEST=1 -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$COOP_HEAD_CACHE:/root/.cache/vllm" \
  -v "$COOP_HEAD_STAGE/exl3-cooperative.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro" \
  -v "$COOP_HEAD_STAGE/test_exl3_overlay.py:/opt/dsv41/test_exl3_overlay.py:ro" \
  -v "$COOP_HEAD_STAGE/test_cuda_integration.py:/opt/dsv41/test_cuda_integration.py:ro" \
  --entrypoint python3 "$COOP_IMAGE" /opt/dsv41/test_cuda_integration.py \
  2>&1 | tee "$COOP_RUN/cuda-head.log"

ssh -o BatchMode=yes "$COOP_WORKER" "docker run --rm --network none --gpus all --cpus 2 --memory 6g --memory-swap 6g \
  -e DSV41_COOP_MAINTENANCE_TEST=1 -e MAX_JOBS=2 -e OMP_NUM_THREADS=1 \
  -e OPENBLAS_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v '$COOP_WORKER_CACHE:/root/.cache/vllm' \
  -v '$COOP_WORKER_STAGE/exl3-cooperative.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py:ro' \
  -v '$COOP_WORKER_STAGE/test_exl3_overlay.py:/opt/dsv41/test_exl3_overlay.py:ro' \
  -v '$COOP_WORKER_STAGE/test_cuda_integration.py:/opt/dsv41/test_cuda_integration.py:ro' \
  --entrypoint python3 '$COOP_IMAGE' /opt/dsv41/test_cuda_integration.py" \
  2>&1 | tee "$COOP_RUN/cuda-worker.log"
```

## 6. Select the profile in `.env` and start both nodes

Print the generated overlay path:

```bash
printf 'EXL3_OVERLAY_HOST=%s\n' "$COOP_HEAD_STAGE/exl3-cooperative.py"
```

Edit `.env`: **replace** any existing assignments below rather than leaving
conflicting entries. Use the absolute overlay path printed above. These values
select the validated lane; retain the rest of your working topology and memory
configuration. Do not blindly copy another cluster's `.env`.

```dotenv
EXL3_OVERLAY_HOST=/absolute/path/printed/above/exl3-cooperative.py
DSV41_EXL3_SERIAL_STREAMS=1
VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
EXL3_FUSED_MOE=1
EXL3_TEMP_ROWS_FUSED=8
EXL3_FAT_KERNEL=0
EXL3_FAT_GROUPED=1
SPEC_METHOD=dspark
DSPARK_TOKENS=3
MAX_MODEL_LEN=600000
MAX_NUM_SEQS=2
LANGUAGE_MODEL_ONLY=1
```

`EXL3_OVERLAY_HOST` must be set in `.env` when an older override already exists
there: the launcher sources `.env` after reading the process environment, and
does not preserve a command-line overlay override. The generated overlay calls
`install(..., enabled=True)`; setting the standalone enablement variable is not
an additional activation step.

```bash
IMAGE="$COOP_IMAGE" SPEC_METHOD=dspark DSPARK_TOKENS=3 \
  MAX_MODEL_LEN=600000 MAX_NUM_SEQS=2 LANGUAGE_MODEL_ONLY=1 \
  EXL3_FUSED_MOE=1 EXL3_TEMP_ROWS_FUSED=8 EXL3_FAT_KERNEL=0 EXL3_FAT_GROUPED=1 \
  SKIP_PULL=1 SKIP_BUILD=1 SKIP_SHIP=1 SKIP_SYNC=1 \
  ./start.sh start 2>&1 | tee "$COOP_RUN/start.log"
```

This command uses the image already verified/pulled on both nodes. `SKIP_SYNC=1`
assumes the pre-existing weight/Engram setup still works; this is not an initial
model-installation procedure. For matching the published benchmark settings,
also use the documented 3072-token batch budget and 2816-token long-prefill
threshold. Those are performance-comparison settings, not activation switches.

For later starts, keep the verified image reference in `.env` as `IMAGE` if this
profile remains selected. Otherwise an ordinary restart could choose a different
image than the one just tested. Review/revalidate on recipe or image upgrades.

## 7. Verify activation, then send a short request

Wait for the launcher to finish all warmups, not just the first HTTP health
response. Check both nodes for the explicit activation message. The mounted and
installed overlay digests must match the generated host file.

```bash
sha256sum "$COOP_HEAD_STAGE/exl3-cooperative.py"
docker logs "$COOP_HEAD_CONTAINER" 2>&1 | grep -F 'Fixed-shape cooperative MoE enabled'
ssh -o BatchMode=yes "$COOP_WORKER" "docker logs '$COOP_WORKER_CONTAINER' 2>&1 | grep -F 'Fixed-shape cooperative MoE enabled'"
docker exec "$COOP_HEAD_CONTAINER" sha256sum /opt/dsv41/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
ssh -o BatchMode=yes "$COOP_WORKER" "docker exec '$COOP_WORKER_CONTAINER' sha256sum /opt/dsv41/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py"
curl --max-time 10 -fsS "http://127.0.0.1:$COOP_PORT/health"

COOP_AUTH=()
if [ -n "${VLLM_API_KEY:-}" ]; then
  COOP_AUTH=(-H "Authorization: Bearer $VLLM_API_KEY")
fi
curl --max-time 30 -fsS "${COOP_AUTH[@]}" \
  "http://127.0.0.1:$COOP_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-v4.1-Flash-EXL3","messages":[{"role":"user","content":"What is 17 times 19? Reply with the number only."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Use your configured `SERVED_MODEL_NAME` in the JSON if it differs from the
default. Expect the final answer `323` and `finish_reason: stop`. The activation
log establishes that the adapter was installed; the packaged GPU test above
checks native selection and fallback, while a health response alone does not.
Only proceed to bounded C1/C2 tests if the gates and startup pass. Benchmark
prompts, metric definitions and caveats are in [the report](cooperative-moe.md).

## 8. Roll back

Drain clients again. Restore the complete saved configuration, not just one knob:

```bash
./start.sh stop
install -m 600 "$COOP_RUN/original.env" "$COOP_REPO/.env"
env -u IMAGE -u SPEC_METHOD -u DSPARK_TOKENS -u MAX_MODEL_LEN -u MAX_NUM_SEQS \
  -u MAX_NUM_BATCHED_TOKENS -u LONG_PREFILL_TOKEN_THRESHOLD \
  -u LANGUAGE_MODEL_ONLY -u EXL3_FUSED_MOE -u EXL3_TEMP_ROWS_FUSED \
  -u EXL3_FAT_KERNEL -u EXL3_FAT_GROUPED -u EXL3_OVERLAY_HOST \
  ./start.sh start
```

This returns to the previous configuration, which is stock only if you started
from stock. Clearing possible shell overrides prevents them from masking the
restored `.env`. If the shell was closed, use the absolute backup path printed in
step 1. Versioned artifacts can remain for inspection; their presence does not
enable the feature. No model weights, caches or rollback files need deletion.

## Validation status of this procedure

The launcher wiring, immutable image manifest, command syntax, paths, source pins
and host-side tests have been checked. The original implementation has the GPU
and serving evidence in the report. This complete clean-build/two-node runbook
has **not yet been executed end to end** against the final package; do not treat
the commands or a draft PR as a substitute for recording those results.
