# Experimental native vision on two DGX Sparks

This is an **opt-in experimental variant** of the existing EXL3 recipe. The
standard Dockerfile, published image, checkpoint, text-only settings, and
network setup are unchanged. It is not an official DeepSeek or FlashInfer
certification.

## Why the text-only restriction can be lifted

The base image's FlashInfer implementation rejects the 1152-wide image-attention
shape on SM12x. FlashInfer [PR #4802](https://github.com/flashinfer-ai/flashinfer/pull/4802)
expands both decode and prefill envelopes. This variant pins its merged commit
`453aa7c7296e9ec711fd4c1f3aa6ee061a6b69dc`, including its SM121 prefill fix.

The existing `patch_sm120_block64.py` also deliberately forces a text-only SWA
window. `patch_sm12x_vision.py` retains that behavior unless
`DSV41_SM12X_VISION=1`. The new launch checks reject a missing build receipt,
unsupported kernel envelope, text-only/vision flag conflict, or undersized
image budget. FlashInfer's version string alone is not sufficient provenance:
the pinned source still reports `0.6.18`. The receipt is a compatibility marker,
not authentication or protection against privileged image modification.

## Build separately; do not modify a serving container

Build on an idle ARM64 host with CUDA tooling provided by the base image. The
build requires network access, CPU/RAM headroom, and no GPU. Do not compile next
to a nearly full live model. The script verifies archive hashes, preserves the
base torch/vLLM/TVM versions, removes stale FlashInfer cache packages, and builds
only the sparse-MLA SM120/121 module for `12.1a` with one compiler job.

```bash
docker build -f Dockerfile.vision -t dsv41-exl3:vision-experimental .
```

Install the same built image on both nodes using your existing image-transfer
procedure. The original image remains the rollback target. Existing model and
Engram staging is still required; this change does not alter weight downloads.

During an explicitly reserved test window, with competing inference stopped,
use the experimental image and explicit launch overrides:

```bash
IMAGE=dsv41-exl3:vision-experimental \
LANGUAGE_MODEL_ONLY=0 DSV41_SM12X_VISION=1 \
MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=3 MAX_MODEL_LEN=150000 \
KV_CACHE_MEMORY_BYTES=2684354560 \
./start.sh up
```

Keep the existing memory guards. Set `LIMIT_MM` to `{"image":1}` through your
normal configuration mechanism for the tested single-image-per-request scope.
This example does not change network, credentials, or reboot policy. It does
not choose the terminal mode for you: retain a successfully qualified candidate
when that is the requested deployment; restore the original only when rollback
is the intended outcome. Never run both model arrangements at once.

A bidirectional image item has a maximum encoder budget of **1025 tokens** in
the pinned runtime. The text-only 1024 batch limit fails before weights load;
2048 passes that check. This batch limit is separate from context length and
client count.

## Observed runtime evidence

An isolated derivative of base image digest
`sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce`,
using the exact dependency sources and opt-in helper above, was exercised on two
DGX Sparks (SM121) with the original EXL3 checkpoint:

- Seven numerical checks against pinned upstream reference functions passed:
  text width 128; image width 1152 at token counts 1, 2, 3, and 18; a 65-token
  prefill shape; and 128-token dual-cache prefill with extra width 512.
- One native image request passed, then three simultaneous image requests
  passed. Distinct printed codes, colors, and shapes were all correct.
- The three requests overlapped, and scheduler metrics reached **3 running
  requests**. Each contained 259 prompt tokens and produced 30 completion
  tokens. These are short correctness canaries, not a throughput benchmark.
- The runtime reported 644117 shared KV tokens with 150000 maximum context,
  three active sequences, a 2.5 GiB KV pool, and DSpark enabled.
- No OOM or automatic container restart was reported. A post-reload image
  canary also passed, and the candidate was left serving with guards active.

**Not established:** three simultaneous full-150000-token requests; video;
multiple images per request; broad OCR/visual reasoning quality; long soak;
or reboot recovery. Runtime evidence comes from the isolated derivative, not
an end-to-end rebuild of this final packaging Dockerfile. A clean rebuild and
the new launcher path remain acceptance work before treating this packaging as
a supported recipe. Do not overwrite an active deployment merely to run those
checks.

## Local checks

```bash
python3 tests/test_sm12x_vision.py
python3 tests/test_sm120_block64.py
bash -n start.sh
```

The CPU tests verify the negative baseline, opt-in/default behavior,
idempotence, drift rejection, archive pins and hash rejection, image budget and
receipt checks, and both launcher call sites. They do not run inference or
prove a Docker build succeeds. The image build checks actual installed backend
files, but compilation itself does not prove GPU correctness.
