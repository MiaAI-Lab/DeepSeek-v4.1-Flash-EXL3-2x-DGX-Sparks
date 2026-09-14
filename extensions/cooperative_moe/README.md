# Cooperative MoE for DS4.1 TP2

A fixed-shape EXL3 MoE extension for dual-Spark DS4.1 serving. The extension uses
cooperative expert kernels to reduce decode latency while preserving the stock
path for unsupported layers and larger prefills. Activation is explicit; this
change does not modify the default image, launcher, or overlay.

**To try it:** follow the [complete two-node opt-in guide](../../docs/cooperative-moe-quickstart.md).
It includes the pinned image, build commands, artifact copying, per-node GPU
checks, exact `.env` settings, startup verification, and rollback. A standalone
`DSV41_COOPERATIVE_MOE=1` does not activate or install the extension.

Measured improvements include **23.9% higher poetry decode, 10.8% higher coding
decode, and 33.1% higher combined C2 decode** in the paired tests, with 32K prefill
effectively unchanged. See the [benchmark and validation report](../../docs/cooperative-moe.md)
for protocols, sample counts, numerical tolerances, and remaining validation.

## Supported configuration

| Component | Requirement |
|---|---|
| Hardware | Two SM121a Sparks, tensor parallelism 2 |
| Expert shape | Hidden 5120, local intermediate 1152, top-k 6 |
| Quantization | Uniform K2/K3 mul1 gate/up/down projections |
| Physical decode batch | 1–8 rows |
| Stream configuration | `DSV41_EXL3_SERIAL_STREAMS=1`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` |

K4 MTP, larger prefill batches, mixed metadata, and unsupported shapes use stock
dispatch. Scratch is allocated after weight loading and before graph capture.
The adapter never retries stock after a partially launched CUDA operation.

## Build

Requirements: an existing ExLlamaV3 checkout containing commit
`02aef45cd681b960a00afcd0749a4ab99e6c1bfe`, the ARM64 recipe image with CUDA 13,
and an empty output directory. The script archives the pinned commit rather than
using uncommitted upstream changes; it does not download or install anything.

```sh
bash extensions/cooperative_moe/build.sh EXLLAMAV3_CHECKOUT EMPTY_OUTPUT_DIRECTORY
```

Outputs are `cooperative_moe.so`, `runtime.py`, and the build log. To reproduce the
validated native artifact, use the documented image and mount the output at
`/work`. The build retains the original compiler input names and native ABI v1
symbols for binary compatibility. Public source filenames and runtime settings
use descriptive names; no device arithmetic was changed in that cleanup.

The binary and adapter digests are pinned in `prepare_profile.py`; the adapter
also verifies the binary at load time. Different toolchains or compiler paths
may change the binary digest. Do not bypass the check: review the build and
repeat the native/integration gates before repinning a different binary.

## Select a profile

The steps below describe the mechanism. For an executable operator workflow,
including copying to the worker and the actual start command, use the
[opt-in guide](../../docs/cooperative-moe-quickstart.md).

Stage `cooperative_moe.so` and `runtime.py` at the same container-visible path on
**both ranks**. The existing per-node vLLM cache mount can be used. Then generate
a separate overlay with verified artifacts:

```sh
python3 extensions/cooperative_moe/prepare_profile.py \
  --stock overlay/exl3.py \
  --artifacts VERIFIED_ARTIFACT_DIRECTORY \
  --runtime-directory /root/.cache/vllm/cooperative_moe \
  --output NEW_COOPERATIVE_OVERLAY.py
```

The generator refuses changed stock source, mismatched artifacts, unsafe relative
paths, and existing output files. It does not update configuration or restart a
service. During an approved maintenance window, select the generated file with
the launcher's existing `EXL3_OVERLAY_HOST` setting and retain the required stream
configuration. The generated profile explicitly enables the adapter; direct
integrations may use `DSV41_COOPERATIVE_MOE=1` when calling `install()`.
An existing `EXL3_OVERLAY_HOST` assignment in `.env` takes precedence over a
command-line overlay choice, so replace it in `.env` when selecting a profile.

Preserve the previous configuration for rollback. Remove the overlay override
and restore those settings at the next approved restart to return to stock.
Review/rebase the pinned overlay when upgrading the recipe. Merely placing this
directory in the repository does not change serving behavior.

## Tests

CPU-only tests require Python 3.10+ and its standard library:

```sh
python3 extensions/cooperative_moe/test_dispatch.py
python3 extensions/cooperative_moe/test_profile.py
bash -n extensions/cooperative_moe/build.sh
```

`test_cuda_integration.py` exercises actual Torch tensors, graph replay/mutation,
stock fallback, invalid routes, and output casting. It requires the selected
overlay, the recipe's `test_exl3_overlay.py` importable from `/opt/dsv41`, and
`DSV41_COOP_MAINTENANCE_TEST=1`. Run GPU validation only with sufficient free
memory in an approved maintenance window, not alongside active user workloads.

## Attribution

The native implementation derives from Turboderp's
[two-stage cooperative MoE kernel](https://github.com/turboderp-org/exllamav3/commit/58d4d7322a1b3bd70aae8412487b21cc5e205cf4).
The specialization fixes DS4.1 dimensions/invariants at three kernel entry points;
the device arithmetic is otherwise unchanged. The parameter header guards unused
ATen declarations for the standalone C ABI. The upstream MIT license is retained
in `native/LICENSE.exllamav3`.
