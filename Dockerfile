# DeepSeek-V4.1-Flash EXL3 on SM121 (NVIDIA GB10 / DGX Spark).
#
# Base image is the official DeepSeek-V4.1-Flash vLLM build (DSpark, CSA2, Engram).
# We do NOT use the GLM-5.3-Flash glm53-flash-arm64-cu130 image: that tree has
# no DeepseekV41 architecture. GLM NoPE-MLA padding patches do not apply here.
#
# This overlay adds:
#   * native SM121 exllamav3_ext (fused exl3_moe + fat kernels compiled in)
#   * packed EXL3 mul1 loader (mixed K, experts + linears)
#   * operational GB10 patches (stops-in-reasoning, decode floor, spinwait,
#     xgrammar) that are model-agnostic
#
# E2 fat kernels are K4/MCG-only (unused here); the E3 v2 grouped kernels below cover mul1 K2/K3/K4.
# selects them. Fused exl3_moe is the decode path.
#
# Build (context = repo root, aarch64 host):
#   docker build -t dsv41-flash-exl3:local .

ARG BASE=vllm/vllm-openai:deepseekv41-flash-0909
FROM ${BASE}

COPY overlay/patch_exl3_ext_aarch64.py /opt/dsv41/patch_exl3_ext_aarch64.py
COPY overlay/patch_exl3_fat_kernel.py /opt/dsv41/patch_exl3_fat_kernel.py
COPY overlay/exl3_fat_gemm.cu /opt/dsv41/exl3-fat-kernel/exl3_fat_gemm.cu
COPY overlay/exl3_fat_gemm.cuh /opt/dsv41/exl3-fat-kernel/exl3_fat_gemm.cuh
COPY overlay/exl3_fat_moe.cu /opt/dsv41/exl3-fat-kernel/exl3_fat_moe.cu
COPY overlay/exl3_fat_moe.cuh /opt/dsv41/exl3-fat-kernel/exl3_fat_moe.cuh
# v2 grouped fat-expert kernels (bits/codebook generic, separate gate/up inputs):
# built as the additive exl3_fat_moe_ext module after exllamav3_ext (no ext rebuild).
COPY overlay/e3v2/exl3_fat_moe.cu /opt/dsv41/e3v2/exl3_fat_moe.cu
COPY overlay/e3v2/exl3_fat_moe.cuh /opt/dsv41/e3v2/exl3_fat_moe.cuh
COPY overlay/build_exl3_fat_moe_ext.py /opt/dsv41/build_exl3_fat_moe_ext.py
# GB10-shaped exl3_moe (6 cp.async stages, 2 fragment stages, 2 blocks per SM; compile-time
# K=2/3/4 instances): additive exl3_moe_x_ext module, same layered path as the E3 v2 kernels.
COPY overlay/moex/exl3_moe_x.cu /opt/dsv41/moex/exl3_moe_x.cu
COPY overlay/moex/exl3_moe_x.cuh /opt/dsv41/moex/exl3_moe_x.cuh
COPY overlay/moex/build_moe_x.py /opt/dsv41/moex/build_moe_x.py

# v0.0.43 fused exl3_moe TORCH_CHECKs mcg-only. This checkpoint is mul1.
# v1.4.5 instantiates cb1 (mcg) and cb2 (mul1) fused MoE kernels.
ARG EXLLAMAV3_COMMIT=e648f1a131365aae15920073e761a3fa5a527654
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
ENV MAX_JOBS=8
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

# Register EXL3 if this image's vLLM does not already list it.
RUN python3 - <<'PY'
from pathlib import Path
import vllm

site = Path(vllm.__file__).resolve().parent
init = site / "model_executor/layers/quantization/__init__.py"
text = init.read_text()
changed = False
if '"exl3"' not in text.split("QuantizationMethods", 1)[-1][:800]:
    old = 'QuantizationMethods = Literal[\n    "awq",\n'
    new = 'QuantizationMethods = Literal[\n    "exl3",\n    "awq",\n'
    if text.count(old) == 1:
        text = text.replace(old, new)
        changed = True
    elif 'QuantizationMethods = Literal[' in text and '"exl3"' not in text:
        text = text.replace(
            'QuantizationMethods = Literal[\n',
            'QuantizationMethods = Literal[\n    "exl3",\n',
            1,
        )
        changed = True
lazy = (
    "    # Update the `method_to_config` with customized quantization methods.\n"
    "    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)\n"
)
if "from .exl3 import Exl3Config" not in text and lazy in text:
    text = text.replace(
        lazy,
        '    from .exl3 import Exl3Config\n'
        '    method_to_config["exl3"] = Exl3Config\n' + lazy,
        1,
    )
    changed = True
if changed:
    init.write_text(text)
    print("registered exl3 in QUANTIZATION_METHODS")
else:
    print("exl3 already present in quantization registry (overlay will replace the class)")
PY

# The vLLM image keeps CUDA toolkit headers under nvidia/cu13, not
# /usr/local/cuda/include. ExLlamaV3 ships AVX2/AVX512 CPU targets that
# do not compile on aarch64; stub them so the SM121 GEMM still builds.
# After pip install, import torch before exllamav3_ext so libc10.so is
# already mapped (this image does not set LD_LIBRARY_PATH during build).
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 -c "from pathlib import Path; assert (Path('/tmp/exllamav3')/'exllamav3/modules/quant/exl3.py').is_file()"; \
    python3 /opt/dsv41/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /opt/dsv41/patch_exl3_fat_kernel.py /tmp/exllamav3/exllamav3/exllamav3_ext /opt/dsv41/exl3-fat-kernel; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    export CPATH="${PY_SITE}/nvidia/cu13/include${CPATH:+:$CPATH}"; \
    export CPLUS_INCLUDE_PATH="${PY_SITE}/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
    export C_INCLUDE_PATH="${PY_SITE}/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    cd /tmp/exllamav3; \
    TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8 \
      pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch; import exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_gemm'); print('exllamav3_ext', exllamav3_ext.__file__, 'exl3_moe=yes fat_gemm=yes')"; \
    rm -rf /tmp/exllamav3 /root/.cache/pip

# E3 v2 grouped MoE kernels for this checkpoint's K=3/K=2 mul1 experts (and K=4 mcg/mul1):
# the exllamav3_ext-embedded v1 kernels above are K4/MCG-only and are ignored by the
# overlay once this module (abi 2) is importable.
RUN set -eux; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    python3 /opt/dsv41/build_exl3_fat_moe_ext.py --src /opt/dsv41/e3v2 --out /tmp/e3v2 --arch 121a --install "$PY_SITE"; \
    python3 -c "import torch, exl3_fat_moe_ext as m; assert int(m.exl3_fat_moe_abi()) == 2; print('exl3_fat_moe_ext abi', int(m.exl3_fat_moe_abi()))"; \
    rm -rf /tmp/e3v2

# GB10-shaped exl3_moe: the shipped 16-row decode instance is barrier-bound at one block per SM on
# SM121 (54-63 % of bandwidth); six cp.async stages, two fragment stages and two blocks per SM
# reach 87 % (1.4x per launch, identical output at group size 8). Selected at runtime with
# DSV41_EXL3_MOE_X (.env); the shipped kernel stays the default when it is unset.
RUN set -eux; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    python3 /opt/dsv41/moex/build_moe_x.py --src /opt/dsv41/moex --out /tmp/moex --arch 121a --install "$PY_SITE"; \
    python3 -c "import torch, exl3_moe_x_ext as m; assert int(m.moex_variant_index(2, 3)) >= 0; print('exl3_moe_x_ext variants', int(m.moex_num_variants()))"; \
    rm -rf /tmp/moex

# Python overlay AFTER the CUDA compile so edits do not rebuild exllamav3_ext.
COPY overlay/exl3.py /opt/dsv41/exl3.py
COPY overlay/patch_model_overrides.py /opt/dsv41/patch_model_overrides.py
COPY overlay/patch_exl3_packed_names.py /opt/dsv41/patch_exl3_packed_names.py
COPY overlay/patch_exl3_lm_head.py /opt/dsv41/patch_exl3_lm_head.py
COPY overlay/patch_engram_secondary.py /opt/dsv41/patch_engram_secondary.py
COPY overlay/patch_engram_file.py /opt/dsv41/patch_engram_file.py
COPY overlay/engram_layout.py /opt/dsv41/engram_layout.py
COPY overlay/engram_file_backend.py /opt/dsv41/engram_file_backend.py
COPY overlay/row_store.cpp /opt/dsv41/row_store.cpp
COPY overlay/patch_suppress_stops_in_reasoning.py /opt/dsv41/patch_suppress_stops_in_reasoning.py
COPY overlay/patch_scheduler_decode_floor.py /opt/dsv41/patch_scheduler_decode_floor.py
COPY overlay/patch_xgrammar_termination.py /opt/dsv41/patch_xgrammar_termination.py
COPY overlay/patch_spinwait.py /opt/dsv41/patch_spinwait.py
COPY overlay/patch_memory_log.py /opt/dsv41/patch_memory_log.py
COPY overlay/patch_h2d_stage.py /opt/dsv41/patch_h2d_stage.py
COPY overlay/patch_sm120_block64.py /opt/dsv41/patch_sm120_block64.py
COPY files/chat_template.jinja /opt/dsv41/chat_template.jinja
COPY files/exl3_k_map.json /opt/dsv41/exl3_k_map.json
COPY tests/test_exl3_overlay.py /opt/dsv41/test_exl3_overlay.py
COPY tests/test_exl3_lm_head.py /opt/dsv41/test_exl3_lm_head.py
COPY tests/test_suppress_stops.py /opt/dsv41/test_suppress_stops.py
COPY tests/test_scheduler_decode_floor.py /opt/dsv41/test_scheduler_decode_floor.py
COPY tests/test_xgrammar_termination.py /opt/dsv41/test_xgrammar_termination.py
COPY tests/test_spinwait_patch.py /opt/dsv41/test_spinwait_patch.py
COPY tests/test_engram_secondary.py /opt/dsv41/test_engram_secondary.py
COPY tests/test_engram_layout.py /opt/dsv41/test_engram_layout.py
COPY tests/test_row_store.py /opt/dsv41/test_row_store.py
COPY tests/test_memory_log.py /opt/dsv41/test_memory_log.py
COPY tests/test_h2d_stage.py /opt/dsv41/test_h2d_stage.py
COPY tests/test_sm120_block64.py /opt/dsv41/test_sm120_block64.py
COPY tests/test_exl3_pretune.py /opt/dsv41/test_exl3_pretune.py
COPY tests/test_engram_dequant.py /opt/dsv41/test_engram_dequant.py
COPY scripts/pack_engram.py /opt/dsv41/pack_engram.py
COPY scripts/boot-shape-warmup.sh /opt/dsv41/boot-shape-warmup.sh
COPY scripts/pyspy_dump.sh /opt/dsv41/pyspy_dump.sh

RUN python3 - <<'PY'
from pathlib import Path
import vllm
src = Path("/opt/dsv41/exl3.py")
dst = Path(vllm.__file__).resolve().parent / "model_executor/layers/quantization/exl3.py"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(src.read_text())
print("installed overlay exl3.py ->", dst)
PY

RUN python3 /opt/dsv41/patch_model_overrides.py
RUN python3 /opt/dsv41/patch_exl3_packed_names.py
RUN python3 /opt/dsv41/patch_exl3_lm_head.py
RUN python3 /opt/dsv41/patch_engram_secondary.py
RUN python3 /opt/dsv41/patch_engram_file.py
RUN g++ -O3 -shared -fPIC -pthread -o /opt/dsv41/librow_store.so /opt/dsv41/row_store.cpp
# py-spy: stack dumps of a hung rank (start.sh hang detector; needs --cap-add SYS_PTRACE)
RUN pip install --no-cache-dir -q py-spy || echo "WARN: py-spy install failed"
RUN python3 /opt/dsv41/patch_suppress_stops_in_reasoning.py || echo "WARN: suppress-stops patch skipped"
RUN python3 /opt/dsv41/patch_scheduler_decode_floor.py || echo "WARN: scheduler patch skipped"
RUN python3 /opt/dsv41/patch_xgrammar_termination.py || echo "WARN: xgrammar patch skipped (likely already upstream)"
RUN python3 /opt/dsv41/patch_spinwait.py --preflight || echo "WARN: spinwait preflight skipped"
RUN python3 /opt/dsv41/patch_h2d_stage.py
RUN python3 /opt/dsv41/patch_sm120_block64.py
RUN python3 /opt/dsv41/patch_memory_log.py

RUN EXL3_SELFCHECK_GPU=0 python3 /opt/dsv41/test_exl3_overlay.py \
    && python3 /opt/dsv41/test_suppress_stops.py \
    && python3 /opt/dsv41/test_scheduler_decode_floor.py \
    && python3 /opt/dsv41/test_spinwait_patch.py \
    && python3 /opt/dsv41/test_engram_secondary.py \
    && python3 /opt/dsv41/test_exl3_lm_head.py \
    && python3 /opt/dsv41/test_engram_layout.py \
    && python3 /opt/dsv41/test_row_store.py \
    && python3 /opt/dsv41/test_memory_log.py \
    && python3 /opt/dsv41/test_h2d_stage.py \
    && python3 /opt/dsv41/test_sm120_block64.py \
    && EXL3_SELFCHECK_GPU=0 python3 /opt/dsv41/test_exl3_pretune.py \
    && python3 /opt/dsv41/test_engram_dequant.py
RUN python3 /opt/dsv41/test_xgrammar_termination.py \
    || echo "WARN: xgrammar test skipped (patch may already be upstream)"

ARG DSV41_RECIPE_STAMP=unknown
LABEL dsv41.recipe.stamp=${DSV41_RECIPE_STAMP}
# Attaches the GHCR package to the repo (GitHub reads image.source).
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks" \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash EXL3 2.9bpw on 2x DGX Spark (GB10/SM121), vLLM + exllamav3 overlay" \
      org.opencontainers.image.licenses="AGPL-3.0"
