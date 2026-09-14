# SPDX-License-Identifier: Apache-2.0
"""EXL3 trellis quantization for DeepSeek-V4.1-Flash (mul1 codebook).

Checkpoint ABI (this 2.9 bpw conversion):
  quant_method=exl3, codebook=mul1, average bits=2.9
  per matrix: trellis (int16) + suh/svh (fp16) + mul1 (int32 marker 0x83D6B12D)
  K is per-tensor (routed experts K=2 or 3, shared 4/5, attn ~5, engram 4/5,
  lm_head 6, DSpark MTP experts 4). Infer K from trellis.shape[-1] // 16.

Routed experts run through fused exllamav3_ext.exl3_moe (K/mul1 native).
The GLM E2/E3 fat kernels stay loaded but are K4/MCG-only, so they never
win eligibility here — prefill fat experts fall back to the LinearEXL3 loop.
Dense / attention / shared / engram wkv / lm_head also load as packed EXL3 linears.

TP=2 shards gate/up column-wise and down row-wise; the MoE runner all-reduces
the combined output. Ported from the GLM-5.3-Flash EXL3 overlay.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
try:
    from vllm.model_executor.layers.linear import (
        LinearBase,
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
except ImportError:  # vLLM layout drift
    from vllm.model_executor.layers.linear.base import (  # type: ignore
        LinearBase,
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
try:
    from vllm.model_executor.layers.quantization import register_quantization_config
except ImportError:  # pragma: no cover
    def register_quantization_config(_name: str):
        def deco(cls):
            return cls
        return deco

from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

EXLLAMAV3_COMMIT = "e648f1a131365aae15920073e761a3fa5a527654"
EXLLAMAV3_VERSION = "1.4.5"
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
# mul1 codebook marker (unsigned 2212286765 = 0x83D6B12D)
MUL1_MULTIPLIER = 0x83D6B12D
MUL1_MARKER_SIGNED_INT32 = -2082680531
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1")
_K_MAP: dict[str, Any] | None = None
SWIGLU_LIMIT_DEFAULT = 10.0
# Default fused-kernel temp rows/expert. 1024 covers MNBT=1024 in one launch
# but measured slower than 128+fallback (P2b). Override with EXL3_TEMP_ROWS_FUSED.
TEMP_ROWS_FUSED = 128
MOE_ACT_SILU = 0
# Shared fused scratch: decode is sequential across layers.
_FUSED_TEMP_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_FAT_SCRATCH_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_COUNT_CACHE: dict[tuple, tuple[torch.Tensor, torch.cuda.Stream]] = {}
# Grouped (E3) fat-expert scratch: fat-row activation buffers, grown once.
_FAT_GROUPED_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}
_FAT_GROUPED_BYTES: dict[tuple, int] = {}
_FAT_BUCKET_EDGES = (16, 32, 64, 128, 256, 512, 1024, 2048)
_FAT_STATS: dict[str, Any] = {
    "layers": 0,
    "fat_layers": 0,
    "fat_experts": 0,
    "max_rows": 0,
    "sum_max_rows": 0,
    "hist": [0] * (len(_FAT_BUCKET_EDGES) + 1),
}
_FAT_TIERS = ("grouped", "kernel", "batched", "sorted", "legacy")
# Machine-checkable E2/E3 (fat-expert) diagnostics. Load-time fields mirror the
# most recent weight load; counters are monotonic per process and are the
# ground truth for what actually ran. The key set is contract — extend it only
# together with a bump of EXL3_FAT_DIAG_SCHEMA.
# Schema 2 adds the E3 grouped tier: sym_fat_moe, grouped_calls,
# grouped_scratch_bytes, grouped_eligible (and "grouped" in fallback_calls).
EXL3_FAT_DIAG_SCHEMA = 2
EXL3_FAT_DIAG_KEYS = (
    "schema",
    "configured_tier",
    "effective_tier",
    "tier_reason",
    "shared_suh",
    "shared_suh_layers",
    "moe_layers_loaded",
    "sym_exl3_moe",
    "sym_fat_gemm",
    "sym_fat_gemm_scatter",
    "sym_fat_moe",
    "grouped_calls",
    "grouped_scratch_bytes",
    "grouped_eligible",
    "cap_major",
    "cap_minor",
    "cap_ok",
    "tp_rank",
    "tp_size",
    "fused_temps_allocs",
    "fused_temps_bytes",
    "fat_scratch_allocs",
    "fat_scratch_bytes",
    "fat_scratch_peak_bytes",
    "prefill_layer_calls",
    "thin_calls",
    "row_tile_calls",
    "fallback_calls",
    "fallback_reasons",
    "fat_expert_runs",
    "direct_calls",
    "scatter_calls",
    "fat_stat_layers",
    "fat_layers",
    "fat_expert_slots",
    "max_rows",
)
_EXL3_FAT_DIAG: dict[str, Any] = {
    "schema": EXL3_FAT_DIAG_SCHEMA,
    "configured_tier": "legacy",
    "effective_tier": "legacy",
    "tier_reason": "unresolved",
    "shared_suh": False,
    "shared_suh_layers": 0,
    "moe_layers_loaded": 0,
    "sym_exl3_moe": False,
    "sym_fat_gemm": False,
    "sym_fat_gemm_scatter": False,
    "sym_fat_moe": False,
    "grouped_calls": 0,
    "grouped_scratch_bytes": 0,
    "grouped_eligible": False,
    "cap_major": -1,
    "cap_minor": -1,
    "cap_ok": False,
    "tp_rank": -1,
    "tp_size": 1,
    "fused_temps_allocs": 0,
    "fused_temps_bytes": 0,
    "fat_scratch_allocs": 0,
    "fat_scratch_bytes": 0,
    "fat_scratch_peak_bytes": 0,
    "prefill_layer_calls": 0,
    "thin_calls": 0,
    "row_tile_calls": 0,
    "fallback_calls": {tier: 0 for tier in _FAT_TIERS},
    "fallback_reasons": {},
    "fat_expert_runs": 0,
    "direct_calls": 0,
    "scatter_calls": 0,
}
_FAT_SCRATCH_BYTES: dict[tuple, int] = {}
_exl3_fat_tier_logged = False


def fused_moe_row_tile_enabled() -> bool:
    """GPU row tiles instead of LinearEXL3 fallback. Prefill-only; decode stays one launch.

    Measured slower than the 128-row fallback at MNBT=1024 (8 full-grid launches).
    Default off; keep for MNBT > temp rows if a later bump still overflows.
    """
    return os.environ.get("EXL3_MOE_ROW_TILE", "0") != "0"


def temp_rows_fused() -> int:
    raw = os.environ.get("EXL3_TEMP_ROWS_FUSED", "").strip()
    if not raw:
        return int(TEMP_ROWS_FUSED)
    return max(1, int(raw))

def sorted_fat_fallback_enabled() -> bool:
    """Use the existing expert-sorted buffers for oversized prefill experts."""
    return os.environ.get("EXL3_FAT_SORTED", "0") != "0"

def batched_fat_fallback_enabled() -> bool:
    """Enable E1 batched fat experts; implies expert-sorted routing."""
    return os.environ.get("EXL3_FAT_BATCHED", "0") != "0"

def fat_kernel_enabled() -> bool:
    """Enable the E2 fat kernel; implies E1 batching and sorted routing."""
    return os.environ.get("EXL3_FAT_KERNEL", "0") != "0"


def grouped_fat_enabled() -> bool:
    """Enable the E3 grouped fat-expert kernels (experimental, default OFF).

    One gather + one gate/up + one down launch cover every fat expert of a
    layer from device-side segment tables: no per-expert launches and no
    host synchronization on the routing counts. Needs the exl3_fat_moe
    kernels (exllamav3_ext built with exl3_fat_moe.cu, or the additive
    exl3_fat_moe_ext module). EXL3_FAT_GROUPED=0 (default) leaves the E2
    path and its cap untouched.
    """
    return os.environ.get("EXL3_FAT_GROUPED", "0") != "0"


def fat_expert_log_enabled() -> bool:
    """Per-step routing histogram. It costs one host sync per MoE layer under
    the grouped tier (the grouped path has none of its own), so it defaults
    off there."""
    default = "0" if grouped_fat_enabled() else "1"
    return os.environ.get("EXL3_FAT_EXPERT_LOG", default) != "0"

def configured_fat_tier() -> str:
    """Highest fat tier the env requests: grouped > kernel > batched > sorted > legacy."""
    if grouped_fat_enabled():
        return "grouped"
    if fat_kernel_enabled():
        return "kernel"
    if batched_fat_fallback_enabled():
        return "batched"
    if sorted_fat_fallback_enabled():
        return "sorted"
    return "legacy"


def exl3_fat_symbols() -> tuple[bool, bool, bool]:
    """(exl3_moe, exl3_fat_gemm, exl3_fat_gemm_scatter) availability."""
    try:
        ext = load_exllamav3_ext()
    except Exception:
        return False, False, False
    return (
        hasattr(ext, "exl3_moe"),
        hasattr(ext, "exl3_fat_gemm"),
        hasattr(ext, "exl3_fat_gemm_scatter"),
    )


EXL3_FAT_MOE_SYMBOLS = (
    "exl3_fat_moe_gather",
    "exl3_fat_moe_gateup",
    "exl3_fat_moe_down",
    "exl3_fat_moe_tile_rows_gateup",
    "exl3_fat_moe_tile_rows_down",
)
# Grouped kernels: 16 B vector atomics (sm_90+); the image builds sm_121a.
EXL3_FAT_MOE_MIN_CAPABILITY = (9, 0)
# Kernel interface generation this overlay speaks: 2 = (bits, cb)-generic
# grouped kernels with separate gate/up gathered inputs (overlay/e3v2).
EXL3_FAT_MOE_ABI = 2
# (bits, codebook) pairs the v2 kernels are instantiated for.
EXL3_FAT_MOE_SHAPES = {(4, "mcg"), (4, "mul1"), (3, "mul1"), (2, "mul1")}
_FAT_MOE_EXT_CACHE: list = []


def load_fat_moe_ext():
    """Module carrying the E3 kernels, or None.

    Two supported sources: exllamav3_ext itself (bindings patched at the
    full image build by patch_exl3_fat_kernel.py) or the additive
    `exl3_fat_moe_ext` module (layered candidate image). Resolved once.
    """
    if _FAT_MOE_EXT_CACHE:
        return _FAT_MOE_EXT_CACHE[0]
    found = None
    # v2 (bits/codebook generic, separate gate/up inputs) ships as the additive
    # exl3_fat_moe_ext module; it is preferred over kernels embedded in
    # exllamav3_ext, which may still be the K4/MCG-only v1 build.
    candidates = []
    try:
        import exl3_fat_moe_ext  # noqa: F401

        candidates.append(exl3_fat_moe_ext)
    except Exception:
        pass
    try:
        candidates.append(load_exllamav3_ext())
    except Exception:
        pass
    for mod in candidates:
        if not all(hasattr(mod, name) for name in EXL3_FAT_MOE_SYMBOLS):
            continue
        abi = getattr(mod, "exl3_fat_moe_abi", None)
        if abi is None or int(abi()) < EXL3_FAT_MOE_ABI:
            continue
        found = mod
        break
    _FAT_MOE_EXT_CACHE.append(found)
    return found


def exl3_fat_moe_symbols() -> bool:
    """True when every E3 grouped kernel entry point is importable."""
    return load_fat_moe_ext() is not None


def grouped_fat_eligibility(layer: torch.nn.Module) -> tuple[bool, str]:
    """Load-time checkpoint/device eligibility for the v2 grouped kernels.

    The grouped pointer-table interface carries bits/codebook per launch, so
    the checks here are: one (bits, codebook) per layer among the compiled
    set (EXL3_FAT_MOE_SHAPES), tile-friendly dimensions (hidden % 256 for the
    2x128 down tile and the gather's 128-wide blocks, intermediate % 128 for
    the gate/up tile), a device capability with 16 B vector atomics, and every
    packed tensor on one CUDA device. gate/up may carry different input sign
    vectors (v2 gathers each side separately).
    """
    inners = getattr(layer, "_exl3_inners", None) or []
    if not inners:
        return False, "no_inners"
    shapes = set()
    for pack in inners:
        for which in ("gate", "up", "down"):
            inner = pack[which]
            bits = int(getattr(inner, "K", 0) or 0)
            cb = "mul1" if bool(getattr(inner, "mul1", False)) else (
                "mcg" if bool(getattr(inner, "mcg", False)) else "none"
            )
            shapes.add((bits, cb))
    if len(shapes) != 1:
        return False, "mixed_" + "_".join(f"{b}{c}" for b, c in sorted(shapes))
    bits, cb = next(iter(shapes))
    if (bits, cb) not in EXL3_FAT_MOE_SHAPES:
        return False, f"uncompiled_{bits}{cb}"
    k_words = int(getattr(layer, "_exl3_k_words", 0) or 0)
    if k_words and k_words != bits * 16:
        return False, f"k_words_{k_words}"
    layer._exl3_fat_bits = bits
    layer._exl3_fat_cb = 2 if cb == "mul1" else 1
    hidden = int(getattr(layer, "_exl3_hidden_size", 0) or 0)
    inter = int(getattr(layer, "_exl3_intermediate_local", 0) or 0)
    if hidden <= 0 or hidden % 256:
        return False, f"hidden_{hidden}"
    if inter <= 0 or inter % 128:
        return False, f"intermediate_{inter}"
    device = layer.w13_trellis.device
    if device.type != "cuda":
        return False, f"device_{device.type}"
    for name in ("w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"):
        if getattr(layer, name).device != device:
            return False, f"device_mismatch_{name}"
    cap = exl3_device_capability()
    if cap < EXL3_FAT_MOE_MIN_CAPABILITY:
        return False, f"capability_{cap[0]}.{cap[1]}"
    return True, "eligible"


def fat_gemm_eligible(layer: torch.nn.Module) -> bool:
    """E2 fat GEMM TORCH_CHECKs K==4 && mcg && !mul1. This checkpoint is mul1."""
    inners = getattr(layer, "_exl3_inners", None) or []
    if not inners:
        return False
    gate = inners[0].get("gate") if isinstance(inners[0], dict) else None
    if gate is None:
        return False
    return (
        int(getattr(gate, "K", 0)) == 4
        and bool(getattr(gate, "mcg", False))
        and not bool(getattr(gate, "mul1", False))
    )


def exl3_device_capability() -> tuple[int, int]:
    """CUDA capability of the current device; (-1, -1) without a GPU."""
    if not torch.cuda.is_available():
        return -1, -1
    try:
        return tuple(int(v) for v in torch.cuda.get_device_capability())
    except Exception:
        return -1, -1


def _exl3_tp_rank_size() -> tuple[int, int]:
    """TP identity so each rank's diag line is attributable; -1 outside vLLM."""
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        return -1, 1


def resolve_exl3_fat_tier(
    shared_suh: bool,
    symbols: tuple[bool, bool, bool] | None = None,
    fat_moe_symbols: bool | None = None,
    grouped_eligible: tuple[bool, str] | None = None,
) -> tuple[str, str]:
    """Map the configured fat tier onto what this image + checkpoint can run.

    Order matters: the checkpoint cap comes first. E1 batched, the E2 kernel
    and the E3 grouped kernels run gate+up behind a single input Hadamard
    (gate.suh), so a checkpoint without shared SUH caps the tier at sorted —
    a legitimate lower tier that needs no fat symbols, whatever the image.
    Only when the kernel would actually run does a missing symbol become an
    image/flag mismatch, failing closed here at load instead of mid-prefill.

    Grouped: a request without the E3 symbols fails closed. A checkpoint or
    device the grouped kernels cannot run (K4/MCG, dimensions, capability)
    deliberately falls back to the E2 kernel tier with the reason recorded,
    and that fallback is itself subject to the E2 symbol check.
    """
    configured = configured_fat_tier()
    if configured == "legacy":
        return configured, "none_requested"
    if not shared_suh and configured in ("kernel", "batched"):
        return "sorted", "shared_suh_absent"
    reason_prefix = ""
    if configured == "grouped":
        if fat_moe_symbols is None:
            fat_moe_symbols = exl3_fat_moe_symbols()
        if not fat_moe_symbols:
            raise RuntimeError(
                "EXL3_FAT_GROUPED=1 requires "
                + "/".join(EXL3_FAT_MOE_SYMBOLS)
                + " (exllamav3_ext or exl3_fat_moe_ext); this image was built "
                "without exl3_fat_moe.cu — set EXL3_FAT_GROUPED=0 (E2 kernel) "
                "or build the E3 candidate image"
            )
        if grouped_eligible is None:
            grouped_eligible = (True, "eligible")
        if grouped_eligible[0]:
            return configured, "grouped_ok"
        # Ineligible for the grouped kernels: fall back to E2 deliberately.
        configured = "kernel"
        reason_prefix = f"grouped_ineligible_{grouped_eligible[1]}:"
        if not shared_suh:
            # E2 runs gate+up behind one input Hadamard; without shared SUH
            # the layer caps at the sorted tier like a plain E2 request.
            return "sorted", reason_prefix + "shared_suh_absent"
    if symbols is None:
        symbols = exl3_fat_symbols()
    if configured == "kernel":
        missing = [
            name
            for name, present in zip(
                ("exl3_fat_gemm", "exl3_fat_gemm_scatter"), symbols[1:]
            )
            if not present
        ]
        if missing:
            raise RuntimeError(
                "EXL3_FAT_KERNEL=1 requires exllamav3_ext."
                + "/".join(missing)
                + "; this image was built without the fat kernel — unset "
                "EXL3_FAT_KERNEL or serve an E2 image"
            )
    return configured, f"{reason_prefix}{configured}_ok"


def exl3_fat_diag() -> dict[str, Any]:
    """Snapshot of the E2 diagnostics; the key set is EXL3_FAT_DIAG_KEYS."""
    diag = dict(_EXL3_FAT_DIAG)
    diag["fallback_calls"] = dict(_EXL3_FAT_DIAG["fallback_calls"])
    diag["fallback_reasons"] = dict(_EXL3_FAT_DIAG["fallback_reasons"])
    diag.update(
        fat_stat_layers=_FAT_STATS["layers"],
        fat_layers=_FAT_STATS["fat_layers"],
        fat_expert_slots=_FAT_STATS["fat_experts"],
        max_rows=_FAT_STATS["max_rows"],
    )
    return diag


def _exl3_fat_diag_line() -> str:
    d = exl3_fat_diag()
    parts = [
        f"schema={d['schema']}",
        f"configured_tier={d['configured_tier']}",
        f"effective_tier={d['effective_tier']}",
        f"tier_reason={d['tier_reason']}",
        f"shared_suh={int(d['shared_suh'])}",
        f"shared_suh_layers={d['shared_suh_layers']}/{d['moe_layers_loaded']}",
        f"sym_exl3_moe={int(d['sym_exl3_moe'])}",
        f"sym_fat_gemm={int(d['sym_fat_gemm'])}",
        f"sym_fat_gemm_scatter={int(d['sym_fat_gemm_scatter'])}",
        f"sym_fat_moe={int(d['sym_fat_moe'])}",
        f"grouped_eligible={int(d['grouped_eligible'])}",
        f"grouped_calls={d['grouped_calls']}",
        f"grouped_scratch_bytes={d['grouped_scratch_bytes']}",
        f"cap={d['cap_major']}.{d['cap_minor']}",
        f"cap_ok={int(d['cap_ok'])}",
        f"tp_rank={d['tp_rank']} tp_size={d['tp_size']}",
        f"prefill_layer_calls={d['prefill_layer_calls']}",
        f"thin_calls={d['thin_calls']}",
        f"row_tile_calls={d['row_tile_calls']}",
        "fallback_calls="
        + ",".join(f"{t}={d['fallback_calls'][t]}" for t in _FAT_TIERS),
        "fallback_reasons="
        + (
            ",".join(f"{r}={n}" for r, n in sorted(d["fallback_reasons"].items()))
            or "none"
        ),
        f"fat_expert_runs={d['fat_expert_runs']}",
        f"direct_calls={d['direct_calls']}",
        f"scatter_calls={d['scatter_calls']}",
        f"fat_layers={d['fat_layers']}",
        f"fat_expert_slots={d['fat_expert_slots']}",
        f"max_rows={d['max_rows']}",
        f"fused_temps_allocs={d['fused_temps_allocs']}",
        f"fused_temps_bytes={d['fused_temps_bytes']}",
        f"fat_scratch_allocs={d['fat_scratch_allocs']}",
        f"fat_scratch_bytes={d['fat_scratch_bytes']}",
        f"fat_scratch_peak_bytes={d['fat_scratch_peak_bytes']}",
    ]
    return " ".join(parts)


def _record_exl3_fat_reason(reason: str) -> None:
    reasons = _EXL3_FAT_DIAG["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _record_exl3_fat_tier(layer: torch.nn.Module, tier: str, reason: str) -> None:
    _EXL3_FAT_DIAG["fallback_calls"][tier] += 1
    _record_exl3_fat_reason(reason)
    layer._exl3_last_fat_fallback = tier
    layer._exl3_last_fat_reason = reason


def _record_exl3_fat_resolution(layer: torch.nn.Module) -> None:
    """Resolve the E2 tier once per MoE layer at weight load and log once.

    Per-layer truth lands on layer._exl3_fat_effective_tier; the module state
    mirrors the most recent load. A resolution that changes between layers of
    one model is a checkpoint property worth a loud line, not a silent one.
    """
    global _exl3_fat_tier_logged
    shared_suh = bool(getattr(layer, "_exl3_shared_w13_suh", False))
    grouped_eligible = grouped_fat_eligibility(layer)
    layer._exl3_grouped_eligible = grouped_eligible
    effective_tier, tier_reason = resolve_exl3_fat_tier(
        shared_suh, grouped_eligible=grouped_eligible
    )
    layer._exl3_fat_effective_tier = effective_tier
    layer._exl3_fat_tier_reason = tier_reason

    diag = _EXL3_FAT_DIAG
    sym_moe, sym_gemm, sym_scatter = exl3_fat_symbols()
    cap_major, cap_minor = exl3_device_capability()
    diag["moe_layers_loaded"] += 1
    if shared_suh:
        diag["shared_suh_layers"] += 1
    diag["shared_suh"] = diag["shared_suh_layers"] == diag["moe_layers_loaded"]
    diag["configured_tier"] = configured_fat_tier()
    diag["sym_exl3_moe"] = sym_moe
    diag["sym_fat_gemm"] = sym_gemm
    diag["sym_fat_gemm_scatter"] = sym_scatter
    diag["sym_fat_moe"] = exl3_fat_moe_symbols()
    diag["grouped_eligible"] = bool(grouped_eligible[0])
    diag["cap_major"] = cap_major
    diag["cap_minor"] = cap_minor
    # LinearEXL3 (and the fat GEMM built on it) needs >= Ampere; GB10 is SM121.
    diag["cap_ok"] = (cap_major, cap_minor) >= (8, 0)
    diag["tp_rank"], diag["tp_size"] = _exl3_tp_rank_size()

    if diag["tier_reason"] == "unresolved":
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    elif diag["effective_tier"] != effective_tier:
        logger.warning(
            "exl3 e2 diag tier changed %s -> %s (%s): %s",
            diag["effective_tier"],
            effective_tier,
            tier_reason,
            _exl3_fat_diag_line(),
        )
        diag["effective_tier"] = effective_tier
        diag["tier_reason"] = tier_reason
    if not _exl3_fat_tier_logged:
        _exl3_fat_tier_logged = True
        if (
            diag["effective_tier"] != diag["configured_tier"]
            and diag["configured_tier"] != "legacy"
        ):
            logger.warning("exl3 e2 diag degraded %s", _exl3_fat_diag_line())
        else:
            logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())


def reset_exl3_fat_diag_counters() -> None:
    """Zero the E2 runtime counters; load-time fields and live bytes stay.

    Scratch current/peak restart from the resident cache so a windowed read
    (e.g. exactly one cold request) still reports honest byte counts.
    """
    diag = _EXL3_FAT_DIAG
    for key in (
        "prefill_layer_calls",
        "thin_calls",
        "row_tile_calls",
        "fat_expert_runs",
        "direct_calls",
        "scatter_calls",
        "fat_scratch_allocs",
        "grouped_calls",
    ):
        diag[key] = 0
    diag["fallback_calls"] = {tier: 0 for tier in _FAT_TIERS}
    diag["fallback_reasons"] = {}
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]
    diag["grouped_scratch_bytes"] = sum(_FAT_GROUPED_BYTES.values())


def reset_exl3_fat_expert_stats() -> None:
    _FAT_STATS["layers"] = 0
    _FAT_STATS["fat_layers"] = 0
    _FAT_STATS["fat_experts"] = 0
    _FAT_STATS["max_rows"] = 0
    _FAT_STATS["sum_max_rows"] = 0
    _FAT_STATS["hist"] = [0] * (len(_FAT_BUCKET_EDGES) + 1)


def _fat_bucket(n: int) -> int:
    for i, edge in enumerate(_FAT_BUCKET_EDGES):
        if n <= edge:
            return i
    return len(_FAT_BUCKET_EDGES)


def record_exl3_fat_expert_stats(
    counts: torch.Tensor,
    *,
    max_rows: int | None = None,
    counts_host: list[int] | None = None,
) -> dict[str, Any]:
    """Prefill-only routing stats. Reuse an existing host copy when available."""
    if counts_host is None:
        if max_rows is None:
            max_rows = int(counts.max().item())
        n_fat = int((counts > temp_rows_fused()).sum().item())
    else:
        if max_rows is None:
            max_rows = max(counts_host, default=0)
        cap = temp_rows_fused()
        n_fat = sum(n > cap for n in counts_host)
    st = _FAT_STATS
    st["layers"] += 1
    st["sum_max_rows"] += max_rows
    st["hist"][_fat_bucket(max_rows)] += 1
    if max_rows > st["max_rows"]:
        st["max_rows"] = max_rows
    if n_fat:
        st["fat_layers"] += 1
        st["fat_experts"] += n_fat
    # 42 routed-MoE layers per engine step (MoE from layer 3 of 45).
    if st["layers"] % 42 == 0:
        avg = st["sum_max_rows"] / st["layers"]
        le128 = sum(st["hist"][:4])
        gt128 = sum(st["hist"][4:])
        logger.info(
            "exl3 fat-expert P0: layers=%d fat_layers=%d (%.1f%%) fat_expert_slots=%d "
            "max_rows=%d avg_max_rows=%.1f hist_le128=%d hist_gt128=%d hist=%s",
            st["layers"],
            st["fat_layers"],
            100.0 * st["fat_layers"] / st["layers"],
            st["fat_experts"],
            st["max_rows"],
            avg,
            le128,
            gt128,
            st["hist"],
        )
        logger.info("exl3 e2 diag %s", _exl3_fat_diag_line())
    return {
        "max_rows": max_rows,
        "n_fat": n_fat,
        "layers": st["layers"],
        "fat_layers": st["fat_layers"],
    }


# ----------------------------------------------------------------------------
# Host->device staging. safetensors 0.8 maps every shard MAP_PRIVATE|PROT_WRITE
# (rw-p). On GB10 a device copy that reads straight from that mapping pins the
# pages with write intent, which copy-on-writes them into anonymous host memory
# that lives as long as the mapping does; vLLM's DeepseekV41 VL wrapper sorts
# the whole weight stream (all 39 mappings stay alive for the entire load), so
# the 2026-09-11/12 boots lost ~1.2 GiB per layer to that and wedged both Sparks.
# Measured (scratchpad smaps probe): direct copy -> Anonymous 1.2 GiB per layer
# inside the shard mappings; copy through a pinned buffer -> 0. So every copy
# from a CPU (mmap) tensor goes through a reusable pinned buffer.
# ----------------------------------------------------------------------------
_PIN_POOL: dict[torch.dtype, torch.Tensor] = {}
_PIN_MIN_ELEMS = 1 << 20


def _pinned(dtype: torch.dtype, numel: int) -> torch.Tensor:
    buf = _PIN_POOL.get(dtype)
    if buf is None or buf.numel() < numel:
        want = max(int(numel), _PIN_MIN_ELEMS)
        if buf is not None:
            want = max(want, buf.numel() * 2)
        buf = torch.empty(want, dtype=dtype, device="cpu", pin_memory=True)
        _PIN_POOL[dtype] = buf
    return buf[:numel]


def h2d_copy(dest: torch.Tensor, src: torch.Tensor) -> None:
    """dest.copy_(src) that never DMAs from a (possibly mmap-backed) CPU tensor."""
    if src.device.type != "cpu" or dest.device.type == "cpu" or src.numel() == 0:
        dest.copy_(src)
        return
    if src.dtype != dest.dtype:
        src = src.to(dtype=dest.dtype)
    stage = _pinned(src.dtype, src.numel()).view(src.shape)
    stage.copy_(src)  # CPU memcpy: reads the mapping, no page pinning, no COW
    dest.copy_(stage)  # synchronous DMA from pinned memory


def h2d_alloc(src: torch.Tensor, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """src.to(device) without a DMA from the mapping."""
    out = torch.empty(src.shape, dtype=dtype or src.dtype, device=device)
    h2d_copy(out, src)
    return out


def _narrow_tp(tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int) -> torch.Tensor:
    """This rank's slice as a view; h2d_copy handles the strides (one memcpy)."""
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk)


def shard_exl3_col(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded


def shard_exl3_row(loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded


def _install_exllamav3_namespace() -> None:
    """Load LinearEXL3 without running exllamav3/__init__.py (FlashAttention)."""
    if "exllamav3.modules.quant.exl3" in sys.modules:
        return
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 package is not installed in this image")
    package_root = Path(list(spec.submodule_search_locations)[0])

    # Stub only packages whose __init__.py pulls FlashAttention / serving extras.
    # Leave .ext, .util, and .modules.quant as real modules so LinearEXL3 loads.
    for name, path in (
        ("exllamav3", package_root),
        ("exllamav3.modules", package_root / "modules"),
        ("exllamav3.model", package_root / "model"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module

    if "exllamav3.model.config" not in sys.modules:
        config = types.ModuleType("exllamav3.model.config")
        config.__file__ = str(package_root / "model/config.py")
        config.__package__ = "exllamav3.model"
        config.Config = type("Config", (), {})

        class _InferParams:
            no_reconstruct = False

        class _NullConfig:
            infer_params = _InferParams()

        config.InferParams = _InferParams
        config.NullConfig = _NullConfig
        sys.modules[config.__name__] = config


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    marker: torch.Tensor,
    *,
    codebook: str = "mul1",
    out_dtype: torch.dtype = torch.float16,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand."""
    cls = load_linear_exl3_cls()
    kwargs: dict[str, Any] = dict(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        out_dtype=out_dtype,
        transformers_fix=True,
    )
    packed = marker.contiguous()
    if codebook == "mcg":
        kwargs["mcg"] = packed
    else:
        kwargs["mul1"] = packed
    return cls(**kwargs)


def execute_exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    marker: torch.Tensor,
    *,
    codebook: str = "mul1",
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Real EXL3 GEMM entry (LinearEXL3 / exllamav3_ext)."""
    inner = make_linear_exl3(
        trellis, suh, svh, marker, codebook=codebook, out_dtype=torch.float16
    )
    return inner.forward(x.contiguous().half(), {}, out_dtype=out_dtype)


def load_k_map() -> dict[str, Any]:
    global _K_MAP
    if _K_MAP is not None:
        return _K_MAP
    candidates = [
        Path("/opt/dsv41/exl3_k_map.json"),
        Path("/opt/glm53/exl3_k_map.json"),
        Path(__file__).resolve().parent.parent / "files" / "exl3_k_map.json",
    ]
    env = os.environ.get("DSV41_EXL3_K_MAP")
    if env:
        candidates.insert(0, Path(env))
    for path in candidates:
        if path.is_file():
            import json

            _K_MAP = json.loads(path.read_text())
            return _K_MAP
    _K_MAP = {}
    return _K_MAP


def k_for_prefix(prefix: str, codebook_bits: float | int | None = None) -> int:
    """K for a vLLM module prefix. Prefer the baked map; fall back to kind."""
    p = prefix.replace("-", ".")
    km = load_k_map()
    routed = km.get("routed") or {}
    shared = km.get("shared") or {}
    engram = km.get("engram") or {}
    m = re.search(r"(?:layers|layer)[._](\d+)", p)
    layer = m.group(1) if m else None
    if p.startswith("mtp") or ".mtp." in f".{p}.":
        return int(km.get("mtp_bits") or 4)
    # vLLM builds the DSpark draft (checkpoint mtp.{i}.*) as
    # model.layers.{num_hidden_layers + i}.* plus model.main_proj; every draft
    # tensor in this checkpoint is K=4 (mtp_bits), including attn and shared.
    n_main = int(km.get("num_hidden_layers") or 40)
    if layer is not None and int(layer) >= n_main:
        return int(km.get("mtp_bits") or 4)
    if "main_proj" in p:
        return int(km.get("mtp_bits") or 4)
    if "shared_expert" in p and layer is not None:
        k = shared.get(layer, shared.get(str(int(layer))))
        if isinstance(k, int):
            return k
    if "engram" in p and layer is not None:
        k = engram.get(layer, engram.get(str(int(layer))))
        if isinstance(k, int):
            return k
    if ".experts" in p or p.endswith("experts") or "routed" in p:
        if layer is not None:
            k = routed.get(layer, routed.get(str(int(layer))))
            if isinstance(k, int):
                return k
        return 3
    # Lightning indexer wk is packed K=8 (512→128). Generic attn is K=5.
    if "indexer.wk" in p:
        return int(km.get("indexer_wk_bits") or 8)
    if "attn" in p or "self_attn" in p:
        return int(km.get("attn_default") or km.get("head_bits") or 5)
    # Packed EXL3 lm_head (checkpoint head.{trellis,suh,svh,mul1}, K=head_bits).
    if p == "lm_head" or p.endswith(".lm_head") or p.endswith("lm_head"):
        return int(km.get("head_bits") or 6)
    if codebook_bits is not None:
        b = float(codebook_bits)
        if abs(b - round(b)) < 1e-6 and 2 <= round(b) <= 8:
            return int(round(b))
    return 3


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


def _exl3_moe_accepts_num_active(fn) -> bool:
    try:
        import inspect

        if "num_active" in inspect.signature(fn).parameters:
            return True
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    return "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30


def pin_exl3_expert_map(
    layer: torch.nn.Module, device: torch.device
) -> torch.Tensor | None:
    """Move expert_map onto `device` once. CUDA graph capture forbids a CPU→GPU copy."""
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        layer.expert_map = emap.to(device=device, dtype=torch.long)
    return layer.expert_map


def map_topk_to_local(
    ids: torch.Tensor,
    n_local: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """ids (T, K) global expert ids → local ids, invalid/non-local → n_local sentinel.

    `expert_map` must already live on `ids.device` (see pin_exl3_expert_map).
    """
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply (CUDA graphs forbid the copy)"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique-expert LinearEXL3 loop. `only_experts` is local ids (fat-expert fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    unique = torch.unique(ids)
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(act.contiguous().half(), {}, out_dtype=torch.float32)
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out

def apply_exl3_sorted_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run oversized experts from contiguous slices of the existing sort.

    One `counts.tolist()` in the caller replaces the legacy per-expert
    `unique.tolist()`, expert-map `.item()`, and `(ids == expert).nonzero()`
    synchronizations. Slices are views; only the LinearEXL3 inputs and outputs
    allocate, as they do in the legacy fallback.
    """
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1
        token_idx = token_sorted[start:offset]
        h = xh.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h, {}, out_dtype=torch.float32)
        up = pack["up"].forward(h, {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(
            act.contiguous().half(), {}, out_dtype=torch.float32
        )
        scale = weight_sorted[start:offset].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def _fat_scratch(
    device: torch.device,
    rows: int,
    gate: Any,
) -> dict[str, torch.Tensor]:
    """Return shared prefill scratch, growing once if the configured chunk grows."""
    hidden = int(gate.in_features)
    intermediate = int(gate.out_features)
    configured = int(
        os.environ.get(
            "EXL3_FAT_SCRATCH_ROWS",
            os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"),
        )
        or 0
    )
    needed = max(256, rows, configured)
    capacity = 1 << (needed - 1).bit_length()
    key = (
        str(device),
        hidden,
        intermediate,
        int(gate.K),
        int(gate.trellis.shape[-1]),
    )
    scratch = _FAT_SCRATCH_CACHE.get(key)
    if scratch is not None and int(scratch["h"].shape[0]) >= rows:
        return scratch

    in_tiles, out_tiles, k_words = map(int, gate.trellis.shape)
    scratch = {
        "packed13": torch.empty(
            (in_tiles, 2 * out_tiles, k_words),
            dtype=torch.int16,
            device=device,
        ),
        "svh13": torch.empty(
            2 * intermediate, dtype=torch.float16, device=device
        ),
        "w13": torch.empty(
            (hidden, 2 * intermediate), dtype=torch.float16, device=device
        ),
        "w2": torch.empty(
            (intermediate, hidden), dtype=torch.float16, device=device
        ),
        "h": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "h13": torch.empty(
            (capacity, hidden), dtype=torch.float16, device=device
        ),
        "gate_up": torch.empty(
            (capacity, 2 * intermediate), dtype=torch.float32, device=device
        ),
        "act": torch.empty(
            (capacity, intermediate), dtype=torch.float32, device=device
        ),
        "act_h": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "h2": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
        "down": torch.empty(
            (capacity, hidden), dtype=torch.float32, device=device
        ),
    }
    _FAT_SCRATCH_CACHE[key] = scratch
    _FAT_SCRATCH_BYTES[key] = sum(
        t.numel() * t.element_size() for t in scratch.values()
    )
    diag = _EXL3_FAT_DIAG
    diag["fat_scratch_allocs"] += 1
    diag["fat_scratch_bytes"] = sum(_FAT_SCRATCH_BYTES.values())
    if diag["fat_scratch_bytes"] > diag["fat_scratch_peak_bytes"]:
        diag["fat_scratch_peak_bytes"] = diag["fat_scratch_bytes"]
    return scratch


def _stage_counts_to_host(
    counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.cuda.Stream]:
    """Copy routing counts on a side stream before launching thin experts."""
    key = (str(counts.device), int(counts.numel()))
    cached = _FAT_COUNT_CACHE.get(key)
    if cached is None:
        host = torch.empty(
            int(counts.numel()), dtype=counts.dtype, device="cpu", pin_memory=True
        )
        stream = torch.cuda.Stream(device=counts.device)
        cached = (host, stream)
        _FAT_COUNT_CACHE[key] = cached
    host, stream = cached
    current = torch.cuda.current_stream(counts.device)
    with torch.cuda.stream(stream):
        stream.wait_stream(current)
        host.copy_(counts, non_blocking=True)
    return host, stream


def apply_exl3_batched_fat(
    xh: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    counts_host: list[int],
    inners: list[dict[str, Any]],
    limit: float,
    cap: int,
    out: torch.Tensor,
    use_kernel: bool = False,
) -> torch.Tensor:
    """Run fat experts with persistent buffers and optional direct trellis GEMM."""
    ext = load_exllamav3_ext()
    offset = 0
    for e, n_rows in enumerate(counts_host):
        start = offset
        offset += n_rows
        if n_rows <= cap:
            continue
        _EXL3_FAT_DIAG["fat_expert_runs"] += 1

        token_idx = token_sorted[start:offset]
        gate = inners[e]["gate"]
        up = inners[e]["up"]
        down = inners[e]["down"]
        scratch = _fat_scratch(xh.device, n_rows, gate)
        intermediate = int(gate.out_features)

        h = scratch["h"][:n_rows]
        h13 = scratch["h13"][:n_rows]
        torch.index_select(xh, 0, token_idx, out=h)
        ext.had_r_128(h, h13, gate.suh, None, 1.0)

        packed13 = scratch["packed13"]
        out_tiles = int(gate.trellis.shape[1])
        packed13[:, :out_tiles].copy_(gate.trellis)
        packed13[:, out_tiles:].copy_(up.trellis)
        gate_up = scratch["gate_up"][:n_rows]
        svh13 = scratch["svh13"]
        svh13[:intermediate].copy_(gate.svh)
        svh13[intermediate:].copy_(up.svh)
        if use_kernel:
            if not hasattr(ext, "exl3_fat_gemm"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires exllamav3_ext.exl3_fat_gemm"
                )
            ext.exl3_fat_gemm(
                h13, packed13, gate_up, svh13, gate.K, gate.mcg, gate.mul1
            )
            _EXL3_FAT_DIAG["direct_calls"] += 1
        else:
            w13 = scratch["w13"]
            ext.reconstruct(w13, packed13, gate.K, gate.mcg, gate.mul1)
            ext.hgemm(h13, w13, gate_up)
            ext.had_r_128(gate_up, gate_up, None, svh13, 1.0)

        gate_out = gate_up[:, :intermediate]
        up_out = gate_up[:, intermediate:]
        gate_out.clamp_(max=limit)
        up_out.clamp_(min=-limit, max=limit)
        act = scratch["act"][:n_rows]
        torch.sigmoid(gate_out, out=act)
        act.mul_(gate_out).mul_(up_out)
        act_h = scratch["act_h"][:n_rows]
        act_h.copy_(act)

        h2 = scratch["h2"][:n_rows]
        ext.had_r_128(act_h, h2, down.suh, None, 1.0)
        if use_kernel:
            if not hasattr(ext, "exl3_fat_gemm_scatter"):
                raise RuntimeError(
                    "EXL3_FAT_KERNEL=1 requires "
                    "exllamav3_ext.exl3_fat_gemm_scatter"
                )
            ext.exl3_fat_gemm_scatter(
                h2,
                down.trellis,
                out,
                down.svh,
                token_idx,
                weight_sorted[start:offset],
                down.K,
                down.mcg,
                down.mul1,
            )
            _EXL3_FAT_DIAG["scatter_calls"] += 1
        else:
            w2 = scratch["w2"]
            ext.reconstruct(w2, down.trellis, down.K, down.mcg, down.mul1)
            down_out = scratch["down"][:n_rows]
            ext.hgemm(h2, w2, down_out)
            ext.had_r_128(down_out, down_out, None, down.svh, 1.0)
            down_out.mul_(weight_sorted[start:offset].unsqueeze(-1))
            out.index_add_(0, token_idx, down_out)
    return out


def _grouped_scratch(
    device: torch.device, rows: int, hidden: int, intermediate: int
) -> dict[str, torch.Tensor]:
    """Fat-row activation buffers for the grouped tier, grown once.

    Capacity covers every routed slot of the configured prefill chunk
    (MAX_NUM_BATCHED_TOKENS x top-k, EXL3_FAT_GROUPED_TOPK, default 8) so
    steady-state prefill never reallocates; a larger request grows it once
    more (outside CUDA graph capture, where growth would be illegal).
    Persistent: h13 [rows, hidden] fp16 + h2 [rows, intermediate] fp16.
    """
    configured = int(
        os.environ.get(
            "EXL3_FAT_SCRATCH_ROWS",
            os.environ.get("MAX_NUM_BATCHED_TOKENS", "0"),
        )
        or 0
    )
    topk = int(os.environ.get("EXL3_FAT_GROUPED_TOPK", "8") or 8)
    needed = max(256, rows)
    key = (str(device), hidden, intermediate)
    scratch = _FAT_GROUPED_CACHE.get(key)
    if scratch is not None and int(scratch["h13"].shape[0]) >= needed:
        return scratch
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "EXL3 grouped scratch growth during CUDA graph capture; warm the "
            f"largest shape first (need {needed} rows)"
        )
    capacity = max(needed, configured * topk)
    scratch = {
        "h13": torch.empty((capacity, hidden), dtype=torch.float16, device=device),
        "h13u": torch.empty((capacity, hidden), dtype=torch.float16, device=device),
        "h2": torch.empty(
            (capacity, intermediate), dtype=torch.float16, device=device
        ),
    }
    _FAT_GROUPED_CACHE[key] = scratch
    _FAT_GROUPED_BYTES[key] = sum(
        t.numel() * t.element_size() for t in scratch.values()
    )
    _EXL3_FAT_DIAG["grouped_scratch_bytes"] = sum(_FAT_GROUPED_BYTES.values())
    return scratch


def _excl_cumsum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inclusive = torch.cumsum(x, 0)
    return inclusive - x, inclusive


def build_grouped_fat_tables(
    counts: torch.Tensor,
    cap: int,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    rows_cap: int,
    tile_rows: int,
) -> dict[str, torch.Tensor]:
    """Device-side row/segment tables for the grouped fat kernels.

    Fat experts (count > cap) are laid out back to back in expert order in a
    fat-row buffer; each kernel CTA owns one `tile_rows` slice of one expert.
    Everything is computed with device ops on capacity-sized tensors and the
    kernels read the live `num_rows` / `num_segs`, so no host sync happens
    and the layer stays CUDA-graph capturable. `counts` excludes the
    invalid/nonlocal sentinel bucket, which the sort places after every
    real expert, so sentinel routes never enter a fat segment.
    """
    n_exp = int(counts.numel())
    device = counts.device
    fat_rows = torch.where(counts > cap, counts, torch.zeros_like(counts))
    row_off, row_cum = _excl_cumsum(fat_rows)
    sorted_off, _ = _excl_cumsum(counts)
    tiles = (fat_rows + (tile_rows - 1)) // tile_rows
    tile_off, tile_cum = _excl_cumsum(tiles)
    num_segs = tile_cum[-1:].to(torch.int32)
    num_rows = row_cum[-1:].to(torch.int32)
    max_segs = (rows_cap + tile_rows - 1) // tile_rows + n_exp
    seg = torch.arange(max_segs, device=device)
    e = torch.searchsorted(tile_cum, seg, right=True).clamp_(max=n_exp - 1)
    local_tile = seg - tile_off[e]
    seg_row0 = row_off[e] + local_tile * tile_rows
    seg_rows = torch.clamp(fat_rows[e] - local_tile * tile_rows, min=0, max=tile_rows)
    r = torch.arange(rows_cap, device=device)
    re = torch.searchsorted(row_cum, r, right=True).clamp_(max=n_exp - 1)
    src = (sorted_off[re] + (r - row_off[re])).clamp_(max=rows_cap - 1)
    return {
        "seg_expert": e.to(torch.int32),
        "seg_row0": seg_row0.to(torch.int32),
        "seg_rows": seg_rows.to(torch.int32),
        "num_segs": num_segs,
        "num_rows": num_rows,
        "row_expert": re.to(torch.int32),
        "row_token": token_sorted.index_select(0, src),
        "row_weight": weight_sorted.index_select(0, src),
    }


def apply_exl3_grouped_fat(
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    layer: torch.nn.Module,
    cap: int,
    limit: float,
) -> None:
    """E3: every fat expert of the layer in three launches, no host sync."""
    ext = load_fat_moe_ext()
    if ext is None:
        raise RuntimeError("EXL3 grouped tier selected but the E3 kernels are not loaded")
    ptrs = layer._exl3_ptrs
    device = ptrs["gate_trellis"].device
    if xh.device != device or out.device != device or counts.device != device:
        raise RuntimeError(
            f"EXL3 grouped tier: activations on {xh.device}, experts on {device}"
        )
    if not (xh.is_contiguous() and out.is_contiguous() and out.dtype == torch.float32):
        raise RuntimeError("EXL3 grouped tier needs contiguous fp16 input / fp32 output")
    hidden = int(xh.shape[1])
    intermediate = int(layer._exl3_intermediate_local)
    rows_cap = int(token_sorted.numel())
    scratch = _grouped_scratch(device, rows_cap, hidden, intermediate)
    h13 = scratch["h13"][:rows_cap]
    h13u = scratch["h13u"][:rows_cap]
    h2 = scratch["h2"][:rows_cap]
    bits = int(getattr(layer, "_exl3_fat_bits", 0) or 0)
    cb = int(getattr(layer, "_exl3_fat_cb", 0) or 0)
    if not bits or not cb:
        raise RuntimeError("EXL3 grouped tier: layer has no resolved (bits, codebook)")
    tile_gu = int(ext.exl3_fat_moe_tile_rows_gateup())
    tile_dn = int(ext.exl3_fat_moe_tile_rows_down())
    token_sorted = token_sorted.contiguous()
    weight_sorted = weight_sorted.contiguous()
    tg = build_grouped_fat_tables(
        counts, cap, token_sorted, weight_sorted, rows_cap, tile_gu
    )
    td = (
        tg
        if tile_dn == tile_gu
        else build_grouped_fat_tables(
            counts, cap, token_sorted, weight_sorted, rows_cap, tile_dn
        )
    )
    ext.exl3_fat_moe_gather(
        xh, tg["row_token"], tg["row_expert"], ptrs["gate_suh"], h13, tg["num_rows"]
    )
    if bool(getattr(layer, "_exl3_shared_w13_suh", False)):
        h13u = h13
    else:
        ext.exl3_fat_moe_gather(
            xh, tg["row_token"], tg["row_expert"], ptrs["up_suh"], h13u, tg["num_rows"]
        )
    ext.exl3_fat_moe_gateup(
        h13,
        h13u,
        ptrs["gate_trellis"],
        ptrs["up_trellis"],
        ptrs["gate_svh"],
        ptrs["up_svh"],
        ptrs["down_suh"],
        h2,
        tg["seg_expert"],
        tg["seg_row0"],
        tg["seg_rows"],
        tg["num_segs"],
        float(limit),
        bits,
        cb,
    )
    ext.exl3_fat_moe_down(
        h2,
        ptrs["down_trellis"],
        ptrs["down_svh"],
        out,
        td["row_token"],
        td["row_weight"],
        td["seg_expert"],
        td["seg_row0"],
        td["seg_rows"],
        td["num_segs"],
        bits,
        cb,
    )
    _EXL3_FAT_DIAG["grouped_calls"] += 1


def build_exl3_fused_state(layer: torch.nn.Module, inners: list[dict[str, Any]]) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    import exllamav3_ext

    device = layer.w13_trellis.device
    n_exp = len(inners)
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    idx = int(device.index) if device.index is not None else 0
    concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
    if concurrency < 1:
        concurrency = 1
    # exl3_moe_x (GB10 pipeline shape, see moex_config()): temps hold one slot per launched group
    concurrency = max(concurrency, moex_config()[1])
    rows = temp_rows_fused()
    key = (str(device), hidden, intermediate, concurrency, rows)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, hidden), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
            torch.empty((concurrency, rows, intermediate), dtype=torch.float16, device=device),
        )
        _FUSED_TEMP_CACHE[key] = temps
        _EXL3_FAT_DIAG["fused_temps_allocs"] += 1
    # Layers share one cache entry, so assign (never accumulate) the bytes.
    _EXL3_FAT_DIAG["fused_temps_bytes"] = sum(
        t.numel() * t.element_size() for t in temps
    )
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    if moex_config()[0] >= 0:
        moex_locks(device)  # allocated here, never inside CUDA-graph capture
    gate0, up0, down0 = inners[0]["gate"], inners[0]["up"], inners[0]["down"]
    layer._exl3_k = int(gate0.K)
    layer._exl3_k_gate = int(gate0.K)
    layer._exl3_k_up = int(up0.K)
    layer._exl3_k_down = int(down0.K)
    layer._exl3_mcg = bool(gate0.mcg)
    layer._exl3_mul1 = bool(gate0.mul1)


# --- exl3_moe_x: the GB10-shaped exl3_moe (overlay/moex/) -----------------------------------
# DSV41_EXL3_MOE_X="<shape>[,<groups>,<group_size>]" routes every fused MoE launch through
# exl3_moe_x_ext, built into the image by overlay/moex/build_moe_x.py (or loaded from
# DSV41_EXL3_MOE_X_DIR). Same kernel body and inner GEMM as exllamav3's exl3_moe; only the
# cp.async depth, fragment stages, blocks per SM and group grid differ. Shape 2 = 32-K tile,
# 6 stages, 2 fragment stages, 2 blocks per SM: 1.4x per launch on GB10, identical output at
# group size 8. Compile-time K = 2/3/4 instances (a runtime-K switch spills). Unset = shipped kernel.
_MOEX = {"cfg": None, "mod": None, "locks": {}}


def moex_config() -> tuple[int, int, int]:
    """(shape, groups, group_size); shape -1 = off."""
    if _MOEX["cfg"] is None:
        raw = os.environ.get("DSV41_EXL3_MOE_X", "").strip()
        if not raw:
            _MOEX["cfg"] = (-1, 0, 8)
        else:
            parts = [int(p) for p in raw.split(",")]
            _MOEX["cfg"] = (parts[0], parts[1] if len(parts) > 1 else 12, parts[2] if len(parts) > 2 else 8)
    return _MOEX["cfg"]


def moex_module():
    if _MOEX["mod"] is None:
        try:
            import exl3_moe_x_ext  # built into the image
        except ImportError:
            d = os.environ.get("DSV41_EXL3_MOE_X_DIR", "/opt/dsv41/moex")
            if d not in sys.path:
                sys.path.insert(0, d)
            import exl3_moe_x_ext
        _MOEX["mod"] = exl3_moe_x_ext
        shape, g, gs = moex_config()
        for k in (2, 3, 4):
            vi = int(exl3_moe_x_ext.moex_variant_index(shape, k))
            logger.info(
                "exl3_moe_x engaged: shape=%s K=%s -> instance %s (%s) groups=%s group_size=%s "
                "[regs, smem, blocks/SM, block, SMs, local, maxthreads]=%s",
                shape, k, vi, exl3_moe_x_ext.moex_name(vi) if vi >= 0 else "none", g, gs,
                exl3_moe_x_ext.moex_info(vi) if vi >= 0 else None,
            )
    return _MOEX["mod"]


def moex_locks(device: torch.device) -> torch.Tensor:
    key = str(device)
    buf = _MOEX["locks"].get(key)
    if buf is None:
        mod = moex_module()
        buf = _MOEX["locks"][key] = torch.zeros(int(mod.moex_locks_ints()) + 4096, dtype=torch.int32, device=device)
    return buf


def _exl3_moe_launch(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    expert_count: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, torch.Tensor],
    layer: torch.nn.Module,
    limit: float,
    n_active_host: int | None,
) -> None:
    inners = layer._exl3_inners
    gate, up, down = inners[0]["gate"], inners[0]["up"], inners[0]["down"]
    args = (
        xh,
        out,
        expert_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        MOE_ACT_SILU,
        int(gate.K),
        int(up.K),
        int(down.K),
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        bool(gate.mcg),
        bool(gate.mul1),
        bool(up.mcg),
        bool(up.mul1),
        bool(down.mcg),
        bool(down.mul1),
        float(limit),
    )
    shape, groups, gsize = moex_config()
    if shape >= 0:
        moex_module().exl3_moe_x(*args, moex_locks(xh.device), shape, groups, gsize)
        return
    if n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)


def _exl3_moe_row_tiles(
    fn: Any,
    xh: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ptrs: dict[str, torch.Tensor],
    layer: torch.nn.Module,
    limit: float,
    n_active_host: int | None,
    max_rows: int,
) -> None:
    """Launch exl3_moe once per 128-row slice of the sorted expert-token buffer.

    Prefill-only (host syncs). Decode never reaches here: tokens <= temp rows.
    Kernel skips experts with count > temp rows; tiles keep every expert inside temps.
    """
    n_exp = int(counts.shape[0])
    device = counts.device
    tile = int(temps[0].shape[1])
    n_tiles = (max_rows + tile - 1) // tile
    prefix = torch.empty(n_exp, dtype=torch.long, device=device)
    prefix[0] = 0
    if n_exp > 1:
        prefix[1:] = counts[:-1].cumsum(0)
    for t in range(n_tiles):
        row0 = t * tile
        tile_counts = (counts - row0).clamp(min=0, max=tile)
        n_tile = int(tile_counts.sum().item())
        if n_tile == 0:
            continue
        cum = tile_counts.cumsum(0)
        idx = torch.arange(n_tile, device=device)
        expert_id = torch.searchsorted(cum, idx, right=True)
        local_row = idx - (cum[expert_id] - tile_counts[expert_id])
        src = prefix[expert_id] + row0 + local_row
        tile_ec = torch.zeros(n_exp + 1, dtype=torch.long, device=device)
        tile_ec[:n_exp] = tile_counts
        _exl3_moe_launch(
            fn,
            xh,
            out,
            tile_ec,
            token_sorted.index_select(0, src),
            weight_sorted.index_select(0, src),
            temps,
            ptrs,
            layer,
            limit,
            n_active_host,
        )


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor:
    """One exl3_moe launch per layer when tokens or hottest expert fit temp rows.

    Cap is temps dim1 (`EXL3_TEMP_ROWS_FUSED`, default 128). Overflow uses GPU
    row tiles if `EXL3_MOE_ROW_TILE=1`; otherwise fat experts use the highest
    enabled tier: kernel implies batched, batched implies sorted, then legacy.
    Decode (tokens ≤ cap) stays a single graph-safe launch.
    """
    import exllamav3_ext

    tokens, hidden = x2d.shape
    n_exp = len(inners)
    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(tokens, device=x2d.device, dtype=torch.long).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    # scatter_add stays on GPU. torch.bincount can host-stage and break CUDA graphs.
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(
        0, local.long(), torch.ones(local.shape, dtype=torch.long, device=local.device)
    )
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None
    # Actual kernel cap is the allocated temp dim1 (env-selected at load).
    cap = int(temps[0].shape[1])

    # Reset per call so a "kernel" label from an earlier prefill cannot
    # masquerade through later decode/thin/row-tile calls.
    layer._exl3_last_fat_fallback = "none"
    layer._exl3_last_fat_reason = "no_fat_experts"

    if tokens <= cap:
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, layer, limit, n_active_host,
        )
        return out

    # Prefill larger than temps. E1 copies routing counts on a side stream and
    # launches thin experts immediately, overlapping the D2H synchronization.
    # Decode never reaches here (capture sizes << cap).
    _EXL3_FAT_DIAG["prefill_layer_calls"] += 1
    use_row_tiles = fused_moe_row_tile_enabled()
    if (
        grouped_fat_enabled()
        and not use_row_tiles
        and getattr(layer, "_exl3_fat_effective_tier", None) == "grouped"
    ):
        # E3: thin experts in the fused kernel (it skips count > cap), every
        # fat expert in three grouped launches driven by device-side tables.
        # No host sync, so this branch is CUDA-graph capturable.
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, layer, limit, n_active_host,
        )
        apply_exl3_grouped_fat(
            xh, out, counts, token_sorted, weight_sorted, layer, cap, limit
        )
        _record_exl3_fat_tier(layer, "grouped", "grouped_ok")
        if fat_expert_log_enabled():
            record_exl3_fat_expert_stats(counts)
        return out
    want_fat_kernel = fat_kernel_enabled() or grouped_fat_enabled()
    want_batched_fat = batched_fat_fallback_enabled() or want_fat_kernel
    use_sorted_fat = sorted_fat_fallback_enabled() or want_batched_fat
    use_batched_fat = (
        want_batched_fat
        and bool(getattr(layer, "_exl3_shared_w13_suh", False))
    )
    use_fat_kernel = (
        use_batched_fat and want_fat_kernel and fat_gemm_eligible(layer)
    )
    launched = False
    counts_host = None
    if use_batched_fat and not use_row_tiles:
        counts_cpu, count_stream = _stage_counts_to_host(counts)
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, layer, limit, n_active_host,
        )
        launched = True
        count_stream.synchronize()
        counts_host = counts_cpu.tolist()
    elif use_sorted_fat:
        counts_host = counts.tolist()

    max_rows = (
        max(counts_host, default=0)
        if counts_host is not None
        else int(counts.max().item())
    )
    if fat_expert_log_enabled():
        record_exl3_fat_expert_stats(
            counts, max_rows=max_rows, counts_host=counts_host
        )
    if max_rows <= cap:
        _EXL3_FAT_DIAG["thin_calls"] += 1
        _record_exl3_fat_reason("thin_only")
        if not launched:
            _exl3_moe_launch(
                fn, xh, out, expert_count, token_sorted, weight_sorted,
                temps, ptrs, layer, limit, n_active_host,
            )
        return out

    if use_row_tiles:
        _EXL3_FAT_DIAG["row_tile_calls"] += 1
        layer._exl3_last_fat_fallback = "row_tile"
        layer._exl3_last_fat_reason = "row_tile_preempts_fat"
        _record_exl3_fat_reason("row_tile_preempts_fat")
        _exl3_moe_row_tiles(
            fn, xh, out, counts, token_sorted, weight_sorted,
            temps, ptrs, layer, limit, n_active_host, max_rows,
        )
        return out

    if not launched:
        _exl3_moe_launch(
            fn, xh, out, expert_count, token_sorted, weight_sorted,
            temps, ptrs, layer, limit, n_active_host,
        )
    if use_batched_fat:
        if use_fat_kernel:
            _record_exl3_fat_tier(layer, "kernel", "kernel_ok")
        else:
            _record_exl3_fat_tier(layer, "batched", "batched_ok")
        assert counts_host is not None
        apply_exl3_batched_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
            use_kernel=use_fat_kernel,
        )
    elif use_sorted_fat:
        _record_exl3_fat_tier(
            layer,
            "sorted",
            "degraded_shared_suh" if want_batched_fat else "sorted_ok",
        )
        assert counts_host is not None
        apply_exl3_sorted_fat(
            xh,
            token_sorted,
            weight_sorted,
            counts_host,
            inners,
            limit,
            cap,
            out,
        )
    else:
        _record_exl3_fat_tier(layer, "legacy", "legacy_default")
        fat = (counts > cap).nonzero(as_tuple=False).view(-1)
        if fat.numel():
            apply_exl3_python_loop(
                x2d,
                ids,
                weights,
                inners,
                expert_map,
                limit,
                only_experts=set(int(i) for i in fat.tolist()),
                out=out,
            )
            _EXL3_FAT_DIAG["fat_expert_runs"] += int(fat.numel())
    return out


def exl3_shape_key(handle: Any) -> tuple:
    """(in, out, K, mcg, mul1): everything exllamav3's GEMM autotune keys on besides rows."""
    return (
        int(handle.in_features),
        int(handle.out_features),
        int(handle.K),
        bool(getattr(handle, "mcg", False)),
        bool(getattr(handle, "mul1", False)),
    )


def iter_exl3_handles(module: torch.nn.Module):
    """Every LinearEXL3 handle the overlay attached under `module`."""
    for _name, layer in module.named_modules():
        inner = getattr(layer, "_exl3_inner", None)
        if inner is not None:
            yield inner
        for h in getattr(layer, "_exl3_inners", None) or ():
            if isinstance(h, dict):
                yield from h.values()
            else:
                yield h


def pretune_exl3_shapes(
    modules,
    sizes=(1, 2, 4, 8, 16),
    out_dtype: torch.dtype = torch.float32,
) -> tuple[int, int]:
    """Run each distinct EXL3 (in, out, K, codebook) shape once per row bucket.

    exllamav3 autotunes its cooperative trellis GEMM the first time it sees a
    (min(pow2(rows), 16), k, n, K, out dtype) shape, timing candidates with
    cudaStreamSynchronize. Inside CUDA-graph capture that sync is illegal
    ("operation not permitted when stream is capturing", coop_autotune.cu:464;
    boot 15 on 2026-09-12 died there) and at request time it is a ~100 ms stall
    per new shape. Touch every shape before capture instead: five row buckets
    cover all rows <= 144 (larger rows take the reconstruct + hgemm path).
    Returns (distinct shapes, launches)."""
    seen: dict[tuple, Any] = {}
    for module in modules:
        if module is None:
            continue
        for h in iter_exl3_handles(module):
            try:
                key = exl3_shape_key(h)
            except Exception:
                continue
            seen.setdefault(key, h)
    launches = 0
    for key, h in seen.items():
        dev = h.trellis.device
        for m in sizes:
            x = torch.zeros(int(m), key[0], device=dev, dtype=torch.float16)
            h.forward(x, {}, out_dtype=out_dtype)
            launches += 1
    if seen and torch.cuda.is_available():
        torch.cuda.synchronize()
    return len(seen), launches


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float = SWIGLU_LIMIT_DEFAULT,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)
    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "fused"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for suffix in EXL3_SUFFIXES:
        if tail == suffix or tail.endswith("_" + suffix):
            return suffix
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Packed EXL3 (mul1 or mcg) for DeepSeek-V4.1-Flash: experts + linears."""

    def __init__(
        self,
        bits: float | int = 2.9,
        codebook: str = "mul1",
        scope: str = "dsv41_packed",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.avg_bits = float(bits)
        self.bits = self.avg_bits
        self.codebook = str(codebook).lower()
        self.scope = str(scope)
        self.raw_config = dict(kwargs)
        if self.codebook not in ("mul1", "mcg"):
            raise ValueError(
                f"this overlay implements codebook=mul1|mcg; got {self.codebook!r}"
            )

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # LinearEXL3 uses CUDA >= Ampere; GB10 is SM121.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skip = {
            "bits",
            "avg_bits",
            "codebook",
            "scope",
            "quant_method",
            # per-tensor ledger is tens of MiB; keep it off the config object.
            "tensor_storage",
        }
        return cls(
            bits=float(config.get("avg_bits", config.get("bits", 2.9))),
            codebook=str(config.get("codebook", "mul1")),
            scope=str(config.get("scope", "dsv41_packed")),
            **{k: v for k, v in config.items() if k not in skip},
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self, prefix=prefix)
        # ParallelLMHead is not LinearBase; create_weights still uses the linear ABI
        # (K=hidden, N=vocab shard). Checkpoint stores packed head.* not head.weight.
        if isinstance(layer, ParallelLMHead):
            return Exl3LinearMethod(self, prefix=prefix)
        if isinstance(layer, LinearBase) and not _skip_exl3_linear(prefix):
            return Exl3LinearMethod(self, prefix=prefix)
        return None


def _skip_exl3_linear(prefix: str) -> bool:
    """Native (not packed) linears: vision, router, MTP markov head, indexer proj.

    The main `lm_head` is packed EXL3 (head_bits=6). ParallelLMHead is dispatched
    in get_quant_method before this skip list.
    """
    p = prefix.replace("-", ".").lower()
    if any(
        x in p
        for x in (
            "vision",
            "visual",
            "aligner",
            "markov_head",
            "confidence_head",
            "weights_proj",
        )
    ):
        return True
    if p == "embed" or p.endswith(".embed") or (".embed." in p and "engram" not in p):
        return True
    # Router gate is dense; compressor.wgate is packed EXL3.
    if p.endswith(".gate") or ".ffn.gate" in p or ".mlp.gate" in p:
        return True
    return False


# ----------------------------------------------------------------------------
# [glm53-dense-fp8] Optional FP8 weight-only (Marlin) path for the BF16 dense
# projections. GLM53_DENSE_FP8=dense,kda (default since 2026-09-09) | off | comma list of groups:
#   shared  mlp.shared_experts.{gate_up_proj,down_proj}
#   dense   mlp.{gate_up_proj,down_proj} of the dense-MLP layers
#   kda     self_attn.{in_proj_qkvbfg_a,f_b_proj,g_b_proj,o_proj} of KDA layers
#   mla     self_attn.{fused_qkv_a_proj,q_b_proj,o_proj} of MLA layers (kv_b_proj
#           stays BF16: MLA reads its weight directly for the absorbed matmuls)
# Weights load as BF16 exactly as today (all custom loaders untouched, ABLIT edits
# o_proj at the end of load_weights), then process_weights_after_loading quantizes
# per output channel to FP8 e4m3 and repacks for the Marlin kernel. KL panel (2026-09-09,
# scripts/quality/kl_panel.py, logs/quality-20260909): 0.005-0.017 nats/token vs BF16 on 57k
# tokens of code/prose, flat across 0-32k depth, argmax agreement 95-98 %; tool-calling unchanged.
# ----------------------------------------------------------------------------
_GLM53_DENSE_FP8_SUFFIXES = {
    "shared": (".mlp.shared_experts.gate_up_proj", ".mlp.shared_experts.down_proj"),
    "dense": (".mlp.gate_up_proj", ".mlp.down_proj"),
    "kda": (".self_attn.in_proj_qkvbfg_a", ".self_attn.f_b_proj", ".self_attn.g_b_proj", ".self_attn.o_proj"),
    "mla": (".self_attn.fused_qkv_a_proj", ".self_attn.q_b_proj", ".self_attn.o_proj"),
}


def _glm53_dense_fp8_groups() -> set[str]:
    # Off on this checkpoint: dense/attn are already packed EXL3, not BF16 Marlin.
    raw = os.environ.get("GLM53_DENSE_FP8", os.environ.get("DSV41_DENSE_FP8", "off")).strip().lower()
    if raw in ("", "off", "0", "no", "none"):
        return set()
    if raw in ("all", "on", "1"):
        return {"shared", "dense", "kda", "mla"}
    groups = {g.strip() for g in raw.split(",") if g.strip()}
    unknown = groups - set(_GLM53_DENSE_FP8_SUFFIXES)
    if unknown:
        raise ValueError(f"GLM53_DENSE_FP8: unknown group(s) {sorted(unknown)}")
    return groups


def _glm53_layer_types() -> list[str] | None:
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config().model_config.hf_text_config
        lt = getattr(cfg, "layer_types", None)
        return list(lt) if lt else None
    except Exception:  # noqa: BLE001
        return None


def _glm53_dense_fp8_group(prefix: str, groups: set[str] | None = None, layer_types: list[str] | None = None) -> str | None:
    """Group name if `prefix` (vLLM module path) is an allow-listed dense projection."""
    groups = _glm53_dense_fp8_groups() if groups is None else groups
    if not groups:
        return None
    if ".mtp" in prefix or "visual" in prefix or "draft" in prefix:
        return None
    m = re.search(r"\.layers\.(\d+)\.", prefix)
    layer_idx = int(m.group(1)) if m else None
    for group in ("shared", "dense", "kda", "mla"):
        if group not in groups:
            continue
        if not any(prefix.endswith(s) for s in _GLM53_DENSE_FP8_SUFFIXES[group]):
            continue
        if group == "dense" and ".shared_experts." in prefix:
            continue
        if group in ("kda", "mla"):
            lt = _glm53_layer_types() if layer_types is None else layer_types
            if lt is None or layer_idx is None or layer_idx >= len(lt):
                continue
            is_kda = lt[layer_idx] == "linear_attention"
            if (group == "kda") != is_kda:
                continue
        return group
    return None


class Glm53DenseFp8Method(UnquantizedLinearMethod):
    """BF16 weight at load time; per-output-channel FP8 e4m3 + Marlin at apply."""

    def __init__(self, group: str) -> None:
        super().__init__()
        self.group = group
        self.ready = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )

        w = layer.weight.data
        if w.dtype != torch.bfloat16 and w.dtype != torch.float16:
            raise RuntimeError(f"[glm53-dense-fp8] expected a BF16/FP16 weight, got {w.dtype} for {self.group}")
        n, k = w.shape
        assert n == layer.output_size_per_partition and k == layer.input_size_per_partition, (w.shape, layer)
        wf = w.float()
        scales = wf.abs().amax(dim=1).clamp(min=1e-12) / 448.0  # [N] per output channel
        fp8 = (wf / scales[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        del wf
        layer.orig_dtype = w.dtype
        layer.weight = torch.nn.Parameter(fp8, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scales.to(layer.orig_dtype), requires_grad=False)
        layer.weight_block_size = None
        prepare_fp8_layer_for_marlin(layer, size_k_first=False)
        layer.glm53_fp8_n, layer.glm53_fp8_k = n, k
        self.ready = True

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if not self.ready:
            return super().apply(layer, x, bias)
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
        )

        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.glm53_fp8_n,
            size_k=layer.glm53_fp8_k,
            bias=bias,
        )


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed mul1/mcg trellis experts: create/load packed tensors, LinearEXL3 apply."""

    def __init__(self, moe, quant_config: Exl3Config, prefix: str = "") -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.prefix = prefix
        self.k = k_for_prefix(prefix, quant_config.avg_bits)
        self.bits = self.k
        self.codebook = quant_config.codebook
        self._logged = False

    def get_fused_moe_quant_config(self, layer: "RoutedExperts") -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        if hidden_size % 16 or intermediate_size_per_partition % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_size_per_partition}"
            )
        k_words = int(self.k) * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_size_per_partition // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        # w13_* : stacked [expert, {gate=0, up=1}, ...] so the stock
        # expert_params_mapping (experts.w13_ + suffix) hits these names.
        w13_trellis = Parameter(
            torch.empty(
                num_experts, 2, in_tiles, out_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w13_suh = Parameter(
            torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w13_svh = Parameter(
            torch.empty(
                num_experts, 2, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w13_mul1 = Parameter(
            torch.empty(num_experts, 2, 1, dtype=torch.int32),
            requires_grad=False,
        )
        w2_trellis = Parameter(
            torch.empty(
                num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
            ),
            requires_grad=False,
        )
        w2_suh = Parameter(
            torch.empty(
                num_experts, intermediate_size_per_partition, dtype=torch.float16
            ),
            requires_grad=False,
        )
        w2_svh = Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float16),
            requires_grad=False,
        )
        w2_mul1 = Parameter(
            torch.empty(num_experts, 1, dtype=torch.int32),
            requires_grad=False,
        )

        packed = {
            "w13_trellis": w13_trellis,
            "w13_suh": w13_suh,
            "w13_svh": w13_svh,
            "w13_mul1": w13_mul1,
            "w2_trellis": w2_trellis,
            "w2_suh": w2_suh,
            "w2_svh": w2_svh,
            "w2_mul1": w2_mul1,
        }
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer
        # Alias so stock mappings that still look for *.mcg hit the mul1 marker.
        layer.w13_mcg = w13_mul1
        layer.w2_mcg = w2_mul1
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError("EXL3 create_weights must not allocate dense expert weights")

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_size_per_partition
        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        layer = param
        # param is the Parameter; expert_id is already physical. Map to local
        # via the owning module if present on the weight_loader closure... we
        # look up from param's __dict__ after register. RoutedExperts.weight_loader
        # maps global→local; glm5next calls *our* loader, so map here.
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        suffix = _suffix_from_mapped_name(weight_name)
        loaded = loaded_weight.detach()  # keep the mmap view; h2d_copy stages it

        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = shard_exl3_col(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id, shard_idx]
        elif shard_id == "w2":
            sharded = shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            dest = param.data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        if tuple(dest.shape) != tuple(sharded.shape):
            if dest.numel() == sharded.numel():
                h2d_copy(dest, sharded.reshape(dest.shape))
                return True if return_success else None
            raise RuntimeError(
                f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                f"expert={expert_id}: dest {tuple(dest.shape)} != "
                f"loaded {tuple(sharded.shape)}"
            )
        h2d_copy(dest, sharded)
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer, "w13_trellis"):
            return
        # Bind owner for any late loads; stitch LinearEXL3 handles.
        for name in (
            "w13_trellis",
            "w13_suh",
            "w13_svh",
            "w13_mul1",
            "w2_trellis",
            "w2_suh",
            "w2_svh",
            "w2_mul1",
        ):
            getattr(layer, name)._exl3_owner = layer

        marker = getattr(layer, "w13_mul1", None)
        if marker is None:
            marker = layer.w13_mcg
        marker2 = getattr(layer, "w2_mul1", None)
        if marker2 is None:
            marker2 = layer.w2_mcg
        want = (
            MUL1_MARKER_SIGNED_INT32
            if self.codebook == "mul1"
            else MCG_MARKER_SIGNED_INT32
        )
        if not torch.all(marker.reshape(-1) == want) or not torch.all(
            marker2.reshape(-1) == want
        ):
            raise RuntimeError(
                f"EXL3 {self.codebook} marker is not {want}; packed ABI mismatch"
            )

        n_exp = int(layer.w13_trellis.shape[0])
        layer._exl3_shared_w13_suh = bool(
            torch.equal(layer.w13_suh[:, 0], layer.w13_suh[:, 1])
        )
        cb = self.codebook
        inners: list[dict[str, Any]] = []
        for e in range(n_exp):
            gate = make_linear_exl3(
                layer.w13_trellis[e, 0],
                layer.w13_suh[e, 0],
                layer.w13_svh[e, 0],
                marker[e, 0],
                codebook=cb,
            )
            up = make_linear_exl3(
                layer.w13_trellis[e, 1],
                layer.w13_suh[e, 1],
                layer.w13_svh[e, 1],
                marker[e, 1],
                codebook=cb,
            )
            down = make_linear_exl3(
                layer.w2_trellis[e],
                layer.w2_suh[e],
                layer.w2_svh[e],
                marker2[e],
                codebook=cb,
            )
            inners.append({"gate": gate, "up": up, "down": down})
        layer._exl3_inners = inners
        layer._exl3_codebook = cb
        # Tier resolution needs the LinearEXL3 handles (K/mcg/mul1) for the
        # grouped eligibility check, so it runs once the inners exist.
        _record_exl3_fat_resolution(layer)
        fused_ok = False
        fused_err = None
        if fused_moe_enabled():
            try:
                import exllamav3_ext

                if hasattr(exllamav3_ext, "exl3_moe"):
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                else:
                    fused_err = "exllamav3_ext.exl3_moe missing"
            except Exception as exc:
                fused_err = repr(exc)
                layer._exl3_ptrs = None
        if not self._logged:
            if fused_ok:
                logger.info(
                    "EXL3 mul1 trellis engaged for routed experts: K=%s avg_bits=%s "
                    "mcg=%s mul1=%s experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=exl3_moe concurrency=%s "
                    "(no BF16 expert reconstruct at load; E3 fat kernels are K4/MCG-only)",
                    self.k,
                    self.quant_config.avg_bits,
                    int(bool(getattr(layer, "_exl3_mcg", False))),
                    int(bool(getattr(layer, "_exl3_mul1", False))),
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    getattr(layer, "_exl3_fused_concurrency", "?"),
                )
            else:
                logger.info(
                    "EXL3 mul1 trellis engaged for routed experts: K=%s avg_bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "fused_moe=python_loop (%s) "
                    "(no BF16 expert reconstruct at load)",
                    self.k,
                    self.quant_config.avg_bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    fused_err or "EXL3_FUSED_MOE=0",
                )
            self._logged = True

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        limit = getattr(self.moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT
        return apply_exl3_experts(
            x, topk_ids, topk_weights, layer, limit=float(limit)
        )


def _tp_rank_size() -> tuple[int, int]:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    try:
        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except (AssertionError, ValueError):
        return 0, 1


class Exl3LinearMethod(LinearMethodBase):
    """Packed EXL3 for attention / shared-expert / engram wkv linears."""

    def __init__(self, quant_config: Exl3Config, prefix: str = "") -> None:
        super().__init__()
        self.quant_config = quant_config
        self.prefix = prefix
        self.k = k_for_prefix(prefix, quant_config.avg_bits)
        self.codebook = quant_config.codebook

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}
        parts = [int(x) for x in output_partition_sizes]
        output_partition_size = int(sum(parts))
        if input_size_per_partition % 16 or output_partition_size % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"in_part={input_size_per_partition} out_part={output_partition_size} "
                f"prefix={self.prefix}"
            )
        k_words = int(self.k) * 16
        in_tiles = input_size_per_partition // 16
        out_tiles = output_partition_size // 16
        trellis = Parameter(
            torch.empty(in_tiles, out_tiles, k_words, dtype=torch.int16),
            requires_grad=False,
        )
        suh = Parameter(
            torch.empty(input_size_per_partition, dtype=torch.float16),
            requires_grad=False,
        )
        svh = Parameter(
            torch.empty(output_partition_size, dtype=torch.float16),
            requires_grad=False,
        )
        mul1 = Parameter(torch.empty((), dtype=torch.int32), requires_grad=False)
        packed = {"trellis": trellis, "suh": suh, "svh": svh, "mul1": mul1}
        # vLLM's LinearBase.create_weights call does not pass input_dim/output_dim.
        shard = "none"
        if extra.get("output_dim") is not None:
            shard = "col"
        elif extra.get("input_dim") is not None:
            shard = "row"
        elif input_size_per_partition < input_size:
            shard = "row"
        elif output_partition_size < output_size:
            shard = "col"
        for name, param in packed.items():
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra)
            param.weight_loader = self._load
            param._exl3_suffix = name
            param._exl3_shard = shard
            param._exl3_owner = layer
        layer.mcg = mul1
        layer._exl3_input_size = input_size_per_partition
        layer._exl3_output_size = output_partition_size
        layer._exl3_output_partition_sizes = parts
        layer._exl3_fused_stash = {}
        layer._exl3_k = self.k
        layer._exl3_codebook = self.codebook
        layer._exl3_prefix = self.prefix
        layer._exl3_shard = shard

    def _load(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | int | None = None,
        **kwargs: Any,
    ) -> None:
        weight_name = kwargs.get("weight_name")
        shard_id: int | None = None
        if isinstance(loaded_shard_id, int):
            shard_id = loaded_shard_id
        elif isinstance(loaded_shard_id, str):
            if loaded_shard_id.isdigit():
                shard_id = int(loaded_shard_id)
            else:
                weight_name = loaded_shard_id
        name = getattr(param, "_exl3_suffix", None)
        if not name:
            try:
                name = _suffix_from_mapped_name(str(weight_name or ""))
            except ValueError:
                name = "trellis"
                if param.dim() == 1:
                    name = "suh"
                elif param.dim() == 0:
                    name = "mul1"
        loaded = loaded_weight.detach()  # mmap view until staged
        tp_rank, tp_size = _tp_rank_size()
        shard = getattr(param, "_exl3_shard", "none")
        owner = getattr(param, "_exl3_owner", None)
        pack_suffix = "mcg" if name == "mul1" else name
        # wo_a is ColumnParallel over o_groups. Checkpoint packs one tensor
        # per group (slice.i); this rank owns a contiguous range of groups.
        n_local = getattr(owner, "bmm_batch_size", None) if owner is not None else None
        skip_tp = False
        if (
            shard_id is not None
            and shard == "col"
            and isinstance(n_local, int)
            and n_local > 0
        ):
            skip_tp = True
            if tp_size > 1:
                lo = tp_rank * n_local
                hi = lo + n_local
                if shard_id < lo or shard_id >= hi:
                    return
                shard_id = shard_id - lo
        if not skip_tp:
            if tp_size > 1 and shard == "col":
                loaded = shard_exl3_col(loaded, pack_suffix, tp_rank, tp_size)
            elif tp_size > 1 and shard == "row":
                loaded = shard_exl3_row(loaded, pack_suffix, tp_rank, tp_size)
        # Safetensors yields CPU (mmap) tensors; packed params live on the
        # CUDA device. copy_ is the cross-device path (no second CUDA copy);
        # set_ is not. Only the fused-stash path below needs a device copy.
        if name == "mul1":
            loaded = loaded.reshape(()).to(dtype=torch.int32)
        parts = list(getattr(owner, "_exl3_output_partition_sizes", None) or [])
        fused = owner is not None and (shard_id is not None or len(parts) > 1)
        if fused and loaded.device != param.device:
            loaded = h2d_alloc(loaded, param.device, dtype=param.dtype)
        if fused and owner is not None:
            if shard_id is None:
                shard_id = 0
            stash = owner._exl3_fused_stash.setdefault(int(shard_id), {})
            stash[name] = loaded
            # The stash is the source of truth for fused linears. The packed
            # Parameter created in create_weights (full merged size, on CUDA)
            # is never read again; drop its storage now so the merged linears
            # do not sit twice in UMA (attn/shared/MTP: a few GiB per rank).
            if param.dim() > 0 and param.numel() > 0:
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            return
        if param.dim() == 0:
            h2d_copy(param, loaded.reshape(()))
            return
        if name == "trellis" and tuple(param.shape) != tuple(loaded.shape):
            # K guessed from the map differs from the checkpoint (e.g. layer 29
            # shared down_proj is K=5): adopt the packed shape, on the device.
            param.set_(h2d_alloc(loaded, param.device, dtype=param.dtype))
            return
        if tuple(param.shape) != tuple(loaded.shape):
            raise RuntimeError(
                f"EXL3 linear load shape mismatch {weight_name}: "
                f"dest {tuple(param.shape)} != loaded {tuple(loaded.shape)} "
                f"(prefix={self.prefix} guessed K={self.k} suffix={name})"
            )
        h2d_copy(param, loaded)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        stash = getattr(layer, "_exl3_fused_stash", None) or {}
        parts = list(getattr(layer, "_exl3_output_partition_sizes", None) or [])
        if stash:
            inners = []
            n = len(parts) if len(parts) > 1 else (max(stash) + 1)
            for i in range(n):
                sh = stash.get(i)
                if not sh or not {"trellis", "suh", "svh"} <= set(sh):
                    raise RuntimeError(
                        f"EXL3 fused linear {self.prefix} missing packed shard {i}: "
                        f"have {sorted((sh or {}).keys())}"
                    )
                marker = sh.get("mul1", sh.get("mcg"))
                inners.append(
                    make_linear_exl3(
                        sh["trellis"],
                        sh["suh"],
                        sh["svh"],
                        marker,
                        codebook=self.codebook,
                    )
                )
            layer._exl3_inners = inners
            layer._exl3_inner = None
            layer._exl3_k = int(stash[0]["trellis"].shape[-1]) // 16
            if os.environ.get("DSV41_EXL3_LINEAR_LOG", "0") != "0":
                logger.info(
                    "EXL3 fused linear %s shards=%s K0=%s %s",
                    getattr(layer, "_exl3_prefix", self.prefix),
                    n,
                    layer._exl3_k,
                    self.codebook,
                )
            return
        marker = getattr(layer, "mul1", None)
        if marker is None:
            marker = layer.mcg
        layer._exl3_inner = make_linear_exl3(
            layer.trellis,
            layer.suh,
            layer.svh,
            marker,
            codebook=self.codebook,
        )
        layer._exl3_k = int(layer.trellis.shape[-1]) // 16
        if os.environ.get("DSV41_EXL3_LINEAR_LOG", "0") != "0":
            logger.info(
                "EXL3 linear %s K=%s %s in=%s out=%s",
                getattr(layer, "_exl3_prefix", self.prefix),
                layer._exl3_k,
                self.codebook,
                int(layer.suh.numel()),
                int(layer.svh.numel()),
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inners = getattr(layer, "_exl3_inners", None)
        inner = getattr(layer, "_exl3_inner", None)
        if inner is None and not inners:
            self.process_weights_after_loading(layer)
            inners = getattr(layer, "_exl3_inners", None)
            inner = layer._exl3_inner
        xh = x.contiguous().half()
        # exllamav3 outputs fp16/fp32 only: the decode kernel (run_alloc) and the
        # prefill path (reconstruct + ext.hgemm, "c must be float32 or float16")
        # both reject bf16. Ask for fp32 and cast to the model dtype.
        odt = torch.float32
        if (
            inners
            and getattr(layer, "is_bmm", False)
            and xh.dim() == 3
            and int(xh.shape[1]) == len(inners)
        ):
            # Grouped (bmm) linear such as V4.1 wo_a: one packed matrix per
            # local o_group, loaded from wo_a.slice.i.*. Group g of the input
            # [T, G, in] goes through inner g -> [T, G, out_g].
            out = torch.stack(
                [
                    h.forward(xh[:, g, :].contiguous(), {}, out_dtype=odt)
                    for g, h in enumerate(inners)
                ],
                dim=1,
            )
        elif inners:
            out = torch.cat(
                [h.forward(xh, {}, out_dtype=odt) for h in inners],
                dim=-1,
            )
        else:
            out = inner.forward(xh, {}, out_dtype=odt)
        out = out.to(dtype=x.dtype)
        if bias is not None:
            out = out + bias
        return out

