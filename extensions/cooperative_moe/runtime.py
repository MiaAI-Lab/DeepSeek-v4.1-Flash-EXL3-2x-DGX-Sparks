"""Fixed-shape cooperative MoE adapter for the DS4.1 TP2 EXL3 overlay.

Requires the saved serialized EXL3-stream contract, like stock's shared fused
scratch. Only K2/K3 mul1 TP-shaped layers and <=8 physical decode rows qualify.
Prefill/unsupported layers remain stock. Allocate/prepare only after weight load,
never during graph capture. Disabled unless explicitly selected.

Shared-scratch contract: install() caches one CoopLaunch per device and attaches
it to every eligible layer. The workspace is reused across layers and invocations.
Overlapping cooperative launches on different CUDA streams are unsupported, so
ubatching and dual-batch overlap are rejected at configuration time.

The three goal50_coop_* C symbols are the validated native ABI v1. Their names
are retained to load the existing verified binary without changing device code.
"""

import ctypes as C
import hashlib
import math
import os
import re
from pathlib import Path

import torch

SHA256 = "a09a589cbdcecb5372991c7b091d732236d58bc5f5aea14ab91e38e426f08d78"
PTR_KEYS = (
    "gate_trellis",
    "gate_suh",
    "gate_svh",
    "up_trellis",
    "up_suh",
    "up_svh",
    "down_trellis",
    "down_suh",
    "down_svh",
)
HIDDEN = 5120
INTERMEDIATE_LOCAL = 1152
TOPK = 6
ROWS_MAX = 8
EXPERTS_MAX = 384
CTR_LEN = 851
PARAMS_SIZE = 344
_OVERLAP_ENV = (
    "VLLM_UBATCH",
    "VLLM_UBATCHING",
    "VLLM_ENABLE_UBATCHING",
    "VLLM_USE_UBATCHING",
    "VLLM_V1_UBATCH",
    "VLLM_V1_ENABLE_UBATCHING",
    "VLLM_DUAL_BATCH",
    "VLLM_DUAL_BATCH_OVERLAP",
    "VLLM_ENABLE_DUAL_BATCH",
    "VLLM_ENABLE_DBO",
    "VLLM_DBO",
)
_OVERLAP_ARG_RE = re.compile(
    r"(--ubatch(?:ing)?(?:\b|=)|--enable-ubatch(?:ing)?\b|"
    r"--dual-batch(?:-overlap)?\b|--enable-dbo\b)",
    re.IGNORECASE,
)


class CooperativeMoEError(RuntimeError):
    """Required cooperative-MoE initialization or configuration failure."""


class CoopDiagnostics:
    """Host-side selection log; capture counts are not graph-replay counts."""

    def __init__(self):
        self.native_prepared = False
        self.eligible_layers = 0
        self.ineligible_layers = 0
        self.ineligible_reasons = {}
        self.capture_selection = {}
        self.eager_selection = {}
        self.summary_logged = False
        self.seen_selection = set()


def _require(condition, message):
    if not condition:
        raise CooperativeMoEError(message)


def enforce_serialized_execution(extra_args=None):
    """Reject configurations that can overlap launches on shared scratch."""
    if (
        os.environ.get("DSV41_EXL3_SERIAL_STREAMS") != "1"
        or os.environ.get("VLLM_DISABLE_SHARED_EXPERTS_STREAM") != "1"
    ):
        raise CooperativeMoEError(
            "cooperative MoE requires the saved serialized EXL3/shared-expert "
            "stream configuration"
        )
    for key in _OVERLAP_ENV:
        value = os.environ.get(key)
        if value is None or str(value).strip() == "":
            continue
        if str(value).strip().lower() in ("0", "off", "false", "no"):
            continue
        raise CooperativeMoEError(
            "cooperative MoE reuses one per-device scratch workspace and "
            f"cannot overlap launches; unsupported {key}={value!r}"
        )
    extra = os.environ.get("EXTRA_ARGS", "") if extra_args is None else extra_args
    if extra and _OVERLAP_ARG_RE.search(str(extra)):
        raise CooperativeMoEError(
            "cooperative MoE reuses one per-device scratch workspace and "
            f"cannot overlap launches; unsupported EXTRA_ARGS={extra!r}"
        )


def layer_ineligible_reason(layer):
    bits = getattr(layer, "_exl3_k_gate", None)
    inners = getattr(layer, "_exl3_inners", None)
    if bits not in (2, 3):
        return f"gate_bits_{bits}"
    if getattr(layer, "_exl3_k_up", None) != bits:
        return "up_bits"
    if getattr(layer, "_exl3_k_down", None) != bits:
        return "down_bits"
    if not getattr(layer, "_exl3_mul1", False):
        return "mul1_disabled"
    if getattr(layer, "_exl3_mcg", True):
        return "mcg_enabled"
    if getattr(layer, "_exl3_hidden_size", None) != HIDDEN:
        return "hidden"
    if getattr(layer, "_exl3_intermediate_local", None) != INTERMEDIATE_LOCAL:
        return "intermediate_local"
    if not inners:
        return "no_inners"
    if not 1 <= len(inners) <= EXPERTS_MAX:
        return f"expert_count_{len(inners)}"
    if not getattr(layer, "_exl3_ptrs", None):
        return "no_ptrs"
    if getattr(layer, "_exl3_fused_temps", None) is None:
        return "no_fused_temps"
    if not all(
        int(pack[p].K) == bits and bool(pack[p].mul1) and not bool(pack[p].mcg)
        for pack in inners
        for p in ("gate", "up", "down")
    ):
        return "mixed_expert_quant"
    return None


def layer_eligible(layer):
    return layer_ineligible_reason(layer) is None


def call_ineligible_reason(x, ids, weights, layer, limit):
    native = getattr(layer, "_dsv41_coop_native", None)
    if native is None:
        return "native_unattached"
    if len(getattr(x, "shape", ())) != 2:
        return "input_rank"
    rows = int(x.shape[0])
    if not 1 <= rows <= ROWS_MAX:
        return "rows_out_of_range"
    if x.shape[1] != HIDDEN:
        return "hidden_mismatch"
    if tuple(ids.shape) != (rows, TOPK) or tuple(weights.shape) != (rows, TOPK):
        return "route_shape"
    if not getattr(x, "is_cuda", False):
        return "input_not_cuda"
    if ids.device != native.device or weights.device != native.device or x.device != native.device:
        return "device_mismatch"
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "input_dtype"
    if ids.dtype != torch.int64:
        return "ids_dtype"
    if weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "weights_dtype"
    try:
        finite_limit = math.isfinite(float(limit)) and float(limit) >= 0
    except (TypeError, ValueError):
        finite_limit = False
    if not finite_limit:
        return "limit_invalid"
    return None


def call_eligible(x, ids, weights, layer, limit):
    return call_ineligible_reason(x, ids, weights, layer, limit) is None


class CoopLaunch:
    def __init__(self, device, library_root):
        self.device = device
        capability = torch.cuda.get_device_capability(device)
        _require(
            capability == (12, 1),
            f"cooperative MoE requires SM121, got {capability}",
        )
        _require(
            not torch.cuda.is_current_stream_capturing(),
            "initialize cooperative MoE before graph capture",
        )
        path = Path(library_root) / "cooperative_moe.so"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(digest == SHA256, f"unvalidated cooperative_moe.so digest {digest}")
        self.library = C.CDLL(str(path))
        abi = int(self.library.goal50_coop_abi())
        _require(abi == 1, f"unexpected goal50_coop_abi {abi}")
        self.launch = self.library.goal50_coop_launch
        self.launch.argtypes = [
            C.POINTER(C.c_void_p),
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_float,
            C.c_int,
            C.c_int,
            C.c_void_p,
        ]
        self.launch.restype = C.c_int
        self.occupancy = {}
        with torch.cuda.device(device):
            for bits in (2, 3):
                info = (C.c_int * 18)()
                status = int(self.library.goal50_coop_info(bits, 1, info))
                _require(status == 0, f"goal50_coop_info K{bits} failed status={status}")
                _require(
                    info[16] == CTR_LEN and info[17] == PARAMS_SIZE,
                    f"native layout mismatch K{bits} ctr={info[16]} params={info[17]}",
                )
                occupancy = (int(info[4]), int(info[9]), int(info[14]))
                _require(
                    all(value >= 1 for value in occupancy),
                    f"cooperative K{bits} kernel occupancy < 1: {occupancy}",
                )
                self.occupancy[bits] = occupancy
            self.scratch = [
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in (
                    ((48, HIDDEN), torch.float16),
                    ((48, HIDDEN), torch.float16),
                    ((48, INTERMEDIATE_LOCAL), torch.float16),
                    ((48, INTERMEDIATE_LOCAL), torch.float16),
                    ((48, INTERMEDIATE_LOCAL), torch.float16),
                    ((48, HIDDEN), torch.float32),
                )
            ]
            self.counters = torch.zeros(CTR_LEN, dtype=torch.int32, device=device)

    def __call__(self, module, x, ids, weights, layer, inners, expert_map, limit):
        _require(
            call_eligible(x, ids, weights, layer, limit),
            "cooperative launch preconditions failed",
        )
        rows = int(x.shape[0])
        experts = len(inners)
        xh = x.contiguous().half()
        local = (
            module.map_topk_to_local(ids, experts, expert_map)
            .reshape(rows, TOPK)
            .contiguous()
        )
        rw = weights.to(dtype=torch.float16).contiguous()
        out = torch.empty((rows, HIDDEN), dtype=torch.float32, device=x.device)
        ptrs = layer._exl3_ptrs
        _require(
            all(key in ptrs for key in PTR_KEYS),
            "cooperative pointer table mapping is incomplete",
        )
        tables = [ptrs[key] for key in PTR_KEYS]
        _require(
            all(
                table.device == x.device
                and table.dtype == torch.int64
                and table.is_contiguous()
                and tuple(table.shape) == (experts,)
                for table in tables
            ),
            "cooperative pointer tables are not contiguous int64 expert vectors",
        )
        tensors = [xh, local, rw, *tables, *self.scratch, self.counters, out]
        pointers = (C.c_void_p * 20)(*[tensor.data_ptr() for tensor in tensors])
        status = self.launch(
            pointers,
            int(layer._exl3_k_gate),
            rows,
            experts,
            float(limit),
            1,
            0,
            C.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            # Never retry stock after a partially launched CUDA operation.
            raise CooperativeMoEError(
                f"cooperative MoE kernel CUDA launch status {status}"
            )
        layer._exl3_last_fat_fallback = "none"
        layer._exl3_last_fat_reason = "no_fat_experts"
        return out


def _log_selection(module, diag, rows, selected, reason, capturing):
    kind = "capture" if capturing else "eager"
    key = (kind, int(rows), bool(selected), str(reason))
    if key in diag.seen_selection:
        return
    diag.seen_selection.add(key)
    record = {"selected": bool(selected), "reason": reason, "kind": kind}
    store = diag.capture_selection if capturing else diag.eager_selection
    store[int(rows)] = record
    module.logger.info(
        "cooperative MoE %s-time selection rows=%s selected=%s reason=%s; "
        "this is not a CUDA-graph replay count"
        % (kind, rows, bool(selected), reason)
    )


def install(module, library_root="/root/.cache/vllm/cooperative_moe", *, enabled=False):
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if not enabled and os.environ.get("DSV41_COOPERATIVE_MOE", "0") != "1":
        return False
    if getattr(module, "_dsv41_coop_installed", False):
        raise RuntimeError("cooperative MoE already installed")
    enforce_serialized_execution()
    original_process = module.Exl3MoEMethod.process_weights_after_loading
    original_apply = module.apply_exl3_fused_moe
    kernels = {}
    diag = CoopDiagnostics()

    def process(method, layer):
        result = original_process(method, layer)
        if hasattr(layer, "_dsv41_coop_native"):
            delattr(layer, "_dsv41_coop_native")
        reason = layer_ineligible_reason(layer)
        if reason is None:
            device = layer.w13_trellis.device
            key = str(device)
            if key not in kernels:
                kernels[key] = CoopLaunch(device, library_root)
                diag.native_prepared = True
                occupancy = kernels[key].occupancy
                module.logger.info(
                    "cooperative MoE native prepared: "
                    "K2 occupancy_a/b/rot=%s/%s/%s "
                    "K3 occupancy_a/b/rot=%s/%s/%s library=%s"
                    % (*occupancy[2], *occupancy[3], library_root)
                )
            layer._dsv41_coop_native = kernels[key]
            diag.eligible_layers += 1
        else:
            diag.ineligible_layers += 1
            diag.ineligible_reasons[reason] = diag.ineligible_reasons.get(reason, 0) + 1
        return result

    def apply(x, ids, weights, layer, inners, expert_map, limit):
        if not diag.summary_logged:
            diag.summary_logged = True
            module.logger.info(
                "cooperative MoE prepared-layer summary: eligible=%s ineligible=%s "
                "ineligible_reasons=%s native_prepared=%s"
                % (
                    diag.eligible_layers,
                    diag.ineligible_layers,
                    diag.ineligible_reasons or "{}",
                    diag.native_prepared,
                )
            )
        capturing = bool(torch.cuda.is_current_stream_capturing())
        reason = call_ineligible_reason(x, ids, weights, layer, limit)
        selected = reason is None
        rows = int(getattr(x, "shape", [0])[0]) if getattr(x, "shape", None) else 0
        _log_selection(
            module,
            diag,
            rows,
            selected,
            "cooperative" if selected else reason,
            capturing,
        )
        if not selected:
            return original_apply(x, ids, weights, layer, inners, expert_map, limit)
        return layer._dsv41_coop_native(
            module, x, ids, weights, layer, inners, expert_map, limit
        )

    module.Exl3MoEMethod.process_weights_after_loading = process
    module.apply_exl3_fused_moe = apply
    module._dsv41_coop_original_apply = original_apply
    module._dsv41_coop_installed = True
    module._dsv41_coop_diag = diag
    module.logger.info(
        "Fixed-shape cooperative MoE wrappers installed: K2/K3 mul1, 1..8 rows; "
        "native prepare runs after weight load; other calls stay stock. "
        "Shared scratch must not overlap."
    )
    return True
