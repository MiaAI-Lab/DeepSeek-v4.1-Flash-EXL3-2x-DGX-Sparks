"""Fixed-shape cooperative MoE adapter for the DS4.1 TP2 EXL3 overlay.

Requires the saved serialized EXL3-stream contract, like stock's shared fused
scratch. Only K2/K3 mul1 TP-shaped layers and <=8 physical decode rows qualify.
Prefill/unsupported layers remain stock. Allocate/prepare only after weight load,
never during graph capture. Disabled unless explicitly selected.

The three goal50_coop_* C symbols are the validated native ABI v1. Their names
are retained to load the existing verified binary without changing device code.
"""

import ctypes as C
import hashlib
import math
import os
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


def layer_eligible(layer):
    bits = getattr(layer, "_exl3_k_gate", None)
    inners = getattr(layer, "_exl3_inners", None)
    if (
        bits not in (2, 3)
        or getattr(layer, "_exl3_k_up", None) != bits
        or getattr(layer, "_exl3_k_down", None) != bits
        or not getattr(layer, "_exl3_mul1", False)
        or getattr(layer, "_exl3_mcg", True)
        or getattr(layer, "_exl3_hidden_size", None) != 5120
        or getattr(layer, "_exl3_intermediate_local", None) != 1152
        or not inners
        or not 1 <= len(inners) <= 384
        or not getattr(layer, "_exl3_ptrs", None)
        or getattr(layer, "_exl3_fused_temps", None) is None
    ):
        return False
    # A mixed projection/expert ABI cannot be inferred from the first expert.
    return all(
        int(pack[p].K) == bits and bool(pack[p].mul1) and not bool(pack[p].mcg)
        for pack in inners
        for p in ("gate", "up", "down")
    )


def call_eligible(x, ids, weights, layer, limit):
    native = getattr(layer, "_dsv41_coop_native", None)
    return (
        native is not None
        and len(x.shape) == 2
        and 1 <= x.shape[0] <= 8
        and x.shape[1] == 5120
        and tuple(ids.shape) == tuple(weights.shape) == (x.shape[0], 6)
        and x.is_cuda
        and ids.device == weights.device == x.device == native.device
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and ids.dtype == torch.int64
        and weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and math.isfinite(float(limit))
        and float(limit) >= 0
    )


class CoopLaunch:
    def __init__(self, device, library_root):
        self.device = device
        assert torch.cuda.get_device_capability(device) == (12, 1)
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "initialize before graph capture"
        path = Path(library_root) / "cooperative_moe.so"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256
        self.library = C.CDLL(str(path))
        assert self.library.goal50_coop_abi() == 1
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
        with torch.cuda.device(device):
            for bits in (2, 3):
                info = (C.c_int * 18)()
                assert self.library.goal50_coop_info(bits, 1, info) == 0
                assert info[16] == 851 and info[17] == 344
            self.scratch = [
                torch.empty(shape, dtype=dtype, device=device)
                for shape, dtype in (
                    ((48, 5120), torch.float16),
                    ((48, 5120), torch.float16),
                    ((48, 1152), torch.float16),
                    ((48, 1152), torch.float16),
                    ((48, 1152), torch.float16),
                    ((48, 5120), torch.float32),
                )
            ]
            self.counters = torch.zeros(851, dtype=torch.int32, device=device)

    def __call__(self, module, x, ids, weights, layer, inners, expert_map, limit):
        assert call_eligible(x, ids, weights, layer, limit)
        rows = int(x.shape[0])
        experts = len(inners)
        xh = x.contiguous().half()
        local = (
            module.map_topk_to_local(ids, experts, expert_map)
            .reshape(rows, 6)
            .contiguous()
        )
        rw = weights.to(dtype=torch.float16).contiguous()
        out = torch.empty((rows, 5120), dtype=torch.float32, device=x.device)
        tables = [layer._exl3_ptrs[k] for k in PTR_KEYS]
        assert all(
            t.device == x.device
            and t.dtype == torch.int64
            and t.is_contiguous()
            and tuple(t.shape) == (experts,)
            for t in tables
        )
        tensors = [xh, local, rw, *tables, *self.scratch, self.counters, out]
        pointers = (C.c_void_p * 20)(*[t.data_ptr() for t in tensors])
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
            raise RuntimeError(f"cooperative MoE kernel CUDA launch status {status}")
        layer._exl3_last_fat_fallback = "none"
        layer._exl3_last_fat_reason = "no_fat_experts"
        return out


def install(module, library_root="/root/.cache/vllm/cooperative_moe", *, enabled=False):
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if not enabled and os.environ.get("DSV41_COOPERATIVE_MOE", "0") != "1":
        return False
    if (
        os.environ.get("DSV41_EXL3_SERIAL_STREAMS") != "1"
        or os.environ.get("VLLM_DISABLE_SHARED_EXPERTS_STREAM") != "1"
    ):
        raise RuntimeError(
            "cooperative MoE requires the saved serialized EXL3/shared-expert stream configuration"
        )
    if getattr(module, "_dsv41_coop_installed", False):
        raise RuntimeError("cooperative MoE already installed")
    original_process = module.Exl3MoEMethod.process_weights_after_loading
    original_apply = module.apply_exl3_fused_moe
    kernels = {}

    def process(method, layer):
        result = original_process(method, layer)
        if hasattr(layer, "_dsv41_coop_native"):
            delattr(layer, "_dsv41_coop_native")
        if layer_eligible(layer):
            device = layer.w13_trellis.device
            key = str(device)
            if key not in kernels:
                kernels[key] = CoopLaunch(device, library_root)
            layer._dsv41_coop_native = kernels[key]
        return result

    def apply(x, ids, weights, layer, inners, expert_map, limit):
        if not call_eligible(x, ids, weights, layer, limit):
            return original_apply(x, ids, weights, layer, inners, expert_map, limit)
        return layer._dsv41_coop_native(
            module, x, ids, weights, layer, inners, expert_map, limit
        )

    module.Exl3MoEMethod.process_weights_after_loading = process
    module.apply_exl3_fused_moe = apply
    module._dsv41_coop_original_apply = original_apply
    module._dsv41_coop_installed = True
    module.logger.info(
        "Fixed-shape cooperative MoE enabled: K2/K3 mul1, 1..8 rows; other calls stay stock"
    )
    return True
