"""CPU stub tests of actual adapter dispatch; NOT Torch/CUDA integration proof."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch


def main():
    capturing = {"on": False}
    fake_torch = NS(
        float16="fp16", bfloat16="bf16", float32="fp32", int64="i64",
        cuda=NS(is_current_stream_capturing=lambda: capturing["on"]),
    )
    name = "runtime.py"
    spec = importlib.util.spec_from_file_location(
        "cooperative_moe_runtime", Path(__file__).with_name(name)
    )
    adapter = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, torch=fake_torch):
        spec.loader.exec_module(adapter)
    calls = []
    native_creations = []

    class Native:
        def __init__(self, device, root):
            self.device = device
            self.occupancy = {2: (1, 1, 1), 3: (2, 2, 2)}
            native_creations.append(device)

        def __call__(self, *args):
            calls.append("candidate")
            return "candidate"

    adapter.CoopLaunch = Native

    class Method:
        def process_weights_after_loading(self, layer):
            calls.append("process")
            return "loaded"

    def stock(*args):
        calls.append("stock")
        return "stock"

    module = NS(
        Exl3MoEMethod=Method,
        apply_exl3_fused_moe=stock,
        logger=NS(info=lambda *args: None),
    )
    with patch.dict(os.environ, {"DSV41_COOPERATIVE_MOE": "0"}):
        assert adapter.install(module) is False and module.apply_exl3_fused_moe is stock
    with patch.dict(
        os.environ, {"DSV41_COOPERATIVE_MOE": "1", "DSV41_EXL3_SERIAL_STREAMS": "0"}
    ):
        try:
            adapter.install(module)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unsafe stream configuration accepted")

    def layer(bits=3):
        packs = [
            {p: NS(K=bits, mul1=True, mcg=False) for p in ("gate", "up", "down")}
            for _ in range(6)
        ]
        return NS(
            _exl3_k_gate=bits,
            _exl3_k_up=bits,
            _exl3_k_down=bits,
            _exl3_mul1=True,
            _exl3_mcg=False,
            _exl3_hidden_size=5120,
            _exl3_intermediate_local=1152,
            _exl3_inners=packs,
            _exl3_ptrs={"gate_trellis": True},
            _exl3_fused_temps=(True,),
            w13_trellis=NS(device="cuda:0"),
        )

    invalid_layers = [
        ("_exl3_k_gate", 4),
        ("_exl3_k_up", 2),
        ("_exl3_k_down", 2),
        ("_exl3_mul1", False),
        ("_exl3_mcg", True),
        ("_exl3_hidden_size", 4096),
        ("_exl3_intermediate_local", 2304),
        ("_exl3_inners", []),
        ("_exl3_ptrs", None),
        ("_exl3_fused_temps", None),
    ]
    checks = 2
    for attr, value in invalid_layers:
        obj = layer()
        setattr(obj, attr, value)
        assert not adapter.layer_eligible(obj)
        checks += 1
    obj = layer()
    obj._exl3_inners[-1]["down"].K = 2
    assert not adapter.layer_eligible(obj)
    checks += 1
    with patch.dict(
        os.environ,
        {
            "DSV41_COOPERATIVE_MOE": "1",
            "DSV41_EXL3_SERIAL_STREAMS": "1",
            "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
        },
    ):
        assert adapter.install(module)
        try:
            adapter.install(module)
        except RuntimeError:
            pass
        else:
            raise AssertionError("double install accepted")

    def tensor(shape, dtype, device="cuda:0"):
        return NS(
            shape=shape, dtype=dtype, device=device, is_cuda=device.startswith("cuda")
        )

    for bits in (2, 3, 4):
        obj = layer(bits)
        assert Method().process_weights_after_loading(obj) == "loaded"
        assert hasattr(obj, "_dsv41_coop_native") == (bits in (2, 3))
        for rows in (0, 1, 2, 3, 4, 6, 8, 9, 12, 3072):
            for topk in (4, 6, 8):
                x = tensor((rows, 5120), "bf16")
                ids = tensor((rows, topk), "i64")
                rw = tensor((rows, topk), "fp32")
                result = module.apply_exl3_fused_moe(
                    x, ids, rw, obj, obj._exl3_inners, None, 10.0
                )
                expected = (
                    "candidate"
                    if bits in (2, 3) and 1 <= rows <= 8 and topk == 6
                    else "stock"
                )
                assert result == expected, (bits, rows, topk, result)
                checks += 1
    assert native_creations == [
        "cuda:0"
    ], "scratch not shared or created for unsupported layer"
    obj = layer()
    Method().process_weights_after_loading(obj)
    obj._exl3_mcg = True
    Method().process_weights_after_loading(obj)
    assert not hasattr(
        obj, "_dsv41_coop_native"
    ), "stale candidate after unsupported weight reload"
    checks += 1
    obj = layer()
    Method().process_weights_after_loading(obj)
    for changed in (
        "input_device",
        "input_dtype",
        "ids_device",
        "ids_dtype",
        "weights_device",
        "weights_shape",
        "limit",
        "hidden",
    ):
        x = tensor((3, 5120), "bf16")
        ids = tensor((3, 6), "i64")
        rw = tensor((3, 6), "fp32")
        limit = 10.0
        if changed == "input_device":
            x.device = "cpu"
            x.is_cuda = False
        elif changed == "input_dtype":
            x.dtype = "i64"
        elif changed == "ids_device":
            ids.device = "cuda:1"
        elif changed == "ids_dtype":
            ids.dtype = "i32"
        elif changed == "weights_device":
            rw.device = "cuda:1"
        elif changed == "weights_shape":
            rw.shape = (3, 5)
        elif changed == "limit":
            limit = float("nan")
        elif changed == "hidden":
            x.shape = (3, 4096)
        assert (
            module.apply_exl3_fused_moe(x, ids, rw, obj, obj._exl3_inners, None, limit)
            == "stock"
        ), changed
        checks += 1
    fresh = NS(
        Exl3MoEMethod=type(
            "FreshMethod", (), {"process_weights_after_loading": lambda *args: None}
        ),
        apply_exl3_fused_moe=stock,
        logger=NS(info=lambda *args: None),
    )
    with patch.dict(
        os.environ,
        {
            "DSV41_COOPERATIVE_MOE": "0",
            "DSV41_EXL3_SERIAL_STREAMS": "1",
            "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
        },
    ):
        assert adapter.install(fresh, enabled=True)
    checks += 1
    capturing["on"] = True
    for rows in (1, 2, 3, 4, 6, 8, 12, 18, 24):
        result = module.apply_exl3_fused_moe(
            tensor((rows, 5120), "bf16"), tensor((rows, 6), "i64"),
            tensor((rows, 6), "fp32"), obj, obj._exl3_inners, None, 10.0,
        )
        record = module._dsv41_coop_diag.capture_selection[rows]
        assert record["selected"] == (rows <= 8)
        assert record["kind"] == "capture"
        assert result == ("candidate" if rows <= 8 else "stock")
        checks += 1
    capturing["on"] = False
    assert module._dsv41_coop_diag.native_prepared
    assert module._dsv41_coop_diag.ineligible_reasons["gate_bits_4"] == 1
    assert module._dsv41_coop_diag.eager_selection[3072]["selected"] is False

    # A failed native call must propagate, never fall back after a partial launch.
    def failed_launch(*args):
        raise RuntimeError("partial launch")

    obj._dsv41_coop_native = NS(device="cuda:0")
    class FailedNative:
        device = "cuda:0"
        __call__ = failed_launch
    obj._dsv41_coop_native = FailedNative()
    before = calls.count("stock")
    try:
        module.apply_exl3_fused_moe(
            tensor((3, 5120), "bf16"), tensor((3, 6), "i64"),
            tensor((3, 6), "fp32"), obj, obj._exl3_inners, None, 10.0,
        )
    except RuntimeError as exc:
        assert str(exc) == "partial launch"
    else:
        raise AssertionError("failed native launch swallowed")
    assert calls.count("stock") == before
    checks += 1
    print(
        {
            "status": "pass",
            "checks": checks,
            "cpu_stub_only": True,
            "cuda_adapter_verified": False,
        }
    )


if __name__ == "__main__":
    main()
