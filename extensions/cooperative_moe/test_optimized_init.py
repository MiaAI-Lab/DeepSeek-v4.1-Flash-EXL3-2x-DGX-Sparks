"""CPU-only native-init regressions, including required calls under python -O.

Adapted from GLM-5.3-Flash PR #202; tests both DS4.1 K2 and K3 initialization.
Run normally AND with python -O. unittest checks are not optimized away.
"""

import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class Device:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class NativeInitTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = Path(workspace.name)
        self.binary = self.root / "cooperative_moe.so"
        self.binary.write_bytes(b"native fixture")
        self.calls = []
        self.allocations = []
        self.capability = (12, 1)
        self.capturing = False
        self.abi = 1
        self.bad_bits = None
        self.info_status = 0
        self.ctr_len = 851
        self.params_size = 344
        self.occupancy = 2
        owner = self

        class Library:
            def __init__(self, path):
                owner.calls.append("cdll")
                self.goal50_coop_launch = NS(argtypes=None, restype=None)

            def goal50_coop_abi(self):
                owner.calls.append("abi")
                return owner.abi

            def goal50_coop_info(self, bits, geometry, info):
                owner.calls.append(("info", bits, geometry))
                bad = owner.bad_bits is None or owner.bad_bits == bits
                info[4] = info[9] = info[14] = owner.occupancy if bad else 2
                info[16] = owner.ctr_len if bad else 851
                info[17] = owner.params_size if bad else 344
                return owner.info_status if bad else 0

        def allocate(*args, **kwargs):
            self.allocations.append((args, kwargs))
            return "tensor"

        torch = NS(
            float16="fp16", bfloat16="bf16", float32="fp32", int64="i64", int32="i32",
            cuda=NS(
                get_device_capability=lambda device: self.capability,
                is_current_stream_capturing=lambda: self.capturing,
                device=lambda device: Device(),
            ),
            empty=allocate, zeros=allocate,
        )
        spec = importlib.util.spec_from_file_location(
            "coop_runtime_test", Path(__file__).with_name("runtime.py")
        )
        self.adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, torch=torch):
            spec.loader.exec_module(self.adapter)
        self.adapter.SHA256 = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        loader = patch.object(ctypes, "CDLL", Library)
        loader.start()
        self.addCleanup(loader.stop)

    def launch(self):
        return self.adapter.CoopLaunch("cuda:0", self.root)

    def test_both_quantizations_prepared_before_scratch(self):
        launch = self.launch()
        self.assertEqual(self.calls, ["cdll", "abi", ("info", 2, 1), ("info", 3, 1)])
        self.assertEqual(launch.occupancy, {2: (2, 2, 2), 3: (2, 2, 2)})
        self.assertEqual(len(self.allocations), 7)
        self.assertEqual(self.allocations[-1][0], (851,))

    def test_wrong_device_rejected_before_loading(self):
        self.capability = (12, 0)
        with self.assertRaisesRegex(self.adapter.CooperativeMoEError, "SM121"):
            self.launch()
        self.assertEqual(self.calls, [])

    def test_capture_rejected_before_loading(self):
        self.capturing = True
        with self.assertRaisesRegex(self.adapter.CooperativeMoEError, "before graph capture"):
            self.launch()
        self.assertEqual(self.calls, [])

    def test_bad_hash_rejected_before_loading(self):
        self.binary.write_bytes(b"unvalidated binary")
        with self.assertRaisesRegex(self.adapter.CooperativeMoEError, "digest"):
            self.launch()
        self.assertEqual(self.calls, [])

    def test_bad_abi_rejected_before_scratch(self):
        self.abi = 2
        with self.assertRaisesRegex(self.adapter.CooperativeMoEError, "abi"):
            self.launch()
        self.assertEqual(self.allocations, [])

    def test_each_quantization_status_layout_and_occupancy(self):
        for bits in (2, 3):
            for field, value, message in (
                ("info_status", 1, "failed status"),
                ("ctr_len", 3587, "layout mismatch"),
                ("params_size", 1, "layout mismatch"),
                ("occupancy", 0, "occupancy"),
            ):
                with self.subTest(bits=bits, field=field):
                    self.bad_bits = bits
                    old = getattr(self, field)
                    setattr(self, field, value)
                    with self.assertRaisesRegex(self.adapter.CooperativeMoEError, message):
                        self.launch()
                    self.assertEqual(self.allocations, [])
                    setattr(self, field, old)

    def test_shared_scratch_guards(self):
        safe = {
            "DSV41_EXL3_SERIAL_STREAMS": "1",
            "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "1",
        }
        with patch.dict(os.environ, safe, clear=True):
            self.adapter.enforce_serialized_execution("")
            for key in self.adapter._OVERLAP_ENV:
                for value in ("1", "2", "true", "yes"):
                    with self.subTest(key=key, value=value), patch.dict(os.environ, {key: value}):
                        with self.assertRaisesRegex(self.adapter.CooperativeMoEError, "cannot overlap"):
                            self.adapter.enforce_serialized_execution("")
                for value in ("", "0", "off", "false", "no"):
                    with patch.dict(os.environ, {key: value}):
                        self.adapter.enforce_serialized_execution("")
            for arg in ("--enable-dbo", "--ubatch=2", "--enable-ubatching", "--dual-batch-overlap"):
                with self.subTest(arg=arg), patch.dict(os.environ, EXTRA_ARGS=arg):
                    with self.assertRaisesRegex(self.adapter.CooperativeMoEError, "cannot overlap"):
                        self.adapter.enforce_serialized_execution()
            for key in safe:
                with patch.dict(os.environ, {key: "0"}):
                    with self.assertRaisesRegex(self.adapter.CooperativeMoEError, "serialized"):
                        self.adapter.enforce_serialized_execution("")


if __name__ == "__main__":
    unittest.main()
