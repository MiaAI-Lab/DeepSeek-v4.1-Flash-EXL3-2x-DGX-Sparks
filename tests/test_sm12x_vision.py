#!/usr/bin/env python3
"""CPU-only regression tests; never load a model or use a GPU."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "overlay"), str(ROOT / "scripts")]
from patch_sm120_block64 import ATTN_HELPER, SWA_META_HELPER
from patch_sm12x_vision import apply, check_configuration
from build_vision_flashinfer import REVISION, SOURCES, unpack


def helper_value(source, flag=None, sm12=True):
    platforms = types.ModuleType("vllm.platforms")
    setattr(platforms, "current_platform", types.SimpleNamespace(is_device_capability_family=lambda value: sm12))
    scope = {}
    with patch.dict(sys.modules, {"vllm.platforms": platforms}), patch.dict(os.environ, {}, clear=True):
        if flag is not None:
            os.environ["DSV41_SM12X_VISION"] = flag
        exec(compile(source, "helper", "exec"), scope)
        return scope["_dsv41_text_only_swa"]()


class VisionTests(unittest.TestCase):
    def test_negative_baseline_proves_regression(self):
        for original in (ATTN_HELPER, SWA_META_HELPER):
            self.assertTrue(helper_value(original, "1"))
            self.assertFalse(helper_value(apply(original), "1"))

    def test_defaults_and_other_platforms_preserved(self):
        for original in (ATTN_HELPER, SWA_META_HELPER):
            fixed = apply(original)
            for flag in (None, "", "0", "invalid"):
                self.assertTrue(helper_value(fixed, flag))
            self.assertFalse(helper_value(fixed, "0", False))

    def test_idempotence_and_drift(self):
        for original in (ATTN_HELPER, SWA_META_HELPER):
            fixed = apply(original)
            self.assertEqual(apply(fixed), fixed)
            with self.assertRaises(ValueError):
                apply(fixed.replace('"DSV41_SM12X_VISION", "0"', '"DRIFT", "0"'))
        for source in ("x=1\n", SWA_META_HELPER + SWA_META_HELPER):
            with self.assertRaises(ValueError):
                apply(source)

    def test_budget_and_manifest_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            env = {"LANGUAGE_MODEL_ONLY": "0", "MAX_NUM_BATCHED_TOKENS": "2048"}
            with self.assertRaises(ValueError):
                check_configuration(env, receipt)
            receipt.write_text(json.dumps({"flashinfer_revision": REVISION}))
            check_configuration(env, receipt)
            for change in ({"MAX_NUM_BATCHED_TOKENS": "1024"}, {"LANGUAGE_MODEL_ONLY": "1"}):
                with self.assertRaises(ValueError):
                    check_configuration(env | change, receipt)
            receipt.write_text(json.dumps({"flashinfer_revision": "wrong"}))
            with self.assertRaises(ValueError):
                check_configuration(env, receipt)

    def test_submodule_unpack_preserves_populated_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "flashinfer/3rdparty/cutlass"
            destination.mkdir(parents=True)
            sentinel = root / "flashinfer/source.txt"
            sentinel.write_text("parent source must remain")
            archive = root / "submodule.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("submodule/include/header.h")
                member.size = 2
                bundle.addfile(member, io.BytesIO(b"ok"))
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            unpack(archive, destination, digest)
            self.assertEqual((destination / "include/header.h").read_text(), "ok")
            self.assertEqual(sentinel.read_text(), "parent source must remain")
            with self.assertRaises(OSError):
                unpack(archive, destination, digest)
            self.assertEqual((destination / "include/header.h").read_text(), "ok")

    def test_pins_and_archive_gate(self):
        self.assertEqual(len(SOURCES), 4)
        for _, revision, digest, _ in SOURCES:
            self.assertRegex(revision, r"^[0-9a-f]{40}$")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                item = tarfile.TarInfo("source/value.txt")
                item.size = 2
                bundle.addfile(item, io.BytesIO(b"ok"))
            with self.assertRaises(ValueError):
                unpack(archive, root / "out", "0" * 64)
            self.assertFalse((root / "out").exists())
            unpack(archive, root / "out", hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual((root / "out/value.txt").read_text(), "ok")

    def test_both_rank_launchers_forward_opt_in(self):
        source = (ROOT / "start.sh").read_text()
        self.assertEqual(source.count("python3 /opt/dsv41/patch_sm12x_vision.py --check"), 2)
        self.assertIn("LANGUAGE_MODEL_ONLY DSV41_SM12X_VISION SKIP_MM_PROFILING", source)
        self.assertIn('-e DSV41_SM12X_VISION="$DSV41_SM12X_VISION"', source)
        self.assertIn('DSV41_SM12X_VISION="${DSV41_SM12X_VISION:-0}"', source)
        self.assertIn('_cli_vision="${DSV41_SM12X_VISION-}"', source)


if __name__ == "__main__":
    unittest.main()
