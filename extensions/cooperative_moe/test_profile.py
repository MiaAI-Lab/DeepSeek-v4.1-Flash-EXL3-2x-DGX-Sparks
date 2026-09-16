"""CPU-only checks for profile integrity, path quoting, and exclusive creation."""

import ast
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import prepare_profile


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        root = Path(self.workspace.name)
        self.stock = root / "stock.py"
        self.artifacts = root / "artifacts"
        self.artifacts.mkdir()
        self.output = root / "selected.py"
        self.stock.write_bytes(b"sentinel = True\n")
        (self.artifacts / "cooperative_moe.so").write_bytes(b"native fixture")
        (self.artifacts / "runtime.py").write_bytes(b"adapter fixture")
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        pins = patch.multiple(
            prepare_profile,
            STOCK_SHA=digest(self.stock),
            BINARY_SHA=digest(self.artifacts / "cooperative_moe.so"),
            ADAPTER_SHA=digest(self.artifacts / "runtime.py"),
        )
        pins.start()
        self.addCleanup(pins.stop)

    def generate(self, runtime="/root/.cache/vllm/cooperative_moe"):
        prepare_profile.make_profile(self.stock, self.artifacts, runtime, self.output)

    def execute_footer(self, runtime):
        self.generate(runtime)
        module = types.ModuleType("profile_fixture")
        install = Mock()
        with (
            patch.dict(sys.modules, {module.__name__: module}),
            patch("runpy.run_path", return_value={"install": install}) as loader,
        ):
            exec(
                compile(self.output.read_bytes(), str(self.output), "exec"),
                module.__dict__,
            )
        self.assertTrue(module.sentinel)
        loader.assert_called_once_with(runtime + "/runtime.py")
        install.assert_called_once_with(module, library_root=runtime, enabled=True)

    def test_verified_profile_selects_explicit_artifacts(self):
        self.execute_footer("/root/.cache/vllm/cooperative_moe")

    def test_quoted_path_is_data_not_python(self):
        self.execute_footer("/runtime/contains'quote")

    def test_existing_output_is_not_overwritten(self):
        self.generate()
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(self.output.read_bytes(), original)

    def test_modified_stock_is_rejected(self):
        self.stock.write_bytes(b"different stock")
        with self.assertRaises(ValueError):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_modified_binary_is_rejected(self):
        (self.artifacts / "cooperative_moe.so").write_bytes(b"different binary")
        with self.assertRaises(ValueError):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_modified_adapter_is_rejected(self):
        (self.artifacts / "runtime.py").write_bytes(b"different adapter")
        with self.assertRaises(ValueError):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_invalid_runtime_paths_are_rejected(self):
        for path in ("relative", "../relative", "/runtime/../parent"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.generate(path)
            self.assertFalse(self.output.exists())


class SourcePinTests(unittest.TestCase):
    def test_checked_in_adapter_matches_profile_pin(self):
        root = Path(__file__).resolve().parent
        runtime = root / "runtime.py"
        self.assertEqual(
            hashlib.sha256(runtime.read_bytes()).hexdigest(),
            prepare_profile.ADAPTER_SHA,
        )
        assignments = {
            node.targets[0].id: node.value
            for node in ast.parse(runtime.read_text()).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }
        self.assertEqual(ast.literal_eval(assignments["SHA256"]), prepare_profile.BINARY_SHA)


if __name__ == "__main__":
    unittest.main()
