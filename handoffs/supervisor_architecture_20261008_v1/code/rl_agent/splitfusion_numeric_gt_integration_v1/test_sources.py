"""Source pinning uses verified bytes, not repeated per-frame disk reads."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .contract import ContractError, sha256
from .sources import VerifiedSources, verify_sources_once


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="numeric-gt-source-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = b"MARK = 5\ndef f(x):\n    return x + 1\n"
        (self.root / "example.py").write_bytes(self.raw)
        self.pins = {"original": ("example.py", sha256(self.raw))}

    def test_verified_original_bytes_are_immutable_and_compile_without_reads(self):
        bundle = verify_sources_once(self.root, self.pins)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no live read")):
            ns = {}; exec(bundle.compile("original"), ns)
            only = {}; exec(bundle.compile("original", functions=("f",)), only)
        self.assertEqual(ns["MARK"], 5)
        self.assertEqual(only["f"](4), 5)
        self.assertNotIn("MARK", only)
        self.assertEqual(bundle.source("original").raw, self.raw)
        self.assertEqual(bundle.verify_unchanged()["example.py"], sha256(self.raw))

    def test_postflight_detects_disk_mutation_without_changing_executed_bytes(self):
        bundle = verify_sources_once(self.root, self.pins)
        (self.root / "example.py").write_bytes(b"MARK = 9\n")
        ns = {}; exec(bundle.compile("original"), ns)
        self.assertEqual(ns["MARK"], 5)
        with self.assertRaises(ContractError):
            bundle.verify_unchanged()

    def test_wrong_pin_traversal_duplicate_and_symlink_rejected(self):
        (self.root / "alias.py").symlink_to(self.root / "example.py")
        invalid = ({"a": ("example.py", "a" * 64)}, {"a": ("../example.py", sha256(self.raw))},
                   {"a": ("alias.py", sha256(self.raw))},
                   {"a": self.pins["original"], "b": self.pins["original"]})
        for pins in invalid:
            with self.assertRaises(ContractError):
                verify_sources_once(self.root, pins)

    def test_missing_function_duplicate_selection_and_constructor_refused(self):
        bundle = verify_sources_once(self.root, self.pins)
        for functions in (("missing",), ("f", "f"), ()):
            with self.assertRaises(ContractError):
                bundle.compile("original", functions=functions)
        with self.assertRaises(ContractError):
            VerifiedSources()

    def test_compile_never_inherits_compiler_future_flags(self):
        for future, expected in ((b"", 3), (b"from __future__ import annotations\n", "1 + 2")):
            raw = future + b"def f(x: 1+2):\n    return x\n"
            (self.root / "annotations.py").write_bytes(raw)
            bundle = verify_sources_once(self.root, {"a": ("annotations.py", sha256(raw))})
            for functions in (None, ("f",)):
                ns = {}; exec(bundle.compile("a", functions=functions), ns)
                self.assertEqual(ns["f"].__annotations__["x"], expected)


if __name__ == "__main__":
    unittest.main()
