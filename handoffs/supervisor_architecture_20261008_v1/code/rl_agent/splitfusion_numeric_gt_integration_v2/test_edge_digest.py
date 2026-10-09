"""Edge-only imports and byte parity with the pinned producer implementation."""
from __future__ import annotations

from types import MappingProxyType
import subprocess
import sys
import unittest

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from rl_agent.splitfusion_numeric_gt_producer_v1 import payload as P
from . import edge_digest as D


class EdgeDigestTests(unittest.TestCase):
    def test_producer_semantic_digest_byte_parity(self):
        self.assertEqual(D.PAYLOAD_SCHEMA, P.SCHEMA)
        for metadata, segmentation, ignore in (
            (b"", b"", b""),
            (b'{"frame":10,"timestamp":-0.0}', b"\0\1\2\xff", b"\xff\0"),
            (bytes(range(256)), bytes(range(255, -1, -1)), b"\0" * 8192),
        ):
            with self.subTest(size=len(metadata)):
                self.assertEqual(D.semantic_digest(metadata, segmentation, ignore),
                                 P.semantic_digest(metadata, segmentation, ignore))

    def test_source_json_normalization_byte_parity(self):
        values = (
            {"camera": (np.float64(-0.0), np.float64(1.25)),
             "sources": MappingProxyType({"rgb": [2, np.float64(3.5)]})},
            {"nested": [{"rows": ("a", "b")}, None, False]},
            {"labels": frozenset(("vehicle", "pedestrian"))},
        )
        for value in values:
            with self.subTest(value=str(value)):
                self.assertEqual(T._canonical(value), P.A.canonical(P.json_value(value)))

    def test_fresh_edge_import_does_not_load_producer_scipy_or_carla(self):
        script = '''\
import importlib.abc
import sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("scipy", "carla") or fullname.startswith(("scipy.", "carla.")):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, Block())
from rl_agent.splitfusion_numeric_gt_runtime_v1 import edge_live, live_archive, scorer
assert live_archive._legacy_files(None, None) == {}
assert not any(name.startswith(("scipy", "carla", "rl_agent.splitfusion_numeric_gt_producer_v1"))
               for name in sys.modules)
'''
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
