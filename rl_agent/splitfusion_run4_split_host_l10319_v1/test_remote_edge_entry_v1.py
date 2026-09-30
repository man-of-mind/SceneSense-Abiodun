"""CPU-only tests for the measured-GPU compatibility entry seam."""

from __future__ import annotations

import unittest

from . import contract as C
from . import remote_edge_entry_v1 as G
from .test_remote_edge_lifecycle_contract_v1 import measured_binding


class _FakeCuda:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_device_name(self, _device: object = None) -> str:
        return self.name


class _FakeTorch:
    def __init__(self, name: str) -> None:
        self.cuda = _FakeCuda(name)


class RemoteGpuEntryTests(unittest.TestCase):
    def test_measured_row_is_parsed_and_matches_binding(self) -> None:
        row = G.parse_nvidia_smi_row(
            "NVIDIA GeForce RTX 5090 Laptop GPU, "
            "GPU-b8c4646c-abb0-5d63-679b-49622ce057b6, 24463, 610.43.02\n"
        )
        G.validate_measured_gpu(row, measured_binding())
        self.assertEqual(row["memory_total_mib"], 24463)

    def test_any_remote_gpu_fact_drift_is_refused(self) -> None:
        expected = {
            "model": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "uuid": "GPU-b8c4646c-abb0-5d63-679b-49622ce057b6",
            "memory_total_mib": 24463,
            "driver_version": "610.43.02",
        }
        for key in expected:
            with self.subTest(key=key):
                bad = dict(expected)
                bad[key] = 1 if key == "memory_total_mib" else "drift"
                with self.assertRaises(G.RemoteEdgeGpuError):
                    G.validate_measured_gpu(bad, measured_binding())

    def test_legacy_alias_is_scoped_and_restored(self) -> None:
        torch = _FakeTorch("NVIDIA GeForce RTX 5090 Laptop GPU")
        seen = []

        def frozen(argv: object) -> int:
            seen.append((torch.cuda.get_device_name(0), torch.cuda.get_device_name(0), list(argv)))
            return 7

        result = G.delegate_frozen_service(
            torch_module=torch, frozen_main=frozen, frozen_argv=("--edge",),
            measured_model="NVIDIA GeForce RTX 5090 Laptop GPU",
        )
        self.assertEqual(result, 7)
        self.assertEqual(seen, [(G.LEGACY_DEVICE_NAME, "NVIDIA GeForce RTX 5090 Laptop GPU", ["--edge"])])
        self.assertEqual(torch.cuda.get_device_name(0),
                         "NVIDIA GeForce RTX 5090 Laptop GPU")

    def test_alias_is_never_applied_to_unbound_model(self) -> None:
        torch = _FakeTorch("foreign GPU")
        called = []
        with self.assertRaises(G.RemoteEdgeGpuError):
            G.delegate_frozen_service(
                torch_module=torch, frozen_main=lambda _argv: called.append(True) or 0,
                frozen_argv=(), measured_model=measured_binding().gpu.model,
            )
        self.assertFalse(called)
        self.assertEqual(torch.cuda.get_device_name(0), "foreign GPU")

    def test_parser_rejects_multiple_or_malformed_gpus(self) -> None:
        for text in ("", "a,b,c", "a,b,1,d\na,b,1,d", "a,b,not-int,d"):
            with self.subTest(text=text):
                with self.assertRaises(G.RemoteEdgeGpuError):
                    G.parse_nvidia_smi_row(text)


if __name__ == "__main__":
    unittest.main()
