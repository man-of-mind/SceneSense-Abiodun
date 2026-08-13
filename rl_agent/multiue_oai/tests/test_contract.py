from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rl_agent.multiue_oai.analyze import pair_effect, service_family
from rl_agent.multiue_oai.endpoint import (
    FRAME_HEADER,
    MAGIC,
    VERSION,
    build_frame_blob,
    chunks_per_frame,
    frame_onwire_bytes,
)
from rl_agent.multiue_oai.runner import DEFAULT_CONFIG, Runner, load_config


class EndpointContractTest(unittest.TestCase):
    def test_frame_metadata_and_wire_accounting(self) -> None:
        blob = build_frame_blob(409600, 1, 1234, 987654321)
        self.assertEqual(len(blob), 409600)
        magic, version, ue_id, _flags, frame_id, scheduled, payload_bytes, _crc = FRAME_HEADER.unpack_from(blob)
        self.assertEqual(magic, MAGIC)
        self.assertEqual(version, VERSION)
        self.assertEqual(ue_id, 1)
        self.assertEqual(frame_id, 1234)
        self.assertEqual(scheduled, 987654321)
        self.assertEqual(payload_bytes, 409600)
        self.assertEqual(chunks_per_frame(409600, 60000), 7)
        self.assertEqual(frame_onwire_bytes(409600, 60000), 409600 + 7 * 36)


class ConfigContractTest(unittest.TestCase):
    def test_stage_is_n2_and_cannot_chain(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["radio"]["ue_count"], 2)
        self.assertEqual(config["transport"]["payload_bytes"], 409600)
        self.assertEqual(config["c1"]["pessimism_factor"], 0.70)
        self.assertEqual(config["c1"]["estimator_window_s"], 1.0)
        self.assertEqual(config["c1"]["estimator_ewma_alpha"], 0.20)
        self.assertIn("DG-B", config["authorization_boundary"]["forbidden"])
        self.assertEqual([row["id"] for row in config["trials"]], [f"A{i}" for i in range(1, 10)])

    def test_strong_channel_has_explicit_models_for_both_ues(self) -> None:
        runner = Runner(DEFAULT_CONFIG, Path("/tmp/not-created"), dry_run=True)
        channel_path = runner.paths["oai_ran_conf"] / runner.config["radio"]["channel_config"]
        channel = channel_path.read_text(encoding="utf-8")
        for index in range(2):
            self.assertIn(f'model_name     = "rfsimu_channel_enB{index}"', channel)
            self.assertIn(f'model_name     = "rfsimu_channel_ue{index}"', channel)

    def test_attach_smoke_mode_is_explicit_and_cannot_be_negative(self) -> None:
        runner = Runner(
            DEFAULT_CONFIG,
            Path("/tmp/not-created"),
            dry_run=True,
            attach_smoke_repeats=3,
        )
        self.assertEqual(runner.attach_smoke_repeats, 3)
        with self.assertRaises(ValueError):
            Runner(
                DEFAULT_CONFIG,
                Path("/tmp/not-created"),
                dry_run=True,
                attach_smoke_repeats=-1,
            )

    def test_dry_run_writes_completion_without_oai(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dry"
            status = Runner(DEFAULT_CONFIG, output, dry_run=True).run()
            self.assertEqual(status, 0)
            self.assertTrue((output / "COMPLETED.json").exists())
            self.assertFalse((output / "FAILED.json").exists())


class DecisionContractTest(unittest.TestCase):
    def test_pair_requires_effect_and_goodput(self) -> None:
        config = load_config(DEFAULT_CONFIG)["decision"]
        greedy = {
            "trial_id": "A6",
            "demand_trace_sha256": "same",
            "worst_deadline_0.25s_fraction": 0.40,
            "worst_deadline_0.50s_fraction": 0.50,
            "worst_complete_latency_p95_ms": 500.0,
            "worst_max_starvation_ms": 1000.0,
            "aggregate_complete_goodput_mbps": 7.0,
        }
        central = {
            "trial_id": "A7",
            "demand_trace_sha256": "same",
            "worst_deadline_0.25s_fraction": 0.46,
            "worst_deadline_0.50s_fraction": 0.56,
            "worst_complete_latency_p95_ms": 400.0,
            "worst_max_starvation_ms": 750.0,
            "aggregate_complete_goodput_mbps": 6.9,
        }
        effect = pair_effect(greedy, central, config)
        self.assertTrue(effect["meaningful_gap"])
        central["aggregate_complete_goodput_mbps"] = 6.0
        self.assertFalse(pair_effect(greedy, central, config)["meaningful_gap"])

    def test_service_families_reproduce_n2(self) -> None:
        for family in ("constant_ceiling", "saturating", "power_law"):
            self.assertAlmostEqual(service_family(family, 2, 10.4, 9.8, 36.7), 9.8)


if __name__ == "__main__":
    unittest.main()
