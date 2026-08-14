from __future__ import annotations

import json
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
from rl_agent.multiue_oai.runner import (
    DEFAULT_CONFIG,
    Runner,
    load_config,
    parse_channel_models,
    parse_interface_ipv4,
    per_ue_radio_summary,
    receiver_identity_report,
    sender_route_report,
)


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
        self.assertEqual(config["radio"]["expected_receiver_nat_sources"], ["192.168.70.134"])
        self.assertEqual(config["instrumentation"]["tunnel_tx_to_sender_ratio_min"], 1.0)
        self.assertEqual(config["instrumentation"]["tunnel_tx_to_sender_ratio_max"], 1.08)
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
        self.assertEqual(runner.attach_channel_mode, "strong")
        clean = Runner(
            DEFAULT_CONFIG,
            Path("/tmp/not-created"),
            dry_run=True,
            attach_smoke_repeats=1,
            attach_channel_mode="clean",
        )
        self.assertEqual(clean.attach_channel_mode, "clean")
        clean._materialize_ue_config()
        self.assertEqual(
            clean.runtime_ue_config,
            clean.paths["oai_ran_conf"] / clean.config["radio"]["ue_base_config"],
        )
        with self.assertRaises(ValueError):
            Runner(
                DEFAULT_CONFIG,
                Path("/tmp/not-created"),
                dry_run=True,
                attach_smoke_repeats=-1,
            )
        with self.assertRaises(ValueError):
            Runner(
                DEFAULT_CONFIG,
                Path("/tmp/not-created"),
                dry_run=True,
                attach_channel_mode="clean",
            )
        with self.assertRaises(ValueError):
            Runner(
                DEFAULT_CONFIG,
                Path("/tmp/not-created"),
                dry_run=True,
                attach_smoke_repeats=1,
                runtime_switch_smoke=True,
            )

    def test_runtime_switch_materializes_initial_clean_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner = Runner(
                DEFAULT_CONFIG,
                Path(temp) / "switch",
                dry_run=True,
                runtime_switch_smoke=True,
            )
            runner._materialize_ue_config()
            ue_config = runner.runtime_ue_config.read_text(encoding="utf-8")
            gnb_config = runner.runtime_gnb_config.read_text(encoding="utf-8")
            self.assertEqual(ue_config.count("noise_power_dB = -50;"), 4)
            self.assertEqual(gnb_config.count("noise_power_dB = -50;"), 4)
            self.assertNotIn("noise_power_dB = -4;", ue_config)
            self.assertNotIn("@include", gnb_config)
            self.assertIn("Active_gNBs", gnb_config)

    def test_interface_identity_is_stable_while_ip_is_discovered(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(
            [(row["ue_id"], row["iface"]) for row in config["radio"]["expected_tunnels"]],
            [(0, "oaitun_ue1"), (1, "oaitun_ue2")],
        )
        payload = json.dumps(
            [
                {
                    "ifname": "oaitun_ue1",
                    "addr_info": [{"family": "inet", "local": "10.0.0.3", "prefixlen": 24}],
                }
            ]
        )
        self.assertEqual(parse_interface_ipv4(payload, "oaitun_ue1"), "10.0.0.3")
        self.assertIsNone(parse_interface_ipv4("[]", "oaitun_ue1"))
        report = receiver_identity_report(
            [
                {"ue_id": 0, "source_ip": "192.168.70.134", "message_id": 1 << 28},
                {"ue_id": 1, "source_ip": "192.168.70.134", "message_id": 2 << 28},
            ],
            {0, 1},
            {"192.168.70.134"},
        )
        self.assertEqual(report["invalid_message_id_count"], 0)
        self.assertEqual(report["unexpected_nat_sources"], [])
        self.assertEqual(report["missing_ues"], [])
        mismatch = receiver_identity_report(
            [{"ue_id": 0, "source_ip": "192.168.70.200", "message_id": 2 << 28}],
            {0, 1},
            {"192.168.70.134"},
        )
        self.assertEqual(mismatch["invalid_message_id_count"], 1)
        self.assertEqual(mismatch["unexpected_nat_sources"], ["192.168.70.200"])

    def test_sender_route_uses_bind_and_tunnel_evidence_before_nat(self) -> None:
        sender = {
            "socket_bindings": {
                "0": {"requested_bind_ip": "10.0.0.3", "actual_local_ip": "10.0.0.3", "actual_local_port": 10001},
                "1": {"requested_bind_ip": "10.0.0.2", "actual_local_ip": "10.0.0.2", "actual_local_port": 10002},
            },
            "per_ue": {
                "0": {"sent_onwire_bytes": 3_688_668},
                "1": {"sent_onwire_bytes": 4_098_520},
            },
        }
        network = {
            0: {"ue_id": 0, "iface": "oaitun_ue1", "ip": "10.0.0.3"},
            1: {"ue_id": 1, "iface": "oaitun_ue2", "ip": "10.0.0.2"},
        }
        tunnels = {
            "ue0": {"tx_bytes_delta": 3_737_808},
            "ue1": {"tx_bytes_delta": 4_153_120},
        }
        report = sender_route_report(
            sender, network, tunnels, tx_ratio_min=1.0, tx_ratio_max=1.08
        )
        self.assertTrue(report["pass"])
        sender["socket_bindings"]["0"]["actual_local_ip"] = "10.0.0.2"
        self.assertFalse(
            sender_route_report(
                sender, network, tunnels, tx_ratio_min=1.0, tx_ratio_max=1.08
            )["pass"]
        )

    def test_channel_state_and_per_ue_radio_evidence_are_parsed(self) -> None:
        channel = parse_channel_models(
            "model 2 rfsimu_channel_ue0 type AWGN:\n"
            "max Doppler: 0 path loss: 0.000000  noise: -4.000000 rchannel offset: 0\n"
            "model 3 rfsimu_channel_ue1 type AWGN:\n"
            "max Doppler: 0 path loss: 0.000000  noise: -4.000000 rchannel offset: 0\n"
        )
        self.assertEqual(channel["rfsimu_channel_ue0"]["model_index"], 2)
        self.assertEqual(channel["rfsimu_channel_ue1"]["noise_power_db"], -4.0)
        radio = per_ue_radio_summary(
            [
                {"rnti": "100", "snrx10": "82", "mcs": "9"},
                {"rnti": "100", "snrx10": "84", "mcs": "10"},
                {"rnti": "200", "snrx10": "80", "mcs": "8"},
            ],
            [
                {"ue_id": "0", "rnti": "100"},
                {"ue_id": "0", "rnti": "100"},
                {"ue_id": "1", "rnti": "200"},
            ],
        )
        self.assertEqual(radio["0"]["rnti"], 100)
        self.assertAlmostEqual(radio["0"]["median_pusch_snr_db"], 8.3)
        self.assertEqual(radio["1"]["median_ul_mcs"], 8.0)

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
