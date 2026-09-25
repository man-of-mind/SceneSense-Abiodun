"""Offline tests for the bounded capacity qualification stage."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rl_agent import ue_n3_structured_udp_receiver as U3
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_qualification as CQ
from rl_agent.ue_mcs_backlog_near_capacity_v1 import capacity_runner as R


ROOT = Path(__file__).resolve().parents[2]


def fake_actions() -> list[dict[str, object]]:
    values = (2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)
    return [{
        "action_id": index, "profile_id": f"a{index}",
        "payload_bytes": 25_000 * (index + 1),
        "checkpoint_sha256": f"digest-{index}", "offered_mbps": rate,
    } for index, rate in enumerate(values)]


def point(label: str, p50: float = 10.0, **overrides: object) -> CQ.CapacityPoint:
    targets = CQ.ADVERSE_OPERATING_POINTS_DB
    values = {
        "achieved_pusch_snr_db_p50": targets[label],
        "achieved_pusch_snr_samples": CQ.MIN_PUSCH_SNR_SAMPLES,
        "label": label, "target_snr_db": targets[label],
        "commanded_noise_power_db": -4.0,
        "service_mbps_p10": p50 * 0.9,
        "service_mbps_p50": p50,
        "service_mbps_p90": p50 * 1.1,
        "samples": 100, "backlogged_fraction": 0.95,
    }
    values.update(overrides)
    return CQ.CapacityPoint(**values)


class CapacityAuditTests(unittest.TestCase):
    def clean(self, samples=None):
        return CQ.audit_points(
            [point("p25", 9.0), point("p50", 10.0), point("p75", 11.0)],
            boundary_service_samples=samples or [10.0] * 100,
            actions=fake_actions())

    def test_clean_surface_and_stable_tiers_qualify(self):
        report = self.clean()
        self.assertTrue(report["qualified"], report["problems"])
        self.assertTrue(report["achieved_snr_strictly_ordered"])
        self.assertTrue(report["tier_stability"]["stable"])

    def test_bootstrap_is_deterministic(self):
        values = [9.5 + (index % 7) * 0.1 for index in range(100)]
        self.assertEqual(CQ.bootstrap_median_interval(values, draws=200),
                         CQ.bootstrap_median_interval(values, draws=200))

    def test_unstable_tier_triplet_refuses(self):
        report = self.clean([8.0] * 50 + [12.0] * 50)
        self.assertFalse(report["qualified"])
        self.assertFalse(report["tier_stability"]["stable"])

    def test_missing_bootstrap_samples_refuses(self):
        report = CQ.audit_points(
            [point("p25", 9.0), point("p50", 10.0), point("p75", 11.0)],
            actions=fake_actions())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("bootstrap" in item for item in report["problems"]))

    def test_duplicate_and_unknown_labels_refuse(self):
        duplicate = CQ.audit_points(
            [point("p25"), point("p25"), point("p75")],
            boundary_service_samples=[10.0] * 100, actions=fake_actions())
        self.assertFalse(duplicate["qualified"])
        self.assertTrue(any("duplicate" in item for item in duplicate["problems"]))
        bad = CQ.CapacityPoint(
            achieved_pusch_snr_db_p50=8.0, achieved_pusch_snr_samples=30,
            label="mystery", target_snr_db=8.0, commanded_noise_power_db=-4.0,
            service_mbps_p10=8, service_mbps_p50=9, service_mbps_p90=10,
            samples=100, backlogged_fraction=1.0)
        unknown = CQ.audit_points(
            [point("p25"), point("p50"), bad],
            boundary_service_samples=[10.0] * 100, actions=fake_actions())
        self.assertFalse(unknown["qualified"])
        self.assertTrue(any("unknown" in item for item in unknown["problems"]))

    def test_target_snr_must_match_exactly(self):
        report = CQ.audit_points(
            [point("p25"), point("p50", target_snr_db=8.6081), point("p75")],
            boundary_service_samples=[10.0] * 100, actions=fake_actions())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("target SNR" in item for item in report["problems"]))

    def test_achieved_snr_needs_30_samples(self):
        report = CQ.audit_points(
            [point("p25"), point("p50", achieved_pusch_snr_samples=29), point("p75")],
            boundary_service_samples=[10.0] * 100, actions=fake_actions())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("PUSCH SNR samples" in item for item in report["problems"]))

    def test_achieved_snr_must_track_target_within_one_db(self):
        report = CQ.audit_points(
            [point("p25"), point("p50", achieved_pusch_snr_db_p50=9.609),
             point("p75")], boundary_service_samples=[10.0] * 100,
            actions=fake_actions())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("differs from target" in item for item in report["problems"]))

    def test_achieved_snr_medians_must_be_strictly_ordered(self):
        report = CQ.audit_points(
            [point("p25", achieved_pusch_snr_db_p50=8.0),
             point("p50", achieved_pusch_snr_db_p50=8.0), point("p75")],
            boundary_service_samples=[10.0] * 100, actions=fake_actions())
        self.assertFalse(report["qualified"])
        self.assertFalse(report["achieved_snr_strictly_ordered"])

    def test_nonfinite_unordered_and_bad_backlog_refuse(self):
        report = CQ.audit_points(
            [point("p25"), point("p50", service_mbps_p10=12.0,
                                 service_mbps_p90=8.0),
             point("p75", commanded_noise_power_db=math.nan)],
            boundary_service_samples=[10.0] * 100, actions=fake_actions())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("not ordered" in item for item in report["problems"]))
        self.assertTrue(any("non-finite" in item for item in report["problems"]))
        report = CQ.audit_points(
            [point("p25"), point("p50", backlogged_fraction=1.1), point("p75")],
            boundary_service_samples=[10.0] * 100, actions=fake_actions())
        self.assertFalse(report["qualified"])
        self.assertTrue(any("outside [0,1]" in item for item in report["problems"]))

    def test_probe_identity_is_exact_catalogue_maximum(self):
        probe = CQ.verify_probe_action(ROOT)
        self.assertTrue(probe["verified_largest_eligible_action"])
        self.assertEqual(probe["action_id"], CQ.PROBE_ACTION_ID)
        self.assertEqual(probe["offered_mbps"], 285.46608)

    def test_capacity_packetization_is_mtu_safe_and_production_is_unchanged(self):
        identity = CQ.probe_packetization_identity()
        self.assertEqual(identity["frame_payload_bytes"], 3_568_326)
        self.assertEqual(identity["chunk_payload_bytes"], 1_200)
        self.assertEqual(identity["chunks_per_frame"], 2_974)
        self.assertGreater(identity["chunks_per_frame"],
                           U3.MAX_CHUNKS_PER_FRAME_LIMIT)
        self.assertEqual(identity["last_chunk_payload_bytes"], 726)
        self.assertEqual(identity["full_ipv4_packet_bytes"], 1_252)
        self.assertLessEqual(identity["full_ipv4_packet_bytes"],
                             identity["path_mtu_bytes"])
        self.assertEqual(identity["production_scientific_chunk_bytes"], 60_000)

    def test_capacity_packetization_drift_is_refused(self):
        value = CQ.probe_packetization_identity()
        value["chunk_payload_bytes"] = 1_201
        with self.assertRaisesRegex(CQ.CapacityQualificationError,
                                    "packetization differs"):
            CQ.verify_probe_packetization(
                value, production_chunk_bytes=60_000,
                ssburst_header_bytes=U3.HEADER.size)

class SinkAccountingTests(unittest.TestCase):
    @staticmethod
    def datagram(frame: int, chunk: int, chunks: int, payload: bytes) -> bytes:
        return U3.HEADER.pack(
            U3.MAGIC, frame, chunk, chunks, U3.HEADER.size + len(payload)) + payload

    def test_unique_payload_is_binned_and_duplicate_is_not(self):
        sink = R.SinkAccounting(
            epoch_ns=1_000_000_000, duration_bins=3,
            expected_frames=3, expected_chunks=CQ.PROBE_CHUNKS_PER_FRAME,
            expected_frame_payload_bytes=CQ.PROBE_PAYLOAD_BYTES,
            expected_chunk_payload_bytes=CQ.PROBE_CHUNK_PAYLOAD_BYTES)
        data = self.datagram(
            0, 0, CQ.PROBE_CHUNKS_PER_FRAME,
            b"x" * CQ.PROBE_CHUNK_PAYLOAD_BYTES)
        first = sink.ingest(data, monotonic_ns=1_010_000_000)
        duplicate = sink.ingest(data, monotonic_ns=1_020_000_000)
        self.assertEqual(first["status"], "ACCEPTED_UNIQUE")
        self.assertEqual(duplicate["status"], "DUPLICATE")
        self.assertEqual(sink.summary()["payload_bytes_per_100ms_bin"],
                         [CQ.PROBE_CHUNK_PAYLOAD_BYTES, 0, 0])

    def test_malformed_foreign_and_header_exclusion(self):
        sink = R.SinkAccounting(
            epoch_ns=0, duration_bins=2, expected_frames=1,
            expected_chunks=CQ.PROBE_CHUNKS_PER_FRAME,
            expected_frame_payload_bytes=CQ.PROBE_PAYLOAD_BYTES,
            expected_chunk_payload_bytes=CQ.PROBE_CHUNK_PAYLOAD_BYTES)
        sink.ingest(b"short", monotonic_ns=1)
        sink.ingest(self.datagram(
            2, 0, CQ.PROBE_CHUNKS_PER_FRAME,
            b"x" * CQ.PROBE_CHUNK_PAYLOAD_BYTES), monotonic_ns=2)
        sink.ingest(self.datagram(
            0, 0, CQ.PROBE_CHUNKS_PER_FRAME,
            b"x" * CQ.PROBE_CHUNK_PAYLOAD_BYTES), monotonic_ns=3)
        summary = sink.summary()
        self.assertEqual(summary["malformed_datagrams"], 1)
        self.assertEqual(summary["outside_registered_probe"], 1)
        self.assertEqual(summary["payload_bytes_per_100ms_bin"],
                         [CQ.PROBE_CHUNK_PAYLOAD_BYTES, 0])

    def test_custom_capacity_sink_accepts_chunk_indexes_above_1024(self):
        sink = R.SinkAccounting(
            epoch_ns=0, duration_bins=2, expected_frames=1,
            expected_chunks=CQ.PROBE_CHUNKS_PER_FRAME,
            expected_frame_payload_bytes=CQ.PROBE_PAYLOAD_BYTES,
            expected_chunk_payload_bytes=CQ.PROBE_CHUNK_PAYLOAD_BYTES)
        high = sink.ingest(self.datagram(
            0, 1_024, CQ.PROBE_CHUNKS_PER_FRAME,
            b"x" * CQ.PROBE_CHUNK_PAYLOAD_BYTES), monotonic_ns=1)
        tail = sink.ingest(self.datagram(
            0, CQ.PROBE_CHUNKS_PER_FRAME - 1, CQ.PROBE_CHUNKS_PER_FRAME,
            b"x" * CQ.PROBE_LAST_CHUNK_PAYLOAD_BYTES), monotonic_ns=2)
        self.assertEqual(high["status"], "ACCEPTED_UNIQUE")
        self.assertEqual(tail["status"], "ACCEPTED_UNIQUE")
        summary = sink.summary()
        self.assertEqual(summary["accepted_unique_chunks"], 2)
        self.assertEqual(summary["packetization_mismatch_datagrams"], 0)
        self.assertEqual(summary["payload_bytes_per_100ms_bin"][0], 1_926)

    def test_custom_capacity_sink_rejects_payload_size_drift(self):
        sink = R.SinkAccounting(
            epoch_ns=0, duration_bins=1, expected_frames=1,
            expected_chunks=CQ.PROBE_CHUNKS_PER_FRAME,
            expected_frame_payload_bytes=CQ.PROBE_PAYLOAD_BYTES,
            expected_chunk_payload_bytes=CQ.PROBE_CHUNK_PAYLOAD_BYTES)
        event = sink.ingest(self.datagram(
            0, 0, CQ.PROBE_CHUNKS_PER_FRAME,
            b"x" * (CQ.PROBE_CHUNK_PAYLOAD_BYTES - 1)), monotonic_ns=1)
        self.assertEqual(event["status"], "PACKETIZATION_MISMATCH")
        self.assertEqual(sink.summary()["packetization_mismatch_datagrams"], 1)


class TelemetryAnalysisTests(unittest.TestCase):
    def test_rlc_tick_aggregation_sums_lcids(self):
        rows = [
            (0, 100, "0,1,0,10,2,1,0,40,0,0,0"),
            (0, 101, "0,1,0,10,2,2,0,60,0,0,0"),
            (0, 200, "0,1,0,10,3,1,0,0,0,0,0"),
        ]
        ticks = R.aggregate_rlc_ticks(rows)
        self.assertEqual([row["backlog_bytes"] for row in ticks], [100, 0])

    def test_backlogged_bin_requires_ticks_and_all_positive(self):
        start = 1_000_000_000
        rows = [
            (0, start + 1, "0,1,0,1,1,1,0,5,0,0,0"),
            (0, start + R.PERIOD_NS + 1, "0,1,0,1,2,1,0,0,0,0,0"),
        ]
        report = R.backlogged_bins(rows, start_ns=start, bins=3)
        self.assertEqual(report["bin_backlogged"], [True, False, False])
        self.assertAlmostEqual(report["backlogged_fraction"], 1 / 3)

    def test_drain_proof_uses_only_trailing_post_boundary_zero_ticks(self):
        ticks = [
            {"receipt_monotonic_ns": 90, "backlog_bytes": 0},
            {"receipt_monotonic_ns": 101, "backlog_bytes": 0},
            {"receipt_monotonic_ns": 102, "backlog_bytes": 0},
            {"receipt_monotonic_ns": 103, "backlog_bytes": 7},
            {"receipt_monotonic_ns": 104, "backlog_bytes": 0},
            {"receipt_monotonic_ns": 105, "backlog_bytes": 0},
        ]
        run, retained = R.trailing_zero_backlog_run(ticks, after_ns=100)
        self.assertEqual(run, 2)
        self.assertEqual([row["receipt_monotonic_ns"] for row in retained],
                         [101, 102, 103, 104, 105])

    def test_latest_ingress_uses_embedded_monotonic_stamp(self):
        rows = [
            (0, 9999, "0,2,10,0,1,20"),
            (0, 1, "0,3,15,0,1,20"),
            (0, 2, "malformed"),
        ]
        latest, count = R.latest_mono_event_ns(rows, after_ns=2_500_000_000)
        self.assertEqual(latest, 3_000_000_015)
        self.assertEqual(count, 1)

    def test_drain_requires_observation_stable_ingress_quiet_interval(self):
        class Live:
            def __init__(self, rows):
                self.rows = rows

            def snapshot(self):
                return list(self.rows)

        class Clock:
            def __init__(self):
                self.ns = 1_000_000_000

            def monotonic(self):
                return self.ns / 1e9

            def monotonic_ns(self):
                return self.ns

            def sleep(self, seconds):
                self.ns += int(seconds * 1e9)

        runner = object.__new__(R.Runner)
        runner.config = {"capacity_qualification": {"service_measurement": {
            "drain_zero_consecutive_samples": 5,
            "drain_timeout_s": 2.0,
            "drain_poll_s": 0.1,
            "drain_quiet_interval_s": CQ.DRAIN_QUIET_INTERVAL_S,
        }}}
        zero_rows = [
            (0, 1_100_000_000 + index * 10_000_000,
             f"0,1,0,{index},0,1,0,0,0,0,0")
            for index in range(6)
        ]
        runner.capacity_live = {
            "pdcp_sdu": Live([]), "rlc_sdu": Live([]),
            "rlc_buffer": Live(zero_rows),
        }
        clock = Clock()
        with mock.patch.object(R.time, "monotonic", side_effect=clock.monotonic),              mock.patch.object(R.time, "monotonic_ns", side_effect=clock.monotonic_ns),              mock.patch.object(R.time, "sleep", side_effect=clock.sleep):
            proof = runner.prove_zero_backlog(after_ns=500_000_000, label="test")
        self.assertTrue(proof["drained"])
        self.assertTrue(proof[
            "no_new_pdcp_or_rlc_ingress_during_quiet_interval"])
        self.assertGreaterEqual(proof["quiet_elapsed_ns"], 500_000_000)
        self.assertEqual(proof["observed_consecutive_zero_ticks"], 5)

    def test_point_analysis_uses_100ms_extdn_payload_bins_and_snr(self):
        epoch = 10_000_000_000
        measure_start = epoch + R.SETTLE_BINS * R.PERIOD_NS
        rlc, rlc_sdu, dequeue, delivered = [], [], [], []
        for index in range(R.MEASURE_BINS):
            stamp = measure_start + index * R.PERIOD_NS + 1_000
            rlc.append((0, stamp, f"0,1,0,{index},0,1,0,100,0,0,0"))
            sec, nsec = divmod(stamp, 1_000_000_000)
            rlc_sdu.append((0, stamp, f"0,{sec},{nsec},0,1,1000"))
            dequeue.append((0, stamp, f"0,{sec},{nsec},0,1,900"))
            delivered.append((0, stamp, f"0,{sec},{nsec},0,1,800"))
        sink = {"payload_bytes_per_100ms_bin":
                [0] * R.SETTLE_BINS + [1000] * R.MEASURE_BINS}
        observed, samples, corroboration = R.analyze_point(
            label="p50", target_snr_db=8.608, commanded_noise_power_db=-4.0,
            epoch_ns=epoch, sink_summary=sink,
            telemetry={"rlc_buffer": rlc, "rlc_sdu": rlc_sdu,
                       "rlc_dequeue": dequeue,
                       "gnb_pdcp_deliver": delivered},
            achieved_pusch_snr_values=[8.6] * 40)
        self.assertEqual(len(samples), 100)
        self.assertAlmostEqual(observed.service_mbps_p50, 0.08)
        self.assertEqual(observed.backlogged_fraction, 1.0)
        self.assertEqual(observed.achieved_pusch_snr_samples, 40)
        self.assertAlmostEqual(observed.achieved_pusch_snr_db_p50, 8.6)
        self.assertTrue(corroboration["corroboration_complete"])


class EvidenceBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = R.source_inventory(ROOT)

    def _checks(self):
        return [{
            "stage": stage,
            "source_inventory": {"verified": True},
            "radio_binding": {"verified": True},
            "protected_evidence": {"all_unchanged": True},
        } for stage in (
            "before_preflight", "before_point_p25", "before_point_p50",
            "before_point_p75", "final_sealing")]

    @staticmethod
    def _container_images():
        names = json.loads(R.DEFAULT_CONFIG.read_text())["radio"]["core_containers"]
        return {name: {
            "configured_image": f"example/{name}:registered",
            "image_id": "sha256:" + f"{index + 1:064x}",
            "repo_digests": [f"example/{name}@sha256:" + f"{index + 2:064x}"],
            "container_id": f"container-{index}",
        } for index, name in enumerate(names)}

    def _make_bound_result(self, root: Path) -> Path:
        evidence = root / "evidence" / "measurement.txt"
        evidence.parent.mkdir(parents=True)
        evidence.write_text("retained measurement\n")
        points = [
            point(label, rate, service_mbps_p10=rate, service_mbps_p90=rate)
            for label, rate in (("p25", 80.0), ("p50", 85.0), ("p75", 90.0))
        ]
        samples = {label: [rate] * 100 for label, rate in (
            ("p25", 80.0), ("p50", 85.0), ("p75", 90.0))}
        audit = CQ.audit_points(
            points, boundary_service_samples=samples["p50"], repo_root=ROOT)
        self.assertTrue(audit["qualified"], audit["problems"])
        tiers = CQ.select_tiers(85.0, repo_root=ROOT)
        records = []
        packetization = CQ.probe_packetization_identity()
        for measured in points:
            payload_per_bin = int(round(
                samples[measured.label][0] * CQ.SAMPLE_PERIOD_S * 1e6 / 8.0))
            achieved_values = [measured.achieved_pusch_snr_db_p50] * 30
            backlog_flags = [True] * 95 + [False] * 5
            backlog_counts = [1] * 95 + [0] * 5
            backlog_minima = [100] * 95 + [None] * 5
            records.append({
                "label": measured.label,
                "target_snr_db": measured.target_snr_db,
                "prime": {"commanded_noise_power_db": -4.0,
                          "read_back_noise_power_db": -4.0},
                "capacity_point": measured.to_json(),
                "service_mbps_samples": samples[measured.label],
                "packetization": copy.deepcopy(packetization),
                "sender": {
                    "frames": R.POINT_FRAMES,
                    "payload_bytes": CQ.PROBE_PAYLOAD_BYTES,
                    "chunk_bytes": CQ.PROBE_CHUNK_PAYLOAD_BYTES,
                    "chunks_per_frame": R.PROBE_CHUNKS,
                    "packetization": copy.deepcopy(packetization),
                    "chunks_handed_to_socket":
                        R.POINT_FRAMES * R.PROBE_CHUNKS,
                    "chunks_dropped_by_socket": 0,
                    "unexpected_socket_errors": 0,
                },
                "sink": {
                    "clean_duration_complete": True,
                    "expected_frames": R.POINT_FRAMES,
                    "expected_chunks_per_frame": R.PROBE_CHUNKS,
                    "header_bytes_excluded": U3.HEADER.size,
                    "packetization": copy.deepcopy(packetization),
                    "malformed_datagrams": 0,
                    "packetization_mismatch_datagrams": 0,
                    "outside_registered_probe": 0,
                    "payload_bytes_per_100ms_bin":
                        [0] * R.SETTLE_BINS
                        + [payload_per_bin] * R.MEASURE_BINS,
                },
                "achieved_pusch_snr_db": {
                    "values": achieved_values,
                    "samples": len(achieved_values),
                    "p50": measured.achieved_pusch_snr_db_p50,
                },
                "corroboration": {
                    "primary_extdn_unique_application_payload_bytes":
                        payload_per_bin * R.MEASURE_BINS,
                    "ue_rlc_tx_sdu": {"events": 1, "bytes": 1},
                    "ue_rlc_tx_dequeue": {"events": 1, "bytes": 1},
                    "gnb_pdcp_rx_deliver": {"events": 1, "bytes": 1},
                    "backlog": {
                        "bin_backlogged": backlog_flags,
                        "tick_counts_per_bin": backlog_counts,
                        "minimum_backlog_per_bin": backlog_minima,
                        "backlogged_fraction": 0.95,
                    },
                    "corroboration_complete": True,
                },
                "post_probe_drain": {
                    "drained": True,
                    "quiet_interval_ns": 500_000_000,
                    "quiet_elapsed_ns": 600_000_000,
                    "no_new_pdcp_or_rlc_ingress_during_quiet_interval": True,
                },
            })
        result = {
            "schema": R.RESULT_SCHEMA,
            "stage_id": CQ.STAGE_ID,
            "status": R.STATUS_CAPTURED,
            "qualified": True,
            "failure": None,
            "radio_profile_id": R.RB.RADIO_PROFILE_ID,
            "primary_service_measurement":
                "EXT_DN_UNIQUE_SSBURST_APPLICATION_PAYLOAD_BYTES_PER_FIXED_100MS_MONOTONIC_WINDOW",
            "pusch_tb_is_primary": False,
            "probe_identity": CQ.verify_probe_action(ROOT),
            "probe_packetization": copy.deepcopy(packetization),
            "container_images": self._container_images(),
            "initial_drain": {
                "drained": True,
                "no_new_pdcp_or_rlc_ingress_during_quiet_interval": True,
            },
            "audit": audit,
            "adverse_capacity_mbps": 85.0,
            "selected_tiers": [tier.to_json() for tier in tiers],
            "points": records,
            "source_inventory": copy.deepcopy(self.inventory),
            "source_verifications": self._checks(),
            "radio_lineage": {"run_id": "offline-fixture"},
            "evidence_files": R._manifest_files(
                root, excluded=(R.RESULT_FILENAME, R.MANIFEST_FILENAME,
                                R.TERMINAL_FILENAME)),
            "final_cold_state": {
                "schema": "scenesense.capacity_final_cold_state.v1",
                "cold": True, "orphan_processes": {},
                "residual_ue_tunnels": [], "carla_processes": [],
                "probe_errors": [],
                "core_containers": {
                    name: "ABSENT" for name in self._container_images()},
            },
            "teardown": {
                "extract_ttracer_ok": True, "ran_notes": [],
                "core": {"stopped": True, "returncode": 0,
                         "core_after": {
                             name: "ABSENT" for name in self._container_images()}},
            },
        }
        result_path = root / R.RESULT_FILENAME
        R.write_json_create(result_path, result)
        self._reseal(root)
        return result_path

    def _reseal(self, root: Path) -> None:
        result_path = root / R.RESULT_FILENAME
        manifest_path = root / R.MANIFEST_FILENAME
        terminal_path = root / R.TERMINAL_FILENAME
        inventory = json.loads(result_path.read_text())["source_inventory"]
        manifest = {
            "schema": R.MANIFEST_SCHEMA,
            "status": R.STATUS_CAPTURED,
            "result_sha256": R.sha256_file(result_path),
            "source_inventory_sha256": inventory["inventory_sha256"],
            "files": R._manifest_files(
                root, excluded=(R.MANIFEST_FILENAME, R.TERMINAL_FILENAME)),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        terminal = {
            "schema": R.TERMINAL_SCHEMA,
            "status": R.STATUS_CAPTURED,
            "result_sha256": R.sha256_file(result_path),
            "manifest_sha256": R.sha256_file(manifest_path),
            "source_inventory_sha256": inventory["inventory_sha256"],
        }
        terminal_path.write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")

    def test_bound_result_reopens_only_after_full_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._make_bound_result(Path(temporary))
            self.assertTrue(R.verify_bound_capacity_result(result)["binding_verified"])

    def test_result_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._make_bound_result(Path(temporary))
            result.write_text(result.read_text() + " ")
            with self.assertRaises(R.CapacityRunError):
                R.verify_bound_capacity_result(result)

    def test_resealed_audit_recomputation_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            payload = json.loads(result.read_text())
            payload["audit"]["adverse_capacity_mbps"] = 86.0
            result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self._reseal(root)
            with self.assertRaisesRegex(R.CapacityRunError, "does not reproduce"):
                R.verify_bound_capacity_result(result)

    def test_resealed_primary_service_sample_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            payload = json.loads(result.read_text())
            payload["points"][0]["service_mbps_samples"][0] += 1.0
            result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self._reseal(root)
            with self.assertRaisesRegex(R.CapacityRunError, "ext-DN payload bins"):
                R.verify_bound_capacity_result(result)

    def test_resealed_capacity_percentile_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            payload = json.loads(result.read_text())
            payload["points"][1]["capacity_point"]["service_mbps_p50"] += 1.0
            result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self._reseal(root)
            with self.assertRaisesRegex(R.CapacityRunError,
                                        "percentiles do not reproduce"):
                R.verify_bound_capacity_result(result)

    def test_resealed_backlog_summary_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            payload = json.loads(result.read_text())
            payload["points"][1]["corroboration"]["backlog"][
                "backlogged_fraction"] = 1.0
            result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self._reseal(root)
            with self.assertRaisesRegex(R.CapacityRunError,
                                        "saturation summary does not reproduce"):
                R.verify_bound_capacity_result(result)

    def test_resealed_achieved_snr_sample_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            payload = json.loads(result.read_text())
            payload["points"][1]["achieved_pusch_snr_db"]["values"] = [
                value + 2.0 for value in payload["points"][1]["achieved_pusch_snr_db"]["values"]]
            result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self._reseal(root)
            with self.assertRaisesRegex(R.CapacityRunError, "summary does not reproduce"):
                R.verify_bound_capacity_result(result)

    def test_resealed_top_packetization_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            payload = json.loads(result.read_text())
            payload["probe_packetization"]["chunk_payload_bytes"] = 1_201
            result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self._reseal(root)
            with self.assertRaisesRegex(R.CapacityRunError, "packetization"):
                R.verify_bound_capacity_result(result)

    def test_resealed_point_packetization_tamper_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            payload = json.loads(result.read_text())
            payload["points"][0]["sink"]["expected_chunks_per_frame"] = 1_024
            result.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self._reseal(root)
            with self.assertRaisesRegex(R.CapacityRunError,
                                        "sink packetization summary"):
                R.verify_bound_capacity_result(result)

    def test_manifest_path_traversal_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            manifest_path = root / R.MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][0]["relative_path"] = "../escape"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            terminal_path = root / R.TERMINAL_FILENAME
            terminal = json.loads(terminal_path.read_text())
            terminal["manifest_sha256"] = R.sha256_file(manifest_path)
            terminal_path.write_text(json.dumps(terminal, indent=2) + "\n")
            with self.assertRaisesRegex(R.CapacityRunError, "canonical|escapes"):
                R.verify_bound_capacity_result(result)

    def test_unmanifested_extra_file_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self._make_bound_result(root)
            (root / "late_unbound_file.txt").write_text("not sealed\n")
            with self.assertRaisesRegex(R.CapacityRunError, "exact current file inventory"):
                R.verify_bound_capacity_result(result)

    def test_stale_current_source_inventory_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self._make_bound_result(Path(temporary))
            stale = copy.deepcopy(self.inventory)
            stale["inventory_sha256"] = "0" * 64
            with mock.patch.object(R, "source_inventory", return_value=stale):
                with self.assertRaisesRegex(R.CapacityRunError, "exact current"):
                    R.verify_bound_capacity_result(result)

    def test_source_inventory_covers_live_transitive_chain(self):
        files = self.inventory["repo_files"]
        for required in (
                "rl_agent/ue_mcs_backlog_near_capacity_v1/capacity_runner.py",
                "rl_agent/ue_mcs_backlog_near_capacity_v1/"
                "CAPACITY_RETRY_AMENDMENT_MTU_SAFE_V1.md",
                "rl_agent/ue_n3_structured_udp_receiver.py",
                "uplink_only_spatial_map_pipeline/run_splitfusion_oai_100mhz_4d5u_v1.sh",
                "OAI/oai-cn5g/docker-compose.yaml",
                "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
                "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem"):
            self.assertIn(required, files)

    def test_import_launches_no_process_socket_or_file_write(self):
        code = r'''
import builtins, socket, subprocess

def forbidden(*args, **kwargs):
    raise RuntimeError("side effect during import")
subprocess.Popen = forbidden
subprocess.run = forbidden
socket.socket = forbidden
builtins.open = forbidden
import rl_agent.ue_mcs_backlog_near_capacity_v1.capacity_runner
print("IMPORT_PURE")
'''
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=str(ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("IMPORT_PURE", completed.stdout)


class LifecycleHardeningTests(unittest.TestCase):
    @staticmethod
    def runner() -> R.Runner:
        runner = object.__new__(R.Runner)
        runner.config = {
            "capacity_qualification": {
                "subprocess_timeouts_s": {
                    "launcher": 180.0, "process_probe": 10.0,
                    "docker_inspect": 15.0, "core_down": 60.0,
                    "ttracer_extract": 180.0, "signal_command": 10.0,
                    "route_probe": 15.0,
                }
            },
            "radio": {"core_containers": ["oai-amf"]},
        }
        return runner

    def test_bounded_subprocess_timeout_fails_closed(self):
        runner = self.runner()
        with mock.patch.object(
                R.subprocess, "run",
                side_effect=subprocess.TimeoutExpired(["probe"], 10.0)):
            with self.assertRaisesRegex(R.CapacityRunError, "exceeded 10.0s"):
                runner._run_external(["probe"], timeout_name="process_probe")

    def test_container_binding_records_immutable_image_identity(self):
        runner = self.runner()
        image_id = "sha256:" + "a" * 64
        container = [{"Id": "container-id", "Image": image_id,
                      "Config": {"Image": "example/oai-amf:latest"}}]
        image = [{"Id": image_id,
                  "RepoDigests": ["example/oai-amf@sha256:" + "b" * 64]}]
        runner._run_external = mock.Mock(side_effect=[
            subprocess.CompletedProcess([], 0, json.dumps(container), ""),
            subprocess.CompletedProcess([], 0, json.dumps(image), ""),
        ])
        binding = runner._container_image_bindings()
        self.assertEqual(binding["oai-amf"]["image_id"], image_id)
        self.assertEqual(len(binding["oai-amf"]["repo_digests"]), 1)

    def test_final_cold_record_is_create_only(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            runner.output_dir = Path(temporary)
            runner._run_external = mock.Mock(return_value=
                subprocess.CompletedProcess([], 1, "", ""))
            runner._tunnel_interfaces = mock.Mock(return_value=[])
            runner._container_states = mock.Mock(return_value={"oai-amf": "ABSENT"})
            first = runner.final_cold_with_core()
            self.assertTrue(first["cold"])
            with self.assertRaises(FileExistsError):
                runner.final_cold_with_core()


class RegisteredConfigTests(unittest.TestCase):
    def test_config_registers_gates_corroboration_quiet_and_timeouts(self):
        config = json.loads(R.DEFAULT_CONFIG.read_text())
        self.assertIn("NR_RLC_TX_DEQUEUE", config["telemetry"]["events"]["ue"])
        self.assertIn("NR_RLC_TX_SDU", config["telemetry"]["events"]["ue"])
        self.assertIn("GNB_PDCP_RX_DELIVER", config["telemetry"]["events"]["gnb"])
        qualification = config["capacity_qualification"]
        service = qualification["service_measurement"]
        self.assertFalse(service["pusch_tb_is_primary"])
        self.assertEqual(service["drain_zero_consecutive_samples"], 5)
        self.assertEqual(service["drain_quiet_interval_s"], CQ.DRAIN_QUIET_INTERVAL_S)
        self.assertEqual(qualification["min_pusch_snr_samples"], 30)
        self.assertEqual(qualification["max_achieved_target_snr_error_db"], 1.0)
        self.assertEqual(set(qualification["subprocess_timeouts_s"]), {
            "launcher", "process_probe", "docker_inspect", "core_down",
            "ttracer_extract", "signal_command", "route_probe"})
        packetization = qualification["probe"]["packetization"]
        self.assertEqual(packetization, CQ.probe_packetization_identity())
        self.assertEqual(R.C.CHUNK_BYTES, 60_000)
        self.assertEqual(packetization["chunks_per_frame"], 2_974)
        self.assertGreater(packetization["chunks_per_frame"],
                           U3.MAX_CHUNKS_PER_FRAME_LIMIT)
        self.assertTrue(packetization["mtu_safe_without_ipv4_fragmentation"])

    def test_stage_plan_exposes_registered_gates(self):
        plan = CQ.stage_plan()
        self.assertFalse(plan["pusch_tb_is_primary"])
        self.assertEqual(plan["bootstrap"]["seed"], CQ.BOOTSTRAP_SEED)
        self.assertEqual(plan["inter_point_drain"]["consecutive_zero_ticks"], 5)
        self.assertEqual(plan["inter_point_drain"]["pdcp_rlc_quiet_interval_s"], 0.5)
        self.assertEqual(plan["min_pusch_snr_samples"], 30)


if __name__ == "__main__":
    unittest.main()
