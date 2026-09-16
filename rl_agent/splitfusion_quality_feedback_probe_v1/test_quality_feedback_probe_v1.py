#!/usr/bin/env python3
"""Focused, non-live regression tests for the privileged quality probe."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_direct_edge_map_v1.ue_ledger import DirectTerminalLedger

from . import protocol
from .coordinator import BoundedQualityCoordinator, QualityCoordinatorError
from .gt_evidence import (
    GroundTruthEvidenceError,
    read_ground_truth,
    write_object_ground_truth,
    write_semantic_ground_truth,
)
from .ledger import QualityFeedbackLedger, QualityLedgerError
from .scoring import (
    QualityInputs,
    mean_or_nan as edge_mean_or_nan,
    oriented_footprint_iou as edge_oriented_footprint_iou,
    require_exact_parity,
    score_concurrent,
    score_serial,
    segmentation_quality_columns as edge_segmentation_quality_columns,
)


def identity(frame: int = 7) -> dict[str, object]:
    return {
        "run_id": "quality-run",
        "cell_id": "a50__adverse_stable",
        "stream_id": "ue-quality",
        "frame_id": frame,
        "action_id": 50,
        "profile_id": "split_ae64_uint4_q5000",
        "capture_timestamp_ns": time.time_ns() - 5_000_000,
    }


def objects() -> list[dict[str, object]]:
    return [
        {
            "class_name": "vehicle", "world_x": 1.0, "world_y": 2.0,
            "world_z": 0.0, "size_x": 4.0, "size_y": 2.0,
            "size_z": 1.5, "model_yaw_deg": 5.0,
        },
        {
            "class_name": "person", "world_x": 3.0, "world_y": 4.0,
            "world_z": 0.0, "size_x": 0.5, "size_y": 0.4,
            "size_z": 1.7, "model_yaw_deg": 0.0,
        },
    ]


def quality() -> dict[str, object]:
    return {
        "segmentation": {
            "miou_vehicle_iou": 0.8, "miou_person_iou": 0.5,
            "miou_3class_macro": 0.7, "gt_vehicle_pixels": 20,
            "gt_person_pixels": 5,
        },
        "localization": {
            name: {
                "recall": 1.0, "source_time_world_xy_error_m": 0.1,
                "footprint_iou": 0.9, "tp": 1, "fn": 0,
            }
            for name in ("vehicle", "person")
        },
    }


def success_ack(value: dict[str, object]) -> dict[str, object]:
    base = time.time_ns()
    timing = {name: base + index for index, name in enumerate(protocol.TIMING_FIELDS)}
    detail = protocol.build_detail(
        identity_fields=value, frozen_carla_frame_id=int(value["frame_id"]),
        timing=timing, quality=quality(), evaluator_mode="TEST",
    )
    return protocol.build_ack(
        identity_fields=value, frozen_carla_frame_id=int(value["frame_id"]),
        timing=timing, quality=quality(), evaluator_mode="TEST",
        detail_sha256=protocol.detail_digest(detail),
    )


class ProtocolTests(unittest.TestCase):
    def test_compact_nonterminal_privileged_ack_is_sub_mtu_and_record_free(self) -> None:
        message = success_ack(identity())
        encoded = protocol.canonical_bytes(message)
        self.assertLessEqual(len(encoded), 1200)
        self.assertFalse(any(token in encoded for token in (b'"records"', b'"objects"', b'"semantic_labels"')))
        self.assertTrue(message["nt"])
        self.assertTrue(message["pg"])
        self.assertFalse(message["dp"])

    def test_negative_or_inverted_timing_is_rejected(self) -> None:
        message = success_ack(identity())
        message["t"][protocol.TIMING_FIELDS.index("evaluation_started_wall_ns")] = 1
        with self.assertRaises(protocol.QualityProtocolError):
            protocol.validate(message)

    def test_every_identity_component_is_part_of_the_join_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = QualityFeedbackLedger(Path(directory) / "quality.csv")
            base = identity()
            ledger.register(base, registered_wall_ns=time.time_ns())
            for field in protocol.IDENTITY_FIELDS:
                mutated = identity()
                if field in {"frame_id", "action_id", "capture_timestamp_ns"}:
                    mutated[field] = int(mutated[field]) + 1
                else:
                    mutated[field] = str(mutated[field]) + "-wrong"
                with self.assertRaises(QualityLedgerError, msg=field):
                    ledger.record(
                        success_ack(mutated), received_wall_ns=time.time_ns() + 1_000_000,
                        message_bytes=500, source_address="10.0.0.1:1",
                    )
            ledger.close()

    def test_sigterm_unwinds_runtime_and_closes_report_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "closed"
            ready = Path(directory) / "ready"
            code = "\n".join(
                (
                    "import signal, sys",
                    "from pathlib import Path",
                    "from rl_agent.splitfusion_quality_feedback_probe_v1.signal_shutdown import run_with_shutdown_handlers",
                    "marker, ready = Path(sys.argv[1]), Path(sys.argv[2])",
                    "def owned_runtime():",
                    "    ready.write_text('ready', encoding='utf-8')",
                    "    try: signal.pause()",
                    "    finally: marker.write_text('closed', encoding='utf-8')",
                    "run_with_shutdown_handlers(owned_runtime)",
                )
            )
            process = subprocess.Popen(
                [sys.executable, "-c", code, str(marker), str(ready)],
                cwd=str(Path(__file__).resolve().parents[2]),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(os.environ),
            )
            deadline = time.monotonic() + 10.0
            while (
                not ready.exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            if not ready.exists():
                process.kill()
                stdout, stderr = process.communicate(timeout=5.0)
                self.fail(f"signal-test child not ready: {stdout!r} {stderr!r}")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=10.0)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
            self.assertEqual(marker.read_text(encoding="utf-8"), "closed")


class ScoringAndEvidenceTests(unittest.TestCase):
    @staticmethod
    def _normalize_formula_result(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): ScoringAndEvidenceTests._normalize_formula_result(item)
                for key, item in value.items()
            }
        if isinstance(value, (float, np.floating)) and np.isnan(float(value)):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return value

    def test_edge_formula_replicas_match_qualified_host_evaluator(self) -> None:
        from rl_agent.ue_route_b_split_cell_adapter_v1 import (
            mean_or_nan as qualified_mean_or_nan,
            oriented_footprint_iou as qualified_oriented_footprint_iou,
            segmentation_quality_columns as qualified_segmentation_quality_columns,
        )

        mask_cases = (
            (
                np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.uint8),
                np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.uint8),
            ),
            (
                np.zeros((3, 5), dtype=np.uint8),
                np.zeros((2, 3), dtype=np.uint8),
            ),
            (
                np.asarray([[1, 1], [2, 0]], dtype=np.uint8),
                np.asarray([[2]], dtype=np.uint8),
            ),
        )
        for predicted, truth in mask_cases:
            self.assertEqual(
                self._normalize_formula_result(
                    edge_segmentation_quality_columns(predicted, truth)
                ),
                self._normalize_formula_result(
                    qualified_segmentation_quality_columns(predicted, truth)
                ),
            )

        footprint_cases = (
            (
                {"world_x": 0, "world_y": 0, "size_x": 4, "size_y": 2,
                 "model_yaw_deg": 15},
                {"world_x": 0.5, "world_y": -0.25, "size_x": 4.2,
                 "size_y": 1.9, "yaw_deg": -10},
            ),
            (
                {"world_x": 5, "world_y": 8, "size_x": 0.5, "size_y": 0.4,
                 "yaw_sin": 1, "yaw_cos": 0},
                {"world_x": 5, "world_y": 8, "size_x": 0.5, "size_y": 0.4,
                 "yaw_sin": 1, "yaw_cos": 0},
            ),
            (
                {"world_x": -2, "world_y": 1, "size_x": 1, "size_y": 1,
                 "yaw_deg": ""},
                {"world_x": 20, "world_y": 20, "size_x": 1, "size_y": 1},
            ),
        )
        for prediction, truth in footprint_cases:
            self.assertEqual(
                edge_oriented_footprint_iou(prediction, truth),
                qualified_oriented_footprint_iou(prediction, truth),
            )

        for values in ([1.0, 2.0, 4.0], [float("nan"), 2.0], [], [float("inf")]):
            self.assertEqual(
                self._normalize_formula_result(edge_mean_or_nan(values)),
                self._normalize_formula_result(qualified_mean_or_nan(values)),
            )

    def test_serial_and_concurrent_exact_parity(self) -> None:
        mask = np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.uint8)
        inputs = QualityInputs.own(
            predicted_mask=mask, ground_truth_mask=mask,
            predictions=objects(), ground_truth_objects=objects(),
            match_distance_m=3.0,
        )
        require_exact_parity(score_serial(inputs), score_concurrent(inputs))

    def test_gt_requires_exact_frame_schema_shape_dtype_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = identity()
            mask = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
            write_semantic_ground_truth(
                root, identity=item, frozen_carla_frame_id=7, mask=mask
            )
            write_object_ground_truth(
                root, identity=item, frozen_carla_frame_id=7, rows=objects()
            )
            loaded = read_ground_truth(root, expected_identity=item, timeout_s=0.01)
            self.assertTrue(np.array_equal(loaded["semantic"], mask))
            with self.assertRaises(GroundTruthEvidenceError):
                write_object_ground_truth(
                    root, identity={**item, "frame_id": 8},
                    frozen_carla_frame_id=7, rows=objects(),
                )


class LedgerAndCoordinatorTests(unittest.TestCase):
    def test_quality_ledger_does_not_close_map_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = DirectTerminalLedger(
                output_csv=root / "map.csv", experiment_id="e", cell_id="c"
            )
            direct.register_capture(
                stream_id="ue-quality", capture_id="ue-quality:7", frame_id=7,
                capture_at=time.time(), action_id="50",
                profile_id="split_ae64_uint4_q5000",
                service_deadline_at=time.time() + 1,
                ack_timeout_at=time.time() + 2,
            )
            quality_ledger = QualityFeedbackLedger(root / "quality.csv")
            item = identity()
            quality_ledger.register(item, registered_wall_ns=time.time_ns())
            ack = success_ack(item)
            quality_ledger.record(
                ack, received_wall_ns=time.time_ns() + 1_000_000,
                message_bytes=len(protocol.canonical_bytes(ack)),
                source_address="10.0.0.1:9",
            )
            self.assertEqual(direct.summary()["obligations_open"], 1)
            self.assertEqual(quality_ledger.summary()["quality_ack_outcomes"], 1)
            quality_ledger.close()
            direct.close()

    def test_early_segmentation_really_completes_before_object_gt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = identity()
            mask = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
            write_semantic_ground_truth(
                root, identity=item, frozen_carla_frame_id=7, mask=mask
            )
            messages: list[dict[str, object]] = []
            details: list[dict[str, object]] = []
            coordinator = BoundedQualityCoordinator(
                gt_directory=root, emit_ack=messages.append,
                record_detail=details.append, parity_sample_limit=1,
            )
            coordinator.submit_early_segmentation(item, mask)
            key = coordinator._key(item)
            deadline = time.monotonic() + 1.0
            while not coordinator._early[key].score_future.done() and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(coordinator._early[key].score_future.done())
            write_object_ground_truth(
                root, identity=item, frozen_carla_frame_id=7, rows=objects()
            )
            now, mono = time.time_ns(), time.monotonic_ns()
            coordinator.submit_final(
                identity=item, frozen_carla_frame_id=7, records=objects(),
                final_mask=mask, production_mask_for_parity=mask,
                model_ready_wall_ns=now - 1000,
                model_ready_monotonic_ns=mono - 1000,
                final_prediction_ready_wall_ns=now,
                final_prediction_ready_monotonic_ns=mono,
            )
            report = coordinator.close()
            self.assertFalse(report["failures"])
            self.assertEqual(len(messages), 1)
            self.assertTrue(details[0]["timing"]["early_segmentation_branch"])

    def test_parity_sample_compares_early_mask_with_production_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = identity()
            early = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
            production = np.asarray([[0, 2], [2, 0]], dtype=np.uint8)
            write_semantic_ground_truth(
                root, identity=item, frozen_carla_frame_id=7, mask=early
            )
            write_object_ground_truth(
                root, identity=item, frozen_carla_frame_id=7, rows=objects()
            )
            coordinator = BoundedQualityCoordinator(
                gt_directory=root, emit_ack=lambda _row: None,
                parity_sample_limit=1,
            )
            coordinator.submit_early_segmentation(item, early)
            now, mono = time.time_ns(), time.monotonic_ns()
            coordinator.submit_final(
                identity=item, frozen_carla_frame_id=7, records=objects(),
                final_mask=early, production_mask_for_parity=production,
                model_ready_wall_ns=now - 1000,
                model_ready_monotonic_ns=mono - 1000,
                final_prediction_ready_wall_ns=now,
                final_prediction_ready_monotonic_ns=mono,
            )
            report = coordinator.close()
            self.assertTrue(
                any("early semantic branch differs" in value for value in report["failures"])
            )

    def test_identical_duplicate_is_diagnostic_but_conflict_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = QualityFeedbackLedger(Path(directory) / "quality.csv")
            item = identity()
            message = success_ack(item)
            ledger.register(item, registered_wall_ns=time.time_ns())
            args = {
                "received_wall_ns": time.time_ns() + 2_000_000,
                "message_bytes": len(protocol.canonical_bytes(message)),
                "source_address": "10.0.0.1:5",
            }
            ledger.record(message, **args)
            ledger.record(message, **args)
            conflict = dict(message)
            conflict["q"] = list(conflict["q"])
            conflict["q"][0] = 0.1
            with self.assertRaises(QualityLedgerError):
                ledger.record(conflict, **args)
            self.assertEqual(ledger.summary()["identical_duplicates"], 1)
            ledger.close()


if __name__ == "__main__":
    unittest.main()
