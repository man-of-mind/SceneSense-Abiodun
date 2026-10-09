from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from ..state_reward_transition_contract import LocalizationCombiner, RewardSpecV1
from .contract import (
    REQUIRED_BEHAVIORAL_SOURCE_ROLES,
    EXPECTED_GRID_ROWS,
    PROVISIONAL_QUALITY_CALIBRATION,
    RUNTIME_BEHAVIORAL_ARTIFACTS,
    SFD1_COMMON_HEADER_BYTES,
    SFD1_CONTEXT_FIXED_BYTES,
    SFD1_CONTEXT_PROTOCOL_VERSION,
    UDP_CHUNK_HEADER_BYTES,
    UDP_CHUNK_BYTES_INCLUDING_HEADER,
    datagram_count,
    keep_count,
    canonical_json_bytes,
    canonical_sha256,
    repository_root,
    sfd1_accounting_identity,
    sfd1_accounting_stream_id,
)
from .manifest import (
    completion_manifest_document,
    run_manifest_document,
    row_schema_document,
    validate_completion_manifest,
)
from .quality import calibration_status, load_reward_spec
from .schema import finalize_row, validate_row
from .selection import (
    _candidate_record,
    _load_current_sweep_p40,
    assign_strata,
    describe_candidates,
    proportional_stratified_select,
    verify_selection_sources,
)
from .store import ExactRowStore
from .preflight import behavioral_source_bindings
from .contract import OfflineGridContractError
from .executor import (
    _frozen_detection_match_frame,
    _rehash_bound_runtime_sources,
    _rehash_selection_global_sources,
    _runtime_preflight_evidence,
    _verify_runtime_input_binding,
    _verify_runtime_scoring_binding,
    _wire_accounting,
)


def reward_spec() -> RewardSpecV1:
    return RewardSpecV1(
        spec_id="offline-grid-synthetic-hypothesis",
        spec_version=1,
        w_loc_person=0.6,
        w_loc_vehicle=0.4,
        tau_person_m=2.0,
        tau_vehicle_m=3.0,
        localization_combiner=LocalizationCombiner.WEIGHTED_GEOMETRIC_MEAN,
        w_seg_person=0.6,
        w_seg_vehicle=0.4,
        seg_reference_person_iou=0.6,
        seg_reference_vehicle_iou=0.8,
        segmentation_modulation_beta=0.3,
        w_quality=1.0,
        w_latency=0.5,
        lambda_mode=0.01,
        lambda_q=0.01,
        r_registered_failure=-1.0,
        gamma_per_tensor=0.99,
        provenance={"purpose": "unit-test", "calibration_status": "HYPOTHESIS"},
    )


def manifest(spec: RewardSpecV1) -> dict:
    return run_manifest_document(
        selection_manifest={
            "selection_manifest_sha256": "a" * 64,
            "selected_frame_count": 768,
        },
        preflight={
            "preflight_binding_sha256": "b" * 64,
            "source_bindings": {"frozen.py": "c" * 64},
            "source_roles": {"frozen.py": "synthetic frozen behavior"},
            "evaluation_source_bindings": {"ground_truth.csv": "e" * 64},
            "checkpoints": {
                "FCOS": {"path": "checkpoint.pt", "sha256": "d" * 64}
            },
        },
        reward_spec=spec,
    )


def partial_row(run_manifest: dict) -> dict:
    episode_id = "canonical_v3_05_val_30_30_s601_tm1601"
    mode_id = 0
    q_e4 = 5000
    stream_id = sfd1_accounting_stream_id(episode_id, mode_id)
    action_id, action_status = sfd1_accounting_identity(mode_id, q_e4)
    inner_bytes = 12_493
    outer_bytes = SFD1_COMMON_HEADER_BYTES + SFD1_CONTEXT_FIXED_BYTES + len(
        stream_id.encode("utf-8")
    )
    total_bytes = inner_bytes + outer_bytes
    datagrams = datagram_count(total_bytes)
    return {
        "run_binding_sha256": run_manifest["run_binding_sha256"],
        "selection_manifest_sha256": run_manifest["selection_manifest_sha256"],
        "episode_manifest_sha256": "e" * 64,
        "source_binding_sha256": "f" * 64,
        "episode_id": episode_id,
        "sample_id": "canonical_v3_05_val_30_30_s601_tm1601_000001_frame1",
        "frame_id": 1,
        "timestamp": 0.1,
        "grid_split": "fit",
        "selection_rank_within_split": 0,
        "inclusion_probability": 0.5,
        "sampling_weight": 2.0,
        "mode_id": mode_id,
        "family": "noAE",
        "quantizer": "UINT8",
        "q_e4": q_e4,
        "keep_count": keep_count(q_e4),
        "camera_si": 12.5,
        "camera_si_valid": True,
        "camera_si_status": "VALID",
        "radar_p40": 0.25,
        "radar_p40_valid": True,
        "radar_p40_status": "VALID",
        "scientific_inner_payload_bytes": inner_bytes,
        "scientific_inner_payload_sha256": hashlib.sha256(b"payload").hexdigest(),
        "sfd1_protocol_version": SFD1_CONTEXT_PROTOCOL_VERSION,
        "sfd1_action_id_field": action_id,
        "sfd1_action_identity_status": action_status,
        "sfd1_stream_id": stream_id,
        "sfd1_common_header_bytes": SFD1_COMMON_HEADER_BYTES,
        "sfd1_frame_context_bytes": SFD1_CONTEXT_FIXED_BYTES + len(stream_id.encode("utf-8")),
        "sfd1_outer_envelope_bytes": outer_bytes,
        "total_transmitted_bytes": total_bytes,
        "total_transmitted_sha256": hashlib.sha256(b"wire").hexdigest(),
        "datagram_count": datagrams,
        "udp_chunk_bytes_including_header": UDP_CHUNK_BYTES_INCLUDING_HEADER,
        "udp_chunk_header_bytes_per_datagram": UDP_CHUNK_HEADER_BYTES,
        "udp_chunk_header_bytes_total": datagrams * UDP_CHUNK_HEADER_BYTES,
        "udp_application_bytes": total_bytes + datagrams * UDP_CHUNK_HEADER_BYTES,
        "seg_vehicle_gt_pixels": 5,
        "seg_vehicle_pred_pixels": 4,
        "seg_vehicle_intersection_pixels": 3,
        "seg_vehicle_union_pixels": 6,
        "seg_person_gt_pixels": 0,
        "seg_person_pred_pixels": 0,
        "seg_person_intersection_pixels": 0,
        "seg_person_union_pixels": 0,
        "loc_vehicle_eligible_gt": 2,
        "loc_vehicle_tp": 1,
        "loc_vehicle_fp": 1,
        "loc_vehicle_fn": 1,
        "loc_vehicle_ignored_predictions": 0,
        "loc_vehicle_matched_xy_errors_m": [1.25],
        "loc_person_eligible_gt": 0,
        "loc_person_tp": 0,
        "loc_person_fp": 0,
        "loc_person_fn": 0,
        "loc_person_ignored_predictions": 0,
        "loc_person_matched_xy_errors_m": [],
    }


class OfflineQualityGridContractTest(unittest.TestCase):
    @staticmethod
    def _pose_fields() -> dict[str, str]:
        return {
            "anchor_x": "1.0", "anchor_y": "2.0", "anchor_z": "3.0",
            "anchor_pitch": "0.0", "anchor_yaw": "5.0", "anchor_roll": "0.0",
        }

    def test_frame_matcher_uses_the_frozen_detection_module_api(self) -> None:
        calls: list[tuple[object, object]] = []
        expected = ({0}, {1}, {0: 1})

        def match_frame(predictions, targets):
            calls.append((predictions, targets))
            return expected

        predictions = [{"class_name": "vehicle"}]
        targets = [{"class_name": "person"}, {"class_name": "vehicle"}]
        scorers = SimpleNamespace(
            detection=SimpleNamespace(match_frame=match_frame)
        )
        self.assertEqual(
            _frozen_detection_match_frame(scorers, predictions, targets), expected
        )
        self.assertEqual(calls, [(predictions, targets)])
        with self.assertRaisesRegex(RuntimeError, "does not expose match_frame"):
            _frozen_detection_match_frame(SimpleNamespace(), predictions, targets)

    def test_frame_matcher_seam_agrees_with_the_real_frozen_matcher(self) -> None:
        from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
            load_frozen_scorers,
        )

        scorers = load_frozen_scorers()
        predictions = [
            {"class_name": "vehicle", "world_x": 0.1, "world_y": 0.0},
            {"class_name": "vehicle", "world_x": 1.0, "world_y": 0.0},
            {"class_name": "person", "world_x": 0.0, "world_y": 0.0},
        ]
        targets = [
            {"class_name": "vehicle", "world_x": 0.0, "world_y": 0.0},
            {"class_name": "vehicle", "world_x": 2.0, "world_y": 0.0},
            {"class_name": "person", "world_x": 0.0, "world_y": 0.0},
        ]
        expected = scorers.detection.match_frame(predictions, targets)
        self.assertEqual(
            expected, ({0, 1, 2}, {0, 1, 2}, {0: 0, 1: 1, 2: 2})
        )
        self.assertEqual(
            _frozen_detection_match_frame(scorers, predictions, targets), expected
        )

    def test_behavioral_source_closure_has_required_paths_and_roles(self) -> None:
        hashes, roles = behavioral_source_bindings(repository_root())
        self.assertGreater(len(hashes), len(REQUIRED_BEHAVIORAL_SOURCE_ROLES))
        self.assertEqual(set(hashes), set(roles))
        for path, role in REQUIRED_BEHAVIORAL_SOURCE_ROLES.items():
            self.assertEqual(roles.get(path), role, path)
            self.assertRegex(hashes[path], r"^[0-9a-f]{64}$")
        for path, (digest, role) in RUNTIME_BEHAVIORAL_ARTIFACTS.items():
            self.assertEqual(hashes[path], digest)
            self.assertEqual(roles[path], role)

    def test_frozen_cardinality_and_exact_keep_rounding(self) -> None:
        self.assertEqual(EXPECTED_GRID_ROWS, 101_376)
        self.assertEqual(keep_count(0), 21_504)
        self.assertEqual(keep_count(5000), 10_752)
        self.assertEqual(keep_count(9800), 430)
        self.assertEqual(datagram_count(12_492), 1)
        self.assertEqual(datagram_count(12_493), 2)
        catalog = json.loads(
            (
                repository_root()
                / "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json"
            ).read_text(encoding="utf-8")
        )
        profiles = {int(row["action_id"]): row for row in catalog["profiles"]}
        for mode_id in range(12):
            for anchor_index, q_e4 in enumerate((0, 3000, 5000, 7000, 9000, 9800)):
                action_id, status = sfd1_accounting_identity(mode_id, q_e4)
                self.assertEqual(action_id, mode_id * 6 + anchor_index)
                self.assertEqual(status, "REGISTERED_ANCHOR_ACTION_ID")
                self.assertEqual(profiles[action_id]["q_e4"], q_e4)

    def test_row_retains_raw_evidence_and_marks_quality_provisional(self) -> None:
        spec = reward_spec()
        run = manifest(spec)
        row = finalize_row(partial_row(run), reward_spec=spec)
        validate_row(row)
        self.assertEqual(row["seg_vehicle_iou"], 0.5)
        self.assertEqual(row["loc_vehicle_recall"], 0.5)
        self.assertEqual(row["loc_vehicle_matched_xy_median_m"], 1.25)
        self.assertEqual(row["loc_person_status"], "UNDEFINED_NO_ELIGIBLE_GT")
        self.assertTrue(row["quality_valid"])
        self.assertEqual(row["quality_calibration_status"], PROVISIONAL_QUALITY_CALIBRATION)
        self.assertTrue(row["raw_sufficient_statistics_are_primary"])
        self.assertFalse(row["scalar_reward_weights_used_by_extraction"])
        self.assertEqual(row_schema_document()["primary_key"], "row_key_sha256")

    def test_append_store_refuses_duplicate_and_mixed_resume(self) -> None:
        spec = reward_spec()
        run = manifest(spec)
        row = finalize_row(partial_row(run), reward_spec=spec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.sqlite3"
            expected = frozenset({row["row_key_sha256"]})
            with ExactRowStore(path, run, resume=False, expected_row_keys=expected) as store:
                store.insert(row)
                self.assertEqual(store.row_count, 1)
                with self.assertRaisesRegex(OfflineGridContractError, "duplicate"):
                    store.insert(row)
            with ExactRowStore(path, run, resume=True, expected_row_keys=expected) as store:
                self.assertEqual(store.audit_rows()["rows"], 1)
            changed = dict(run)
            changed["run_binding_sha256"] = "0" * 64
            with self.assertRaises((OfflineGridContractError, ValueError)):
                ExactRowStore(path, changed, resume=True, expected_row_keys=expected)

    def test_candidate_description_does_not_hash_unselected_heavy_payloads(self) -> None:
        episode = "synthetic_ep"
        sample = "synthetic_ep_000001_frame1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_root = root / "data_collection/experiments/route_b_perception_v3" / episode
            (episode_root / "rgb").mkdir(parents=True)
            (episode_root / "radar").mkdir()
            rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            rgb[:, 640:] = 255
            self.assertTrue(cv2.imwrite(str(episode_root / "rgb/frame.png"), rgb))
            np.savez(
                episode_root / "radar/points.npz",
                sweep_offset=np.asarray([0, 0, -1], dtype=np.int32),
                original_range_m=np.asarray([10.0, 50.0, 5.0], dtype=np.float32),
            )
            row = {
                "experiment_id": episode,
                "sample_id": sample,
                "frame_id": "1",
                "timestamp": "0.1",
                "rgb_path": "rgb/frame.png",
                "radar_points_path": "radar/points.npz",
                # These heavy, unneeded candidate files deliberately do not exist.
                "mask_path": "heavy/mask.png",
                "instance_raw_path": "heavy/instances.npy",
                "radar_tensor_path": "heavy/tensor.npy",
                "vehicle_pixels": "10",
                "person_pixels": "0",
                **self._pose_fields(),
            }
            described = describe_candidates(
                root=root,
                episode_id=episode,
                grid_split="fit",
                rows=[row],
                executable_ids={sample},
                progress_every=0,
            )
            self.assertEqual(len(described), 1)
            assigned, _edges = assign_strata(described)
            selected, _report = proportional_stratified_select(assigned, 1)
            with self.assertRaisesRegex(OfflineGridContractError, "missing"):
                _candidate_record(root, selected[0][0], 0, 1.0, 1.0)

    def test_stratified_selection_is_deterministic_with_explicit_weights(self) -> None:
        # Reuse a fully validated Candidate shape and vary only stratification values.
        episode = "synthetic_ep"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_root = root / "data_collection/experiments/route_b_perception_v3" / episode
            (episode_root / "rgb").mkdir(parents=True)
            (episode_root / "radar").mkdir()
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(episode_root / "rgb/frame.png"), image))
            np.savez(
                episode_root / "radar/points.npz",
                sweep_offset=np.asarray([0], dtype=np.int32),
                original_range_m=np.asarray([20.0], dtype=np.float32),
            )
            row = {
                "experiment_id": episode,
                "sample_id": "synthetic_ep_0",
                "frame_id": "0",
                "timestamp": "0.0",
                "rgb_path": "rgb/frame.png",
                "radar_points_path": "radar/points.npz",
                "mask_path": "unused", "instance_raw_path": "unused",
                "radar_tensor_path": "unused", "vehicle_pixels": "0",
                "person_pixels": "0",
                **self._pose_fields(),
            }
            base = describe_candidates(
                root=root, episode_id=episode, grid_split="fit", rows=[row],
                executable_ids={"synthetic_ep_0"}, progress_every=0,
            )[0]
            candidates = [
                replace(
                    base,
                    sample_id=f"synthetic_ep_{index}", frame_id=index,
                    camera_si=float(index), radar_p40=index / 10.0,
                    vehicle_pixels=index * 10, vehicle_present=index > 0,
                    person_pixels=(7 - index) * 5, person_present=index < 7,
                )
                for index in range(8)
            ]
            assigned, _ = assign_strata(candidates)
            first, report = proportional_stratified_select(assigned, 4)
            second, _ = proportional_stratified_select(assigned, 4)
            self.assertEqual(
                [(item.sample_id, probability, weight) for item, probability, weight in first],
                [(item.sample_id, probability, weight) for item, probability, weight in second],
            )
            self.assertFalse(report["oversampled_with_replacement"])
            self.assertTrue(report["sampling_weights_stored"])

    def test_reward_spec_requires_exact_canonical_bytes_and_hash(self) -> None:
        spec = reward_spec()
        payload = canonical_json_bytes(spec.to_canonical_dict())
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_bytes(payload)
            self.assertEqual(load_reward_spec(path, digest), spec)
            with self.assertRaisesRegex(OfflineGridContractError, "drift"):
                load_reward_spec(path, "0" * 64)
            path.write_bytes(payload + b"\n")
            with self.assertRaisesRegex(OfflineGridContractError, "canonical"):
                load_reward_spec(path, hashlib.sha256(payload + b"\n").hexdigest())

    def test_invalid_current_sweep_is_not_filtered_into_a_valid_p40(self) -> None:
        invalid_vectors = (
            np.asarray([10.0, np.nan], dtype=np.float32),
            np.asarray([10.0, 0.0], dtype=np.float32),
            np.asarray([10.0, 120.01], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "radar.npz"
            for ranges in invalid_vectors:
                np.savez(
                    path,
                    sweep_offset=np.zeros(ranges.size, dtype=np.int32),
                    original_range_m=ranges,
                )
                value, valid, status, raw, accepted, rejected = _load_current_sweep_p40(path)
                self.assertIsNone(value)
                self.assertFalse(valid)
                self.assertEqual(status, "InvalidRadarRangesError")
                self.assertEqual((raw, accepted, rejected), (2, 1, 1))

    def test_self_rehashed_selection_tamper_fails_canonical_source_rebuild(self) -> None:
        rebuilt = {"schema": "synthetic", "camera_si": 1.0}
        tampered = {"schema": "synthetic", "camera_si": 99.0}
        tampered["selection_manifest_sha256"] = canonical_sha256(tampered)
        with mock.patch(
            "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.selection.validate_selection_manifest",
            return_value=tampered["selection_manifest_sha256"],
        ), mock.patch(
            "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.selection.build_selection_manifest",
            return_value=rebuilt,
        ):
            with self.assertRaisesRegex(OfflineGridContractError, "deterministic source rebuild"):
                verify_selection_sources(Path.cwd(), tampered)

    def test_caller_provenance_cannot_upgrade_quality_calibration(self) -> None:
        forged = replace(
            reward_spec(), provenance={"purpose": "unit-test", "calibration_status": "FROZEN"}
        )
        self.assertEqual(calibration_status(forged), PROVISIONAL_QUALITY_CALIBRATION)

    def test_exact_sfd1_and_udp_accounting_uses_total_not_inner_bytes(self) -> None:
        import torch
        from phase2_map_sharing.transport import CHUNK_HEADER, chunk_payload
        from rl_agent.splitfusion_live_dispatch_v1.envelope import (
            CONTEXT_FIXED_BYTES,
            CONTEXT_PROTOCOL_VERSION,
            HEADER_BYTES,
            pack_envelope,
            unpack_envelope,
        )
        from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
        from .contract import UDP_PAYLOAD_BYTES_PER_DATAGRAM

        self.assertFalse(torch.cuda.is_initialized())
        frame = {
            "episode_id": "synthetic_episode",
            "frame_id": 17,
            "timestamp": 1.25,
            "ego_world_pose": {
                "x": 1.0, "y": 2.0, "z": 3.0,
                "pitch": 0.0, "yaw": 5.0, "roll": 0.0,
            },
        }
        stream = sfd1_accounting_stream_id(frame["episode_id"], 0)
        outer_bytes = HEADER_BYTES + CONTEXT_FIXED_BYTES + len(stream.encode("utf-8"))
        inner = b"x" * (UDP_PAYLOAD_BYTES_PER_DATAGRAM - outer_bytes + 1)
        accounting = _wire_accounting(
            modules={
                "build_frame_context_v1": build_frame_context_v1,
                "pack_envelope": pack_envelope,
                "unpack_envelope": unpack_envelope,
                "CONTEXT_PROTOCOL_VERSION": CONTEXT_PROTOCOL_VERSION,
                "CONTEXT_FIXED_BYTES": CONTEXT_FIXED_BYTES,
                "HEADER_BYTES": HEADER_BYTES,
                "CHUNK_HEADER": CHUNK_HEADER,
                "chunk_payload": chunk_payload,
            },
            frame=frame,
            mode_id=0,
            q_e4=4000,
            inner_payload=inner,
        )
        self.assertEqual(accounting["scientific_inner_payload_bytes"], len(inner))
        self.assertEqual(accounting["total_transmitted_bytes"], UDP_PAYLOAD_BYTES_PER_DATAGRAM + 1)
        self.assertEqual(accounting["datagram_count"], 2)
        self.assertEqual(
            accounting["sfd1_action_identity_status"],
            "EXACT_V2_SIZE_ACCOUNTING_WITH_NONDISPATCHABLE_DYNAMIC_ACTION_SENTINEL",
        )
        self.assertFalse(torch.cuda.is_initialized())

    def test_transport_accounting_tamper_is_rejected(self) -> None:
        spec = reward_spec()
        row = finalize_row(partial_row(manifest(spec)), reward_spec=spec)
        changed = dict(row)
        changed["total_transmitted_bytes"] += 1
        with self.assertRaisesRegex(OfflineGridContractError, "accounting drift"):
            validate_row(changed)

    def test_runtime_input_symlink_and_bytes_are_reverified(self) -> None:
        class Inference:
            def __init__(self, dataset: Path) -> None:
                self.dataset = dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = "episode05"
            raw = root / "data_collection/experiments/route_b_perception_v3" / episode
            (raw / "rgb").mkdir(parents=True)
            (raw / "radar_tensors").mkdir()
            (raw / "rgb/frame.jpg").write_bytes(b"rgb")
            (raw / "radar_tensors/frame.npy").write_bytes(b"radar")
            dataset = root / "model/dataset"
            dataset.mkdir(parents=True)
            (dataset / episode).symlink_to(raw, target_is_directory=True)
            source_paths = {
                "rgb_path": "rgb/frame.jpg",
                "radar_tensor_path": "radar_tensors/frame.npy",
            }
            frame = {
                "episode_id": episode,
                "sample_id": "episode05_frame1",
                "source_paths": source_paths,
                "source_sha256": {
                    name: hashlib.sha256((raw / relative).read_bytes()).hexdigest()
                    for name, relative in source_paths.items()
                },
            }
            inference_row = {
                name: f"{episode}/{relative}" for name, relative in source_paths.items()
            }
            with mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.executor.repository_root",
                return_value=root,
            ):
                observed = _verify_runtime_input_binding(
                    frame=frame,
                    inference=Inference(dataset),
                    inference_row=inference_row,
                )
                self.assertEqual(observed, frame["source_sha256"])
                (raw / "radar_tensors/frame.npy").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "bytes drifted"):
                    _verify_runtime_input_binding(
                        frame=frame,
                        inference=Inference(dataset),
                        inference_row=inference_row,
                    )

    def test_runtime_source_tamper_after_preflight_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bound.py"
            evaluation = root / "evaluation.csv"
            checkpoint = root / "checkpoint.pt"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            evaluation.write_text("truth\n", encoding="utf-8")
            checkpoint.write_bytes(b"weights")
            bound = {
                "run_binding": {
                    "source_module_sha256": {
                        "bound.py": hashlib.sha256(source.read_bytes()).hexdigest()
                    },
                    "evaluation_source_sha256": {
                        "evaluation.csv": hashlib.sha256(evaluation.read_bytes()).hexdigest()
                    },
                    "checkpoint_sha256": {
                        "checkpoint.pt": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    },
                }
            }
            with mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.executor.repository_root",
                return_value=root,
            ):
                self.assertEqual(_rehash_bound_runtime_sources(bound)["files"], 3)
                evaluation.write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                    _rehash_bound_runtime_sources(bound)

    def test_selection_global_truth_is_rehashed_immediately(self) -> None:
        from .contract import EPISODES

        names = (
            "manifest.csv", "metadata.json", "resolved_config.json",
            "object_boxes.csv", "object_visibility.csv", "depth_frames.csv",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for episode in EPISODES:
                episode_root = root / episode.root_relpath
                episode_root.mkdir(parents=True)
                expected[episode.episode_id] = {}
                for name in names:
                    path = episode_root / name
                    path.write_text(f"{episode.episode_id}:{name}\n", encoding="utf-8")
                    expected[episode.episode_id][name] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
            selection = {"source_global_sha256": expected}
            with mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.executor.repository_root",
                return_value=root,
            ):
                self.assertEqual(_rehash_selection_global_sources(selection)["files"], 12)
                changed = root / EPISODES[0].root_relpath / "object_boxes.csv"
                changed.write_text("changed\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "changed before truth load"):
                    _rehash_selection_global_sources(selection)

    def test_scoring_masks_are_rehashed_at_use_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_id = "frame_1"
            dataset = root / "dataset"
            relatives = {
                "segmentation_gt_path":
                    f"contracts/v010/val/segmentation_masks/{sample_id}.png",
                "object_ignore_mask_path":
                    f"contracts/v010/val/object_ignore_masks/{sample_id}.png",
            }
            hashes = {}
            paths = {}
            for name, relative in relatives.items():
                path = dataset / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode("ascii"))
                paths[name] = str(path.relative_to(root))
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = {
                "sample_id": sample_id,
                "evaluation_source_paths": paths,
                "evaluation_source_sha256": hashes,
            }
            with mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.executor.repository_root",
                return_value=root,
            ):
                self.assertEqual(
                    _verify_runtime_scoring_binding(
                        frame=frame, runtime={"dataset_root": dataset}
                    ),
                    hashes,
                )
                (dataset / relatives["object_ignore_mask_path"]).write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "bytes drifted"):
                    _verify_runtime_scoring_binding(
                        frame=frame, runtime={"dataset_root": dataset}
                    )

    def test_runtime_preflight_tensor_payload_becomes_json_evidence(self) -> None:
        import torch

        tensor = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        original = {
            "checkpoint_payloads": {"AE32": {"autoencoder": {"w": tensor}}}
        }
        evidence = _runtime_preflight_evidence(original, torch)
        descriptor = evidence["checkpoint_payloads"]["AE32"]["autoencoder"]["w"]
        self.assertEqual(descriptor["record"], "checkpoint_tensor_evidence_v1")
        self.assertEqual(descriptor["shape"], [1, 2])
        self.assertEqual(descriptor["dtype"], "torch.float32")
        self.assertEqual(descriptor["storage_bytes"], 8)
        self.assertRegex(descriptor["value_bytes_sha256"], r"^[0-9a-f]{64}$")
        self.assertIs(
            original["checkpoint_payloads"]["AE32"]["autoencoder"]["w"], tensor
        )
        self.assertIsInstance(json.dumps(evidence, allow_nan=False), str)
        self.assertFalse(torch.cuda.is_initialized())

    def test_runtime_preflight_refuses_unbound_non_json_objects(self) -> None:
        import torch

        with self.assertRaisesRegex(RuntimeError, "unsupported evidence type"):
            _runtime_preflight_evidence({"unexpected": object()}, torch)
        self.assertEqual(
            _runtime_preflight_evidence({5000: {"q": 0.5}}, torch),
            {"5000": {"q": 0.5}},
        )
        with self.assertRaisesRegex(RuntimeError, "normalization collision"):
            _runtime_preflight_evidence({5000: 1, "5000": 2}, torch)

    def test_complete_manifest_requires_exact_complete_store(self) -> None:
        initial = manifest(reward_spec())
        execution = {"status": "COMPLETE"}
        audit = {"complete": True, "rows": EXPECTED_GRID_ROWS, "expected_rows": EXPECTED_GRID_ROWS}
        complete = completion_manifest_document(
            initial_manifest=initial,
            execution_result=execution,
            artifact_sha256={"quality_rows.sqlite3": "a" * 64},
            store_audit=audit,
        )
        self.assertEqual(validate_completion_manifest(complete), complete["run_manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "exact expected-key"):
            completion_manifest_document(
                initial_manifest=initial,
                execution_result=execution,
                artifact_sha256={"quality_rows.sqlite3": "a" * 64},
                store_audit={"complete": False, "rows": EXPECTED_GRID_ROWS - 1},
            )


if __name__ == "__main__":
    unittest.main()
