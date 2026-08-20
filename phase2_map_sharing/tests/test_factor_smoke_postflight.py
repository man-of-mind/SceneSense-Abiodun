from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd
import yaml

from data_collection.validate_phase2_factor_realization_smoke import (
    ContractError,
    build_plan,
    load_config,
    validate_results,
)
from data_collection.run_phase2_calibration_audit import (
    _persist_completed_factor_pair_postflights,
)
from phase2_map_sharing.factor_smoke_postflight import (
    DEPENDENCY_PATHS,
    _captured_role_provenance,
    _retention_window_evidence,
    _verify_role_artifact_manifest,
    analyze_batch_artifacts,
    canonical_sha256,
)
from phase2_map_sharing.factor_smoke_runtime_contract import (
    CausalPolicyRuntimeAuditor,
    FEATURE_SOURCE_STAGE,
    FeatureComponent,
    FeatureSample,
    RecipientAvailabilityRecorder,
    analyze_installed_track_guardrails,
    build_recipient_available_endpoint,
    build_recipient_map_target_match,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _samples(config: dict, names: list[str]) -> dict[str, FeatureSample]:
    fixtures = set(config["policy_projection_exercise"]["fixture_backed_fields"])
    abstracted = set(
        config["policy_projection_exercise"]
        ["local_loopback_transport_abstracted_fields"]["fields"]
    )
    result = {}
    for name in names:
        stage = FEATURE_SOURCE_STAGE[name]
        result[name] = FeatureSample(
            value=0.0,
            source_stage=stage,
            observed_at_s=0.8,
            available_at_s=0.9,
            component_provenance=(
                (
                    FeatureComponent("helper_localization", 0.8, 0.8),
                    FeatureComponent("recipient_state_transport", 0.8, 0.9),
                )
                if stage == "derived_relative_kinematics"
                else ()
            ),
            evidence_kind=(
                "preregistered_fixture"
                if name in fixtures
                else "local_loopback_transport_abstraction"
                if name in abstracted
                else "observed"
            ),
        )
    return result


class SyntheticExact16:
    """Tiny artifact tree exercising the real assembler and validator boundary."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = load_config()
        design = Path(self.config["source_design"]["design_config"])
        self.config["source_design"]["design_config_sha256"] = _sha(design)
        for name, path in DEPENDENCY_PATHS.items():
            self.config["recipient_endpoint_runtime"]["dependency_sha256"][name] = {
                "path": str(path),
                "sha256": _sha(path),
            }
        self.plan = build_plan(self.config)
        self.checkpoint = root / "frozen_m_prime.pt"
        self.checkpoint.write_bytes(b"frozen-model-checkpoint")
        self.model_sha = _sha(self.checkpoint)
        (root / "plan.json").write_text("{}\n", encoding="utf-8")
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "synthetic",
                    "ambient_traffic": {
                        "traffic_sanity_gate": {
                            "minimum_static_collision_horizontal_impulse": 50.0
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        self.postflights: dict[str, dict] = {}
        batch_rows = []
        for row in self.plan["rows"]:
            batch_rows.append(self._make_trajectory(row))
        _write_json(
            root / "batch_manifest.json",
            {"schema": "synthetic.raw.exact16", "trajectories": batch_rows},
        )

    def _role_tree(self, trajectory_dir: Path, role: str, frames: list[int], times: list[float]) -> None:
        role_dir = trajectory_dir / role
        metrics_path = role_dir / "streams/synthetic_metrics.csv"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "frame_id": frames,
                "carla_timestamp": [100.0 + value for value in times],
            }
        ).to_csv(metrics_path, index=False)
        entries = []
        for frame in frames:
            path = role_dir / f"retained_inputs/frame_{frame}_inputs.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{role}:{frame}".encode())
            entries.append(
                {
                    "path": str(path.relative_to(role_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )
        entries.append(
            {
                "path": str(metrics_path.relative_to(role_dir)),
                "bytes": metrics_path.stat().st_size,
                "sha256": _sha(metrics_path),
            }
        )
        _write_json(
            role_dir / "artifact_manifest.json",
            {
                "trajectory_id": trajectory_dir.name,
                "source_role": role,
                "files": entries,
            },
        )

    def _traffic_tree(self, trajectory_dir: Path) -> dict:
        traffic_dir = trajectory_dir / "traffic_sanity"
        traffic_dir.mkdir(parents=True, exist_ok=True)
        for name, header in (
            ("npc_collision_events.csv", ["frame_id", "actor_id"]),
            ("npc_trajectories.csv", ["frame_id", "actor_id"]),
            ("ambient_actor_trajectories.csv", ["frame_id", "actor_id"]),
        ):
            with (traffic_dir / name).open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(header)
        traffic = {
            "pass": True,
            "collision_events": 0,
            "collision_callback_rows": 0,
            "ignored_static_contact_rows": 0,
            "collision_events_by_owner_scope": {},
            "failures": [],
            "trajectory_csv": str((traffic_dir / "npc_trajectories.csv").resolve()),
            "ambient_actor_trajectory_csv": str(
                (traffic_dir / "ambient_actor_trajectories.csv").resolve()
            ),
        }
        _write_json(traffic_dir / "traffic_sanity_summary.json", traffic)
        return traffic

    def _make_trajectory(self, row: dict) -> dict:
        trajectory_id = row["trajectory_id"]
        directory = self.root / trajectory_id
        scenario = directory / "scenario"
        scenario.mkdir(parents=True)
        requested = copy.deepcopy(row["requested_factor_contract"])
        positive = bool(row["controlled_hazard_present"])
        onset = max(float(requested["requested_hazard_onset_s"]), 0.1) if positive else 1.0
        first_elapsed = max(0.0, onset - 1.0)
        elapsed = [first_elapsed + 0.1 * index for index in range(40)]
        frames = list(range(1000, 1040))
        pd.DataFrame({"frame_id": frames, "elapsed_s": elapsed}).to_csv(
            scenario / "realized_trace.csv", index=False
        )
        for role in ("helper", "recipient"):
            self._role_tree(directory, role, frames, elapsed)
        static_token = directory / "static_environment_truth/static_environment_objects.csv"
        static_token.parent.mkdir(parents=True)
        static_token.write_text("id,class\n1,Car\n", encoding="utf-8")

        context = "c" * 64
        factor = {
            "trajectory_id": trajectory_id,
            "trajectory_row_sha256": row["trajectory_row_sha256"],
            "scenario_role": row["scenario_role"],
            "requested_factors": requested,
            "nontreatment_plan_sha256": context,
        }
        if positive:
            factor.update(
                {
                    "factor_realization_gate": {"pass": True},
                    "realized_factors": self._realized(requested, onset),
                }
            )
        else:
            factor.update(
                {
                    "registered_target_absent": True,
                    "realized_factors_status": (
                        "not_applicable_matched_benign_registered_target_absent"
                    ),
                    "factor_reference_trajectory_id": (
                        trajectory_id.removesuffix("_ben") + "_pos"
                    ),
                }
            )
        _write_json(scenario / "factor_realization.json", factor)
        traffic = self._traffic_tree(directory)
        gate = {"pass": True, "failures": []}
        batch_record = {
            "trajectory_id": trajectory_id,
            "status": "complete",
            "traffic_sanity": traffic,
            "trajectory_verification": {"pass": True, "trajectory_id": trajectory_id},
            "matched_pair_initial_realization_gate": gate,
            "matched_pair_owned_nontreatment_gate": gate,
            "matched_pair_static_environment_gate": gate,
            "matched_pair_full_trajectory_gate": gate,
        }
        self.postflights[trajectory_id] = self._postflight(
            row, directory, onset if positive else None, static_token
        )
        return batch_record

    @staticmethod
    def _realized(requested: dict, onset: float) -> dict:
        return {
            "realized_hazard_onset_s": onset,
            "realized_helper_speed_mps": requested["requested_helper_speed_mps"],
            "realized_recipient_speed_mps": requested["requested_recipient_speed_mps"],
            "realized_hazard_actor_speed_mps": requested["requested_hazard_actor_speed_mps"],
            "realized_onset_driver_speed_mps": requested["requested_onset_driver_speed_mps"],
            "pre_intervention_radial_closing_speed_mps": requested["requested_closing_speed_target_mps"],
            "pre_intervention_hazard_proximity_horizon_s": requested["requested_proximity_horizon_target_s"],
            "pre_intervention_minimum_surface_clearance_m": 1.0,
            "geometry_measurement_basis": requested["geometry_measurement_basis"],
            "closing_speed_measurement_basis": requested["closing_speed_measurement_basis"],
            "proximity_horizon_measurement_basis": requested["proximity_horizon_measurement_basis"],
        }

    def _audit(self, trajectory_id: str) -> dict:
        auditor = CausalPolicyRuntimeAuditor.from_config(
            self.config,
            trajectory_id=trajectory_id,
            arm_id="fixed_local_loopback_projection_audit",
            clock_id="carla_simulation_time",
            decision_locus="helper",
        )
        auditor.record_policy_state_exposure(
            sample_at_s=0.5, source_track_count=0, installed_map_track_count=0
        )
        feature = self.config["policy_feature_contract"]
        placement = _samples(self.config, feature["placement_features"])
        publication = _samples(self.config, feature["publication_features"])
        actions = self.config["policy_projection_exercise"]["fixed_actions"]
        auditor.consume(
            stage="placement",
            decision_id=f"{trajectory_id}:placement",
            decision_at_s=1.0,
            action=actions["placement"],
            samples=placement,
        )
        auditor.consume(
            stage="publication",
            decision_id=f"{trajectory_id}:publication",
            decision_at_s=1.0,
            action=actions["publication"],
            samples=publication,
        )
        auditor.exercise_forbidden_canary(
            stage="placement",
            decision_id=f"{trajectory_id}:canary",
            decision_at_s=1.0,
            action=actions["placement"],
            valid_samples=placement,
        )
        return auditor.to_record()

    def _availability(self, trajectory_id: str, positive: bool, hazard_class: str):
        recorder = RecipientAvailabilityRecorder(
            trajectory_id=trajectory_id, clock_id="carla_simulation_time"
        )
        matches = []
        truth = {}
        endpoint = None
        if positive:
            helper_obs, recipient_obs = "a" * 64, "b" * 64
            recorder.register_source_observation(
                source_role="helper", source_track_id="helper-target",
                observation_sha256=helper_obs, observed_at_s=0.8,
            )
            recorder.record_source_confirmation(
                source_role="helper", source_track_id="helper-target", confirmed_at_s=1.0
            )
            recorder.register_source_observation(
                source_role="recipient", source_track_id="recipient-target",
                observation_sha256=recipient_obs, observed_at_s=1.8,
            )
            recorder.record_source_confirmation(
                source_role="recipient", source_track_id="recipient-target", confirmed_at_s=2.0
            )
            recorder.record_recipient_local_install(
                local_install_id="recipient-local-1", source_track_id="recipient-target",
                source_observation_sha256=recipient_obs, recipient_map_track_id="map-target",
                confirmed_at_s=2.0, installed_at_s=2.05, available_at_s=2.1,
            )
            recorder.record_install_attempt(
                attempt_id="helper-attempt-1", contribution_id="helper-contribution-1",
                source_role="helper", source_track_id="helper-target",
                source_observation_sha256=helper_obs, published_at_s=1.1,
                attempted_at_s=1.2, install_status="accepted",
                recipient_map_track_id="map-target", installed_at_s=1.2,
                available_at_s=1.3,
            )
        availability = recorder.to_record()
        if positive:
            for kind, ref, role, track, available in (
                ("helper_install_attempt", "helper-attempt-1", "helper", "helper-target", 1.3),
                ("recipient_local_install", "recipient-local-1", "recipient", "recipient-target", 2.1),
            ):
                matches.append(
                    build_recipient_map_target_match(
                        trajectory_id=trajectory_id, install_kind=kind,
                        install_ref_id=ref, source_role=role, source_track_id=track,
                        recipient_map_track_id="map-target", available_at_s=available,
                        canonical_map_state={"class_name": hazard_class, "x_m": 1.0, "y_m": 2.0, "snapshot_at_s": available},
                        target_truth_state={"class_name": hazard_class, "x_m": 1.0, "y_m": 2.0, "observed_at_s": available},
                        center_gate_m=5.0,
                    )
                )
            truth = {"helper-attempt-1": True}
            endpoint = build_recipient_available_endpoint(
                availability, helper_source_track_id="helper-target",
                recipient_source_track_id="recipient-target",
                recipient_map_track_id="map-target", evaluation_horizon_s=10.0,
                evaluation_recipient_map_target_matches=matches,
            )
        return availability, truth, matches, endpoint

    def _postflight(self, row: dict, directory: Path, onset: float | None, static_token: Path) -> dict:
        positive = bool(row["controlled_hazard_present"])
        availability, truth, matches, endpoint = self._availability(
            row["trajectory_id"], positive, row["hazard_class"]
        )
        checkpoint_identity = {
            role: {
                "model_sha256": self.model_sha,
                "config_sha256": "d" * 64,
                "checkpoint_hash_status": "capture_time_sha256_recomputed_equal",
                "checkpoint_path_at_capture": str(self.checkpoint.resolve()),
                "checkpoint_sha256_at_capture": self.model_sha,
                "checkpoint_sha256_recomputed": self.model_sha,
                "checkpoint_sha256_equal": True,
                "checkpoint_identity_basis": "capture_time_file_bytes",
                "manifest_sha256": "e" * 64,
            }
            for role in ("helper", "recipient")
        }
        dependencies = {
            name: {"path": str(path), "sha256": _sha(path)}
            for name, path in DEPENDENCY_PATHS.items()
        }
        result = {
            "schema": "scenesense.phase2_factor_smoke_postflight.v1",
            "trajectory_id": row["trajectory_id"],
            "transport_scope": "local_loopback_only_no_oai_claim",
            "warnings_generated": False,
            "recipient_availability_provenance": availability,
            "installed_track_guardrails": analyze_installed_track_guardrails(
                availability, evaluation_truth_match_by_attempt_id=truth
            ),
            "causal_policy_audit": self._audit(row["trajectory_id"]),
            "evaluation_truth_match_by_attempt_id": truth,
            "evaluation_truth_coverage": {"basis": "synthetic_test_fixture"},
            "tracker_diagnostics": {"helper": {}, "recipient": {}},
            "role_model_provenance": checkpoint_identity,
            "retention_window_evidence": _retention_window_evidence(
                directory, realized_onset_s=onset
            ),
            "dependency_fingerprints": dependencies,
            "input_fingerprints": {str(static_token.resolve()): _sha(static_token)},
            "collector_artifact_manifests": {
                role: _verify_role_artifact_manifest(directory / role)
                for role in ("helper", "recipient")
            },
        }
        if positive:
            result["evaluation_recipient_map_target_matches"] = matches
            result["installed_track_endpoint"] = endpoint
        result["postflight_sha256"] = canonical_sha256(result)
        return result

    def analyze(self):
        with mock.patch(
            "phase2_map_sharing.factor_smoke_postflight.analyze_trajectory_artifacts",
            side_effect=lambda *, trajectory_dir, trajectory_row, smoke_config: copy.deepcopy(
                self.postflights[trajectory_row["trajectory_id"]]
            ),
        ):
            return analyze_batch_artifacts(
                batch_root=self.root,
                smoke_config=self.config,
                factor_plan=self.plan,
                write_outputs=True,
            )


class Exact16AssemblerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticExact16(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact16_assembler_reaches_canonical_atomic_pass(self):
        result, validation = self.fixture.analyze()
        self.assertEqual(validation["verdict"], "PASS_ATOMIC_EXACT_16_ADMITTED")
        self.assertEqual(len(result["trajectories"]), 16)
        self.assertEqual(
            len(
                {
                    role["model_sha256"]
                    for row in result["trajectories"]
                    for role in row["capture_model_identity"].values()
                }
            ),
            1,
        )
        self.assertEqual(result["policy_feature_projection"]["trajectory_audit_count"], 16)

    def _admitted_then_tamper(self, relative: str) -> tuple[dict, Path]:
        result, _ = self.fixture.analyze()
        first = self.fixture.plan["rows"][0]["trajectory_id"]
        path = self.fixture.root / first / relative
        path.write_bytes(path.read_bytes() + b"tamper")
        return result, path

    def test_static_truth_tamper_is_rejected(self):
        result, _ = self._admitted_then_tamper(
            "static_environment_truth/static_environment_objects.csv"
        )
        with self.assertRaisesRegex(ContractError, "postflight input bytes drifted"):
            validate_results(result, self.fixture.config, self.fixture.plan)

    def test_role_listed_input_tamper_is_rejected(self):
        result, _ = self._admitted_then_tamper(
            "helper/retained_inputs/frame_1000_inputs.npz"
        )
        with self.assertRaisesRegex(ContractError, "artifact bytes drifted"):
            validate_results(result, self.fixture.config, self.fixture.plan)

    def test_factor_artifact_tamper_is_rejected(self):
        result, _ = self._admitted_then_tamper("scenario/factor_realization.json")
        with self.assertRaisesRegex(ContractError, "factor artifact bytes drifted"):
            validate_results(result, self.fixture.config, self.fixture.plan)

    def test_postflight_artifact_tamper_is_rejected(self):
        result, _ = self._admitted_then_tamper("scenario/factor_smoke_postflight.json")
        with self.assertRaisesRegex(ContractError, "postflight artifact bytes drifted"):
            validate_results(result, self.fixture.config, self.fixture.plan)

    def test_structural_traffic_artifact_tamper_is_rejected(self):
        result, _ = self._admitted_then_tamper("traffic_sanity/npc_collision_events.csv")
        with self.assertRaisesRegex(ContractError, "traffic-sanity artifact drifted"):
            validate_results(result, self.fixture.config, self.fixture.plan)

    def test_ignored_static_settlement_callback_does_not_fail_collision_gate(self):
        first = self.fixture.plan["rows"][0]["trajectory_id"]
        directory = self.fixture.root / first / "traffic_sanity"
        collision_path = directory / "npc_collision_events.csv"
        with collision_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "frame_id", "npc_actor_id", "contact_owner_scope",
                    "other_actor_id", "other_type_id", "normal_impulse_x",
                    "normal_impulse_y", "normal_impulse_z",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "frame_id": 1000, "npc_actor_id": 7,
                    "contact_owner_scope": "controlled_ego",
                    "other_actor_id": 0, "other_type_id": "static.ground",
                    "normal_impulse_x": 0.0, "normal_impulse_y": 0.0,
                    "normal_impulse_z": 200.0,
                }
            )
        batch_path = self.fixture.root / "batch_manifest.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        record = next(item for item in batch["trajectories"] if item["trajectory_id"] == first)
        record["traffic_sanity"]["collision_callback_rows"] = 1
        record["traffic_sanity"]["ignored_static_contact_rows"] = 1
        _write_json(directory / "traffic_sanity_summary.json", record["traffic_sanity"])
        _write_json(batch_path, batch)
        _, validation = self.fixture.analyze()
        self.assertEqual(validation["verdict"], "PASS_ATOMIC_EXACT_16_ADMITTED")

    def test_pair_postflight_waits_for_second_row_then_writes_both_once(self):
        pair_root = self.fixture.root / "pair_order"
        positive = {
            "trajectory_id": "pair_pos", "group_id": "pair",
            "scenario_role": "controlled_positive_occlusion", "status": "complete",
        }
        benign = {
            "trajectory_id": "pair_ben", "group_id": "pair",
            "scenario_role": "matched_benign_negative", "status": "complete",
        }
        batch = {"trajectories": [positive]}
        plan = {"rows": [{"trajectory_id": "pair_pos"}, {"trajectory_id": "pair_ben"}]}

        def write_postflight(*, trajectory_dir, trajectory_row, smoke_config):
            path = Path(trajectory_dir) / "scenario/factor_smoke_postflight.json"
            payload = {"postflight_sha256": "f" * 64}
            _write_json(path, payload)
            return payload

        with mock.patch(
            "phase2_map_sharing.factor_smoke_postflight.analyze_and_persist_trajectory_artifacts",
            side_effect=write_postflight,
        ) as analyzer:
            self.assertEqual(
                _persist_completed_factor_pair_postflights(
                    batch=batch, group_id="pair", output_dir=pair_root,
                    factor_smoke_config={}, factor_smoke_plan=plan,
                ),
                0,
            )
            self.assertEqual(analyzer.call_count, 0)
            gate = {"pass": True}
            for record in (positive, benign):
                for name in (
                    "matched_pair_initial_realization_gate",
                    "matched_pair_owned_nontreatment_gate",
                    "matched_pair_static_environment_gate",
                    "matched_pair_full_trajectory_gate",
                ):
                    record[name] = gate
            batch["trajectories"].append(benign)
            self.assertEqual(
                _persist_completed_factor_pair_postflights(
                    batch=batch, group_id="pair", output_dir=pair_root,
                    factor_smoke_config={}, factor_smoke_plan=plan,
                ),
                2,
            )
            self.assertEqual(
                _persist_completed_factor_pair_postflights(
                    batch=batch, group_id="pair", output_dir=pair_root,
                    factor_smoke_config={}, factor_smoke_plan=plan,
                ),
                0,
            )
            self.assertEqual(analyzer.call_count, 2)


class CaptureCheckpointIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.role_dirs = {
            role: self.root / "trajectory" / role for role in ("helper", "recipient")
        }
        self.checkpoint = self.root / "model.pt"
        self.checkpoint.write_bytes(b"one frozen model")
        for role in self.role_dirs:
            self._write_role(role, self.checkpoint)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_role(self, role: str, checkpoint: Path, *, omit_sha: bool = False) -> None:
        directory = self.role_dirs[role] / "manifests"
        directory.mkdir(parents=True, exist_ok=True)
        identity = {
            "checkpoint_path_at_capture": str(checkpoint.resolve()),
            "checkpoint_identity_basis": "capture_time_file_bytes",
        }
        if not omit_sha:
            identity["checkpoint_sha256"] = _sha(checkpoint)
        manifest = {
            **identity,
            "checkpoint_path": str(checkpoint.resolve()),
            "phase2_paired_causal": dict(identity),
        }
        _write_json(directory / f"{role}_manifest.json", manifest)
        _write_json(directory / f"{role}_resolved_config.json", {"role": role})

    def test_capture_checkpoint_identity_recomputes_and_matches_roles(self):
        identity = _captured_role_provenance(self.role_dirs)
        self.assertEqual(identity["helper"]["model_sha256"], _sha(self.checkpoint))
        self.assertTrue(identity["recipient"]["checkpoint_sha256_equal"])

    def test_missing_capture_time_checkpoint_sha_is_rejected(self):
        self._write_role("helper", self.checkpoint, omit_sha=True)
        with self.assertRaisesRegex(ValueError, "root/nested checkpoint identity differs|lacks capture-time"):
            _captured_role_provenance(self.role_dirs)

    def test_checkpoint_mutation_after_manifest_is_rejected(self):
        self.checkpoint.write_bytes(b"mutated after capture")
        with self.assertRaisesRegex(ValueError, "differ from capture-time"):
            _captured_role_provenance(self.role_dirs)

    def test_cross_role_checkpoint_difference_is_rejected(self):
        other = self.root / "other.pt"
        other.write_bytes(b"different detector")
        self._write_role("recipient", other)
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            _captured_role_provenance(self.role_dirs)


if __name__ == "__main__":
    unittest.main()
