"""Adversarial tests for the Run-4 quality/payload bridge."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import random
import socket
import subprocess
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import held_payload
from rl_agent.splitfusion_hybrid_sac_run4_v1 import quality_adapter as adapter
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import scientific_basis
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.corrected_p40_sidecar import (
    load_exact_corrected_p40_sidecar,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_quality_surface import (
    EXACT_GRID_ROW_EVIDENCE,
    MODELED_SAME_FRAME_EVIDENCE,
    BUNDLE_RELATIVE_PATH,
    EmpiricalQualitySurface,
    QualityComponent,
    SplitAccessError,
    load_empirical_quality_surface,
)
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.contract import (
    Q_E4_GRID,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
    UnreconciledActionIdentityError,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class _StubRng:
    def __init__(self, draw: float) -> None:
        self.draw = draw
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.draw


def _curves(
    *, selection_sha: str, database_sha: str
) -> tuple[held_payload.ScenePayloadCurveV1, ...]:
    result = []
    for scene in range(2):
        probability = (0.5, 0.25)[scene]
        for mode_id in range(12):
            base = 1_500_000 + scene * 20_000 + mode_id * 10_000
            nodes = tuple(
                held_payload.PayloadNodeV1(
                    q_e4=q_e4,
                    total_transmitted_bytes=base - 100 * q_e4,
                    source_row_sha256=_digest(
                        f"held-row:{scene}:{mode_id}:{q_e4}"
                    ),
                )
                for q_e4 in Q_E4_GRID
            )
            result.append(
                held_payload.ScenePayloadCurveV1(
                    sample_id=f"train-held-{scene}",
                    episode_id=f"train-episode-{scene}",
                    frame_id=9000 + scene,
                    selection_rank_within_train=scene,
                    source_split="train",
                    inclusion_probability=probability,
                    sampling_weight=1.0 / probability,
                    mode_id=mode_id,
                    scene_source_sha256=_digest(f"held-scene:{scene}"),
                    source_selection_sha256=selection_sha,
                    source_database_sha256=database_sha,
                    nodes=nodes,
                )
            )
    return tuple(result)


class Run4QualityPayloadAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.sidecar = load_exact_corrected_p40_sidecar(root=cls.root)
        cls.surface = load_empirical_quality_surface(
            cls.sidecar, project_root=cls.root
        )
        cls.catalog = action_contract.load_contract()
        material = _curves(
            selection_sha=cls.surface.binding.selection_file_sha256,
            database_sha=cls.surface.binding.database_file_sha256,
        )
        cls.held_provider = held_payload.HeldPayloadProviderV1(
            material,
            expected_inventory_sha256=(
                held_payload.payload_curve_inventory_sha256(material)
            ),
        )
        cls.surface_binding_sha = adapter.surface_binding_sha256(
            cls.surface.binding
        )
        cls.bridge = adapter.Run4QualityPayloadAdapterV1(
            surface=cls.surface,
            held_provider=cls.held_provider,
            expected_surface_binding_sha256=cls.surface_binding_sha,
            expected_held_provider_binding_sha256=(
                cls.held_provider.binding_sha256
            ),
        )
        cls.fit_record = next(
            record for record in cls.sidecar.records if record.grid_split == "fit"
        )
        cls.held_record = next(
            record
            for record in cls.sidecar.records
            if record.grid_split == "held_scene"
        )
        seed_query = cls.surface.query_fit_q_e4(
            cls.fit_record.sample_id, 0, 3000
        )
        cls.selection = adapter.FitSceneSelectionV1.from_query(
            seed_query, surface_binding_sha256=cls.surface_binding_sha
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.surface.close()

    @classmethod
    def action(cls, mode_id: int = 0, q_e4: int = 3000) -> ExecutedActionIdentity:
        executable = cls.catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, cls.catalog
        )

    def test_binding_includes_reviewed_quality_scientific_basis(self) -> None:
        self.assertEqual(
            self.bridge.binding.scientific_basis_sha256,
            scientific_basis.SCIENTIFIC_BASIS_SHA256,
        )
        original = self.bridge.binding
        object.__setattr__(
            self.bridge,
            "binding",
            replace(original, scientific_basis_sha256=_digest("changed-basis")),
        )
        try:
            with self.assertRaisesRegex(
                adapter.QualityAdapterBindingError, "scientific-basis"
            ):
                self.bridge.reward_tensor(
                    selection=self.selection,
                    action=self.action(),
                    tensor_seq=9,
                )
        finally:
            object.__setattr__(self.bridge, "binding", original)

    @staticmethod
    def counter(ordinal: int = 1) -> held_payload.HeldSelectionCounterV1:
        return held_payload.HeldSelectionCounterV1(
            session_id="quality-adapter-session",
            decision_seq=7,
            held_ordinal=ordinal,
            rng_stream_id="quality-adapter-local-rng",
        )

    def test_reward_exact_node_preserves_direct_quality_payload_and_digests(self) -> None:
        action = self.action(q_e4=3000)
        result = self.bridge.reward_tensor(
            selection=self.selection, action=action, tensor_seq=10
        )
        source = self.surface.query_fit_q_e4(
            self.selection.sample_id, action.mode_id, action.q_e4
        )
        self.assertEqual(result.q_perc, source.policy.component("q_perc").value)
        self.assertEqual(
            result.tensor.offered_payload_bytes,
            source.policy.payload.total_transmitted_bytes,
        )
        self.assertIs(type(result.tensor.offered_payload_bytes), int)
        self.assertTrue(result.tensor.reward_requested)
        self.assertEqual(
            result.tensor.payload_evidence_class,
            run4_contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE,
        )
        self.assertEqual(
            result.evidence.surface_evidence_status, EXACT_GRID_ROW_EVIDENCE
        )
        self.assertEqual(
            result.tensor.payload_provenance_sha256,
            result.evidence.canonical_sha256,
        )
        self.assertEqual(
            tuple(item.row_sha256 for item in result.evidence.endpoints),
            tuple(item.row_sha256 for item in source.hidden.endpoint_evidence),
        )

    def test_reward_off_anchor_is_same_scene_interpolation_not_nearest_anchor(self) -> None:
        action = self.action(q_e4=2000)
        result = self.bridge.reward_tensor(
            selection=self.selection, action=action, tensor_seq=11
        )
        source = self.surface.query_fit_q_e4(
            self.selection.sample_id, action.mode_id, action.q_e4
        )
        self.assertEqual(
            source.policy.evidence_status, MODELED_SAME_FRAME_EVIDENCE
        )
        self.assertEqual(result.q_perc, source.policy.component("q_perc").value)
        self.assertEqual(
            result.tensor.offered_payload_bytes,
            source.policy.payload.total_transmitted_bytes,
        )
        self.assertEqual(
            result.tensor.payload_evidence_class,
            run4_contract.PayloadEvidenceClass.MODELED_SAME_SCENE_INTERPOLATION,
        )
        self.assertEqual(
            tuple(item.q_e4 for item in result.evidence.endpoints), (1500, 3000)
        )

    def test_policy_scene_exposes_exactly_camera_si_and_radar_p40(self) -> None:
        result = self.bridge.reward_tensor(
            selection=self.selection, action=self.action(), tensor_seq=12
        )
        self.assertEqual(
            tuple(item.name for item in fields(result.policy_scene)),
            ("camera_si", "radar_p40"),
        )
        self.assertEqual(
            set(result.policy_scene.to_dict()), {"camera_si", "radar_p40"}
        )
        self.assertEqual(
            result.policy_scene_features(),
            (self.selection.camera_si, self.selection.radar_p40),
        )
        serialized = repr(result.policy_scene.to_dict())
        for hidden in ("sample_id", "episode_id", "frame_id", "grid_split"):
            self.assertNotIn(hidden, serialized)

    def test_held_tensor_uses_provider_and_can_never_request_reward(self) -> None:
        action = self.action(mode_id=3, q_e4=2000)
        rng = _StubRng(0.0)
        result = self.bridge.held_tensor(
            counter=self.counter(1),
            rng=rng,
            action=action,
            tensor_seq=13,
        )
        self.assertFalse(result.tensor.reward_requested)
        self.assertEqual(rng.calls, 1)
        self.assertEqual(
            (result.estimate.mode_id, result.estimate.q_e4),
            (action.mode_id, action.q_e4),
        )
        self.assertEqual(
            result.tensor.payload_provenance_sha256,
            result.estimate.canonical_sha256,
        )
        self.assertEqual(
            result.tensor.payload_evidence_class,
            run4_contract.PayloadEvidenceClass.MODELED_SAME_SCENE_INTERPOLATION,
        )
        self.assertFalse(hasattr(result, "q_perc"))

    def test_held_tensor_exact_node_keeps_integer_measured_class(self) -> None:
        result = self.bridge.held_tensor(
            counter=self.counter(2),
            rng=_StubRng(0.5),
            action=self.action(mode_id=4, q_e4=5000),
            tensor_seq=14,
        )
        self.assertIs(type(result.tensor.offered_payload_bytes), int)
        self.assertEqual(
            result.tensor.payload_evidence_class,
            run4_contract.PayloadEvidenceClass.MEASURED_EXACT_ACTION_NODE,
        )
        self.assertFalse(result.tensor.reward_requested)

    def test_held_scene_is_categorically_unavailable_for_reward(self) -> None:
        held = self.surface.evaluate_held(self.held_record.sample_id, 0, 0.3)
        with self.assertRaises(adapter.FitSceneRejected):
            adapter.FitSceneSelectionV1.from_query(
                held, surface_binding_sha256=self.surface_binding_sha
            )
        with self.assertRaises(SplitAccessError):
            self.surface.query_fit_q_e4(self.held_record.sample_id, 0, 3000)

    def test_selection_identity_and_scene_join_are_rechecked(self) -> None:
        genuine = self.surface.query_fit_q_e4(
            self.selection.sample_id, 0, 3000
        )
        foreign = self.surface.query_fit_q_e4(
            next(
                record.sample_id
                for record in self.sidecar.records
                if record.grid_split == "fit"
                and record.sample_id != self.selection.sample_id
            ),
            0,
            3000,
        )
        with mock.patch.object(
            EmpiricalQualitySurface,
            "query_fit_q_e4",
            return_value=foreign,
        ):
            with self.assertRaises(adapter.FitSceneRejected):
                self.bridge.reward_tensor(
                    selection=self.selection,
                    action=self.action(),
                    tensor_seq=15,
                )
        self.assertNotEqual(genuine.hidden.sample_id, foreign.hidden.sample_id)

    def test_action_surface_join_is_rechecked(self) -> None:
        genuine = self.surface.query_fit_q_e4(
            self.selection.sample_id, 0, 3000
        )
        forged_policy = replace(genuine.policy, mode_id=1)
        forged = replace(genuine, policy=forged_policy)
        with mock.patch.object(
            EmpiricalQualitySurface,
            "query_fit_q_e4",
            return_value=forged,
        ):
            with self.assertRaises(adapter.ActionJoinError):
                self.bridge.reward_tensor(
                    selection=self.selection,
                    action=self.action(),
                    tensor_seq=16,
                )

    def test_invalid_or_out_of_range_qperc_fails_closed(self) -> None:
        genuine = self.surface.query_fit_q_e4(
            self.selection.sample_id, 0, 3000
        )
        for component in (
            QualityComponent("q_perc", None, False, "UNDEFINED_TEST"),
            QualityComponent("q_perc", 1.1, True, "OUT_OF_RANGE_TEST"),
        ):
            quality = tuple(
                component if item.name == "q_perc" else item
                for item in genuine.policy.quality
            )
            forged = replace(genuine, policy=replace(genuine.policy, quality=quality))
            with self.subTest(component=component):
                with mock.patch.object(
                    EmpiricalQualitySurface,
                    "query_fit_q_e4",
                    return_value=forged,
                ):
                    with self.assertRaises(adapter.QualityUnavailable):
                        self.bridge.reward_tensor(
                            selection=self.selection,
                            action=self.action(),
                            tensor_seq=17,
                        )

    def test_result_attestation_fields_cannot_drift_independently(self) -> None:
        result = self.bridge.reward_tensor(
            selection=self.selection, action=self.action(), tensor_seq=171
        )
        with self.assertRaises(adapter.QualityUnavailable):
            replace(result, q_perc=result.q_perc - 0.01)
        with self.assertRaises(adapter.FitSceneRejected):
            replace(
                result,
                policy_scene=replace(
                    result.policy_scene,
                    camera_si=result.policy_scene.camera_si + 0.01,
                ),
            )
        with self.assertRaises(adapter.QualityAdapterError):
            replace(
                result,
                tensor=replace(
                    result.tensor,
                    offered_payload_bytes=result.tensor.offered_payload_bytes + 1,
                ),
            )

    def test_unreconciled_or_subclassed_action_is_refused(self) -> None:
        action = self.action()
        unreconciled = replace(action, _reconciliation=None)
        with self.assertRaises(UnreconciledActionIdentityError):
            self.bridge.reward_tensor(
                selection=self.selection,
                action=unreconciled,
                tensor_seq=18,
            )

        class ForeignAction(ExecutedActionIdentity):
            pass

        foreign = ForeignAction(**action.to_canonical_dict())
        with self.assertRaises(adapter.ActionJoinError):
            self.bridge.reward_tensor(
                selection=self.selection,
                action=foreign,
                tensor_seq=18,
            )

    def test_selection_and_provider_binding_mismatches_fail_closed(self) -> None:
        wrong_selection = replace(
            self.selection, surface_binding_sha256=_digest("wrong-surface")
        )
        with self.assertRaises(adapter.QualityAdapterBindingError):
            self.bridge.reward_tensor(
                selection=wrong_selection, action=self.action(), tensor_seq=19
            )
        with self.assertRaises(adapter.QualityAdapterBindingError):
            adapter.Run4QualityPayloadAdapterV1(
                surface=self.surface,
                held_provider=self.held_provider,
                expected_surface_binding_sha256=_digest("not-this-surface"),
                expected_held_provider_binding_sha256=(
                    self.held_provider.binding_sha256
                ),
            )

    def test_cross_source_held_provider_is_refused(self) -> None:
        material = _curves(
            selection_sha=_digest("foreign-selection"),
            database_sha=self.surface.binding.database_file_sha256,
        )
        provider = held_payload.HeldPayloadProviderV1(
            material,
            expected_inventory_sha256=(
                held_payload.payload_curve_inventory_sha256(material)
            ),
        )
        with self.assertRaises(adapter.QualityAdapterBindingError):
            adapter.Run4QualityPayloadAdapterV1(
                surface=self.surface,
                held_provider=provider,
                expected_surface_binding_sha256=self.surface_binding_sha,
                expected_held_provider_binding_sha256=provider.binding_sha256,
            )

    def test_surface_binding_drift_after_construction_is_detected(self) -> None:
        original = self.surface.binding
        object.__setattr__(
            self.surface,
            "binding",
            replace(original, database_file_sha256=_digest("changed-database")),
        )
        try:
            with self.assertRaises(adapter.QualityAdapterBindingError):
                self.bridge.reward_tensor(
                    selection=self.selection,
                    action=self.action(),
                    tensor_seq=20,
                )
        finally:
            object.__setattr__(self.surface, "binding", original)

    def test_bool_and_negative_tensor_sequence_are_refused(self) -> None:
        for invalid in (True, -1, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(adapter.QualityAdapterError):
                    self.bridge.reward_tensor(
                        selection=self.selection,
                        action=self.action(),
                        tensor_seq=invalid,  # type: ignore[arg-type]
                    )

    def test_adapter_contains_no_latency_network_or_hidden_policy_features(self) -> None:
        result = self.bridge.reward_tensor(
            selection=self.selection, action=self.action(), tensor_seq=21
        )
        policy_keys = set(result.policy_scene.to_dict())
        self.assertEqual(policy_keys, {"camera_si", "radar_p40"})
        for forbidden in (
            "latency",
            "network",
            "profile",
            "sample_id",
            "episode_id",
            "frame_id",
        ):
            self.assertTrue(all(forbidden not in key for key in policy_keys))

    def test_z_import_performs_no_io_rng_socket_process_or_evidence_load(self) -> None:
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("filesystem access")
        ) as opened, mock.patch.object(random, "seed") as seeded, mock.patch.object(
            random, "random"
        ) as sampled, mock.patch.object(
            socket, "socket", side_effect=AssertionError("socket opened")
        ) as socket_opened, mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("process launched")
        ) as popen, mock.patch.object(
            EmpiricalQualitySurface,
            "load",
            side_effect=AssertionError("evidence loaded"),
        ) as loaded:
            importlib.reload(adapter)
        opened.assert_not_called()
        seeded.assert_not_called()
        sampled.assert_not_called()
        socket_opened.assert_not_called()
        popen.assert_not_called()
        loaded.assert_not_called()

    def test_real_evidence_files_are_read_only_during_queries(self) -> None:
        database = self.root / BUNDLE_RELATIVE_PATH / "quality_rows.sqlite3"
        before = database.stat()
        self.bridge.reward_tensor(
            selection=self.selection, action=self.action(), tensor_seq=22
        )
        after = database.stat()
        self.assertEqual((before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
