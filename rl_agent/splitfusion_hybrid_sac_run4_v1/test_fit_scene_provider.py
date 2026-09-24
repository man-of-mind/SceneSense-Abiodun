"""Adversarial tests for the Run-4 train-only fit-scene provider."""

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

from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
    fit_scene_provider as provider_module,
)
from rl_agent.splitfusion_hybrid_sac_run4_v1 import held_payload, quality_adapter
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.corrected_p40_sidecar import (
    load_exact_corrected_p40_sidecar,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from rl_agent.splitfusion_hybrid_sac_v1.empirical_quality_surface import (
    EmpiricalQualitySurface,
    PolicySceneView,
    load_empirical_quality_surface,
)
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.contract import (
    Q_E4_GRID,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _synthetic_train_curves(
    inventory: provider_module.RegisteredTrainSceneInventoryV1,
) -> tuple[held_payload.ScenePayloadCurveV1, ...]:
    """Complete mechanics-only curves with the exact registered scene identities."""

    result = []
    for train_rank, record in enumerate(inventory.records):
        selection = record.selection
        for mode_id in range(12):
            base = 5_000_000 + train_rank * 100 + mode_id * 10_000
            nodes = tuple(
                held_payload.PayloadNodeV1(
                    q_e4=q_e4,
                    total_transmitted_bytes=base - 200 * q_e4,
                    source_row_sha256=_digest(
                        f"mechanics:{selection.sample_id}:{mode_id}:{q_e4}"
                    ),
                )
                for q_e4 in Q_E4_GRID
            )
            result.append(
                held_payload.ScenePayloadCurveV1(
                    sample_id=selection.sample_id,
                    episode_id=selection.episode_id,
                    frame_id=selection.frame_id,
                    selection_rank_within_train=train_rank,
                    source_split="train",
                    inclusion_probability=selection.inclusion_probability,
                    sampling_weight=selection.sampling_weight,
                    mode_id=mode_id,
                    scene_source_sha256=selection.canonical_sha256,
                    source_selection_sha256=inventory.selection_manifest_sha256,
                    source_database_sha256=inventory.database_sha256,
                    nodes=nodes,
                )
            )
    return tuple(result)


class Run4FitSceneProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.partition = load_registered_empirical_fit_partition(
            project_root=cls.root
        )
        cls.sidecar = load_exact_corrected_p40_sidecar(root=cls.root)
        cls.surface = load_empirical_quality_surface(
            cls.sidecar, project_root=cls.root
        )
        cls.surface_binding_sha = quality_adapter.surface_binding_sha256(
            cls.surface.binding
        )
        cls.train_ids = tuple(
            sorted(
                row.sample_id
                for row in cls.partition.scene_assignments
                if row.split == TRAIN_SPLIT
            )
        )
        cls.validation_ids = frozenset(
            row.sample_id
            for row in cls.partition.scene_assignments
            if row.split == FIT_VALIDATION_SPLIT
        )
        selections = tuple(
            quality_adapter.FitSceneSelectionV1.from_query(
                cls.surface.query_fit_q_e4(sample_id, 0, 0),
                surface_binding_sha256=cls.surface_binding_sha,
            )
            for sample_id in cls.train_ids
        )
        cls.inventory = provider_module.RegisteredTrainSceneInventoryV1.from_verified_selections(
            selections,
            registered_partition_sha256=(
                provider_module.REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256
            ),
            selection_manifest_sha256=cls.surface.binding.selection_file_sha256,
            database_sha256=cls.surface.binding.database_file_sha256,
            surface_binding_sha256=cls.surface_binding_sha,
        )

        curves = _synthetic_train_curves(cls.inventory)
        cls.held_provider = held_payload.HeldPayloadProviderV1(
            curves,
            expected_inventory_sha256=(
                held_payload.payload_curve_inventory_sha256(curves)
            ),
        )
        cls.adapter = quality_adapter.Run4QualityPayloadAdapterV1(
            surface=cls.surface,
            held_provider=cls.held_provider,
            expected_surface_binding_sha256=cls.surface_binding_sha,
            expected_held_provider_binding_sha256=cls.held_provider.binding_sha256,
        )
        cls.envelope = provider_module.VerifiedFitSceneEnvelopeV1(
            inventory=cls.inventory,
            quality_adapter_binding_sha256=cls.adapter.binding.canonical_sha256,
            held_provider_binding_sha256=cls.held_provider.binding_sha256,
            held_provider_inventory_sha256=cls.held_provider.inventory_sha256,
            held_scene_identity_inventory_sha256=cls.inventory.canonical_sha256,
            held_provider_scene_count=cls.held_provider.scene_count,
            verification_report_sha256=_digest("test-only-verification-report"),
        )
        cls.catalog = action_contract.load_contract()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.surface.close()

    @classmethod
    def make_provider(
        cls, *, scene_seed: int = 1701, held_seed: int = 2903
    ) -> provider_module.Run4FitSceneProviderV1:
        return provider_module.Run4FitSceneProviderV1(
            envelope=cls.envelope,
            adapter=cls.adapter,
            expected_envelope_sha256=cls.envelope.canonical_sha256,
            scene_rng_seed=scene_seed,
            held_rng_seed=held_seed,
            scene_rng_stream_id="run4-test-fit-scenes",
            held_rng_stream_id="run4-test-held-scenes",
        )

    @classmethod
    def action(
        cls, mode_id: int = 3, q_e4: int = 3000
    ) -> ExecutedActionIdentity:
        executable = cls.catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
        )
        return ExecutedActionIdentity.from_executable_action(
            executable, cls.catalog
        )

    @staticmethod
    def counter(
        ordinal: int, *, session: str = "fit-scene-provider-test"
    ) -> held_payload.HeldSelectionCounterV1:
        return held_payload.HeldSelectionCounterV1(
            session_id=session,
            decision_seq=9,
            held_ordinal=ordinal,
            rng_stream_id="run4-test-held-scenes",
        )

    def test_registered_inventory_is_exactly_391_train_scenes(self) -> None:
        self.assertEqual(
            provider_module.REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        )
        self.assertEqual(len(self.inventory.records), 391)
        self.assertEqual(
            self.inventory.canonical_sha256,
            provider_module.REGISTERED_TRAIN_SCENE_INVENTORY_SHA256,
        )
        inventory_ids = {
            item.selection.sample_id for item in self.inventory.records
        }
        self.assertEqual(inventory_ids, set(self.train_ids))
        self.assertTrue(inventory_ids.isdisjoint(self.validation_ids))

    def test_fit_validation_scene_cannot_be_represented_as_train(self) -> None:
        sample_id = sorted(self.validation_ids)[0]
        selection = quality_adapter.FitSceneSelectionV1.from_query(
            self.surface.query_fit_q_e4(sample_id, 0, 0),
            surface_binding_sha256=self.surface_binding_sha,
        )
        with self.assertRaises(provider_module.InventoryRejected):
            provider_module.TrainSceneInventoryRecordV1.from_selection(selection)
        with self.assertRaises(provider_module.InventoryRejected):
            provider_module.TrainSceneInventoryRecordV1(
                selection=selection,
                partition_split="train",
                temporal_block=selection.frame_id // 100,
            )

    def test_weighted_draw_is_deterministic_and_global_rng_neutral(self) -> None:
        global_before = random.getstate()
        seed = 1701
        probe = random.Random(seed)
        expected_draw = probe.random()
        threshold = expected_draw * self.inventory.total_sampling_weight
        cumulative = 0.0
        expected = self.inventory.records[-1]
        for record in self.inventory.records:
            cumulative += record.selection.sampling_weight
            if threshold < cumulative:
                expected = record
                break

        provider = self.make_provider(scene_seed=seed)
        observed = provider.select_scene()
        self.assertEqual(observed.rng_draw, expected_draw)
        self.assertEqual(observed.selection, expected.selection)
        self.assertEqual(random.getstate(), global_before)

    def test_checkpoint_restore_continues_both_local_rng_streams_exactly(self) -> None:
        first = self.make_provider(scene_seed=37, held_seed=41)
        first.select_scene()
        first.held_tensor(
            counter=self.counter(1, session="before-checkpoint"),
            action=self.action(),
            tensor_seq=1,
        )
        checkpoint = first.state_dict()

        restored = self.make_provider(scene_seed=37, held_seed=41)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.state_dict(), checkpoint)

        first_scenes = tuple(first.select_scene().selection.sample_id for _ in range(8))
        restored_scenes = tuple(
            restored.select_scene().selection.sample_id for _ in range(8)
        )
        self.assertEqual(first_scenes, restored_scenes)

        first_held = first.held_tensor(
            counter=self.counter(2, session="continuation-a"),
            action=self.action(mode_id=7, q_e4=4000),
            tensor_seq=2,
        )
        restored_held = restored.held_tensor(
            counter=self.counter(2, session="continuation-b"),
            action=self.action(mode_id=7, q_e4=4000),
            tensor_seq=2,
        )
        self.assertEqual(
            first_held.estimate.rng_draw, restored_held.estimate.rng_draw
        )
        self.assertEqual(
            first_held.estimate.selected_sample_id,
            restored_held.estimate.selected_sample_id,
        )
        self.assertEqual(
            first.state_dict().held_rng_state,
            restored.state_dict().held_rng_state,
        )

    def test_reward_and_held_payload_close_to_exact_train_inventory(self) -> None:
        provider = self.make_provider(scene_seed=101, held_seed=103)
        draw = provider.select_scene()
        action = self.action(mode_id=5, q_e4=2000)
        reward = provider.reward_tensor(draw=draw, action=action, tensor_seq=10)
        self.assertEqual(
            reward.evidence.fit_selection_sha256,
            draw.selection.canonical_sha256,
        )
        self.assertEqual(reward.policy_scene_features(), draw.policy_scene_features())
        self.assertTrue(reward.tensor.reward_requested)

        held = provider.held_tensor(
            counter=self.counter(1, session="exact-held-join"),
            action=action,
            tensor_seq=11,
        )
        selected = next(
            item.selection
            for item in self.inventory.records
            if item.selection.sample_id == held.estimate.selected_sample_id
        )
        self.assertEqual(
            held.estimate.selected_scene_source_sha256,
            selected.canonical_sha256,
        )
        self.assertFalse(held.tensor.reward_requested)

    def test_policy_projection_contains_only_si_and_p40(self) -> None:
        draw = self.make_provider().select_scene()
        scene = draw.policy_scene()
        self.assertIs(type(scene), PolicySceneView)
        self.assertEqual(
            tuple(item.name for item in fields(scene)),
            ("camera_si", "radar_p40"),
        )
        serialized = repr(scene.to_dict())
        for forbidden in (
            "sample_id",
            "episode_id",
            "frame_id",
            "sampling_weight",
            "partition",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_zero_si_and_p40_are_valid_but_missing_is_not_zero(self) -> None:
        genuine = self.inventory.records[0].selection
        zero = replace(genuine, camera_si=0.0, radar_p40=0.0)
        record = provider_module.TrainSceneInventoryRecordV1.from_selection(zero)
        draw = provider_module.FitSceneDrawV1(
            provider_binding_sha256=_digest("zero-provider"),
            draw_ordinal=1,
            rng_draw=0.0,
            selection=record.selection,
        )
        self.assertEqual(draw.policy_scene_features(), (0.0, 0.0))
        for field in ("camera_si", "radar_p40"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(genuine, **{field: None})

    def test_binding_and_inventory_drift_fail_closed(self) -> None:
        with self.assertRaises(provider_module.EnvelopeBindingError):
            provider_module.Run4FitSceneProviderV1(
                envelope=self.envelope,
                adapter=self.adapter,
                expected_envelope_sha256=_digest("wrong-envelope"),
                scene_rng_seed=1,
                held_rng_seed=2,
                scene_rng_stream_id="scene",
                held_rng_stream_id="held",
            )
        foreign = replace(
            self.envelope,
            quality_adapter_binding_sha256=_digest("foreign-adapter"),
        )
        with self.assertRaises(provider_module.EnvelopeBindingError):
            provider_module.Run4FitSceneProviderV1(
                envelope=foreign,
                adapter=self.adapter,
                expected_envelope_sha256=foreign.canonical_sha256,
                scene_rng_seed=1,
                held_rng_seed=2,
                scene_rng_stream_id="scene",
                held_rng_stream_id="held",
            )
        with self.assertRaises(provider_module.InventoryRejected):
            replace(
                self.inventory,
                selection_manifest_sha256=_digest("foreign-selection"),
            )

    def test_count_only_held_population_attestation_is_insufficient(self) -> None:
        original = self.held_provider._scenes
        forged_first = replace(
            original[0], sample_id="foreign-train-scene-with-same-count"
        )
        self.held_provider._scenes = (forged_first,) + original[1:]
        try:
            with self.assertRaises(provider_module.EnvelopeBindingError):
                self.make_provider()
        finally:
            self.held_provider._scenes = original

    def test_stale_or_forged_draw_is_refused(self) -> None:
        provider = self.make_provider()
        stale = provider.select_scene()
        current = provider.select_scene()
        with self.assertRaises(provider_module.SceneDrawError):
            provider.reward_tensor(
                draw=stale, action=self.action(), tensor_seq=20
            )
        forged = replace(current, rng_draw=current.rng_draw / 2.0)
        with self.assertRaises(provider_module.SceneDrawError):
            provider.reward_tensor(
                draw=forged, action=self.action(), tensor_seq=20
            )

    def test_invalid_checkpoint_is_atomic(self) -> None:
        provider = self.make_provider()
        provider.select_scene()
        before = provider.state_dict()
        foreign = replace(before, provider_binding_sha256=_digest("foreign-state"))
        with self.assertRaises(provider_module.ProviderStateError):
            provider.load_state_dict(foreign)
        self.assertEqual(provider.state_dict(), before)

    def test_z_import_performs_no_io_rng_socket_process_or_evidence_load(self) -> None:
        global_before = random.getstate()
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("filesystem access")
        ) as opened, mock.patch.object(
            socket, "socket", side_effect=AssertionError("socket opened")
        ) as socket_opened, mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("process launched")
        ) as popen, mock.patch.object(
            EmpiricalQualitySurface,
            "load",
            side_effect=AssertionError("evidence loaded"),
        ) as loaded:
            importlib.reload(provider_module)
        opened.assert_not_called()
        socket_opened.assert_not_called()
        popen.assert_not_called()
        loaded.assert_not_called()
        self.assertEqual(random.getstate(), global_before)


if __name__ == "__main__":
    unittest.main()
