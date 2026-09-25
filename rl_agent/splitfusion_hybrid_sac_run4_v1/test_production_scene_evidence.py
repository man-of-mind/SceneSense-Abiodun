"""Adversarial tests for the exact Run-4 production scene evidence."""

from __future__ import annotations

import builtins
import importlib
import random
import socket
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
    production_scene_evidence as subject,
)
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_hybrid_sac_v1.empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    load_registered_empirical_fit_partition,
)
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import (
    ExecutedActionIdentity,
)


class ProductionSceneEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        cls.evidence = subject.load_production_scene_evidence(
            repository_root=cls.root
        )
        cls.catalog = action_contract.load_contract()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.evidence.close()

    def test_registered_bundle_closes_exactly(self) -> None:
        evidence = self.evidence
        self.assertEqual(len(evidence.inventory.records), 391)
        self.assertEqual(evidence.held_provider.scene_count, 391)
        self.assertEqual(
            evidence.held_provider.inventory_sha256,
            subject.REGISTERED_HELD_CURVE_INVENTORY_SHA256,
        )
        self.assertEqual(
            evidence.held_provider.binding_sha256,
            subject.REGISTERED_HELD_PROVIDER_BINDING_SHA256,
        )
        self.assertEqual(
            evidence.adapter.binding.canonical_sha256,
            subject.REGISTERED_QUALITY_ADAPTER_BINDING_SHA256,
        )
        self.assertEqual(
            evidence.verification.canonical_sha256,
            subject.REGISTERED_VERIFICATION_REPORT_SHA256,
        )
        self.assertEqual(
            evidence.envelope.canonical_sha256,
            subject.REGISTERED_SCENE_ENVELOPE_SHA256,
        )
        self.assertEqual(evidence.verification.curve_count, 391 * 12)

    def test_inventory_is_train_only_and_disjoint_from_validation(self) -> None:
        partition = load_registered_empirical_fit_partition(
            project_root=self.root
        )
        validation = {
            row.sample_id
            for row in partition.scene_assignments
            if row.split == FIT_VALIDATION_SPLIT
        }
        train = {
            record.selection.sample_id
            for record in self.evidence.inventory.records
        }
        self.assertEqual(len(train), 391)
        self.assertTrue(train.isdisjoint(validation))

    def test_exact_reward_node_matches_pinned_surface(self) -> None:
        selection = self.evidence.inventory.records[0].selection
        mode_id = 7
        q_e4 = 7000
        executable = self.catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE)
        )
        action = ExecutedActionIdentity.from_executable_action(
            executable, self.catalog
        )
        direct = self.evidence.surface.query_fit_q_e4(
            selection.sample_id, mode_id, q_e4
        )
        produced = self.evidence.adapter.reward_tensor(
            selection=selection, action=action, tensor_seq=1
        )
        self.assertEqual(
            produced.tensor.offered_payload_bytes,
            direct.policy.payload.total_transmitted_bytes,
        )
        self.assertEqual(
            produced.q_perc, direct.policy.component("q_perc").value
        )
        self.assertEqual(produced.policy_scene.camera_si, selection.camera_si)
        self.assertEqual(produced.policy_scene.radar_p40, selection.radar_p40)

    def test_camera_scaling_is_fit_only_and_registered(self) -> None:
        scaling = self.evidence.verification.camera_si_scaling
        self.assertEqual(scaling.scene_count, 391)
        self.assertEqual(scaling.weighted_median, 116.39183807373047)
        self.assertEqual(scaling.weighted_q25, 107.39614868164062)
        self.assertEqual(scaling.weighted_q75, 124.9004898071289)
        self.assertEqual(scaling.robust_scale_iqr, 17.50434112548828)
        self.assertEqual(
            scaling.scene_inventory_sha256,
            self.evidence.inventory.canonical_sha256,
        )

    def test_digest_drift_closes_owned_resource(self) -> None:
        closed = mock.Mock()
        fake = SimpleNamespace(
            held_provider=SimpleNamespace(
                inventory_sha256=subject.REGISTERED_HELD_CURVE_INVENTORY_SHA256,
                binding_sha256=subject.REGISTERED_HELD_PROVIDER_BINDING_SHA256,
            ),
            adapter=SimpleNamespace(
                binding=SimpleNamespace(
                    canonical_sha256=(
                        subject.REGISTERED_QUALITY_ADAPTER_BINDING_SHA256
                    )
                )
            ),
            verification=SimpleNamespace(
                canonical_sha256=subject.REGISTERED_VERIFICATION_REPORT_SHA256
            ),
            envelope=SimpleNamespace(
                canonical_sha256=subject.REGISTERED_SCENE_ENVELOPE_SHA256
            ),
            close=closed,
        )
        with mock.patch.object(
            subject, "REGISTERED_HELD_CURVE_INVENTORY_SHA256", "0" * 64
        ), mock.patch.object(subject, "_assemble_unregistered", return_value=fake):
            with self.assertRaisesRegex(
                subject.ProductionSceneEvidenceError,
                "held curve inventory digest drifted",
            ):
                subject.load_production_scene_evidence(
                    repository_root=self.root
                )
        closed.assert_called_once_with()

    def test_import_has_no_evidence_io_rng_socket_or_process_side_effect(self) -> None:
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("filesystem access")
        ) as opened, mock.patch.object(
            random, "seed", side_effect=AssertionError("global RNG seeded")
        ) as seeded, mock.patch.object(
            random, "random", side_effect=AssertionError("global RNG sampled")
        ) as sampled, mock.patch.object(
            socket, "socket", side_effect=AssertionError("socket opened")
        ) as socket_opened, mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("process launched")
        ) as popen:
            importlib.reload(subject)
        opened.assert_not_called()
        seeded.assert_not_called()
        sampled.assert_not_called()
        socket_opened.assert_not_called()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
