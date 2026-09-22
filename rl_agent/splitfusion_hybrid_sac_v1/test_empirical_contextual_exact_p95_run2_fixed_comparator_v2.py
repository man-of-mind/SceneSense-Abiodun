from __future__ import annotations

import ast
import math
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from . import empirical_contextual_exact_p95_run2_fixed_comparator_v2 as subject
from .empirical_contextual_contract import (
    DIRECT_QUALITY_COMPONENT,
    fixed_stage_latency_ms,
)
from .empirical_contextual_environment import EmpiricalOneStepEnvironmentV1
from .empirical_contextual_exact_p95_deadline_penalty_v2 import (
    base_p95_expected_utility64_v2,
    shaped_p95_expected_utility64_v2,
)
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .payload_network_surrogate import UDP_PAYLOAD_CAPACITY_BYTES


class ExactP95Run2FixedComparatorV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partition = load_registered_empirical_fit_partition()
        cls.contexts = subject.build_train_contexts_v2(cls.partition)

    def test_registered_inventory_and_context_order(self) -> None:
        actions = subject.enumerate_supported_actions_v2()
        self.assertEqual(len(actions), 52_240)
        self.assertEqual(len(self.contexts), 1_564)
        self.assertEqual(self.contexts[0].context_index, 0)
        self.assertEqual(self.contexts[-1].context_index, 1_563)
        self.assertEqual(
            subject._context_order_sha256(self.contexts),
            "3c77a1386a3deed508f50105491a0148f54400ad4c0d041dd8419ba329857a23",
        )

    def test_train_and_validation_ids_are_disjoint(self) -> None:
        train_ids = {item.sample_id for item in self.contexts}
        validation_ids = {
            item.sample_id
            for item in self.partition.scene_assignments
            if item.split == FIT_VALIDATION_SPLIT
        }
        declared_train = {
            item.sample_id
            for item in self.partition.scene_assignments
            if item.split == TRAIN_SPLIT
        }
        self.assertEqual(train_ids, declared_train)
        self.assertEqual(len(train_ids), 391)
        self.assertEqual(len(validation_ids), 85)
        self.assertTrue(train_ids.isdisjoint(validation_ids))

    def test_tie_rule_is_mean_then_mode_then_q(self) -> None:
        actions = ((0, 4), (0, 5), (1, 0), (1, 1))
        means = np.asarray((0.4, 0.5, 0.5, 0.3), dtype=np.float64)
        self.assertEqual(subject.select_fixed_action_v2(actions, means), (1, 2))
        actions = ((0, 4), (0, 5), (1, 0))
        means = np.asarray((0.5, 0.5, 0.5), dtype=np.float64)
        self.assertEqual(subject.select_fixed_action_v2(actions, means), (0, 1))

    def test_binary64_operation_order_and_float32_emission(self) -> None:
        p = np.asarray((0.25, 0.75), dtype=np.float64)
        quality = np.asarray((0.6, 0.4), dtype=np.float64)
        latency = np.asarray((199.0, 201.0), dtype=np.float64)
        base, shaped, emitted = subject._reward_vectors(
            p_admit=p, quality=quality, latency_p95=latency
        )
        for index in range(2):
            expected_base = base_p95_expected_utility64_v2(
                p_admit=float(p[index]),
                q_perc=float(quality[index]),
                latency_p95_ms=float(latency[index]),
            )
            expected_shaped = shaped_p95_expected_utility64_v2(
                p_admit=float(p[index]),
                q_perc=float(quality[index]),
                latency_p95_ms=float(latency[index]),
                deadline_penalty=subject.RUN2_V2_DEADLINE_PENALTY,
            )
            self.assertEqual(float(base[index]), expected_base)
            self.assertEqual(float(shaped[index]), expected_shaped)
            expected_bits = struct.pack(">f", expected_shaped)
            self.assertEqual(struct.pack(">f", float(emitted[index])), expected_bits)

    def test_emission_is_explicit_cpu_under_meta_default(self) -> None:
        original = torch.get_default_device()
        try:
            torch.set_default_device("meta")
            vector = subject._emit_cpu_float32_vector(
                np.asarray((0.1, -0.2), dtype=np.float64)
            )
            scalar = subject._emit_cpu_float32_scalar(0.1)
            self.assertEqual(vector.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(vector)))
            self.assertEqual(struct.pack(">f", scalar), struct.pack(">f", 0.1))
        finally:
            torch.set_default_device(original)

    def test_surface_and_network_vector_match_scalar_public_path(self) -> None:
        environment = EmpiricalOneStepEnvironmentV1.load_registered(seed=0)
        try:
            context = self.contexts[0]
            mode_id = 11
            vector = subject._interpolate_same_frame_mode(
                environment._surface, context.sample_id, mode_id
            )
            network = subject._network_vector(
                environment._network,
                context.network_profile,
                vector["payload"],
                vector["datagrams"],
            )
            for index in (0, len(vector["q"]) // 2, len(vector["q"]) - 1):
                q_e4 = int(vector["q"][index])
                query = environment._surface.query_fit_q_e4(
                    context.sample_id, mode_id, q_e4
                )
                component = query.policy.component(DIRECT_QUALITY_COMPONENT)
                self.assertTrue(component.valid)
                self.assertEqual(float(vector["quality"][index]), component.value)
                payload = float(query.policy.payload.total_transmitted_bytes)
                self.assertEqual(float(vector["payload"][index]), payload)
                prediction = environment._prediction_session.predict(
                    network_profile=context.network_profile,
                    payload_bytes=payload,
                    datagram_count=math.ceil(payload / UDP_PAYLOAD_CAPACITY_BYTES),
                )
                latency = prediction.conditional_retained_survivor_latency_model()
                self.assertLessEqual(
                    abs(
                        float(network["p_admit"][index])
                        - prediction.p_edge_admission_given_sent
                    ),
                    subject.SCALAR_VECTOR_ABS_TOLERANCE,
                )
                self.assertLessEqual(
                    abs(
                        float(network["p95"][index])
                        - (fixed_stage_latency_ms() + latency.p95_ms)
                    ),
                    subject.SCALAR_VECTOR_ABS_TOLERANCE,
                )
        finally:
            environment.close()

    def test_frozen_source_hash_drift_fails_closed(self) -> None:
        with mock.patch.object(
            subject, "TRAIN_PENALTY_SUMMARY_SHA256", "0" * 64
        ):
            with self.assertRaisesRegex(
                subject.ExactP95Run2FixedComparatorV2Error, "hash drift"
            ):
                subject._require_source_bindings(subject._project_root())

    def test_source_has_no_validation_or_checkpoint_dependency(self) -> None:
        path = Path(subject.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        self.assertFalse(
            any("fit_validation_evaluator" in name for name in imports)
        )
        self.assertFalse(any("fit_validation_panel" in name for name in imports))
        self.assertNotIn("torch.load", source)
        self.assertNotIn("checkpoint_latest", source)
        self.assertNotIn("checkpoint_000", source)
        self.assertNotIn(".pt\"", source)
        self.assertEqual(subject._require_source_bindings(subject._project_root())["actor_checkpoint_read_count"], 0)

    def test_import_and_helpers_do_not_initialize_cuda(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
