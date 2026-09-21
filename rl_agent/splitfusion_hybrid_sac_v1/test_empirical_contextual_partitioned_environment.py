"""Focused CPU tests for partition-aware D1 sampling."""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_contract import require_supported_action
from .empirical_contextual_environment import EmpiricalEnvironmentError
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    TRAIN_SPLIT,
    load_registered_empirical_fit_partition,
)
from .empirical_contextual_partitioned_environment import (
    PartitionedEmpiricalEnvironmentStateV1,
    PartitionedEmpiricalOneStepEnvironmentV1,
    load_registered_partitioned_d1_environment,
)
from .empirical_radio_context import (
    OaiRadioCalibrationStoreV1,
    RadioCalibrationError,
)


class PartitionedEmpiricalEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partition = load_registered_empirical_fit_partition()
        cls.train = load_registered_partitioned_d1_environment(
            seed=20260921, split=TRAIN_SPLIT
        )
        # Reuse the already qualified immutable dependencies so the focused
        # test pays the exhaustive D1 preflight cost only once.
        cls.validation = PartitionedEmpiricalOneStepEnvironmentV1(
            sidecar=cls.train._sidecar,
            surface=cls.train._surface,
            network=cls.train._network,
            radio_store=cls.train._radio_store,
            contexts=cls.train._contexts,
            normalization=cls.train.normalization,
            freshness=cls.train.freshness,
            seed=20260922,
            preflight=cls.train.preflight,
            partition=cls.partition,
            split=FIT_VALIDATION_SPLIT,
        )
        cls.action = require_supported_action(0, 8812)

    @classmethod
    def tearDownClass(cls) -> None:
        # Both views deliberately share this focused test's read-only surface.
        cls.train.close()

    @staticmethod
    def _episode(env):
        observation = env.reset()
        result = env.step(PartitionedEmpiricalEnvironmentTest.action)
        return observation, result

    def test_exact_registered_counts_bindings_and_disjoint_populations(self) -> None:
        self.assertEqual(self.train.scene_population_count, 391)
        self.assertEqual(self.validation.scene_population_count, 85)
        self.assertEqual(
            dict(self.train.radio_profile_population_counts),
            {
                "FAVORABLE_STABLE": 80,
                "MID_VARIABLE": 80,
                "ADVERSE_STABLE": 79,
                "FADE_RECOVERY": 80,
            },
        )
        self.assertEqual(
            dict(self.validation.radio_profile_population_counts),
            {profile: 20 for profile in NETWORK_PROFILE_ORDER},
        )
        train_scenes = {item.sample_id for item in self.train._split_contexts}
        validation_scenes = {
            item.sample_id for item in self.validation._split_contexts
        }
        self.assertFalse(train_scenes & validation_scenes)
        self.assertEqual(
            train_scenes | validation_scenes,
            {item.sample_id for item in self.train._contexts.contexts},
        )
        self.assertEqual(
            self.train.fit_partition_sha256,
            REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
        )
        self.assertNotEqual(
            self.train.sampling_contract_sha256,
            self.validation.sampling_contract_sha256,
        )
        self.assertEqual(
            self.partition.source_bindings["d1_pilot_binding_sha256"],
            self.train.binding.canonical_sha256(),
        )
        self.assertGreater(self.train.scene_sampling_weight_total, 0.0)
        self.assertGreater(self.validation.scene_sampling_weight_total, 0.0)

    def test_scene_draw_is_exact_restricted_weighted_draw(self) -> None:
        checkpoint = self.train.state_dict()
        probe = random.Random()
        probe.setstate(checkpoint.d1_state.context_rng_state)
        threshold = probe.random() * self.train.scene_sampling_weight_total
        cumulative = 0.0
        expected = self.train._split_contexts[-1]
        for context in self.train._split_contexts:
            cumulative += context.sampling_weight
            if threshold < cumulative:
                expected = context
                break
        self.train.reset()
        self.assertEqual(self.train._active_context, expected)
        self.train.step(self.action)

    def test_profile_is_uniform_before_restricted_row_draw(self) -> None:
        for env in (self.train, self.validation):
            checkpoint = env.state_dict()
            profile_probe = random.Random()
            profile_probe.setstate(
                checkpoint.d1_state.radio_sampler_state.profile_rng_state
            )
            expected_profile = NETWORK_PROFILE_ORDER[
                profile_probe.randrange(len(NETWORK_PROFILE_ORDER))
            ]
            row_probe = random.Random()
            row_probe.setstate(
                checkpoint.d1_state.radio_sampler_state.row_rng_state
            )
            allowed_in_store_order = tuple(
                row
                for row in env._radio_store.rows
                if row.network_profile == expected_profile
                and next(
                    assignment
                    for assignment in self.partition.radio_assignments
                    if assignment.csv_row_number == row.csv_row_number
                ).split
                == env.sampling_split
            )
            expected_row = allowed_in_store_order[
                row_probe.randrange(len(allowed_in_store_order))
            ]
            _observation, result = self._episode(env)
            self.assertEqual(result.audit.hidden_network_profile, expected_profile)
            self.assertEqual(
                result.audit.hidden_radio_csv_row_number,
                expected_row.csv_row_number,
            )
            allowed = {
                row.csv_row_number
                for row in self.partition.radio_assignments
                if row.split == env.sampling_split
                and row.network_profile == expected_profile
            }
            self.assertIn(result.audit.hidden_radio_csv_row_number, allowed)

    def test_many_draws_never_cross_the_requested_split(self) -> None:
        scene_split = {
            row.sample_id: row.split for row in self.partition.scene_assignments
        }
        radio_split = {
            row.csv_row_number: row.split
            for row in self.partition.radio_assignments
        }
        for env in (self.train, self.validation):
            observed_profiles = set()
            for _ in range(128):
                _observation, result = self._episode(env)
                self.assertEqual(scene_split[result.audit.sample_id], env.sampling_split)
                self.assertEqual(
                    radio_split[result.audit.hidden_radio_csv_row_number],
                    env.sampling_split,
                )
                observed_profiles.add(result.audit.hidden_network_profile)
            self.assertEqual(observed_profiles, set(NETWORK_PROFILE_ORDER))

    def test_policy_observation_keeps_the_frozen_d1_binding_and_no_split(self) -> None:
        observation, _result = self._episode(self.train)
        self.assertEqual(
            observation.environment_binding_sha256,
            self.train.binding.canonical_sha256(),
        )
        self.assertFalse(hasattr(observation, "sampling_split"))
        feature_names = " ".join(observation.policy_feature_order).lower()
        self.assertNotIn("split", feature_names)
        self.assertNotIn("profile", feature_names)

    def test_checkpoint_round_trip_is_exact_and_split_bound(self) -> None:
        checkpoint = self.train.state_dict()
        expected = self._episode(self.train)
        self.train.load_state_dict(checkpoint)
        actual = self._episode(self.train)
        self.assertEqual(actual, expected)

        validation_before = self.validation.state_dict()
        with self.assertRaisesRegex(
            EmpiricalEnvironmentError, "sampling-split mismatch"
        ):
            self.validation.load_state_dict(checkpoint)
        self.assertEqual(self.validation.state_dict(), validation_before)

        forged = replace(checkpoint, sampling_contract_sha256="0" * 64)
        train_before = self.train.state_dict()
        with self.assertRaisesRegex(
            EmpiricalEnvironmentError, "sampling-contract mismatch"
        ):
            self.train.load_state_dict(forged)
        self.assertEqual(self.train.state_dict(), train_before)

        nested = checkpoint.d1_state
        malformed_radio = replace(
            nested.radio_sampler_state, row_rng_state=("malformed",)
        )
        malformed = replace(
            checkpoint,
            d1_state=replace(nested, radio_sampler_state=malformed_radio),
        )
        train_before = self.train.state_dict()
        with self.assertRaises(RadioCalibrationError):
            self.train.load_state_dict(malformed)
        self.assertEqual(self.train.state_dict(), train_before)

    def test_state_requires_exact_type_and_registered_split(self) -> None:
        with self.assertRaises(EmpiricalEnvironmentError):
            self.train.load_state_dict(self.train.state_dict().d1_state)
        with self.assertRaises(EmpiricalEnvironmentError):
            PartitionedEmpiricalEnvironmentStateV1(
                sampling_split="held",
                fit_partition_sha256="0" * 64,
                sampling_contract_sha256="1" * 64,
                d1_state=self.train.state_dict().d1_state,
            )
        with self.assertRaises(EmpiricalEnvironmentError):
            load_registered_partitioned_d1_environment(seed=1, split="held")

    def test_identity_join_and_partition_hash_gates_fail_closed(self) -> None:
        contexts = list(self.train._contexts.contexts)
        contexts[0] = replace(contexts[0], frame_id=contexts[0].frame_id + 1)
        with self.assertRaisesRegex(EmpiricalEnvironmentError, "identity mismatch"):
            self.train._join_scene_population(tuple(contexts))

        rows = list(self.train._radio_store.rows)
        rows[0] = replace(rows[0], trace_step_index=rows[0].trace_step_index + 1)
        bad_store = OaiRadioCalibrationStoreV1(
            rows=tuple(rows),
            rows_by_profile=self.train._radio_store.rows_by_profile,
            source_sha256=self.train._radio_store.source_sha256,
        )
        with self.assertRaisesRegex(EmpiricalEnvironmentError, "identity mismatch"):
            self.train._join_radio_population(bad_store)

        bad_bindings = dict(self.partition.source_bindings)
        bad_bindings["d1_pilot_binding_sha256"] = "0" * 64
        bad_partition = replace(self.partition, source_bindings=bad_bindings)
        with self.assertRaisesRegex(EmpiricalEnvironmentError, "hash drift"):
            PartitionedEmpiricalOneStepEnvironmentV1(
                sidecar=self.train._sidecar,
                surface=self.train._surface,
                network=self.train._network,
                radio_store=self.train._radio_store,
                contexts=self.train._contexts,
                normalization=self.train.normalization,
                freshness=self.train.freshness,
                seed=8,
                preflight=self.train.preflight,
                partition=bad_partition,
                split=TRAIN_SPLIT,
            )

    def test_local_sampling_never_advances_module_global_rng(self) -> None:
        before = random.getstate()
        self._episode(self.train)
        self._episode(self.validation)
        self.assertEqual(random.getstate(), before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
