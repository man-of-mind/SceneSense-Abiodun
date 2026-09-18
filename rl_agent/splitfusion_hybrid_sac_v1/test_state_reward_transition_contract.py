"""Tests for the Phase-4a causal-state, reward and replay-transition contract.

Every value is deterministic and injected: no clock is read, nothing sleeps, and
no CARLA, OAI, Docker, CUDA or network resource is touched.  Test 18 proves the
import-side-effect claim in a clean subprocess with the filesystem and sockets
blocked.

Twenty-two test methods.  Tests 1-18 are one per originally required proof;
19 closes the remaining branch coverage; 20-22 cover the mandatory Phase-4a
clarifications (anchor-only ACK, privileged non-deployable GT, undefined-class
masks, no F1 claim, UE-local monotonic reward latency, bound evidence hashes,
and the candidate status of SI/P40):

 1. valid minimal causal state and deterministic feature order
 2. identifiers never enter the policy vector
 3. NaN/Inf/range/type rejection, including bool-as-int
 4. missing or stale scene/radio telemetry fails closed
 5. normalization requires explicit train-fit provenance
 6. previous-action identity stays catalog reconciled
 7. lower localization error strictly increases normalized quality
 8. beta < 0.5 keeps localization the larger share
 9. absent components renormalize; insufficient support fails
10. latency is derived exactly from controller timestamps
11. timeout is censored; feedback-only loss is never punished
12. action-path and adjudicated service failure are registered negatives
13. infrastructure fault is excluded
14. identity mismatch and cross-session next state fail closed
15. d and gamma**d are derived from the frozen hold
16. canonical serialization and hashing are byte deterministic
17. the schema binds all four frozen dependencies
18. no filesystem, network, CUDA, CARLA or OAI side effect on import
19. remaining adjudication verdicts, derived accessors and record guards
20. the quality ACK is anchor-only and never snaps an off-anchor q
21. GT-absent classes are masked, not rewarded; no F1/precision claim
22. reward latency is UE-local monotonic; evidence hashes are bound in

Dependency hashes in test 17 and canonical hashes in test 16 are recomputed with
a locally written canonicalizer rather than by calling the module's own helper.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np

from . import action_contract as ac
from . import reward_ticket_controller as rtc
from . import scene_descriptors as sd
from . import state_reward_transition_contract as src
from . import transaction_identity as ti

SESSION = "3f263fce-cc44-476e-93b5-19d09d439471"
OTHER_SESSION = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"

MS = 1_000_000
T0 = 1_000_000_000
B = rtc.B_REWARD_DEADLINE_NS

HEX64 = "a" * 64
HEX64_B = "b" * 64


def _plain(value: Any) -> Any:
    """Locally written flattener for read-only mappings/tuples to JSON types."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _independent_canonical_bytes(payload: Any) -> bytes:
    """A locally written canonicalizer, independent of the module under test."""
    return json.dumps(
        _plain(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _independent_sha256(payload: Any) -> str:
    return hashlib.sha256(_independent_canonical_bytes(payload)).hexdigest()


class StateRewardTransitionContractTest(unittest.TestCase):
    """Phase-4a contract semantics and fail-closed rules."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ac.load_contract()

    # -- builders ---------------------------------------------------------- #

    def _action(
        self, mode_id: int = 5, q_e4: int = 5000
    ) -> ti.ExecutedActionIdentity:
        executable = self.contract.resolve(mode_id, q_e4 / ac.Q_E4_SCALE)
        return ti.ExecutedActionIdentity.from_executable_action(
            executable, self.contract
        )

    def _scene(self, camera_si: float = 42.5, radar_p40: float = 0.375):
        return sd.SceneDescriptorSample(camera_si=camera_si, radar_p40=radar_p40)

    def _norm(self, **overrides: Any) -> src.StateNormalizationSpecV1:
        kwargs: dict = dict(
            spec_id="phase4a_unit_test_norm",
            spec_version=1,
            train_split_id="unit-test-synthetic-split",
            fit_population_count=1234,
            fit_config_sha256=HEX64,
            camera_si_clip_min=0.0,
            camera_si_clip_max=100.0,
            achieved_snr_db_clip_min=-5.0,
            achieved_snr_db_clip_max=35.0,
            bsr_log1p_scale=math.log1p(1_000_000.0),
            mcs_table_id="oai_ul_table_1",
            mcs_table_max_index=27,
            provenance={"origin": "unit test; not a fitted production spec"},
        )
        kwargs.update(overrides)
        return src.StateNormalizationSpecV1(**kwargs)

    def _freshness(self, **overrides: Any) -> src.StateFreshnessPolicyV1:
        kwargs: dict = dict(
            policy_id="phase4a_unit_test_freshness",
            max_scene_age_ns=150 * MS,
            max_snr_age_ns=200 * MS,
            max_bsr_age_ns=200 * MS,
            max_mcs_age_ns=200 * MS,
            provenance={"origin": "unit test bounds; not a measured budget"},
        )
        kwargs.update(overrides)
        return src.StateFreshnessPolicyV1(**kwargs)

    def _reward_spec(self, **overrides: Any) -> src.RewardSpecV1:
        kwargs: dict = dict(
            spec_id="phase4a_unit_test_reward",
            spec_version=1,
            w_seg_person=2.0,
            w_seg_vehicle=1.0,
            w_loc_person=2.0,
            w_loc_vehicle=1.0,
            tau_person_m=1.5,
            tau_vehicle_m=3.0,
            segmentation_mix_beta=0.3,
            w_quality=1.0,
            w_latency=0.25,
            r_registered_failure=-1.0,
            gamma_per_tensor=0.99,
            min_valid_quality_components=2,
            provenance={"origin": "unit test; every value is a hypothesis"},
        )
        kwargs.update(overrides)
        return src.RewardSpecV1(**kwargs)

    def _previous(self, **overrides: Any) -> src.PreviousOutcomeV1:
        kwargs: dict = dict(
            action=self._action(mode_id=3, q_e4=9800),
            terminal_class=rtc.TerminalClass.REWARD_FINAL_EXACT,
            decision_seq=0,
            quality_valid=True,
            latency_valid=True,
            quality_normalized=0.75,
            latency_normalized=0.4,
        )
        kwargs.update(overrides)
        return src.PreviousOutcomeV1(**kwargs)

    def _state(self, **overrides: Any) -> src.CausalStateV1:
        kwargs: dict = dict(
            scene=self._scene(),
            scene_age_ns=20 * MS,
            achieved_snr_db=18.25,
            bsr_bytes=4096,
            mcs_table_id="oai_ul_table_1",
            mcs_index=13,
            snr_age_ns=30 * MS,
            bsr_age_ns=25 * MS,
            mcs_age_ns=35 * MS,
            session_uuid=SESSION,
            observed_ns=T0,
            tensor_seq=10,
            carla_frame_id=500,
            scene_source_id="route_b_cell_rgb_radar_v1",
            network_source_id="oai_ue_mac_stats_v1",
            previous=None,
        )
        kwargs.update(overrides)
        return src.CausalStateV1(**kwargs)

    def _ack(
        self, action: Optional[ti.ExecutedActionIdentity] = None
    ) -> src.QualityAckBindingV1:
        return src.QualityAckBindingV1.for_executed_action(
            action if action is not None else self._action(),
            raw_quality_ack_sha256="c" * 64,
            detailed_evidence_sha256="d" * 64,
            evaluator_mode="exact_carla_gt_v1",
        )

    def _evidence(
        self,
        action: Optional[ti.ExecutedActionIdentity] = None,
        *,
        with_ack: bool = True,
    ) -> src.QualityEvidenceV1:
        action = action if action is not None else self._action()
        return src.QualityEvidenceV1.for_action(
            action,
            gt_source_detail="carla_0_10_town10hd_opt_actor_origin_gt",
            ack_binding=self._ack(action) if with_ack else None,
        )

    def _quality(self, **overrides: Any) -> src.QualityComponentsV1:
        kwargs: dict = dict(
            evidence=self._evidence(),
            vehicle_gt_support=4,
            person_gt_support=2,
            vehicle_seg_valid=True,
            person_seg_valid=True,
            vehicle_loc_valid=True,
            person_loc_valid=True,
            vehicle_seg_iou=0.80,
            person_seg_iou=0.60,
            vehicle_localization_error_m=0.95,
            person_localization_error_m=0.50,
        )
        kwargs.update(overrides)
        return src.QualityComponentsV1(**kwargs)

    def _completed_ticket(
        self,
        *,
        terminal: rtc.TerminalClass = rtc.TerminalClass.REWARD_FINAL_EXACT,
        decision_seq: int = 1,
        first_tensor_seq: int = 10,
        first_frame_id: int = 500,
        extra_reuses: int = 0,
        action: Optional[ti.ExecutedActionIdentity] = None,
        opened_ns: int = T0,
        resolution_offset_ns: int = 50 * MS,
        session_uuid: str = SESSION,
    ) -> rtc.CompletedTicket:
        """Drive the real Phase-3b controller to produce a genuine ticket."""
        action = action if action is not None else self._action()
        controller = rtc.RewardTicketController(session_uuid)
        controller.open_decision(
            decision_seq=decision_seq,
            tensor_seq=first_tensor_seq,
            carla_frame_id=first_frame_id,
            action=action,
            now_ns=opened_ns,
        )
        # The k_min reuse must not postdate the terminal event: the controller
        # enforces a monotonic clock, so a very small feedback offset forces the
        # reuse to share that instant.
        terminal_offset = (
            B + 1
            if terminal is rtc.TerminalClass.FEEDBACK_TIMEOUT
            else resolution_offset_ns
        )
        tensor_seq = first_tensor_seq
        frame_id = first_frame_id
        for index in range(1 + extra_reuses):
            tensor_seq += 1
            frame_id += 1
            controller.reuse_held_action(
                tensor_seq=tensor_seq,
                carla_frame_id=frame_id,
                now_ns=opened_ns + min((index + 1) * 10 * MS, terminal_offset),
            )
        if terminal is rtc.TerminalClass.FEEDBACK_TIMEOUT:
            completed = controller.observe(opened_ns + B + 1).completed_ticket
        elif terminal is rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED:
            completed = controller.record_infrastructure_fault(
                decision_seq=decision_seq,
                now_ns=opened_ns + resolution_offset_ns,
                detail="unit-test evaluator fault",
            ).completed_ticket
        else:
            status = (
                rtc.FeedbackTerminalStatus.REWARD_FINAL
                if terminal is rtc.TerminalClass.REWARD_FINAL_EXACT
                else rtc.FeedbackTerminalStatus.ACTION_PATH_FAILURE
            )
            completed = controller.submit_feedback(
                rtc.RewardFeedbackMessage(
                    identity=ti.RewardFeedbackIdentity(
                        session_uuid=session_uuid,
                        decision_seq=decision_seq,
                        reward_tensor_seq=first_tensor_seq,
                        carla_frame_id=first_frame_id,
                        action=action,
                    ),
                    terminal_status=status,
                ),
                opened_ns + resolution_offset_ns,
            ).completed_ticket
        assert completed is not None
        assert completed.terminal_class is terminal, completed.terminal_class
        return completed

    def _adjudication(
        self, verdict: src.Adjudication
    ) -> src.AdjudicationRecordV1:
        return src.AdjudicationRecordV1(
            verdict=verdict,
            adjudicator_id="unit_test_reconciler",
            evidence_sha256=HEX64_B,
            adjudicated_ns=T0 + 10 * B,
            detail=f"unit-test verdict {verdict.value}",
        )

    # ------------------------------------------------------------------ 1 -- #

    def test_valid_minimal_state_and_deterministic_feature_order(self) -> None:
        state = self._state()
        norm = self._norm()
        fresh = self._freshness()

        # a minimal episode-start state has an explicitly absent previous outcome
        self.assertFalse(state.has_previous_decision)
        self.assertIsNone(state.previous)
        self.assertEqual(state.camera_si, 42.5)
        self.assertEqual(state.radar_p40, 0.375)

        vector = src.build_policy_features(state, norm, fresh)
        self.assertEqual(len(vector.as_tuple()), src.POLICY_FEATURE_COUNT)
        self.assertEqual(vector.feature_names, src.POLICY_FEATURE_ORDER)
        self.assertEqual(
            vector.state_normalization_spec_sha256, norm.canonical_sha256()
        )
        named = vector.as_mapping()

        # every scalar transformation is exactly the configured one
        self.assertAlmostEqual(named["scene_camera_si_scaled"], 42.5 / 100.0, 12)
        self.assertEqual(named["scene_radar_p40"], 0.375)
        self.assertAlmostEqual(
            named["radio_achieved_snr_db_scaled"], (18.25 + 5.0) / 40.0, 12
        )
        self.assertAlmostEqual(
            named["radio_bsr_log1p_scaled"],
            math.log1p(4096.0) / math.log1p(1_000_000.0),
            12,
        )
        self.assertAlmostEqual(named["radio_mcs_index_scaled"], 13.0 / 27.0, 12)

        # absence is carried by masks, not by an indistinguishable zero
        self.assertEqual(named["prev_present_mask"], 0.0)
        self.assertEqual(named["prev_quality_valid_mask"], 0.0)
        self.assertEqual(named["prev_latency_valid_mask"], 0.0)
        self.assertEqual(
            [named[f"prev_joint_mode_onehot_{i:02d}"] for i in range(12)],
            [0.0] * 12,
        )

        # order is deterministic and repeated builds are identical
        self.assertEqual(
            src.build_policy_features(state, norm, fresh).as_tuple(),
            vector.as_tuple(),
        )
        # as_mapping is a defensive copy: mutating it cannot corrupt the record
        named["scene_radar_p40"] = 99.0
        self.assertEqual(vector.as_mapping()["scene_radar_p40"], 0.375)

        # a present previous decision fills exactly one one-hot slot
        with_prev = self._state(previous=self._previous())
        prev_named = src.build_policy_features(
            with_prev, norm, fresh
        ).as_mapping()
        self.assertEqual(prev_named["prev_present_mask"], 1.0)
        self.assertEqual(prev_named["prev_joint_mode_onehot_03"], 1.0)
        self.assertEqual(
            sum(
                prev_named[f"prev_joint_mode_onehot_{i:02d}"] for i in range(12)
            ),
            1.0,
        )
        # q enters as the wire value over its maximum, never as an anchor id
        self.assertEqual(prev_named["prev_q_normalized"], 9800.0 / 9800.0)
        self.assertEqual(prev_named["prev_quality_normalized"], 0.75)
        self.assertEqual(prev_named["prev_latency_normalized"], 0.4)
        self.assertEqual(prev_named["prev_quality_valid_mask"], 1.0)

        # a mid-range q is not snapped to an anchor
        mid = self._state(
            previous=self._previous(action=self._action(mode_id=0, q_e4=4321))
        )
        self.assertAlmostEqual(
            src.build_policy_features(mid, norm, fresh).as_mapping()[
                "prev_q_normalized"
            ],
            4321.0 / 9800.0,
            12,
        )

    # ------------------------------------------------------------------ 2 -- #

    def test_identifiers_never_enter_the_policy_vector(self) -> None:
        # the executable guard runs at import and stays true
        src.assert_policy_features_exclude_forbidden_fields()

        for metadata in src.CausalStateV1.metadata_field_names():
            with self.subTest(field=metadata):
                self.assertNotIn(metadata, src.POLICY_FEATURE_ORDER)
        for age in src.CausalStateV1.measurement_age_field_names():
            with self.subTest(field=age):
                self.assertNotIn(age, src.POLICY_FEATURE_ORDER)
        for forbidden in src.FORBIDDEN_POLICY_FEATURE_SUBSTRINGS:
            with self.subTest(substring=forbidden):
                for name in src.POLICY_FEATURE_ORDER:
                    self.assertNotIn(forbidden, name)

        # changing only identifiers/ages must not move a single feature
        norm, fresh = self._norm(), self._freshness()
        base = self._state()
        relabelled = self._state(
            session_uuid=OTHER_SESSION,
            observed_ns=T0 + 7 * B,
            tensor_seq=987,
            carla_frame_id=65535,
            scene_source_id="a-different-source",
            network_source_id="a-different-radio-source",
            scene_age_ns=1,
            snr_age_ns=2,
            bsr_age_ns=3,
            mcs_age_ns=4,
        )
        self.assertEqual(
            src.build_policy_features(base, norm, fresh).as_tuple(),
            src.build_policy_features(relabelled, norm, fresh).as_tuple(),
        )
        # ... while a real observation change does move one
        self.assertNotEqual(
            src.build_policy_features(base, norm, fresh).as_tuple(),
            src.build_policy_features(
                self._state(bsr_bytes=4097), norm, fresh
            ).as_tuple(),
        )

        # the previous action's anchor id is never a feature even when present
        anchored = self._previous(action=self._action(mode_id=3, q_e4=9800))
        self.assertTrue(anchored.action.is_registered_anchor)
        self.assertIsNotNone(anchored.action.action_id)
        named = src.build_policy_features(
            self._state(previous=anchored), norm, fresh
        ).as_mapping()
        self.assertNotIn("prev_action_id", named)
        self.assertNotIn(float(anchored.action.action_id), set(named.values()))

    # ------------------------------------------------------------------ 3 -- #

    def test_nan_inf_range_and_type_rejection(self) -> None:
        nan, inf = float("nan"), float("inf")

        # scene values are guarded by the Phase-3 record itself
        for bad in (nan, inf, -1.0, True, "0.5", None):
            with self.subTest(camera_si=bad):
                with self.assertRaises(sd.SceneDescriptorError):
                    sd.SceneDescriptorSample(camera_si=bad, radar_p40=0.5)
        for bad in (nan, inf, -0.01, 1.01, True):
            with self.subTest(radar_p40=bad):
                with self.assertRaises(sd.SceneDescriptorError):
                    sd.SceneDescriptorSample(camera_si=1.0, radar_p40=bad)

        # state scalars and exact integers
        for field, bad in (
            ("achieved_snr_db", nan),
            ("achieved_snr_db", inf),
            ("achieved_snr_db", True),
            ("achieved_snr_db", "18"),
            ("bsr_bytes", -1),
            ("bsr_bytes", 4096.0),
            ("bsr_bytes", True),
            ("mcs_index", -1),
            ("mcs_index", 13.0),
            ("mcs_index", True),
            ("mcs_table_id", ""),
            ("mcs_table_id", 1),
            ("scene_age_ns", -1),
            ("scene_age_ns", True),
            ("snr_age_ns", 1.5),
            ("bsr_age_ns", True),
            ("mcs_age_ns", -5),
            ("observed_ns", -1),
            ("tensor_seq", True),
            ("carla_frame_id", -1),
            ("session_uuid", SESSION.upper()),
            ("session_uuid", "not-a-uuid"),
            ("scene_source_id", ""),
            ("network_source_id", ""),
            ("scene", "not-a-sample"),
            ("previous", "not-a-previous-outcome"),
        ):
            with self.subTest(field=field, value=bad):
                with self.assertRaises(src.CausalStateError):
                    self._state(**{field: bad})

        # bool is rejected everywhere an exact int is required
        with self.assertRaises(src.CausalStateError):
            self._state(bsr_bytes=True)
        self.assertIsInstance(True, int)  # ... precisely because bool is an int

        # quality components
        for field, bad in (
            ("vehicle_seg_iou", nan),
            ("vehicle_seg_iou", 1.5),
            ("vehicle_seg_iou", -0.1),
            ("vehicle_seg_iou", True),
            ("person_localization_error_m", nan),
            ("person_localization_error_m", inf),
            ("person_localization_error_m", -0.5),
            ("evidence", "not-evidence"),
            ("vehicle_gt_support", -1),
            ("vehicle_gt_support", True),
            ("vehicle_gt_support", 1.0),
        ):
            with self.subTest(field=field, value=bad):
                with self.assertRaises(src.QualityContractError):
                    self._quality(**{field: bad})

        # reward-spec domains
        for field, bad in (
            ("w_seg_person", 0.0),
            ("w_seg_person", -1.0),
            ("w_loc_vehicle", nan),
            ("tau_person_m", 0.0),
            ("segmentation_mix_beta", 0.5),
            ("segmentation_mix_beta", 0.6),
            ("segmentation_mix_beta", -0.1),
            ("w_quality", 0.0),
            ("w_latency", -0.1),
            ("r_registered_failure", 0.5),
            ("gamma_per_tensor", 0.0),
            ("gamma_per_tensor", 1.01),
            ("min_valid_quality_components", 0),
            ("min_valid_quality_components", 5),
            ("min_valid_quality_components", True),
            ("spec_id", ""),
            ("provenance", {}),
        ):
            with self.subTest(field=field, value=bad):
                with self.assertRaises(src.RewardSpecError):
                    self._reward_spec(**{field: bad})

    # ------------------------------------------------------------------ 4 -- #

    def test_missing_or_stale_telemetry_fails_closed(self) -> None:
        # missing scene input is a Phase-3 fail-closed condition, never P40=0
        with self.assertRaises(sd.RadarUnavailableError):
            sd.radar_proximity_p40(None)
        with self.assertRaises(sd.RadarUnavailableError):
            sd.radar_proximity_p40(np.array([], dtype=np.float64))
        with self.assertRaises(sd.InvalidRadarRangesError):
            sd.radar_proximity_p40(np.array([10.0, float("nan")]))
        # the guard is an exception, not a substituted number
        try:
            sd.radar_proximity_p40(None)
        except sd.RadarUnavailableError as exc:
            self.assertNotIn("0.0", str(exc))

        norm, fresh = self._norm(), self._freshness()
        # a fresh state vectorizes
        src.build_policy_features(self._state(), norm, fresh)

        # each source going stale fails closed on its own
        for field, bound_name in (
            ("scene_age_ns", "max_scene_age_ns"),
            ("snr_age_ns", "max_snr_age_ns"),
            ("bsr_age_ns", "max_bsr_age_ns"),
            ("mcs_age_ns", "max_mcs_age_ns"),
        ):
            with self.subTest(source=field):
                bound = getattr(fresh, bound_name)
                at_bound = self._state(**{field: bound})
                # the bound itself is admissible
                src.build_policy_features(at_bound, norm, fresh)
                stale = self._state(**{field: bound + 1})
                with self.assertRaises(src.StaleTelemetryError) as caught:
                    fresh.assert_fresh(stale)
                self.assertIn("registered fallback", str(caught.exception))
                # and there is no path from a stale state to a feature vector
                with self.assertRaises(src.StaleTelemetryError):
                    src.build_policy_features(stale, norm, fresh)

        # an MCS index from another table is a silent-unit-error risk, so it
        # fails closed rather than being scaled by the wrong maximum
        with self.assertRaises(src.NormalizationSpecError):
            src.build_policy_features(
                self._state(mcs_table_id="oai_ul_table_2"), norm, fresh
            )
        with self.assertRaises(src.NormalizationSpecError):
            src.build_policy_features(self._state(mcs_index=28), norm, fresh)

        # freshness bounds must themselves be explicit and positive
        for field in (
            "max_scene_age_ns",
            "max_snr_age_ns",
            "max_bsr_age_ns",
            "max_mcs_age_ns",
        ):
            with self.subTest(bound=field):
                with self.assertRaises(src.NormalizationSpecError):
                    self._freshness(**{field: 0})

    # ------------------------------------------------------------------ 5 -- #

    def test_normalization_requires_explicit_train_fit_provenance(self) -> None:
        # there is no zero-argument or defaulted normalization spec
        with self.assertRaises(TypeError):
            src.StateNormalizationSpecV1()  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            src.RewardSpecV1()  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            src.StateFreshnessPolicyV1()  # type: ignore[call-arg]

        # the module exports no fitted constant to fall back on
        for name in dir(src):
            if name.startswith("DEFAULT_") or name.endswith("_DEFAULT"):
                self.fail(f"{name} looks like an invented fitted default")

        for field, bad in (
            ("train_split_id", ""),
            ("fit_population_count", 0),
            ("fit_population_count", -1),
            ("fit_population_count", True),
            ("fit_config_sha256", "deadbeef"),
            ("fit_config_sha256", HEX64.upper()),
            ("fit_config_sha256", 1),
            ("spec_id", ""),
            ("spec_version", 0),
            ("provenance", {}),
            ("provenance", {"origin": ""}),
            ("provenance", "a string"),
            ("camera_si_clip_min", -1.0),
            ("camera_si_clip_max", 0.0),
            ("achieved_snr_db_clip_max", -10.0),
            ("bsr_log1p_scale", 0.0),
            ("bsr_log1p_scale", -1.0),
            ("mcs_table_id", ""),
            ("mcs_table_max_index", 0),
        ):
            with self.subTest(field=field, value=bad):
                with self.assertRaises(src.NormalizationSpecError):
                    self._norm(**{field: bad})

        norm = self._norm()
        # provenance is frozen and the hash is reproducible independently
        self.assertIsInstance(norm.provenance, MappingProxyType)
        with self.assertRaises(TypeError):
            norm.provenance["origin"] = "tampered"  # type: ignore[index]
        self.assertEqual(
            norm.canonical_sha256(), _independent_sha256(norm.to_canonical_dict())
        )
        # the vector carries the spec hash, so scaling is always traceable
        vector = src.build_policy_features(
            self._state(), norm, self._freshness()
        )
        self.assertEqual(
            vector.state_normalization_spec_sha256, norm.canonical_sha256()
        )
        self.assertNotEqual(
            self._norm(camera_si_clip_max=120.0).canonical_sha256(),
            norm.canonical_sha256(),
        )
        # a non-spec object cannot be smuggled in
        with self.assertRaises(src.NormalizationSpecError):
            src.build_policy_features(
                self._state(), {"camera_si_clip_max": 100.0}, self._freshness()
            )

    # ------------------------------------------------------------------ 6 -- #

    def test_previous_action_identity_stays_catalog_reconciled(self) -> None:
        keep, drop = ac.keep_drop_counts(5000)
        fabricated = ti.ExecutedActionIdentity(
            execution_mode=ac.EXECUTION_MODE,
            mode_id=2,
            family=self.contract.mode(2).family,
            quantizer=self.contract.mode(2).quantizer,
            q_e4=5000,
            keep_count=keep,
            drop_count=drop,
        )
        self.assertFalse(fabricated.is_catalog_reconciled)
        with self.assertRaises(ti.UnreconciledActionIdentityError):
            self._previous(action=fabricated)
        with self.assertRaises(src.CausalStateError):
            self._previous(action="SPLIT/AE32/UINT4")

        # a reconciled identity keeps its exact catalog binding
        previous = self._previous()
        self.assertTrue(previous.action.is_catalog_reconciled)
        self.assertEqual(previous.action.catalog_sha256, ac.CATALOG_SHA256)
        self.assertEqual(previous.action.execution_mode, "SPLIT")
        self.assertLess(previous.action.mode_id, ac.EXPECTED_MODE_COUNT)

        # every one of the 12 joint modes is representable and separable
        seen = set()
        for mode_id in range(ac.EXPECTED_MODE_COUNT):
            named = src.build_policy_features(
                self._state(
                    previous=self._previous(
                        action=self._action(mode_id=mode_id, q_e4=0)
                    )
                ),
                self._norm(),
                self._freshness(),
            ).as_mapping()
            hot = [
                index
                for index in range(12)
                if named[f"prev_joint_mode_onehot_{index:02d}"] == 1.0
            ]
            self.assertEqual(hot, [mode_id])
            seen.add(mode_id)
        self.assertEqual(len(seen), 12)

        # non-REWARD_FINAL_EXACT terminals cannot claim an exact Q or L
        for terminal in (
            rtc.TerminalClass.ACTION_PATH_FAILURE,
            rtc.TerminalClass.FEEDBACK_TIMEOUT,
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
        ):
            with self.subTest(terminal=terminal):
                with self.assertRaises(src.CausalStateError):
                    self._previous(terminal_class=terminal)
                # ... but they are representable with both flags false
                ok = self._previous(
                    terminal_class=terminal,
                    quality_valid=False,
                    latency_valid=False,
                    quality_normalized=None,
                    latency_normalized=None,
                )
                self.assertIs(ok.terminal_class, terminal)

        # validity flags and values must agree; no sentinels
        with self.assertRaises(src.CausalStateError):
            self._previous(quality_valid=True, quality_normalized=None)
        with self.assertRaises(src.CausalStateError):
            self._previous(quality_valid=False, quality_normalized=0.0)
        with self.assertRaises(src.CausalStateError):
            self._previous(latency_valid=False, latency_normalized=0.0)
        with self.assertRaises(src.CausalStateError):
            self._previous(quality_normalized=1.5)

    # ------------------------------------------------------------------ 7 -- #

    def test_lower_localization_error_strictly_increases_quality(self) -> None:
        spec = self._reward_spec()
        errors = [0.0, 0.25, 0.5, 0.95, 2.0, 5.0, 25.0]
        values = [spec.normalize_localization(e, spec.tau_person_m) for e in errors]
        # strictly decreasing in error, i.e. strictly increasing in accuracy
        for lower, higher in zip(values, values[1:]):
            self.assertGreater(lower, higher)
        self.assertEqual(values[0], 1.0)
        for value in values:
            self.assertTrue(0.0 <= value <= 1.0)
        # the registered closed form, recomputed locally
        self.assertAlmostEqual(
            spec.normalize_localization(0.95, 3.0), math.exp(-0.95 / 3.0), 12
        )
        # tau is per class and the two classes are scaled independently
        self.assertNotEqual(spec.tau_person_m, spec.tau_vehicle_m)
        self.assertGreater(
            spec.normalize_localization(1.0, spec.tau_vehicle_m),
            spec.normalize_localization(1.0, spec.tau_person_m),
        )

        # end to end: a strictly better localization gives a strictly higher Q
        previous_quality = -1.0
        for error in reversed(errors):
            evaluation = spec.evaluate_quality(
                self._quality(
                    vehicle_localization_error_m=error,
                    person_localization_error_m=error,
                )
            )
            self.assertGreater(evaluation.quality, previous_quality)
            previous_quality = evaluation.quality

    # ------------------------------------------------------------------ 8 -- #

    def test_beta_below_half_keeps_localization_the_larger_share(self) -> None:
        for beta in (0.0, 0.1, 0.3, 0.49):
            with self.subTest(beta=beta):
                spec = self._reward_spec(segmentation_mix_beta=beta)
                self.assertAlmostEqual(spec.localization_mix, 1.0 - beta, 12)
                self.assertGreater(spec.localization_mix, beta)

        spec = self._reward_spec(segmentation_mix_beta=0.3)
        # perfect localization with poor segmentation beats the converse
        loc_strong = spec.evaluate_quality(
            self._quality(
                vehicle_seg_iou=0.0,
                person_seg_iou=0.0,
                vehicle_localization_error_m=0.0,
                person_localization_error_m=0.0,
            )
        )
        seg_strong = spec.evaluate_quality(
            self._quality(
                vehicle_seg_iou=1.0,
                person_seg_iou=1.0,
                vehicle_localization_error_m=1e9,
                person_localization_error_m=1e9,
            )
        )
        self.assertAlmostEqual(loc_strong.quality, 0.7, 9)
        self.assertAlmostEqual(seg_strong.quality, 0.3, 9)
        self.assertGreater(loc_strong.quality, seg_strong.quality)
        # the top-level mix is exactly beta/1-beta, never renormalized
        mixed = spec.evaluate_quality(self._quality())
        self.assertAlmostEqual(
            mixed.quality, 0.3 * mixed.q_seg + 0.7 * mixed.q_loc, 12
        )
        # beta >= 0.5 is unrepresentable
        for beta in (0.5, 0.51, 0.9, 1.0):
            with self.subTest(beta=beta):
                with self.assertRaises(src.RewardSpecError):
                    self._reward_spec(segmentation_mix_beta=beta)

    # ------------------------------------------------------------------ 9 -- #

    def test_absent_components_renormalize_and_low_support_fails(self) -> None:
        spec = self._reward_spec(min_valid_quality_components=2)

        # both classes valid: weights 2 (person) and 1 (vehicle)
        both = spec.evaluate_quality(
            self._quality(person_seg_iou=0.6, vehicle_seg_iou=0.9)
        )
        self.assertAlmostEqual(
            both.q_seg, (2.0 * 0.6 + 1.0 * 0.9) / 3.0, 12
        )
        self.assertEqual(
            dict(both.segmentation_weights_used), {"person": 2.0, "vehicle": 1.0}
        )

        # vehicle segmentation absent: Q_seg is exactly the person IoU, i.e.
        # renormalized over the valid weight only -- not 0.6*2/3, and neither
        # scored zero nor scored perfect for the missing class
        person_only = spec.evaluate_quality(
            self._quality(
                vehicle_seg_valid=False,
                vehicle_seg_iou=None,
                person_seg_iou=0.6,
            )
        )
        self.assertAlmostEqual(person_only.q_seg, 0.6, 12)
        self.assertEqual(
            dict(person_only.segmentation_weights_used), {"person": 2.0}
        )
        self.assertNotAlmostEqual(person_only.q_seg, (2.0 * 0.6) / 3.0, 6)
        self.assertNotIn("vehicle", person_only.segmentation_weights_used)

        # the absent class is recorded as absent, not as a number
        self.assertIsNone(person_only.components.vehicle_seg_iou)
        self.assertFalse(person_only.components.vehicle_seg_valid)

        # same rule for localization
        veh_loc_only = spec.evaluate_quality(
            self._quality(
                person_loc_valid=False,
                person_localization_error_m=None,
                vehicle_localization_error_m=2.0,
            )
        )
        self.assertAlmostEqual(
            veh_loc_only.q_loc, math.exp(-2.0 / spec.tau_vehicle_m), 12
        )
        self.assertIsNone(veh_loc_only.loc_person_normalized)
        self.assertEqual(
            dict(veh_loc_only.localization_weights_used), {"vehicle": 1.0}
        )

        # a whole family absent is NOT silently folded away: the top-level mix
        # is not renormalized, so it fails
        with self.assertRaises(src.InsufficientQualitySupportError):
            spec.evaluate_quality(
                self._quality(
                    vehicle_seg_valid=False,
                    person_seg_valid=False,
                    vehicle_seg_iou=None,
                    person_seg_iou=None,
                )
            )
        with self.assertRaises(src.InsufficientQualitySupportError):
            spec.evaluate_quality(
                self._quality(
                    vehicle_loc_valid=False,
                    person_loc_valid=False,
                    vehicle_localization_error_m=None,
                    person_localization_error_m=None,
                )
            )

        # the configured minimum support is enforced
        minimal = self._quality(
            vehicle_seg_valid=False,
            person_loc_valid=False,
            vehicle_seg_iou=None,
            person_localization_error_m=None,
        )
        self.assertEqual(minimal.valid_component_count, 2)
        spec.evaluate_quality(minimal)  # exactly at the configured floor
        with self.assertRaises(src.InsufficientQualitySupportError):
            self._reward_spec(min_valid_quality_components=3).evaluate_quality(
                minimal
            )
        self._reward_spec(min_valid_quality_components=4).evaluate_quality(
            self._quality()
        )
        with self.assertRaises(src.InsufficientQualitySupportError):
            self._reward_spec(min_valid_quality_components=4).evaluate_quality(
                minimal
            )

    # ----------------------------------------------------------------- 10 -- #

    def test_latency_is_derived_exactly_from_controller_timestamps(self) -> None:
        for offset in (0, 1, 37 * MS, 199 * MS, B):
            with self.subTest(offset_ns=offset):
                ticket = self._completed_ticket(resolution_offset_ns=offset)
                latency = src.LatencyMeasurementV1.from_completed_ticket(ticket)
                self.assertEqual(latency.opened_ns, ticket.opened_ns)
                self.assertEqual(latency.resolution_ns, ticket.resolution_ns)
                self.assertEqual(
                    latency.l_ns, ticket.resolution_ns - ticket.opened_ns
                )
                self.assertEqual(latency.l_ns, offset)
                self.assertEqual(
                    latency.normalized_latency, float(offset) / float(B)
                )
                # it agrees with the controller's own raw measurement
                self.assertEqual(latency.l_ns, ticket.feedback_latency_ns)
                # the controller refuses post-deadline feedback, so L <= B
                self.assertLessEqual(latency.normalized_latency, 1.0)

        # there is no way to hand in a latency: it is always derived
        ticket = self._completed_ticket(resolution_offset_ns=50 * MS)
        reward = src.evaluate_completed_decision(
            ticket, self._reward_spec(), quality_components=self._quality()
        )
        assert reward.latency is not None
        self.assertEqual(reward.latency.l_ns, 50 * MS)
        self.assertEqual(reward.latency.normalized_latency, 0.25)

        # r = w_quality * Q - w_latency * normalized_latency, recomputed here
        spec = self._reward_spec()
        assert reward.quality is not None
        self.assertAlmostEqual(
            reward.scalar_reward,
            spec.w_quality * reward.quality.quality - spec.w_latency * 0.25,
            12,
        )

        # a censored or excluded ticket has no latency to derive
        for terminal in (
            rtc.TerminalClass.FEEDBACK_TIMEOUT,
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
        ):
            with self.subTest(terminal=terminal):
                with self.assertRaises(src.StateRewardContractError):
                    src.LatencyMeasurementV1.from_completed_ticket(
                        self._completed_ticket(terminal=terminal)
                    )

        # a hand-built latency record must still satisfy the exact algebra
        with self.assertRaises(src.StateRewardContractError):
            src.LatencyMeasurementV1(
                opened_ns=T0,
                resolution_ns=T0 + 10,
                l_ns=11,
                normalized_latency=10.0 / B,
                deadline_ns=T0 + B,
            )
        with self.assertRaises(src.StateRewardContractError):
            src.LatencyMeasurementV1(
                opened_ns=T0,
                resolution_ns=T0 + 10,
                l_ns=10,
                normalized_latency=0.5,
                deadline_ns=T0 + B,
            )

    # ----------------------------------------------------------------- 11 -- #

    def test_timeout_is_censored_and_feedback_only_loss_is_not_punished(
        self,
    ) -> None:
        spec = self._reward_spec()
        ticket = self._completed_ticket(
            terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT
        )
        self.assertIs(ticket.terminal_class, rtc.TerminalClass.FEEDBACK_TIMEOUT)
        self.assertIsNone(ticket.resolution_ns)

        # unadjudicated: censored, no reward at all, in either direction
        censored = src.evaluate_completed_decision(ticket, spec)
        self.assertIs(
            censored.eligibility,
            src.LearningEligibility.CENSORED_PENDING_ADJUDICATION,
        )
        self.assertFalse(censored.learning_eligible)
        self.assertIsNone(censored.scalar_reward)
        self.assertIsNone(censored.quality)
        self.assertIsNone(censored.latency)
        self.assertEqual(censored.costs.c_deadline, 1.0)
        # the excess magnitude is unmeasurable without a receipt: null, not 0
        self.assertIsNone(censored.costs.c_latency_excess)
        self.assertIsNone(censored.costs.c_authoritative_failure)
        # this matches the controller's own disposition label
        self.assertEqual(
            rtc.TERMINAL_LEARNING_DISPOSITION[ticket.terminal_class],
            "censored_pending_post_run_reconciliation",
        )

        # an explicit PENDING verdict changes nothing
        pending = src.evaluate_completed_decision(
            ticket, spec, adjudication=self._adjudication(src.Adjudication.PENDING)
        )
        self.assertIs(
            pending.eligibility,
            src.LearningEligibility.CENSORED_PENDING_ADJUDICATION,
        )
        self.assertIsNone(pending.scalar_reward)

        # a proven feedback-only control loss must NOT penalize the action
        only_loss = src.evaluate_completed_decision(
            ticket,
            spec,
            adjudication=self._adjudication(src.Adjudication.FEEDBACK_ONLY_LOSS),
        )
        self.assertIs(
            only_loss.eligibility,
            src.LearningEligibility.CENSORED_FEEDBACK_ONLY_LOSS,
        )
        self.assertIsNone(only_loss.scalar_reward)
        self.assertEqual(only_loss.costs.c_authoritative_failure, 0.0)
        assert only_loss.adjudication is not None
        self.assertIs(
            only_loss.adjudication.verdict, src.Adjudication.FEEDBACK_ONLY_LOSS
        )
        # and it is definitely not the registered negative
        self.assertNotEqual(only_loss.scalar_reward, spec.r_registered_failure)

        # a timeout never carries a fabricated quality
        with self.assertRaises(src.QualityContractError):
            src.evaluate_completed_decision(
                ticket, spec, quality_components=self._quality()
            )

        # adjudication is meaningless for an already-authoritative terminal
        for terminal in (
            rtc.TerminalClass.REWARD_FINAL_EXACT,
            rtc.TerminalClass.ACTION_PATH_FAILURE,
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
        ):
            with self.subTest(terminal=terminal):
                with self.assertRaises(src.AdjudicationError):
                    src.evaluate_completed_decision(
                        self._completed_ticket(terminal=terminal),
                        spec,
                        quality_components=(
                            self._quality()
                            if terminal is rtc.TerminalClass.REWARD_FINAL_EXACT
                            else None
                        ),
                        adjudication=self._adjudication(
                            src.Adjudication.FEEDBACK_ONLY_LOSS
                        ),
                    )

        # an adjudication record needs an attributed verdict and evidence
        for field, bad in (
            ("verdict", "FEEDBACK_ONLY_LOSS"),
            ("adjudicator_id", ""),
            ("evidence_sha256", "deadbeef"),
            ("adjudicated_ns", -1),
            ("detail", ""),
        ):
            with self.subTest(field=field):
                kwargs = dict(
                    verdict=src.Adjudication.FEEDBACK_ONLY_LOSS,
                    adjudicator_id="x",
                    evidence_sha256=HEX64_B,
                    adjudicated_ns=1,
                    detail="d",
                )
                kwargs[field] = bad
                with self.assertRaises(src.AdjudicationError):
                    src.AdjudicationRecordV1(**kwargs)

    # ----------------------------------------------------------------- 12 -- #

    def test_service_failure_is_a_registered_negative(self) -> None:
        spec = self._reward_spec(r_registered_failure=-2.5)

        # a proven action-path failure is the registered negative, with no Q
        ticket = self._completed_ticket(
            terminal=rtc.TerminalClass.ACTION_PATH_FAILURE
        )
        outcome = src.evaluate_completed_decision(ticket, spec)
        self.assertIs(outcome.eligibility, src.LearningEligibility.ELIGIBLE)
        self.assertTrue(outcome.learning_eligible)
        self.assertEqual(outcome.scalar_reward, -2.5)
        self.assertIsNone(outcome.quality)
        self.assertEqual(outcome.costs.c_authoritative_failure, 1.0)
        # latency is still preserved as a raw measurement
        assert outcome.latency is not None
        self.assertEqual(outcome.latency.l_ns, 50 * MS)
        self.assertEqual(
            rtc.TERMINAL_LEARNING_DISPOSITION[ticket.terminal_class],
            "included_registered_negative_service_reward",
        )
        # Q is never fabricated for a failure
        with self.assertRaises(src.QualityContractError):
            src.evaluate_completed_decision(
                ticket, spec, quality_components=self._quality()
            )

        # a reconciled late service failure becomes authoritative only through
        # an explicit adjudication record
        timeout = self._completed_ticket(
            terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT
        )
        adjudicated = src.evaluate_completed_decision(
            timeout,
            spec,
            adjudication=self._adjudication(
                src.Adjudication.AUTHORITATIVE_SERVICE_FAILURE
            ),
        )
        self.assertIs(adjudicated.eligibility, src.LearningEligibility.ELIGIBLE)
        self.assertEqual(adjudicated.scalar_reward, -2.5)
        self.assertEqual(adjudicated.costs.c_authoritative_failure, 1.0)
        self.assertEqual(adjudicated.costs.c_deadline, 1.0)
        assert adjudicated.adjudication is not None
        self.assertIs(
            adjudicated.adjudication.verdict,
            src.Adjudication.AUTHORITATIVE_SERVICE_FAILURE,
        )
        self.assertEqual(adjudicated.adjudication.evidence_sha256, HEX64_B)
        # without the record the same ticket stays censored
        self.assertIsNone(
            src.evaluate_completed_decision(timeout, spec).scalar_reward
        )
        # the registered negative is required to be non-positive
        self.assertLessEqual(spec.r_registered_failure, 0.0)

    # ----------------------------------------------------------------- 13 -- #

    def test_infrastructure_fault_is_excluded(self) -> None:
        spec = self._reward_spec()
        ticket = self._completed_ticket(
            terminal=rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED
        )
        outcome = src.evaluate_completed_decision(ticket, spec)
        self.assertIs(
            outcome.eligibility,
            src.LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
        )
        self.assertFalse(outcome.learning_eligible)
        self.assertIsNone(outcome.scalar_reward)
        self.assertIsNone(outcome.quality)
        self.assertIsNone(outcome.latency)
        # never recorded as a service failure, and never a penalty
        self.assertIsNone(outcome.costs.c_authoritative_failure)
        self.assertNotEqual(outcome.scalar_reward, spec.r_registered_failure)
        self.assertEqual(
            rtc.TERMINAL_LEARNING_DISPOSITION[ticket.terminal_class],
            "excluded_reported_as_experimental_failure",
        )

        # an excluded outcome cannot be reshaped into a scored one
        with self.assertRaises(src.StateRewardContractError):
            src.DecisionOutcomeV1(
                terminal_class=rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
                eligibility=src.LearningEligibility.ELIGIBLE,
                costs=src.ConstraintCostsV1(None, None, None),
                reward_spec_sha256=spec.canonical_sha256(),
                scalar_reward=-1.0,
            )
        with self.assertRaises(src.StateRewardContractError):
            src.DecisionOutcomeV1(
                terminal_class=rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
                eligibility=(
                    src.LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT
                ),
                costs=src.ConstraintCostsV1(1.0, None, 1.0),
                reward_spec_sha256=spec.canonical_sha256(),
            )
        # a censored/excluded eligibility can never carry a reward
        for eligibility in (
            src.LearningEligibility.CENSORED_PENDING_ADJUDICATION,
            src.LearningEligibility.CENSORED_FEEDBACK_ONLY_LOSS,
            src.LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
        ):
            with self.subTest(eligibility=eligibility):
                with self.assertRaises(src.StateRewardContractError):
                    src.DecisionOutcomeV1(
                        terminal_class=rtc.TerminalClass.FEEDBACK_TIMEOUT,
                        eligibility=eligibility,
                        costs=src.ConstraintCostsV1(1.0, None, None),
                        reward_spec_sha256=spec.canonical_sha256(),
                        scalar_reward=-1.0,
                    )
        # ... and an eligible one must carry one
        with self.assertRaises(src.StateRewardContractError):
            src.DecisionOutcomeV1(
                terminal_class=rtc.TerminalClass.ACTION_PATH_FAILURE,
                eligibility=src.LearningEligibility.ELIGIBLE,
                costs=src.ConstraintCostsV1(0.0, 0.0, 1.0),
                reward_spec_sha256=spec.canonical_sha256(),
            )
        # cost indicators are strict
        for bad in (0.5, -1.0, 2.0, float("nan")):
            with self.subTest(c_deadline=bad):
                with self.assertRaises(src.StateRewardContractError):
                    src.ConstraintCostsV1(bad, 0.0, 0.0)
        with self.assertRaises(src.StateRewardContractError):
            src.ConstraintCostsV1(0.0, -0.1, 0.0)

    # ----------------------------------------------------------------- 14 -- #

    def test_identity_mismatch_and_cross_session_next_state_fail_closed(
        self,
    ) -> None:
        spec, norm = self._reward_spec(), self._norm()
        ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500
        )
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._quality()
        )
        state = self._state(tensor_seq=10, carla_frame_id=500)
        next_state = self._state(
            tensor_seq=12, carla_frame_id=502, observed_ns=T0 + 100 * MS
        )

        # the happy path assembles
        transition = src.build_replay_transition(
            state=state,
            next_state=next_state,
            completed_ticket=ticket,
            outcome=outcome,
            reward_spec=spec,
            normalization=norm,
        )
        self.assertEqual(transition.session_uuid, SESSION)
        self.assertEqual(transition.decision_seq, 1)
        self.assertEqual(transition.reward_tensor_seq, 10)
        self.assertEqual(transition.reward_carla_frame_id, 500)

        def _build(**overrides: Any) -> src.ReplayTransitionV1:
            kwargs: dict = dict(
                state=state,
                next_state=next_state,
                completed_ticket=ticket,
                outcome=outcome,
                reward_spec=spec,
                normalization=norm,
            )
            kwargs.update(overrides)
            return src.build_replay_transition(**kwargs)

        # the state must join the reward-requested tensor and its exact frame
        with self.assertRaises(src.TransitionIdentityError):
            _build(state=self._state(tensor_seq=11, carla_frame_id=500))
        with self.assertRaises(src.TransitionIdentityError):
            _build(state=self._state(tensor_seq=10, carla_frame_id=501))

        # one exact session throughout
        with self.assertRaises(src.TransitionIdentityError):
            _build(state=self._state(
                session_uuid=OTHER_SESSION, tensor_seq=10, carla_frame_id=500
            ))
        # a next state can never cross a session
        with self.assertRaises(src.TransitionIdentityError) as caught:
            _build(next_state=self._state(
                session_uuid=OTHER_SESSION,
                tensor_seq=12,
                observed_ns=T0 + 100 * MS,
            ))
        self.assertIn("crosses out of session", str(caught.exception))

        # the next state must be causally later than every governed tensor
        for bad_seq in (10, 11):
            with self.subTest(next_tensor_seq=bad_seq):
                with self.assertRaises(src.TransitionIdentityError):
                    _build(next_state=self._state(
                        tensor_seq=bad_seq, observed_ns=T0 + 100 * MS
                    ))
        with self.assertRaises(src.TransitionIdentityError):
            _build(next_state=self._state(
                tensor_seq=12, observed_ns=state.observed_ns - 1
            ))

        # a non-terminal transition needs a next state; a terminal one may omit
        with self.assertRaises(src.TransitionIdentityError):
            _build(next_state=None)
        terminal_transition = _build(
            next_state=None, terminated=True, episode_end_reason="route complete"
        )
        self.assertTrue(terminal_transition.terminated)
        self.assertFalse(terminal_transition.truncated)
        with self.assertRaises(src.TransitionIdentityError):
            _build(next_state=None, terminated=True)
        with self.assertRaises(src.TransitionIdentityError):
            _build(terminated=True, truncated=True, episode_end_reason="both")
        with self.assertRaises(src.TransitionIdentityError):
            _build(episode_end_reason="reason without an ending")

        # the executed action must equal the hold's action
        other_action = self._action(mode_id=9, q_e4=0)
        with self.assertRaises(src.TransitionIdentityError):
            src.ReplayTransitionV1(
                state=state,
                executed_action=other_action,
                completed_ticket=ticket,
                outcome=outcome,
                gamma_per_tensor=spec.gamma_per_tensor,
                reward_spec_sha256=spec.canonical_sha256(),
                state_normalization_spec_sha256=norm.canonical_sha256(),
                terminated=False,
                truncated=False,
                next_state=next_state,
            )

        # the outcome must have been measured for this ticket and spec
        other_ticket = self._completed_ticket(
            terminal=rtc.TerminalClass.ACTION_PATH_FAILURE,
            decision_seq=1,
            first_tensor_seq=10,
            first_frame_id=500,
        )
        with self.assertRaises(src.TransitionIdentityError):
            _build(completed_ticket=other_ticket)
        with self.assertRaises(src.TransitionIdentityError):
            _build(reward_spec=self._reward_spec(w_quality=2.0))

        # a state may never carry its own or a future outcome
        with self.assertRaises(src.TransitionIdentityError):
            _build(state=self._state(
                tensor_seq=10,
                carla_frame_id=500,
                previous=self._previous(decision_seq=1),
            ))
        with self.assertRaises(src.TransitionIdentityError):
            _build(state=self._state(
                tensor_seq=10,
                carla_frame_id=500,
                previous=self._previous(decision_seq=7),
            ))
        # a genuinely earlier previous decision is fine
        _build(state=self._state(
            tensor_seq=10, carla_frame_id=500,
            previous=self._previous(decision_seq=0),
        ))

    # ----------------------------------------------------------------- 15 -- #

    def test_duration_and_discount_are_derived_from_the_frozen_hold(self) -> None:
        norm = self._norm()
        for extra_reuses, expected_d in ((0, 2), (1, 3), (3, 5)):
            with self.subTest(extra_reuses=extra_reuses):
                gamma = 0.97
                spec = self._reward_spec(gamma_per_tensor=gamma)
                ticket = self._completed_ticket(
                    decision_seq=1,
                    first_tensor_seq=10,
                    first_frame_id=500,
                    extra_reuses=extra_reuses,
                )
                self.assertEqual(ticket.hold_duration_tensors, expected_d)
                outcome = src.evaluate_completed_decision(
                    ticket, spec, quality_components=self._quality()
                )
                transition = src.build_replay_transition(
                    state=self._state(tensor_seq=10, carla_frame_id=500),
                    next_state=self._state(
                        tensor_seq=10 + expected_d,
                        observed_ns=T0 + 500 * MS,
                    ),
                    completed_ticket=ticket,
                    outcome=outcome,
                    reward_spec=spec,
                    normalization=norm,
                )
                # d comes from the frozen Phase-2 hold, not from a caller
                self.assertEqual(transition.hold_duration_tensors, expected_d)
                self.assertEqual(
                    transition.hold_duration_tensors, ticket.hold.tensor_count
                )
                self.assertEqual(len(ticket.hold.tensor_seqs), expected_d)
                self.assertAlmostEqual(
                    transition.discount_multiplier, gamma ** expected_d, 12
                )
                self.assertGreaterEqual(expected_d, rtc.K_MIN_TENSORS)

        # neither d nor the multiplier is a constructor parameter
        import inspect

        parameters = set(
            inspect.signature(src.ReplayTransitionV1.__init__).parameters
        )
        for forbidden in (
            "hold_duration_tensors",
            "discount_multiplier",
            "d",
            "duration",
        ):
            with self.subTest(parameter=forbidden):
                self.assertNotIn(forbidden, parameters)
        for forbidden in ("hold_duration_tensors", "discount_multiplier"):
            self.assertNotIn(
                forbidden,
                set(inspect.signature(src.build_replay_transition).parameters),
            )
        # gamma = 1 is admissible and yields a unit multiplier
        one = self._reward_spec(gamma_per_tensor=1.0)
        ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500
        )
        self.assertEqual(
            src.build_replay_transition(
                state=self._state(tensor_seq=10, carla_frame_id=500),
                next_state=self._state(tensor_seq=14, observed_ns=T0 + 500 * MS),
                completed_ticket=ticket,
                outcome=src.evaluate_completed_decision(
                    ticket, one, quality_components=self._quality()
                ),
                reward_spec=one,
                normalization=norm,
            ).discount_multiplier,
            1.0,
        )

    # ----------------------------------------------------------------- 16 -- #

    def test_canonical_serialization_and_hashing_are_deterministic(self) -> None:
        spec, norm, fresh = self._reward_spec(), self._norm(), self._freshness()
        ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500
        )
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._quality()
        )
        state = self._state(
            tensor_seq=10, carla_frame_id=500, previous=self._previous()
        )
        next_state = self._state(tensor_seq=12, observed_ns=T0 + 100 * MS)
        transition = src.build_replay_transition(
            state=state,
            next_state=next_state,
            completed_ticket=ticket,
            outcome=outcome,
            reward_spec=spec,
            normalization=norm,
        )

        # every record's hash is reproducible with a locally written canonicalizer
        for label, record in (
            ("state", state),
            ("normalization", norm),
            ("reward_spec", spec),
            ("feature_vector", src.build_policy_features(state, norm, fresh)),
            ("transition", transition),
        ):
            with self.subTest(record=label):
                payload = record.to_canonical_dict()
                self.assertEqual(
                    record.canonical_sha256(), _independent_sha256(payload)
                )
                # byte deterministic across repeated calls
                self.assertEqual(
                    _independent_canonical_bytes(payload),
                    _independent_canonical_bytes(record.to_canonical_dict()),
                )

        self.assertEqual(
            transition.canonical_bytes(),
            _independent_canonical_bytes(transition.to_canonical_dict()),
        )
        # rebuilding the identical transition gives identical bytes
        rebuilt = src.build_replay_transition(
            state=state,
            next_state=next_state,
            completed_ticket=ticket,
            outcome=outcome,
            reward_spec=spec,
            normalization=norm,
        )
        self.assertEqual(rebuilt.canonical_bytes(), transition.canonical_bytes())
        self.assertEqual(rebuilt.canonical_sha256(), transition.canonical_sha256())
        # and any real change moves the hash
        self.assertNotEqual(
            src.build_replay_transition(
                state=state,
                next_state=self._state(
                    tensor_seq=13, observed_ns=T0 + 100 * MS
                ),
                completed_ticket=ticket,
                outcome=outcome,
                reward_spec=spec,
                normalization=norm,
            ).canonical_sha256(),
            transition.canonical_sha256(),
        )

        # the serialized transition preserves everything the contract requires
        payload = transition.to_canonical_dict()
        for key in (
            "session_uuid",
            "decision_seq",
            "reward_tensor_seq",
            "reward_carla_frame_id",
            "state",
            "next_state",
            "executed_action",
            "completed_ticket",
            "outcome",
            "hold_duration_tensors",
            "discount_multiplier",
            "gamma_per_tensor",
            "reward_spec_sha256",
            "state_normalization_spec_sha256",
            "terminated",
            "truncated",
            "episode_end_reason",
            "schema_id",
            "schema_sha256",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertIn("hold", payload["completed_ticket"])
        self.assertIsNotNone(payload["outcome"]["quality"]["components"])
        self.assertEqual(payload["reward_spec_sha256"], spec.canonical_sha256())
        self.assertEqual(
            payload["state_normalization_spec_sha256"], norm.canonical_sha256()
        )
        # every serializable record round-trips through canonical bytes, and
        # every hash is independently reproducible
        for label, record in (
            ("freshness_policy", fresh),
            ("previous_outcome", self._previous()),
            ("quality_components", self._quality()),
            ("quality_evaluation", spec.evaluate_quality(self._quality())),
            ("latency", src.LatencyMeasurementV1.from_completed_ticket(ticket)),
            ("ack_binding", self._ack()),
            ("quality_evidence", self._evidence()),
            ("constraint_costs", src.ConstraintCostsV1(0.0, 0.0, 1.0)),
            ("adjudication", self._adjudication(src.Adjudication.PENDING)),
            ("decision_outcome", outcome),
        ):
            with self.subTest(record=label):
                payload = record.to_canonical_dict()
                raw = _independent_canonical_bytes(payload)
                self.assertEqual(raw, _independent_canonical_bytes(payload))
                self.assertIn("record", payload)
                if hasattr(record, "canonical_sha256"):
                    self.assertEqual(
                        record.canonical_sha256(), _independent_sha256(payload)
                    )
                if hasattr(record, "canonical_bytes"):
                    self.assertEqual(record.canonical_bytes(), raw)

        # records are immutable
        for record in (state, spec, norm, transition, outcome):
            with self.subTest(record=type(record).__name__):
                with self.assertRaises(Exception):
                    record.session_uuid = "x"  # type: ignore[misc]

    # ----------------------------------------------------------------- 17 -- #

    def test_schema_binds_all_four_frozen_dependencies(self) -> None:
        descriptor = src.SCHEMA_DESCRIPTOR
        self.assertIsInstance(descriptor, MappingProxyType)
        self.assertEqual(
            src.SCHEMA_ID, "splitfusion_hybrid_sac_state_reward_transition_v1"
        )
        self.assertEqual(src.SCHEMA_VERSION, 1)
        self.assertEqual(src.SCHEMA_SHA256, _independent_sha256(descriptor))
        self.assertEqual(len(src.SCHEMA_SHA256), 64)
        int(src.SCHEMA_SHA256, 16)

        dependencies = descriptor["dependencies"]
        # 1. the locked action catalog
        self.assertEqual(
            dependencies["action_catalog"]["sha256"], ac.CATALOG_SHA256
        )
        self.assertEqual(
            dependencies["action_catalog"]["sha256"],
            "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3",
        )
        self.assertEqual(
            dependencies["action_catalog"]["schema"], ac.CATALOG_SCHEMA
        )
        self.assertEqual(dependencies["action_catalog"]["joint_mode_count"], 12)
        self.assertEqual(
            tuple(dependencies["action_catalog"]["q_e4_bounds"]), (0, 9800)
        )
        # 2. transaction identity (and the action-identity sub-schema)
        self.assertEqual(
            dependencies["transaction_identity"]["schema_id"], ti.SCHEMA_ID
        )
        self.assertEqual(
            dependencies["transaction_identity"]["schema_sha256"],
            ti.SCHEMA_SHA256,
        )
        self.assertEqual(
            dependencies["transaction_identity"]["schema_version"],
            ti.SCHEMA_VERSION,
        )
        self.assertEqual(
            dependencies["executed_action_identity"]["schema_sha256"],
            ti.ACTION_IDENTITY_SCHEMA_SHA256,
        )
        # 3. scene descriptors, SI and P40 only
        self.assertEqual(
            dependencies["scene_descriptors"]["schema_id"], sd.SCHEMA_ID
        )
        self.assertEqual(
            dependencies["scene_descriptors"]["schema_sha256"], sd.SCHEMA_SHA256
        )
        self.assertEqual(
            tuple(dependencies["scene_descriptors"]["descriptors"]),
            ("camera_si", "radar_p40"),
        )
        # 4. the Phase-3b.1 reward-ticket controller, at version 2
        self.assertEqual(
            dependencies["reward_ticket_controller"]["schema_id"],
            rtc.CONTROLLER_SCHEMA_ID,
        )
        self.assertEqual(
            dependencies["reward_ticket_controller"]["schema_sha256"],
            rtc.CONTROLLER_SCHEMA_SHA256,
        )
        self.assertEqual(
            dependencies["reward_ticket_controller"]["schema_version"], 2
        )
        self.assertEqual(
            dependencies["reward_ticket_controller"]["reward_deadline_ns"],
            200_000_000,
        )
        self.assertEqual(
            dependencies["reward_ticket_controller"]["minimum_hold_tensors"], 2
        )

        # the terminal treatments agree with the controller's own dispositions
        self.assertEqual(
            set(descriptor["reward"]["terminal_handling"]),
            {terminal.value for terminal in rtc.TerminalClass},
        )
        self.assertEqual(
            dict(descriptor["reward"]["controller_dispositions"]),
            {
                terminal.value: disposition
                for terminal, disposition in (
                    rtc.TERMINAL_LEARNING_DISPOSITION.items()
                )
            },
        )
        # the frozen feature order is part of the hashed contract
        self.assertEqual(
            tuple(descriptor["causal_state"]["policy_feature_order"]),
            src.POLICY_FEATURE_ORDER,
        )
        self.assertEqual(
            descriptor["causal_state"]["policy_feature_count"],
            src.POLICY_FEATURE_COUNT,
        )
        # the deployment boundary is stated, not implied
        self.assertIn("does NOT claim", descriptor["quality"]["deployment_claim"])
        self.assertIn("CARLA", descriptor["quality"]["ground_truth_source"])
        # the superseded action-space sketch is explicitly disowned
        self.assertIn("superseded", descriptor["action_space"])
        self.assertIn("feed-forward", descriptor["policy_class"])
        # immutable at depth
        with self.assertRaises(TypeError):
            descriptor["version"] = 2  # type: ignore[index]
        with self.assertRaises(TypeError):
            descriptor["dependencies"]["action_catalog"]["sha256"] = "x"  # type: ignore[index]

    # ----------------------------------------------------------------- 18 -- #

    def test_no_filesystem_network_or_accelerator_side_effect_on_import(
        self,
    ) -> None:
        """Import the contract with the filesystem and sockets blocked.

        Third-party initialisation (numpy, OpenCV) is allowed to touch the
        filesystem before the block goes up -- that is outside this repository's
        control.  The claim proved here is the one this phase owns: importing the
        four first-party contract modules reads no file, opens no socket and
        initialises no accelerator or simulator.
        """
        probe = r'''
import builtins, io, os, sys

# Let third-party libraries finish their own initialisation first.
import numpy, cv2  # noqa: F401

_forbidden = []


def _blocked_open(*args, **kwargs):
    _forbidden.append(("open", args[:1]))
    raise AssertionError(f"filesystem access during import: {args[:1]}")


builtins.open = _blocked_open
io.open = _blocked_open

import socket


def _blocked_socket(*args, **kwargs):
    raise AssertionError("network access during import")


socket.socket = _blocked_socket
socket.create_connection = _blocked_socket

before = set(sys.modules)
from abiodun.rl_agent.splitfusion_hybrid_sac_v1 import (  # noqa: F401
    state_reward_transition_contract as contract,
)

# The locked catalog must NOT have been read at import time.
from abiodun.rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac
assert ac.default_contract.cache_info().currsize == 0, "catalog read on import"

for banned in ("carla", "torch", "docker", "pycuda", "tensorflow"):
    assert banned not in sys.modules, f"{banned} imported by the contract"

assert not _forbidden, _forbidden
assert contract.SCHEMA_VERSION == 1
assert len(contract.SCHEMA_SHA256) == 64
print("IMPORT_CLEAN", contract.SCHEMA_SHA256)
'''
        package_root = Path(__file__).resolve().parents[3]
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(package_root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"probe failed\nstdout:\n{completed.stdout}\nstderr:\n"
            f"{completed.stderr}",
        )
        self.assertIn("IMPORT_CLEAN", completed.stdout)
        self.assertIn(src.SCHEMA_SHA256, completed.stdout)

        # the module also declares no environment, storage or training symbol
        for banned in (
            "Env",
            "GymEnv",
            "ReplayBuffer",
            "Actor",
            "Critic",
            "train",
            "optimizer",
            "LSTM",
            "GRU",
            "step",
            "reset",
        ):
            with self.subTest(symbol=banned):
                self.assertNotIn(banned, src.__all__)

    # ----------------------------------------------------------------- 19 -- #

    def test_remaining_verdicts_accessors_and_record_guards(self) -> None:
        spec, norm = self._reward_spec(), self._norm()

        # an adjudicated instrument fault on a timeout is excluded, not scored
        timeout = self._completed_ticket(
            terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT
        )
        excluded = src.evaluate_completed_decision(
            timeout,
            spec,
            adjudication=self._adjudication(src.Adjudication.INFRASTRUCTURE_FAULT),
        )
        self.assertIs(
            excluded.eligibility,
            src.LearningEligibility.EXCLUDED_INFRASTRUCTURE_FAULT,
        )
        self.assertIsNone(excluded.scalar_reward)
        self.assertIsNone(excluded.costs.c_authoritative_failure)
        self.assertEqual(excluded.costs.c_deadline, 1.0)
        assert excluded.adjudication is not None
        self.assertIs(
            excluded.adjudication.verdict, src.Adjudication.INFRASTRUCTURE_FAULT
        )
        self.assertEqual(len(src.Adjudication), 4)

        # no terminal class other than REWARD_FINAL_EXACT accepts a quality
        for terminal in (
            rtc.TerminalClass.INFRASTRUCTURE_FAULT_EXCLUDED,
            rtc.TerminalClass.ACTION_PATH_FAILURE,
            rtc.TerminalClass.FEEDBACK_TIMEOUT,
        ):
            with self.subTest(terminal=terminal):
                with self.assertRaises(src.QualityContractError):
                    src.evaluate_completed_decision(
                        self._completed_ticket(terminal=terminal),
                        spec,
                        quality_components=self._quality(),
                    )
        # ... and REWARD_FINAL_EXACT requires one
        with self.assertRaises(src.QualityContractError):
            src.evaluate_completed_decision(self._completed_ticket(), spec)

        # non-record arguments fail closed everywhere
        ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500
        )
        with self.assertRaises(src.StateRewardContractError):
            src.evaluate_completed_decision("not-a-ticket", spec)
        with self.assertRaises(src.RewardSpecError):
            src.evaluate_completed_decision(ticket, {"w_quality": 1.0})
        with self.assertRaises(src.QualityContractError):
            src.evaluate_completed_decision(
                ticket, spec, quality_components={"iou": 1.0}
            )
        with self.assertRaises(src.AdjudicationError):
            src.evaluate_completed_decision(
                ticket, spec, quality_components=self._quality(),
                adjudication="FEEDBACK_ONLY_LOSS",
            )
        with self.assertRaises(src.QualityContractError):
            spec.evaluate_quality({"vehicle_seg_iou": 1.0})
        with self.assertRaises(src.StateRewardContractError):
            src.LatencyMeasurementV1.from_completed_ticket("not-a-ticket")
        with self.assertRaises(src.CausalStateError):
            self._freshness().assert_fresh("not-a-state")
        with self.assertRaises(src.CausalStateError):
            src.build_policy_features("not-a-state", norm, self._freshness())
        with self.assertRaises(src.RewardSpecError):
            src.build_replay_transition(
                state=self._state(tensor_seq=10, carla_frame_id=500),
                next_state=None, completed_ticket=ticket,
                outcome=src.evaluate_completed_decision(
                    ticket, spec, quality_components=self._quality()
                ),
                reward_spec="spec", normalization=norm,
                terminated=True, episode_end_reason="x",
            )
        with self.assertRaises(src.NormalizationSpecError):
            src.build_replay_transition(
                state=self._state(tensor_seq=10, carla_frame_id=500),
                next_state=None, completed_ticket=ticket,
                outcome=src.evaluate_completed_decision(
                    ticket, spec, quality_components=self._quality()
                ),
                reward_spec=spec, normalization="norm",
                terminated=True, episode_end_reason="x",
            )
        with self.assertRaises(src.TransitionIdentityError):
            src.build_replay_transition(
                state=self._state(tensor_seq=10, carla_frame_id=500),
                next_state=None, completed_ticket="ticket",
                outcome=src.evaluate_completed_decision(
                    ticket, spec, quality_components=self._quality()
                ),
                reward_spec=spec, normalization=norm,
                terminated=True, episode_end_reason="x",
            )

        # the transition's documented accessors all delegate to frozen records
        outcome = src.evaluate_completed_decision(
            ticket, spec, quality_components=self._quality()
        )
        transition = src.build_replay_transition(
            state=self._state(tensor_seq=10, carla_frame_id=500),
            next_state=self._state(tensor_seq=13, observed_ns=T0 + 300 * MS),
            completed_ticket=ticket,
            outcome=outcome,
            reward_spec=spec,
            normalization=norm,
        )
        self.assertIs(transition.hold, ticket.hold)
        self.assertIs(transition.quality, outcome.quality)
        self.assertIs(transition.latency, outcome.latency)
        self.assertIs(transition.costs, outcome.costs)
        self.assertIs(transition.eligibility, outcome.eligibility)
        self.assertIs(transition.terminal_class, ticket.terminal_class)
        self.assertEqual(transition.scalar_reward, outcome.scalar_reward)
        self.assertTrue(transition.learning_eligible)
        self.assertIs(transition.executed_action, ticket.action)

        # direct construction still validates every field
        valid: dict = dict(
            state=self._state(tensor_seq=10, carla_frame_id=500),
            executed_action=ticket.action,
            completed_ticket=ticket,
            outcome=outcome,
            gamma_per_tensor=spec.gamma_per_tensor,
            reward_spec_sha256=spec.canonical_sha256(),
            state_normalization_spec_sha256=norm.canonical_sha256(),
            terminated=False,
            truncated=False,
            next_state=self._state(tensor_seq=13, observed_ns=T0 + 300 * MS),
        )
        src.ReplayTransitionV1(**valid)
        for label, override in (
            ("gamma zero", {"gamma_per_tensor": 0.0}),
            ("gamma above one", {"gamma_per_tensor": 1.5}),
            ("gamma nan", {"gamma_per_tensor": float("nan")}),
            ("gamma bool", {"gamma_per_tensor": True}),
            ("bad reward hash", {"reward_spec_sha256": "deadbeef"}),
            ("bad norm hash", {"state_normalization_spec_sha256": "x" * 64}),
            ("terminated not bool", {"terminated": 1}),
            ("truncated not bool", {"truncated": 1}),
            ("state wrong type", {"state": "s"}),
            ("action wrong type", {"executed_action": "a"}),
            ("ticket wrong type", {"completed_ticket": "t"}),
            ("outcome wrong type", {"outcome": "o"}),
            ("next state wrong type", {"next_state": "n"}),
        ):
            with self.subTest(case=label):
                with self.assertRaises(src.TransitionIdentityError):
                    src.ReplayTransitionV1(**{**valid, **override})

        # a truncated transition is the other legal way to omit a next state
        truncated = src.build_replay_transition(
            state=self._state(tensor_seq=10, carla_frame_id=500),
            next_state=None,
            completed_ticket=ticket,
            outcome=outcome,
            reward_spec=spec,
            normalization=norm,
            truncated=True,
            episode_end_reason="time limit",
        )
        self.assertTrue(truncated.truncated)
        self.assertFalse(truncated.terminated)
        self.assertIsNone(truncated.next_state)
        self.assertIsNone(truncated.to_canonical_dict()["next_state"])

        # quality/evaluation records validate their own derived fields
        with self.assertRaises(src.QualityContractError):
            src.QualityEvaluationV1(
                components=self._quality(), q_seg=1.5, q_loc=0.5, quality=0.8,
                loc_vehicle_normalized=None, loc_person_normalized=None,
                segmentation_weights_used={"person": 1.0},
                localization_weights_used={"person": 1.0},
                reward_spec_sha256=spec.canonical_sha256(),
            )
        with self.assertRaises(src.QualityContractError):
            src.QualityEvaluationV1(
                components="c", q_seg=0.5, q_loc=0.5, quality=0.5,
                loc_vehicle_normalized=None, loc_person_normalized=None,
                segmentation_weights_used={"person": 1.0},
                localization_weights_used={"person": 1.0},
                reward_spec_sha256=spec.canonical_sha256(),
            )
        with self.assertRaises(src.QualityContractError):
            src.QualityEvaluationV1(
                components=self._quality(), q_seg=0.5, q_loc=0.5, quality=0.5,
                loc_vehicle_normalized=None, loc_person_normalized=None,
                segmentation_weights_used={},
                localization_weights_used={"person": 1.0},
                reward_spec_sha256=spec.canonical_sha256(),
            )
        # the weights actually used are frozen on the evaluation record
        evaluation = spec.evaluate_quality(self._quality())
        self.assertIsInstance(
            evaluation.segmentation_weights_used, MappingProxyType
        )
        with self.assertRaises(TypeError):
            evaluation.segmentation_weights_used["person"] = 9.0  # type: ignore[index]

        # the feature-vector record validates width, finiteness and provenance
        for label, override in (
            ("wrong width", {"values": (0.0,)}),
            ("not a tuple", {"values": [0.0] * src.POLICY_FEATURE_COUNT}),
            ("nan value", {
                "values": (float("nan"),) + (0.0,) * (
                    src.POLICY_FEATURE_COUNT - 1
                )
            }),
            ("int value", {
                "values": (0,) + (0.0,) * (src.POLICY_FEATURE_COUNT - 1)
            }),
            ("bad spec hash", {"state_normalization_spec_sha256": "nope"}),
            ("empty policy id", {"freshness_policy_id": ""}),
        ):
            with self.subTest(case=label):
                kwargs: dict = dict(
                    values=(0.0,) * src.POLICY_FEATURE_COUNT,
                    state_normalization_spec_sha256=norm.canonical_sha256(),
                    freshness_policy_id="p",
                )
                kwargs.update(override)
                with self.assertRaises(src.StateRewardContractError):
                    src.PolicyFeatureVectorV1(**kwargs)

        # the frozen-order guard is executable and catches a tampered order
        original = src.POLICY_FEATURE_ORDER
        try:
            src.POLICY_FEATURE_ORDER = original + ("carla_frame_id",)  # type: ignore[misc]
            with self.assertRaises(src.StateRewardContractError):
                src.assert_policy_features_exclude_forbidden_fields()
            src.POLICY_FEATURE_ORDER = original + (original[0],)  # type: ignore[misc]
            with self.assertRaises(src.StateRewardContractError):
                src.assert_policy_features_exclude_forbidden_fields()
        finally:
            src.POLICY_FEATURE_ORDER = original  # type: ignore[misc]
        src.assert_policy_features_exclude_forbidden_fields()

    # ----------------------------------------------------------------- 20 -- #

    def test_quality_ack_is_anchor_only_and_never_snaps_off_anchor(self) -> None:
        """Clarifications 1-4: anchor-only adapter, fail closed, identity key."""
        anchor_q = set(self.contract.q_anchor_order)
        self.assertTrue(src.QUALITY_ACK_IS_ANCHOR_ONLY)
        self.assertEqual(src.QUALITY_ACK_SCHEMA, "sf_priv_quality_ack.v1")
        self.assertEqual(
            src.QUALITY_ACK_REQUIRED_ANCHOR_FIELDS, ("action_id", "profile_id")
        )
        self.assertEqual(src.QUALITY_ACK_ANCHOR_ACTION_COUNT, 72)

        # an exactly-registered anchor binds, and names that exact anchor
        on_anchor = self._action(mode_id=5, q_e4=5000)
        self.assertIn(on_anchor.q_e4, anchor_q)
        self.assertTrue(on_anchor.is_registered_anchor)
        binding = self._ack(on_anchor)
        self.assertEqual(binding.action_id, on_anchor.action_id)
        self.assertEqual(binding.profile_id, on_anchor.profile_id)
        self.assertLess(binding.action_id, 72)
        binding.assert_matches_action(on_anchor)

        # an arbitrary continuous q has no anchor identity and is REFUSED
        off_anchor = self._action(mode_id=5, q_e4=4321)
        self.assertNotIn(off_anchor.q_e4, anchor_q)
        self.assertFalse(off_anchor.is_registered_anchor)
        self.assertIsNone(off_anchor.action_id)
        self.assertIsNone(off_anchor.profile_id)
        with self.assertRaises(src.OffAnchorQualityAckError) as caught:
            self._ack(off_anchor)
        message = str(caught.exception)
        # the refusal explains itself and names the protocol-v2 requirement
        self.assertIn("never be snapped", src.PROTOCOL_V2_REQUIREMENT)
        self.assertIn("nearest anchor", message)
        self.assertIn("protocol-v2", message)
        # nothing was snapped: the nearest anchor's id appears nowhere
        nearest = self.contract.find_anchor(
            off_anchor.family, off_anchor.quantizer, 5000
        )
        self.assertIsNotNone(nearest)
        with self.assertRaises(src.OffAnchorQualityAckError):
            src.QualityEvidenceV1.for_action(
                off_anchor,
                gt_source_detail="x",
                ack_binding=self._ack(on_anchor),
            )

        # evidence for an off-anchor action is keyed on the FULL identity hash
        off_evidence = self._evidence(off_anchor, with_ack=False)
        self.assertEqual(
            off_evidence.executed_action_sha256, off_anchor.canonical_sha256()
        )
        self.assertFalse(off_evidence.is_anchor_evidenced)
        self.assertIsNone(off_evidence.raw_quality_ack_sha256)
        self.assertIsNone(off_evidence.detailed_evidence_sha256)
        self.assertIn(
            "protocol-v2",
            off_evidence.to_canonical_dict()["protocol_v2_requirement"],
        )
        # the identity hash key is well defined on anchor too, and differs
        self.assertNotEqual(
            off_evidence.executed_action_sha256,
            self._evidence(on_anchor).executed_action_sha256,
        )

        # an ACK may never be re-attributed to a different anchor
        other_anchor = self._action(mode_id=6, q_e4=9000)
        with self.assertRaises(src.QualityContractError):
            self._ack(on_anchor).assert_matches_action(other_anchor)
        with self.assertRaises(src.QualityContractError):
            src.QualityEvidenceV1.for_action(
                other_anchor,
                gt_source_detail="x",
                ack_binding=self._ack(on_anchor),
            )

        # binding fields are validated fail-closed
        for field, bad in (
            ("raw_quality_ack_sha256", "deadbeef"),
            ("detailed_evidence_sha256", ("e" * 64).upper()),
            ("action_id", 72),
            ("action_id", -1),
            ("profile_id", ""),
            ("evaluator_mode", ""),
            ("ack_schema", "sf_priv_quality_ack.v2"),
            ("ack_protocol_version", 2),
            ("ack_source", "SOMETHING_ELSE"),
        ):
            with self.subTest(field=field, value=bad):
                kwargs: dict = dict(
                    raw_quality_ack_sha256="c" * 64,
                    detailed_evidence_sha256="d" * 64,
                    action_id=3,
                    profile_id="p",
                    evaluator_mode="m",
                )
                kwargs[field] = bad
                with self.assertRaises(src.QualityContractError):
                    src.QualityAckBindingV1(**kwargs)

        # an anchor action's transition must carry its ACK evidence
        spec, norm = self._reward_spec(), self._norm()
        ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500,
            action=on_anchor,
        )
        unbound = self._quality(evidence=self._evidence(on_anchor, with_ack=False))
        with self.assertRaises(src.TransitionIdentityError) as caught:
            src.build_replay_transition(
                state=self._state(tensor_seq=10, carla_frame_id=500),
                next_state=self._state(tensor_seq=13, observed_ns=T0 + 300 * MS),
                completed_ticket=ticket,
                outcome=src.evaluate_completed_decision(
                    ticket, spec, quality_components=unbound
                ),
                reward_spec=spec,
                normalization=norm,
            )
        self.assertIn("evidence packet", str(caught.exception))

        # an off-anchor action is a legitimate, fully evaluable transition with
        # no v1 ACK at all -- which is exactly the protocol-v2 gap
        off_ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500,
            action=off_anchor,
        )
        off_transition = src.build_replay_transition(
            state=self._state(tensor_seq=10, carla_frame_id=500),
            next_state=self._state(tensor_seq=13, observed_ns=T0 + 300 * MS),
            completed_ticket=off_ticket,
            outcome=src.evaluate_completed_decision(
                off_ticket,
                spec,
                quality_components=self._quality(
                    evidence=self._evidence(off_anchor, with_ack=False)
                ),
            ),
            reward_spec=spec,
            normalization=norm,
        )
        self.assertIsNone(off_transition.raw_quality_ack_sha256)
        self.assertEqual(
            off_transition.executed_action_sha256, off_anchor.canonical_sha256()
        )

    # ----------------------------------------------------------------- 21 -- #

    def test_gt_absent_classes_are_masked_not_rewarded(self) -> None:
        """Clarifications 6, 7, 9: privileged GT, undefined masks, no F1."""
        # CARLA_GT_EXACT is privileged and non-deployable, as a typed value
        self.assertEqual(len(src.GroundTruthSource), 1)
        source = src.GroundTruthSource.CARLA_GT_EXACT
        self.assertEqual(source.value, "CARLA_GT_EXACT")
        self.assertTrue(source.privileged)
        self.assertFalse(source.deployable)
        components = self._quality()
        self.assertIs(components.gt_source, source)
        payload = components.to_canonical_dict()["evidence"]
        self.assertTrue(payload["gt_privileged"])
        self.assertFalse(payload["gt_deployable"])
        self.assertEqual(payload["gt_source"], "CARLA_GT_EXACT")
        with self.assertRaises(src.QualityContractError):
            src.QualityEvidenceV1(
                gt_source="CARLA_GT_EXACT",
                executed_action_sha256="a" * 64,
                gt_source_detail="x",
            )

        # a class with zero GT support may not be marked valid at all
        for support in (0, None):
            with self.subTest(person_gt_support=support):
                with self.assertRaises(src.UndefinedClassSupportError) as caught:
                    self._quality(person_gt_support=support)
                if support == 0:
                    self.assertIn("perfect recall", str(caught.exception))
        # ... in particular a GT-absent localization cannot be a zero error
        with self.assertRaises(src.UndefinedClassSupportError):
            self._quality(
                person_gt_support=0,
                person_seg_valid=False,
                person_seg_iou=None,
                person_localization_error_m=0.0,
            )
        # the only admissible representation is the undefined mask
        masked = self._quality(
            person_gt_support=0,
            person_seg_valid=False,
            person_loc_valid=False,
            person_seg_iou=None,
            person_localization_error_m=None,
        )
        self.assertEqual(masked.undefined_classes, ("person",))
        self.assertEqual(masked.person_gt_support, 0)
        self.assertIsNone(masked.person_localization_error_m)
        # and it is excluded from renormalization, not scored 0 or 1
        spec = self._reward_spec()
        evaluation = spec.evaluate_quality(masked)
        self.assertIsNone(evaluation.loc_person_normalized)
        self.assertNotIn("person", evaluation.localization_weights_used)
        self.assertNotIn("person", evaluation.segmentation_weights_used)
        self.assertAlmostEqual(
            evaluation.q_loc,
            math.exp(-0.95 / spec.tau_vehicle_m),
            12,
        )
        # a perfect-recall reading would have given q_loc = 1.0; it did not
        self.assertNotAlmostEqual(evaluation.q_loc, 1.0, 6)
        self.assertEqual(masked.to_canonical_dict()["undefined_classes"], ["person"])

        # no false positives exist in the v1 evidence, so no F1/precision claim
        self.assertFalse(src.QUALITY_ACK_FALSE_POSITIVE_COUNTS_AVAILABLE)
        self.assertFalse(components.supports_detection_f1_claim)
        self.assertFalse(components.evidence.supports_detection_f1_claim)
        self.assertEqual(
            src.QUALITY_ACK_DERIVABLE_DETECTION_METRICS, ("recall",)
        )
        for metric in ("precision", "f1", "comprehensive_detection_quality"):
            with self.subTest(metric=metric):
                self.assertIn(metric, src.QUALITY_ACK_UNDERIVABLE_DETECTION_METRICS)
                self.assertNotIn(metric, src.QUALITY_ACK_DERIVABLE_DETECTION_METRICS)
        # nothing in the module claims to compute them
        for banned in ("f1", "precision", "average_precision", "map_score"):
            with self.subTest(symbol=banned):
                self.assertFalse(
                    any(banned == name.lower() for name in src.__all__)
                )

        # SI and P40 are candidates, not established predictors
        self.assertEqual(src.CANDIDATE_STATE_FEATURES, ("camera_si", "radar_p40"))
        for phrase in ("candidate", "ablation", "SHAP", "NOT established"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, src.CANDIDATE_FEATURE_VALIDATION_REQUIRED)

    # ----------------------------------------------------------------- 22 -- #

    def test_latency_clock_domain_and_bound_evidence_hashes(self) -> None:
        """Clarifications 5, 8, and the declared-literal cross-check."""
        spec, norm = self._reward_spec(), self._norm()

        # reward latency is UE-local monotonic, from both controller timestamps
        self.assertEqual(src.REWARD_LATENCY_CLOCK_DOMAIN, "UE_LOCAL_MONOTONIC")
        ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500,
            resolution_offset_ns=64 * MS,
        )
        latency = src.LatencyMeasurementV1.from_completed_ticket(ticket)
        self.assertEqual(latency.clock_domain, "UE_LOCAL_MONOTONIC")
        self.assertEqual(latency.opened_ns, ticket.opened_ns)
        self.assertEqual(latency.resolution_ns, ticket.resolution_ns)
        self.assertEqual(latency.l_ns, 64 * MS)
        payload = latency.to_canonical_dict()
        self.assertEqual(payload["clock_domain"], "UE_LOCAL_MONOTONIC")
        for forbidden in (
            "map_install_ack_latency",
            "quality_ack_wall_timings",
            "wall_clock",
            "mixed_wall_and_monotonic",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIn(forbidden, payload["forbidden_sources"])
        # the ACK's own timings are wall-clock and are not part of the record
        self.assertEqual(src.QUALITY_ACK_TIMING_CLOCK_DOMAIN, "WALL")
        for wall_field in (
            "gt_ready_wall_ns",
            "evaluation_completed_wall_ns",
            "ack_emit_start_wall_ns",
            "model_ready_wall_ns",
        ):
            with self.subTest(field=wall_field):
                self.assertNotIn(wall_field, payload)
                self.assertNotIn(
                    wall_field, str(src.SCHEMA_DESCRIPTOR["reward"]["latency"])
                )

        # every quality-bearing transition binds both evidence hashes
        on_anchor = self._action(mode_id=5, q_e4=5000)
        anchor_ticket = self._completed_ticket(
            decision_seq=1, first_tensor_seq=10, first_frame_id=500,
            action=on_anchor,
        )
        transition = src.build_replay_transition(
            state=self._state(tensor_seq=10, carla_frame_id=500),
            next_state=self._state(tensor_seq=13, observed_ns=T0 + 300 * MS),
            completed_ticket=anchor_ticket,
            outcome=src.evaluate_completed_decision(
                anchor_ticket,
                spec,
                quality_components=self._quality(
                    evidence=self._evidence(on_anchor)
                ),
            ),
            reward_spec=spec,
            normalization=norm,
        )
        self.assertEqual(transition.raw_quality_ack_sha256, "c" * 64)
        self.assertEqual(transition.detailed_evidence_sha256, "d" * 64)
        self.assertEqual(
            transition.executed_action_sha256, on_anchor.canonical_sha256()
        )
        serialized = transition.to_canonical_dict()
        for key in (
            "raw_quality_ack_sha256",
            "detailed_evidence_sha256",
            "executed_action_sha256",
            "quality_evidence",
            "reward_latency_clock_domain",
        ):
            with self.subTest(key=key):
                self.assertIn(key, serialized)
        self.assertEqual(serialized["raw_quality_ack_sha256"], "c" * 64)
        self.assertEqual(serialized["detailed_evidence_sha256"], "d" * 64)
        self.assertEqual(
            serialized["reward_latency_clock_domain"], "UE_LOCAL_MONOTONIC"
        )
        self.assertEqual(
            serialized["quality_evidence"]["ack_binding"]["ack_schema"],
            "sf_priv_quality_ack.v1",
        )
        # evidence may never be re-attributed to another action
        with self.assertRaises(src.TransitionIdentityError):
            src.build_replay_transition(
                state=self._state(tensor_seq=10, carla_frame_id=500),
                next_state=None,
                completed_ticket=anchor_ticket,
                outcome=src.evaluate_completed_decision(
                    anchor_ticket,
                    spec,
                    quality_components=self._quality(
                        evidence=self._evidence(self._action(mode_id=6, q_e4=9000))
                    ),
                ),
                reward_spec=spec,
                normalization=norm,
                terminated=True,
                episode_end_reason="end",
            )
        # a censored transition has no quality and therefore no evidence hashes
        timeout = self._completed_ticket(
            terminal=rtc.TerminalClass.FEEDBACK_TIMEOUT,
            decision_seq=1, first_tensor_seq=10, first_frame_id=500,
        )
        censored = src.build_replay_transition(
            state=self._state(tensor_seq=10, carla_frame_id=500),
            next_state=self._state(tensor_seq=13, observed_ns=T0 + 500 * MS),
            completed_ticket=timeout,
            outcome=src.evaluate_completed_decision(timeout, spec),
            reward_spec=spec,
            normalization=norm,
        )
        self.assertIsNone(censored.quality_evidence)
        self.assertIsNone(censored.raw_quality_ack_sha256)
        self.assertIsNone(censored.detailed_evidence_sha256)
        self.assertIsNone(censored.scalar_reward)

        # the declared ACK literals still match the real protocol module.  It is
        # AST-parsed, never imported: that module resolves its own dependency
        # through an absolute rl_agent.* import rooted elsewhere, so importing
        # it here would need a sys.path mutation at import time.
        import ast

        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "splitfusion_quality_feedback_probe_v1"
            / "protocol.py"
        )
        self.assertTrue(protocol_path.is_file(), protocol_path)
        tree = ast.parse(protocol_path.read_text())
        literals: dict = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        literals[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
        self.assertEqual(
            literals["QUALITY_EVALUATED_ACK_SCHEMA"], src.QUALITY_ACK_SCHEMA
        )
        self.assertEqual(
            literals["QUALITY_EVALUATION_FAILED_ACK_SCHEMA"],
            src.QUALITY_ACK_FAILURE_SCHEMA,
        )
        self.assertEqual(
            literals["PROTOCOL_VERSION"], src.QUALITY_ACK_PROTOCOL_VERSION
        )
        self.assertEqual(literals["SOURCE"], src.QUALITY_ACK_SOURCE)
        # the anchor fields really are required identity fields
        for field in src.QUALITY_ACK_REQUIRED_ANCHOR_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, literals["IDENTITY_FIELDS"])
        # every ACK timing field really is a wall clock
        for field in literals["TIMING_FIELDS"]:
            with self.subTest(field=field):
                self.assertTrue(field.endswith("_wall_ns"), field)
        # tp and fn exist; no false-positive field does
        quality_fields = literals["QUALITY_FIELDS"]
        for field in ("vehicle_tp", "vehicle_fn", "person_tp", "person_fn"):
            self.assertIn(field, quality_fields)
        for field in quality_fields:
            with self.subTest(field=field):
                self.assertFalse(field.endswith("_fp"), field)
                self.assertNotIn("false_positive", field)
                self.assertNotIn("precision", field)
        # the 72-anchor bound is the one the protocol validates
        self.assertIn(
            "action_id outside catalog", protocol_path.read_text()
        )
        self.assertIn(
            f"< {src.QUALITY_ACK_ANCHOR_ACTION_COUNT}", protocol_path.read_text()
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
