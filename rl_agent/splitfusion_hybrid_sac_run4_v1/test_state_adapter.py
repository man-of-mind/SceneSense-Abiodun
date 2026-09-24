"""Adversarial tests for the pure Run-4 UE state adapter."""

from __future__ import annotations

import builtins
import importlib
import random
import socket
import subprocess
import unittest
from dataclasses import replace
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import state_adapter as src


SESSION = "11111111-1111-4111-8111-111111111111"
FOREIGN_SESSION = "22222222-2222-4222-8222-222222222222"
UE_ID = "ue-1"
FOREIGN_UE = "ue-2"
CLOCK = "RUN4_TEST_MONOTONIC"


class StateAdapterTest(unittest.TestCase):
    def boundary(self, *, commit_ns: int = 1_000) -> contract.DecisionBoundaryV1:
        return contract.DecisionBoundaryV1(
            identity=contract.DecisionIdentityV1(SESSION, UE_ID, 7),
            state_commit_timestamp_ns=commit_ns,
            action_open_timestamp_ns=commit_ns + 10,
            clock_domain=CLOCK,
        )

    def sample_identity(
        self,
        sequence: int,
        *,
        session: str = SESSION,
        ue_id: str = UE_ID,
    ) -> contract.SampleIdentityV1:
        return contract.SampleIdentityV1(session, ue_id, sequence)

    def grant(
        self,
        sequence: int = 1,
        *,
        source_ns: int = 800,
        available_ns: int = 850,
        mcs: int = 12,
        table: int = contract.UL_MCS_TABLE_ID,
        round_: int = 0,
        ndi: int = 1,
        direction: contract.LinkDirection = contract.LinkDirection.UPLINK,
        policy: str = contract.UL_MCS_POLICY_ID,
        session: str = SESSION,
        ue_id: str = UE_ID,
        grant_identity: str | None = None,
        clock: str = CLOCK,
    ) -> src.RawUeUlDciGrantCandidateV1:
        return src.RawUeUlDciGrantCandidateV1(
            identity=self.sample_identity(sequence, session=session, ue_id=ue_id),
            grant_identity=(
                f"grant-{sequence}"
                if grant_identity is None
                else grant_identity
            ),
            link_direction=direction,
            mcs_table=table,
            mcs_index=mcs,
            harq_round=round_,
            new_data_indicator=ndi,
            scheduler_policy_id=policy,
            source="unit-test:ue-ul-dci",
            source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=clock,
        )

    def backlog(
        self,
        sequence: int = 1,
        *,
        source_ns: int = 800,
        available_ns: int = 850,
        value: int = 1234,
        direction: contract.LinkDirection = contract.LinkDirection.UPLINK,
        session: str = SESSION,
        ue_id: str = UE_ID,
        clock: str = CLOCK,
    ) -> src.RawUeRlcBacklogSampleV1:
        return src.RawUeRlcBacklogSampleV1(
            identity=self.sample_identity(sequence, session=session, ue_id=ue_id),
            backlog_bytes=value,
            link_direction=direction,
            source="unit-test:ue-rlc-buffer-status",
            source_timestamp_ns=source_ns,
            available_timestamp_ns=available_ns,
            clock_domain=clock,
        )

    # UL-MCS selection -------------------------------------------------

    def test_unsorted_grants_select_latest_source_instant(self) -> None:
        selected = src.select_prior_new_data_ul_mcs(
            [
                self.grant(3, source_ns=700, available_ns=990, mcs=7),
                self.grant(1, source_ns=900, available_ns=920, mcs=21),
                self.grant(2, source_ns=800, available_ns=810, mcs=15),
            ],
            self.boundary(),
        )
        self.assertEqual(selected.observation.value, 21)
        self.assertEqual(selected.grant_identity, "grant-1")
        self.assertEqual(selected.observation.metadata.source_timestamp_ns, 900)
        self.assertEqual(selected.observation.metadata.available_timestamp_ns, 920)

    def test_mcs_zero_is_valid_and_distinct_from_missing(self) -> None:
        valid = src.select_prior_new_data_ul_mcs(
            [self.grant(mcs=0)], self.boundary()
        )
        missing = src.select_prior_new_data_ul_mcs([], self.boundary())
        self.assertTrue(valid.observation.metadata.valid)
        self.assertEqual(valid.observation.value, 0)
        self.assertIsNone(valid.observation.missing_reason)
        self.assertFalse(missing.observation.metadata.valid)
        self.assertIsNone(missing.observation.value)
        self.assertEqual(
            missing.observation.missing_reason,
            src.MISSING_PRIOR_UL_GRANT,
        )
        self.assertIsNone(missing.harq_round)
        self.assertIsNone(missing.grant_identity)

    def test_startup_has_explicit_missing_grant(self) -> None:
        result = src.select_prior_new_data_ul_mcs([], self.boundary())
        self.assertEqual(
            result.observation.metadata.kind,
            contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX,
        )
        self.assertEqual(result.observation.metadata.observer, contract.Observer.UE)
        self.assertEqual(
            result.observation.metadata.link_direction,
            contract.LinkDirection.UPLINK,
        )
        self.assertEqual(
            result.observation.metadata.source_timestamp_ns,
            self.boundary().state_commit_timestamp_ns,
        )

    def test_future_and_boundary_equal_grants_are_not_causal(self) -> None:
        result = src.select_prior_new_data_ul_mcs(
            [
                self.grant(1, source_ns=1_000, available_ns=1_000),
                self.grant(2, source_ns=1_001, available_ns=1_001),
                self.grant(3, source_ns=900, available_ns=1_000),
            ],
            self.boundary(),
        )
        self.assertFalse(result.observation.metadata.valid)

    def test_only_registered_new_data_table0_ul_grant_is_selected(self) -> None:
        candidates = [
            self.grant(1, source_ns=910, available_ns=920, round_=1, mcs=28),
            self.grant(2, source_ns=900, available_ns=910, table=1, mcs=27),
            self.grant(
                3,
                source_ns=890,
                available_ns=900,
                direction=contract.LinkDirection.DOWNLINK,
                mcs=26,
            ),
            self.grant(
                4,
                source_ns=880,
                available_ns=890,
                policy="foreign-scheduler",
                mcs=25,
            ),
            self.grant(5, source_ns=870, available_ns=880, mcs=17, ndi=0),
        ]
        result = src.select_prior_new_data_ul_mcs(candidates, self.boundary())
        self.assertEqual(result.observation.value, 17)
        self.assertEqual(result.new_data_indicator, 0)
        self.assertEqual(result.harq_round, 0)
        self.assertEqual(result.mcs_table, 0)
        self.assertEqual(result.scheduler_policy_id, contract.UL_MCS_POLICY_ID)

    def test_foreign_session_and_ue_cannot_supply_grant(self) -> None:
        result = src.select_prior_new_data_ul_mcs(
            [
                self.grant(1, session=FOREIGN_SESSION, mcs=28),
                self.grant(2, ue_id=FOREIGN_UE, mcs=27),
            ],
            self.boundary(),
        )
        self.assertFalse(result.observation.metadata.valid)

    def test_wrong_clock_domain_cannot_supply_grant(self) -> None:
        result = src.select_prior_new_data_ul_mcs(
            [self.grant(clock="FOREIGN_CLOCK")], self.boundary()
        )
        self.assertFalse(result.observation.metadata.valid)

    def test_exact_duplicate_grant_is_deduplicated(self) -> None:
        grant = self.grant()
        result = src.select_prior_new_data_ul_mcs(
            [grant, grant], self.boundary()
        )
        self.assertEqual(result.observation.value, grant.mcs_index)

    def test_same_grant_identity_with_different_content_is_conflict(self) -> None:
        first = self.grant(1, grant_identity="durable-grant")
        second = replace(first, mcs_index=13)
        with self.assertRaises(src.EvidenceIdentityConflictError):
            src.select_prior_new_data_ul_mcs(
                [first, second], self.boundary()
            )

    def test_same_sample_identity_with_different_grant_identity_is_conflict(self) -> None:
        first = self.grant(1, grant_identity="grant-a")
        second = replace(first, grant_identity="grant-b")
        with self.assertRaises(src.EvidenceIdentityConflictError):
            src.select_prior_new_data_ul_mcs(
                [first, second], self.boundary()
            )

    def test_distinct_latest_grants_at_same_source_instant_are_ambiguous(self) -> None:
        with self.assertRaises(src.AmbiguousLatestEvidenceError):
            src.select_prior_new_data_ul_mcs(
                [
                    self.grant(1, source_ns=900, available_ns=910),
                    self.grant(2, source_ns=900, available_ns=920),
                ],
                self.boundary(),
            )

    def test_eligible_table0_mcs_out_of_range_fails_closed(self) -> None:
        with self.assertRaises(src.RawEvidenceError):
            src.select_prior_new_data_ul_mcs(
                [self.grant(mcs=29)], self.boundary()
            )

    # RLC backlog selection --------------------------------------------

    def test_unsorted_backlog_selects_latest_strictly_pre_boundary(self) -> None:
        selected = src.select_pre_action_rlc_backlog(
            [
                self.backlog(1, source_ns=700, available_ns=710, value=10),
                self.backlog(3, source_ns=900, available_ns=910, value=30),
                self.backlog(2, source_ns=800, available_ns=810, value=20),
            ],
            self.boundary(),
            payload_enqueue_timestamp_ns=1_100,
        )
        self.assertEqual(selected.value, 30)
        self.assertEqual(selected.metadata.identity.sample_seq, 3)

    def test_valid_zero_backlog_is_preserved(self) -> None:
        selected = src.select_pre_action_rlc_backlog(
            [self.backlog(value=0)],
            self.boundary(),
            payload_enqueue_timestamp_ns=1_100,
        )
        self.assertTrue(selected.metadata.valid)
        self.assertEqual(selected.value, 0)
        self.assertIsNone(selected.missing_reason)

    def test_startup_backlog_missing_is_not_zero(self) -> None:
        result = src.select_pre_action_rlc_backlog(
            [],
            self.boundary(),
            payload_enqueue_timestamp_ns=1_100,
        )
        self.assertFalse(result.metadata.valid)
        self.assertIsNone(result.value)
        self.assertEqual(
            result.missing_reason,
            src.MISSING_PRE_ACTION_RLC_BACKLOG,
        )

    def test_backlog_must_precede_both_enqueue_and_state_commit(self) -> None:
        # Enqueue is the earlier boundary in this case.  Equality with either
        # boundary is intentionally ineligible.
        result = src.select_pre_action_rlc_backlog(
            [
                self.backlog(1, source_ns=899, available_ns=899, value=1),
                self.backlog(2, source_ns=900, available_ns=900, value=2),
                self.backlog(3, source_ns=950, available_ns=950, value=3),
                self.backlog(4, source_ns=800, available_ns=900, value=4),
            ],
            self.boundary(commit_ns=1_000),
            payload_enqueue_timestamp_ns=900,
        )
        self.assertEqual(result.value, 1)

    def test_state_commit_is_cutoff_when_it_precedes_enqueue(self) -> None:
        result = src.select_pre_action_rlc_backlog(
            [
                self.backlog(1, source_ns=990, available_ns=999, value=1),
                self.backlog(2, source_ns=999, available_ns=1_000, value=2),
            ],
            self.boundary(commit_ns=1_000),
            payload_enqueue_timestamp_ns=1_100,
        )
        self.assertEqual(result.value, 1)

    def test_backlog_does_not_forward_fill_between_calls(self) -> None:
        first = src.select_pre_action_rlc_backlog(
            [self.backlog(value=55)],
            self.boundary(),
            payload_enqueue_timestamp_ns=1_100,
        )
        second = src.select_pre_action_rlc_backlog(
            [],
            self.boundary(),
            payload_enqueue_timestamp_ns=1_100,
        )
        self.assertEqual(first.value, 55)
        self.assertIsNone(second.value)
        self.assertFalse(second.metadata.valid)

    def test_foreign_or_downlink_backlog_is_not_selected(self) -> None:
        result = src.select_pre_action_rlc_backlog(
            [
                self.backlog(1, session=FOREIGN_SESSION),
                self.backlog(2, ue_id=FOREIGN_UE),
                self.backlog(
                    3, direction=contract.LinkDirection.DOWNLINK
                ),
                self.backlog(4, clock="FOREIGN_CLOCK"),
            ],
            self.boundary(),
            payload_enqueue_timestamp_ns=1_100,
        )
        self.assertFalse(result.metadata.valid)

    def test_exact_duplicate_backlog_is_deduplicated(self) -> None:
        sample = self.backlog()
        result = src.select_pre_action_rlc_backlog(
            [sample, sample],
            self.boundary(),
            payload_enqueue_timestamp_ns=1_100,
        )
        self.assertEqual(result.value, sample.backlog_bytes)

    def test_conflicting_backlog_sample_identity_fails_closed(self) -> None:
        first = self.backlog(1)
        second = replace(first, backlog_bytes=999)
        with self.assertRaises(src.EvidenceIdentityConflictError):
            src.select_pre_action_rlc_backlog(
                [first, second],
                self.boundary(),
                payload_enqueue_timestamp_ns=1_100,
            )

    def test_distinct_latest_backlog_samples_at_same_instant_are_ambiguous(self) -> None:
        with self.assertRaises(src.AmbiguousLatestEvidenceError):
            src.select_pre_action_rlc_backlog(
                [
                    self.backlog(1, source_ns=900, available_ns=910),
                    self.backlog(2, source_ns=900, available_ns=920),
                ],
                self.boundary(),
                payload_enqueue_timestamp_ns=1_100,
            )

    # Input integrity and import purity --------------------------------

    def test_raw_records_reject_bool_and_negative_numeric_fields(self) -> None:
        with self.assertRaises(src.RawEvidenceError):
            self.grant(mcs=True)  # type: ignore[arg-type]
        with self.assertRaises(src.RawEvidenceError):
            self.grant(round_=-1)
        with self.assertRaises(src.RawEvidenceError):
            self.backlog(value=-1)

    def test_source_cannot_follow_availability(self) -> None:
        with self.assertRaises(src.RawEvidenceError):
            self.grant(source_ns=900, available_ns=899)
        with self.assertRaises(src.RawEvidenceError):
            self.backlog(source_ns=900, available_ns=899)

    def test_import_has_no_rng_filesystem_socket_or_process_side_effect(self) -> None:
        with mock.patch.object(
            builtins, "open", side_effect=AssertionError("filesystem access")
        ) as opened, mock.patch.object(random, "seed") as seeded, mock.patch.object(
            random, "random"
        ) as sampled, mock.patch.object(
            socket, "socket", side_effect=AssertionError("socket opened")
        ) as socket_opened, mock.patch.object(
            subprocess, "Popen", side_effect=AssertionError("process launched")
        ) as popen:
            importlib.reload(src)
        opened.assert_not_called()
        seeded.assert_not_called()
        sampled.assert_not_called()
        socket_opened.assert_not_called()
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
