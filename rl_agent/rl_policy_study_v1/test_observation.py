from __future__ import annotations

import unittest

import torch

from rl_agent.rl_policy_study_v1.delayed_feedback import (
    PendingTicket,
    PendingTransitionLedger,
    QualitySource,
    TerminalFeedback,
    TerminalOutcome,
    TicketKey,
)
from rl_agent.rl_policy_study_v1.model import SplitOnlyRecurrentActorCritic
from rl_agent.rl_policy_study_v1.observation import (
    OBSERVATION_FEATURES_V1,
    CurrentSignalsV1,
    ObservationLimitsV1,
    assemble_observation_v1,
)


def _ticket() -> PendingTicket:
    return PendingTicket(
        key=TicketKey("session", "ue-1", 1, 71),
        decision_epoch=1,
        opened_at_ms=130.0,
        service_started_at_ms=100.0,
        previous_action_id=70,
        segmentation_quality_anchor=0.7,
        localization_quality_anchor=0.8,
        quality_source=QualitySource.FROZEN_ACTION_VALIDATION,
        quality_version="quality-v1",
        quality_catalog_sha256="a" * 64,
        payload_bytes=6464,
        compute_ms_charged=21.0,
        old_log_probability=-1.0,
        old_value=0.0,
        incoming_lstm_h=(0.0,),
        incoming_lstm_c=(0.0,),
        deadline_ms=140.0,
    )


def _feedback(item: PendingTicket) -> TerminalFeedback:
    return TerminalFeedback(
        key=item.key,
        outcome=TerminalOutcome.TAIL_COMPLETED_ACK_RECEIVED,
        observed_at_ms=235.0,
        payload_bytes_charged=6400,
        compute_ms_charged=20.5,
        feedback_latency_ms=135.0,
        deadline_ms=140.0,
        deadline_met=True,
        segmentation_quality=0.7,
        localization_quality=0.8,
        quality_source=QualitySource.FROZEN_ACTION_VALIDATION,
        quality_version="quality-v1",
        quality_catalog_sha256="a" * 64,
        quality_available=True,
    )


class ObservationV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = ObservationLimitsV1(
            deadline_ms=140.0,
            snr_min_db=-10.0,
            snr_max_db=40.0,
            maximum_mcs=28,
            maximum_bsr_bytes=10_000_000,
            maximum_uplink_mbps=1000.0,
            maximum_payload_bytes=1_300_000,
            maximum_frame_lag=16,
            maximum_pending_count=8,
            maximum_feedback_age_ms=1000.0,
        )
        self.signals = CurrentSignalsV1(
            sensor_elapsed_ms=30.0,
            snr_db=20.0,
            mcs=20,
            bsr_bytes=4096,
            uplink_throughput_mbps=75.0,
            previous_action_id=70,
            previous_payload_bytes=7000,
        )

    def _vector(
        self,
        ledger: PendingTransitionLedger,
        *,
        cutoff_ms: float,
    ) -> tuple[float, ...]:
        state = ledger.observation(
            session_id="session",
            ue_id="ue-1",
            current_frame_id=2,
            now_ms=cutoff_ms,
        )
        return assemble_observation_v1(self.signals, state, self.limits)

    def test_schema_is_fixed_finite_and_bounded(self) -> None:
        values = self._vector(PendingTransitionLedger(), cutoff_ms=200.0)
        self.assertEqual(len(values), len(OBSERVATION_FEATURES_V1))
        self.assertTrue(all(torch.isfinite(torch.tensor(values))))

    def test_feedback_enters_only_decisions_after_its_receipt_cutoff(self) -> None:
        ledger = PendingTransitionLedger()
        item = _ticket()
        ledger.open(item)
        before_close = self._vector(ledger, cutoff_ms=234.0)
        ledger.close(_feedback(item))

        # Historical reconstruction stays identical even though the ledger is
        # now closed in present time: the ACK had not arrived by this cutoff.
        reconstructed_before = self._vector(ledger, cutoff_ms=234.0)
        after = self._vector(ledger, cutoff_ms=235.0)
        self.assertEqual(before_close, reconstructed_before)
        self.assertNotEqual(reconstructed_before, after)

        torch.manual_seed(19)
        model = SplitOnlyRecurrentActorCritic(
            observation_dim=len(OBSERVATION_FEATURES_V1),
            encoder_dim=24,
            hidden_dim=16,
        )
        mask = torch.ones(1, 72, dtype=torch.bool)
        before_output = model(
            torch.tensor([reconstructed_before], dtype=torch.float32),
            action_mask=mask,
        )
        after_output = model(
            torch.tensor([after], dtype=torch.float32),
            action_mask=mask,
        )
        self.assertFalse(torch.equal(before_output.logits, after_output.logits))


if __name__ == "__main__":
    unittest.main()
