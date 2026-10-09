from __future__ import annotations

import unittest

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as SNR

from . import b_production_dependencies_v1 as D


class ProductionDependenciesTest(unittest.TestCase):
    def test_run5b_reader_uses_frozen_lease_and_causal_cutoff(self):
        adapter = SNR.ModeledLeaseSnrAdapterV1(
            provider_id="test", session_uuid="00000000-0000-4000-8000-000000000001",
            ue_id="ue")
        adapter.record_command_ack_at(at_ns=80, command_id="c1", status="ACK",
                                      clamped=False, target_snr_db=12.5)
        adapter.record_heartbeat_at(at_ns=90, active_command_id="c1")
        reader = D.CausalSnrLeaseReaderV1(adapter)
        self.assertEqual(reader.observe_db(
            cutoff_raw_ns=100,
            session_uuid="00000000-0000-4000-8000-000000000001",
            decision_seq=0), 12.5)

    def test_future_command_is_not_visible(self):
        adapter = SNR.ModeledLeaseSnrAdapterV1(
            provider_id="test", session_uuid="00000000-0000-4000-8000-000000000001",
            ue_id="ue")
        adapter.record_command_ack_at(at_ns=101, command_id="future", status="ACK",
                                      clamped=False, target_snr_db=20.0)
        adapter.record_heartbeat_at(at_ns=102, active_command_id="future")
        reader = D.CausalSnrLeaseReaderV1(adapter)
        with self.assertRaises(Exception):
            reader.observe_db(
                cutoff_raw_ns=100,
                session_uuid="00000000-0000-4000-8000-000000000001",
                decision_seq=0)

    def test_import_does_not_initialize_cuda(self):
        import torch
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__": unittest.main()
