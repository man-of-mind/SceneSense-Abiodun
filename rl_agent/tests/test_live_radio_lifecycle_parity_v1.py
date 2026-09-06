#!/usr/bin/env python3
"""Offline contract tests; these tests never import or start CARLA, OAI or Docker.

Regression cover for the full-288 fresh-radio lifecycle defect: run_one_cell()
gated _start_live_radio() on campaign_kind == "live_pilot_16", so the 288-cell
campaign started CARLA and the adapter with no gNB/UE, and the target-SNR runtime
failed connecting to the telnet actuator.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from abiodun.rl_agent import ue_288_campaign_supervisor as supervisor


def synthetic_cell() -> supervisor.Cell:
    return supervisor.Cell(
        cell_id="a00__favorable_stable",
        action_index=0,
        action_id=0,
        profile_id="split_noae_uint8_q0000",
        model_family="noae",
        network_profile_id="FAVORABLE_STABLE",
        trace_id="synthetic-trace-not-for-collection",
        seed=31,
    )


def synthetic_config(campaign_kind: str) -> dict:
    return {
        "campaign_kind": campaign_kind,
        "measurement_contract": {"expected_prepared_hz": 10.0},
        "cell": {"expected_outputs": []},
    }


class _Recorder:
    """Stand-in for the qualified radio/CARLA lifecycle, started by nothing real."""

    def __init__(self, attach_fails: bool = False) -> None:
        self.attach_fails = attach_fails
        self.radio_starts = 0
        self.radio_stops = 0
        self.carla_starts = 0
        self.adapter_runs = 0
        self.order: list[str] = []


class LiveRadioLifecycleParityTest(unittest.TestCase):
    def _run_cell(self, campaign_kind: str, attach_fails: bool = False) -> _Recorder:
        recorder = _Recorder(attach_fails=attach_fails)
        cell = synthetic_cell()
        config = synthetic_config(campaign_kind)

        def fake_start_radio(config, cell, attempt, service_log_dir):
            recorder.order.append("radio_start")
            if recorder.attach_fails:
                raise supervisor.CampaignError("qualified gnb three-process topology drift")
            recorder.radio_starts += 1
            return Path(service_log_dir) / "ns", Path(service_log_dir) / "state", {"status": "ATTACHED"}

        def fake_stop_radio(config, namespace, radio_state, attached):
            recorder.order.append("radio_stop")
            recorder.radio_stops += 1
            return {"all_lifecycle_gates_passed": True}

        class FakeLifecycle:
            @staticmethod
            def start_carla(port, carla_log):
                recorder.order.append("carla_start")
                recorder.carla_starts += 1
                return object(), 4242

            @staticmethod
            def wait_for_rpc(port, timeout):
                return "synthetic"

            @staticmethod
            def stop_carla(server, pgid, port):
                recorder.order.append("carla_stop")
                return {"shutdown_verified": True}

            @staticmethod
            def child_env():
                return {}

        class FakeCompleted:
            returncode = 0

        def fake_subprocess_run(argv, **kwargs):
            recorder.order.append("adapter")
            recorder.adapter_runs += 1
            return FakeCompleted()

        patches = {
            "_start_live_radio": fake_start_radio,
            "_stop_live_radio": fake_stop_radio,
            "import_lifecycle_helper": lambda config: FakeLifecycle,
            "_stop_phase15_application": lambda config: {},
            "_require_phase15_application_cold": lambda config: {},
            "_compact_log_diagnostics": lambda directory, include_tails: [],
        }
        originals = {name: getattr(supervisor, name) for name in patches}
        original_run = supervisor.subprocess.run
        for name, value in patches.items():
            setattr(supervisor, name, value)
        supervisor.subprocess.run = fake_subprocess_run
        try:
            with tempfile.TemporaryDirectory() as scratch:
                supervisor.run_one_cell(
                    config=config,
                    cell=cell,
                    adapter=Path(scratch) / "adapter.py",
                    campaign_root=Path(scratch) / "campaign",
                    ledger_rows=[],
                    port=2000,
                )
        finally:
            for name, value in originals.items():
                setattr(supervisor, name, value)
            supervisor.subprocess.run = original_run
        return recorder

    def test_live_pilot_16_starts_and_stops_the_radio_exactly_once(self) -> None:
        recorder = self._run_cell("live_pilot_16")
        self.assertEqual(recorder.radio_starts, 1)
        self.assertEqual(recorder.radio_stops, 1)

    def test_full_288_starts_and_stops_the_radio_exactly_once(self) -> None:
        recorder = self._run_cell("full_288")
        self.assertEqual(recorder.radio_starts, 1)
        self.assertEqual(recorder.radio_stops, 1)

    def test_full_288_orders_radio_before_carla_and_adapter(self) -> None:
        recorder = self._run_cell("full_288")
        self.assertEqual(
            recorder.order,
            ["radio_start", "carla_start", "adapter", "carla_stop", "radio_stop"],
        )

    def test_offline_kind_never_touches_the_radio(self) -> None:
        recorder = self._run_cell("offline_replay")
        self.assertEqual(recorder.radio_starts, 0)
        self.assertEqual(recorder.radio_stops, 0)

    def test_carla_and_adapter_cannot_begin_when_radio_attachment_fails(self) -> None:
        recorder = self._run_cell("full_288", attach_fails=True)
        self.assertEqual(recorder.carla_starts, 0)
        self.assertEqual(recorder.adapter_runs, 0)
        self.assertEqual(recorder.radio_stops, 0)

    def test_registered_live_kinds_are_exactly_the_two_hardware_campaigns(self) -> None:
        self.assertEqual(supervisor.LIVE_CAMPAIGN_KINDS, ("live_pilot_16", "full_288"))
        self.assertTrue(supervisor.is_live_campaign({"campaign_kind": "live_pilot_16"}))
        self.assertTrue(supervisor.is_live_campaign({"campaign_kind": "full_288"}))
        self.assertFalse(supervisor.is_live_campaign({"campaign_kind": "offline_replay"}))
        self.assertFalse(supervisor.is_live_campaign({}))


if __name__ == "__main__":
    unittest.main()
