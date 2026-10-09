"""CPU/local-IPC composition tests; never start CARLA, OAI, CUDA or radio."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as FA
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2 as LS
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import (
    Pipeline, _fresh_radio, _scene, MS,
)
from . import actor_binding as A
from . import runtime_policy as P
from . import snr_live as S
from .test_actor_binding import state_fixture


CTRL = "11111111-1111-4111-8111-111111111111"


class Actor:
    boundary_sha256 = A.ACTOR_TREE_SHA256
    calls = 0

    def act(self, vector):
        if type(vector) is not A.S5.Run5FeatureVectorV2:
            raise AssertionError("non-Run5 feature vector")
        if len(vector.as_tuple()) != 22:
            raise AssertionError("not 22 features")
        self.calls += 1
        return SimpleNamespace(mode_id=11, q_e4=4321)


class PolicyRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.actor = Actor()
        self.label = state_fixture()[0]["identity"].ue_id
        self.runtime = SimpleNamespace(actor=self.actor,
            provider=SimpleNamespace(ue_label=self.label), engine=SimpleNamespace(ue_id=self.label))
        self.factory = mock.Mock(side_effect=lambda **kwargs: self.runtime)
        self.originals = (FA.load_registered_actor, LS.build_live_state,
                          pinned.LivePilotCellRuntime, pinned.start_target_snr)
        patcher = mock.patch.object(pinned, "LivePilotCellRuntime", self.factory)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(A, "load_source_export", return_value=self.actor)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.handle = P.install_policy_seams(attempt_dir=self.root, export_dir=self.root,
                                            controller_id=CTRL)
        self.addCleanup(self.handle.close)

    def bind(self):
        return pinned.LivePilotCellRuntime(campaign={})

    def inject(self, boundary, *, offset=10_000_000, target=15., seq=1):
        beat = boundary.state_commit_timestamp_ns - offset
        value = S.SnrSnapshot(S.SCHEMA, CTRL, seq, S.CLOCK_DOMAIN,
                              f"ack-{seq}", beat - 1, beat, target, True, None)
        self.assertTrue(self.handle.provider.accept_packet(value.encode(), beat + 1))

    def test_receiver_precedes_controller_and_actor_loader_is_run5(self):
        self.assertTrue(self.handle.provider.path.is_socket())
        self.assertIs(FA.load_registered_actor(), self.actor)
        record = json.loads((self.root / "run5_policy/POLICY_INSTALL.json").read_text())
        self.assertEqual(record["state_dim"], 22)
        self.assertEqual(record["socket_uid"], record["process_uid"])
        self.assertEqual(self.handle.provider.path.stat().st_mode & 0o777, 0o600)
        self.assertFalse(self.handle._controller_started)

    def test_unbound_runtime_refuses_not_zero_fill(self):
        kwargs, _ = state_fixture()
        with self.assertRaises(P.PolicySeamError):
            LS.build_live_state(**kwargs)

    def test_runtime_binds_actual_ue_only_once(self):
        self.assertIs(self.bind(), self.runtime)
        self.assertEqual(self.handle.provider.ue_id, self.label)
        with self.assertRaises(P.PolicySeamError):
            self.bind()

    def test_runtime_actor_drift_fails_closed(self):
        self.runtime.actor = SimpleNamespace(boundary_sha256="a" * 64)
        with self.assertRaises(P.PolicySeamError):
            self.bind()
        self.assertFalse(self.handle._runtime_bound)

    def test_runtime_identity_disagreement_fails_closed(self):
        self.runtime.engine.ue_id = "foreign"
        with self.assertRaises(P.PolicySeamError):
            self.bind()

    def test_causal_22d_state_and_frame_scoped_audit(self):
        self.bind()
        kwargs, _ = state_fixture()
        self.inject(kwargs["boundary"])
        _, expected = A._RUN4_BUILD_LIVE_STATE(**kwargs)
        guard, vector = LS.build_live_state(**kwargs)
        self.assertEqual(vector.as_tuple()[:21], expected.as_tuple())
        self.assertEqual(vector.as_tuple()[21], .5)
        self.assertEqual(guard.boundary, kwargs["boundary"])
        row = self.handle.evidence()["decision_snr"][0]
        self.assertTrue(row["admitted"])
        self.assertEqual(row["frame_id"], kwargs["camera_si"].metadata.identity.sample_seq)
        self.assertEqual(row["decision_identity"]["session_uuid"], kwargs["identity"].session_uuid)
        self.assertLess(row["snr"]["heartbeat_ns"], row["boundary"]["state_commit_timestamp_ns"])

    def test_missing_and_stale_snr_use_registered_fallback_exception(self):
        self.bind()
        kwargs, _ = state_fixture()
        with self.assertRaises(A.R4.ExternalFallbackRequired):
            LS.build_live_state(**kwargs)
        self.inject(kwargs["boundary"], offset=200_000_001)
        with self.assertRaises(A.R4.ExternalFallbackRequired):
            LS.build_live_state(**kwargs)
        self.assertFalse(any(row["admitted"] for row in self.handle.evidence()["decision_snr"]))

    def test_future_sample_does_not_enter_state(self):
        self.bind()
        kwargs, _ = state_fixture()
        self.inject(kwargs["boundary"], offset=-1)
        with self.assertRaises(A.R4.ExternalFallbackRequired):
            LS.build_live_state(**kwargs)

    def test_controller_reuses_legacy_paths_stop_and_campaign(self):
        campaign = {"runtime": {"target_snr_runtime":
            "rl_agent/splitfusion_live_dispatch_v1/live_pilot_target_snr_runtime.py"}}
        process = object()
        with mock.patch.object(P.subprocess, "Popen", return_value=process) as launch:
            actual, output, stop = pinned.start_target_snr(campaign, campaign_path=self.root / "c.yaml",
                profile_id="FAVORABLE_STABLE", temporary_dir=self.root, start_file=self.root / "start")
        self.assertIs(actual, process)
        self.assertEqual(output, self.root / "radio_trace.csv")
        self.assertEqual(stop, self.root / "stop_target_snr")
        argv = launch.call_args.args[0]
        self.assertEqual(argv[2], P.CONTROLLER_MODULE)
        self.assertIn("--controller", argv)
        self.assertEqual(argv[argv.index("--snr-controller-id") + 1], CTRL)
        self.assertEqual(argv[argv.index("--snr-socket") + 1], str(self.handle.provider.path))
        self.assertEqual(argv[argv.index("--profile-id") + 1], "FAVORABLE_STABLE")
        self.assertNotIn("sudo", argv)
        self.assertEqual(launch.call_args.kwargs["cwd"], str(pinned.ROOT))
        with self.assertRaises(P.PolicySeamError):
            pinned.start_target_snr(campaign, campaign_path=self.root / "c.yaml", profile_id="x",
                                   temporary_dir=self.root, start_file=self.root / "start")

    def test_foreign_controller_source_refused(self):
        campaign = {"runtime": {"target_snr_runtime": "rl_agent/__init__.py"}}
        with mock.patch.object(P.subprocess, "Popen") as launch:
            with self.assertRaises(P.PolicySeamError):
                pinned.start_target_snr(campaign, campaign_path=self.root / "c.yaml", profile_id="x",
                                       temporary_dir=self.root, start_file=self.root / "start")
        launch.assert_not_called()

    def test_close_retains_evidence_and_removes_only_own_socket(self):
        directory = self.handle.socket_dir
        self.bind()
        kwargs, _ = state_fixture()
        self.inject(kwargs["boundary"])
        LS.build_live_state(**kwargs)
        self.handle.close()
        self.assertFalse(directory.exists())
        self.assertIs(FA.load_registered_actor, self.originals[0])
        self.assertIs(LS.build_live_state, self.originals[1])
        self.assertIs(pinned.start_target_snr, self.originals[3])
        saved = json.loads((self.root / "run5_policy/SNR_EVIDENCE.json").read_text())
        self.assertEqual(saved["state_dim"], 22)
        self.assertEqual(len(saved["decision_snr"]), 1)
        self.assertEqual(saved["provider"]["controller_uuid"], CTRL)
        self.handle.close()

    def test_profiler_wrapper_installed_afterwards_sees_22_features(self):
        self.bind()
        kwargs, _ = state_fixture()
        self.inject(kwargs["boundary"])
        wrapper = mock.Mock(wraps=LS.build_live_state)
        with mock.patch.object(LS, "build_live_state", wrapper):
            _, vector = LS.build_live_state(**kwargs)
        self.assertEqual(len(vector.as_tuple()), 22)
        wrapper.assert_called_once()

    def test_real_pipeline_holds_do_not_query_snr_or_reinvoke_actor(self):
        pipe = Pipeline(actor=self.actor)
        self.runtime.provider = pipe.h.provider
        self.runtime.engine = pipe.engine
        self.bind()
        pipe.h.host.advance(100 * MS)
        _fresh_radio(pipe.h)
        now = pipe.h.host.raw()
        value = S.SnrSnapshot(S.SCHEMA, CTRL, 1, S.CLOCK_DOMAIN, "ack-1", now - 2,
                              now - 1, 15., True, None)
        self.handle.provider.accept_packet(value.encode(), now)
        first = pipe.pipe.plan(frame_id=1001, capture_wall_ns=1_790_000_000_000_000_000,
                               scene=_scene(pipe.h))
        self.assertEqual(first.plan.kind, E.FrameKind.POLICY_DECISION)
        self.assertEqual(self.actor.calls, 1)
        self.assertEqual(len(pipe.pipe.decisions[-1]["features"]), 22)
        pipe.h.host.advance(100 * MS)
        held = pipe.pipe.plan(frame_id=1002, capture_wall_ns=1_790_000_000_100_000_000,
                              scene=_scene(pipe.h))
        self.assertEqual(held.plan.kind, E.FrameKind.POLICY_HOLD)
        self.assertEqual(self.actor.calls, 1)
        self.assertEqual(len(self.handle.evidence()["decision_snr"]), 1)
        self.assertEqual(first.plan.profile, held.plan.profile)


if __name__ == "__main__":
    unittest.main()
