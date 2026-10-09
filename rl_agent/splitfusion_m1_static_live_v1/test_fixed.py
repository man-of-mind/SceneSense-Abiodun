"""CPU-only focused checks for the direct fixed-action engine seam."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import CONTRACT
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as state_contract
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E

from .fixed import FixedLog, make_engine_class
from .source_verify import paths_sha256, verify_registered_sources


class NeverActor:
    def __init__(self): self.calls = 0
    def act(self, _):
        self.calls += 1
        raise AssertionError("actor must not be called")


class FixedEngineTest(unittest.TestCase):
    def build(self):
        temporary = tempfile.TemporaryDirectory()
        log = FixedLog(Path(temporary.name) / "rows.jsonl")
        actor = NeverActor()
        cls = make_engine_class(E.Run4DecisionEngineV2, log)
        engine = cls(contract_=CONTRACT, actor=actor, ue_id="ue-test",
            controller_lineage_sha256=hashlib.sha256(b"m1-static").hexdigest(),
            session_factory=lambda: "11111111-1111-4111-8111-111111111111")
        self.addCleanup(log.close); self.addCleanup(temporary.cleanup)
        return engine, actor, log

    def test_typed_admission_refusal_records_and_executes_a0(self):
        engine, actor, log = self.build()
        record = {"action_open": {"domain": "CLOCK_MONOTONIC_RAW", "ns": 100}}
        def builder(_identity, _previous):
            record.update(radio_reasons=["RLC_BACKLOG_STALE"])
            raise state_contract.ExternalFallbackRequired("RLC_BACKLOG_STALE")
        plan = engine.plan_frame(frame_id=1, capture_wall_ns=10, now_raw_ns=90,
                                 build_state=builder)
        self.assertEqual((plan.kind, plan.profile.mode_id, plan.profile.q_e4,
                          plan.profile.action_id), (E.FrameKind.POLICY_DECISION, 0, 0, 0))
        self.assertEqual(actor.calls, 0)
        self.assertEqual(engine.counters.fallbacks, 0)
        self.assertEqual(log.counters["observational_admission_refused"], 1)
        self.assertEqual(engine.opportunities[0]["observational_reasons"], ["RLC_BACKLOG_STALE"])

    def test_valid_observation_does_not_call_actor_and_hold_stays_a0(self):
        engine, actor, _log = self.build()
        guarded = SimpleNamespace(boundary=SimpleNamespace(action_open_timestamp_ns=100))
        decision = engine.plan_frame(frame_id=1, capture_wall_ns=10, now_raw_ns=90,
            build_state=lambda *_: (guarded, object()))
        hold = engine.plan_frame(frame_id=2, capture_wall_ns=20, now_raw_ns=110,
            build_state=lambda *_: (_ for _ in ()).throw(AssertionError("hold built state")))
        self.assertEqual((decision.kind, hold.kind),
                         (E.FrameKind.POLICY_DECISION, E.FrameKind.POLICY_HOLD))
        self.assertEqual((hold.profile.mode_id, hold.profile.q_e4, hold.profile.action_id), (0, 0, 0))
        self.assertEqual(actor.calls, 0)
        self.assertEqual(engine.counters.actor_calls, 0)
        self.assertEqual(engine.counters.fallbacks, 0)

    def test_untyped_observation_fault_is_not_bypassed(self):
        engine, _actor, _log = self.build()
        with self.assertRaisesRegex(RuntimeError, "telemetry corrupt"):
            engine.plan_frame(frame_id=1, capture_wall_ns=10, now_raw_ns=90,
                build_state=lambda *_: (_ for _ in ()).throw(RuntimeError("telemetry corrupt")))


class ImportSafeSourceVerifierTest(unittest.TestCase):
    def test_valid_inventory_passes_and_mismatch_is_rejected(self):
        root = Path(__file__).resolve().parents[2]
        from . import contract as C
        sources = C.exact_sources(root)
        result = verify_registered_sources(sources, root)
        self.assertEqual(result["checked"], len(sources))
        self.assertEqual(result["paths_sha256"], paths_sha256(sources))
        mismatch = dict(sources)
        mismatch["rl_agent/splitfusion_live_parity_repair_v1/source_geometry.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source drift"):
            verify_registered_sources(mismatch, root)


if __name__ == "__main__":
    unittest.main()
