import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_exact_gt_live_v2.ue import EarlyGtUeHandoff

from .abort_evidence import ARTIFACT, install_abort_evidence


class Action:
    def to_canonical_dict(self): return {"mode_id": 3}


@dataclass
class Frame:
    frame_id: int = 101
    action: object = Action()


class Resolution:
    def to_canonical_dict(self): return {"frame_id": 99, "resolved_by": "FEEDBACK"}


class Reader:
    def __init__(self):
        self.event, self.counters, self.stops = "NRUE_MAC_DCI_GRANT", {"rows": 3}, 0
    def stop(self): self.stops += 1


class Audit:
    def __init__(self): self.stops = 0
    def stop(self): self.stops += 1


class Controller:
    def __init__(self):
        self.session_uuid = "session-1"
        self.current = NS(resolution=None)
        self.ledger = NS(feedback=[{"frame_id": 99}], frames=[Frame()])
    def resolutions(self): return [Resolution()]


@dataclass
class Counters:
    policy_decisions: int = 3


class Engine:
    def __init__(self):
        self.controllers, self.counters, self.faulted = [Controller()], Counters(), "service failed"
        self.opportunities, self.fallback_log = [{"frame_id": 99}], []
    def coverage(self): return {"policy_decisions": 3, "resolved": 1}


class GtLog:
    def snapshot(self): return {"tickets": []}
    def missing_high_outputs(self): return []


class FakeRuntime:
    def __init__(self):
        self.failure = BrokenPipeError("exact-GT socket closed")
        self.close_calls = 0
        self.engine, self.pipeline = Engine(), NS(decisions=[{"frame_id": 99}, {"frame_id": 101}])
        self._run4_identity = {99: {"ticket_seq": 1}, 101: {"ticket_seq": 2}}
        self._feedback_rows, self._terminal_rows = [{"frame_id": 99}], []
        self.cycle_budget = NS(frame_budget=3900, decision_cap=1950)
        self.cycle_boundary_reached, self.prewarm_report = False, {"completed": True}
        self.gt_log, self.gt_queue = GtLog(), None
        self.provider = NS(counters={"snapshots": 4}, bridge=NS(counters={"published": 4}))
        self.readers, self.audit = [Reader()], Audit()
    def _run4_object_gt_evidence(self):
        return {"gt_reward_gate": None, "gt_last_datagram": {}, "gt_builder_overlap": {}}
    def close(self):
        self.close_calls += 1
        raise self.failure


class AbortEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.old_factory = U.build_run4_runtime_class

    def tearDown(self):
        U.build_run4_runtime_class = self.old_factory

    def test_service_failure_exports_partial_once_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            U.build_run4_runtime_class = lambda bindings: FakeRuntime
            bindings = NS(evidence_dir=root / "evidence")
            attempt = root / "attempt"
            attempt.mkdir()
            handoff = EarlyGtUeHandoff(attempt_dir=attempt,
                registration={"exact_gt_limits": {"ue_freeze_queue": 1}}, service_config={})
            with install_abort_evidence() as state:
                runtime = U.build_run4_runtime_class(bindings)()
                runtime._numeric_handoff = handoff
                with self.assertRaises(BrokenPipeError) as first:
                    runtime.close()
                self.assertIs(first.exception, runtime.failure)
                with self.assertRaises(BrokenPipeError) as second:
                    runtime.close()
                self.assertIs(second.exception, runtime.failure)
                with self.assertRaises(BrokenPipeError) as third:
                    handoff.close()
                self.assertIs(third.exception, runtime.failure)
                with self.assertRaises(BrokenPipeError):
                    handoff.close()
                document = json.loads((bindings.evidence_dir / ARTIFACT).read_bytes())
                self.assertEqual(document["artifact_status"], "PARTIAL_ABORTED")
                self.assertTrue(document["partial"] and document["aborted"])
                self.assertEqual(document["unresolved_tickets_at_close"], 1)
                self.assertEqual(document["resolutions"], [{"frame_id": 99, "resolved_by": "FEEDBACK"}])
                self.assertFalse(document["export_semantics"]["controller_poll_performed"])
                self.assertEqual(runtime.close_calls, 1)
                self.assertEqual(runtime.readers[0].stops, 1)
                self.assertEqual(runtime.audit.stops, 1)
                self.assertEqual(state["snapshot_attempts"], state["snapshot_writes"])
                self.assertEqual(state["snapshot_writes"], state["handoff_aborts"])
                self.assertTrue((handoff.directory / "UE_ABORT.json").is_file())

    def test_snapshot_write_failure_never_replaces_original(self):
        with tempfile.TemporaryDirectory() as scratch:
            U.build_run4_runtime_class = lambda bindings: FakeRuntime
            bindings = NS(evidence_dir=Path(scratch) / "evidence")
            def fail_write(path, document):
                raise OSError("injected snapshot write failure")
            with install_abort_evidence(snapshot_writer=fail_write) as state:
                runtime = U.build_run4_runtime_class(bindings)()
                with self.assertRaises(BrokenPipeError) as caught:
                    runtime.close()
                self.assertIs(caught.exception, runtime.failure)
                self.assertEqual(state["snapshot_attempts"], 1)
                self.assertEqual(state["snapshot_writes"], 0)
                self.assertEqual(state["snapshot_write_errors"], ["OSError: injected snapshot write failure"])
                self.assertEqual(runtime.readers[0].stops, 1)


if __name__ == "__main__":
    unittest.main()
