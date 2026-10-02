"""Tests for the edge evaluator-submit-before-map ordering seam.

Uses the actual Phase-6 composition with the profiler seams installed: real
SFD4 wires from the Phase-6 test pipeline, the real ``Run4EdgeProcessorV2``
(profiled subclass), the real map-update builder/validator/encoder and the real
``Run4EvaluatorV2`` with a fixed ground truth.  CPU only.
"""

from __future__ import annotations

import dataclasses
import json
import queue
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_direct_edge_map_v1 import protocol as DP
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as ER
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import (
    CONTRACT,
)
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import (
    SPEC,
    FakeActor,
    Pipeline,
)

from . import edge_ordering_v1 as OR
from . import edge_profiled_v1 as EP
from . import recorder_v1 as RV
from . import runner_v1 as RUN
from . import threadcap_v1 as TC

GT = {"semantic": np.zeros((4, 4), np.uint8),
      "objects": [{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0}]}
RECORDS = [{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0, "score": 0.9},
           {"class_name": "person", "world_x": 3.0, "world_y": 4.0, "score": 0.5}]


def fake_compute(**kwargs):
    # fresh objects per call, like the live closure (records list of dicts, numpy labels)
    return [dict(r) for r in RECORDS], np.zeros((4, 4), np.uint8)


class _Saved:
    def __init__(self, *targets):
        self._saved = [(m, n, getattr(m, n)) for m, n in targets]

    def restore(self):
        for m, n, v in self._saved:
            setattr(m, n, v)


class EarlySubmitTest(unittest.TestCase):
    """Each test builds the profiled classes, then the ordering seam on top."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = _Saved((ER, "Run4EdgeProcessorV2"), (ER, "Run4EvaluatorV2"),
                            (PW, "publish_ready_after_warmup"))
        OR.reset_for_tests()
        self.rec = RV.activate(RV.ProfileRecorderV1(Path(self.tmp.name) / "e.jsonl", role="edge",
                                                    flush_interval_s=0.01).start())
        self.rec.set_phase("live")

    def tearDown(self) -> None:
        self.rec.close()
        RV.deactivate()
        self.saved.restore()
        OR.reset_for_tests()
        self.tmp.cleanup()

    def _install(self, ordering: str) -> None:
        # the profiled classes the live edge uses, then the ordering seam on top
        class Profiled(ER.Run4EdgeProcessorV2):
            def process(self, frame, *, edge_timing, segmentation=None):
                with RV.active().span("edge.process"):
                    return super().process(frame, edge_timing=edge_timing, segmentation=segmentation)
        ER.Run4EdgeProcessorV2 = Profiled
        OR.install(edge_module=ER, prewarm_module=PW, map_protocol=MP, ordering=ordering,
                   counters_dir=Path(self.tmp.name), recorder_fn=RV.active)

    def _service(self, ordering: str, *, send=None, queue_depth: int = 64, start: bool = True,
                 gt=None, compute=None):
        self._install(ordering)
        sent: list = []
        evaluator = ER.Run4EvaluatorV2(spec=SPEC, send=send or sent.append, match_distance_m=3.0,
                                       gt_timeout_s=0.5, queue_depth=queue_depth,
                                       read_ground_truth=gt or (lambda **kw: GT))
        if start:
            evaluator.start()
        processor = ER.Run4EdgeProcessorV2(contract=CONTRACT, run_id="run", cell_id="cell",
                                           service_deadline_s=0.1, ack_timeout_s=0.5,
                                           compute=compute or fake_compute)
        OR.mark_live()
        return processor, evaluator, sent

    @staticmethod
    def _wire(reward: bool = True):
        p = Pipeline(FakeActor(mode_id=9, q_e4=4321))
        prepared = p.step()                 # decision: reward requested
        if not reward:
            prepared = p.step()             # the held tensor: no reward request
        return prepared

    # -- identity, content, exactly once ---------------------------------------
    def test_early_ticket_equals_serial_ticket_and_map_update_is_identical(self) -> None:
        wire = self._wire().wire
        serial_processor, serial_evaluator, _ = self._service(OR.SERIAL, start=False)
        serial = serial_processor.process(wire, edge_timing={"reassembly_complete_wall_s": 1.0})
        self.assertIsNotNone(serial.evaluation)
        self.saved.restore(); OR.reset_for_tests()
        captured = []
        early_processor, early_evaluator, _ = self._service(OR.EARLY, start=False)
        early_evaluator.submit = captured.append       # capture instead of queueing
        early = early_processor.process(wire, edge_timing={"reassembly_complete_wall_s": 1.0})
        self.assertIsNone(early.evaluation)             # the loop cannot submit again
        self.assertEqual(len(captured), 1)
        t_serial, t_early = serial.evaluation, captured[0]
        self.assertEqual(t_serial.envelope, t_early.envelope)
        self.assertEqual(t_serial.context, t_early.context)
        self.assertEqual(list(t_serial.records), list(t_early.records))
        self.assertTrue(np.array_equal(t_serial.predicted_mask, t_early.predicted_mask))
        self.assertEqual(dict(t_serial.gt_identity), dict(t_early.gt_identity))
        self.assertEqual(early.update, serial.update)   # map content byte-identical
        self.assertEqual(DP.encode(early.update), DP.encode(serial.update))
        self.assertEqual(OR.counters_snapshot()["counters"], {"reward_frames_seen": 1,
                                                              "early_submissions": 1,
                                                              "loop_submission_suppressed": 1})

    def test_hold_frames_and_pre_ready_frames_never_gain_evaluations(self) -> None:
        hold = self._wire(reward=False).wire
        processor, evaluator, _ = self._service(OR.EARLY, start=False)
        captured = []
        evaluator.submit = captured.append
        processed = processor.process(hold, edge_timing={})
        self.assertIsNone(processed.evaluation)
        self.assertEqual(captured, [])
        OR._STATE["live"] = False                        # before READY (prewarm window)
        reward = self._wire().wire
        processed = processor.process(reward, edge_timing={})
        self.assertEqual(captured, [])
        self.assertIsNotNone(processed.evaluation)       # base behaviour kept when not live
        self.assertEqual(OR.counters_snapshot()["counters"]["live_gate_rejected"], 1)

    # -- ownership / no mutation, both completion orders -------------------------
    def test_neither_branch_mutates_shared_inputs_in_either_completion_order(self) -> None:
        wire = self._wire().wire
        for delay_gt in (0.0, 0.15):                      # evaluator finishes before / after map
            self.saved.restore(); OR.reset_for_tests()
            def slow_gt(_d=delay_gt, **kw):
                time.sleep(_d)
                return GT
            processor, evaluator, sent = self._service(OR.EARLY, gt=slow_gt)
            tickets: list = []
            inner_submit = evaluator.submit
            evaluator.submit = lambda ticket: (tickets.append(ticket), inner_submit(ticket))
            processed = processor.process(wire, edge_timing={"reassembly_complete_wall_s": 1.0})
            before = json.dumps(processed.update, sort_keys=True, default=str)
            ticket = tickets[0]
            ticket_records_before = [dict(r) for r in ticket.records]
            mask_before = ticket.predicted_mask.copy()
            # map branch (as the loop does)
            MP.validate_run4_map_update(processed.update)
            payload = DP.encode(dict(processed.update))
            evaluator.close()
            self.assertEqual(len(sent), 1)
            feedback, reason = W.decode_feedback(sent[0])
            self.assertEqual(feedback.kind, "DELIVERED_SUCCESS")
            self.assertEqual(json.dumps(processed.update, sort_keys=True, default=str), before)
            self.assertEqual(DP.encode(dict(processed.update)), payload)
            # the ticket the evaluator consumed is unchanged, as are the map update's records
            self.assertEqual([dict(r) for r in ticket.records], ticket_records_before)
            self.assertTrue(np.array_equal(ticket.predicted_mask, mask_before))
            self.assertEqual(RECORDS, [{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0, "score": 0.9},
                                       {"class_name": "person", "world_x": 3.0, "world_y": 4.0, "score": 0.5}])
            self.assertEqual(evaluator.records[0]["frame_id"], processed.envelope.frame_id)
            self.assertEqual(evaluator.counters["emitted"], 1)
            self.assertEqual(evaluator.counters["submitted"], 1)

    def test_qperc_identical_between_orderings(self) -> None:
        wire = self._wire().wire
        outputs = []
        for ordering in (OR.SERIAL, OR.EARLY):
            self.saved.restore(); OR.reset_for_tests()
            processor, evaluator, sent = self._service(ordering)
            processed = processor.process(wire, edge_timing={"reassembly_complete_wall_s": 1.0})
            if processed.evaluation is not None:
                evaluator.submit(processed.evaluation)   # the loop's submission (serial)
            evaluator.close()
            feedback, reason = W.decode_feedback(sent[0])
            outputs.append((feedback.kind, feedback.q_perc, reason.name,
                            evaluator.records[0]["measurement"]))
        self.assertEqual(outputs[0], outputs[1])

    # -- overflow and send errors never block or stop the map path ---------------
    def test_queue_full_emits_infrastructure_fault_synchronously_and_map_continues(self) -> None:
        wire = self._wire().wire
        processor, evaluator, sent = self._service(OR.EARLY, queue_depth=1, start=False)
        evaluator._queue.put_nowait(object())            # pre-fill: the next put is Full
        processed = processor.process(wire, edge_timing={})
        self.assertIsNone(processed.evaluation)
        self.assertEqual(evaluator.counters["queue_overflow"], 1)
        self.assertEqual(len(sent), 1)                   # overflow feedback was emitted (as today)
        feedback, reason = W.decode_feedback(sent[0])
        self.assertEqual(feedback.kind, "INFRASTRUCTURE_FAULT")
        self.assertEqual(OR.counters_snapshot()["counters"]["early_submissions"], 1)
        MP.validate_run4_map_update(processed.update)   # map update intact

    def test_send_error_on_overflow_is_counted_and_map_update_still_built(self) -> None:
        wire = self._wire().wire

        def failing_send(payload):
            raise OSError("network unreachable")
        processor, evaluator, _ = self._service(OR.EARLY, queue_depth=1, start=False,
                                                send=failing_send)
        evaluator._queue.put_nowait(object())
        processed = processor.process(wire, edge_timing={})
        self.assertIsNone(processed.evaluation)
        counters = OR.counters_snapshot()
        self.assertEqual(counters["counters"]["early_submit_errors"], 1)
        self.assertEqual(len(counters["errors"]), 1)
        MP.validate_run4_map_update(processed.update)

    # -- map failures keep their accounting -----------------------------------------
    def test_map_build_failure_after_early_submit_still_raises_and_submits_once(self) -> None:
        wire = self._wire().wire
        processor, evaluator, sent = self._service(OR.EARLY)
        saved = MP.build_run4_map_update
        MP.build_run4_map_update = lambda **kw: (_ for _ in ()).throw(RuntimeError("map build broke"))
        try:
            with self.assertRaises(RuntimeError):         # the loop counts edge_processing_failed
                processor.process(wire, edge_timing={})
        finally:
            MP.build_run4_map_update = saved
        evaluator.close()
        self.assertEqual(evaluator.counters["submitted"], 1)   # evaluation happened once
        self.assertEqual(len(sent), 1)                         # quality feedback is not map proof
        self.assertEqual(OR.counters_snapshot()["counters"]["early_submissions"], 1)

    def test_counters_file_is_written_once_at_evaluator_close(self) -> None:
        processor, evaluator, _ = self._service(OR.EARLY)
        evaluator.close()
        path = Path(self.tmp.name) / OR.COUNTERS_NAME
        document = json.loads(path.read_text())
        self.assertEqual(document["ordering"], "early")
        evaluator.close()                                 # idempotent (no FileExistsError)

    def test_serial_install_leaves_the_processor_class_alone(self) -> None:
        evaluator_before = ER.Run4EvaluatorV2
        self._install(OR.SERIAL)
        self.assertEqual(ER.Run4EdgeProcessorV2.__name__, "Profiled")     # not wrapped
        self.assertTrue(issubclass(ER.Run4EvaluatorV2, evaluator_before))  # counters hook only
        self.assertIsNot(ER.Run4EvaluatorV2, evaluator_before)
        # a serial processor still returns the ticket for the loop to submit
        processor = ER.Run4EdgeProcessorV2(contract=CONTRACT, run_id="run", cell_id="cell",
                                           service_deadline_s=0.1, ack_timeout_s=0.5,
                                           compute=fake_compute)
        OR.mark_live()
        processed = processor.process(self._wire().wire, edge_timing={})
        self.assertIsNotNone(processed.evaluation)
        self.assertEqual(OR.counters_snapshot()["counters"], {})


class ArmWiringTest(unittest.TestCase):
    def test_arms_c2_and_e(self) -> None:
        self.assertEqual(TC.ORDERINGS["C2"], "serial")
        self.assertEqual(TC.ORDERINGS["E"], "early")
        self.assertEqual(TC.ARMS["E"], TC.TREATMENT)
        self.assertEqual(TC.BUNDLES["E"], "production")
        env = TC.arm_environment({"PATH": "/bin"}, "E")
        self.assertEqual(env[TC.BUNDLE_ENV], "production")

    def test_ordering_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(OR.read_ordering(Path(tmp) / OR.ORDERING_FILE), "serial")
            OR.write_ordering_config(Path(tmp), "early", "E")
            self.assertEqual(OR.read_ordering(Path(tmp) / OR.ORDERING_FILE), "early")
            with self.assertRaises(FileExistsError):
                OR.write_ordering_config(Path(tmp), "early", "E")

    def test_e5_requires_exactly_once_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cell = Path(tmp)
            (cell / "profiler" / "edge").mkdir(parents=True)
            (cell / "run4_phase6").mkdir()
            (cell / "run4_phase6" / "edge_report.json").write_text(json.dumps(
                {"evaluator": {"submitted": 10}, "counters": {"evaluations_submitted": 0}}))
            (cell / "profiler" / "edge" / "EDGE_ORDERING.json").write_text(json.dumps(
                {"ordering": "early", "counters": {"early_submissions": 10, "loop_submission_suppressed": 10}}))
            self.assertTrue(RUN.ordering_evidence(cell, expected="early")["passed"])
            self.assertFalse(RUN.ordering_evidence(cell, expected="serial")["passed"])
            (cell / "profiler" / "edge" / "EDGE_ORDERING.json").write_text(json.dumps(
                {"ordering": "early", "counters": {"early_submissions": 9, "loop_submission_suppressed": 9}}))
            self.assertFalse(RUN.ordering_evidence(cell, expected="early")["passed"])

    def test_entry_module_name_matches_the_pinned_sampler_edge_rule(self) -> None:
        from . import sampler_v1 as SA

        self.assertEqual(SA.classify("python3", "python3 -u -m rl_agent.splitfusion_single_gpu_"
                                     "profiler_v1.edge_profiled_v1_ordering --edge"), "edge")


if __name__ == "__main__":
    unittest.main()
