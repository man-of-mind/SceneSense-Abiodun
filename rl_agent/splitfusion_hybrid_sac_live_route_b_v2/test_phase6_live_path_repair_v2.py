"""Offline acceptance tests for addendum 5 (the three Phase-6 live-path repairs).

1. plan-before-preparation split (bit-identity, ordering, fail-closed binding);
2. evaluator without head-of-line blocking (stalled A cannot delay ready B);
3. registered edge terminals resolve the exact reward ticket.

Clock-controlled; no CARLA, OAI, Docker, CUDA or network.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract

from . import phase6_decision_engine_v2 as E
from . import phase6_edge_runtime_v2 as ER
from . import phase6_ue_runtime_v2 as U
from . import reward_hold_controller_v2 as R
from . import run4_live_wire_v2 as W
from .test_phase3_continuous_execution_v2 import CONTRACT, runtimes
from .test_phase6_live_integration_v2 import (
    LINEAGE, MS, SPEC, FakeActor, _fresh_radio, _scene, processor,
)
from .test_ue_telemetry_provider_v2 import Harness

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent.splitfusion_hybrid_sac_live_route_b_v2"
PACKAGE_PATH = "rl_agent/splitfusion_hybrid_sac_live_route_b_v2"
REPAIR_BASE = "8cfb4eda3561b69282b400aa0471f5bfb562096a"
SESSION = "5a5a5a5a-5a5a-4a5a-8a5a-5a5a5a5a5a5a"
WINDOW = {"callbacks": 4, "sweep_indices": [11, 12, 13, 14], "returns": 321,
          "window_span_s": 0.2}
VALID = {
    **{f"seg_{c}_{k}": 10 for c in ("vehicle", "person")
       for k in ("gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels")},
    **{f"loc_{c}_{k}": v for c in ("vehicle", "person")
       for k, v in (("eligible_gt", 1), ("tp", 1), ("fn", 0), ("fp", 0))},
    "loc_vehicle_matched_xy_errors_m": [0.1], "loc_person_matched_xy_errors_m": [0.1],
}


def committed_module(name: str, commit: str = REPAIR_BASE) -> types.ModuleType:
    """The pre-repair module, executed in the package namespace (never saved)."""
    source = subprocess.run(["git", "show", f"{commit}:{PACKAGE_PATH}/{name}.py"],
                            cwd=ROOT, capture_output=True, check=True, text=True).stdout
    module = types.ModuleType(f"{PACKAGE}._before_addendum5_{name}")
    module.__package__ = PACKAGE
    module.__file__ = str(ROOT / PACKAGE_PATH / f"{name}.py")
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def build(module=U, actor=None):
    h = Harness()
    actor = actor or FakeActor()
    engine = E.Run4DecisionEngineV2(contract_=CONTRACT, actor=actor,
                                    ue_id=h.provider.ue_label,
                                    controller_lineage_sha256=LINEAGE,
                                    session_factory=lambda: SESSION)
    ue, _edge, _ = runtimes()
    pipe = module.Run4FramePipelineV2(engine=engine, continuous_ue=ue, provider=h.provider,
                                      stream_id="ue288_phase6_test", run_id="run",
                                      cell_id="cell", clock=h._ticking_clock())
    return h, engine, pipe, actor


def planner(h, pipe, *, scene_calls=None):
    def scene_fn(frame_bgr, window_meta, *, source_raw_ns):
        if scene_calls is not None:
            scene_calls.append(source_raw_ns)
        return U.SceneDescriptorsV2(camera_si=112.0, radar_p40=0.4, camera_status="VALID",
                                    radar_status="VALID", source_raw_ns=source_raw_ns,
                                    available_raw_ns=h.host.raw())
    pipe.stage_clock = h.host.raw
    return U.SensorFirstPlannerV2(pipeline=pipe, lock=threading.Lock(), scene_fn=scene_fn,
                                  clock=h.host.raw)


CAPTURE0 = 1_790_000_000_000_000_000


def run_frame(h, plan, frame_id, *, rgb_age_ns=20 * MS, raster_ns=25 * MS, window=WINDOW,
              prep_window=None, input_7ch=None, drop=()):
    """One frame: radar raster and RGB preparation first, then plan + materialize."""
    h.host.advance(100 * MS)
    _fresh_radio(h)
    rgb = h.host.raw() - rgb_age_ns
    ready = h.host.raw()
    start = h.host.raw()
    h.host.advance(raster_ns)
    timestamp = frame_id / 10.0
    preparation = {"radar_window_ready_raw_ns": ready, "radar_tensor_start_raw_ns": start,
                   "radar_tensor_end_raw_ns": h.host.raw(),
                   "radar_window_sha256": prep_window or U.radar_window_sha256(window,
                                                                               timestamp),
                   "sensor_prepared_raw_ns": h.host.raw()}
    for key in drop:
        preparation.pop(key)
    return plan.plan_and_materialize(
        frame_id=frame_id, capture_wall_ns=CAPTURE0 + frame_id * 100 * MS,
        carla_timestamp=timestamp, frame_bgr=object(), window_meta=window,
        rgb_raw_ns=rgb, preparation=preparation, ego_pose=(1.0, 2.0, 0.1, 0.0, 90.0, 0.0),
        input_7ch=input_7ch or (lambda: object()))


# ---------------------------------------------------------------------------
# Repair 1 (restored by addendum 6): sensor-first planning at the contract boundary
# ---------------------------------------------------------------------------


class PlanningSplitTest(unittest.TestCase):
    def test_01_split_is_bit_identical_to_pre_repair_process(self) -> None:
        try:
            from . import frozen_actor_v2 as FA
            actor = FA.load_registered_actor()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"registered actor unavailable: {exc}")
        old = committed_module("phase6_ue_runtime_v2")
        results = []
        for module, split in ((old, False), (U, True)):
            h, engine, pipe, _ = build(module, actor)
            out = []
            for step in range(6):
                h.host.advance(100 * MS)
                _fresh_radio(h)
                frame_id = 2000 + step
                kwargs = dict(frame_id=frame_id, capture_wall_ns=CAPTURE0 + step * 100 * MS)
                if split:
                    planned = pipe.plan(scene=_scene(h), **kwargs)
                    prepared = pipe.materialize(planned, ego_pose=(1, 2, 0, 0, 90, 0),
                                                input_7ch=lambda: object(), **kwargs)
                else:
                    prepared = pipe.process(scene=_scene(h), ego_pose=(1, 2, 0, 0, 90, 0),
                                            input_7ch=lambda: object(), **kwargs)
                out.append((prepared.plan.kind.value, prepared.plan.profile.mode_id,
                            prepared.plan.profile.q_e4, prepared.wire))
            features = [d.get("features") for d in pipe.decisions]
            results.append((out, features, engine.counters.actor_calls))
        self.assertEqual(results[0][0], results[1][0])      # kinds, mode, q_e4, wire
        self.assertEqual(results[0][1], results[1][1])      # exact feature vectors
        self.assertEqual(results[0][2], results[1][2])      # actor calls
        self.assertTrue(any(f is not None for f in results[1][1]))

    def test_02_sensor_preparation_precedes_action_open_and_front_follows(self) -> None:
        h, engine, pipe, actor = build()
        plan = planner(h, pipe)
        prepared = run_frame(h, plan, 10, raster_ns=60 * MS,
                             input_7ch=lambda: h.host.advance(40 * MS) or object())
        self.assertEqual(prepared.plan.kind, E.FrameKind.POLICY_DECISION)
        record = pipe.decisions[-1]
        self.assertEqual(U.boundary_violations(record), [])
        stages = record["stages"]
        opened = engine.controller.current.action_open_ns
        self.assertEqual(record["action_open"]["ns"], opened)          # never re-stamped
        self.assertLessEqual(stages["radar_tensor_end_raw_ns"], stages["si_p40_start_raw_ns"])
        self.assertLessEqual(stages["si_p40_end_raw_ns"], record["state_commit"]["ns"])
        self.assertLessEqual(opened, stages["input_7ch_start_raw_ns"])
        self.assertGreaterEqual(stages["front_start_raw_ns"] - opened, 40 * MS)
        self.assertEqual(actor.calls, 1)
        processed = processor().process(prepared.wire, edge_timing={})
        feedback, _ = W.quality_feedback(SPEC, {"frame_id": 10, **VALID}, processed.envelope)
        self.assertEqual(engine.on_feedback(feedback, receipt_raw_ns=opened + 170 * MS),
                         R.FeedbackClass.ACCEPTED)

    def test_02b_front_inside_the_clock_can_cause_a_timeout(self) -> None:
        h, engine, pipe, _ = build()
        prepared = run_frame(h, planner(h, pipe), 10,
                             input_7ch=lambda: h.host.advance(150 * MS) or object())
        opened = engine.controller.current.action_open_ns
        processed = processor().process(prepared.wire, edge_timing={})
        feedback, _ = W.quality_feedback(SPEC, {"frame_id": 10, **VALID}, processed.envelope)
        self.assertEqual(engine.on_feedback(feedback, receipt_raw_ns=opened + 170 * MS + 1),
                         R.FeedbackClass.LATE_ORPHAN)

    def test_03_window_and_identity_mismatches_fail_closed(self) -> None:
        h, engine, pipe, actor = build()
        with self.assertRaisesRegex(U.Phase6UeError, "rasterized radar window"):
            run_frame(h, planner(h, pipe), 10, prep_window="0" * 64)
        self.assertIsNone(engine.controller.current)                   # nothing assigned
        self.assertEqual(actor.calls, 0)
        with self.assertRaisesRegex(U.Phase6UeError, "did not complete"):
            run_frame(h, planner(h, pipe), 11, drop=("radar_tensor_end_raw_ns",))
        with self.assertRaisesRegex(U.Phase6UeError, "incomplete radar window"):
            run_frame(h, planner(h, pipe), 12, window={**WINDOW, "callbacks": 3})
        for name, kwargs in {"frame": dict(frame_id=99),
                             "capture": dict(capture_wall_ns=CAPTURE0 + 1),
                             "carla": dict(carla_timestamp=9.9),
                             "window": dict(radar_window_sha256="1" * 64)}.items():
            with self.subTest(case=name):
                h2, engine2, pipe2, _ = build()
                h2.host.advance(100 * MS)
                _fresh_radio(h2)
                planned = pipe2.plan(frame_id=10, capture_wall_ns=CAPTURE0, scene=_scene(h2),
                                     carla_timestamp=1.0, radar_window_sha256="a" * 64)
                args = dict(frame_id=10, capture_wall_ns=CAPTURE0, carla_timestamp=1.0,
                            radar_window_sha256="a" * 64)
                args.update(kwargs)
                with self.assertRaises(E.InfrastructureFault):
                    pipe2.materialize(planned, ego_pose=(1, 2, 0, 0, 90, 0),
                                      input_7ch=lambda: object(), **args)
                self.assertIsNotNone(engine2.faulted)

    def test_03b_failed_front_is_registered_not_left_open(self) -> None:
        h, engine, pipe, _ = build()

        def broken():
            raise OSError("7-channel construction failed")
        with self.assertRaises(E.InfrastructureFault):
            run_frame(h, planner(h, pipe), 10, input_7ch=broken)
        self.assertIn("7-channel construction failed", engine.faulted)

    def test_04_rgb_source_timestamp_and_100ms_rule_unchanged(self) -> None:
        h, engine, pipe, _ = build()
        calls = []
        prepared = run_frame(h, planner(h, pipe, scene_calls=calls), 10, rgb_age_ns=20 * MS)
        self.assertEqual(calls[-1], pipe.decisions[-1]["stages"]["rgb_receipt_raw_ns"])
        self.assertEqual(prepared.plan.kind, E.FrameKind.POLICY_DECISION)
        h2, engine2, pipe2, _ = build()
        stale = run_frame(h2, planner(h2, pipe2), 10, rgb_age_ns=80 * MS, raster_ns=25 * MS)
        self.assertEqual(stale.plan.kind, E.FrameKind.FALLBACK)          # 105 ms > 100 ms
        self.assertTrue(any("camera_si" in r for r in stale.plan.fallback_reasons))
        from .ue_telemetry_provider_v2 import TRAINING_FRESHNESS
        self.assertIn("100", repr(TRAINING_FRESHNESS).replace("_", ""))

    def test_05_legacy_default_paths_unchanged(self) -> None:
        for path in ("rl_agent/ue_route_b_split_cell_adapter_v1.py",
                     "rl_agent/splitfusion_direct_edge_map_v1/adapter_direct_v1.py",
                     "rl_agent/splitfusion_quality_feedback_probe_v1/adapter_quality_v1.py",
                     "rl_agent/splitfusion_quality_feedback_probe_v1/live_cell_child.py",
                     "rl_agent/splitfusion_quality_feedback_probe_v1/gt_evidence.py",
                     "rl_agent/splitfusion_live_dispatch_v1/live_pilot_runtime.py"):
            committed = subprocess.run(["git", "show", f"{REPAIR_BASE}:{path}"], cwd=ROOT,
                                       capture_output=True, check=True).stdout
            self.assertEqual((ROOT / path).read_bytes(), committed, path)
        calls = []
        original = types.SimpleNamespace(build_radar_sample=lambda **kw: ("real", kw),
                                         other="forwarded")
        proxy = U._PreparationTimer(original, lambda real, **kw: calls.append(kw) or real(**kw))
        self.assertEqual(proxy.other, "forwarded")
        self.assertEqual(proxy.build_radar_sample(frame_time_s=1.0), ("real",
                                                                        {"frame_time_s": 1.0}))
        self.assertEqual(calls, [{"frame_time_s": 1.0}])
        self.assertEqual(original.build_radar_sample(x=1), ("real", {"x": 1}))


# ---------------------------------------------------------------------------
# Repair 2: evaluator without head-of-line blocking
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _tickets():
    h, engine, pipe, _ = build()
    prepared = run_frame(h, planner(h, pipe), 10)
    base = processor().process(prepared.wire, edge_timing={}).evaluation
    a = dataclasses.replace(base, gt_identity={"frame_id": "A"})
    b = dataclasses.replace(
        base, gt_identity={"frame_id": "B"},
        envelope=dataclasses.replace(base.envelope, decision_seq=1, ticket_seq=1,
                                     tensor_seq=5, frame_id=15),
        context=dataclasses.replace(base.context, frame_id=15))
    return a, b


def _gt():
    return {"semantic": np.zeros((4, 4), np.uint8),
            "objects": [{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0}],
            "gt_ready_wall_ns": time.time_ns()}


def _read_factory(ready: set):
    def read(*, expected_identity, timeout_s):
        if expected_identity.get("frame_id") in ready:
            return _gt()
        time.sleep(timeout_s)
        raise RuntimeError("ground truth missing")
    return read


def _wait(predicate, seconds=2.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


class EvaluatorTest(unittest.TestCase):
    def test_06_stalled_gt_a_cannot_delay_ready_gt_b(self) -> None:
        a, b = _tickets()
        sent: list[bytes] = []
        clock = _Clock()
        evaluator = ER.Run4EvaluatorV2(
            spec=SPEC, send=sent.append, match_distance_m=3.0, gt_timeout_s=2.0,
            read_ground_truth=_read_factory({"B"}), probe_timeout_s=0.001,
            poll_interval_s=0.002, clock=clock)
        evaluator.start()
        evaluator.submit(a)
        time.sleep(0.05)
        started = time.monotonic()
        evaluator.submit(b)
        self.assertTrue(_wait(lambda: len(sent) == 1, 1.0))
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.5)                       # B did not wait for A
        first, _ = W.decode_feedback(sent[0])
        self.assertEqual((first.frame_id, first.kind), (15, "DELIVERED_SUCCESS"))
        clock.t = 2.0                                       # A's two-second GT wait ends
        self.assertTrue(_wait(lambda: len(sent) == 2, 1.0))
        second, reason = W.decode_feedback(sent[1])
        self.assertEqual((second.frame_id, second.kind, reason),
                         (10, "EVALUATOR_FAULT", W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE))
        summary = evaluator.close()
        self.assertFalse(summary["worker_alive"])
        self.assertEqual((summary["submitted"], summary["emitted"], summary["gt_expired"]),
                         (2, 2, 1))
        frames = [r["frame_id"] for r in evaluator.records]
        self.assertEqual(sorted(frames), [10, 15])          # A accounted exactly once
        timing_b = next(r["timing"] for r in evaluator.records if r["frame_id"] == 15)
        for key in ("enqueued_wall_ns", "gt_ready_detected_wall_ns",
                    "evaluator_start_wall_ns", "evaluator_end_wall_ns", "gt_ready_wall_ns"):
            self.assertIn(key, timing_b)

    def test_07_shutdown_drains_or_accounts_every_ticket(self) -> None:
        a, b = _tickets()
        sent: list[bytes] = []
        evaluator = ER.Run4EvaluatorV2(
            spec=SPEC, send=sent.append, match_distance_m=3.0, gt_timeout_s=0.2,
            read_ground_truth=_read_factory({"B"}), probe_timeout_s=0.001,
            poll_interval_s=0.002)
        evaluator.start()
        evaluator.submit(a)
        evaluator.submit(b)
        summary = evaluator.close()
        self.assertFalse(summary["worker_alive"])
        self.assertEqual(summary["emitted"], summary["submitted"])
        self.assertEqual(sorted(r["frame_id"] for r in evaluator.records), [10, 15])
        kinds = {r["frame_id"]: r["kind"] for r in evaluator.records}
        self.assertEqual(kinds, {10: "EVALUATOR_FAULT", 15: "DELIVERED_SUCCESS"})
        # at-most-once emission per exact ticket
        evaluator._emit(b, None, W.EvaluatorReason.EVALUATOR_EXCEPTION)
        self.assertEqual(len(sent), 2)
        self.assertEqual(evaluator.counters["duplicate_emission_refused"], 1)

    def test_07b_missing_gt_is_never_a_fabricated_quality(self) -> None:
        a, _b = _tickets()
        sent: list[bytes] = []
        evaluator = ER.Run4EvaluatorV2(
            spec=SPEC, send=sent.append, match_distance_m=3.0, gt_timeout_s=0.05,
            read_ground_truth=_read_factory(set()), probe_timeout_s=0.001,
            poll_interval_s=0.002)
        evaluator.start()
        evaluator.submit(a)
        evaluator.close()
        feedback, reason = W.decode_feedback(sent[0])
        self.assertIsNone(feedback.q_perc)
        self.assertEqual(feedback.kind, "EVALUATOR_FAULT")


# ---------------------------------------------------------------------------
# Repair 3: registered edge terminals -> reward controller
# ---------------------------------------------------------------------------


def _decided(h, engine, pipe, coord, frame_id):
    return run_frame(h, coord, frame_id)


def _terminal_doc(prepared, outcome="SUPERSEDED_PENDING", stage="EDGE_PENDING_REPLACED"):
    return processor().terminal(prepared.wire, outcome=outcome, stage=stage)


class TerminalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.h, self.engine, self.pipe, _ = build()
        self.coord = planner(self.h, self.pipe)
        self.prepared = _decided(self.h, self.engine, self.pipe, self.coord, 10)
        self.opened = self.engine.controller.current.action_open_ns

    def terminal(self, doc):
        terminal = U.registered_terminal_from_message(doc)
        self.assertIsNotNone(terminal)
        return terminal

    def test_08_superseded_closes_active_ticket_as_service_failure(self) -> None:
        terminal = self.terminal(_terminal_doc(self.prepared))
        self.assertEqual(terminal.agent_credit, "CREDIT_SUPERSEDED_BY_FRESHER")
        klass = self.engine.on_registered_terminal(terminal,
                                                   receipt_raw_ns=self.opened + 30 * MS)
        self.assertEqual(klass, R.FeedbackClass.ACCEPTED)
        resolution = self.engine.controller.current.resolution
        self.assertIs(resolution.terminal, contract.RewardTerminal.REGISTERED_SERVICE_FAILURE)
        self.assertEqual(resolution.reward, contract.REGISTERED_FAILURE_REWARD)
        self.assertEqual(resolution.reward, -1.0)
        self.assertIsNone(resolution.q_perc)
        self.assertTrue(resolution.learning_included)
        self.assertEqual(self.engine.counters.session_rollovers, 0)

    def test_09_duplicate_conflict_late_hold_and_fallback(self) -> None:
        doc = _terminal_doc(self.prepared)
        first = self.engine.on_registered_terminal(self.terminal(doc),
                                                   receipt_raw_ns=self.opened + 10 * MS)
        self.assertEqual(first, R.FeedbackClass.ACCEPTED)
        self.assertEqual(self.engine.on_registered_terminal(
            self.terminal(doc), receipt_raw_ns=self.opened + 20 * MS),
            R.FeedbackClass.DUPLICATE_IGNORED)
        with self.assertRaises(R.ConflictingFeedbackError):
            self.engine.on_registered_terminal(
                self.terminal(_terminal_doc(self.prepared, "STALE_BEFORE_EDGE",
                                            "EDGE_STAGE_AFTER_REASSEMBLY")),
                receipt_raw_ns=self.opened + 30 * MS)

    def test_09b_late_terminal_never_attaches_to_newer_action(self) -> None:
        # second tensor of the hold, then time out the first decision
        hold_prepared = run_frame(self.h, self.coord, 11)
        self.assertEqual(hold_prepared.plan.kind, E.FrameKind.POLICY_HOLD)
        self.assertIsNone(U.registered_terminal_from_message(
            _terminal_doc(hold_prepared)))                    # hold: ledger only
        newer = _decided(self.h, self.engine, self.pipe, self.coord, 12)
        self.assertEqual(newer.plan.kind, E.FrameKind.POLICY_DECISION)
        late = self.engine.on_registered_terminal(
            self.terminal(_terminal_doc(self.prepared)),
            receipt_raw_ns=self.engine.controller.current.action_open_ns + 1)
        self.assertEqual(late, R.FeedbackClass.LATE_ORPHAN)
        self.assertIsNone(self.engine.controller.current.resolution)   # newer untouched

    def test_09c_fallback_and_map_feedback_are_ledger_only(self) -> None:
        h, engine, pipe, _ = build()
        prepared = run_frame(h, planner(h, pipe), 10, rgb_age_ns=150 * MS)
        self.assertEqual(prepared.plan.kind, E.FrameKind.FALLBACK)
        self.assertIsNone(U.registered_terminal_from_message(_terminal_doc(prepared)))
        doc = _terminal_doc(self.prepared)
        self.assertIsNone(U.registered_terminal_from_message(
            {**doc, "schema": "splitfusion_direct_map_feedback.run4.v1"}))

    def test_09d_identity_must_match_fully(self) -> None:
        doc = _terminal_doc(self.prepared)
        forged = dict(doc)
        forged["capture_timestamp_ns"] = int(doc["capture_timestamp_ns"]) + 1
        with self.assertRaises(R.ConflictingFeedbackError):
            self.engine.on_registered_terminal(self.terminal(forged),
                                               receipt_raw_ns=self.opened + 10 * MS)


# ---------------------------------------------------------------------------
# Constants, frozen files and imports
# ---------------------------------------------------------------------------


class FrozenTest(unittest.TestCase):
    def test_10_scientific_constants_unchanged(self) -> None:
        self.assertEqual(contract.REWARD_DEADLINE_NS, 170_000_000)
        self.assertEqual(R.K_MIN, 2)
        self.assertEqual(contract.REGISTERED_FAILURE_REWARD, -1.0)
        self.assertEqual((E.FALLBACK["mode_id"], E.FALLBACK["q_e4"]), (11, 9800))
        self.assertEqual((E.COVERAGE_MIN_FRACTION, E.COVERAGE_WARMUP_OPPORTUNITIES),
                         (0.95, 10))
        from . import frozen_actor_v2 as FA
        self.assertEqual(FA.SELECTED.actor_boundary_sha256,
                         "b61f27a9bcd3512ecf52bc35854f6a723d550092db51cf3055347297039cebd3")
        for name in ("frozen_actor_v2.py", "live_state_v2.py", "continuous_execution_v2.py",
                     "run4_live_wire_v2.py", "run4_map_protocol_v2.py",
                     "ue_telemetry_provider_v2.py", "phase6_result_reporting_v2.py",
                     "live_qualification_300_v2.json", "ACTOR_BINDING_V2.json",
                     "phase6_edge_launch_v2.py"):
            # phase6_live_child_v2 / _nobuild_v2: addendum-6 stop and GT seams only.
            committed = subprocess.run(["git", "show", f"{REPAIR_BASE}:{PACKAGE_PATH}/{name}"],
                                       cwd=ROOT, capture_output=True, check=True).stdout
            self.assertEqual((ROOT / PACKAGE_PATH / name).read_bytes(), committed, name)
        for path in ("rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
                     "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/quality.py"):
            committed = subprocess.run(["git", "show", f"{REPAIR_BASE}:{path}"], cwd=ROOT,
                                       capture_output=True, check=True).stdout
            self.assertEqual((ROOT / path).read_bytes(), committed, path)

    def test_10b_quality_calculation_unchanged(self) -> None:
        old = committed_module("phase6_edge_runtime_v2")
        a, _b = _tickets()
        outputs = []
        for module in (old, ER):
            sent = []
            evaluator = module.Run4EvaluatorV2(
                spec=SPEC, send=sent.append, match_distance_m=3.0, gt_timeout_s=0.1,
                read_ground_truth=lambda **kw: _gt())
            evaluator.evaluate(a)
            outputs.append(sent[0])
        self.assertEqual(outputs[0], outputs[1])

    def test_11_prior_attempts_byte_identical(self) -> None:
        import hashlib
        import json as _json
        document = _json.loads((ROOT / PACKAGE_PATH / "phase6_live_path_repair_addendum_5.json")
                               .read_text(encoding="utf-8"))
        self.assertEqual(document["base_commit"], REPAIR_BASE)
        self.assertEqual(len(document["preserved_attempts_sha256"]), 4)
        for files in document["preserved_attempts_sha256"].values():
            for path, digest in files.items():
                target = ROOT / path
                if not target.exists():
                    self.skipTest("prior attempt evidence not present on this host")
                self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), digest, path)

    def test_12_imports_start_nothing(self) -> None:
        code = (
            "import sys\n"
            "seen = []\n"
            "sys.addaudithook(lambda e, a: seen.append(e) if e in "
            "('socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system') else None)\n"
            "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_ue_runtime_v2\n"
            "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2\n"
            "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.reward_hold_controller_v2\n"
            "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_engineering_gates_v2\n"
            "import torch\n"
            "print(seen, torch.cuda.is_initialized())\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                              text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin"})
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertEqual(done.stdout.strip(), "[] False")


if __name__ == "__main__":
    unittest.main()
