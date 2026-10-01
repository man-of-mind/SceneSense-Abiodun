"""Offline and composed tests for the single-GPU profiler (CPU; no CARLA/OAI/Docker).

Run: ``env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m unittest
rl_agent.splitfusion_single_gpu_profiler_v1.test_single_gpu_profiler_v1``
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2 as LS
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as ER
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import reward_hold_controller_v2 as R
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import ue_telemetry_provider_v2 as T
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import (
    CONTRACT,
    runtimes,
)
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import (
    LINEAGE,
    MS,
    SPEC,
    FakeActor,
    _fresh_radio,
    _scene,
)
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_ue_telemetry_provider_v2 import Harness
from rl_agent.splitfusion_live_dispatch_v1 import timing as TM

from . import analyze_v1 as AN
from . import child_v1 as CH
from . import identity_v1 as IV
from . import recorder_v1 as RV
from . import runner_v1 as RUN
from . import sampler_v1 as SA
from . import seams_ue_v1 as SU
from . import stages_v1 as SV

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
BASE_COMMIT = "6199d49a4a6091b5f6fa0be9f94d1c4afd0dff0b"


def records(path: Path) -> list[dict]:
    return list(RV.iter_records(path))


class Saved:
    """Save and restore module attributes a seam replaces."""

    def __init__(self, *targets: tuple) -> None:
        self._saved = [(module, name, getattr(module, name)) for module, name in targets]

    def restore(self) -> None:
        for module, name, value in self._saved:
            setattr(module, name, value)


class FakeEvent:
    synchronize_calls = 0

    def __init__(self, ready_after: int = 1) -> None:
        self.recorded = 0
        self.queries = 0
        self.ready_after = ready_after

    def record(self) -> None:
        self.recorded += 1
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.queries > self.ready_after

    def elapsed_time(self, other: "FakeEvent") -> float:
        return 1.25

    def synchronize(self) -> None:          # the recorder must never call this
        FakeEvent.synchronize_calls += 1
        raise AssertionError("synchronize called")


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------


class RecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        RV.deactivate()
        self.tmp.cleanup()

    def test_create_only_refuses_an_existing_file_and_never_overwrites(self) -> None:
        path = self.dir / "p.jsonl"
        path.write_text("keep\n")
        with self.assertRaises(FileExistsError):
            RV.ProfileRecorderV1(path, role="t").start()
        self.assertEqual(path.read_text(), "keep\n")

    def test_spans_nest_batch_and_close_with_footer(self) -> None:
        path = self.dir / "p.jsonl"
        rec = RV.ProfileRecorderV1(path, role="t", flush_interval_s=0.01).start()
        rec.bind(frame_id=7, tensor_seq=3)
        with rec.span("outer"):
            with rec.span("inner") as span:
                span.annotate(result=1)
            rec.emit_span("derived", RV.raw_ns() - 10, RV.raw_ns())
        rec.mark("m", value=2)
        time.sleep(0.05)
        summary = rec.close()
        rows = records(path)
        self.assertEqual(rows[0]["r"], "header")
        self.assertEqual(rows[-1]["r"], "footer")
        spans = {r["stage"]: r for r in rows if r.get("r") == "span"}
        self.assertEqual(spans["inner"]["parent"], spans["outer"]["seq"])
        self.assertEqual(spans["derived"]["parent"], spans["outer"]["seq"])
        self.assertEqual(spans["inner"]["result"], 1)
        self.assertEqual(spans["outer"]["frame_id"], 7)
        self.assertGreaterEqual(summary["counters"]["batches_written"], 2)
        self.assertEqual(summary["counters"].get("dropped_records", 0), 0)
        self.assertFalse(rec._thread.is_alive())          # complete teardown
        self.assertIsNone(rec._handle)

    def test_instrumented_exception_propagates_unchanged(self) -> None:
        rec = RV.ProfileRecorderV1(self.dir / "p.jsonl", role="t").start()
        error = KeyError("boom")
        with self.assertRaises(KeyError) as caught:
            with rec.span("s"):
                raise error
        self.assertIs(caught.exception, error)
        rec.close()
        span = [r for r in records(self.dir / "p.jsonl") if r.get("r") == "span"][0]
        self.assertFalse(span["ok"])

    def test_write_failure_never_raises_and_is_counted(self) -> None:
        class Broken:
            def __init__(self) -> None:
                self.calls = 0

            def write(self, _text: str) -> None:
                self.calls += 1
                if self.calls > 1:                     # header ok, then "disk full"
                    raise OSError(28, "No space left on device")

            def flush(self) -> None:
                pass

            def fileno(self) -> int:
                raise OSError("closed")

            def close(self) -> None:
                pass

        rec = RV.ProfileRecorderV1(self.dir / "p.jsonl", role="t", opener=lambda p: Broken(),
                                   flush_interval_s=0.01).start()
        for _ in range(20):
            with rec.span("s"):
                pass
        time.sleep(0.05)
        summary = rec.close()
        self.assertEqual(summary["counters"]["write_failed"], 1)
        self.assertGreater(summary["counters"]["dropped_records"], 0)

    def test_bounded_buffer_drops_and_counts(self) -> None:
        rec = RV.ProfileRecorderV1(self.dir / "p.jsonl", role="t", max_buffered_records=5,
                                   flush_interval_s=60.0)
        rec._handle = open(self.dir / "p.jsonl", "x")
        for _ in range(9):
            rec.mark("m")
        # the first record of a thread is its "thread" record: 1 + 4 marks kept, 5 dropped
        self.assertEqual(rec.counters["records"], 5)
        self.assertEqual(rec.counters["dropped_records"], 5)
        rec._handle.close()

    def test_cuda_events_resolve_by_query_without_synchronize(self) -> None:
        FakeEvent.synchronize_calls = 0
        rec = RV.ProfileRecorderV1(self.dir / "p.jsonl", role="t", flush_interval_s=0.01,
                                   cuda_events=lambda: FakeEvent(ready_after=2)).start()
        for _ in range(4):
            with rec.span("g", cuda=True):
                pass
        time.sleep(0.15)
        summary = rec.close()
        cuda = [r for r in records(self.dir / "p.jsonl") if r.get("r") == "cuda"]
        self.assertEqual(len(cuda), 4)
        self.assertTrue(all(r["status"] == "RESOLVED" and r["gpu_ms"] == 1.25 for r in cuda))
        self.assertEqual(FakeEvent.synchronize_calls, 0)
        self.assertEqual(summary["counters"]["cuda_resolved"], 4)

    def test_unresolved_cuda_is_reported_not_waited_for(self) -> None:
        rec = RV.ProfileRecorderV1(self.dir / "p.jsonl", role="t",
                                   cuda_events=lambda: FakeEvent(ready_after=10**9)).start()
        with rec.span("g", cuda=True):
            pass
        started = time.monotonic()
        summary = rec.close(cuda_drain_s=0.1)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertEqual(summary["counters"]["cuda_unresolved"], 1)

    def test_cuda_is_never_initialized_by_the_recorder(self) -> None:
        import torch

        rec = RV.ProfileRecorderV1(self.dir / "p.jsonl", role="t").start()
        with rec.span("g", cuda=True):
            pass
        summary = rec.close()
        self.assertFalse(torch.cuda.is_initialized())
        self.assertEqual(summary["cuda_state"], "NOT_INITIALIZED_BY_PROCESS")
        self.assertEqual(summary["counters"]["cuda_spans_without_events"], 1)

    def test_broken_gauge_is_counted_and_the_run_continues(self) -> None:
        rec = RV.ProfileRecorderV1(self.dir / "p.jsonl", role="t", gauge_interval_s=0.01).start()
        rec.register_gauge("bad", lambda: 1 / 0)
        rec.register_gauge("good", lambda: 3)
        time.sleep(0.08)
        summary = rec.close()
        self.assertGreater(summary["gauge_errors"]["bad"], 0)
        gauges = [r for r in records(self.dir / "p.jsonl") if r.get("r") == "gauge"]
        self.assertTrue(gauges and all(g["values"] == {"good": 3} for g in gauges))

    def test_null_recorder_is_inert(self) -> None:
        null = RV.NULL
        with null.span("x") as span:
            span.annotate(a=1)
        null.mark("m")
        null.bind(frame_id=1)
        self.assertFalse(null.bound())
        self.assertEqual(null.close(), {"enabled": False})
        self.assertIs(RV.active(), RV.NULL)


class StageRecorderTest(unittest.TestCase):
    def test_profiled_stage_recorder_keeps_the_base_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rec = RV.activate(RV.ProfileRecorderV1(Path(tmp) / "p.jsonl", role="t").start())
            try:
                Profiled = SV.make_profiled_stage_recorder(TM.StageRecorder, "ue.codec",
                                                           RV.active)
                base, profiled = TM.StageRecorder(TM.UE_STAGES), Profiled(TM.UE_STAGES)
                for recorder in (base, profiled):
                    with recorder.stage("total_ue_preparation"):
                        with recorder.stage("front_backbone"):
                            pass
                    with self.assertRaises(ValueError):
                        with recorder.stage("not_a_stage"):
                            pass
                self.assertEqual([b.name for b in base.snapshot().boundaries],
                                 [b.name for b in profiled.snapshot().boundaries])
                self.assertEqual(base.snapshot().clock, profiled.snapshot().clock)
                rec.bind(frame_id=1)
                with profiled.stage("front_backbone"):
                    pass
            finally:
                rec.close()
                RV.deactivate()
            spans = [r for r in records(Path(tmp) / "p.jsonl") if r.get("r") == "span"]
            self.assertEqual([s["stage"] for s in spans], ["ue.codec.front_backbone"])


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------


class SamplerTest(unittest.TestCase):
    def test_short_run_reports_intervals_and_survives_missing_nvml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "s.jsonl"

            def no_nvml():
                raise RuntimeError("NVML unavailable in this test")

            sampler = SA.SamplerV1(out, interval_ms=20.0, watch_pid=None,
                                   extra_roles={os.getpid(): "runner"}, nvml_factory=no_nvml)
            summary = sampler.run(max_samples=10)
            rows = records(out)
            self.assertEqual(rows[0]["r"], "header")
            self.assertEqual(rows[-1]["r"], "footer")
            samples = [r for r in rows if r["r"] == "sample"]
            self.assertEqual(len(samples), 10)
            self.assertIn("NVML unavailable", rows[0]["nvml_error"])
            self.assertTrue(all(any(p["role"] == "runner" for p in s["procs"]) for s in samples))
            self.assertEqual(summary["interval"]["n"], 9)
            with self.assertRaises(FileExistsError):
                SA.SamplerV1(out, interval_ms=20.0, watch_pid=None, extra_roles={},
                             nvml_factory=no_nvml)

    def test_a_late_tick_counts_dropped_samples_instead_of_bursting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sampler = SA.SamplerV1(Path(tmp) / "s.jsonl", interval_ms=20.0, watch_pid=None,
                                   extra_roles={}, nvml_factory=lambda: None)
            real = sampler.sample
            calls = {"n": 0}

            def slow(scheduled, previous):
                calls["n"] += 1
                if calls["n"] == 3:
                    time.sleep(0.07)                     # 3.5 intervals
                return real(scheduled, previous)

            sampler.sample = slow
            summary = sampler.run(max_samples=6)
            self.assertGreaterEqual(summary["dropped_samples"], 2)
            self.assertLess(summary["interval"]["p50_ms"], 40.0)

    def test_sigterm_tears_the_sampler_down_with_a_footer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out, ready = Path(tmp) / "s.jsonl", Path(tmp) / "ready.json"
            process = subprocess.Popen(
                [sys.executable, "-m", "rl_agent.splitfusion_single_gpu_profiler_v1.sampler_v1",
                 "--out", str(out), "--interval-ms", "50", "--ready-file", str(ready)],
                cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + 20
            while not ready.is_file() and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(0.4)
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=20), 0)
            rows = records(out)
            self.assertEqual(rows[-1]["r"], "footer")
            self.assertGreater(rows[-1]["samples"], 2)

    def test_stat_parsing_tolerates_spaces_and_parentheses_in_comm(self) -> None:
        fields = " ".join(["S", "1"] + ["0"] * 9 + ["11", "12"] + ["0"] * 4 + ["5"] + ["0"] * 3
                          + ["77"] + ["0"] * 14 + ["3"])
        stat = SA.parse_stat(f"42 (a (b) c) {fields}")
        self.assertEqual((stat["comm"], stat["utime"], stat["stime"], stat["threads"],
                          stat["rss_pages"], stat["processor"]), ("a (b) c", 11, 12, 5, 77, 3))

    def test_role_rules(self) -> None:
        self.assertEqual(SA.classify("python3", "python3 -m rl_agent.splitfusion_single_gpu_"
                                     "profiler_v1.child_v1 --x"), "ue")
        self.assertEqual(SA.classify("python3", "python3 -m rl_agent.splitfusion_single_gpu_"
                                     "profiler_v1.edge_profiled_v1 --edge"), "edge")
        self.assertEqual(SA.classify("CarlaUnreal-Lin", "CarlaUnreal-Linux-Shipping"), "carla")
        self.assertEqual(SA.classify("nr-softmodem", ""), "oai_gnb")
        self.assertEqual(SA.classify("nr-uesoftmodem", ""), "oai_ue")
        self.assertIsNone(SA.classify("bash", "bash"))


# ---------------------------------------------------------------------------
# analyzer
# ---------------------------------------------------------------------------


def _span(seq, stage, start, end, parent=None, tid=1, **extra):
    return {"r": "span", "seq": seq, "stage": stage, "start_raw_ns": start, "end_raw_ns": end,
            "parent": parent, "tid": tid, "ph": "live", **extra}


def _stream(*spans):
    return [{"r": "header", "clock": {"raw_ns": 0, "wall_ns": 10}}, *spans,
            {"r": "footer", "t_raw_ns": 1000}]


class AnalyzerTest(unittest.TestCase):
    def test_clean_stream_passes(self) -> None:
        result = AN.validate_stream(_stream(
            _span(1, "ue.worker_frame", 0, 100),
            _span(2, "ue.submit", 10, 90, parent=1),
            _span(3, "ue.plan", 20, 30, parent=2),
            _span(4, "agent.actor", 22, 28, parent=3),
            _span(5, "ue.input_7ch", 31, 40, parent=2)), source="t")
        self.assertTrue(result["passed"], result)

    def test_each_structural_defect_is_detected(self) -> None:
        cases = {
            "negative_interval": [_span(1, "ue.worker_frame", 10, 5)],
            "crossed_siblings": [_span(1, "ue.worker_frame", 0, 100),
                                 _span(2, "ue.sensor_wait", 10, 50, parent=1),
                                 _span(3, "ue.radar_window", 40, 60, parent=1)],
            "nesting_registry": [_span(1, "ue.worker_frame", 0, 100),
                                 _span(2, "agent.actor", 10, 20, parent=1)],
            "outside_parent": [_span(1, "ue.worker_frame", 0, 100),
                               _span(2, "ue.submit", 50, 150, parent=1)],
            "non_monotonic_span_start": [_span(1, "ue.worker_frame", 100, 200),
                                         _span(2, "ue.worker_frame", 50, 60)],
            "duplicate_seq": [_span(1, "ue.worker_frame", 0, 1),
                              _span(1, "ue.worker_frame", 2, 3)],
            "unregistered_stage": [_span(1, "nope", 0, 1)],
        }
        for problem, spans in cases.items():
            result = AN.validate_stream(_stream(*spans), source=problem)
            self.assertIn(problem, result["problems"], (problem, result))
            self.assertFalse(result["passed"])
        missing_footer = AN.validate_stream(_stream()[:-1], source="f")
        self.assertIn("footer_count", missing_footer["problems"])

    def test_identity_conflicts_and_foreign_frames(self) -> None:
        transmitted = {5: {"tensor_seq": 2, "session_uuid": "s", "decision_seq": 1}}
        rows = [_span(1, "edge.process", 0, 1, frame_id=5, tensor_seq=2, session_uuid="s"),
                _span(2, "edge.process", 0, 1, frame_id=5, tensor_seq=3),
                _span(3, "edge.process", 0, 1, frame_id=6),
                _span(4, "ue.worker_frame", 0, 1, frame_id=9)]
        result = AN.identity_conflicts(rows, transmitted, pre_send_stages=AN.PRE_SEND_STAGES)
        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(result["foreign_frames"], 1)     # pre-send stage 9 is allowed

    def test_coverage_against_registered_denominators(self) -> None:
        rows = {"ue": [_span(1, "ue.submit", 0, 1, frame_id=1),
                       _span(2, "ue.submit", 2, 3, frame_id=2)]}
        result = AN.coverage(rows, {"sent": {1, 2, 3}})
        self.assertEqual((result["ue.submit"]["covered"], result["ue.submit"]["missing"],
                          result["ue.submit"]["missing_examples"]), (2, 1, [3]))
        self.assertEqual(set(result), set(SV.STAGES))

    def test_clock_report_detects_a_foreign_clock(self) -> None:
        same = {"a": [{"r": "header", "clock": {"raw_ns": 0, "wall_ns": 100}, "boot_id": "b"}],
                "b": [{"r": "header", "clock": {"raw_ns": 5, "wall_ns": 105}, "boot_id": "b"}]}
        self.assertTrue(AN.clock_report(same, [])["single_raw_domain"])
        other = dict(same, b=[{"r": "header", "clock": {"raw_ns": 0, "wall_ns": 10**10},
                               "boot_id": "b"}])
        self.assertFalse(AN.clock_report(other, [])["single_raw_domain"])

    def test_reward_semantics_use_the_registered_inclusive_deadline(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as C

        opened = 1_000_000_000
        q = 0.75
        good = {"terminal": "SUCCESS", "action_open_timestamp_ns": opened,
                "resolution_timestamp_ns": opened + 170 * MS, "q_perc": q,
                "reward": q - C.REWARD_LATENCY_WEIGHT * (170.0 / C.REWARD_DEADLINE_MS),
                "learning_included": True}
        timeout = {"terminal": "TIMEOUT", "action_open_timestamp_ns": opened,
                   "resolution_timestamp_ns": opened + 170 * MS + 1, "reward": -1.0,
                   "learning_included": True}
        ue = {"resolutions": [good, timeout],
              "decisions": [{"kind": "POLICY_DECISION", "frame_id": 1,
                             "action_open": {"ns": opened}}],
              "feedback_rows": [{"frame_id": 1, "class": "ACCEPTED",
                                 "receipt": {"ns": opened + 170 * MS}}]}
        kwargs = {"contract_module": "rl_agent.splitfusion_hybrid_sac_run4_v1.run4_contract",
                  "controller_module":
                      "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.reward_hold_controller_v2"}
        self.assertTrue(AN.reward_semantics(ue, **kwargs)["passed"])
        tampered = dict(ue, resolutions=[dict(good, reward=good["reward"] + 1e-12), timeout])
        self.assertEqual(AN.reward_semantics(tampered, **kwargs)["resolution_errors"], 1)
        late = dict(ue, feedback_rows=[{"frame_id": 1, "class": "ACCEPTED",
                                        "receipt": {"ns": opened + 170 * MS + 1}}])
        self.assertEqual(AN.reward_semantics(late, **kwargs)["feedback_class_errors"], 1)

    def test_actor_replay_detects_a_changed_decision(self) -> None:
        span = _span(1, "agent.actor", 0, 1, frame_id=4, features=[0.1, 0.2],
                     decided_mode_id=3, decided_q_e4=500, state_sha256="s", heads_sha256="h")
        decisions = [{"frame_id": 4, "kind": "POLICY_DECISION", "mode_id": 3, "q_e4": 500}]
        loader = f"{__name__}:_fake_actor_loader"
        self.assertTrue(AN.actor_replay([span], decisions, loader=loader,
                                        method="act_on_vector")["passed"])
        changed = [dict(decisions[0], q_e4=501)]
        self.assertFalse(AN.actor_replay([span], changed, loader=loader,
                                         method="act_on_vector")["passed"])


class _FakeReplayActor:
    def act_on_vector(self, values):
        return SimpleNamespace(mode_id=3, q_e4=500, state_sha256="s", heads_sha256="h")


def _fake_actor_loader():
    return _FakeReplayActor()


# ---------------------------------------------------------------------------
# composed: the real Phase-6 pipeline/engine/controller/edge with CPU fakes
# ---------------------------------------------------------------------------


class _DeterministicPipeline:
    """The Phase-6 test pipeline with deterministic decision sessions."""

    def __init__(self) -> None:
        self.h = Harness()
        counter = iter(range(10**6))
        self.actor = FakeActor()
        self.engine = E.Run4DecisionEngineV2(
            contract_=CONTRACT, actor=self.actor, ue_id=self.h.provider.ue_label,
            controller_lineage_sha256=LINEAGE,
            session_factory=lambda: f"00000000-0000-4000-8000-{next(counter):012d}")
        ue, _edge, _ = runtimes()
        self.pipe = U.Run4FramePipelineV2(
            engine=self.engine, continuous_ue=ue, provider=self.h.provider,
            stream_id="ue288_profiler_test", run_id="run", cell_id="cell",
            clock=self.h._ticking_clock())
        self.identity: dict[int, dict] = {}
        self.sent = 0
        self.sensor_stages: dict = {}

    def submit(self, *, frame_id: int, fresh: bool = True, invalid_scene: bool = False) -> dict:
        self.h.host.advance(100 * MS)
        if fresh:
            _fresh_radio(self.h)
        scene = _scene(self.h)
        if invalid_scene:                          # forces the registered fallback
            scene = U.SceneDescriptorsV2(camera_si=None, radar_p40=scene.radar_p40,
                                         camera_status="TEST_INVALID",
                                         radar_status="VALID",
                                         source_raw_ns=scene.source_raw_ns,
                                         available_raw_ns=scene.available_raw_ns)
        prepared = self.pipe.process(
            frame_id=frame_id, capture_wall_ns=1_790_000_000_000_000_000 + frame_id * 100 * MS,
            ego_pose=(1.0, 2.0, 0.1, 0.0, 90.0, 0.0), scene=scene,
            input_7ch=lambda: object())
        self.identity[frame_id] = dict(prepared.identity)
        self.sent += 1
        return {"sent": True, "prepared": prepared}


def run_scenario(profiled: bool, directory: Path, *, opener=None) -> dict:
    """Eight frames, edge processing, evaluator feedback and resolution."""
    saved = Saved((X, "StageRecorder"), (W, "pack_sfd4"), (LS, "build_live_state"),
                  (T, "open_decision"))
    rec = None
    try:
        p = _DeterministicPipeline()
        runtime = SimpleNamespace(engine=p.engine, pipeline=p.pipe, submit=p.submit,
                                  _run4_identity=p.identity, sent=0, sensor_stages={})
        if profiled:
            rec = RV.ProfileRecorderV1(directory / "ue.profile.jsonl", role="ue",
                                       flush_interval_s=0.01,
                                       **({} if opener is None else {"opener": opener})).start()
            RV.activate(rec)
            rec.register_gauge("explodes", lambda: 1 / 0)          # a failing sampler input
            X.StageRecorder = SV.make_profiled_stage_recorder(TM.StageRecorder, "ue.codec",
                                                              RV.active)
            W.pack_sfd4 = SU.bound_span(RV.active, "ue.sfd4_pack", W.pack_sfd4)
            LS.build_live_state = SU.bound_span(RV.active, "agent.state_build",
                                                LS.build_live_state)
            T.open_decision = SU.bound_span(RV.active, "agent.telemetry_snapshot",
                                            T.open_decision)
            SU.instrument_runtime(runtime, rec, policy_identity={"binding": "test"})
            # No phase mark: the harness feeds simulated receipt stamps, and a
            # real-clock mark on the same thread would (correctly) fail monotonicity.
            rec._phase = "live"
        out = {"plans": [], "wires": [], "feedback": [], "classes": []}
        processor = ER.Run4EdgeProcessorV2(
            contract=CONTRACT, run_id="run", cell_id="cell", service_deadline_s=0.1,
            ack_timeout_s=0.5,
            compute=lambda **kw: ([{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0}],
                                  np.zeros((4, 4), np.uint8)))
        sent = []
        gt = {"semantic": np.zeros((4, 4), np.uint8),
              "objects": [{"class_name": "vehicle", "world_x": 1.0, "world_y": 2.0}]}
        evaluator = ER.Run4EvaluatorV2(spec=SPEC, send=sent.append, match_distance_m=3.0,
                                       gt_timeout_s=0.1, read_ground_truth=lambda **kw: gt)
        for index, frame in enumerate(range(2001, 2009)):
            if rec is not None:                     # as the collector worker does
                rec.bind(frame_id=frame)
                with rec.span("ue.worker_frame"):
                    result = runtime.submit(frame_id=frame, invalid_scene=(index == 6))
                rec.unbind()
            else:
                result = runtime.submit(frame_id=frame, invalid_scene=(index == 6))
            prepared = result["prepared"]
            out["plans"].append((prepared.plan.kind.value, prepared.plan.profile.mode_id,
                                 prepared.plan.profile.q_e4, prepared.plan.tensor_seq))
            out["wires"].append(hashlib.sha256(prepared.wire).hexdigest())
            processed = processor.process(prepared.wire, edge_timing={})
            if processed.evaluation is not None:
                evaluator.evaluate(processed.evaluation)
                feedback, _reason = W.decode_feedback(sent[-1])
                out["feedback"].append(hashlib.sha256(sent[-1]).hexdigest())
                ticket = p.engine.controller.current
                outcome = runtime.engine.on_feedback(
                    feedback, receipt_raw_ns=ticket.action_open_ns + (60 + 70 * index) * MS)
                out["classes"].append(outcome.value)
        out["resolutions"] = [(r.terminal.value, r.reward, r.latency_ms)
                              for c in p.engine.controllers for r in c.resolutions()]
        out["counters"] = dict(vars(p.engine.counters))
        out["actor_calls"] = p.actor.calls
        out["identity"] = p.identity
        return out
    finally:
        if rec is not None:
            rec.close()
        RV.deactivate()
        saved.restore()


class ComposedPipelineTest(unittest.TestCase):
    def test_profiler_changes_no_action_wire_feedback_terminal_or_reward(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain = run_scenario(False, Path(a))
            profiled = run_scenario(True, Path(b))
            for key in ("plans", "wires", "feedback", "classes", "resolutions", "counters",
                        "actor_calls"):
                self.assertEqual(plain[key], profiled[key], key)
            self.assertIn("FALLBACK", [kind for kind, *_ in plain["plans"]])
            self.assertTrue(plain["resolutions"])
            rows = records(Path(b) / "ue.profile.jsonl")
            stages = {r["stage"] for r in rows if r.get("r") == "span"}
            for expected in ("ue.plan", "agent.actor", "agent.state_build",
                             "agent.telemetry_snapshot", "ue.codec.total_ue_preparation",
                             "ue.codec.front_backbone", "ue.sfd4_pack"):
                self.assertIn(expected, stages)
            transmitted = {f: SU.wire_identity(i) for f, i in profiled["identity"].items()}
            joined = AN.identity_conflicts(rows, transmitted, pre_send_stages=AN.PRE_SEND_STAGES)
            self.assertEqual(joined["conflicts"], 0)
            self.assertGreater(joined["checked"], 0)
            valid = AN.validate_stream(rows, source="composed")
            self.assertTrue(valid["passed"], valid)
            marks = [r for r in rows if r.get("stage") == "ue.feedback_receipt"]
            self.assertEqual(len(marks), len(profiled["classes"]))

    def test_profiler_failures_do_not_change_policy_semantics(self) -> None:
        class FailingFile:
            def __init__(self) -> None:
                self.n = 0

            def write(self, _text: str) -> None:
                self.n += 1
                if self.n > 1:
                    raise OSError(28, "No space left on device")

            def flush(self) -> None:
                pass

            def fileno(self) -> int:
                raise OSError("x")

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            plain = run_scenario(False, Path(a))
            broken = run_scenario(True, Path(b), opener=lambda p: FailingFile())
            for key in ("plans", "wires", "feedback", "classes", "resolutions", "counters"):
                self.assertEqual(plain[key], broken[key], key)

    def test_actor_proxy_forwards_identity_attributes(self) -> None:
        actor = SimpleNamespace(boundary_sha256="b" * 64, act=lambda f: None)
        proxy = SU._ActorProxy(actor, RV.NULL)
        self.assertEqual(proxy.boundary_sha256, "b" * 64)
        self.assertEqual(IV.runtime_actor_identity(actor)["boundary_sha256"], "b" * 64)


# ---------------------------------------------------------------------------
# seams, disabled behaviour, identity, isolation
# ---------------------------------------------------------------------------


class SeamInstallTest(unittest.TestCase):
    def test_ue_seams_wrap_pass_through_and_redirect_edge_and_map(self) -> None:
        from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned

        SU.ensure_legacy_fusion_namespace(pinned.ROOT)
        from data_collection import run_route_b_perception_collection_v2 as RB
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base

        saved = Saved((X, "StageRecorder"), (W, "pack_sfd4"), (LS, "build_live_state"),
                      (T, "open_decision"), (base, "_prepare_live_input"),
                      (pinned, "carla_image_to_bgr"), (RB, "SamplingWorld"),
                      (pinned, "PassiveSplitCollector"), (pinned, "LivePilotCellRuntime"),
                      (pinned, "seed_cell_edge_state"), (pinned, "stop_tail"),
                      (D, "DIRECT_EDGE_MODULE"), (D, "DIRECT_MAP_SERVER"))
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_launch_v2 as EL

        saved_launch = Saved((EL, "up_command"))
        originals = {name: getattr(module, name) for module, name, _ in saved._saved}
        env = os.environ.get(SU.PROFILER_DIR_ENV)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rec = RV.ProfileRecorderV1(Path(tmp) / "p.jsonl", role="ue").start()
                SU.install_ue_seams(rec, attempt_dir=Path(tmp), policy_identity={})
                self.assertIs(W.pack_sfd4.__wrapped__, originals["pack_sfd4"])
                self.assertIs(base._prepare_live_input.__wrapped__,
                              originals["_prepare_live_input"])
                self.assertTrue(issubclass(RB.SamplingWorld, originals["SamplingWorld"]))
                self.assertTrue(issubclass(pinned.PassiveSplitCollector,
                                           originals["PassiveSplitCollector"]))
                self.assertEqual(D.DIRECT_EDGE_MODULE, SU.EDGE_MODULE)
                self.assertTrue((ROOT / D.DIRECT_MAP_SERVER).is_file())
                # unbound wrappers are exact pass-throughs
                self.assertEqual(W.pack_sfd4.__name__, originals["pack_sfd4"].__name__)
                world = SimpleNamespace(tick=lambda: 41, other=5)
                timed = SU._TickTimer(world, RV.active)
                self.assertEqual((timed.tick(), timed.other), (41, 5))
                rec.close()
        finally:
            saved.restore()
            saved_launch.restore()
            if env is None:
                os.environ.pop(SU.PROFILER_DIR_ENV, None)
            else:
                os.environ[SU.PROFILER_DIR_ENV] = env

    def test_edge_seams_install_and_keep_processor_outputs(self) -> None:
        from . import edge_profiled_v1 as EP

        def make(cls):
            return cls(contract=CONTRACT, run_id="run", cell_id="cell", service_deadline_s=0.1,
                       ack_timeout_s=0.5,
                       compute=lambda **kw: ([{"class_name": "vehicle", "world_x": 1.0,
                                               "world_y": 2.0}], np.zeros((4, 4), np.uint8)))

        p = _DeterministicPipeline()
        wire = p.submit(frame_id=3001)["prepared"].wire
        plain = make(ER.Run4EdgeProcessorV2).process(wire, edge_timing={})
        from rl_agent.splitfusion_edge_optimization_v1 import detached_edge_preload_v3 as PL
        from rl_agent.splitfusion_edge_optimization_v1 import detached_runtime as DRT
        from rl_agent.splitfusion_edge_optimization_v1.detached_tail_v3 import (
            DetachedOptimizedTailAdapterV3 as TA,
        )
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as base

        saved = Saved((ER, "StageRecorder"), (DRT, "StageRecorder"), (ER, "ChunkReassembler"),
                      (base, "LatestFramePendingSlot"), (ER, "Run4EdgeProcessorV2"),
                      (ER, "Run4MapPublisherV2"), (ER, "Run4EvaluatorV2"),
                      (PL, "preload_detached_optimized_edge_v3"),
                      (PW, "publish_ready_after_warmup"), (MP, "build_run4_map_update"))
        snapshot = TA.__dict__["snapshot"]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rec = RV.activate(RV.ProfileRecorderV1(Path(tmp) / "e.jsonl", role="edge",
                                                       flush_interval_s=0.01).start())
                rec.set_phase("live")
                EP.install_edge_seams(rec)
                profiled = make(ER.Run4EdgeProcessorV2).process(wire, edge_timing={})
                self.assertEqual(plain.update, profiled.update)
                self.assertEqual(plain.identity, profiled.identity)
                slot = base.LatestFramePendingSlot()
                self.assertEqual(slot.offer("s", {"frame_id": 1}, sequence=1), (True, None))
                self.assertEqual(slot.take(0.01), ("s", {"frame_id": 1}))
                rec.close()
                rows = records(Path(tmp) / "e.jsonl")
                stages = [r["stage"] for r in rows if r.get("r") in ("span", "mark")]
                for expected in ("edge.process", "edge.verify", "edge.compute",
                                 "edge.map_update_build", "edge.queue_offer", "edge.queue_take"):
                    self.assertIn(expected, stages)
                process = next(r for r in rows if r.get("stage") == "edge.process")
                self.assertEqual(process["tensor_seq"], plain.envelope.tensor_seq)
                self.assertTrue(AN.validate_stream(rows, source="edge")["passed"])
        finally:
            saved.restore()
            TA.snapshot = snapshot
            RV.deactivate()

    def test_repository_mount_override_rebinds_only_the_repository(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_launch_v2 as EL

        saved = Saved((EL, "up_command"))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                variables = {name: "v" for name in EL.COMPOSE_VARIABLES}
                before = EL.up_command(variables)
                override = SU.install_repository_mount_override(EL, ROOT, Path(tmp))
                after = EL.up_command(variables)
                self.assertEqual(after, before[:-len(EL.UP_ARGUMENTS)]
                                 + ["-f", str(override)] + list(EL.UP_ARGUMENTS))
                self.assertEqual(EL.forbidden_operations(after), [])
                text = override.read_text()
                self.assertIn(f"source: {ROOT.resolve()}", text)
                self.assertIn("target: /work/abiodun", text)
                self.assertIn("read_only: true", text)
                with self.assertRaises(FileExistsError):      # create-only
                    SU.install_repository_mount_override(EL, ROOT, Path(tmp))
        finally:
            saved.restore()

    def test_edge_profile_copy_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "scratch"
            (scratch / SU.EDGE_SCRATCH_LEAF).mkdir(parents=True)
            (scratch / SU.EDGE_SCRATCH_LEAF / "edge.profile.jsonl").write_text("new\n")
            target = Path(tmp) / "out"
            target.mkdir()
            (target / "edge.profile.jsonl").write_text("old\n")
            result = SU.copy_edge_profile(str(scratch), target)
            self.assertEqual(result["copied"], [])
            self.assertEqual((target / "edge.profile.jsonl").read_text(), "old\n")


class DisabledBehaviourTest(unittest.TestCase):
    def test_absent_or_disabled_config_means_the_unchanged_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp) / "cells" / "c"
            attempt.mkdir(parents=True)
            self.assertIsNone(CH.load_config(attempt))
            config = Path(tmp) / CH.CONFIG_NAME
            config.write_text(json.dumps({"schema": CH.CONFIG_SCHEMA, "enabled": False}))
            self.assertIsNone(CH.load_config(attempt))
            config.unlink()
            config.write_text(json.dumps({"schema": "other", "enabled": True}))
            with self.assertRaises(CH.ProfilerConfigError):
                CH.load_config(attempt)
            self.assertEqual(CH.attempt_dir_from_argv(["--x", "1", "--attempt-dir", "/a"]),
                             Path("/a"))

    def test_disabled_child_delegates_to_the_nobuild_child_with_identical_argv(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
            phase6_live_child_nobuild_v2 as NB,
        )

        seen = []
        saved = Saved((NB, "main"), (NB, "install_run4_seams_nobuild"))
        original_install = NB.install_run4_seams_nobuild
        try:
            NB.main = lambda args: seen.append(list(args)) or 0
            with tempfile.TemporaryDirectory() as tmp:
                attempt = Path(tmp) / "cells" / "c"
                attempt.mkdir(parents=True)
                argv = ["--attempt-dir", str(attempt), "--carla-port", "2000"]
                self.assertEqual(CH.main(argv), 0)
            self.assertEqual(seen, [argv])
            self.assertIs(NB.install_run4_seams_nobuild, original_install)
            self.assertIs(RV.active(), RV.NULL)
        finally:
            saved.restore()


class IdentityAndIsolationTest(unittest.TestCase):
    def test_binding_identity_is_policy_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "w.pt"
            weights.write_bytes(b"weights")
            digest = hashlib.sha256(b"weights").hexdigest()
            binding = Path(tmp) / "binding.json"
            binding.write_text(json.dumps({
                "schema": "some.other.policy.v1", "actor": {"weights_file_name": "w.pt",
                                                            "weights_file_sha256": digest,
                                                            "boundary_sha256": "c" * 64},
                "binding": {"feature_schema_id": "run5_like_22",
                            "policy_feature_order": [f"f{i}" for i in range(22)]}}))
            identity = IV.binding_identity(binding)
            self.assertEqual(identity["state_dim"], 22)
            self.assertTrue(identity["weights_match_declaration"])
            self.assertEqual(identity["feature_schema_id"], "run5_like_22")

    def test_registered_run4_binding_resolves_to_the_registered_actor(self) -> None:
        qualification = RUN.load_qualification()
        identity = IV.binding_identity(ROOT / qualification["policy"]["binding_path"],
                                       ROOT / qualification["policy"]["weights_path"])
        self.assertEqual(identity["weights_file_sha256_observed"],
                         qualification["policy"]["expected_weights_sha256"])
        self.assertEqual(identity["state_dim"], 21)

    def test_profiler_core_names_no_policy(self) -> None:
        core = ("recorder_v1.py", "sampler_v1.py", "nvml_v1.py", "identity_v1.py",
                "stages_v1.py", "analyze_v1.py", "child_v1.py")
        forbidden = ("d064013d", "b61f27a9", "seed43", "run4_contract", "frozen_actor_v2",
                     "POLICY_FEATURE_COUNT")
        for name in core:
            text = (PACKAGE / name).read_text()
            for token in forbidden:
                self.assertNotIn(token, text, f"{name} names {token}")

    def test_tracked_sources_outside_the_package_equal_the_base_commit(self) -> None:
        commit = RUN.base_commit_identity(BASE_COMMIT)
        self.assertTrue(commit["base_is_ancestor"])
        self.assertEqual(commit["tracked_changes_outside_package"], [])

    def test_gate_evaluation_requires_every_gate(self) -> None:
        result = RUN.evaluate_gates(cell_report={}, analysis={}, timing={}, audits={},
                                    expected_weights_sha256="x", sampler_stop={},
                                    leftovers=[], transmitted_budget=300)
        self.assertFalse(result["passed"])
        self.assertEqual(result["verdict"], "INSTRUMENTATION_NOT_QUALIFIED")
        self.assertEqual(set(result["gates"]), set(RUN.load_qualification()["gates"]))


if __name__ == "__main__":
    unittest.main()
