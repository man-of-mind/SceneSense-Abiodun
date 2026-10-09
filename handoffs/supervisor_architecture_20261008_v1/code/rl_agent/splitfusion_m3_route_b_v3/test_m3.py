"""Focused M3.1 offline checks (CPU only; no CARLA/OAI/GPU). Run from the live worktree root:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v rl_agent.splitfusion_m3_route_b_v3.test_m3

Groups: S (GOOD_ALL schedule + actor-state isolation), R (route termination incl. the TypeError cause through the real
``run_route_b``), K (capacity substitutions), M (map journal, map delivery vs feedback, freshness horizon), E (edge
evaluator overflow guard on the real early-GT processor), P (retention plan).
"""
import argparse
import importlib
import inspect
import json
import socket
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from . import capacity as CAP
from . import contract as C
from . import good_all_controller as G
from . import map_journal as MJ
from . import route_completion as RC

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------------------------------------ S
class ScheduleTests(unittest.TestCase):
    def test_s1_registration_is_constant_21p5_with_measured_mapping_point(self):
        reg = C.channel_registration(ROOT)
        s = reg["schedule"]
        self.assertEqual((s["schedule_id"], s["target_snr_db"], s["expected_rfsim_command_db"]), ("GOOD_ALL", 21.5, -11.0))
        self.assertEqual(s["prefix_samples"], 4200)
        self.assertEqual(s["trace_sha256"], C.schedule_prefix(4200)[1])
        self.assertFalse(reg["carrier_cell"]["profile_substituted"])
        self.assertEqual(reg["radio_profile"], "OAI_N78_100MHZ_273PRB_4D5U_V1")
        self.assertNotEqual(s["trace_id"], C.CARRIER_TRACE["trace_id"])

    def test_s2_retained_inverse_interpolation_maps_21p5_to_minus_11(self):
        from rl_agent import ue_target_snr_cell_runtime_v1 as base
        config = json.loads((ROOT / C.LIVE_CONFIG).read_bytes())
        replay = base.load_replay_module()
        mapping = base.load_mapping(ROOT / config["network"]["mapping_csv"], replay)
        self.assertEqual(replay.inverse_interpolate(21.5, mapping, 0.25), -11.0)

    def test_s3_prepare_runs_retained_checks_then_returns_constant(self):
        calls = []

        def original(*, campaign, profile_id):
            calls.append(profile_id)
            return "unused", [], {}
        reg = C.channel_registration(ROOT)
        campaign = dict(network=dict(prefix_samples=4200))
        sequence, prefix, frozen = G.build_prepare(original, reg)(campaign=campaign, profile_id="GOOD_ALL")
        self.assertEqual(calls, ["FAVORABLE_STABLE"])
        self.assertEqual(len(prefix), 4200)
        self.assertEqual(set(prefix), {(2, 21.5)})
        self.assertEqual([sequence.next_sample() for _ in range(5)], [(2, 21.5)] * 5)
        self.assertEqual(frozen, dict(profile_id="GOOD_ALL", trace_id="CONST_GOOD_ALL_21P5DB_V1", seed=0,
                                      trace_sha256=reg["schedule"]["trace_sha256"]))
        with self.assertRaises(ValueError):
            G.build_prepare(original, reg)(campaign=campaign, profile_id="FAVORABLE_STABLE")

    def test_s4_profile_argument_rewrite_is_exact(self):
        self.assertEqual(G.rewrite_profile(["--a", "1", "--profile-id", "FAVORABLE_STABLE"]),
                         ["--a", "1", "--profile-id", "GOOD_ALL"])
        for bad in (["--profile-id", "FADE_RECOVERY"], [], ["--profile-id", "FAVORABLE_STABLE"] * 2):
            with self.assertRaises(ValueError):
                G.rewrite_profile(bad)

    def test_s5_actor_sees_only_the_commanded_value_constant_feature(self):
        from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as S
        from rl_agent.splitfusion_run5_live_avo_v1 import snr_live as L
        self.assertAlmostEqual(S.scale_snr_db(21.5), 16.0 / 19.0, places=15)
        fields = set(L.SnrSnapshot.__dataclass_fields__)
        self.assertFalse({f for f in fields if any(t in f for t in ("profile", "trace", "step", "schedule", "route"))})

    def test_s6_controller_module_seam_sets_and_restores(self):
        from rl_agent.splitfusion_run5_full_live_v1 import runtime_policy as RP
        from . import entries
        with tempfile.TemporaryDirectory() as tmp:
            with entries.install_good_all_controller(Path(tmp)):
                self.assertEqual(RP.CONTROLLER_MODULE, C.CONTROLLER_MODULE)
                source = inspect.getsource(RP)
                self.assertIn('argv = [sys.executable, "-m", CONTROLLER_MODULE', source)   # read at call time
            self.assertEqual(RP.CONTROLLER_MODULE, C.RETAINED_CONTROLLER_MODULE)
            self.assertTrue(json.loads((Path(tmp) / "m3_route/CONTROLLER_MODULE_SEAM.json").read_text())["restored"])


# ------------------------------------------------------------------------------------------------------------ R
class _FakeBase:
    """Stands in for PassiveSplitCollector: finish() returns the retained bool cleanup flag."""
    cleanup_value = True

    def __init__(self, **kwargs):
        self.attempt_dir = Path(kwargs["attempt_dir"])
        self.sent, self.dropped, self.failures, self.cleanup_ok = 0, 0, [], False
        self.transport_counters = types.SimpleNamespace(bump=lambda name: None, snapshot=lambda: {})
        self.ticks = 0

    def on_world_tick(self, frame_id, route_tick):
        self.ticks += 1
        if self.ticks % 2 == 0:
            self.sent += 1

    def _fetch_installed_record(self, expected_frame):
        return "OK", []

    def write_per_frame(self):
        (self.attempt_dir / "per_frame_metrics.csv").write_text("frame_id\n")

    def write_perception(self):
        (self.attempt_dir / "perception_metrics.csv").write_text("frame_id\n")

    def finish(self):
        self.cleanup_ok = type(self).cleanup_value
        return self.cleanup_ok


def _early_gt_like(inherited, closed):
    """Faithful replica of exact_gt_live_v2.ue EarlyGtCollector.finish (source asserted in test_r1)."""
    class EarlyGtLike(inherited):
        def finish(self):
            result = super().finish()
            result["numeric_gt"] = closed()
            return result
    return EarlyGtLike


class RouteTerminationTests(unittest.TestCase):
    def test_r1_retained_early_gt_finish_assigns_into_inner_result(self):
        source = inspect.getsource(importlib.import_module("rl_agent.splitfusion_exact_gt_live_v2.ue"))
        self.assertEqual(source.count('result = super().finish()\n'), 1)
        self.assertEqual(source.count('result["numeric_gt"] = handoff.close()'), 1)
        adapter = inspect.getsource(importlib.import_module("rl_agent.ue_route_b_split_cell_adapter_v1"))
        self.assertIn("cleanup_ok = collector.finish()", adapter)

    def test_r2_finish_result_semantics(self):
        for flag in (True, False):
            result = RC.FinishResult(flag)
            result["numeric_gt"] = {"x": 1}
            self.assertIs(bool(result), flag)
        with self.assertRaises(TypeError):
            RC.FinishResult(1)

    def test_r3_layer_repairs_typeerror_and_preserves_cleanup_flag(self):
        closes = []
        with tempfile.TemporaryDirectory() as tmp:
            broken = _early_gt_like(_FakeBase, lambda: closes.append(1) or {"closed": True})(attempt_dir=tmp)
            with self.assertRaises(TypeError):                     # the retained defect, reproduced
                broken.finish()
            self.assertEqual(closes, [1])                          # handoff closed BEFORE the TypeError
            for flag in (True, False):
                _FakeBase.cleanup_value = flag
                try:
                    cls = _early_gt_like(RC.build_layer(_FakeBase), lambda: {"closed": True})
                    with tempfile.TemporaryDirectory() as tmp2:
                        result = cls(attempt_dir=tmp2).finish()
                        evidence = json.loads((Path(tmp2) / "m3_route/FINISH_RESULT.json").read_text())
                    self.assertIs(bool(result), flag)
                    self.assertEqual(result["numeric_gt"], {"closed": True})
                    self.assertEqual((evidence["inner_type"], evidence["cleanup_ok"]), ("bool", flag))
                finally:
                    _FakeBase.cleanup_value = True

    def _run_route(self, tmp, *, layered, stop_at=None, ticks=40):
        """Real ue_route_b_split_cell_adapter_v1.run_route_b with stubbed CARLA drive/runner."""
        import data_collection.run_route_b_density_loop as density
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        from rl_agent.splitfusion_quality_feedback_probe_v1 import live_cell_child as LC
        attempt, artifacts = Path(tmp) / "attempt", Path(tmp) / "artifacts"
        attempt.mkdir(); artifacts.mkdir()
        inner = RC.build_layer(_FakeBase) if layered else _FakeBase
        collector_cls = LC.build_bounded_collector_class(_early_gt_like(inner, lambda: {"closed": True}),
                                                         transmitted_budget=stop_at or 10**6, safety_timeout_s=900.0,
                                                         artifacts_dir=artifacts)
        route_json = ROOT / C.ROUTE_JSON
        campaign = dict(route_b=dict(route_json=C.ROUTE_JSON, scenario_seed=31, traffic_manager_seed=31))

        class World:
            frame = 100

            def tick(self):
                World.frame += 1
                return World.frame

        def drive(world, vehicle, agent, route_value, collisions, run_args, loop_index, maintain, janitor):
            for _ in range(ticks):
                world.tick()
            return dict(loop_index=1, completed=True, abort_reason="", intervention_count=0, intervention_events=[],
                        watchdog_aborted=False)

        def run(args):
            rows = [density.drive_one_loop_with_traffic(World(), None, None, {}, None, args, 1, None, None)]
            summary = dict(route_id=json.loads(route_json.read_text())["name"], density="traffic_50_50",
                           npc_vehicles_requested=50, npc_pedestrians_requested=50, seed=31, lane_offset_m=-0.5,
                           target_speed_kph=25.0, loops=1, loops_completed=1, status="PASS", cleanup_succeeded=True,
                           watchdog_aborted_any=False, intervention_count_total=0, collision_incident_count_total=0,
                           walker_brake_ticks_total=3, rows=len(rows))
            Path(args.summary_json).write_text(json.dumps(summary))
            return 0
        saved = (P.PassiveSplitCollector, density.drive_one_loop_with_traffic, density.run)
        P.PassiveSplitCollector, density.drive_one_loop_with_traffic, density.run = collector_cls, drive, run
        try:
            return P.run_route_b(campaign=campaign, cell=dict(cell_id="c"), row={}, binding={}, attempt_dir=attempt,
                                 carla_host="127.0.0.1", carla_port=2000, map_api_port=1, feedback_port=1,
                                 edge_evidence_dir=Path(tmp), maximum_loop_sim_s=900.0)
        finally:
            P.PassiveSplitCollector, density.drive_one_loop_with_traffic, density.run = saved

    def test_r4_real_run_route_b_completion_needs_the_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            accepted, detail, collector = self._run_route(tmp, layered=False)
            self.assertFalse(accepted)
            self.assertTrue(detail["error"].startswith("TypeError: 'bool' object does not support item assignment"))
            self.assertFalse(detail["route_summary_persisted"])          # route result discarded by the TypeError
        with tempfile.TemporaryDirectory() as tmp:
            accepted, detail, collector = self._run_route(tmp, layered=True)
            self.assertTrue(accepted)
            self.assertEqual(detail["error"], "")
            self.assertTrue(detail["route_completed"] and detail["route_summary_persisted"]
                            and detail["route_summary_identity_ok"])
            result = {}
            self.assertEqual(RC.accept_route(collector=collector, detail=detail, accepted=accepted, ceiling=C.FRAME_CEILING,
                                             result=result), RC.ROUTE_COMPLETED)
            self.assertEqual(result["collector"]["stop_reason"], RC.ROUTE_COMPLETED)
            self.assertEqual(result["m3_route_completion"]["terminal"], RC.ROUTE_COMPLETED)

    def test_r5_ceiling_stop_is_now_reported_as_itself_and_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            accepted, detail, collector = self._run_route(tmp, layered=True, stop_at=5)
            self.assertFalse(accepted)
            self.assertEqual(detail["error"], "RouteBudgetReached: TRANSMITTED_BUDGET_REACHED")
            with self.assertRaises(RC.IncompleteRoute):
                RC.accept_route(collector=collector, detail=detail, accepted=accepted, ceiling=5, result={})

    def test_r6_accept_route_refuses_every_incomplete_case(self):
        good = dict(route_completed=True, route_summary_persisted=True, route_summary_identity_ok=True, error="")
        def collector(stop="", sent=3000):
            c = types.SimpleNamespace(probe_stop_reason=stop, sent=sent, _persist_probe_summary=lambda: None,
                                      probe_summary=lambda: {})
            return c
        for kwargs in (dict(collector=collector("SAFETY_TIMEOUT_EXPIRED"), detail=good, accepted=True),
                       dict(collector=collector("DECISION_CYCLE_BOUNDARY"), detail=good, accepted=True),
                       dict(collector=collector(), detail=dict(good, route_summary_persisted=False), accepted=True),
                       dict(collector=collector(), detail=dict(good, route_completed=False), accepted=True),
                       dict(collector=collector(), detail=good, accepted=False),
                       dict(collector=collector(sent=C.FRAME_CEILING + 1), detail=good, accepted=True)):
            with self.assertRaises(RC.IncompleteRoute):
                RC.accept_route(ceiling=C.FRAME_CEILING, result={}, **kwargs)

    def test_r7_child_stop_rule_substitution_installs_once_and_restores(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_child_v2 as CHILD
        original = CHILD.run
        with RC.install_child_stop_rule() as installed:
            self.assertIsNot(CHILD.run, original)
            self.assertEqual(CHILD.run.__m3_capacity_substitution__, "R1_CHILD_ROUTE_STOP_RULE")
            self.assertEqual(installed.records[0]["id"], "R1_CHILD_ROUTE_STOP_RULE")
            self.assertIs(CHILD._M3_ROUTE_ACCEPT, RC.accept_route)
        self.assertIs(CHILD.run, original)
        self.assertFalse(hasattr(CHILD, "_M3_ROUTE_ACCEPT"))

    def test_r8_g1_route_branch(self):
        base = lambda **kw: dict(branch=None, stop_reason=kw["child"].get("stop_reason"), sent=3000,
                                 checks=dict(counts_match=True), passed=False)
        g1 = RC.g1_route_branch(base)
        child = dict(stop_reason="ROUTE_COMPLETED", m3_route_completion=dict(terminal="ROUTE_COMPLETED"),
                     route_detail=dict(route_completed=True, route_summary_persisted=True, route_summary_identity_ok=True,
                                       error=""))
        self.assertTrue(g1(child=child, timing={}, analysis={}, transmitted_budget=C.FRAME_CEILING)["passed"])
        bad = dict(child, route_detail=dict(child["route_detail"], route_summary_identity_ok=False))
        self.assertFalse(g1(child=bad, timing={}, analysis={}, transmitted_budget=C.FRAME_CEILING)["passed"])
        ceiling = g1(child=dict(stop_reason="TRANSMITTED_BUDGET_REACHED"), timing={}, analysis={},
                     transmitted_budget=C.FRAME_CEILING)
        self.assertEqual((ceiling["branch"], ceiling["passed"]), ("FRAME_CEILING_STOP_INCOMPLETE_ROUTE", False))

    def test_r9_p3_final_cycle_exception_is_narrow(self):
        def frame(fid, ticket, reward, session="s2"):
            return dict(frame_id=fid, ticket_seq=ticket, reward_requested=reward, session_uuid=session)
        def closed():
            return [frame(1, 0, True, "s1"), frame(2, 0, False, "s1"), frame(3, 0, True), frame(4, 0, False)]
        ok = dict(sessions=["s1", "s2"], frames=closed() + [frame(5, 1, True)])
        self.assertTrue(RC.final_cycle_exception(ok)["applies"])
        middle = dict(sessions=["s1", "s2"], frames=[frame(1, 0, True, "s1"), frame(3, 0, True), frame(4, 0, False)])
        self.assertFalse(RC.final_cycle_exception(middle)["applies"])
        two = dict(sessions=["s1", "s2"], frames=closed() + [frame(5, 1, True), frame(6, 2, True)])
        self.assertFalse(RC.final_cycle_exception(two)["applies"])
        not_last = dict(sessions=["s1", "s2"], frames=closed()[:2] + [frame(5, 1, True)] + closed()[2:])
        not_last["frames"][-1]["frame_id"] = 9
        self.assertFalse(RC.final_cycle_exception(not_last)["applies"])
        gates = dict(P0_POLICY_COVERAGE=True, P3_HOLD_TICKET=False, P4_ACCOUNTING=True)
        wrapped = RC.p3_final_cycle(lambda **kw: dict(gates=dict(gates), verdict="FAILED"))
        out = wrapped(ue=ok, edge={}, map_identity_rows=[], feedback_packets_on_ue_tunnel=[], cleanup_ok=True)
        self.assertEqual((out["gates"]["P3_HOLD_TICKET"], out["verdict"]), (True, "PASSED"))
        out = wrapped(ue=two, edge={}, map_identity_rows=[], feedback_packets_on_ue_tunnel=[], cleanup_ok=True)
        self.assertEqual((out["gates"]["P3_HOLD_TICKET"], out["verdict"]), (False, "FAILED"))


# ------------------------------------------------------------------------------------------------------------ K
class CapacityTests(unittest.TestCase):
    def test_k1_every_registered_replacement_occurs_exactly_once(self):
        for item in C.CAPACITY_SUBSTITUTIONS:
            original, text = CAP.substituted_source(item["id"])
            for old, new in item["replacements"]:
                self.assertEqual(original.count(old), 1)
                self.assertIn(new, text)
            self.assertNotIn("super()", original)

    def test_k2_ue_child_substitutions_raise_the_pose_bound_and_restore(self):
        from rl_agent.splitfusion_live_parity_runtime_v1 import pose
        from rl_agent.splitfusion_exact_gt_live_v2 import ue
        originals = (pose.PoseArchive.__init__, ue.EarlyGtUeHandoff.stage_before_action)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                pose.PoseArchive(Path(tmp) / "a", side="ue", max_frames=C.FRAME_CEILING)
            installed = CAP.install(CAP.UE_CHILD)
            try:
                archive = pose.PoseArchive(Path(tmp) / "b", side="ue", max_frames=C.FRAME_CEILING)
                self.assertEqual(archive.max_frames, C.FRAME_CEILING)
                with self.assertRaises(Exception):
                    pose.PoseArchive(Path(tmp) / "c", side="ue", max_frames=C.FRAME_CEILING + 1)
                self.assertIn(f"len(self._tickets) < {C.FRAME_CEILING}", CAP.substituted_source("C4_UE_TICKETS")[1])
                self.assertEqual(ue.EarlyGtUeHandoff.stage_before_action.__m3_capacity_substitution__, "C4_UE_TICKETS")
                with self.assertRaises(CAP.CapacitySubstitutionError):
                    CAP.install(("C5_UE_POSE_ARCHIVE",))
            finally:
                installed.uninstall()
        self.assertEqual((pose.PoseArchive.__init__, ue.EarlyGtUeHandoff.stage_before_action), originals)

    def test_k3_prep_and_service_substitutions_keep_signatures(self):
        for ids in (CAP.PREP_WORKER, CAP.GT_SERVICE):
            targets = [CAP._resolve(CAP._entry(i)["target"]) for i in ids]
            before = [inspect.signature(t[3]) for t in targets]
            installed = CAP.install(ids)
            try:
                after = [inspect.signature(getattr(t[1], t[2])) for t in targets]
                self.assertEqual(before, after)
            finally:
                installed.uninstall()
            self.assertEqual([getattr(t[1], t[2]) for t in targets], [t[3] for t in targets])
            self.assertTrue(installed.document()["restored"])

    def test_k4_ceiling_limits_archive_and_lifecycle(self):
        from rl_agent.splitfusion_live_gt_handoff_v1 import archive as AR
        self.assertEqual(C.ARCHIVE_JOB_CAP, AR.MAX_CUMULATIVE_JOBS)
        self.assertLessEqual(C.FRAME_CEILING, C.ARCHIVE_JOB_CAP)       # ~1 archive job per transmitted frame
        self.assertEqual(C.limits(C.FRAME_CEILING)["max_frames"], C.FRAME_CEILING)
        with self.assertRaises(ValueError):
            C.limits(1000)
        from . import host
        self.assertEqual(host.LIFECYCLE, {C.FRAME_CEILING: dict(safety_timeout_s=900.0, child_timeout_s=2400.0)})
        from rl_agent.splitfusion_run5_live_avo_v1 import host_capture as H
        self.assertEqual(inspect.getsource(H.launch).count("deadline = time.monotonic() + 1000"), 1)

    def test_k5_unsubstituted_service_refuses_the_ceiling_at_ready(self):
        from rl_agent.splitfusion_live_gt_handoff_v1 import archive as AR
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp).resolve())
            with self.assertRaisesRegex(Exception, "archive frame bound differs"):
                AR._configuration([root], 4, 1 << 20, 4 * C.FRAME_CEILING, {})       # retained 4 x max_frames sizing
            self.assertEqual(AR._configuration([root], 4, 1 << 20, C.ARCHIVE_JOB_CAP, {})["max_frames"], 4000)
        original, text = CAP.substituted_source("C3_SERVICE_ARCHIVE_SIZING")
        self.assertIn('max_frames=4 * self.limits["max_frames"]', original)
        self.assertIn(f"max_frames={C.ARCHIVE_JOB_CAP}", text)


# ------------------------------------------------------------------------------------------------------------ M
class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("/404"):
            self.send_response(404); self.end_headers(); return
        frame = int(self.path.rsplit("/", 1)[1])
        body = json.dumps(dict(status="INSTALLED", record=dict(stream_id="s", frame_id=frame, objects=[
            dict(type="Vehicle", location=dict(x=1.5, y=2.5, z=0.0), dimensions=dict(length=4, width=2, height=1.5),
                 score=0.9)]))).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class MapTests(unittest.TestCase):
    def test_m1_journal_tees_the_retained_fetch_exactly(self):
        from rl_agent import ue_route_b_split_cell_adapter_v1 as P
        server = HTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                layer = RC.build_layer(type("Base", (), dict(
                    __init__=lambda self, **kw: setattr(self, "attempt_dir", kw["attempt_dir"]),
                    _fetch_installed_record=P.PassiveSplitCollector._fetch_installed_record)))
                collector = layer(attempt_dir=tmp)
                collector.stream_id, collector.map_api_port = "s", server.server_address[1]
                MJ.install_tee()
                try:
                    outcome, predictions = collector._fetch_installed_record(42)
                    missing = collector._fetch_installed_record(404)
                    other = []
                    t = threading.Thread(target=lambda: other.append(P.urllib.request.urlopen(
                        f"http://127.0.0.1:{server.server_address[1]}/x/7").read()))
                    t.start(); t.join()
                finally:
                    MJ.uninstall_tee()
                summary = collector._m3_journal.close()
                rows = [json.loads(line) for line in (Path(tmp) / "m3_map/installed_map_records.jsonl").read_text().splitlines()]
                latest = json.loads((Path(tmp) / "m3_map/map_state_latest.json").read_text())
            self.assertEqual(outcome, "OK")
            self.assertEqual(predictions[0]["world_x"], 1.5)
            self.assertEqual(missing, ("NOT_FOUND", None))
            self.assertEqual([r["outcome"] for r in rows], ["OK", "NOT_FOUND"])
            self.assertEqual(rows[0]["map_document"]["record"]["objects"][0]["location"]["x"], 1.5)
            self.assertEqual(latest["frame_id"], 42)
            self.assertEqual(summary["counts"]["installed"], 1)
            self.assertEqual(summary["counts"]["write_errors"], 0)
            self.assertIsNone(MJ._STATE["original"])                       # tee uninstalled
            self.assertEqual(len(other[0]) > 0, True)                         # other threads untouched, not journaled
        finally:
            server.shutdown()
            server.server_close()

    def test_m2_map_freshness_horizon_is_500ms_capture_age_not_the_170ms_deadline(self):
        config = json.loads((ROOT / C.LIVE_CONFIG).read_bytes())
        self.assertEqual((config["cell"]["ack_timeout_ms"], config["cell"]["service_deadline_ms"]), (500, 100))
        adapter = inspect.getsource(importlib.import_module("rl_agent.splitfusion_direct_edge_map_v1.adapter_direct_v1"))
        self.assertIn('"--direct-processing-horizon-ms",\n        str(float(campaign["cell"]["ack_timeout_ms"])),', adapter)
        ingest = inspect.getsource(importlib.import_module("rl_agent.splitfusion_direct_edge_map_v1.map_ingest"))
        self.assertIn("age_s = ingest_at - capture_ns / 1_000_000_000.0", ingest)
        self.assertIn("if age_s > self.processing_horizon_s:", ingest)
        self.assertIn("if capture_ns <= newest_capture:", ingest)
        for module in ("rl_agent.splitfusion_direct_edge_map_v1.map_ingest",
                       "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2"):
            self.assertNotIn("170", inspect.getsource(importlib.import_module(module)))

    def test_m3_prediction_older_than_170ms_is_installed_and_older_than_500ms_is_stale(self):
        from rl_agent.splitfusion_direct_edge_map_v1.tests import test_direct_edge_map_v1 as T
        with tempfile.TemporaryDirectory() as tmp:
            installer = T.RecordingInstaller()
            service = T.make_service(Path(tmp), installer)
            try:
                now = time.time()
                edge = service.ingest(T.make_update(frame_id=7, capture_offset_s=-0.4999, now=now), ingest_at=now,
                                      emit=False)
                late = service.ingest(T.make_update(frame_id=8, capture_offset_s=-0.30, now=now), ingest_at=now, emit=False)
                stale = service.ingest(T.make_update(frame_id=10, capture_offset_s=-0.51, now=now), ingest_at=now,
                                       emit=False)
                older = service.ingest(T.make_update(frame_id=6, capture_offset_s=-0.45, now=now), ingest_at=now,
                                       emit=False)
            finally:
                service.close()
        self.assertEqual(late["outcome"], T.OUTCOME_RESULT_INSTALLED)        # 300 ms after capture: map-eligible
        self.assertEqual(edge["outcome"], T.OUTCOME_RESULT_INSTALLED)
        self.assertEqual(stale["outcome"], T.OUTCOME_STALE_BEFORE_MAP)
        self.assertEqual(older["outcome"], T.OUTCOME_SUPERSEDED_PENDING)     # newer capture already installed
        self.assertEqual([d["frame_id"] for d in installer.installed], [7, 8])

    def test_m4_edge_publishes_before_and_independently_of_feedback(self):
        source = inspect.getsource(importlib.import_module("rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2"))
        loop = source[source.index("    def process_loop() -> None:"):source.index("    ready = Path(args.ready_file)")]
        self.assertLess(loop.index("publisher.publish(processed.update)"), loop.index("evaluator.submit(processed.evaluation)"))
        self.assertNotIn("read_ground_truth", loop)
        early = inspect.getsource(importlib.import_module("rl_agent.splitfusion_exact_gt_live_v2.edge").EarlyGtProcessorV1)
        self.assertIn("evaluator.submit(ticket)                       # bounded put_nowait; never blocks compute", early)
        self.assertLess(early.index("evaluator.submit(ticket)"), early.index("MP.build_run4_map_update("))
        edge_source = inspect.getsource(importlib.import_module("rl_agent.splitfusion_exact_gt_live_v2.edge").EarlyGtEvaluatorV1)
        self.assertNotIn("def submit", edge_source)     # retained base put_nowait / overflow branch applies


class RadioObserverTests(unittest.TestCase):
    def test_m5_radio_observer_timestamps_stats_and_copies_logs_before_removal(self):
        from . import radio_observer as RO
        with tempfile.TemporaryDirectory() as tmp:
            state, out = Path(tmp) / "00_FAVORABLE_STABLE", Path(tmp) / "capture"
            state.mkdir()
            (state / "gnb.log").write_text("boot\n")
            (state / "ue.log").write_text("ue\n")
            observer = RO.RadioObserver(out / "m3_radio")
            observer.start(state)
            with (state / "gnb.log").open("a") as handle:
                handle.write("UE b251: ulsch_rounds 28/0/0/0, ulsch_errors 0, MCS (0) 28 NPRB 5 SNR 22.8 (+7.8) dB\n")
            time.sleep(0.8)
            summary = observer.finish()
            rows = [json.loads(l) for l in (out / "m3_radio/gnb_stats_observed.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertIn("SNR 22.8", rows[0]["line"])
            self.assertGreater(rows[0]["observed_raw_ns"], 0)
            self.assertEqual(sorted(summary["copied"]), ["gnb.log", "ue.log"])
            self.assertTrue((out / "m3_radio/radio_state_copy/gnb.log").read_text().endswith("dB\n"))


# ------------------------------------------------------------------------------------------------------------ E
class EdgeGuardTests(unittest.TestCase):
    def setUp(self):
        from rl_agent.splitfusion_exact_gt_live_v2 import edge as EG
        self.EG = EG
        self.saved = (EG.EarlyGtEvaluatorV1._binding, EG.EarlyGtEvaluatorV1._instance)
        EG.EarlyGtEvaluatorV1._binding = dict(socket_path="/nonexistent", capability_path="/nonexistent",
                                              connect_timeout_s=0.1)

    def tearDown(self):
        self.EG.EarlyGtEvaluatorV1._binding, self.EG.EarlyGtEvaluatorV1._instance = self.saved

    def _process(self, evaluator_cls, send):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase3_continuous_execution_v2 import CONTRACT
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_phase6_live_integration_v2 import (
            SPEC, FakeActor, Pipeline)
        from rl_agent.splitfusion_single_gpu_profiler_v1.test_early_submit_v1 import GT, fake_compute
        evaluator = evaluator_cls(spec=SPEC, send=send, match_distance_m=3.0, gt_timeout_s=0.5, queue_depth=1,
                                  read_ground_truth=lambda **kw: GT)
        evaluator._queue.put_nowait(object())            # the next put is queue.Full -> retained overflow branch
        processor = self.EG.EarlyGtProcessorV1(contract=CONTRACT, run_id="run", cell_id="cell", service_deadline_s=0.1,
                                               ack_timeout_s=0.5, compute=fake_compute)
        wire = Pipeline(FakeActor(mode_id=9, q_e4=4321)).step().wire
        return evaluator, processor, wire, MP

    def test_e1_retained_overflow_send_error_escapes_and_would_skip_publication(self):
        def failing(payload):
            raise OSError("network unreachable")
        evaluator, processor, wire, _ = self._process(self.EG.EarlyGtEvaluatorV1, failing)
        with self.assertRaises(OSError):
            processor.process(wire, edge_timing={})
        self.assertEqual(evaluator.counters["queue_overflow"], 1)
        loop = inspect.getsource(importlib.import_module("rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2"))
        self.assertIn('bump("edge_processing_failed")', loop)       # process() exception -> no publication

    def test_e2_guarded_overflow_send_error_still_publishes_completed_prediction(self):
        from .edge_guard import build_guarded_evaluator
        def failing(payload):
            raise OSError("network unreachable")
        evaluator, processor, wire, MP = self._process(build_guarded_evaluator(self.EG.EarlyGtEvaluatorV1), failing)
        processed = processor.process(wire, edge_timing={})
        MP.validate_run4_map_update(processed.update)
        self.assertEqual(evaluator.counters["queue_overflow"], 1)
        self.assertEqual(evaluator.counters["m3_submit_guarded_errors"], 1)
        self.assertEqual(evaluator.m3_guarded_errors[0]["error"], "OSError: network unreachable")

    def test_e3_guarded_overflow_normal_send_is_bounded_and_unchanged(self):
        from .edge_guard import build_guarded_evaluator
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        sent = []
        evaluator, processor, wire, MP = self._process(build_guarded_evaluator(self.EG.EarlyGtEvaluatorV1), sent.append)
        processed = processor.process(wire, edge_timing={})
        MP.validate_run4_map_update(processed.update)
        self.assertEqual(W.decode_feedback(sent[0])[0].kind, "INFRASTRUCTURE_FAULT")   # retained fault, as before
        self.assertEqual(evaluator.counters["m3_submit_guarded_errors"], 0)
        self.assertLess(evaluator.counters["m3_overflow_branch_max_ns"], 50_000_000)


# ------------------------------------------------------------------------------------------------------------ P
class PlanTests(unittest.TestCase):
    def test_p1_retention_plan_spans_the_typical_loop_outcome_blind(self):
        plan = C.retention_plan(C.FRAME_CEILING)
        self.assertEqual(plan["selection_inputs"]["planned_frames"], C.RETENTION_PLANNED_FRAMES)
        self.assertEqual(len(plan["ordinals"]), 100)
        self.assertEqual(plan["ordinals"][-1], 1492)
        self.assertTrue({"latency", "quality", "action", "deadline_outcome"} <= set(plan["independent_of"]))

    def test_p2_lineage_binds_schedule_route_capacity_and_evidence(self):
        a = C.lineage(ROOT, "a" * 64, "b" * 64, C.FRAME_CEILING, "candidate", "v2", "c" * 64)
        self.assertEqual(len(a), 64)
        with self.assertRaises(ValueError):
            C.retention_plan(1000)
        self.assertEqual(C.route_registration(ROOT)["completion"]["runner_criteria"]["waypoints_expected"], 19)
        self.assertEqual([s["id"] for s in C.capacity_registration()["substitutions"]],
                         ["C1_PREP_DEFERRED_ENGINE", "C2_PREP_PRODUCER_SOURCE", "C3_SERVICE_ARCHIVE_SIZING",
                          "C4_UE_TICKETS", "C5_UE_POSE_ARCHIVE"])


if __name__ == "__main__":
    unittest.main()
