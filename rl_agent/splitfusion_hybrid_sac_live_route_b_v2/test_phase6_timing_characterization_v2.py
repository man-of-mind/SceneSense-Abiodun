"""Addendum-11 offline tests: timing-characterization analysis and CARLA trace."""

from __future__ import annotations

import csv
import json
import tempfile
import types
import unittest
from pathlib import Path

from . import phase6_timing_characterization_v2 as TC
from . import phase6_ue_runtime_v2 as U

MS = 1_000_000
BRIDGE = 10_000_000_000          # wall = raw + BRIDGE (ns)
SESSION = "s-1"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) or ["frame_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_cell(root: Path, *, fb_ms=(150.0, 190.0), duplicate=False, drop_terminal=False,
               bad_identity=False) -> Path:
    """Two reward tickets (k_min=2) on frames 10/14, holds 12/16, fallback 18."""
    kinds = {10: "POLICY_DECISION", 12: "POLICY_HOLD", 14: "POLICY_DECISION",
             16: "POLICY_HOLD", 18: "FALLBACK"}
    decisions, idents, tickets, ingest, terminals, per_frame = [], [], [], [], [], []
    evaluations, feedback, resolutions = [], [], []
    seq = {10: 0, 12: 0, 14: 1, 16: 1}
    for i, (f, kind) in enumerate(sorted(kinds.items())):
        t0 = 1_000 * MS + i * 100 * MS                  # frame plan instant (raw)
        rgb = t0 - 25 * MS
        stages = {"rgb_receipt_raw_ns": rgb, "si_p40_end_raw_ns": t0,
                  "input_7ch_start_raw_ns": t0 + 1 * MS, "front_start_raw_ns": t0 + 9 * MS,
                  "front_end_raw_ns": t0 + 27 * MS, "first_packet_send_raw_ns": t0 + 28 * MS,
                  "last_packet_send_raw_ns": t0 + 29 * MS}
        rec = {"frame_id": f, "kind": kind, "mode_id": 7 if kind != "FALLBACK" else 11,
               "q_e4": 6800 if kind != "FALLBACK" else 9800, "tensor_seq": i,
               "capture": {"domain": "WALL", "ns": rgb + BRIDGE + 3_000}, "stages": stages}
        if kind in ("POLICY_DECISION", "FALLBACK"):
            rec["action_open"] = {"domain": "CLOCK_MONOTONIC_RAW", "ns": t0 + 200_000}
        decisions.append(rec)
        ident = {"frame_id": f, "frame_kind": kind, "session_uuid": SESSION,
                 "decision_seq": seq.get(f, (1 << 64) - 1), "ticket_seq": seq.get(f, (1 << 64) - 1),
                 "reward_requested": kind == "POLICY_DECISION", "mode_id": rec["mode_id"],
                 "q_e4": rec["q_e4"], "tensor_seq": i}
        idents.append({"frame_id": f, "run4_identity": ident})
        reasm = t0 + 90 * MS
        ingest.append({"frame_id": f, "outcome": "RESULT_INSTALLED",
                       "edge_reassembly_complete_wall_s": (reasm + BRIDGE) / 1e9,
                       "edge_compute_start_wall_s": (reasm + 1 * MS + BRIDGE) / 1e9,
                       "edge_publish_start_wall_s": (reasm + 61 * MS + BRIDGE) / 1e9,
                       "map_install_at": (reasm + 64 * MS + BRIDGE) / 1e9})
        terminals.append({"frame_id": f, "terminal": "True", "outcome": "RESULT_INSTALLED",
                          "superseded_by_frame_id": ""})
        per_frame.append({"frame_id": f, "prepare_status": "SENT",
                          "payload_bytes": 312000 if kind != "FALLBACK" else 6800,
                          "payload_chunks": 25 if kind != "FALLBACK" else 1})
        tickets.append({"frame_id": f, "queue_class": "HIGH" if kind == "POLICY_DECISION" else "LOW",
                        "objects_write_end_wall_ns": t0 + 48 * MS + BRIDGE})
        if kind == "POLICY_DECISION":
            k = 0 if f == 10 else 1
            ao = rec["action_open"]["ns"]
            receipt = ao + int(fb_ms[k] * MS)
            evaluations.append({"frame_id": f, "run4_identity": dict(ident),
                                "enqueued_wall_ns": ao + 140 * MS + BRIDGE,
                                "emit_wall_ns": receipt - 2 * MS + BRIDGE,
                                "kind": "DELIVERED_SUCCESS"})
            late = fb_ms[k] > 170.0
            feedback.append({"frame_id": f, "class": "LATE_ORPHAN" if late else "ACCEPTED",
                             "kind": "DELIVERED_SUCCESS", "q_perc": 0.4,
                             "receipt": {"ns": receipt}})
            resolutions.append({"identity": {"session_uuid": SESSION, "decision_seq": k},
                                "terminal": "TIMEOUT" if late else "SUCCESS",
                                "learning_included": True, "reward": -1.0 if late else 0.4})
    if bad_identity:
        evaluations[0]["run4_identity"]["tensor_seq"] = 99
    if duplicate:
        feedback.append(dict(feedback[0]))
    if drop_terminal:
        terminals = terminals[:-1]
    ue = {"decisions": decisions, "transmitted_identities": idents,
          "gt_objects": {"tickets": tickets}, "feedback_rows": feedback,
          "resolutions": resolutions, "unresolved_tickets_at_close": 0, "faulted": None,
          "carla_trace": {"ticks": [[f, 2 * i, d["stages"]["rgb_receipt_raw_ns"] - 20 * MS]
                                    for i, (f, d) in enumerate(zip(sorted(kinds), decisions))],
                          "rgb": [[d["frame_id"], d["stages"]["rgb_receipt_raw_ns"]]
                                  for d in decisions]}}
    cell = root / "cell"
    (cell / "run4_phase6").mkdir(parents=True)
    (cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").write_text(json.dumps(ue))
    (cell / "run4_phase6/edge_report.json").write_text(json.dumps(
        {"evaluations": evaluations, "evaluator": {"excluded": 0}}))
    _write_csv(cell / "map_feedback.csv", terminals)
    _write_csv(cell / "direct_edge_map/direct_map_ingest.csv", ingest)
    _write_csv(cell / "per_frame_metrics.csv", per_frame)
    return cell


class TimingCharacterizationTest(unittest.TestCase):
    def analyze(self, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return TC.analyze(build_cell(Path(tmp.name), **kw))

    def test_two_independent_deadlines_and_next_eligible_decision(self) -> None:
        r = self.analyze(fb_ms=(150.0, 190.0))
        self.assertEqual(r["policy_deadline_action_open_to_feedback_le_170"]["count"], 1)
        self.assertEqual(r["policy_deadline_action_open_to_feedback_le_170"]["n"], 2)
        # capture precedes action-open by 25.2 ms: 175.2 and 215.2 ms
        kpi = r["system_kpi_capture_to_feedback_le_200"]
        self.assertEqual((kpi["count"], kpi["n"]), (1, 2))
        t0, t1 = r["tickets"]
        self.assertEqual((t0["next_eligible_frame"], t0["next_eligible_kind"]), (14, "POLICY_DECISION"))
        self.assertAlmostEqual(t0["action_open_to_next_eligible_ms"], 199.8)
        self.assertTrue(t0["feedback_before_next_eligible"])
        self.assertEqual(t1["next_eligible_frame"], 18)
        self.assertTrue(t1["feedback_before_next_eligible"])     # 190 < 199.8
        self.assertEqual(t0["tensors_carried"], 2)
        self.assertEqual(r["outcomes"]["timeouts"], 1)
        self.assertEqual(r["outcomes"]["late_feedback"], 1)
        self.assertTrue(r["integration"]["passed"], r["integration"])
        self.assertLess(r["clock_bridge"]["spread_us"], 1.0)

    def test_feedback_after_the_next_eligible_plan_instant_is_counted(self) -> None:
        r = self.analyze(fb_ms=(205.0, 150.0))
        self.assertFalse(r["tickets"][0]["feedback_before_next_eligible"])
        self.assertEqual(r["feedback_before_next_eligible_decision"]["count"], 1)

    def test_missing_feedback_is_a_measured_miss_not_censored(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cell = build_cell(Path(tmp.name))
        path = cell / "run4_phase6/PHASE6_UE_EVIDENCE.json"
        ue = json.loads(path.read_text())
        ue["feedback_rows"] = ue["feedback_rows"][:1]
        path.write_text(json.dumps(ue))
        r = TC.analyze(cell)
        d = r["policy_deadline_action_open_to_feedback_le_170"]
        self.assertEqual((d["count"], d["n"], d["undetermined"]), (1, 2, 0))
        self.assertEqual(r["distributions_ms"]["action_open_to_feedback"]["missing"], 1)
        self.assertFalse(r["tickets"][1]["feedback_before_next_eligible"])

    def test_stage_decomposition_is_uncensored_for_every_sent_frame(self) -> None:
        r = self.analyze()
        self.assertEqual(r["counts"]["sent_frames"], 5)
        f = {x["frame_id"]: x for x in r["frames"]}
        self.assertAlmostEqual(f[10]["capture_to_anchor_ms"], 25.2)
        self.assertAlmostEqual(f[12]["capture_to_anchor_ms"], 25.0)      # hold: plan instant
        self.assertEqual(f[12]["anchor"], "plan_instant")
        # edge wall stamps carry the 3-us capture-pair bridge error by construction
        self.assertAlmostEqual(f[10]["last_datagram_to_uplink_complete_ms"], 61.0, delta=0.01)
        self.assertAlmostEqual(f[10]["edge_queue_ms"], 1.0, delta=0.01)
        self.assertAlmostEqual(f[10]["edge_compute_ms"], 60.0, delta=0.01)
        self.assertAlmostEqual(f[10]["capture_to_map_install_ms"], 179.0, delta=0.01)
        self.assertEqual(f[10]["action_reuse_count"], 2)
        self.assertEqual(f[10]["payload_bytes"], 312000)
        self.assertEqual(r["distributions_ms"]["capture_to_map_install"]["n"], 5)

    def test_percentiles_are_nearest_rank(self) -> None:
        values = list(range(1, 101))
        s = TC.summary(values + [None])
        self.assertEqual((s["p50"], s["p90"], s["p95"], s["p99"], s["max"]), (50, 90, 95, 99, 100))
        self.assertEqual(s["missing"], 1)

    def test_gpu_overlap_between_front_edge_and_carla(self) -> None:
        r = self.analyze()
        g = {x["frame_id"]: x for x in r["gpu_overlap"]}
        self.assertAlmostEqual(g[10]["front_ms"], 18.0)
        self.assertIn("front_overlap_edge_ms", g[10])
        self.assertIn("edge_overlap_carla_ms", g[10])
        # frame 12's front (t0+9..27 = 1109..1127 ms) overlaps frame 10's edge tail (1091..1151)
        self.assertAlmostEqual(g[12]["front_overlap_edge_ms"], 18.0, delta=0.01)

    def test_integration_failures_are_detected(self) -> None:
        self.assertIn(10, self.analyze(duplicate=True)["integration"]["duplicate_feedback_frames"])
        self.assertFalse(self.analyze(duplicate=True)["integration"]["passed"])
        self.assertEqual(self.analyze(drop_terminal=True)["integration"]["frames_without_terminal"], [18])
        self.assertEqual(self.analyze(bad_identity=True)["integration"]
                         ["evaluation_identity_mismatches"], [10])


class CarlaTraceTest(unittest.TestCase):
    def test_collector_records_every_tick_and_rgb_without_changing_behaviour(self) -> None:
        calls = []

        class Base:
            def on_world_tick(self, frame_id, route_tick):
                calls.append(("tick", frame_id))

            def _on_rgb(self, image):
                calls.append(("rgb", image.frame))

        cls = U.build_run4_collector_class(Base)
        host = types.SimpleNamespace(
            _run4_carla_trace={"ticks": [], "rgb": []}, _run4_lock=__import__("threading").Lock(),
            _run4_rgb_raw=__import__("collections").OrderedDict(),
            live=types.SimpleNamespace(infrastructure_fault=None), failures=[])
        # a real instance (super() needs one) carrying only the fake host's attributes
        obj = cls.__new__(cls)
        obj.__dict__.update(host.__dict__)
        obj.on_world_tick(7, 3)
        obj._on_rgb(types.SimpleNamespace(frame=7))
        self.assertEqual(calls, [("tick", 7), ("rgb", 7)])
        self.assertEqual([t[:2] for t in obj._run4_carla_trace["ticks"]], [(7, 3)])
        self.assertEqual(obj._run4_carla_trace["rgb"][0][0], 7)
        self.assertEqual(obj._run4_rgb_raw[7], obj._run4_carla_trace["rgb"][0][1])


class AddendumElevenTest(unittest.TestCase):
    def test_addendum_is_prospective_and_binds_prior_evidence(self) -> None:
        import hashlib

        doc = json.loads((Path(__file__).parent / "phase6_timing_characterization_addendum_11.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual(doc["status"], "REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE")
        self.assertEqual(doc["classification"], "TIMING_CHARACTERIZATION_NOT_QUALIFICATION")
        self.assertFalse(doc["run"]["timeout_ms_changed"])
        self.assertEqual(doc["run"]["runs"], 1)
        self.assertIn("--transmitted-budget 300", doc["run"]["command"])
        self.assertNotIn("--stop-after-decisions", doc["run"]["command"])
        self.assertEqual(len(doc["evidence_sha256_v8_v9_v10"]), 167)
        root = Path(__file__).resolve().parents[2]
        for rel, digest in doc["evidence_sha256_v8_v9_v10"].items():
            path = root / rel
            if path.is_file():             # evidence is kept off Git; verify when present
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, rel)
        self.assertEqual(TC.POLICY_DEADLINE_MS, 170.0)
        self.assertEqual(TC.SYSTEM_KPI_MS, 200.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
