"""Focused CPU tests: classifier (real captured errors + mutations), adoption/installation guards, bounded
diagnostics, and the actor-visible feedback/controller semantics of a GT-unavailable outcome.

    cd <checkout>; env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
      PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m unittest rl_agent.splitfusion_gt_geometry_containment_v1.test_containment
"""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_avo_tilt_v1 import switch
from rl_agent.splitfusion_exact_gt_service_proto_v2 import freeze_codec as FZ, parity_common as P
from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as R
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

from . import classify as CL, core as K, engine as E
from .fixtures import FixtureCorpus, S1_TILT

PLAN = {7: ("TILT", "146", *S1_TILT, "S1_TILT_TARGET"),
        19: ("RELOCATE_TILT", "101", 20.0, 0.0, "COMPETITOR"),
        20: ("BBOX_ROTATION", "101", 5.0, "BBOX")}


def capture(corpus, index):
    driver = P.DeferredDriver(corpus)                 # unchanged engine: the error is the production error
    try:
        driver.unbound(corpus.cases[index])
    except LiveGtContractError as error:
        return error
    finally:
        driver.engine.close(abandon=True)             # the producer owner is process-wide: release it
    raise AssertionError(f"case {index} did not raise")


class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        P.environment_ok()
        switch.install()
        cls.corpus = FixtureCorpus(PLAN)
        cls.target = capture(cls.corpus, 7)
        cls.competitor = capture(cls.corpus, 19)
        cls.bbox = capture(cls.corpus, 20)

    @classmethod
    def tearDownClass(cls):
        switch.uninstall()

    def test_real_target_refusal_recognised(self):
        record, why = CL.recognise(self.target)
        self.assertIsNone(why)
        self.assertEqual(record["reason"], "GT_GEOMETRY_ENVELOPE_TARGET")
        self.assertEqual(record["offender"]["gt_actor_id"], "146")
        self.assertEqual(record["offender"]["footprint_tilt_m"], 0.03709273558898407)   # = S1 actor 106 value
        self.assertEqual(record["message"], str(self.target))

    def test_real_competitor_refusal_recognised(self):
        record, why = CL.recognise(self.competitor)
        self.assertIsNone(why)
        self.assertEqual(record["reason"], "GT_GEOMETRY_ENVELOPE_POSSIBLE_COMPETITOR")
        self.assertEqual(record["offender"]["role"], "STRUCTURAL_IGNORE")
        self.assertEqual(record["offender"]["structural_reasons"], ["distance_gt_40m"])

    def test_unrelated_contract_error_not_recognised(self):
        self.assertIn("bounding-box rotation", str(self.bbox))
        self.assertEqual(CL.recognise(self.bbox), (None, "RAISE_SITE"))

    def _mutated(self, error, mutate):
        saved = dict(diagnostics=copy.deepcopy(error.geometry_diagnostics), args=error.args,
                     cause=error.__cause__, context=error.__context__, cls=error.__class__, tb=error.__traceback__)
        try:
            mutate(error)
            return CL.recognise(error)
        finally:
            error.geometry_diagnostics, error.args = saved["diagnostics"], saved["args"]
            error.__cause__, error.__context__, error.__class__ = saved["cause"], saved["context"], saved["cls"]
            error.__traceback__ = saved["tb"]

    def test_malformed_or_inconsistent_evidence_never_recognised(self):
        class Sub(LiveGtContractError):
            pass

        def diag(change):
            def apply(error):
                d = copy.deepcopy(error.geometry_diagnostics)
                change(d)
                error.geometry_diagnostics = d
            return apply
        cases = {
            "DIAGNOSTICS_SHAPE": [lambda e: setattr(e, "geometry_diagnostics", None),
                                  diag(lambda d: d.update(extra=1))],
            "STATUS": [diag(lambda d: d.update(status="TILTED_SCOREABLE_TARGET"))],
            "OFFENDER_SHAPE": [diag(lambda d: d["offender"].pop("role")), diag(lambda d: d["offender"].update(x=1))],
            "OFFENDER_ID": [diag(lambda d: d["offender"].update(gt_actor_id=146)),
                            diag(lambda d: d["offender"].update(gt_actor_id=""))],
            "OFFENDER_NUMBERS": [diag(lambda d: d["offender"].update(pitch_deg=math.nan)),
                                 diag(lambda d: d["offender"].update(roll_deg=8)),
                                 diag(lambda d: d["offender"].update(footprint_tilt_m=math.inf))],
            "INSIDE_ENVELOPE": [diag(lambda d: d["offender"].update(footprint_tilt_m=0.03)),
                                diag(lambda d: d["offender"].update(footprint_tilt_m=0.0299))],
            "STRUCTURAL_REASONS": [diag(lambda d: d["offender"].update(structural_reasons=("x",)))],
            "ROLE_STATUS_MISMATCH": [diag(lambda d: d["offender"].update(role="STRUCTURAL_IGNORE")),
                                     diag(lambda d: d["offender"].update(structural_reasons=["distance_gt_40m"]))],
            "MESSAGE_MISMATCH": [diag(lambda d: d["offender"].update(footprint_tilt_m=0.04)),
                                 lambda e: setattr(e, "args", (str(e) + " ",))],
            "CHAINED": [lambda e: setattr(e, "__cause__", ValueError("cleanup")),
                        lambda e: setattr(e, "__context__", ValueError("context"))],
            "TYPE": [lambda e: setattr(e, "__class__", Sub)],
            "RAISE_SITE": [lambda e: setattr(e, "__traceback__", None)],
        }
        for why, mutations in cases.items():
            for mutate in mutations:
                with self.subTest(why=why):
                    self.assertEqual(self._mutated(self.target, mutate), (None, why))
        self.assertIsNone(CL.recognise(self.target)[1])          # restored after every mutation

    def test_competitor_role_rules(self):
        def to_target(e):
            d = copy.deepcopy(e.geometry_diagnostics)
            d["status"] = CL.STATUS_TARGET
            e.geometry_diagnostics = d
        self.assertEqual(self._mutated(self.competitor, to_target), (None, "ROLE_STATUS_MISMATCH"))

    def test_foreign_raise_site_with_identical_evidence(self):
        try:
            error = LiveGtContractError(*self.target.args)
            error.geometry_diagnostics = copy.deepcopy(self.target.geometry_diagnostics)
            raise error
        except LiveGtContractError as forged:
            self.assertEqual(CL.recognise(forged), (None, "RAISE_SITE"))

    def test_classifier_never_raises(self):
        class Hostile(LiveGtContractError):
            pass
        self.assertEqual(CL.recognise(object()), (None, "TYPE"))
        self.assertEqual(CL.recognise(ValueError("x")), (None, "TYPE"))
        self.assertEqual(CL.recognise(Hostile("x")), (None, "TYPE"))


class GuardTest(unittest.TestCase):
    def test_contained_engine_is_never_constructed(self):
        with self.assertRaises(E.ContainmentError):
            E.ContainedEngine()

    def test_adopt_requires_exact_fresh_engine(self):
        with self.assertRaises(E.ContainmentError):
            E.adopt(object())

    def test_install_requires_lifetime_repair_and_restores(self):
        self.assertIs(PC.PrepCore, R.ORIGINAL_PREP_CORE)
        with self.assertRaises(RuntimeError):
            with K.install(None):
                pass
        with R.install(None):
            with K.install("/nonexistent/trace"):
                self.assertIs(PC.PrepCore, K.PrepCoreContained)
                self.assertTrue(issubclass(PC.PrepCore, R.PrepCoreLifetime))
            self.assertIs(PC.PrepCore, R.PrepCoreLifetime)
            self.assertIsNone(K.PrepCoreContained.diagnostics_directory)
        self.assertIs(PC.PrepCore, R.ORIGINAL_PREP_CORE)


class DiagnosticsTest(unittest.TestCase):
    def _fd(self, payload):
        fd, size = FZ.seal([memoryview(payload)])
        return fd

    def test_bounded_records_and_sources(self):
        payload = os.urandom(4096)
        with tempfile.TemporaryDirectory() as tmp:
            d = K.Diagnostics(tmp)
            fd = self._fd(payload)
            try:
                for n in range(K.MAX_RECORDS + 40):
                    d.record(dict(descriptor="{}"), fd, dict(status=CL.STATUS_TARGET, n=n), job=n)
            finally:
                os.close(fd)
            root = Path(tmp) / "geometry_refusal_v1"
            lines = (root / "REFUSALS.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), K.MAX_RECORDS)
            self.assertEqual(sorted(p.name for p in root.iterdir() if p.is_dir()),
                             [f"source_{i:03d}" for i in range(1, K.MAX_SOURCES + 1)])
            for i in range(1, K.MAX_SOURCES + 1):
                ready = json.loads((root / f"source_{i:03d}/READY.json").read_text())
                self.assertEqual((root / f"source_{i:03d}/SOURCE.bin").read_bytes(), payload)
                self.assertTrue(ready["complete"])
            counters = json.loads((root / "COUNTERS.json").read_text())
            self.assertEqual((counters["refusals"], counters["records_written"], counters["records_dropped"],
                              counters["sources_written"], counters["write_errors"]),
                             (K.MAX_RECORDS + 40, K.MAX_RECORDS, 40, K.MAX_SOURCES, 0))

    def test_write_error_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "geometry_refusal_v1").write_text("not a directory")
            d = K.Diagnostics(tmp)
            d.record(dict(descriptor="{}"), None, dict(status=CL.STATUS_TARGET), job=1)
            self.assertEqual(d.counters["refusals"], 1)
            self.assertGreaterEqual(d.counters["write_errors"], 1)


class ActorVisibleOutcomeTest(unittest.TestCase):
    """The edge's GROUND_TRUTH_UNAVAILABLE feedback through the real wire codec and reward/hold controller."""

    def test_timely_and_late_gt_unavailable(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import reward_hold_controller_v2 as RC
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import test_phase4_reward_hold_v2 as T
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as RCON
        ms = T.MS
        for receipt_ms, expected in ((169, "TIMELY"), (171, "LATE")):
            with self.subTest(receipt_ms=receipt_ms):
                c = T.controller()
                _, _, first = T.opened(c)
                fb = T.feedback(first, kind="EVALUATOR_FAULT")       # what the edge emits for UNSCORED
                payload = W.encode_feedback(fb, W.EvaluatorReason.GROUND_TRUTH_UNAVAILABLE)
                decoded, reason = W.decode_feedback(payload)
                self.assertEqual((decoded.kind, decoded.q_perc, reason.name),
                                 ("EVALUATOR_FAULT", None, "GROUND_TRUTH_UNAVAILABLE"))
                cls = c.on_feedback(decoded, receipt_ns=1_000 * ms + receipt_ms * ms)
                resolution = c.current.resolution
                if expected == "TIMELY":
                    self.assertIs(cls, RC.FeedbackClass.ACCEPTED)
                    self.assertIs(resolution.terminal, RCON.RewardTerminal.EVALUATOR_FAULT)
                    self.assertFalse(resolution.learning_included)
                    with self.assertRaises(RC.SessionBreakRequired):       # next decision starts from genesis
                        c.previous_for_next_decision()
                else:
                    self.assertIs(cls, RC.FeedbackClass.LATE_ORPHAN)
                    self.assertIs(resolution.terminal, RCON.RewardTerminal.TIMEOUT)
                    self.assertTrue(resolution.learning_included)
                    self.assertIsNotNone(c.previous_for_next_decision())   # TIMEOUT history, no session break


if __name__ == "__main__":
    unittest.main()


class MetricsTest(unittest.TestCase):
    def test_taxonomy_keeps_every_decision_and_separates_refusals(self):
        from . import metrics as MT
        rows = [dict(service_outcome="SCORED", quality_defined="True", feedback_receipt_ms="150", timely_170="True"),
                dict(service_outcome="SCORED", quality_defined="False", feedback_receipt_ms="160", timely_170="True"),
                dict(service_outcome="SCORED", quality_defined="True", feedback_receipt_ms="190", timely_170="False"),
                dict(service_outcome=MT.GEOMETRY_PREFIX + "TARGET", quality_defined="", feedback_receipt_ms="120",
                     timely_170="True"),
                dict(service_outcome="NO_GT_FREEZE_QUEUE_FULL", quality_defined="", feedback_receipt_ms="",
                     timely_170="False")]
        s = MT.summarise(rows)
        self.assertEqual((s["policy_decisions"], s["any_feedback_le170"], s["scored_feedback_le170"], s["scored"],
                          s["scored_zero_eligible"], s["gt_unavailable_geometry"], s["gt_unavailable_other"],
                          s["gt_unavailable_feedback_le170"], s["late"], s["missing_feedback"]),
                         (5, 3, 2, 3, 1, 1, 1, 1, 1, 1))


# ---------------------------------------------------------------- review_v1 blocker 2: plan validation regression
def _plan_mutations(corpus, case, other_case):
    """(name, plan bytes, frame context) per mutation; names 1-3 are the independent review's counterexamples."""
    import dataclasses
    from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as NC

    plan = NC._parse(corpus.plan(case))
    context = corpus.context(case)

    def edit(change):
        value = copy.deepcopy(plan)
        change(value)
        return NC.canonical(value)
    pose = context.ego_world
    return [
        ("missing_action", edit(lambda p: p.pop("action")), context),
        ("missing_frame_context", edit(lambda p: p.pop("frame_context")), context),
        ("action_frame_mismatch", edit(lambda p: p["action"].update(frame_id=p["action"]["frame_id"] + 1)), context),
        ("missing_capture", edit(lambda p: p.pop("capture")), context),
        ("capture_schema_invalid", edit(lambda p: p["capture"].update(frame_id="x")), context),
        ("frame_context_not_dict", edit(lambda p: p.update(frame_context=[1])), context),
        ("not_reward_requested", edit(lambda p: p["action"].update(reward_requested=False)), context),
        ("hold_frame_kind", edit(lambda p: p["action"].update(frame_kind="HOLD")), context),
        ("run4_identity_invalid", edit(lambda p: p["action"].pop("session_uuid")), context),
        ("non_canonical_bytes", NC.canonical(plan) + b" ", context),
        ("not_json", b"{", context),
        ("foreign_stream_capture", edit(lambda p: p["capture"].update(stream_id=p["capture"]["stream_id"] + "_x")),
         context),
        ("tensor_seq_disagrees_with_context",
         edit(lambda p: p["action"].update(tensor_seq=p["action"]["tensor_seq"] + 1)), context),
        ("other_frame_context", NC.canonical(plan), corpus.context(other_case)),
        ("context_pose_shifted_1m", NC.canonical(plan),
         dataclasses.replace(context, ego_world=dataclasses.replace(pose, x=pose.x + 1.0))),
        ("valid_anchor_plan", P.anchor_variant(corpus, case, 71)["plan_bytes"], context),
        ("valid_plan", NC.canonical(plan), context),
    ]


def _bind_category(corpus, case, contained, plan_bytes, context):
    driver = P.DeferredDriver(corpus)
    try:
        if contained:
            E.adopt(driver.engine)
        handle = driver.unbound(case)
        if contained:
            assert driver.engine.contained_refusal(handle) is not None
        try:
            driver.engine.bind(handle, plan_bytes, frame_context=context)
            category = ("OK", None)
        except Exception as error:
            from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as D
            if isinstance(error, D.DeferralRefused):
                category = ("GEOMETRY_REFUSED" if error.reason.startswith("GT_GEOMETRY_ENVELOPE_") else "REFUSED",
                            error.reason)
            else:
                category = ("FATAL", type(error).__name__)
        return dict(category=category, engine_failed=driver.engine._failure is not None,
                    owner_failed=driver.engine._owner._failure is not None,
                    snapshot_valid=driver.engine.snapshot()["valid"])
    finally:
        driver.engine.close(abandon=True)


class PlanValidationTest(unittest.TestCase):
    """A contained (refused) frame validates its plan exactly like a normal frame; only a fully valid plan gets the
    geometry refusal.  Normal frame: case 6 on the UNCHANGED engine; refused frame: case 7 (S1 tilt) contained."""

    @classmethod
    def setUpClass(cls):
        P.environment_ok()
        switch.install()
        cls.corpus = FixtureCorpus({7: ("TILT", "146", *S1_TILT, "review_counterexample_tilt")})
        normal, refused = cls.corpus.cases[6], cls.corpus.cases[7]
        cls.results = {}
        for (name, n_plan, n_ctx), (name2, r_plan, r_ctx) in zip(_plan_mutations(cls.corpus, normal, cls.corpus.cases[8]),
                                                                 _plan_mutations(cls.corpus, refused, cls.corpus.cases[8])):
            assert name == name2
            cls.results[name] = dict(normal=_bind_category(cls.corpus, normal, False, n_plan, n_ctx),
                                     refused=_bind_category(cls.corpus, refused, True, r_plan, r_ctx))

    @classmethod
    def tearDownClass(cls):
        switch.uninstall()

    def _expected(self, normal):
        return ("GEOMETRY_REFUSED", CL.REASONS[CL.STATUS_TARGET]) if normal["category"] == ("OK", None) \
            else normal["category"]

    def test_review_counterexamples_remain_fatal(self):
        for name in ("missing_action", "missing_frame_context", "action_frame_mismatch"):
            with self.subTest(name=name):
                r = self.results[name]
                self.assertEqual(r["normal"]["category"][0], "FATAL")
                self.assertEqual(r["refused"]["category"], r["normal"]["category"])
                self.assertTrue(r["refused"]["engine_failed"] and r["refused"]["owner_failed"])
                self.assertFalse(r["refused"]["snapshot_valid"])

    def test_every_mutation_classified_like_a_normal_frame(self):
        for name, r in self.results.items():
            with self.subTest(name=name):
                self.assertEqual(r["refused"]["category"], self._expected(r["normal"]))
                self.assertEqual(r["refused"]["engine_failed"], r["normal"]["engine_failed"])
                self.assertEqual(r["refused"]["owner_failed"], r["normal"]["owner_failed"])

    def test_coverage_of_outcome_classes(self):
        normal = {r["normal"]["category"][0] for r in self.results.values()}
        self.assertEqual(normal, {"FATAL", "REFUSED", "OK"})
        self.assertEqual(self.results["valid_plan"]["refused"]["category"],
                         ("GEOMETRY_REFUSED", "GT_GEOMETRY_ENVELOPE_TARGET"))
        self.assertEqual(self.results["valid_anchor_plan"]["refused"]["category"],
                         ("GEOMETRY_REFUSED", "GT_GEOMETRY_ENVELOPE_TARGET"))
        self.assertFalse(self.results["valid_plan"]["refused"]["engine_failed"])


# ---------------------------------------------------------------- review_v1 blocker 1: prep stack composition
class WorkerEntryCompositionTest(unittest.TestCase):
    """The candidate prep worker composes the CURRENT V16 stack: accounting V3.v2 session + timer ACTIVE when the
    retained worker runs, plus tilt AVO, FREEZE lifetime repair and containment; everything restored afterwards
    (the independent review's entry probe, retained main mocked)."""

    def _probe(self, entry):
        from unittest.mock import patch
        from rl_agent.splitfusion_avo_tilt_v1 import quality as Q
        from rl_agent.splitfusion_gt_storage_accounting_v3 import accounting as A, switch as SW
        from rl_agent.splitfusion_m3_route_b_v16.accounting_timing import AccountingTimer
        from rl_agent.splitfusion_m3_route_b_v4 import worker_entry as LEAF
        from rl_agent.splitfusion_m3_route_b_v7 import worker_entry as V7W
        from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
        from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA
        seen = {}
        with tempfile.TemporaryDirectory() as tmp:
            def observe(args):
                active = OWN.payload_storage_bytes
                seen.update(timer=type(active) is AccountingTimer,
                            session=isinstance(getattr(active, "__wrapped__", None), A.AccountingSession),
                            avo=NA.numeric_avo is Q.numeric_avo, prep_core=PC.PrepCore)
                return 0
            with patch.object(V7W, "role_and_trace", return_value=("prep", Path(tmp))), \
                    patch.object(LEAF, "main", side_effect=observe):
                rc = entry.main([])
            records = {p.name: json.loads(p.read_text()) for p in Path(tmp).iterdir()}
        restored = (OWN.payload_storage_bytes is SW.ORIGINAL_PAYLOAD_STORAGE_BYTES and SW.session() is None
                    and NA.numeric_avo is switch.ORIGINAL_NUMERIC_AVO and PC.PrepCore is R.ORIGINAL_PREP_CORE)
        return rc, seen, records, restored

    def test_candidate_matches_v16_stack_plus_containment(self):
        from rl_agent.splitfusion_m3_route_b_v16 import contract as V16, worker_entry as V16W
        from . import contract as CC, worker_entry as CW
        rc16, seen16, rec16, ok16 = self._probe(V16W)
        rc, seen, rec, ok = self._probe(CW)
        self.assertEqual((rc16, rc), (0, 0))
        self.assertTrue(ok16 and ok)
        for key in ("timer", "session", "avo"):
            self.assertTrue(seen16[key] and seen[key], key)
        self.assertIs(seen16["prep_core"], R.PrepCoreLifetime)
        self.assertIs(seen["prep_core"], K.PrepCoreContained)
        self.assertLessEqual(set(rec16), set(rec))
        self.assertEqual(set(rec) - set(rec16), {CC.CONTAINMENT_BINDING_RECORD})
        acct = rec[CC.ACCOUNTING_BINDING_RECORD]
        self.assertTrue(acct["installed"] and acct["timer_installed"] and acct["timer_removed"] and acct["restored"]
                        and acct["fast_path_available"] and acct["accounting_session_closed"])
        self.assertEqual(acct["version"], V16.ACCOUNTING_VERSION)
        self.assertEqual(rec[CC.ACCOUNTING_TIMING_RECORD]["wrapped"], rec16[V16.ACCOUNTING_TIMING_RECORD]["wrapped"])
        for name in (CC.AVO_BINDING_RECORD, CC.PREP_BINDING_RECORD, CC.CONTAINMENT_BINDING_RECORD):
            self.assertTrue(rec[name]["installed"] and rec[name]["restored"], name)

    def test_registration_carries_the_v16_accounting(self):
        from rl_agent.splitfusion_m3_route_b_v16 import contract as V16
        from . import contract as CC
        mine, v16 = CC.gc_registration("candidate", 3900, "v2"), V16.gc_registration("candidate", 3900, "v2")
        self.assertEqual(mine["storage_accounting"], v16["storage_accounting"])
        self.assertEqual({k: v for k, v in mine.items() if k not in ("trampolines", "implementations", "seam",
                                                                  "geometry_containment")},
                         {k: v for k, v in v16.items() if k not in ("trampolines", "implementations", "seam")})
        root = Path(__file__).resolve().parents[2]
        self.assertLessEqual(set(V16.exact_sources(root)), set(CC.exact_sources(root)))

    def test_capability_refusal_is_preserved(self):
        from unittest.mock import patch
        from rl_agent.splitfusion_gt_storage_accounting_v3 import accounting as A
        from rl_agent.splitfusion_m3_route_b_v7 import worker_entry as V7W
        from . import worker_entry as CW
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(V7W, "role_and_trace", return_value=("prep", Path(tmp))), \
                patch.object(A, "capabilities", return_value=dict(dict_tags=None, mappingproxy="unavailable",
                                                                  pair_history=None)):
            with self.assertRaisesRegex(RuntimeError, "REFUSED: registered V3 storage-accounting capabilities"):
                CW.main([])
