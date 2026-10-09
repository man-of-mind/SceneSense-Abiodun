"""Static-mode v13 tests (CPU only, no live service).

The ``Drain`` tests are the V12 drain tests run unchanged against this package's drain layer (the bounded end-of-run
drain is preserved at the budget stop); the remaining tests cover the registered v13 changes.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m unittest -v rl_agent.splitfusion_static_modes_v13.test_static_modes_v13
"""
import ast
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from . import campaign as CAMP
from . import contract as C
from . import fixed_engine as FE
from . import route_drain as RD

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / C.PACKAGE


class _FakeLedger:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}
        self.closed = {}

    def summary(self):
        return dict(obligations_closed=len(self.closed), obligations_open=len(self.pending))


class _FakeFeedback:
    def __init__(self):
        self._ledger = _FakeLedger()
        self._inbox = types.SimpleNamespace(empty=lambda: True, qsize=lambda: 0)

    def record_expired(self, now=None):
        observed = time.time() if now is None else float(now)
        count = 0
        with self._ledger.lock:
            for key, row in list(self._ledger.pending.items()):
                if observed >= float(row["ack_timeout_at"]):
                    self._ledger.pending.pop(key)
                    self._ledger.closed[key] = "FEEDBACK_TIMEOUT"
                    count += 1
        return count

    def summary(self):
        return self._ledger.summary()


class _Base:
    """Minimal stand-in for the retained collector below the drain layer."""

    def __init__(self, **kwargs):
        self.attempt_dir = kwargs["attempt_dir"]
        self.ack_timeout_s = 0.5
        self.sent = 0
        self.dropped = 0
        self.failures = []
        self.rows = []
        self.finished = []
        self.feedback = _FakeFeedback()
        self.transport_counters = types.SimpleNamespace(bump=lambda name: self.rows.append(("bump", name)))

    def _append_row(self, row):
        self.rows.append(row)

    def _process_token(self, token):
        self.sent += 1
        key = f"c:{token['frame_id']}"
        with self.feedback._ledger.lock:
            self.feedback._ledger.pending[key] = dict(ack_timeout_at=time.time() + self.ack_timeout_s)
        return "sent"

    def finish(self):
        self.finished.append(dict(open=len(self.feedback._ledger.pending)))
        return True


class Drain(unittest.TestCase):
    def layer(self, directory):
        return RD.build_drain_layer(_Base)(attempt_dir=directory)

    def evidence(self, directory):
        return json.loads((Path(directory) / "m3_route" / "END_OF_ROUTE_DRAIN.json").read_text())

    def test_in_flight_token_is_waited_for_and_its_terminal_arrives_before_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            started, release = threading.Event(), threading.Event()

            retained = _Base._process_token

            def slow(self, token):                    # patched on the class: the drain layer's super() call
                started.set()
                release.wait(5)
                return retained(self, token)
            with patch.object(_Base, "_process_token", slow):
                worker = threading.Thread(target=collector._process_token, args=({"frame_id": 8595},))
                worker.start()
                self.assertTrue(started.wait(5))

                def ack():                      # the real ACK arrives while the drain is waiting
                    time.sleep(0.15)
                    with collector.feedback._ledger.lock:
                        for key in list(collector.feedback._ledger.pending):
                            collector.feedback._ledger.pending.pop(key)
                            collector.feedback._ledger.closed[key] = "RESULT_INSTALLED"
                acker = threading.Thread(target=ack)
                release.set()
                worker.join(5)
                acker.start()
                self.assertTrue(collector.finish())
                acker.join(5)
            evidence = self.evidence(temporary)
        self.assertIs(evidence["in_flight_completed"], True)
        self.assertEqual(evidence["sent_at_end"], 1)
        self.assertEqual(evidence["open_at_end"], [])
        self.assertEqual(evidence["expired_on_real_clock"], 0)                 # a real ACK, not a timeout
        self.assertEqual(evidence["ledger_summary"]["obligations_closed"], 1)
        self.assertIs(evidence["closed_equals_sent"], True)
        self.assertEqual(collector.finished, [dict(open=0)])                   # the retained finish still ran

    def test_tokens_offered_during_the_drain_are_dropped_and_never_create_an_obligation(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            collector.finish()
            self.assertIsNone(collector._process_token({"frame_id": 9001}))
            self.assertEqual(collector.sent, 0)
            self.assertEqual(collector.dropped, 1)
            self.assertEqual(collector.feedback._ledger.pending, {})
            evidence = self.evidence(temporary)
        self.assertEqual([r for r in collector.rows if isinstance(r, dict)][0]["prepare_status"], RD.DROP_STATUS)
        self.assertEqual(evidence["tokens_dropped_during_drain"], [])          # dropped after the evidence was written

    def test_an_obligation_that_never_acks_becomes_an_honest_timeout_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            collector._process_token({"frame_id": 4242})
            self.assertTrue(collector.finish())
            evidence = self.evidence(temporary)
        self.assertEqual(evidence["expired_on_real_clock"], 1)
        self.assertEqual(evidence["open_at_end"], [])
        self.assertEqual(collector.feedback._ledger.closed, {"c:4242": "FEEDBACK_TIMEOUT"})
        self.assertIs(evidence["missing_terminal_declared_successful"], False)
        self.assertGreaterEqual(evidence["waited_s"], collector.ack_timeout_s)

    def test_the_drain_never_waits_past_the_absolute_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            with collector.feedback._ledger.lock:                              # an obligation with a far horizon
                collector.feedback._ledger.pending["c:1"] = dict(ack_timeout_at=time.time() + 10_000)
            with patch.object(C, "DRAIN_ABSOLUTE_CAP_S", 0.4):
                began = time.monotonic()
                self.assertTrue(collector.finish())
                elapsed = time.monotonic() - began
            evidence = self.evidence(temporary)
        self.assertLess(elapsed, 5.0)
        self.assertEqual(evidence["open_before_expiry"], ["c:1"])
        self.assertEqual(evidence["expired_on_real_clock"], 0)                 # still inside its own horizon
        self.assertEqual(evidence["open_at_end"], ["c:1"])                     # reported, never silently closed

    def test_a_drain_failure_is_recorded_and_never_replaces_the_retained_teardown(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            with patch.object(collector, "_m3_open_obligations", side_effect=RuntimeError("ledger unavailable")):
                self.assertTrue(collector.finish())
            evidence = self.evidence(temporary)
        self.assertIs(evidence["completed"], False)
        self.assertIn("ledger unavailable", evidence["error"])
        self.assertEqual(collector.finished, [dict(open=0)])
        self.assertTrue(any("end-of-route drain" in f for f in collector.failures))
        self.assertTrue(collector._m3_draining)

    def test_the_layer_sits_directly_above_the_unchanged_v3_layer(self):
        from rl_agent.splitfusion_m3_route_b_v3 import route_completion as RC
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        original = LG.build_live_gt_collector
        with RD.install_collector_layer() as built:
            self.assertIsNot(LG.build_live_gt_collector, original)
            layer = LG.build_live_gt_collector(_Base)
            self.assertEqual(built, [layer])
        self.assertIs(LG.build_live_gt_collector, original)
        names = [cls.__name__ for cls in layer.__mro__]
        # the drain layer wraps the UNCHANGED v3 M3 layer, which wraps whatever the retained builder produced
        self.assertEqual(names[:2], ["M3EndOfRouteDrainLayer", "M3CollectorLayer"])
        self.assertEqual(names[2], original(_Base).__name__)
        self.assertIn(_Base, layer.__mro__)
        self.assertEqual(RC.build_layer(_Base).__name__, "M3CollectorLayer")

    def test_registration_states_what_is_unchanged(self):
        registration = RD.drain_registration()
        self.assertEqual(registration["bounds"]["settle_margin_s"], C.DRAIN_SETTLE_MARGIN_S)
        self.assertIn("170 ms decision deadline and reward resolution", registration["unchanged"])
        self.assertTrue(any("missing terminal successful" in item for item in registration["never"]))




class DrainAtBudgetStop(unittest.TestCase):
    def test_the_budget_stop_drains_like_a_route_end_and_closes_every_sent_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = RD.build_drain_layer(_Base)(attempt_dir=temporary)
            for frame in range(1, 1001):                          # exactly the registered budget is sent
                collector._process_token({"frame_id": frame})
            with collector.feedback._ledger.lock:                  # all but the last ACKed before the stop
                for key in list(collector.feedback._ledger.pending)[:-1]:
                    collector.feedback._ledger.pending.pop(key)
                    collector.feedback._ledger.closed[key] = "RESULT_INSTALLED"
            self.assertTrue(collector.finish())                    # finish() is reached on ANY stop, budget included
            self.assertIsNone(collector._process_token({"frame_id": 1001}))
            evidence = json.loads((Path(temporary) / "m3_route" / "END_OF_ROUTE_DRAIN.json").read_text())
        self.assertEqual(evidence["sent_at_end"], 1000)
        self.assertIs(evidence["closed_equals_sent"], True)
        self.assertEqual(evidence["expired_on_real_clock"], 1)      # the last frame timed out honestly
        self.assertEqual(collector.sent, 1000)


class Contract(unittest.TestCase):
    def test_budget_limits_lifecycle(self):
        self.assertEqual(C.BUDGETS, (1000,))
        self.assertEqual(C.limits(1000)["max_frames"], 1000)
        with self.assertRaises(ValueError):
            C.limits(3900)
        self.assertEqual(C.FRAME_CEILING, 3900)                    # capacity constants unchanged
        self.assertEqual(C.LIFECYCLE, dict(safety_timeout_s=900.0, child_timeout_s=2400.0))

    def test_retention_is_disabled_and_refuses_a_plan(self):
        self.assertIs(C.retention_registration()["enabled"], False)
        with self.assertRaises(ValueError):
            C.retention_registration("0" * 64)

    def test_thirteen_arms_unique_tokens_and_frozen_order(self):
        import numpy as np
        self.assertEqual(len(C.ARM_KEYS), 13)
        tokens = {C.token(arm) for arm in C.ARM_KEYS}
        self.assertEqual(len(tokens), 13)
        permutation = list(np.random.default_rng(20261010).permutation(12))
        self.assertEqual(C.RUN_ORDER, ("agent",) + tuple(f"mode{int(m):02d}" for m in permutation))

    def test_frozen_identity_table_matches_the_contract_profiles(self):
        table = json.loads((ROOT / C.FIXED_BUNDLES_FILE).read_text())
        for mode, (family, quantizer, action, profile, recorded) in C.FIXED_IDENTITIES.items():
            self.assertEqual(table["profiles"][str(mode)], [family, quantizer, action, profile])
            self.assertEqual(action, 6 * mode)
            if recorded is not None:
                self.assertEqual(table["modes"][str(mode)], recorded)
        fixed = C.fixed_action("mode05")
        self.assertEqual((fixed["q_e4"], fixed["keep_count"], fixed["drop_count"], fixed["ranker_bypassed"]),
                         (0, 21504, 0, True))
        self.assertIsNone(C.fixed_action("agent"))

    def test_only_mode11_q0_is_inside_actor_support_but_all_run_external(self):
        inside = [m for m in range(12) if C.fixed_action(f"mode{m:02d}")["support"].startswith("INSIDE")]
        self.assertEqual(inside, [11])

    def test_channel_is_the_retained_favorable_stable_trace(self):
        registration = C.channel_registration(ROOT)
        self.assertEqual(registration["controller_module"], C.RETAINED_CONTROLLER_MODULE)
        self.assertEqual(registration["schedule"]["trace_sha256"], C.FAVORABLE_TRACE_SHA256)
        self.assertIs(registration["carrier_cell"]["profile_substituted"], False)
        self.assertNotIn("target_snr_db", registration["schedule"])


class Entries(unittest.TestCase):
    def test_child_installs_no_route_stop_rule_no_retention_and_no_controller_substitution(self):
        text = (PACKAGE / "entries.py").read_text()
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("#", '"', "*")))
        for forbidden in ("install_child_stop_rule", "install_retention", "finalize_after_child",
                          "CONTROLLER_MODULE = C.CONTROLLER_MODULE"):
            self.assertNotIn(forbidden, code)

    def test_fixed_scope_is_entered_only_for_mode_arms(self):
        tree = ast.parse((PACKAGE / "entries.py").read_text())
        child = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_child")
        source = ast.unparse(child)
        self.assertIn("_no_scope() if arm == C.AGENT_ARM else install_fixed_engine(", source)


class FixedEngine(unittest.TestCase):
    def test_identity_mismatch_raises_and_is_counted(self):
        fixed = C.fixed_action("mode03")
        fields = {k: fixed[k] for k in ("mode_id", "q_e4", "action_id", "profile_id", "execution_bundle_sha256",
                                        "keep_count", "drop_count", "family", "quantizer")}
        log = types.SimpleNamespace(counters=dict(executed_mismatch=0))
        FE._check(types.SimpleNamespace(profile=types.SimpleNamespace(**fields)), log, fixed)
        with self.assertRaises(FE.FixedBaselineError):
            FE._check(types.SimpleNamespace(profile=types.SimpleNamespace(**dict(fields, mode_id=4))), log, fixed)
        self.assertEqual(log.counters["executed_mismatch"], 1)

    def test_install_replaces_and_restores_the_engine_and_writes_the_summary(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
        original = E.Run4DecisionEngineV2
        with tempfile.TemporaryDirectory() as temporary:
            with FE.install_fixed_engine(arm="mode08", fixed_action=C.fixed_action("mode08"), attempt=temporary):
                self.assertTrue(getattr(E.Run4DecisionEngineV2, "_static_fixed_v13_installed", False))
                with self.assertRaises(FE.FixedBaselineError):
                    with FE.install_fixed_engine(arm="mode08", fixed_action=C.fixed_action("mode08"),
                                                 attempt=temporary):
                        pass
            summary = json.loads((Path(temporary) / FE.LOG_DIR / FE.SUMMARY_FILE).read_text())
        self.assertIs(E.Run4DecisionEngineV2, original)
        self.assertIs(summary["restored"], True)
        self.assertEqual(summary["counters"]["actor_calls"], 0)


class Campaign(unittest.TestCase):
    def test_launch_refuses_without_a_matching_authorization(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "STORAGE_PLAN.json"
            plan.write_text("{}")
            auth = Path(temporary) / "AUTH.json"
            auth.write_text(json.dumps(dict(schema=CAMP.AUTH_SCHEMA, explicit_live_launch_approved=False)))
            with self.assertRaises(SystemExit):
                CAMP._authorization(auth, ["agent"], plan)
            auth.write_text(json.dumps(dict(schema=CAMP.AUTH_SCHEMA, explicit_live_launch_approved=True,
                                            approved_by="x", campaign=CAMP.CAMPAIGN, arms=["mode08", "agent"],
                                            storage_plan_sha256=CAMP.sha(plan))))
            with self.assertRaises(SystemExit):                     # not the frozen order
                CAMP._authorization(auth, ["mode08", "agent"], plan)

    def test_storage_plan_counts_only_arms_that_keep_the_gate(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(CAMP, "free_bytes", return_value=25_650_000_000):
            document = CAMP.plan(Path(temporary) / "plan")
        self.assertEqual(document["first_stage"], ["agent", "mode08", "mode11"])
        self.assertIs(document["deletion_authorized"], False)


if __name__ == "__main__":
    unittest.main()
