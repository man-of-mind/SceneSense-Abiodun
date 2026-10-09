"""Boundary accounting, the explicit exact-GT service queue, deadline edges and live outcome/state parity."""
import json
import math
import unittest
from dataclasses import replace

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract

from . import calibration as K
from . import chains as CH
from . import outcome as O

BINDING = K.PACKAGE_DIR / "bindings" / "LIVE100_REGRESSION_BINDING_V2.json"
SCREEN_BINDING = K.PACKAGE_DIR / "bindings" / "SCREEN_V2_ARMS0_1_PARTIAL_BINDING_V2.json"


def _calibration(path=BINDING, **policy):
    doc = json.loads(path.read_text())
    doc["policy"].update(policy)
    return K.CalibrationSetV1(doc)


class ExactReproduction(unittest.TestCase):
    """Every recorded chain (cold included) is reproduced exactly by the queue model before any other use."""

    @classmethod
    def setUpClass(cls):
        cls.cals = {"live100": _calibration(exclude_cold=False),
                    "screen": _calibration(SCREEN_BINDING, exclude_cold=False)}

    def test_all_143_chains_reproduce_every_boundary(self):
        self.assertEqual(sum(len(c.chains) for c in self.cals.values()), 48 + 95)
        for cal in self.cals.values():
            for cid, chain in cal.chains.items():
                vector = cal.vectors[cid]
                comp = CH.compose(vector)
                self.assertIs(comp.ordering, vector.realized_ordering, cid)
                for name, value in comp.instants.items():
                    self.assertAlmostEqual(value, chain.instants[name], delta=1e-9, msg=f"{cid} {name}")
                req = chain.service.requests
                self.assertAlmostEqual(comp.gt_ready, req["PUBLISH"]["svc_end"], delta=1e-9)
                self.assertAlmostEqual(comp.channel_busy_until, chain.service.post_scoring_end, delta=1e-9)

    def test_critical_path_counts_each_segment_once_and_sums_to_receipt(self):
        for cal in self.cals.values():
            for cid in cal.chains:
                comp = CH.compose(cal.vectors[cid])
                names = [n for n, _ in comp.critical_path]
                self.assertEqual(len(names), len(set(names)), cid)
                self.assertAlmostEqual(sum(v for _, v in comp.critical_path), comp.receipt, delta=1e-9)
                self.assertTrue(all(v >= -CH.TOLERANCE_MS for _, v in comp.critical_path), cid)
                self.assertTrue(names[-2].endswith("_score_execute"), cid)   # scoring needs both inputs

    def test_gt_boundaries_stay_separate(self):
        vector = next(iter(self.cals["live100"].vectors.values()))
        names = [n for n, _ in vector.gt_segments]
        expected = ["gt_freeze_handoff"] + [f"gt_{op.lower()}_{part}" for op in CH.GT_REQUESTS
                                            for part in ("dispatch", "send", "execute", "reply")]
        self.assertEqual(names, expected)
        # Preparation (FREEZE execute), binding, publication and dispatch waits are distinct terms.
        self.assertGreater(dict(vector.gt_segments)["gt_freeze_execute"], 10.0)

    def test_queue_rule_and_measured_carry_in(self):
        counts = {"GT_FIRST": 0, "PREDICTION_FIRST": 0, "carry": 0}
        for cal in self.cals.values():
            for cid, vector in cal.vectors.items():
                counts[vector.realized_ordering.value] += 1
                comp = CH.compose(vector)
                if comp.critical_path[0][0] == "score_channel_carry_in":
                    counts["carry"] += 1
                    self.assertTrue(cid.endswith("#1362"), cid)   # cold frame blocked by the previous decision
        self.assertEqual(counts, {"GT_FIRST": 43 + 44 + 16, "PREDICTION_FIRST": 5 + 4 + 31, "carry": 1})


class QueueMechanics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cal = _calibration(SCREEN_BINDING, exclude_cold=False)
        cls.cal = cal
        warm = [v for cid, v in cal.vectors.items() if not cal.chains[cid].context.cold]   # cold scorer start-up
        cls.gt_first = next(v for v in warm if v.realized_ordering is CH.Ordering.GT_FIRST
                            and CH.compose(v).gt_ready + 20 < CH.compose(v).prediction_at_service)
        cls.pred_first = next(v for v in warm
                              if v.realized_ordering is CH.Ordering.PREDICTION_FIRST
                              and (v.carry_in is None or v.carry_in < CH.compose(v).prediction_at_service)
                              and CH.compose(v).prediction_at_service + 5 < CH.compose(v).gt_ready)

    def _shift_gt(self, vector, delta):
        segments = tuple((n, v + delta if n == "gt_freeze_execute" else v) for n, v in vector.gt_segments)
        return replace(vector, gt_segments=segments, bind_end=vector.bind_end + delta)

    def test_gt_preparation_is_concurrent_not_added(self):
        base = CH.compose(self.gt_first)
        slack = base.prediction_at_service - (base.gt_ready + self.gt_first.dispatch_target
                                              + sum(vars(self.gt_first.costs.target_admit).values())
                                              - self.gt_first.costs.target_admit.dispatch)
        self.assertGreater(slack, 5)
        later = CH.compose(self._shift_gt(self.gt_first, slack / 2))
        self.assertAlmostEqual(later.receipt, base.receipt, delta=1e-9)

    def test_prediction_first_waits_for_gt_ready_at_service(self):
        base = CH.compose(self.pred_first)
        shifted = CH.compose(self._shift_gt(self.pred_first, 10.0))
        self.assertAlmostEqual(shifted.receipt - base.receipt, 10.0, delta=1e-9)
        self.assertEqual(shifted.critical_path[0][0], "gt_freeze_handoff")

    def test_priority_prediction_wins_when_both_pending(self):
        # Block the channel until both inputs are pending: PREDICTION (priority 0) must be dispatched first.
        v = self.gt_first
        late_free = max(CH.compose(v).prediction_at_service, CH.compose(v).gt_ready) + 1.0
        with self.assertRaises(CH.UncalibratedTermError):      # its own chain lacks prediction_store/target_score
            CH.compose(v, carry_in=late_free)
        donor = self.pred_first.costs
        filled = v.with_costs(replace(v.costs, prediction_store=donor.prediction_store,
                                      target_score=donor.target_score, donor_chain_id=self.pred_first.chain_id))
        comp = CH.compose(filled, carry_in=late_free)
        self.assertIs(comp.ordering, CH.Ordering.PREDICTION_FIRST)
        self.assertEqual(comp.critical_path[0], ("score_channel_carry_in", late_free))

    def test_channel_is_serial_and_carry_in_delays(self):
        v = self.gt_first
        base = CH.compose(v, carry_in=None)
        blocked = CH.compose(v, carry_in=base.gt_ready + 3.0)   # channel busy just after GT is ready
        self.assertGreaterEqual(blocked.receipt, base.receipt)
        self.assertIs(blocked.ordering, CH.Ordering.GT_FIRST)

    def test_send_after_bind_is_uncalibrated(self):
        v = self.pred_first
        late = replace(v, C=v.C + (v.bind_end - (v.A + v.B + v.C + v.D)) + 1.0)
        with self.assertRaises(CH.UncalibratedTermError) as raised:
            CH.compose(late)
        self.assertEqual(raised.exception.term, "gt_confirm_send_dependency")

    def test_malformed_and_queue_violating_chains_are_rejected(self):
        chain = next(iter(self.cal.chains.values()))
        bad = dict(chain.instants, ue_input_end=chain.instants["ue_input_start"] - 1.0)
        with self.assertRaises(CH.ChainError):
            CH.decompose(replace(chain, instants=bad))
        with self.assertRaises(CH.ChainError):
            CH.decompose(replace(chain, service=None))
        # TARGET dispatched first while the prediction was already pending violates priority.
        gt_first_chain = self.cal.chains[self.gt_first.chain_id]
        req = {k: dict(v) for k, v in gt_first_chain.service.requests.items()}
        early_prediction = req["TARGET"]["svc_begin"] - 1.0
        tampered = replace(gt_first_chain.service, prediction_received=early_prediction)
        instants = dict(gt_first_chain.instants, service_prediction_received=early_prediction)
        with self.assertRaises(CH.ChainError):
            CH.decompose(replace(gt_first_chain, service=tampered, instants=instants))
        row = {name: chain.instants[name] for name in CH.INSTANTS}
        row.update(frame_id=1, scoring_request_op=chain.scoring_request_op)
        row.pop("gt_published")
        with self.assertRaises(CH.ChainError):
            CH.chain_from_boundary_row(row, source_id="legacy", context=chain.context)

class DeadlineAndOutcome(unittest.TestCase):
    def test_exactly_170_ms_is_timely_and_just_over_is_timeout(self):
        on = O.classify(170_000_000, 0.6)
        self.assertIs(on.kind, contract.RewardEventKind.DELIVERED_SUCCESS)
        self.assertEqual(on.resolution_elapsed_ns, 170_000_000)
        over = O.classify(170_000_001, 0.6)
        self.assertIs(over.kind, contract.RewardEventKind.TIMEOUT)
        self.assertEqual(over.resolution_elapsed_ns, 170_000_001)
        self.assertEqual(over.feedback_class, "LATE_ORPHAN")
        self.assertEqual(O.receipt_elapsed_ns(170.0), 170_000_000)
        self.assertEqual(O.receipt_elapsed_ns(170.000001), 170_000_001)

    def test_reward_through_the_unchanged_resolver(self):
        identity = contract.DecisionIdentityV1("00000000-0000-4000-8000-000000000001", "ue", 0)
        from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
        from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import ExecutedActionIdentity
        catalog = action_contract.load_contract()
        action = ExecutedActionIdentity.from_executable_action(catalog.resolve(8, 0.35), catalog)
        for receipt, q, terminal, reward in ((170_000_000, 0.6, "SUCCESS", 0.6 - 0.25),
                                             (170_000_001, 0.6, "TIMEOUT", -1.0),
                                             (120_000_000, None, "EVALUATOR_FAULT", None),
                                             (170_000_001, None, "TIMEOUT", -1.0),
                                             (None, 0.6, "TIMEOUT", -1.0)):
            out = O.classify(receipt, q)
            event = contract.RewardEventV1(identity=identity, action=action, kind=out.kind,
                                           action_open_timestamp_ns=1_000_000_000,
                                           resolution_timestamp_ns=1_000_000_000 + out.resolution_elapsed_ns,
                                           clock_domain="CLOCK_MONOTONIC_RAW", source="test",
                                           q_perc=out.q_perc if out.kind.value == "DELIVERED_SUCCESS" else None)
            resolution = contract.resolve_reward(event)
            self.assertEqual(resolution.terminal.value, terminal)
            if reward is None:
                self.assertFalse(resolution.learning_included)
                self.assertTrue(out.breaks_session)
            else:
                self.assertAlmostEqual(resolution.reward, reward, places=12)

    def test_live100_controller_resolutions_are_reproduced(self):
        cal = _calibration(exclude_cold=False)
        counts = {}
        for chain in cal.chains.values():
            out = O.classify(O.receipt_elapsed_ns(chain.instants["ue_receipt"]), chain.q_perc)
            self.assertEqual(O.CONTROLLER_TERMINAL[out.kind], chain.controller_resolution, chain.chain_id)
            counts[chain.controller_resolution] = counts.get(chain.controller_resolution, 0) + 1
        self.assertEqual(counts, {"SUCCESS": 4, "EVALUATOR_FAULT": 1, "TIMEOUT": 43})

    def test_live100_previous_outcome_features_are_reproduced(self):
        """The recorded state of each live decision equals the parity encoding of the preceding outcome."""
        doc = json.loads(BINDING.read_text())
        evidence = json.loads(K.resolve_path(doc["sources"][0]["files"]["evidence"]["path"]).read_text())
        decisions = [d for d in evidence["decisions"] if d.get("kind") == "POLICY_DECISION"]
        by_open = {r["action_open_timestamp_ns"]: r for r in evidence["resolutions"]}
        previous, previous_session, checked = None, None, 0
        for d in decisions:
            res = by_open[d["action_open"]["ns"]]
            if d["decision_session_uuid"] != previous_session:
                expected = O.previous_outcome_features(None, mode_id=None, q_e4=None, q_perc=None, latency_ms=None)
            else:
                kind = contract.RewardEventKind.DELIVERED_SUCCESS if previous["terminal"] == "SUCCESS" \
                    else contract.RewardEventKind.TIMEOUT
                expected = O.previous_outcome_features(kind, mode_id=previous_mode, q_e4=previous_q,
                                                       q_perc=previous["q_perc"], latency_ms=previous["latency_ms"])
            self.assertEqual([float(x).hex() for x in d["features"][4:21]], [float(x).hex() for x in expected],
                             d["frame_id"])
            checked += 1
            previous, previous_session = res, d["decision_session_uuid"]
            previous_mode, previous_q = d["mode_id"], d["q_e4"]
        self.assertEqual(checked, 48)


class NoStaleResidual(unittest.TestCase):
    def test_package_never_uses_the_retained_residual(self):
        for path in K.PACKAGE_DIR.glob("*.py"):
            if path.name.startswith("test_"):
                continue
            text = path.read_text()
            self.assertNotIn("retained_resid", text, path.name)
            self.assertNotIn("SEND_SPAN_NS_PER_BYTE", text, path.name)
            self.assertNotIn("actor_reserve", text, path.name)


if __name__ == "__main__":
    unittest.main()
