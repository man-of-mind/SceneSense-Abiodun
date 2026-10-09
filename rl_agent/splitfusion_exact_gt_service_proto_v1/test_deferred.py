"""Adversarial unit tests for deferred action binding (CPU only, offline fixtures).

    cd abiodun && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /usr/bin/python3 -m unittest rl_agent.splitfusion_exact_gt_service_proto_v1.test_deferred -v
"""
import copy
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_runtime_v1 import producer as PR

from . import parity_common as P
from .deferred import DeferredProducerEngine, DeferralRefused

_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = P.Corpus(4)
    return _CORPUS


def kwargs_of(case, evidence=None):
    kw = {k: case.kwargs[k] for k in P.KEYS}
    if evidence is not None:
        kw["dynamic_snapshot_evidence"] = evidence
    return kw


def extended_population(case, extra_id):
    evidence = copy.deepcopy(case.evidence["dynamic_snapshot_evidence"])
    evidence["registered_population_ids"] = list(evidence["registered_population_ids"]) + [extra_id]
    evidence["population_sha256"] = P.sha(("synthetic-registration-%d" % extra_id).encode())
    return evidence


def history(engine):
    return engine._owner._producer.history_sha256()


class DeferredBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        P.environment_ok()
        cls.c = corpus()
        cls.cases = cls.c.cases

    def deferred(self, offline=True):
        config = self.c.engine_config(offline=offline)
        config.pop("live_process_contract", None)
        return DeferredProducerEngine(**config)

    def unbound(self, engine, case, evidence=None, offline=True):
        extra = self.c.offline_inputs(case) if offline else {}
        return engine.prepare_unbound(kwargs_of(case, evidence),
            population=evidence or case.evidence["dynamic_snapshot_evidence"], **extra)

    def test_cancelled_registration_is_discarded_and_reissued(self):
        f0, f1, f2 = self.cases[:3]
        extra = max(f0.evidence["dynamic_snapshot_evidence"]["registered_population_ids"]) + 100000
        ev1, ev2 = extended_population(f1, extra), extended_population(f2, extra)
        reference = PR.ProducerEngine(**self.c.engine_config())
        try:
            for case, ev in ((f0, None), (f2, ev2)):
                prepared = reference.prepare(self.c.plan(case), kwargs_of(case, ev),
                    frame_context=self.c.context(case), population=ev or case.evidence["dynamic_snapshot_evidence"],
                    **self.c.offline_inputs(case))
                reference.confirm_send(prepared.token, self.c.plan(case))
                expected = (prepared.identity_bytes, prepared.semantic_sha256,
                            reference.publish_ready(prepared.token), history(reference))
        finally:
            reference.close(abandon=True)
        engine = self.deferred()
        try:
            handle = self.unbound(engine, f0)
            prepared = engine.bind(handle, self.c.plan(f0), frame_context=self.c.context(f0))
            engine.confirm_send(prepared.token, self.c.plan(f0))
            before = history(engine)
            producer = engine._owner._producer
            handle = self.unbound(engine, f1, ev1)
            self.assertIsNotNone(producer.pending_registration)  # bound to the staged frame until commit/cancel
            engine.cancel_unbound(handle, "HOLD_NEVER_BOUND")
            self.assertEqual(history(engine), before)
            self.assertIsNone(producer.pending_registration)
            self.assertNotIn(extra, producer.population)
            handle = self.unbound(engine, f2, ev2)
            prepared = engine.bind(handle, self.c.plan(f2), frame_context=self.c.context(f2))
            engine.confirm_send(prepared.token, self.c.plan(f2))
            actual = (prepared.identity_bytes, prepared.semantic_sha256, engine.publish_ready(prepared.token),
                      history(engine))
            self.assertEqual(actual, expected)
            self.assertIn(extra, producer.population)
        finally:
            engine.close(abandon=True)

    def test_hold_plan_cannot_bind_and_history_is_restored(self):
        f0 = self.cases[0]
        engine = self.deferred()
        before = history(engine)
        try:
            handle = self.unbound(engine, f0)
            plan = C._parse(self.c.plan(f0))
            plan["action"] = dict(plan["action"], frame_kind="POLICY_HOLD", reward_requested=False)
            with self.assertRaises(Exception):
                engine.bind(handle, C.canonical(plan), frame_context=self.c.context(f0))
            self.assertFalse(engine.snapshot()["valid"])   # protocol violation fails closed
        finally:
            engine.close(abandon=True)
        self.assertEqual(history(engine), before)

    def test_non_template_source_is_refused_before_compute(self):
        f0 = self.cases[0]
        engine = self.deferred()
        before = history(engine)
        try:
            kw = kwargs_of(f0)
            kw["identity"] = dict(kw["identity"], action_id=71, profile_id="mode11_q9800_a71")
            with self.assertRaises(PR.ProducerError):
                engine.prepare_unbound(kw, population=f0.evidence["dynamic_snapshot_evidence"],
                                       **self.c.offline_inputs(f0))
            self.assertEqual(engine._owner.snapshot()["retained_jobs"], 0)
        finally:
            engine.close(abandon=True)
        self.assertEqual(history(engine), before)

    def test_refused_non_template_plan_does_not_poison(self):
        f0, f1 = self.cases[:2]
        engine = self.deferred()
        try:
            handle = self.unbound(engine, f0)
            plan = C._parse(self.c.plan(f0))
            plan["capture"] = dict(plan["capture"], action_id=71, profile_id="mode11_q9800_a71")
            with self.assertRaises(DeferralRefused):
                engine.bind(handle, C.canonical(plan), frame_context=self.c.context(f0))
            self.assertTrue(engine.snapshot()["valid"])
            handle = self.unbound(engine, f1)
            prepared = engine.bind(handle, self.c.plan(f1), frame_context=self.c.context(f1))
            self.assertTrue(prepared.identity_bytes)
        finally:
            engine.close(abandon=True)

    def test_handle_protocol(self):
        f0, f1 = self.cases[:2]
        engine = self.deferred()
        try:
            handle = self.unbound(engine, f0)
            with self.assertRaises(PR.ProducerError):   # one unbound at a time
                self.unbound(engine, f1)
        finally:
            engine.close(abandon=True)
        engine = self.deferred()
        try:
            handle = self.unbound(engine, f0)
            engine.cancel_unbound(handle, "HOLD_NEVER_BOUND")
            with self.assertRaises(PR.ProducerError):   # stale handle
                engine.bind(handle, self.c.plan(f0), frame_context=self.c.context(f0))
        finally:
            engine.close(abandon=True)
        engine = self.deferred()
        try:
            handle = self.unbound(engine, f1)
            engine.cancel_unbound(handle, "HOLD_NEVER_BOUND")
            with self.assertRaises(Exception):          # regressing frame after a later one
                self.unbound(engine, f0)
        finally:
            engine.close(abandon=True)
        engine = self.deferred()
        try:
            handle = self.unbound(engine, f0)
            prepared = engine.bind(handle, self.c.plan(f0), frame_context=self.c.context(f0))
            with self.assertRaises(PR.ProducerError):   # publication before commit
                engine.publish_ready(prepared.token)
        finally:
            engine.close(abandon=True)

    def test_close_with_unbound_restores_history(self):
        f0 = self.cases[0]
        engine = self.deferred()
        before = history(engine)
        self.unbound(engine, f0)
        engine.close(abandon=True)
        self.assertEqual(history(engine), before)
        self.assertEqual(engine._owner._producer.staged, None)

    def test_live_mode_export_stamp_order(self):
        f0 = self.cases[0]
        engine = self.deferred(offline=False)
        try:
            handle = self.unbound(engine, f0, offline=False)
            prepared = engine.bind(handle, self.c.plan(f0), frame_context=self.c.context(f0))
            engine.confirm_send(prepared.token, self.c.plan(f0))
            engine.publish_ready(prepared.token)
            export = engine.export_live_source(prepared.token, prepared.identity_bytes, prepared.semantic_sha256)
            events = engine._frames[prepared.token.sequence].events
            self.assertLessEqual(events["prepare_start_raw_ns"], events["neutral_ready_raw_ns"])
            self.assertLessEqual(events["neutral_ready_raw_ns"], events["numerical_ready_raw_ns"])
            self.assertEqual(export.semantic_sha256, prepared.semantic_sha256)
        finally:
            engine.close(abandon=True)


if __name__ == "__main__":
    unittest.main()
