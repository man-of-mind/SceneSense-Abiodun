"""CPU only scorer ownership tests for prospective scored live archives.

Synthetic event instants exercise protocol guards; they make no live timing or
authenticity claim. The scorer retains its score until an exact durable receipt.
"""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import time
import unittest

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from . import live_clock as CLOCK, live_receipt as LR, scorer as G
from .test_live_archive import source_sync, legacy_files
from .test_live_receipt import modeled_completion, PREFIX


RESERVATION = 32 * 1024 * 1024


def _event(key, semantic, name, instant):
    return dict(role="archive_client", name=name, raw_ns=instant, pid=os.getpid(),
        boot_id=CLOCK.read_boot_id(), clock_domain=CLOCK.CLOCK_DOMAIN,
        ready_identity=C._parse(key), ready_identity_sha256=C.sha256(key),
        semantic_sha256=semantic)


class LiveScorerArchiveTests(unittest.TestCase):
    def build(self, *, timeout=False):
        engine = G.ScorerEngine(run_binding="4" * 64, producer_binding="c" * 64,
                                max_jobs=2, max_bytes=2 * RESERVATION)
        case = fixture(engine.scorer)
        expected = dict(case["expected"], frame_context=PO.context_document(case["expected"]["frame_context"]))
        key = G.registration_key(expected)
        engine.register(key, expected=expected, reservation_bytes=RESERVATION)
        prediction = copy.deepcopy(case["prediction_kwargs"])
        prediction["labels"] = prediction["labels"].tobytes(order="C")
        prediction["shape"] = [720, 1280]
        engine.put_prediction(key, **prediction)
        engine.put_target(key, R.encode_target(case["target"]))
        score = engine.score_ready(key)
        self.assertIsNotNone(score)
        self.addCleanup(lambda: None if engine._closed else engine.close(abandon=True))
        stamped = engine._rows[key].scorer_events
        first_admission = min(raw_ns for name, raw_ns, _ in stamped if name.endswith("_admitted"))
        score_end = next(raw_ns for name, raw_ns, _ in stamped if name == "score_end")
        observed = first_admission - (250_000_000 if timeout else 5_000_000)
        sync = source_sync(case["target"], observed)
        start = observed - 2_000_000
        events = [("collector", "owned_tick_snapshot", start),
            ("producer", "source_stage_received", start + 250_000),
            ("producer", "prepare_start", start + 500_000),
            ("ue", "action_open", start + 600_000),
            ("producer", "source_alignment_observed", observed),
            ("producer", "numeric_ready", start + 3_000_000),
            ("ue", "last_datagram_sent", start + 4_000_000),
            ("producer", "ready_published", start + 5_000_000)]
        if timeout:
            events.append(("ue", "feedback_timeout", start + 600_000 + CLOCK.DEADLINE_NS + 1))
        else:
            self.assertLessEqual(score_end - (start + 600_000), CLOCK.DEADLINE_NS)
            events.extend((("coordinator", "feedback_emitted", score_end),
                           ("ue", "feedback_received", score_end)))
        ledger = CLOCK.LiveClockLedger(key, case["target"].semantic_sha256)
        boot = CLOCK.read_boot_id()
        for role, name, instant in events:
            ledger.record(role=role, name=name, raw_ns=instant, pid=os.getpid(), boot_id=boot,
                          ready_identity_bytes=key, semantic_sha256=case["target"].semantic_sha256)
        events = [copy.deepcopy(item) for item in ledger._events.values()]
        values = dict(input_metadata=case["source_kwargs"]["input_metadata"],
                      source_synchronization=sync,
                      numerical_avo=C.thaw(case["target"].metadata["avo"]), clock_events=events)
        return engine, case, key, values

    def receipt(self, case, files, prefix="frame_10/scorer/"):
        return dict(status="COMPLETE", durable=True, error=None, job_id=7,
            identity=case["expected"]["expected_capture_identity"],
            written=[dict(root=0, relative=prefix + name, sha256=hashlib.sha256(raw).hexdigest(),
                          bytes=len(raw), ready=False) for name, raw in files.items()])

    def confirm(self, engine, case, key, attached, *, receipt=None):
        if receipt is None:
            receipt = self.receipt(case, attached["files"])
        submitted = _event(key, attached["ready_semantic_sha256"], "archive_submitted",
                           time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        completed = _event(key, attached["ready_semantic_sha256"], "durable_completion_observed",
                           submitted["raw_ns"] + 1000)
        pending = engine.confirm_live_archive(key, receipt, "frame_10/scorer/",
            archive_submitted_event=submitted, durable_completion_event=completed)
        return pending, receipt

    def completion_proof(self, attached, pending, receipt):
        prefix = "frame_10/scorer/"
        selected = {item["relative"]: item for item in receipt["written"]}
        document = dict(schema="splitfusion.numeric_gt.live_receipt.v1", job_id=receipt["job_id"],
            root_index=0, relative_prefix=prefix,
            archive_manifest_sha256=attached["archive_manifest_sha256"],
            files={name: selected[name] for name in sorted(selected)},
            archive_receipt_sha256=C.sha256(C.canonical(receipt)),
            archive_receipt=receipt, clock_completion=C._parse(pending["clock_completion_bytes"]))
        raw = C.canonical(document)
        return dict(schema="splitfusion.numeric_gt.live_completion_durability_proof.v1",
            path="/tmp/owned-attempt/live_archive_completion.json", bytes=len(raw),
            sha256=C.sha256(raw), job_id=receipt["job_id"],
            receipt_sha256=C.sha256(C.canonical(receipt)),
            archive_manifest_sha256=attached["archive_manifest_sha256"],
            clock_completion_sha256=C.sha256(pending["clock_completion_bytes"]),
            clock_completion_bytes=len(pending["clock_completion_bytes"]),
            created_exclusive=True, file_fsynced=True, directory_fsynced=True,
            readback_verified=True, readback_bytes=raw)

    def test_on_time_scored_frame_requires_exact_durable_receipt_before_release(self):
        engine, case, key, values = self.build()
        attached = engine.attach_live_archive(key, **values)
        self.assertEqual(attached["terminal"], "FEEDBACK_ON_TIME")
        self.assertFalse(attached["live_qualified"])
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        pending, receipt = self.confirm(engine, case, key, attached)
        self.assertEqual(pending["status"], "COMPLETION_SIDECAR_PENDING_DURABILITY")
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        result = engine.confirm_live_completion_durable(key, self.completion_proof(attached, pending, receipt))
        self.assertEqual(result["status"], "DURABLE")
        self.assertEqual(result["terminal"], "FEEDBACK_ON_TIME")
        self.assertEqual(engine.snapshot()["retained_jobs"], 0)
        self.assertTrue(engine.close()["valid"])

    def test_scored_timeout_is_not_fabricated_as_feedback_success(self):
        engine, case, key, values = self.build(timeout=True)
        attached = engine.attach_live_archive(key, **values)
        self.assertEqual(attached["terminal"], "FEEDBACK_TIMEOUT")
        self.assertFalse(engine._rows[key].feedback)
        clock = C._parse(attached["clock_descriptor_bytes"])
        self.assertEqual(clock["deadline"]["terminal"], "FEEDBACK_TIMEOUT")
        self.assertEqual(clock["scoring_state"], "SCORED_NOT_EMITTED")
        pending, receipt = self.confirm(engine, case, key, attached)
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        result = engine.confirm_live_completion_durable(key, self.completion_proof(attached, pending, receipt))
        self.assertEqual(result["terminal"], "FEEDBACK_TIMEOUT")
        self.assertEqual(engine.snapshot()["retained_jobs"], 0)
        self.assertTrue(engine.close()["valid"])

    def test_missing_terminal_or_caller_forged_scorer_event_fails_closed(self):
        engine, _, key, values = self.build()
        no_terminal = [e for e in values["clock_events"]
                       if (e["role"], e["name"]) != ("ue", "feedback_received")]
        with self.assertRaises(ValueError):
            engine.attach_live_archive(key, **dict(values, clock_events=no_terminal))
        self.assertFalse(engine.snapshot()["valid"])
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        engine, _, key, values = self.build()
        foreign = copy.deepcopy(values["clock_events"])
        forged = copy.deepcopy(foreign[0])
        forged.update(role="scorer", name="score_end",
                      raw_ns=engine._rows[key].scorer_events[-1][1] - 1,
                      pid=os.getpid(), boot_id=CLOCK.read_boot_id())
        foreign.append(forged)
        with self.assertRaises(ValueError):
            engine.attach_live_archive(key, **dict(values, clock_events=foreign))
        self.assertFalse(engine.snapshot()["valid"])

    def test_source_clock_or_archive_receipt_corruption_never_releases_ticket(self):
        engine, _, key, values = self.build()
        wrong = copy.deepcopy(values["source_synchronization"])
        wrong["observed_raw_ns"] += 1
        with self.assertRaises(ValueError):
            engine.attach_live_archive(key, **dict(values, source_synchronization=wrong))
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        engine, case, key, values = self.build()
        attached = engine.attach_live_archive(key, **dict(values, legacy_artifacts=legacy_files(case)))
        receipt = self.receipt(case, attached["files"])
        receipt["written"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.confirm(engine, case, key, attached, receipt=receipt)
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        self.assertFalse(engine.snapshot()["valid"])

    def test_completion_sidecar_missing_or_changed_fails_closed_without_release(self):
        engine, case, key, values = self.build()
        attached = engine.attach_live_archive(key, **values)
        pending, receipt = self.confirm(engine, case, key, attached)
        with self.assertRaises(ValueError):
            engine.confirm_live_completion_durable(key, dict(
                self.completion_proof(attached, pending, receipt), readback_verified=False))
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        self.assertFalse(engine.snapshot()["valid"])
        engine, case, key, values = self.build()
        attached = engine.attach_live_archive(key, **values)
        pending, receipt = self.confirm(engine, case, key, attached)
        forged = self.completion_proof(attached, pending, receipt)
        forged["clock_completion_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            engine.confirm_live_completion_durable(key, forged)
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        self.assertFalse(engine.snapshot()["valid"])

    def test_duplicate_archive_receipt_before_sidecar_durability_retains_ticket(self):
        engine, case, key, values = self.build()
        attached = engine.attach_live_archive(key, **values)
        self.confirm(engine, case, key, attached)
        with self.assertRaises(G.ScorerError):
            self.confirm(engine, case, key, attached)
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        self.assertFalse(engine.snapshot()["valid"])

    def test_duplicate_sidecar_proof_after_release_is_refused(self):
        engine, case, key, values = self.build()
        attached = engine.attach_live_archive(key, **values)
        pending, receipt = self.confirm(engine, case, key, attached)
        proof = self.completion_proof(attached, pending, receipt)
        engine.confirm_live_completion_durable(key, proof)
        with self.assertRaises(G.ScorerError):
            engine.confirm_live_completion_durable(key, proof)
        self.assertEqual(engine.snapshot()["retained_jobs"], 0)

    def test_scorer_and_archive_helper_complete_one_real_sidecar_write(self):
        engine, _, key, values = self.build()
        attached = engine.attach_live_archive(key, **values)
        reply = dict(attached, files=dict(attached["files"]), sha256=dict(attached["sha256"]))
        archive = LR.archive_from_scorer_reply(key, reply)
        with tempfile.TemporaryDirectory(prefix="scorer-live-proof-", dir="/tmp") as temporary:
            root, sidecars = Path(temporary) / "archive", Path(temporary) / "sidecars"
            root.mkdir()
            sidecars.mkdir()
            client, receipt = modeled_completion(root, archive)
            submitted = LR.observe_archive_submission(client, receipt["job_id"], archive,
                relative_prefix=PREFIX, root_index=0)
            client.pending.clear()
            client.completions.append(receipt)
            completed = LR.observe_durable_completion(client, receipt, archive,
                relative_prefix=PREFIX, root_index=0)
            pending = engine.confirm_live_archive(key, receipt, PREFIX,
                archive_submitted_event=submitted, durable_completion_event=completed)
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
            proof = LR.persist_live_completion(client, receipt, archive, pending,
                submitted, completed, relative_prefix=PREFIX, root_index=0,
                sidecar_directory=sidecars)
            result = engine.confirm_live_completion_durable(key, proof)
            self.assertEqual(result["status"], "DURABLE")
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)
            self.assertTrue(engine.close()["valid"])


if __name__ == "__main__":
    unittest.main()
