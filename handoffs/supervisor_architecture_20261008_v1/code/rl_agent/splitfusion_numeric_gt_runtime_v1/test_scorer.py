"""CPU-only child-engine lifecycle/admission tests; synthetic durability only."""
import copy
from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture, forbidden
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from . import scorer as G

RESERVATION = 32 * 1024 * 1024


class ScorerEngineTests(unittest.TestCase):
    def build(self):
        engine = G.ScorerEngine(run_binding="4" * 64, producer_binding="c" * 64,
                               max_jobs=2, max_bytes=2 * RESERVATION)
        case = fixture(engine.scorer)
        expected = dict(case["expected"], frame_context=PO.context_document(case["expected"]["frame_context"]))
        key = G.registration_key(expected)
        engine.register(key, expected=expected, reservation_bytes=RESERVATION)
        self.addCleanup(lambda: None if engine._closed else engine.close(abandon=True))
        return engine, case, key

    def prediction(self, engine, case, key, **changes):
        kw = copy.deepcopy(case["prediction_kwargs"])
        kw.update(labels=kw["labels"].tobytes(), shape=[720, 1280]); kw.update(changes)
        return engine.put_prediction(key, **kw)

    def target(self, engine, case, key):
        return engine.put_target(key, R.encode_target(case["target"]))

    def scored(self):
        engine, case, key = self.build()
        self.prediction(engine, case, key); self.target(engine, case, key)
        result = engine.score_ready(key)
        return engine, case, key, result

    def archive_kwargs(self, case):
        target = case["prepared"].target
        return dict(ready_semantic_sha256=case["target"].semantic_sha256,
            input_metadata=case["source_kwargs"]["input_metadata"], avo_document=case["source_kwargs"]["avo_document"],
            legacy_target_meta={k: v for k, v in target.items() if k not in
                ("segmentation", "object_ignore", "segmentation_png", "object_ignore_png", "_csv_bytes")},
            segmentation_png=target["segmentation_png"], object_ignore_png=target["object_ignore_png"],
            old_prepared_sha256=case["prepared"].content_sha256)

    def receipt(self, case, files, prefix="frame_10/scorer/"):
        # Shape of a real ArchiveClient completion; no actual disk write here.
        return dict(status="COMPLETE", durable=True, identity=case["expected"]["expected_capture_identity"],
            written=[dict(root=0, relative=prefix + name, sha256=hashlib.sha256(data).hexdigest(),
                          bytes=len(data), ready=False) for name, data in files.items()])

    def test_both_arrival_orders_and_score_once(self):
        for order in ("target", "prediction"):
            engine, case, key = self.build()
            if order == "target":
                self.target(engine, case, key)
                self.assertIsNone(engine.score_ready(key))
                self.prediction(engine, case, key)
            else:
                self.prediction(engine, case, key)
                self.assertIsNone(engine.score_ready(key))
                self.target(engine, case, key)
            result = engine.score_ready(key)
            with patch.dict(engine.scorer._detection, match_frame=forbidden), \
                 patch.dict(engine.scorer._person, greedy_match=forbidden):
                self.assertIs(engine.score_ready(key), result)
            self.assertEqual(engine.snapshot()["scored"], 1)
            with self.assertRaises(TypeError):
                result["q_perc"] = 0.
            engine.close(abandon=True)

    def test_actual_legacy_archive_bytes_no_rematch_and_receipt_releases(self):
        engine, case, key, result = self.scored()
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        engine.confirm_feedback(key)
        with patch.dict(engine.scorer._detection, match_frame=forbidden, load_gt=forbidden, load_predictions=forbidden), \
             patch.dict(engine.scorer._person, greedy_match=forbidden):
            packet = engine.attach_archive(key, **self.archive_kwargs(case))
        oracle = E.evaluate_prepared(identity=case["prediction_kwargs"]["identity"]["action_identity"],
            prepared=case["prepared"], predicted_labels=case["prediction_kwargs"]["labels"],
            predictions=case["prediction_kwargs"]["records"], avo=case["source_kwargs"]["avo_document"]["avo"])
        self.assertEqual(dict(packet["files"]), E.archive_payloads(oracle))
        self.assertEqual(C.thaw(result["measurement"]), oracle.measurement)
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        with self.assertRaises(TypeError):
            packet["files"]["evaluation.json"] = b"mutated"
        receipt = self.receipt(case, packet["files"])
        engine.confirm_archive(key, receipt, "frame_10/scorer/")
        self.assertEqual(engine.snapshot()["retained_jobs"], 0)
        self.assertEqual(engine.snapshot()["reserved_bytes"], 0)
        self.assertEqual(engine.snapshot()["join"]["pending_jobs"], 0)
        self.assertTrue(engine.close()["valid"])

    def test_duplicate_inputs_idempotent_but_conflicting_prediction_sticky(self):
        engine, case, key = self.build()
        self.assertTrue(self.target(engine, case, key)); self.assertFalse(self.target(engine, case, key))
        self.assertTrue(self.prediction(engine, case, key)); self.assertFalse(self.prediction(engine, case, key))
        changed = copy.deepcopy(case["prediction_kwargs"])
        changed["records"][0]["world_x"] += 1
        changed["identity"]["prediction_digest"] = hashlib.sha256(changed["labels"].tobytes()
            + Q.S.canonical(changed["records"])).hexdigest()
        with self.assertRaises(RuntimeError):
            self.prediction(engine, case, key, **{k: v for k, v in changed.items() if k != "labels"})
        self.assertFalse(engine.snapshot()["valid"])
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)

    def test_foreign_registration_and_target_and_full_action_fail_closed(self):
        engine, case, key = self.build()
        with self.assertRaises(ValueError):
            engine.register(key, expected=dict(case["expected"], run_binding="9" * 64), reservation_bytes=RESERVATION)
        self.assertFalse(engine.snapshot()["valid"])
        engine, case, key = self.build()
        other = R.admit_payload(case["payload"], **dict(case["expected"], plan_sha256="9" * 64))
        with self.assertRaises(ValueError):
            engine.put_target(key, R.encode_target(other))
        self.assertFalse(engine.snapshot()["valid"])
        engine, case, key = self.build()
        identity = copy.deepcopy(case["prediction_kwargs"]["identity"])
        identity["action_identity"]["q_e4"] += 1
        with self.assertRaises(RuntimeError):
            self.prediction(engine, case, key, identity=identity)
        self.assertFalse(engine.snapshot()["valid"])

    def test_no_pickled_or_mutable_prediction_buffers_accepted(self):
        for wrong in (bytearray(720 * 1280), b"pickle-looking\x80", None):
            engine, case, key = self.build()
            wrong = case["prediction_kwargs"]["labels"] if wrong is None else wrong
            with self.assertRaises(ValueError):
                self.prediction(engine, case, key, labels=wrong)
            self.assertFalse(engine.snapshot()["valid"])

    def test_early_feedback_and_archive_prohibited(self):
        engine, case, key = self.build()
        with self.assertRaisesRegex(G.ScorerError, "precedes"):
            engine.confirm_feedback(key)
        engine, case, key, result = self.scored()
        with self.assertRaisesRegex(G.ScorerError, "precedes"):
            engine.attach_archive(key, **self.archive_kwargs(case))
        self.assertEqual(result["q_perc"], result["q_perc"])

    def test_archive_corruption_does_not_replace_numerical_quality(self):
        changes = (dict(old_prepared_sha256="0" * 64), dict(segmentation_png=b"invalid png"),
                   dict(ready_semantic_sha256="0" * 64))
        for change in changes:
            engine, case, key, result = self.scored(); engine.confirm_feedback(key)
            before = C.canonical(result)
            with self.assertRaises(ValueError):
                engine.attach_archive(key, **{**self.archive_kwargs(case), **change})
            self.assertEqual(C.canonical(result), before)
            self.assertFalse(engine.snapshot()["valid"])
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)

    def test_archive_receipt_requires_actual_complete_identity_hashes_bytes(self):
        changes = (lambda r: r.update(status="QUEUED"), lambda r: r.update(durable=False),
            lambda r: r["identity"].update(frame_id=99), lambda r: r["written"].pop(),
            lambda r: r["written"][0].update(sha256="0" * 64),
            lambda r: r["written"][0].update(bytes=0),
            lambda r: r["written"][0].update(root="not-an-archive-root-index"),
            lambda r: r["written"][0].update(root=True),
            lambda r: r["written"][0].update(ready=True),
            lambda r: r["written"].append(dict(r["written"][0])))
        for change in changes:
            engine, case, key, _ = self.scored(); engine.confirm_feedback(key)
            packet = engine.attach_archive(key, **self.archive_kwargs(case))
            receipt = copy.deepcopy(self.receipt(case, packet["files"])); change(receipt)
            with self.assertRaises(ValueError):
                engine.confirm_archive(key, receipt, "frame_10/scorer/")
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
            self.assertFalse(engine.snapshot()["valid"])

    def test_registered_capacity_no_eviction_and_undrained_close_fails(self):
        engine, case, key = self.build()
        with self.assertRaises(G.ScorerError):
            engine.close()
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        self.assertFalse(engine.close(abandon=True)["valid"])
        engine, case, key = self.build()
        with self.assertRaises((ValueError, G.ScorerCapacityError)):
            engine.register(b"unregistered", expected=case["expected"], reservation_bytes=2 * RESERVATION)
        self.assertEqual(engine.snapshot()["retained_jobs"], 1)

    def test_primitive_frame_context_required_and_registered_key_exact(self):
        engine, case, key = self.build()
        self.assertEqual(key, case["target"].identity_bytes)
        expected = dict(case["expected"], frame_context=PO.context_document(case["expected"]["frame_context"]))
        expected["frame_context"]["sequence_id"] += 1
        with self.assertRaises(ValueError):
            G.registration_key(expected)


if __name__ == "__main__":
    unittest.main()
