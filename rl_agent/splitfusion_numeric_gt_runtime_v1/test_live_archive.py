"""CPU only prospective archive tests, with synthetic and retained numerical frames."""
import copy
from dataclasses import replace
import json
import os
import time
import unittest

import numpy as np

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R, scoring as Q
from rl_agent.splitfusion_numeric_gt_integration_v2.test_scoring import fixture
from rl_agent.splitfusion_numeric_gt_integration_v2.test_ready import redigest
from rl_agent.splitfusion_minimal_quality_scorer_v1 import scorer as S
from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O, payload as P
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_run5_live_avo_v1 import gt_source_sync as SY
from . import live_archive as G, live_clock as CLOCK


def source_sync(target, observed_raw_ns):
    meta = C.thaw(target.metadata)
    frame = meta["frame"]
    stamp = dict(frame_id=frame["frame_id"], timestamp=frame["timestamp"],
                 timestamp_repr=repr(frame["timestamp"]), timestamp_hex=float(frame["timestamp"]).hex())
    camera = meta["camera_matrix"]
    image = dict(stamp, width=1280, height=720, camera_matrix=camera,
                 callback_receipt_raw_ns=observed_raw_ns - 100)
    record = dict(schema=SY.SYNC_SCHEMA, identity=meta["capture_identity"],
        sources=dict(rgb=image, semantic=dict(image), depth=dict(image),
            radar=dict(stamp, width=None, height=None, camera_matrix=camera,
                       callback_receipt_raw_ns=observed_raw_ns - 100),
            policy_radar=stamp,
            frozen_world=dict(stamp, snapshot_observed_raw_ns=observed_raw_ns - 200)),
        timestamp_deltas_vs_rgb_s=dict.fromkeys(("rgb", "semantic", "depth", "radar",
                                                   "policy_radar", "frozen_world"), 0.),
        pose_deltas_vs_rgb={role: dict(max_abs_matrix_delta=0., translation_delta_m=0.)
                             for role in ("semantic", "depth")},
        same_frame_required=True, timestamp_tolerance_s=0., pose_matrix_atol=SY.POSE_ATOL,
        observed_raw_ns=observed_raw_ns)
    SY.validate_source_alignment(record)
    return record


def clock_for(target, sync):
    ledger = CLOCK.LiveClockLedger(target.identity_bytes, target.semantic_sha256)
    start = sync["observed_raw_ns"] - 2_000_000
    boot = CLOCK.read_boot_id()
    pid = os.getpid()
    events = (("collector", "owned_tick_snapshot", start),
        ("producer", "source_stage_received", start + 250_000),
        ("producer", "prepare_start", start + 500_000),
        ("ue", "action_open", start + 600_000),
        ("producer", "source_alignment_observed", sync["observed_raw_ns"]),
        ("producer", "numeric_ready", start + 3_000_000),
        ("ue", "last_datagram_sent", start + 4_000_000),
        ("producer", "ready_published", start + 5_000_000),
        ("scorer", "target_admitted", start + 6_000_000),
        ("scorer", "prediction_admitted", start + 7_000_000),
        ("scorer", "score_start", start + 8_000_000),
        ("scorer", "score_end", start + 9_000_000),
        ("coordinator", "feedback_emitted", start + 10_000_000),
        ("ue", "feedback_received", start + 11_000_000),
        ("scorer", "archive_build_start", start + 12_000_000))
    for role, name, instant in events:
        ledger.record(role=role, name=name, raw_ns=instant, pid=pid, boot_id=boot,
            ready_identity_bytes=target.identity_bytes, semantic_sha256=target.semantic_sha256)
    return ledger


def legacy_files(case):
    target = case["target"]
    builder = O.bound().builder
    manifest = case["reference_kwargs"]["target_tables"]["target_manifest.csv"]
    manifest_fields = ("sample_id", "segmentation_mask_path", "segmentation_mask_sha256",
                       "object_ignore_mask_path", "object_ignore_mask_sha256")
    return dict(
        **{"segmentation.png": case["prepared"].target["segmentation_png"],
           "object_ignore.png": case["prepared"].target["object_ignore_png"],
           "object_boxes.csv": P.legacy_csv_bytes(target.metadata["object_boxes"], builder.OBJECT_FIELDS),
           "object_ignore_regions.csv": P.legacy_csv_bytes(target.metadata["object_ignore_regions"],
                                                             builder.IGNORE_FIELDS),
           "target_manifest.csv": P.legacy_csv_bytes(manifest, manifest_fields)})


class LiveArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = S.NumericalScorer()
        cls.spec, _ = quality_bridge()

    def values(self, *, undefined=False):
        case = fixture(self.scorer)
        if undefined:
            payload = replace(case["payload"], object_boxes=(), avo=P.freeze(dict(
                frame_id=10, timestamp=10., threshold=.65, qualified=[], structural_ignored=[])))
            target = R.admit_payload(redigest(payload), **case["expected"])
            case["target"] = target
        target = case["target"]
        frame = Q.join_admitted_inputs(self.scorer, target=target, prediction=case["prediction"])
        result = self.scorer.score(frame)
        from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
        measured = quality.evaluate_exact_quality(self.spec, result.measurement)
        q = None if measured is None else float(measured.q_perc)
        sync = source_sync(target, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        kwargs = dict(ready_target=target, score_result=result, q_perc=q, quality_spec=self.spec,
            input_metadata=case["source_kwargs"]["input_metadata"],
            source_synchronization=sync, numerical_avo=C.thaw(target.metadata["avo"]),
            clock_ledger=clock_for(target, sync))
        return case, kwargs

    def test_live_schema_binds_exact_numeric_score_native_bytes_and_observed_clock(self):
        case, kwargs = self.values()
        value = G.build_live_archive(**kwargs)
        verified = G.verify_live_archive(value)
        archive = json.loads(value.files["live_archive.json"])
        score = json.loads(value.files["numerical_score.json"])
        clock = json.loads(value.files["live_clock.json"])
        self.assertEqual(verified["schema"], G.SCHEMA)
        self.assertEqual(score["measurement"], kwargs["score_result"].measurement)
        self.assertEqual(score["q_perc"], kwargs["q_perc"])
        self.assertEqual(score["ready_identity"], C._parse(case["target"].identity_bytes))
        self.assertEqual(archive["ready_semantic_sha256"], case["target"].semantic_sha256)
        self.assertEqual(value.files["segmentation.u8"], case["target"].segmentation_bytes)
        self.assertEqual(value.files["predicted_labels.u8"], case["prediction"].labels)
        self.assertEqual(clock["schema"], CLOCK.SCHEMA)
        self.assertEqual(clock["deadline"]["within_deadline"], True)
        self.assertFalse(any(event["name"] == "archive_build_end" for event in clock["events"]))
        self.assertEqual(archive["terminal_coverage"], "SCORED_TICKETS_ONLY")
        self.assertFalse(archive["controller_terminal_claim"])
        self.assertNotIn("avo_document_sha256", archive)
        self.assertNotIn("old_prepared_sha256", archive)
        self.assertNotIn("AVO_PRE_SCORE_INPUTS", str(archive) + str(clock))

    def test_undefined_quality_is_recorded_without_zero_or_fabricated_reward(self):
        _, kwargs = self.values(undefined=True)
        self.assertIsNone(kwargs["q_perc"])
        value = G.build_live_archive(**kwargs)
        archive = C._parse(value.manifest_bytes)
        score = C._parse(value.files["numerical_score.json"])
        self.assertIsNone(archive["q_perc"])
        self.assertFalse(archive["quality_defined"])
        self.assertIsNone(score["q_perc"])
        self.assertFalse(score["quality_defined"])
        self.assertTrue(G.verify_live_archive(value)["schema"] == G.SCHEMA)

    def test_optional_original_png_and_tables_require_numerical_parity(self):
        case, kwargs = self.values()
        legacy = legacy_files(case)
        value = G.build_live_archive(**dict(kwargs, legacy_artifacts=legacy))
        self.assertEqual(value.files["segmentation.png"], legacy["segmentation.png"])
        self.assertEqual(value.files["target_manifest.csv"], legacy["target_manifest.csv"])
        self.assertEqual(G.verify_live_archive(value)["files"], 15)
        for name, altered in (("segmentation.png", b"bad"),
                              ("object_boxes.csv", b"bad"),
                              ("target_manifest.csv", legacy["target_manifest.csv"].replace(b"10", b"99"))):
            _, fresh = self.values()
            broken = dict(legacy, **{name: altered})
            with self.subTest(name=name), self.assertRaises(ValueError):
                G.build_live_archive(**dict(fresh, legacy_artifacts=broken))

    def test_mismatch_or_historical_avo_document_cannot_become_live_archive(self):
        case, base = self.values()
        changes = []
        sync = copy.deepcopy(base["source_synchronization"])
        sync["identity"]["frame_id"] += 1
        changes.append(dict(source_synchronization=sync))
        sync = copy.deepcopy(base["source_synchronization"])
        sync["sources"]["rgb"]["camera_matrix"][0][3] += 1
        changes.append(dict(source_synchronization=sync))
        avo = copy.deepcopy(base["numerical_avo"])
        avo["threshold"] = .5
        changes.append(dict(numerical_avo=avo))
        changes.append(dict(numerical_avo=case["source_kwargs"]["avo_document"]))
        metadata = copy.deepcopy(base["input_metadata"])
        metadata["provenance"]["evidence_sha256"]["owned_source"] = "0" * 64
        changes.append(dict(input_metadata=metadata))
        changes.append(dict(q_perc=0.0 if base["q_perc"] is None else base["q_perc"] + .01))
        for index, change in enumerate(changes):
            _, fresh = self.values()
            with self.subTest(index=index), self.assertRaises(ValueError):
                G.build_live_archive(**dict(fresh, **change))

    def test_wrong_clock_event_and_mutated_output_refused(self):
        _, base = self.values()
        wrong_sync = copy.deepcopy(base["source_synchronization"])
        wrong_sync["observed_raw_ns"] += 1
        with self.assertRaisesRegex(G.LiveArchiveError, "RAW instant differs"):
            G.build_live_archive(**dict(base, source_synchronization=wrong_sync))
        _, fresh = self.values()
        value = G.build_live_archive(**fresh)
        broken = dict(value.files)
        broken["predicted_labels.u8"] = b"wrong"
        with self.assertRaisesRegex(G.LiveArchiveError, "inventory/hash differs"):
            G.verify_live_archive(replace(value, files=type(value.files)(broken)))

    def test_retained_native_frame_matches_original_numeric_oracle(self):
        # One sealed live-derived frame exercises the actual source producer,
        # camera/AVO admission and full native prediction. Clock events here
        # are test observations; this test makes no live timing claim.
        from rl_agent.splitfusion_numeric_gt_integration_v2 import composition as COM, replay as H, fixtures as F
        from rl_agent.splitfusion_numeric_gt_integration_v2.ownership import ProducerOwner
        from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K, source as NS, artifacts as AR
        from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import KEYS
        from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E, target_transport as T

        data = H.load_inputs(1)
        case = data["cases"][0]
        pred = data["predictions"][case.frame_id]
        binding = "4" * 64
        expected = H.expectation(data, case, binding)
        plan = F.plan_for(case, pred, expected["frame_context"])
        kwargs = {name: case.kwargs[name] for name in KEYS}
        producer = NS.NumericSourceProducer(**K.producer_arguments(case))
        owner = ProducerOwner(producer, max_retained_jobs=2, max_retained_bytes=128 * 1024 * 1024)
        try:
            COM.register_population(owner, case)
            token = owner.compute(plan, kwargs, reserved_bytes=64 * 1024 * 1024)
            staged = owner._staged_for_assembly(token)
            numeric, avo, payload = COM.assemble(staged, kwargs)
            owner.stage_products(token, target=numeric, avo=avo, payload=payload)
            target = R.admit_payload(payload, **expected)
            owner.set_ready(token, target)
            owner.confirm_send(token, plan)
            admitted = Q.admit_prediction(self.scorer, identity=pred["identity"], labels=pred["labels"],
                records=pred["records"], prediction_contract=expected["prediction_contract"])
            result = self.scorer.score(Q.join_admitted_inputs(self.scorer, target=target, prediction=admitted))
            from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality
            measured = quality.evaluate_exact_quality(self.spec, result.measurement)
            q = None if measured is None else float(measured.q_perc)
            self.assertEqual(result.measurement, pred["evidence"]["measurement"])
            self.assertEqual(q, pred["q_perc"])
            source_sync = staged.sync
            old = AR.legacy_target(staged, numeric)
            attachment = dict(old["_csv_bytes"], **{
                "segmentation.png": old["segmentation_png"], "object_ignore.png": old["object_ignore_png"]})
            value = G.build_live_archive(ready_target=target, score_result=result, q_perc=q,
                quality_spec=self.spec, input_metadata=T._input_metadata(staged.inputs),
                source_synchronization=source_sync, numerical_avo=C.thaw(target.metadata["avo"]),
                clock_ledger=clock_for(target, source_sync), legacy_artifacts=attachment)
            self.assertEqual(value.files["segmentation.png"], case.segmentation_png)
            self.assertEqual(value.files["object_ignore.png"], case.object_ignore_png)
            self.assertEqual(G.verify_live_archive(value)["quality_defined"], q is not None)
            oracle = E.evaluate_prepared(identity=pred["identity"]["action_identity"],
                prepared=case.bundle.prepared, predicted_labels=pred["labels"],
                predictions=pred["records"], avo=case.avo_document["avo"])
            self.assertEqual(result.measurement, oracle.measurement)
            owner.complete_scoring(token)
            owner.confirm_feedback(token)
            owner.complete_archive(token)
            owner.close()
            self.assertTrue(owner.snapshot()["valid"])
        finally:
            if not owner.snapshot()["closed"]:
                owner.close(abandon=True)


if __name__ == "__main__":
    unittest.main()
