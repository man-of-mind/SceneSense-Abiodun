"""CPU-only actual producer transactions and explicitly offline artifact replay.

One test invokes the existing stdlib archive child and verifies real durable
files. Other receipt fault tests use deliberately synthetic receipt structures.
No numerical RPC, simulator, CUDA, service, or retained-evidence write occurs.
"""
from contextlib import contextmanager, ExitStack
import copy
import io
import json
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import carla
import numpy as np

from rl_agent.splitfusion_live_gt_handoff_v1 import archive as DISK
from rl_agent.splitfusion_live_parity_repair_v1 import test_source_geometry as F, evaluation as E, source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v2 import prediction_archive as PA
from rl_agent.splitfusion_numeric_gt_producer_v1 import artifacts as AR, originals as O
from rl_agent.splitfusion_numeric_gt_producer_v1.test_numeric_producer import ARGS, upright
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as R
from rl_agent.splitfusion_numeric_gt_integration_v2.test_ownership import plan_for, context_for, advance
from . import producer as G


CLOCKS = {name: 100 + index for index, name in enumerate(AR.AVO_CLOCKS)}
FILENAMES = dict(depth_bgra="avo_depth.npy", rawsemantic="avo_semantic.npy")


@contextmanager
def fixture(*, rgb=False, **options):
    with F.SourceGeometryTests().fixture() as (directory, kwargs):
        kwargs = upright(kwargs)
        pose = carla.Transform(carla.Location(1.8, 0., 1.55), carla.Rotation(pitch=-4.))
        camera = np.asarray(pose.get_matrix())
        kwargs["camera_matrix"], kwargs["camera_inverse"] = camera, np.linalg.inv(camera)
        kwargs["rgb"]["camera_matrix"] = camera.tolist()
        for name in ("semantic_image", "depth_image"):
            kwargs[name].transform = pose
        kwargs["dynamic_snapshot_evidence"].update(registered_population_ids=[1, 2], population_sha256="d" * 64)
        if rgb:
            kwargs["rgb"]["_rgb_image"] = SimpleNamespace(raw_data=bytearray(720 * 1280 * 4),
                width=1280, height=720, frame=kwargs["identity"]["frame_id"],
                timestamp=kwargs["rgb"]["timestamp"], transform=pose)
        init = dict(producer_args=dict(ARGS, static_catalogue_dir=directory / "catalogue"),
            run_binding="4" * 64, producer_binding=O.bound().binding_sha256,
            prediction_contract=PA.native_prediction_contract().document(), max_frames=8, offline_replay=True)
        init.update(options)
        engine = G.ProducerEngine(**init)
        try:
            yield engine, kwargs, directory
        finally:
            engine.close(abandon=True)


def prepare(engine, kwargs, **changes):
    options = dict(frame_context=context_for(kwargs), population=kwargs["dynamic_snapshot_evidence"],
        archive_clocks=CLOCKS, payload_filenames=FILENAMES, prescore_filename="avo_inputs.json",
        replay_observed_raw_ns=101)
    options.update(changes)
    return engine.prepare(plan_for(kwargs)[0], kwargs, **options)


def commit(engine, kwargs):
    pending = prepare(engine, kwargs)
    engine.confirm_send(pending.token, plan_for(kwargs)[0])
    engine.publish_ready(pending.token)
    return pending


def score(engine, pending):
    engine.score_completed(pending.token, pending.identity_bytes, pending.semantic_sha256)
    engine.feedback_confirmed(pending.token, pending.identity_bytes, pending.semantic_sha256)


def receipt(pending, archive, prefix="frame_100/producer"):
    # Deliberate synthetic receipt for protocol fault tests, not fsync evidence.
    return dict(status="COMPLETE", durable=True, error=None,
        identity=C._parse(pending.identity_bytes)["capture_identity"],
        written=[dict(root=0, relative=prefix + "/" + name, sha256=C.sha256(raw), bytes=len(raw), ready=False)
                 for name, raw in archive.files.items()])


def forbidden(*args, **kwargs):
    raise AssertionError("legacy archive/PNG operation before deferred archive phase")


class ProducerEngineTests(unittest.TestCase):
    def test_short_runtime_budget_preserves_registered_source_history_and_original_prepared_digest(self):
        with fixture(max_frames=2) as (engine, kwargs, directory), patch("time.clock_gettime_ns", return_value=101):
            original = S.LiveCompleteTargetProducer(static_catalogue_dir=directory / "catalogue", **ARGS)
            expected = E.prepare_target(original.build(**kwargs))
            pending = commit(engine, kwargs); score(engine, pending)
            archive = engine.build_archive(pending.token)
            descriptor = json.loads(archive.descriptor_bytes)
            self.assertEqual(engine._owner._producer.max_frames, ARGS["max_frames"])
            self.assertEqual(engine._owner._producer.conflict_state.max_frames, ARGS["max_frames"])
            self.assertEqual(engine.snapshot()["runtime_max_frames"], 2)
            self.assertEqual(engine.snapshot()["producer_max_frames"], ARGS["max_frames"])
            self.assertEqual(descriptor["old_prepared_sha256"], expected.content_sha256)
            self.assertEqual(archive.files["segmentation.png"], expected.target["segmentation_png"])
            self.assertEqual(descriptor["legacy_target_meta"]["tables"], expected.target["tables"])

    def test_invalid_registered_bound_or_runtime_over_capacity_rejected_before_producer_creation(self):
        for source_limit, runtime_limit in ((None, 1), (True, 1), (0, 1), (1001, 1), ("100", 1), (2, 3)):
            with self.subTest(source_limit=source_limit, runtime_limit=runtime_limit), \
                 patch.object(G.NS, "NumericSourceProducer", side_effect=AssertionError("must reject configuration first")):
                with self.assertRaises(G.ProducerError):
                    G.ProducerEngine(producer_args=dict(ARGS, max_frames=source_limit),
                        run_binding="4" * 64, producer_binding=O.bound().binding_sha256,
                        prediction_contract=PA.native_prediction_contract().document(), max_frames=runtime_limit)

    def test_runtime_request_budget_stays_separate_and_is_enforced(self):
        with fixture(max_frames=1) as (engine, kwargs, _):
            pending = prepare(engine, kwargs)
            engine.cancel(pending.token, plan_for(kwargs)[0], "not sent")
            advance(kwargs)
            with patch.object(engine._owner._producer, "compute", wraps=engine._owner._producer.compute) as compute:
                with self.assertRaisesRegex(G.ProducerError, "runtime frame budget exhausted"):
                    prepare(engine, kwargs)
                self.assertEqual(compute.call_count, 0)
            self.assertEqual(engine._owner._producer.max_frames, ARGS["max_frames"])
            self.assertEqual(engine._owner._producer.processed_frames, 0)

    def test_numerical_prepare_and_real_commit_precede_all_archive_operations(self):
        with fixture() as (engine, kwargs, _), ExitStack() as guards:
            for name in ("legacy_target", "prepared_target", "avo_documents"):
                guards.enter_context(patch.object(AR, name, forbidden))
            for name in ("imencode", "imdecode", "imread"):
                guards.enter_context(patch("cv2." + name, forbidden))
            producer = engine._owner._producer
            before = producer.history_sha256()
            pending = prepare(engine, kwargs)
            self.assertEqual(producer.processed_frames, 0)
            with patch.object(producer, "commit", wraps=producer.commit) as committed:
                engine.confirm_send(pending.token, plan_for(kwargs)[0])
                engine.confirm_send(pending.token, plan_for(kwargs)[0])
                self.assertEqual(committed.call_count, 1)
            packet = engine.publish_ready(pending.token)
            self.assertIs(engine.publish_ready(pending.token), packet)
            received = R.decode_target(packet, expected_capture_identity=kwargs["identity"],
                expected_action_identity=plan_for(kwargs)[1], frame_context=context_for(kwargs),
                plan_sha256=C.sha256(plan_for(kwargs)[0]), run_binding="4" * 64,
                producer_binding=O.bound().binding_sha256,
                prediction_contract=PA.native_prediction_contract().document())
            self.assertEqual(received.identity_bytes, pending.identity_bytes)
            self.assertEqual(received.semantic_sha256, pending.semantic_sha256)
            self.assertEqual(received.metadata["history_before_sha256"], before)
            self.assertEqual(received.metadata["history_after_sha256"], producer.history_sha256())
            self.assertEqual(engine.snapshot()["owner"]["commit_calls"], 1)

    def test_cancel_uses_real_rollback_once_including_population_registration(self):
        with fixture() as (engine, kwargs, _):
            producer = engine._owner._producer
            before = producer.history_sha256()
            kwargs["dynamic_snapshot_evidence"].update(registered_population_ids=[1, 2, 3], population_sha256="e" * 64)
            pending = prepare(engine, kwargs)
            with patch.object(producer, "cancel", wraps=producer.cancel) as cancelled:
                engine.cancel(pending.token, plan_for(kwargs)[0], "accepted send cancelled")
                self.assertEqual(cancelled.call_count, 1)
            self.assertEqual(producer.history_sha256(), before)
            self.assertEqual(producer.population, frozenset([1, 2]))
            self.assertIsNone(producer.pending_registration)
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)
            self.assertEqual(engine.snapshot()["events"]["1"]["outcome"], "CANCELLED")
            self.assertTrue(engine.close()["valid"])

    def test_foreign_full_plan_cannot_commit_and_cancel_remains_available(self):
        with fixture() as (engine, kwargs, _):
            pending = prepare(engine, kwargs)
            changed = C._parse(plan_for(kwargs)[0]); changed["action"]["q_e4"] += 1
            with self.assertRaisesRegex(G.ProducerError, "exact full plan"):
                engine.confirm_send(pending.token, C.canonical(changed))
            self.assertEqual(engine._owner._producer.processed_frames, 0)
            engine.cancel(pending.token, plan_for(kwargs)[0], "foreign confirmation")
            self.assertFalse(engine.snapshot()["valid"])

    def test_publish_before_send_is_failstop_and_real_pending_cancelable(self):
        with fixture() as (engine, kwargs, _):
            pending = prepare(engine, kwargs)
            with self.assertRaisesRegex(G.ProducerError, "precedes"):
                engine.publish_ready(pending.token)
            self.assertEqual(engine._owner._producer.processed_frames, 0)
            engine.cancel(pending.token, plan_for(kwargs)[0], "not sent")

    def test_wrong_context_admission_failure_rolls_back_real_transaction(self):
        with fixture() as (engine, kwargs, _):
            producer = engine._owner._producer
            before = producer.history_sha256()
            foreign = dict(kwargs, identity=dict(kwargs["identity"]))
            foreign["identity"]["capture_timestamp_ns"] += 1
            with patch.object(producer, "cancel", wraps=producer.cancel) as cancelled:
                with self.assertRaises(ValueError):
                    prepare(engine, kwargs, frame_context=context_for(foreign))
                self.assertEqual(cancelled.call_count, 1)
            self.assertEqual(producer.history_sha256(), before)
            self.assertFalse(engine.snapshot()["valid"])
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)

    def test_original_prepared_artifacts_and_immutable_owned_rgb(self):
        with fixture(rgb=True) as (engine, kwargs, _):
            pending = commit(engine, kwargs)
            row = engine._frames[pending.token.sequence]
            staged, _, _, _ = engine._owner._archive_for_owner(row.owner_token)
            original = E.prepare_target(staged.inputs)
            kwargs["rgb"]["_rgb_image"].raw_data[:] = b"\xff" * (720 * 1280 * 4)
            score(engine, pending)
            archive = engine.build_archive(pending.token)
            descriptor = json.loads(archive.descriptor_bytes)
            self.assertEqual(descriptor["old_prepared_sha256"], original.content_sha256)
            self.assertEqual(archive.files["segmentation.png"], original.target["segmentation_png"])
            self.assertEqual(archive.files["object_ignore.png"], original.target["object_ignore_png"])
            self.assertEqual(descriptor["legacy_target_meta"]["tables"], original.target["tables"])
            self.assertEqual(np.load(io.BytesIO(archive.files["rgb_bgra.npy"]), allow_pickle=False).sum(), 0)
            self.assertEqual(archive.files[FILENAMES["depth_bgra"]], archive.files["depth_bgra.npy"])
            self.assertEqual(archive.files[FILENAMES["rawsemantic"]], archive.files["raw_semantic.npy"])
            self.assertEqual(archive.archive_sha256, C.sha256(archive.descriptor_bytes))
            self.assertTrue(descriptor["offline_replay"])
            self.assertTrue(descriptor["rgb_source_present"])
            for name, expected in descriptor["files"].items():
                self.assertEqual(expected, dict(bytes=len(archive.files[name]), sha256=C.sha256(archive.files[name])))
            with self.assertRaises(TypeError):
                archive.files["segmentation.png"] = b"mutation"
            self.assertIs(engine.build_archive(pending.token), archive)

    def test_real_archive_child_receipt_and_scorer_ack_both_required(self):
        with fixture() as (engine, kwargs, directory):
            pending = commit(engine, kwargs); score(engine, pending)
            archive = engine.build_archive(pending.token)
            root = (directory / "runtime-archive"); root.mkdir()
            prefix = "frame_100/producer"
            child = DISK.ArchiveClient(roots=[root.resolve()], max_jobs=2,
                max_bytes=G.RESERVATION, max_frames=8, binding={"cpu_test": "producer_engine"})
            try:
                child.submit(kwargs["identity"], {root / prefix / name: raw for name, raw in archive.files.items()})
                completion = child.close()["completions"][0]
            finally:
                if not child.closed:
                    child.abort()
            engine.archive_durable(pending.token, completion, relative_prefix=prefix)
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
            for name, raw in archive.files.items():
                self.assertEqual((root / prefix / name).read_bytes(), raw)
            engine.scorer_archive_completed(pending.token, pending.identity_bytes, pending.semantic_sha256)
            snapshot = engine.snapshot()
            self.assertEqual(snapshot["retained_jobs"], 0)
            self.assertEqual(snapshot["owner"]["reserved_bytes"], 0)
            self.assertEqual(snapshot["events"]["1"]["outcome"], "DURABLE")
            self.assertGreater(snapshot["events"]["1"]["archive_durable_observed_raw_ns"], 0)
            self.assertTrue(engine.close()["valid"])

    def test_scorer_ack_without_fsync_retains_owned_artifacts(self):
        with fixture() as (engine, kwargs, _):
            pending = commit(engine, kwargs); score(engine, pending)
            archive = engine.build_archive(pending.token)
            engine.scorer_archive_completed(pending.token, pending.identity_bytes, pending.semantic_sha256)
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
            engine.archive_durable(pending.token, receipt(pending, archive), relative_prefix="frame_100/producer")
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)

    def test_corrupt_or_nondurable_receipts_retain_owned_frame_and_failstop(self):
        changes = (lambda r: r.update(status="QUEUED"), lambda r: r.update(durable=False),
            lambda r: r["identity"].update(frame_id=999), lambda r: r["written"].pop(),
            lambda r: r["written"][0].update(sha256="0" * 64), lambda r: r["written"][0].update(bytes=0),
            lambda r: r["written"].append(dict(r["written"][0])), lambda r: r["written"][0].update(root=1))
        for change in changes:
            with self.subTest(change=change), fixture() as (engine, kwargs, _):
                pending = commit(engine, kwargs); score(engine, pending)
                archive = engine.build_archive(pending.token)
                result = receipt(pending, archive); change(result)
                with self.assertRaises(G.ProducerError):
                    engine.archive_durable(pending.token, result, relative_prefix="frame_100/producer")
                self.assertFalse(engine.snapshot()["valid"])
                self.assertEqual(engine.snapshot()["retained_jobs"], 1)

    def test_early_archive_or_foreign_notification_failstop(self):
        with fixture() as (engine, kwargs, _):
            pending = commit(engine, kwargs)
            with self.assertRaisesRegex(G.ProducerError, "score and feedback"):
                engine.build_archive(pending.token)
        with fixture() as (engine, kwargs, _):
            pending = commit(engine, kwargs)
            with self.assertRaisesRegex(G.ProducerError, "foreign notification"):
                engine.score_completed(pending.token, pending.identity_bytes, "0" * 64)
        with fixture() as (engine, kwargs, _):
            pending = commit(engine, kwargs)
            with self.assertRaisesRegex(G.ProducerError, "precedes"):
                engine.feedback_confirmed(pending.token, pending.identity_bytes, pending.semantic_sha256)

    def test_recorded_clocks_refused_by_default_and_no_live_archive_clock_fabrication(self):
        with fixture(offline_replay=False) as (engine, kwargs, _):
            with self.assertRaisesRegex(G.ProducerError, "explicit offline replay"):
                prepare(engine, kwargs)
            self.assertEqual(engine.snapshot()["owner"]["commit_calls"], 0)
        with fixture(offline_replay=False) as (engine, kwargs, _):
            pending = prepare(engine, kwargs, archive_clocks=None, replay_observed_raw_ns=None)
            engine.confirm_send(pending.token, plan_for(kwargs)[0]); engine.publish_ready(pending.token)
            score(engine, pending)
            with self.assertRaisesRegex(G.ProducerError, "live legacy archive-clock schema"):
                engine.build_archive(pending.token)
            self.assertFalse(engine.snapshot()["live_archive_clock_schema_implemented"])

    def test_job_and_payload_bounds_reject_without_evicting_owned_frames(self):
        with fixture(max_retained_jobs=1) as (engine, kwargs, _):
            commit(engine, kwargs); advance(kwargs)
            with patch.object(engine._owner._producer, "compute", wraps=engine._owner._producer.compute) as compute:
                with self.assertRaises(RuntimeError):
                    prepare(engine, kwargs)
                self.assertEqual(compute.call_count, 0)
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
        with fixture(reserved_bytes=1024) as (engine, kwargs, _):
            producer = engine._owner._producer
            before = producer.history_sha256()
            with self.assertRaises(RuntimeError):
                prepare(engine, kwargs)
            self.assertEqual(producer.history_sha256(), before)
            self.assertEqual(engine.snapshot()["owner"]["cancel_calls"], 1)
            self.assertEqual(engine.snapshot()["retained_jobs"], 0)

    def test_failed_consumer_cannot_be_revived_and_committed_history_not_rolled_back(self):
        with fixture() as (engine, kwargs, _):
            pending = commit(engine, kwargs); score(engine, pending)
            archive = engine.build_archive(pending.token)
            history = engine._owner._producer.history_sha256()
            engine.fail(pending.token, consumer="archive", reason="archive child died")
            with self.assertRaisesRegex(G.ProducerError, "cannot revive"):
                engine.archive_durable(pending.token, receipt(pending, archive), relative_prefix="frame_100/producer")
            self.assertEqual(engine._owner._producer.history_sha256(), history)
            self.assertFalse(engine.snapshot()["valid"])
            self.assertFalse(engine.close(abandon=True)["valid"])

    def test_undrained_close_preserves_lifetime_and_explicit_abandon_releases(self):
        with fixture() as (engine, kwargs, _):
            commit(engine, kwargs)
            with self.assertRaises(RuntimeError):
                engine.close()
            self.assertFalse(engine.snapshot()["closed"])
            self.assertEqual(engine.snapshot()["retained_jobs"], 1)
            result = engine.close(abandon=True)
            self.assertFalse(result["valid"])
            self.assertTrue(result["closed"])
            self.assertEqual(result["retained_jobs"], 0)

    def test_pending_close_cancels_actual_transaction_and_thread_ownership_is_enforced(self):
        with fixture() as (engine, kwargs, _):
            pending = prepare(engine, kwargs)
            producer = engine._owner._producer
            with patch.object(producer, "cancel", wraps=producer.cancel) as cancel:
                result = engine.close()
                self.assertEqual(cancel.call_count, 1)
            self.assertTrue(result["valid"])
            self.assertEqual(producer.processed_frames, 0)
        with fixture() as (engine, kwargs, _):
            errors = []
            def foreign():
                try:
                    prepare(engine, kwargs)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=foreign); thread.start(); thread.join()
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], G.ProducerError)
            self.assertFalse(engine.snapshot()["valid"])


if __name__ == "__main__":
    unittest.main()
