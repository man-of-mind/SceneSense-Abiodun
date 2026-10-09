"""CPU-only retention tests; real corrected packing and SFD4 identity parsing."""
from pathlib import Path
import json
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_ue_runtime_v2 as U
from rl_agent.splitfusion_live_parity_repair_v1 import perception as P
from . import capture as C, scope as S
from . import test_replay as F


def fixture():
    case = F.ReplayTests()
    case.setUp()
    e = case.envelope
    plan = NS(kind=NS(value="POLICY_DECISION"))
    planned = NS(reward_requested=True, session_uuid=e.session_uuid, frame_id=e.frame_id,
        tensor_seq=e.tensor_seq, decision_seq=e.decision_seq, ticket_seq=e.ticket_seq,
        capture_wall_ns=e.capture_timestamp_ns, mode_id=e.mode_id, q_e4=e.q_e4,
        carla_timestamp=1.0, radar_window_sha256="c"*64, plan=plan)
    prepared = NS(wire=W.pack_sfd4(e, case.context), identity=MP.run4_identity(e),
        gt_identity=MP.gt_identity(run_id="test", cell_id="test", stream_id=case.context.stream_id,
            frame_id=e.frame_id, anchor_action_id=e.anchor_action_id, anchor_profile_id="fixture",
            capture_timestamp_ns=e.capture_timestamp_ns))
    return case, planned, prepared


def collectors(front=False, gt=False):
    return [NS(worker=NS(is_alive=lambda: front), evaluation_worker=NS(is_alive=lambda: gt))]


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.before_threads)

    def populated(self):
        case, planned, prepared = fixture()
        archive = C.InputArchive(max_frames=1)
        record = archive.begin(planned)
        self.assertIs(archive.observe_input(case.input, "cuda:0"), case.input)
        archive.end(record, prepared)
        return archive

    def test_repaired_packing_byte_equal_only_retains_preexisting_cpu_tensor(self):
        rng = np.random.default_rng(11)
        bgr = rng.integers(0, 256, (720,1280,3), dtype=np.uint8)
        radar = rng.random((4,432,768), dtype=np.float32)
        observed = []
        def observer(tensor, device):
            observed.append(tensor)
            return tensor
        replacement = S.retained_packing(P.prepare_live_input, observer)
        baseline = P.prepare_live_input(bgr, radar, "cpu")
        result = replacement(bgr, radar, "cpu")
        self.assertTrue(torch.equal(baseline, result))
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].data_ptr(), result.data_ptr())
        self.assertEqual(tuple(result.shape), C.SHAPE)
        self.assertEqual(torch.count_nonzero(result[:,:,432:,:]), 0)

    def test_live_observer_no_tensor_copy_scan_or_disk_write(self):
        case, planned, prepared = fixture()
        archive = C.InputArchive(max_frames=1)
        record = archive.begin(planned)
        with mock.patch.object(Path, "open", side_effect=AssertionError("disk")), \
                mock.patch.object(torch.Tensor, "clone", side_effect=AssertionError("copy")), \
                mock.patch.object(torch.Tensor, "numpy", side_effect=AssertionError("convert")), \
                mock.patch.object(torch, "isfinite", side_effect=AssertionError("scan")):
            result = archive.observe_input(case.input, "cuda:0")
            archive.end(record, prepared)
        self.assertIs(result, case.input)
        self.assertIs(archive.records[0]["input"], case.input)
        self.assertIs(archive.records[0]["prepared"]["wire"], prepared.wire)
        self.assertEqual(archive.error_count, 0)

    def test_prewarm_hold_and_foreign_reward_kind(self):
        case, planned, _ = fixture()
        archive = C.InputArchive()
        self.assertIs(archive.observe_input(case.input, "cuda:0"), case.input)
        planned.reward_requested = False
        self.assertIsNone(archive.begin(planned))
        self.assertIs(archive.observe_input(case.input, "cuda:0"), case.input)
        self.assertEqual(archive.records, [])
        planned.reward_requested = True
        planned.plan.kind.value = "FALLBACK"
        with self.assertRaisesRegex(ValueError, "policy decision"):
            archive.begin(planned)

    def test_observer_limit_never_raises_into_input_path(self):
        case, planned, prepared = fixture()
        archive = C.InputArchive(max_frames=1, max_bytes=C.INPUT_BYTES)
        record = archive.begin(planned)
        self.assertIs(archive.observe_input(case.input, "cuda:0"), case.input)
        archive.end(record, prepared)  # wire exceeds total cap
        self.assertEqual(archive.error_count, 1)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/"archive"
            with self.assertRaisesRegex(ValueError, "observer faults"):
                archive.flush(destination, collectors=collectors(), source_binding=S.PINS)
            self.assertFalse(destination.exists())

    def test_duplicate_and_frame_overflow_fail_without_unbounded_storage(self):
        archive = self.populated()
        _, planned, _ = fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            archive.begin(planned)
        planned.frame_id += 1
        with self.assertRaisesRegex(ValueError, "frame bound"):
            archive.begin(planned)
        self.assertEqual(len(archive.records), 1)

    def test_both_worker_drains_are_required(self):
        archive = self.populated()
        with tempfile.TemporaryDirectory() as tmp:
            for owned in ([], collectors(front=True), collectors(gt=True), [NS()]):
                with self.assertRaises(ValueError):
                    archive.flush(Path(tmp)/"archive", collectors=owned, source_binding=S.PINS)
            self.assertFalse((Path(tmp)/"archive").exists())

    def test_roundtrip_create_only_and_hash_tamper(self):
        archive = self.populated()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"archive"
            manifest = archive.flush(path, collectors=collectors(), source_binding=S.PINS)
            binding = C.sha(path/"MANIFEST.json")
            self.assertEqual(C.verify_archive(path, expected_manifest_sha256=binding), manifest)
            self.assertEqual(archive.records, [])
            self.assertEqual(manifest["frames"][0]["input_bytes"], 9633792)
            with self.assertRaises(ValueError):
                archive.flush(path, collectors=collectors(), source_binding=S.PINS)
            second = self.populated()
            with self.assertRaises(FileExistsError):
                second.flush(path, collectors=collectors(), source_binding=S.PINS)
            member = path/manifest["frames"][0]["wire_file"]
            with member.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "modified"):
                C.verify_archive(path, expected_manifest_sha256=binding)

    def test_plan_wire_drift_and_cpu_inference_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = self.populated()
            archive.records[0]["planned"].frame_id += 1
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                archive.flush(Path(tmp)/"drift", collectors=collectors(), source_binding=S.PINS)
            archive = self.populated()
            archive.records[0]["transfer_device"] = "cpu"
            with self.assertRaisesRegex(ValueError, "original CUDA"):
                archive.flush(Path(tmp)/"cpu", collectors=collectors(), source_binding=S.PINS)

    def test_manifest_requires_external_hash_and_rejects_semantic_tampering(self):
        mutations = [
            lambda m: m.update(frames=[]),
            lambda m: m.update(max_frames=0),
            lambda m: m.update(retained_bytes=1),
            lambda m: m.update(source_binding={}),
            lambda m: m["drain"]["worker"].update(alive=True),
            lambda m: m["frames"][0].update(index=3),
            lambda m: m["frames"][0].update(transfer_device="cpu"),
            lambda m: m["frames"][0]["gt_identity"].update(frame_id=0),
            lambda m: m["frames"][0].update(stream_id="foreign"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"archive"
                manifest = self.populated().flush(path, collectors=collectors(), source_binding=S.PINS)
                original_binding = C.sha(path/"MANIFEST.json")
                mutation(manifest)
                with (path/"MANIFEST.json").open("w") as stream:
                    json.dump(manifest, stream)
                with self.assertRaises(ValueError):
                    C.verify_archive(path, expected_manifest_sha256=original_binding)
                # Even a newly rebound malformed manifest is refused.
                with self.assertRaises(ValueError):
                    C.verify_archive(path, expected_manifest_sha256=C.sha(path/"MANIFEST.json"))

    def test_observer_fault_disables_storage_and_symlink_manifest_refused(self):
        archive = self.populated()
        archive.fault(ValueError("injected"))
        _, planned, _ = fixture()
        for frame_id in range(1000, 2000):
            planned.frame_id = frame_id
            self.assertIsNone(archive.begin(planned))
        self.assertEqual(len(archive._keys), 1)
        self.assertEqual(len(archive.records), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"archive"
            self.populated().flush(path, collectors=collectors(), source_binding=S.PINS)
            binding = C.sha(path/"MANIFEST.json")
            (path/"MANIFEST.json").rename(Path(tmp)/"external.json")
            (path/"MANIFEST.json").symlink_to(Path(tmp)/"external.json")
            with self.assertRaisesRegex(ValueError, "manifest"):
                C.verify_archive(path, expected_manifest_sha256=binding)

    def test_scope_uses_real_materialization_and_restores(self):
        case, planned, _ = fixture()
        e = case.envelope
        frame_plan = NS(tensor_seq=e.tensor_seq, profile=NS(mode_id=e.mode_id, q_e4=e.q_e4, action_id=e.anchor_action_id,
                                  profile_id="fixture"), frame_identity=e)
        planned.plan = frame_plan
        frame_plan.kind = NS(value="POLICY_DECISION")
        planned.verify = lambda **kwargs: None
        bgr = np.zeros((720,1280,3), dtype=np.uint8)
        radar = np.zeros((4,432,768), dtype=np.float32)
        def make_input():
            # CPU stand-in observes the requested live target but does not use CUDA.
            packing = S.retained_packing(original_packing,
                lambda tensor, device: archive.observe_input(tensor, "cuda:0"))
            return packing(bgr, radar, "cpu")
        pipeline = U.Run4FramePipelineV2(engine=mock.Mock(),
            continuous_ue=NS(prepare=lambda *args: NS(envelope=e)),
            provider=mock.Mock(), stream_id=case.context.stream_id, run_id="test", cell_id="test")
        original_packing, original_materialize = P.prepare_live_input, U.Run4FramePipelineV2.materialize
        archive = C.InputArchive(max_frames=1)
        with S.install_capture(archive):
            self.assertIsNot(P.prepare_live_input, original_packing)
            result = pipeline.materialize(planned, frame_id=e.frame_id,
                capture_wall_ns=e.capture_timestamp_ns, ego_pose=(1.,2.,3.,0.,10.,0.), input_7ch=make_input)
            self.assertEqual(result.envelope, e)
            self.assertEqual(archive.error_count, 0)
            self.assertEqual(len(archive.records), 1)
        self.assertIs(P.prepare_live_input, original_packing)
        self.assertIs(U.Run4FramePipelineV2.materialize, original_materialize)

    def test_production_exception_is_preserved_and_scope_restored(self):
        _, planned, _ = fixture()
        sentinel = RuntimeError("original production error")
        archive = C.InputArchive(max_frames=1)
        before_packing = P.prepare_live_input
        with mock.patch.object(U.Run4FramePipelineV2, "materialize", side_effect=sentinel):
            before = U.Run4FramePipelineV2.materialize
            with self.assertRaises(RuntimeError) as raised:
                with S.install_capture(archive):
                    U.Run4FramePipelineV2.materialize(NS(), planned)
            self.assertIs(raised.exception, sentinel)
            self.assertIs(U.Run4FramePipelineV2.materialize, before)
        self.assertIs(P.prepare_live_input, before_packing)
        self.assertEqual(archive.error_count, 1)

    def test_nested_scope_and_new_threads_do_not_capture(self):
        case, planned, prepared = fixture()
        archive = C.InputArchive()
        with S.install_capture(archive):
            with self.assertRaisesRegex(ValueError, "nested"):
                with S.install_capture(C.InputArchive()):
                    pass
        import threading
        record = archive.begin(planned)
        worker = threading.Thread(target=lambda: archive.observe_input(case.input, "cuda:0"))
        worker.start(); worker.join()
        self.assertIsNone(record["input"])
        archive.observe_input(case.input, "cuda:0")
        archive.end(record, prepared)
        self.assertEqual(archive.error_count, 0)


if __name__ == "__main__":
    unittest.main()
