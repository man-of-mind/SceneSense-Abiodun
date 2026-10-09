"""Focused regressions using real sealed/mapped fixture buffers (CPU only)."""
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_agent.splitfusion_exact_gt_service_proto_v2 import freeze_codec as FZ
from rl_agent.splitfusion_exact_gt_service_proto_v2 import parity_common as P
from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from . import repair as R


class PreparationFailure(ValueError):
    pass


class FakeEngine:
    def __init__(self, fail=False, chained=False, leak=False):
        self.fail, self.chained, self.leak = fail, chained, leak

    def prepare_unbound(self, kwargs, **_):
        view = kwargs["semantic_image"].raw_data
        self.first_byte = view[0]
        if self.leak:
            self.leaked = view
        if self.chained:
            try:
                raise KeyError("INNER_SOURCE_CHECK")
            except KeyError as inner:
                raise PreparationFailure("ORIGINAL_SOURCE_CHECK") from inner
        if self.fail:
            raise PreparationFailure("ORIGINAL_SOURCE_CHECK")
        return "handle"


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = P.Corpus(1)
        cls.case = cls.corpus.cases[0]
        cls.session = "b" * 64
        cls.descriptor, chunks = FZ.freeze(cls.corpus.sourcekwargs(cls.case),
            cls.case.kwargs["policy_timestamp_s"], session_id=cls.session, sequence=1)
        cls.fd, cls.size = FZ.seal(chunks)

    @classmethod
    def tearDownClass(cls):
        os.close(cls.fd)

    def core(self, cls=R.PrepCoreLifetime, **engine):
        core = cls.__new__(cls)
        core.engine = FakeEngine(**engine)
        core.offline = False
        core.gate = FZ.FreezeGate(self.session)
        core.jobs = {}
        return core

    def call(self, core, descriptor=None):
        with contextlib.redirect_stderr(io.StringIO()):
            return core.handle("FREEZE", {"descriptor": (descriptor or self.descriptor).decode()}, self.fd)

    def test_old_code_masks_real_preparation_error(self):
        with self.assertRaisesRegex(BufferError, "cannot close exported pointers") as caught:
            self.call(self.core(R.ORIGINAL_PREP_CORE, fail=True))
        self.assertIsInstance(caught.exception.__context__, PreparationFailure)

    def test_repair_preserves_original_and_no_unbound_job(self):
        core = self.core(fail=True)
        with self.assertRaisesRegex(PreparationFailure, "ORIGINAL_SOURCE_CHECK"):
            self.call(core)
        self.assertEqual(core.jobs, {})

    def test_chained_error_preserved(self):
        with self.assertRaises(PreparationFailure) as caught:
            self.call(self.core(chained=True))
        self.assertIsInstance(caught.exception.__cause__, KeyError)

    def test_true_escaped_view_still_fails_closed(self):
        core = self.core(leak=True)
        try:
            with self.assertRaises(BufferError):
                self.call(core)
            self.assertEqual(core.jobs, {})
        finally:
            core.engine.leaked.release()

    def test_success_identical_except_clock_observations(self):
        old, new = self.call(self.core(R.ORIGINAL_PREP_CORE)), self.call(self.core())
        for result in (old[0], new[0]):
            result.pop("decoded_raw_ns")
            result.pop("decoded_thread_cpu_ns")
        self.assertEqual(old, new)

    def test_malformed_descriptor_not_downgraded(self):
        with self.assertRaises(FZ.FreezeError):
            self.call(self.core(), b"not-json")

    def test_fatal_snapshot_hashes_and_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            core = self.core(fail=True)
            core.failure_directory = temporary
            with self.assertRaises(PreparationFailure):
                self.call(core)
            output = Path(temporary) / "freeze_failure_v1"
            ready = json.loads((output / "READY.json").read_bytes())
            for name, sha in ready["sha256"].items():
                self.assertEqual(C.sha256((output / name).read_bytes()), sha)
            self.assertEqual((output / "SOURCE.bin").stat().st_size, self.size)
            self.assertEqual(json.loads((output / "REQUEST.json").read_bytes())["descriptor"],
                             self.descriptor.decode())
            self.assertEqual(json.loads((output / "ERROR.json").read_bytes())["exception_type"],
                             "PreparationFailure")

    def test_failed_snapshot_never_replaces_original(self):
        core = self.core(fail=True)
        core.failure_directory = "/not-used"
        with patch.object(R, "save_failure", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(PreparationFailure, "ORIGINAL_SOURCE_CHECK"):
                self.call(core)

    def test_no_snapshot_or_extra_decode_on_success(self):
        core = self.core()
        core.failure_directory = "/not-used"
        with patch.object(R, "save_failure", side_effect=AssertionError("success wrote a snapshot")):
            self.call(core)

    def test_switch_restores_and_refuses_double_install(self):
        with R.install():
            self.assertIs(R.PC.PrepCore, R.PrepCoreLifetime)
            with self.assertRaises(RuntimeError):
                with R.install():
                    pass
        self.assertIs(R.PC.PrepCore, R.ORIGINAL_PREP_CORE)

    def test_600_requests_gc_disabled_no_mapping_or_fd_growth(self):
        # One gate and caller across 600 requests. The producer is intentionally
        # minimal: this is a mapping-lifetime soak, not 600 live/scored decisions.
        core = self.core()
        gc.collect()
        maps_before = Path("/proc/self/maps").read_text().count("splitfusion_gt_snapshot_v4")
        fds_before = len(list(Path("/proc/self/fd").iterdir()))
        enabled = gc.isenabled()
        gc.disable()
        try:
            for index in range(600):
                descriptor = json.loads(self.descriptor)
                frame = self.case.frame_id + index
                descriptor["sequence"] = index + 1
                descriptor["frame_id"] = descriptor["capture_template"]["frame_id"] = frame
                descriptor["rgb"]["frame_id"] = frame
                descriptor["world"]["frame_id"] = frame
                descriptor["dynamic_snapshot_evidence"]["frame_id"] = frame
                for image in descriptor["images"]:
                    image["frame"] = frame
                for role in ("semantic_callback", "depth_callback", "radar_callback"):
                    descriptor["rgb"][role]["frame_id"] = frame
                self.call(core, C.canonical(descriptor))
                core.jobs.clear()
        finally:
            if enabled:
                gc.enable()
        self.assertEqual(core.gate.admitted, 600)
        self.assertEqual(Path("/proc/self/maps").read_text().count("splitfusion_gt_snapshot_v4"), maps_before)
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), fds_before)


if __name__ == "__main__":
    unittest.main()
