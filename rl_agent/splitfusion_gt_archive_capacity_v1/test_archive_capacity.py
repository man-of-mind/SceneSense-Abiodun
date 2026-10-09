"""Focused checks of the archive capacity repair (real archive child, real fsync; CPU only, no CARLA/OAI/GPU).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m unittest -v rl_agent.splitfusion_gt_archive_capacity_v1.test_archive_capacity
"""
import json
import os
from pathlib import Path
import select
import signal
import tempfile
import time
import unittest
from unittest import mock

from rl_agent.splitfusion_live_gt_handoff_v1 import archive as A

from . import contract as C

MEASURED = {}


def _rss_kib(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="archive_capacity_")
        self.root = Path(self.tmp.name).resolve()
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            if not client.closed:
                client.abort()
            self.assertIsNotNone(client.process.poll(), "archive child must be reaped")
        self.tmp.cleanup()

    def client(self, **options):
        client = A.ArchiveClient(roots=[str(self.root)], max_jobs=options.pop("max_jobs", 4),
                                 max_bytes=options.pop("max_bytes", 1024 * 1024),
                                 max_frames=options.pop("max_frames", 8), binding=dict(scope="capacity_test"))
        self.clients.append(client)
        return client

    def job(self, client, i, payload=None):
        payload = payload if payload is not None else f"job {i}".encode()
        files = {str(self.root / f"job_{i:05d}.bin"): payload, str(self.root / f"job_{i:05d}.READY"): b"ready"}
        return client.submit(dict(frame_id=i, evidence_role="capacity"), files,
                             ready_paths=[str(self.root / f"job_{i:05d}.READY")])

    def wait_credit(self, client, deadline_s=30.):
        deadline = time.monotonic() + deadline_s
        while client.outstanding_jobs >= client.config["max_jobs"]:
            self.assertLess(time.monotonic(), deadline, "archive credit never returned")
            select.select([client.socket], [], [], 0.05)
            client.poll()

    def drain(self, client, deadline_s=60.):
        deadline = time.monotonic() + deadline_s
        while client.outstanding_jobs:
            self.assertLess(time.monotonic(), deadline, "archive drain stalled")
            select.select([client.socket], [], [], 0.05)
            client.poll()


class ConfigurationBoundTest(Base):
    def test_bound_is_finite_and_matches_the_registered_sizing(self):
        self.assertEqual(A.MAX_CUMULATIVE_JOBS, C.MAX_CUMULATIVE_JOBS)
        self.assertEqual(A.MAX_CUMULATIVE_JOBS, C.JOBS_PER_DECISION_SIZING * C.LARGEST_REGISTERED_BUDGET)
        from rl_agent.splitfusion_run5_pilot_live_v1 import contract as PL
        self.assertEqual(4 * PL.limits(1000)["max_frames"], A.MAX_CUMULATIVE_JOBS)   # service.py:100 / ue.py:132 sizing
        for bad in (0, -1, A.MAX_CUMULATIVE_JOBS + 1, 10 ** 9, 1000.0, True, None):
            with self.assertRaisesRegex(A.ArchiveError, "archive frame bound differs"):
                A._configuration([str(self.root)], 4, 1024, bad, {})
        self.assertEqual(A._configuration([str(self.root)], 4, 1024, 4000, {})["max_frames"], 4000)

    def test_child_starts_with_the_upper_bound_and_closes_cleanly(self):
        client = self.client(max_frames=A.MAX_CUMULATIVE_JOBS)
        self.assertEqual(client.ready["config"]["max_frames"], 4000)
        summary = client.close()
        self.assertEqual((summary["submitted"], summary["completed"], summary["child_returncode"]), (0, 0, 0))


class CumulativeCapacityTest(Base):
    def test_4000_small_jobs_complete_durably_then_the_4001st_is_refused(self):
        client = self.client(max_frames=A.MAX_CUMULATIVE_JOBS, max_jobs=4, max_bytes=1024 * 1024)
        rss_parent0, rss_child0 = _rss_kib(os.getpid()), _rss_kib(client.process.pid)
        started = time.monotonic()
        for i in range(1, A.MAX_CUMULATIVE_JOBS + 1):
            self.wait_credit(client)
            self.assertEqual(self.job(client, i), i)
            self.assertLessEqual(client.outstanding_jobs, 4)
            self.assertLessEqual(client.outstanding_bytes, 1024 * 1024)
        self.drain(client)
        elapsed = time.monotonic() - started
        self.assertEqual(len(client.completions), 4000)
        self.assertTrue(all(row["status"] == "COMPLETE" and row["durable"] for row in client.completions))
        self.assertEqual([row["job_id"] for row in client.completions], list(range(1, 4001)))
        for i in range(1, 4001):                                   # durable readback of every job
            self.assertEqual((self.root / f"job_{i:05d}.bin").read_bytes(), f"job {i}".encode())
            self.assertEqual((self.root / f"job_{i:05d}.READY").read_bytes(), b"ready")
        self.assertEqual(len(list(self.root.iterdir())), 8000)
        bookkeeping, rss_parent1, rss_child1 = client.bookkeeping_bytes, _rss_kib(os.getpid()), _rss_kib(client.process.pid)
        self.assertGreater(bookkeeping, 0)
        self.assertLessEqual(bookkeeping, A.MAX_BOOKKEEPING_BYTES)
        with self.assertRaisesRegex(A.ArchiveError, "archive frame limit exceeded"):
            self.job(client, 4001)
        self.assertFalse((self.root / "job_04001.bin").exists())
        with self.assertRaisesRegex(A.ArchiveError, "archive frame limit exceeded"):
            client.close()                                          # sticky refusal; child still reconciles
        self.assertEqual(client.process.returncode, 0)
        self.assertEqual((client.close_result["submitted"], client.close_result["completed"]), (4000, 4000))
        MEASURED.update(jobs=4000, files=8000, elapsed_s=round(elapsed, 3), bookkeeping_bytes=bookkeeping,
                        bookkeeping_bytes_per_job=round(bookkeeping / 4000, 1), max_bookkeeping_bytes=A.MAX_BOOKKEEPING_BYTES,
                        parent_rss_kib_before_after=[rss_parent0, rss_parent1], child_rss_kib_before_after=[rss_child0, rss_child1])

    def test_child_enforces_its_own_cumulative_bound(self):
        client = self.client(max_frames=3, max_jobs=2)
        for i in (1, 2, 3):
            self.wait_credit(client)
            self.job(client, i)
        self.drain(client)
        client.config["max_frames"] = 4                            # bypass the parent check only
        self.job(client, 4)
        with self.assertRaisesRegex(A.ArchiveError, "archive job order/count differs"):
            self.drain(client)
        self.assertFalse((self.root / "job_00004.bin").exists())


class RetainedLimitsTest(Base):
    def test_outstanding_job_credit_is_enforced_with_a_stalled_child(self):
        client = self.client(max_frames=A.MAX_CUMULATIVE_JOBS, max_jobs=2)
        os.kill(client.process.pid, signal.SIGSTOP)
        try:
            self.job(client, 1)
            self.job(client, 2)
            with self.assertRaisesRegex(A.ArchiveError, "credits exhausted"):
                self.job(client, 3)
        finally:
            os.kill(client.process.pid, signal.SIGCONT)
        self.assertTrue(client.failed)
        client.abort()
        self.assertFalse((self.root / "job_00003.bin").exists())

    def test_outstanding_byte_credit_is_enforced(self):
        client = self.client(max_frames=A.MAX_CUMULATIVE_JOBS, max_bytes=4096)
        self.job(client, 1, payload=b"x" * (4096 - len(b"ready")))  # exactly at the byte credit
        self.drain(client)
        with self.assertRaisesRegex(A.ArchiveError, "credits exhausted"):
            self.job(client, 2, payload=b"x" * 4096)

    def test_duplicate_identity_and_duplicate_path_are_refused(self):
        client = self.client(max_frames=A.MAX_CUMULATIVE_JOBS)
        self.job(client, 1)
        with self.assertRaisesRegex(A.ArchiveError, "duplicate archive frame identity"):
            client.submit(dict(frame_id=1, evidence_role="capacity"), {str(self.root / "other.bin"): b"x"})
        second = self.client(max_frames=A.MAX_CUMULATIVE_JOBS)
        second.submit(dict(frame_id=7), {str(self.root / "shared.bin"): b"x"})
        with self.assertRaisesRegex(A.ArchiveError, "duplicate archive file across jobs"):
            second.submit(dict(frame_id=8), {str(self.root / "shared.bin"): b"y"})
        self.assertEqual((self.root / "shared.bin").read_bytes() if (self.root / "shared.bin").exists() else b"x", b"x")

    def test_bookkeeping_bound_is_enforced_and_accounted_separately(self):
        client = self.client(max_frames=A.MAX_CUMULATIVE_JOBS)
        self.job(client, 1)
        self.drain(client)
        used = client.bookkeeping_bytes
        self.assertEqual(client.outstanding_bytes, 0)              # payload credit fully returned
        self.assertGreater(used, 0)                                # bookkeeping is cumulative
        with mock.patch.object(A, "MAX_BOOKKEEPING_BYTES", used + 10):
            with self.assertRaisesRegex(A.ArchiveError, "archive bookkeeping bound exceeded"):
                self.job(client, 2)
        self.assertFalse((self.root / "job_00002.bin").exists())

    def test_abort_with_owned_jobs_terminates_and_reaps_the_child(self):
        client = self.client(max_frames=A.MAX_CUMULATIVE_JOBS, max_jobs=2)
        os.kill(client.process.pid, signal.SIGSTOP)
        self.job(client, 1)
        client.abort()
        self.assertIsNotNone(client.process.poll())
        self.assertTrue(client.failed)
        self.assertEqual(client.close_result["outstanding_jobs"], 1)          # unresolved job reported, not dropped
        self.assertFalse((self.root / "job_00001.bin").exists())


class SubstitutionTest(unittest.TestCase):
    def test_only_the_registered_two_file_substitution_is_accepted(self):
        root = Path(__file__).resolve().parents[2]
        from rl_agent.splitfusion_exact_gt_live_v2 import contract as E
        current = E.exact_sources(root)
        self.assertEqual(C.repaired(root), {k: current[k] for k in C.ORIGINAL})
        old = dict(current, **C.ORIGINAL)
        self.assertTrue(C.substitution_ok(old, current, root))
        self.assertTrue(C.substitution_ok(current, current, root))
        self.assertFalse(C.substitution_ok(dict(old, **{C.ARCHIVE: "0" * 64}), current, root))
        other = next(k for k in current if k not in C.ORIGINAL)
        self.assertFalse(C.substitution_ok(dict(old, **{other: "0" * 64}), current, root))
        self.assertFalse(C.substitution_ok({k: v for k, v in old.items() if k != other}, current, root))


def tearDownModule():
    target = os.environ.get("ARCHIVE_CAPACITY_MEASURED")
    if target and MEASURED:
        Path(target).write_text(json.dumps(MEASURED, indent=1, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
