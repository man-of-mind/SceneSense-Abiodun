"""CPU-only archive ownership, durability, failure, and actual-child tests."""
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from . import archive as A


def identity(frame=1):
    return dict(frame_id=frame, stream_id="archive-test", session_uuid="cpu-only")


@contextmanager
def configuration():
    with tempfile.TemporaryDirectory(prefix="archive-handoff-test-") as directory:
        root = Path(directory)
        first, second = root / "attempt", root / "edge"
        first.mkdir(); second.mkdir()
        config = A._configuration((first, second), 2, 1024 * 1024, 8, {"test": "owned-cpu"})
        yield root, (first, second), config


def batch(config, files, ready=(), stamped=()):
    records, chunks, size = A._encode_files(files, ready, config["roots"], stamped)
    return dict(command="SUBMIT", job_id=1, identity=identity(), files=records, bytes=size), A._memfd(chunks)


def sentinel():
    return A.canonical(dict(schema="splitfusion.prepared_live_v010_transport.v1",
                            ready_raw_ns=0, ready_wall_ns=0, payloads={"exact": "retained"}))


class ArchivePrimitiveTests(unittest.TestCase):
    def test_boundaries_and_immutable_payload_required(self):
        with configuration() as (_, roots, config):
            for jobs, size, frames in ((1, 100, 1), (9, 100, 1), (True, 100, 1),
                                      (2, 0, 1), (2, A.MAX_BYTES + 1, 1), (2, 100, A.MAX_CUMULATIVE_JOBS + 1)):
                with self.subTest(jobs=jobs, size=size, frames=frames), self.assertRaises(A.ArchiveError):
                    A._configuration(roots, jobs, size, frames, {})
            for payload in (bytearray(b"mutable"), memoryview(b"view"), "text"):
                with self.subTest(payload=type(payload)), self.assertRaises(A.ArchiveError):
                    A._encode_files({roots[0] / "file": payload}, (), config["roots"])

    def test_outside_traversal_symlink_and_bad_ready_paths_rejected(self):
        with configuration() as (outer, roots, config):
            (roots[0] / "escape").symlink_to(roots[1], target_is_directory=True)
            for path in (outer / "outside", str(roots[0]) + "/../outside", "relative",
                         roots[0], roots[0] / "escape" / "file"):
                with self.subTest(path=path), self.assertRaises(A.ArchiveError):
                    A._encode_files({path: b"x"}, (), config["roots"])
            path = roots[0] / "good"
            for ready, stamped in (([roots[0] / "missing"], []), ([path, path], []), ([], [path])):
                with self.assertRaises(A.ArchiveError):
                    A._encode_files({path: b"x"}, ready, config["roots"], stamped)

    def test_memfd_is_fully_sealed_and_exact(self):
        descriptor = A._memfd([b"first", b"\x00\xfflast"])
        try:
            self.assertEqual(os.pread(descriptor, 100, 0), b"first\x00\xfflast")
            self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & A.SEALS, A.SEALS)
            with self.assertRaises(OSError):
                os.pwrite(descriptor, b"X", 0)
            with self.assertRaises(OSError):
                os.ftruncate(descriptor, 0)
        finally:
            os.close(descriptor)

    def test_worker_preserves_bytes_and_stamps_ready_after_other_fsyncs(self):
        with configuration() as (_, roots, config):
            ready, normal = roots[1] / "bundle" / "READY.json", roots[0] / "ordinary.ready"
            payload = roots[1] / "bundle" / "native.npy"
            files = {ready: sentinel(), normal: b"untouched-ready", payload: b"\x00\x80exact", roots[0] / "empty": b""}
            message, descriptor = batch(config, files, [normal, ready], [ready])
            descriptors = A._root_fds(config)
            events, original = [], A._create_file

            def observed(root_fd, relative, data, **options):
                original(root_fd, relative, data, **options)
                events.append((relative, time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)))

            try:
                with mock.patch.object(A, "_create_file", observed):
                    written, finalized = A._write_batch(config, descriptors, message, descriptor, set())
            finally:
                os.close(descriptor)
                for item in descriptors:
                    os.close(item)
            self.assertEqual([item[0] for item in events], ["bundle/native.npy", "empty", "ordinary.ready", "bundle/READY.json"])
            self.assertEqual(payload.read_bytes(), files[payload])
            self.assertEqual(normal.read_bytes(), files[normal])
            self.assertEqual((roots[0] / "empty").read_bytes(), b"")
            actual = json.loads(ready.read_bytes())
            self.assertGreaterEqual(actual["ready_raw_ns"], events[-2][1])
            self.assertLessEqual(actual["ready_raw_ns"], events[-1][1])
            self.assertGreater(actual["ready_wall_ns"], 0)
            self.assertEqual(actual["payloads"], {"exact": "retained"})
            self.assertEqual(finalized, [dict(path=str(ready), sha256=hashlib.sha256(ready.read_bytes()).hexdigest(),
                ready_raw_ns=actual["ready_raw_ns"], ready_wall_ns=actual["ready_wall_ns"])])
            self.assertEqual(written[-1]["sha256"], finalized[0]["sha256"])

    def test_bad_sentinel_and_checksum_fail_before_any_write(self):
        for mode in ("schema", "clock", "bool", "checksum"):
            with self.subTest(mode=mode), configuration() as (_, roots, config):
                document = json.loads(sentinel())
                if mode == "schema": document["schema"] = "foreign"
                if mode == "clock": document["ready_raw_ns"] = 1
                if mode == "bool": document["ready_wall_ns"] = False
                ready, payload = roots[1] / "READY.json", roots[0] / "native"
                message, descriptor = batch(config, {payload: b"payload", ready: A.canonical(document)}, [ready], [ready])
                if mode == "checksum": message["files"][0]["sha256"] = "0" * 64
                descriptors = A._root_fds(config)
                try:
                    with self.assertRaises(A.ArchiveError):
                        A._write_batch(config, descriptors, message, descriptor, set())
                finally:
                    os.close(descriptor)
                    for item in descriptors: os.close(item)
                self.assertFalse(payload.exists())
                self.assertFalse(ready.exists())

    def test_worker_refuses_post_admission_symlink_and_preserves_partial_files(self):
        with configuration() as (_, roots, config):
            first, bad, ready = roots[0] / "first", roots[0] / "late" / "escape", roots[0] / "READY"
            message, descriptor = batch(config, {first: b"preserve", bad: b"never-outside", ready: b"ready"}, [ready])
            (roots[0] / "late").symlink_to(roots[1], target_is_directory=True)
            descriptors = A._root_fds(config)
            try:
                with self.assertRaises(OSError) as caught:
                    A._write_batch(config, descriptors, message, descriptor, set())
                self.assertEqual(len(caught.exception.archive_written), 1)
            finally:
                os.close(descriptor)
                for item in descriptors: os.close(item)
            self.assertEqual(first.read_bytes(), b"preserve")
            self.assertFalse((roots[1] / "escape").exists())
            self.assertFalse(ready.exists())

    def test_atomic_ready_remains_invisible_during_partial_staging_writes(self):
        with configuration() as (_, roots, config):
            descriptors = A._root_fds(config)
            ready, payload = roots[0] / "READY.json", b'{"atomic":"complete-only"}'
            real_write, real_link, real_fsync = os.write, os.link, os.fsync
            events = []

            def slow_write(descriptor, data):
                self.assertFalse(ready.exists(), "partial READY became visible")
                events.append("write")
                return real_write(descriptor, data[:2])

            def synchronized(descriptor):
                real_fsync(descriptor)
                events.append("fsync")

            def published(source, destination, **options):
                self.assertEqual(destination, "READY.json")
                self.assertFalse(ready.exists())
                self.assertEqual((roots[0] / source).read_bytes(), payload)
                self.assertEqual(events[-1], "fsync")
                events.append("link")
                return real_link(source, destination, **options)

            try:
                with mock.patch.object(A.os, "write", slow_write), mock.patch.object(A.os, "link", published), \
                        mock.patch.object(A.os, "fsync", synchronized):
                    A._create_file(descriptors[0], "READY.json", memoryview(payload), atomic=True)
            finally:
                for descriptor in descriptors: os.close(descriptor)
            self.assertEqual(ready.read_bytes(), payload)
            self.assertEqual(events[-3:], ["link", "fsync", "fsync"])
            self.assertEqual(sorted(path.name for path in roots[0].iterdir()), ["READY.json"])

    def test_atomic_ready_collision_preserves_existing_and_staged_evidence(self):
        with configuration() as (_, roots, config):
            ready = roots[0] / "READY.json"
            ready.write_bytes(b"existing")
            descriptors = A._root_fds(config)
            try:
                with self.assertRaises(FileExistsError):
                    A._create_file(descriptors[0], "READY.json", memoryview(b"new-staged"), atomic=True)
            finally:
                for descriptor in descriptors: os.close(descriptor)
            self.assertEqual(ready.read_bytes(), b"existing")
            staged = list(roots[0].glob(".archive-ready-*"))
            self.assertEqual(len(staged), 1)
            self.assertEqual(staged[0].read_bytes(), b"new-staged")


class ArchiveChildTests(unittest.TestCase):
    def client(self, roots, **options):
        return A.ArchiveClient(roots=roots, max_jobs=options.pop("max_jobs", 2),
            max_bytes=options.pop("max_bytes", 1024 * 1024), max_frames=options.pop("max_frames", 8),
            binding={"test": "owned-cpu"}, **options)

    def test_real_child_multi_root_exact_durability_and_clock_finalization(self):
        with configuration() as (_, roots, _):
            before_env, before_fds = dict(os.environ), len(os.listdir("/proc/self/fd"))
            client = self.client(roots)
            try:
                ready = roots[1] / "one" / "READY.json"
                first = client.submit(identity(1), {roots[0] / "one": b"first", ready: sentinel()}, [ready], [ready])
                second = client.submit(identity(2), {roots[0] / "two": b"second", roots[1] / "two": b""})
                self.assertEqual((first, second), (1, 2))
                report = client.close()
                self.assertEqual((report["submitted"], report["completed"], report["outstanding_jobs"]), (2, 2, 0))
                self.assertFalse(report["failed"])
                self.assertEqual(client.process.poll(), 0)
                self.assertEqual((roots[0] / "one").read_bytes(), b"first")
                self.assertEqual((roots[0] / "two").read_bytes(), b"second")
                final = report["completions"][0]["finalized_ready"][0]
                self.assertEqual(final["sha256"], hashlib.sha256(ready.read_bytes()).hexdigest())
                self.assertEqual(client.ready["context"]["environment"], A.ENVIRONMENT)
                self.assertEqual(client.ready["context"]["numerical_imports"], [])
                self.assertEqual(client.ready["context"]["parent_death_signal"], "SIGTERM")
                self.assertEqual(client.close(), report)
            finally:
                if not client.closed: client.abort()
            self.assertEqual(dict(os.environ), before_env)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before_fds)

    def test_submit_transfers_without_disk_ack_and_queue_bound_is_sticky(self):
        with configuration() as (_, roots, _):
            client = self.client(roots, timeout_s=2.)
            stopped = False
            try:
                os.kill(client.process.pid, signal.SIGSTOP)
                os.waitpid(client.process.pid, os.WUNTRACED)
                stopped = True
                self.assertEqual(client.submit(identity(1), {roots[0] / "one": b"one"}), 1)
                self.assertEqual(client.submit(identity(2), {roots[0] / "two": b"two"}), 2)
                self.assertEqual(client.outstanding_jobs, 2)
                self.assertEqual(client.outstanding_bytes, 6)
                self.assertFalse((roots[0] / "one").exists())
                self.assertFalse((roots[0] / "two").exists())
                self.assertTrue(all(row["ownership"] == "SEALED_KERNEL_TRANSFER_NOT_DURABLE" for row in client.submissions))
                with self.assertRaisesRegex(A.ArchiveError, "credits"):
                    client.submit(identity(3), {roots[0] / "three": b"three"})
                os.kill(client.process.pid, signal.SIGCONT)
                stopped = False
                with self.assertRaisesRegex(A.ArchiveError, "credits"):
                    client.close()
                self.assertEqual(client.close_result["completed"], 2)
                self.assertTrue(client.close_result["failed"])
                self.assertFalse((roots[0] / "three").exists())
            finally:
                if stopped and client.process.poll() is None: os.kill(client.process.pid, signal.SIGCONT)
                if not client.closed: client.abort()

    def test_existing_file_failure_preserves_partial_and_rejects_later_jobs(self):
        with configuration() as (_, roots, _):
            existing = roots[0] / "existing"
            existing.write_bytes(b"original")
            client = self.client(roots)
            stopped = False
            try:
                os.kill(client.process.pid, signal.SIGSTOP)
                os.waitpid(client.process.pid, os.WUNTRACED)
                stopped = True
                client.submit(identity(1), {roots[0] / "first": b"preserved", existing: b"overwrite",
                                          roots[0] / "READY": b"never"}, [roots[0] / "READY"])
                client.submit(identity(2), {roots[0] / "later": b"not-after-fault"})
                os.kill(client.process.pid, signal.SIGCONT)
                stopped = False
                with self.assertRaisesRegex(A.ArchiveError, "failed"):
                    client.close()
                self.assertEqual(existing.read_bytes(), b"original")
                self.assertEqual((roots[0] / "first").read_bytes(), b"preserved")
                self.assertFalse((roots[0] / "READY").exists())
                self.assertEqual(client.close_result["completed"], 0)
                self.assertEqual([row["status"] for row in client.completions], ["FAILED", "FAILED"])
                self.assertFalse((roots[0] / "later").exists())
                self.assertEqual(len(client.completions[0]["written"]), 1)
                self.assertEqual(client.process.poll(), 0)
            finally:
                if stopped and client.process.poll() is None: os.kill(client.process.pid, signal.SIGCONT)
                if not client.closed: client.abort()

    def test_drain_timeout_preserves_unresolved_identity_and_joins_child(self):
        with configuration() as (_, roots, _):
            client = self.client(roots)
            try:
                os.kill(client.process.pid, signal.SIGSTOP)
                os.waitpid(client.process.pid, os.WUNTRACED)
                client.submit(identity(7), {roots[0] / "unresolved": b"preserve-job"})
                client.timeout = .05
                with self.assertRaises(A.ArchiveError):
                    client.close()
                self.assertTrue(client.closed and client.failed)
                self.assertIsNotNone(client.process.poll())
                self.assertEqual(client.close_result["outstanding_jobs"], 1)
                self.assertEqual(client.close_result["submissions"][0]["identity"], identity(7))
                self.assertEqual(client.close_result["completed"], 0)
                self.assertFalse((roots[0] / "unresolved").exists())
            finally:
                if not client.closed: client.abort()

    def test_duplicate_and_oversized_admission_fail_closed(self):
        for mode in ("path", "identity", "bytes", "frame_bound"):
            with self.subTest(mode=mode), configuration() as (_, roots, _):
                client = self.client(roots, max_bytes=8, max_frames=1 if mode == "frame_bound" else 8)
                try:
                    client.submit(identity(1), {roots[0] / "one": b"one"})
                    value = identity(1) if mode == "identity" else identity(2)
                    files = {roots[0] / ("one" if mode == "path" else "two"): b"oversized" if mode == "bytes" else b"two"}
                    with self.assertRaises(A.ArchiveError): client.submit(value, files)
                    with self.assertRaises(A.ArchiveError): client.close()
                    self.assertEqual(client.close_result["submitted"], 1)
                    self.assertEqual(client.close_result["completed"], 1)
                    self.assertFalse((roots[0] / "two").exists())
                finally:
                    if not client.closed: client.abort()

    def test_real_restore_hook_cannot_strip_dedicated_child_caps(self):
        from rl_agent.splitfusion_live_gt_optimization_v5.test_bootstrap import _retained_hook
        spec = importlib.util.spec_from_file_location("archive_actual_retained_hook", _retained_hook())
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)
        original, environment = subprocess.Popen.__init__, dict(os.environ)
        hook.install_child_environment_restore({"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
        active = subprocess.Popen.__init__
        try:
            with configuration() as (_, roots, _):
                client = self.client(roots)
                try:
                    self.assertEqual(client.ready["context"]["environment"], A.ENVIRONMENT)
                    client.close()
                finally:
                    if not client.closed: client.abort()
            self.assertIs(subprocess.Popen.__init__, active)
            self.assertEqual(dict(os.environ), environment)
        finally:
            subprocess.Popen.__init__ = original

    def test_bad_ready_aborts_and_joins_owned_child_without_fd_leak(self):
        with configuration() as (_, roots, _):
            original = A._receive
            def corrupt(sock, timeout):
                response, descriptor = original(sock, timeout)
                if response.get("status") == "READY": response["context"]["pid"] = -1
                return response, descriptor
            client = A.ArchiveClient.__new__(A.ArchiveClient)
            before = len(os.listdir("/proc/self/fd"))
            with mock.patch.object(A, "_receive", corrupt), self.assertRaisesRegex(A.ArchiveError, "READY"):
                client.__init__(roots=roots, max_jobs=2, max_bytes=100, max_frames=2, binding={})
            self.assertTrue(client.closed)
            self.assertIsNotNone(client.process.poll())
            self.assertEqual(client.socket.fileno(), -1)
            self.assertEqual(len(os.listdir("/proc/self/fd")), before)

    def test_owner_death_terminates_archive_child(self):
        with configuration() as (_, roots, _):
            code = ("import time; from pathlib import Path; "
                    "from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient; "
                    f"c=ArchiveClient(roots=(Path({str(roots[0])!r}),),max_jobs=2,max_bytes=100,max_frames=2,binding={{}}); "
                    "print(c.process.pid,flush=True); time.sleep(30)")
            owner = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=dict(os.environ, **A.ENVIRONMENT), cwd=Path(__file__).resolve().parents[2])
            child_pid = None
            try:
                child_pid = int(owner.stdout.readline())
                owner.kill(); owner.wait(timeout=3.)
                deadline = time.monotonic() + 3.
                while time.monotonic() < deadline:
                    status = Path(f"/proc/{child_pid}/stat")
                    if not status.exists() or status.read_text().split()[2] == "Z":
                        break
                    time.sleep(.01)
                else:
                    self.fail("archive child survived owner death")
            finally:
                if owner.poll() is None: owner.kill(); owner.wait(timeout=3.)
                owner.stdout.close(); owner.stderr.close()


if __name__ == "__main__":
    unittest.main()
