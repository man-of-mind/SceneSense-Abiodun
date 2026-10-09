"""Real CPU echo-child tests for a persistent PDEATHSIG creator thread."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from .process import ROOT
from .worker_owner import OwnedClient


def _sources():
    names = ("__init__.py", "ipc.py", "process.py", "worker_owner.py")
    base = Path(__file__).parent
    return {str((base / name).relative_to(ROOT)):
            hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}


def _owner(directory, root):
    return OwnedClient("echo_test_only", {"offline_test": True},
        directory=directory, writable_attempt_root=root,
        source_sha256=_sources(), cpu=min(os.sched_getaffinity(0)), timeout_s=15)


def _process_dead(pid, original_start):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
    except FileNotFoundError:
        return True
    return fields[2] == "Z" or fields[21] != original_start


class WorkerOwnerTests(unittest.TestCase):
    def test_real_child_survives_frame_creator_exit_then_closes_cleanly(self):
        with tempfile.TemporaryDirectory(prefix="numeric-owned-", dir="/tmp") as root:
            holder = {}
            def frame_creator():
                try:
                    holder["client"] = _owner(Path(root) / "worker", root)
                except BaseException as exc:
                    holder["error"] = exc
            frame = threading.Thread(target=frame_creator, name="short-lived-frame")
            frame.start(); frame.join(timeout=25)
            self.assertFalse(frame.is_alive())
            if "error" in holder:
                raise holder["error"]
            client = holder["client"]
            try:
                self.assertTrue(client.owner_thread_alive)
                self.assertNotEqual(client.owner_native_id, frame.native_id)
                self.assertNotEqual(client.ready["pid"], os.getpid())
                self.assertEqual(client.call("ECHO", {"after_frame_exit": True}),
                                 {"after_frame_exit": True})
                client.begin("ECHO", {"bounded": b"x" * 100_000})
                self.assertIsNotNone(client.pending)
                self.assertEqual(client.finish(), {"bounded": b"x" * 100_000})
                self.assertEqual(client.close(), {})
                self.assertFalse(client.owner_thread_alive)
                self.assertEqual(client.process.returncode, 0)
                self.assertEqual([item["status"] for item in client.events[-3:]],
                                 ["OK", "OK", "OK"])
            finally:
                if not client.closed:
                    client.abort()

    def test_worker_error_keeps_exact_event_then_owned_abort_reaps(self):
        with tempfile.TemporaryDirectory(prefix="numeric-owned-", dir="/tmp") as root:
            client = _owner(Path(root) / "worker", root)
            try:
                with self.assertRaisesRegex(ValueError, "worker failed"):
                    client.call("UNKNOWN", {})
                self.assertTrue(client.failed)
                self.assertTrue(client.owner_thread_alive)
                self.assertEqual(client.events[-1]["status"], "FAILED")
                with self.assertRaises(ValueError):
                    client.close()
                self.assertTrue(client.closed)
                self.assertFalse(client.owner_thread_alive)
                self.assertIsNotNone(client.process.poll())
            finally:
                if not client.closed:
                    client.abort()

    def test_host_death_still_delivers_child_parent_death_signal(self):
        # Kill only this test's helper parent. Its owner thread remains alive
        # until that kill, so the child must die from the unchanged PDEATHSIG.
        script = '''\
import os, sys, time
from pathlib import Path
from rl_agent.splitfusion_numeric_gt_runtime_v1.test_worker_owner import _owner
client = _owner(Path(sys.argv[1]) / "worker", sys.argv[1])
print(client.ready["pid"], flush=True)
while True:
    time.sleep(1)
'''
        with tempfile.TemporaryDirectory(prefix="numeric-pdeath-", dir="/tmp") as root:
            host = subprocess.Popen([sys.executable, "-B", "-c", script, root],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
            child_pid = None
            original_start = None
            try:
                readable, _, _ = select.select([host.stdout], [], [], 25)
                self.assertTrue(readable, "owned echo-child startup timed out")
                line = host.stdout.readline().strip()
                self.assertTrue(line.isdecimal(), line)
                child_pid = int(line)
                self.assertNotEqual(child_pid, host.pid)
                stat = Path(f"/proc/{child_pid}/stat").read_text().split()
                original_start = stat[21]
                self.assertNotEqual(stat[2], "Z")
                host.kill(); host.wait(timeout=5)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not _process_dead(child_pid, original_start):
                    time.sleep(.02)
                self.assertTrue(_process_dead(child_pid, original_start),
                                "owned child survived its parent process death")
            finally:
                if host.poll() is None:
                    host.kill(); host.wait(timeout=5)
                if child_pid is not None:
                    try:
                        fields = Path(f"/proc/{child_pid}/stat").read_text().split()
                    except FileNotFoundError:
                        fields = None
                    if fields is not None and fields[2] != "Z" and fields[21] == original_start:
                        os.kill(child_pid, signal.SIGKILL)
                host.stdout.close(); host.stderr.close()


if __name__ == "__main__":
    unittest.main()
