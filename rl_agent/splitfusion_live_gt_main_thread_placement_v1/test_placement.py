"""Fresh-process tests for narrow main-TID ownership and mixed-mask honesty."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class MainThreadPolicyTests(unittest.TestCase):
    def child(self, body):
        code = """
import json, os, subprocess, sys, threading
from unittest import mock
from rl_agent.splitfusion_live_gt_main_thread_placement_v1 import placement as P
original = P.scheduling()
archive = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
policy = None
try:
""" + "\n".join("    " + line for line in body.splitlines()) + """
finally:
    if policy is not None and not policy.closed:
        policy.restore()
    archive.terminate()
    archive.wait(timeout=5)
assert P.scheduling() == original
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
            timeout=30, cwd=Path(__file__).resolve().parents[2],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_only_main_tid_is_changed_and_helpers_keep_original_mask(self):
        self.child("""
ready, stop = threading.Event(), threading.Event()
def helper():
    ready.set()
    stop.wait(timeout=5)
thread = threading.Thread(target=helper)
thread.start()
ready.wait(timeout=5)
try:
    policy = P.Placement(requested_cpu=min(original['affinity']), original_context=original,
                         archive_pid=archive.pid, max_frames=2)
    policy.start()
    assert policy.set_calls == [os.getpid()]
    assert sorted(os.sched_getaffinity(thread.native_id)) == original['affinity']
    policy.begin_request()
    policy.identify_request({'frame_id': 1})
    sum(i*i for i in range(10000))
    policy.end_request('OK')
    policy.before_drain()
    policy.restore()
    assert policy.restored and policy.restore_calls == [os.getpid()]
    assert sorted(os.sched_getaffinity(thread.native_id)) == original['affinity']
    scope = policy.snapshot(include_requests=True)
    assert scope['affinity_scope'] == 'main_gt_tid_only'
    assert scope['request_samples'][0]['before']['cpu'] == policy.requested_cpu
    assert scope['request_samples'][0]['after']['cpu'] == policy.requested_cpu
    print(json.dumps(scope))
finally:
    stop.set()
    thread.join(timeout=5)
""")

    def test_late_singleton_helper_is_rejected_without_helper_fixup(self):
        self.child("""
policy = P.Placement(requested_cpu=min(original['affinity']), original_context=original,
                     archive_pid=archive.pid, max_frames=2)
policy.start()
ready, stop = threading.Event(), threading.Event()
def helper():
    ready.set()
    stop.wait(timeout=5)
thread = threading.Thread(target=helper)
thread.start()
ready.wait(timeout=5)
try:
    if len(original['affinity']) > 1:
        try:
            policy.verify()
        except ValueError as error:
            assert 'scheduling differs' in str(error)
        else:
            raise AssertionError('inherited singleton helper accepted')
        assert sorted(os.sched_getaffinity(thread.native_id)) == [policy.requested_cpu]
        assert policy.set_calls == [os.getpid()]
finally:
    stop.set()
    thread.join(timeout=5)
policy.restore()
assert policy.restored and policy.restore_calls == [os.getpid()]
print(json.dumps(policy.snapshot()))
""")

    def test_partial_main_pin_fault_restores_without_archive_mutation(self):
        self.child("""
policy = P.Placement(requested_cpu=min(original['affinity']), original_context=original,
                     archive_pid=archive.pid, max_frames=2)
actual, calls = os.sched_setaffinity, []
def fail_once(tid, mask):
    calls.append(tid)
    actual(tid, mask)
    if len(calls) == 1:
        raise RuntimeError('injected after main pin')
with mock.patch.object(os, 'sched_setaffinity', side_effect=fail_once):
    try:
        policy.start()
    except RuntimeError as error:
        assert 'injected' in str(error)
    else:
        raise AssertionError('partial main pin fault accepted')
assert calls == [os.getpid(), os.getpid()] and policy.restored
assert sorted(os.sched_getaffinity(archive.pid)) == original['affinity']
print(json.dumps(policy.snapshot()))
""")


if __name__ == '__main__':
    unittest.main()
