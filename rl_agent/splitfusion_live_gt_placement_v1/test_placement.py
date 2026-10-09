"""Fresh-process CPU placement unit tests; no simulator, GPU or live partition."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class PlacementPolicyTests(unittest.TestCase):
    def child(self, body):
        code = """
import json, os, subprocess, sys, time
from unittest import mock
from rl_agent.splitfusion_live_gt_placement_v1 import placement as P
original = P.scheduling()
archive = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
try:
    policy = P.Placement(requested_cpu=min(original['affinity']), original_context=original,
                         archive_pid=archive.pid, max_frames=2)
""" + "\n".join("    " + line for line in body.splitlines()) + """
finally:
    if 'policy' in locals() and not policy.closed:
        policy.restore()
    archive.terminate()
    archive.wait(timeout=5)
assert P.scheduling() == original
"""
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2], timeout=30,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(completed.stdout)

    def test_active_request_counters_and_archive_mask_restore(self):
        result = self.child("""
policy.start()
assert P.scheduling()['affinity'] == [policy.requested_cpu]
policy.begin_request()
policy.identify_request({'frame_id': 17})
sum(value * value for value in range(50000))
policy.end_request('OK')
policy.before_drain()
policy.restore()
scope = policy.snapshot(include_requests=True)
assert scope['restored'] and scope['closed']
assert scope['request_count'] == 1
row = scope['request_samples'][0]
assert row['frame_id'] == 17 and row['before']['cpu'] == row['after']['cpu'] == policy.requested_cpu
assert all(value >= 0 for value in row['delta'].values())
assert row['delta']['thread_cpu_ns'] > 0 and row['delta']['run_ns'] > 0
for phase in ('archive_start', 'archive_end'):
    assert all(row['affinity'] == original['affinity'] for row in scope[phase]['threads'].values())
assert scope['set_calls'] == [os.getpid()] and scope['restore_calls'] == [os.getpid()]
print(json.dumps(scope))
""")
        self.assertEqual(result["cpu_id_provider"], "libc.sched_getcpu")

    def test_partial_startup_pin_failure_restores_owned_mask(self):
        self.child("""
actual = os.sched_setaffinity
calls = []
def fail_after_pin(tid, mask):
    calls.append((tid, mask))
    actual(tid, mask)
    if len(calls) == 1:
        raise RuntimeError('injected after affinity syscall')
with mock.patch.object(os, 'sched_setaffinity', side_effect=fail_after_pin):
    try:
        policy.start()
    except RuntimeError as error:
        assert 'injected' in str(error)
    else:
        raise AssertionError('partial startup fault accepted')
assert policy.closed and policy.restored
assert P.scheduling() == original
assert sorted(os.sched_getaffinity(archive.pid)) == original['affinity']
print(json.dumps(policy.snapshot()))
""")

    def test_foreign_archive_mask_rejected_before_pin(self):
        self.child("""
with mock.patch.object(policy, 'observe_archive', side_effect=ValueError('foreign archive mask')), \
        mock.patch.object(os, 'sched_setaffinity', side_effect=AssertionError('must not pin')):
    try:
        policy.start()
    except ValueError as error:
        assert 'foreign archive mask' in str(error)
    else:
        raise AssertionError('foreign archive scope accepted')
assert not policy.set_calls and P.scheduling() == original
print(json.dumps({'rejected_before_pin': True}))
""")

    def test_invalid_cpu_and_false_parent_context_rejected(self):
        self.child("""
for cpu, context in ((True, original), (-1, original), (max(original['affinity'])+10000, original),
                     (min(original['affinity']), dict(original, nice=original['nice']+1))):
    try:
        P.Placement(requested_cpu=cpu, original_context=context, archive_pid=archive.pid, max_frames=2)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid placement accepted')
assert P.scheduling() == original
print(json.dumps({'rejected': True}))
""")

    def test_owned_helper_tid_is_pinned_and_restored_with_main(self):
        self.child("""
import threading
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
    assert len(policy.original_threads) == 2
    assert set(policy.set_calls) == {os.getpid(), thread.native_id}
    assert sorted(os.sched_getaffinity(thread.native_id)) == [policy.requested_cpu]
    policy.restore()
    assert sorted(os.sched_getaffinity(thread.native_id)) == original['affinity']
    print(json.dumps(policy.snapshot()))
finally:
    stop.set()
    thread.join(timeout=5)
""")


if __name__ == "__main__":
    unittest.main()
