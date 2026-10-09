"""Matched control observation must make no affinity syscalls, even on close."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ControlPolicyTests(unittest.TestCase):
    def test_control_has_zero_affinity_writes_through_request_and_close(self):
        code = """
import os, subprocess, sys
from unittest import mock
from rl_agent.splitfusion_live_gt_placement_trial_v1.placement import Placement
from rl_agent.splitfusion_live_gt_placement_v1.placement import scheduling
original = scheduling()
archive = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
try:
    with mock.patch.object(os, 'sched_setaffinity', side_effect=AssertionError('control affinity mutation')):
        placement = Placement(mode='control', original_context=original, archive_pid=archive.pid, max_frames=1)
        placement.start()
        placement.begin_request()
        placement.identify_request({'frame_id': 1})
        sum(i*i for i in range(10000))
        placement.end_request('OK')
        placement.before_drain()
        placement.restore()
        scope = placement.snapshot(include_requests=True)
        assert scope['requested_cpu'] is None and scope['mode'] == 'control_unbound'
        assert scope['set_calls'] == scope['restore_calls'] == []
        assert scope['restored'] and scope['closed'] and scope['request_count'] == 1
        assert scheduling() == original
finally:
    archive.terminate()
    archive.wait(timeout=5)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2], timeout=30,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unregistered_mode_or_cpu_is_rejected_before_spawn(self):
        from .client import Client
        for mode, cpu in (("other", None), (True, None), ("control", 4), ("p4", 2), ("p4", True)):
            with self.subTest(mode=mode, cpu=cpu):
                with self.assertRaises(Exception):
                    Client(mode=mode, placement_cpu=cpu)


if __name__ == "__main__":
    unittest.main()
