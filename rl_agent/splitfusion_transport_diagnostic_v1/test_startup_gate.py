import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
import startup_gate as G


class GateTests(unittest.TestCase):
    def test_setup_barrier_preserves_result_loop_and_live_methods(self):
        class Runtime:
            def __init__(self):
                self.thread = threading.current_thread()
                self.sent = 0; self._run4_ready = threading.Event(); self._run4_ready.set()
                self._feedback_rows = []
                self.receiver = Mock(); self.receiver.getsockname.return_value = ('10.0.0.2', 51014)
                self.remote = ('192.168.70.140', 51002)
            def _result_loop(self): return 'unchanged'
            def submit(self): return 'unchanged'
        receive, submit = Runtime._result_loop, Runtime.submit
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'LISTENER_RELEASE.json').write_text('{}')
            wrapped = G.wrap_runtime_class(Runtime, root)
            instance = wrapped()
            record = json.loads((root / 'LISTENER_READY.json').read_text())
            self.assertEqual(record['sent'], 0)
            self.assertEqual(record['listener']['tid'], threading.get_native_id())
            self.assertIs(Runtime._result_loop, receive); self.assertIs(Runtime.submit, submit)

    def test_stop_return_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'COLLECTORS_STOPPED.json').write_text('{}')
            stop = Mock(return_value=True)
            self.assertTrue(G.teardown_gate(stop, root)())
            stop.assert_called_once_with()

    def test_stop_still_runs_on_collector_timeout(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(G, 'wait', side_effect=TimeoutError('test')):
            root = Path(directory); stop = Mock(return_value=False)
            self.assertFalse(G.teardown_gate(stop, root)())
            self.assertTrue((root / 'COLLECTOR_DRAIN_TIMEOUT.json').exists())
            stop.assert_called_once_with()
