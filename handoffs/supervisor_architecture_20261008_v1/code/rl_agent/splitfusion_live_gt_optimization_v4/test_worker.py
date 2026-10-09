"""V4 runs all V3 contract tests unchanged plus V4-specific lifecycle proofs.

Loading a private test module binds its global W to V4 only; the original test
source and V3 runtime are neither edited nor monkeypatched.
"""
import importlib.util
import json
import os
from pathlib import Path
import time
import unittest
from unittest import mock

from . import worker as W
from . import memfd_transport as M

_path = Path(__file__).parents[1] / 'splitfusion_live_gt_optimization_v3' / 'test_worker.py'
_spec = importlib.util.spec_from_file_location(
    'rl_agent.splitfusion_live_gt_optimization_v3._v4_test_contract_copy', _path)
V3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V3)
V3.W = W


class FramingTests(V3.FramingTests):
    pass


class EngineTests(V3.EngineTests):
    pass


class ChildTests(V3.ChildTests):
    pass


class V4SpecificTests(unittest.TestCase):
    def test_console_creation_failure_closes_both_owned_sockets(self):
        with V3.fixture() as (root, _kwargs, _radar):
            sockets = []
            real_pair, real_open = W.socket.socketpair, Path.open
            def pair(*args, **kwargs):
                result = real_pair(*args, **kwargs)
                sockets.extend(result)
                return result
            def opening(path, *args, **kwargs):
                if path.name == 'worker.log':
                    raise OSError('injected full disk for worker console')
                return real_open(path, *args, **kwargs)
            try:
                with mock.patch.object(W.socket, 'socketpair', side_effect=pair), \
                        mock.patch.object(Path, 'open', opening):
                    try:
                        W.Client(attempt_dir=root / 'child', catalogue_dir=root / 'catalogue',
                                 max_frames=2, expected_sources=W.sources())
                    except OSError as error:
                        self.assertIn('injected full disk', str(error))
                        self.assertEqual(len(sockets), 2)
                        self.assertTrue(all(value.fileno() == -1 for value in sockets))
                    else:
                        self.fail('console creation fault did not propagate')
            finally:
                for value in sockets:
                    value.close()

    def test_child_readiness_preserves_scheduling_and_caps_only_child(self):
        with V3.fixture() as (root, _kwargs, _radar):
            expected = W.scheduling_context()
            before_env = dict(os.environ)
            client = W.Client(attempt_dir=root / 'child', catalogue_dir=root / 'catalogue',
                              max_frames=2, expected_sources=W.sources())
            try:
                context = client.ready['process_context']
                self.assertEqual(context['pid'], client.process.pid)
                self.assertEqual(context['scheduling'], expected)
                self.assertTrue(all(value == expected for value in context['threads'].values()))
                self.assertFalse(context['cuda_initialized'])
                self.assertEqual(context['torch_threads'], 1)
                self.assertEqual(context['environment'], {'CUDA_VISIBLE_DEVICES': '',
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                result = client.close()
                self.assertEqual(result['process_context']['scheduling'], expected)
                self.assertEqual(dict(os.environ), before_env)
                self.assertEqual(W.scheduling_context(), expected)
            finally:
                if not client.closed:
                    client.abort()

    def test_malformed_sealed_snapshot_fails_without_target_or_orphan(self):
        with V3.fixture() as (root, kwargs, _radar):
            client = W.Client(attempt_dir=root / 'child', catalogue_dir=root / 'catalogue',
                              max_frames=2, expected_sources=W.sources())
            descriptor, size = M.create([b'X' * 100])
            try:
                with self.assertRaisesRegex(W.WorkerError, 'snapshot magic mismatch'):
                    client._exchange(dict(command='RUN', directory=str(root / 'edge'),
                        sent_raw_ns=W.raw(), expected_identity=kwargs['identity'], snapshot_bytes=size),
                        descriptor=descriptor)
                self.assertTrue(client.closed and client.failed)
                self.assertIsNotNone(client.process.poll())
                self.assertFalse((root / 'edge').exists())
                manifest = root / 'child' / 'engine' / 'complete_gt_stage_timing' / 'MANIFEST.json'
                self.assertEqual(json.loads(manifest.read_bytes())['records'], 0)
            finally:
                os.close(descriptor)
                client.abort()

    def test_original_semantic_attempt_failure_path_and_real_worker_result(self):
        with V3.fixture() as (root, kwargs, radar):
            semantic = root / 'original_attempt'
            semantic.mkdir()
            client = W.Client(attempt_dir=root / 'child', catalogue_dir=root / 'catalogue',
                max_frames=2, expected_sources=W.sources(), source_attempt_name=semantic.name,
                semantic_attempt_dir=str(semantic))
            kwargs['frozen_world'].population_evidence['registry_complete'] = False
            try:
                with self.assertRaises(W.WorkerError) as caught:
                    client.run(directory=root / 'edge', kwargs=kwargs, radar=radar,
                               sent_raw_ns=W.raw(), sync_config=V3.sync_config(root))
                result = caught.exception.worker_result
                self.assertEqual(result['status'], 'FAILED')
                self.assertEqual(result['identity'], kwargs['identity'])
                self.assertEqual(result['sync']['accepted'], 1)
                self.assertEqual([r['completed'] for r in result['rows']], [True, False])
                self.assertEqual(result['failure_close']['records'], 2)
                self.assertTrue(all(r['process_pid'] == client.process.pid for r in result['rows']))
                rejection = semantic / 'parity_rejected_targets' / V3.T.bundle_path(root, kwargs['identity']).name
                document = json.loads((rejection / 'REJECTION.json').read_bytes())
                self.assertEqual(document['stage'], 'complete_source_build')
                self.assertFalse(document['target_ready'])
                for name, digest in document['sha256'].items():
                    self.assertEqual(V3.sha(rejection / name), digest)
                self.assertFalse((root / 'child' / 'engine' / semantic.name / 'parity_rejected_targets').exists())
                self.assertIsNotNone(client.process.poll())
                self.assertTrue(client.closed and client.failed)
            finally:
                client.abort()

    def test_many_real_requests_preserve_fd_count_and_same_worker_pid(self):
        with V3.fixture() as (root, kwargs, radar):
            baseline = len(os.listdir('/proc/self/fd'))
            client = W.Client(attempt_dir=root / 'child', catalogue_dir=root / 'catalogue',
                              max_frames=8, expected_sources=W.sources())
            active = len(os.listdir('/proc/self/fd'))
            try:
                for index in range(8):
                    if index:
                        V3.advance(kwargs, frame=100 + index)
                    result = client.run(directory=root / 'edge', kwargs=kwargs, radar=radar,
                                        sent_raw_ns=W.raw(), sync_config=V3.sync_config(root))
                    self.assertEqual(result['processed'], index + 1)
                    self.assertEqual(result['worker']['pid'], client.process.pid)
                    self.assertEqual(len(os.listdir('/proc/self/fd')), active)
                    self.assertEqual(result['sync']['accepted'], index + 1)
                result = client.close()
                self.assertEqual(result['timing']['records'], 40)
            finally:
                if not client.closed:
                    client.abort()
            self.assertEqual(len(os.listdir('/proc/self/fd')), baseline)

    def test_concurrent_request_refused_before_descriptor_or_child_work(self):
        with V3.fixture() as (root, kwargs, radar):
            client = W.Client(attempt_dir=root / 'child', catalogue_dir=root / 'catalogue',
                              max_frames=2, expected_sources=W.sources())
            try:
                client.lock.acquire()
                try:
                    with self.assertRaisesRegex(W.WorkerError, 'concurrent'):
                        client.run(directory=root / 'edge', kwargs=kwargs, radar=radar, sent_raw_ns=W.raw())
                finally:
                    client.lock.release()
                self.assertEqual(client.count, 0)
                self.assertFalse(client.failed)
                self.assertEqual(client.close()['processed'], 0)
            finally:
                if not client.closed:
                    client.abort()


if __name__ == '__main__':
    unittest.main()
