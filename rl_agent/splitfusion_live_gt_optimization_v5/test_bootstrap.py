"""CPU-only startup regression using the actual retained restoration hook."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from . import contract as C


def _retained_hook():
    repo = Path(__file__).resolve().parents[2]
    for root in (repo, repo.parent / "abiodun_run4_single_gpu_profiler_v1"):
        path = root / "rl_agent/splitfusion_single_gpu_profiler_v1/threadcap_v1.py"
        if path.is_file():
            return path
    raise FileNotFoundError("actual retained threadcap_v1.py is required")


def _hook_probe():
    """One fresh parent, ordinary children, and the real V4/V5 CPU workers."""
    from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as F
    from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
    from .client import Client
    import cv2
    import torch

    hook_path = _retained_hook()
    hook_digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("retained_v5_hook_regression", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    before = (dict(os.environ), W.scheduling_context(), torch.get_num_threads(),
              torch.get_num_interop_threads(), cv2.getNumThreads(), torch.cuda.is_initialized())
    frozen_sources = W.sources()
    baseline_popen = subprocess.Popen.__init__
    hook.install_child_environment_restore({"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None})
    hooked_popen = subprocess.Popen.__init__
    clients = []

    def ordinary():
        requested = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                         OPENBLAS_NUM_THREADS="7", CUDA_VISIBLE_DEVICES="ordinary-child")
        keys = tuple(C.ENVIRONMENT)
        code = "import json,os; print(json.dumps({k:os.environ.get(k) for k in " + repr(keys) + "}))"
        completed = subprocess.run([sys.executable, "-c", code], env=requested,
                                   capture_output=True, text=True, timeout=15., check=True)
        actual = json.loads(completed.stdout)
        assert actual == {"CUDA_VISIBLE_DEVICES": "ordinary-child", "OMP_NUM_THREADS": None,
                          "MKL_NUM_THREADS": None, "OPENBLAS_NUM_THREADS": "7"}, actual
        return actual

    try:
        ordinary_before = ordinary()
        with F.fixture() as (root, kwargs, radar):
            old = W.Client(attempt_dir=root / "old", catalogue_dir=root / "catalogue",
                           max_frames=2, expected_sources=W.sources())
            clients.append(old)
            old_context = old.ready["process_context"]
            assert old_context["environment"]["OMP_NUM_THREADS"] is None
            assert old_context["environment"]["MKL_NUM_THREADS"] is None
            old.close()

            engine = W.Engine(attempt_dir=root / "direct", catalogue_dir=root / "catalogue",
                              max_frames=2, expected_sources=W.sources(), allow_replay_clock=True,
                              source_attempt_name="v5_bootstrap_parity")
            try:
                expected = engine.process(directory=root / "direct_edge", kwargs=kwargs, radar=radar,
                    sent_raw_ns=W.raw(), sync_config=F.sync_config(root / "direct"),
                    replay_observed_raw_ns=123)
                assert expected["status"] == "OK", expected.get("error")
            finally:
                engine.close()

            worker = Client(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                            max_frames=2, expected_sources=W.sources(),
                            expected_bootstrap_sources=C.sources(), allow_replay_clock=True,
                            source_attempt_name="v5_bootstrap_parity")
            clients.append(worker)
            ready = worker.ready["process_context"]
            C.validate_context(ready, C.sources())
            assert ready["scheduling"] == before[1]
            assert all(value == before[1] for value in ready["threads"].values())
            actual = worker.run(directory=root / "child_edge", kwargs=kwargs, radar=radar,
                sent_raw_ns=W.raw(), sync_config=F.sync_config(root / "child"),
                replay_observed_raw_ns=123)
            assert actual["status"] == "OK" and actual["processed"] == 1
            expected_bundle = F.T.verify_full_inputs(root / "direct_edge", expected_identity=kwargs["identity"])
            actual_bundle = F.T.verify_full_inputs(root / "child_edge", expected_identity=kwargs["identity"])
            assert actual_bundle.manifest["payloads"] == expected_bundle.manifest["payloads"]
            assert actual_bundle.prepared.content_sha256 == expected_bundle.prepared.content_sha256
            assert F.tree(Path(actual["record"]["complete_sources"]["directory"])) == F.tree(
                Path(expected["record"]["complete_sources"]["directory"]))
            first_avo = json.loads(Path(expected["record"]["path"]).read_bytes())
            child_avo = json.loads(Path(actual["record"]["path"]).read_bytes())
            for key in ("avo", "raw_pedestrian_rows", "payloads"):
                assert first_avo[key] == child_avo[key], key
            F.advance(kwargs, replenish=True)
            second = worker.run(directory=root / "child_edge", kwargs=kwargs, radar=radar,
                sent_raw_ns=W.raw(), sync_config=F.sync_config(root / "child"),
                replay_observed_raw_ns=124)
            assert second["processed"] == second["sync"]["accepted"] == 2
            assert second["worker"]["pid"] == actual["worker"]["pid"] == worker.process.pid
            closed = worker.close()
            assert closed["timing"]["records"] == 10 and worker.process.poll() == 0
            C.validate_context(closed["process_context"], C.sources())

        ordinary_after = ordinary()
        after = (dict(os.environ), W.scheduling_context(), torch.get_num_threads(),
                 torch.get_num_interop_threads(), cv2.getNumThreads(), torch.cuda.is_initialized())
        assert after == before, "worker changed parent settings"
        assert subprocess.Popen.__init__ is hooked_popen, "worker changed retained launch hook"
        assert hashlib.sha256(hook_path.read_bytes()).hexdigest() == hook_digest
        assert W.sources() == frozen_sources, "frozen worker/GT sources changed"
        assert hook.restore_counters()["launches"] == 4
        return dict(old_context=old_context, ready=ready, ordinary_before=ordinary_before,
                    ordinary_after=ordinary_after, hook_sha256=hook_digest,
                    hook_counters=hook.restore_counters(), parent_unchanged=True,
                    parity_verified=True, same_worker_two_requests=True,
                    frozen_sources=W.sources(), bootstrap_sources=C.sources())
    finally:
        for client in clients:
            if not client.closed:
                client.abort()
        subprocess.Popen.__init__ = baseline_popen


class BootstrapTests(unittest.TestCase):
    def test_real_retained_hook_parent_isolation_and_exact_request_parity(self):
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        completed = subprocess.run([sys.executable, "-m",
            "rl_agent.splitfusion_live_gt_optimization_v5.test_bootstrap", "--hook-probe"],
            cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True,
            text=True, timeout=90.)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        proof = json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                               if line.startswith("V5_BOOTSTRAP_PROOF=")))
        self.assertEqual(proof["ready"]["environment"], C.ENVIRONMENT)
        self.assertEqual(proof["ready"]["torch_threads"], 1)
        self.assertTrue(proof["parent_unchanged"] and proof["parity_verified"])
        self.assertEqual(proof["ordinary_before"], proof["ordinary_after"])

    def test_effective_contract_rejects_missing_wrong_or_unbound_caps(self):
        valid = dict(environment=dict(C.ENVIRONMENT), torch_threads=1, cuda_initialized=False,
                     bootstrap_sources=C.sources(), torch_interop_threads=24, opencv_threads=24)
        self.assertEqual(C.validate_context(valid, C.sources()), valid)
        mutations = [("torch_threads", 24), ("torch_threads", True), ("torch_threads", None),
                     ("cuda_initialized", True), ("bootstrap_sources", {})]
        for key, value in mutations:
            bad = copy.deepcopy(valid)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                C.validate_context(bad, C.sources())
        for key in C.ENVIRONMENT:
            bad = copy.deepcopy(valid)
            bad["environment"][key] = None
            with self.subTest(variable=key), self.assertRaises(ValueError):
                C.validate_context(bad, C.sources())

    def test_bootstrap_refuses_preimported_numerics_without_changing_parent(self):
        from . import bootstrap
        before = dict(os.environ)
        with mock.patch.dict(sys.modules, {"numpy": mock.Mock()}):
            with self.assertRaisesRegex(RuntimeError, "precede numerical imports"):
                bootstrap.main()
        self.assertEqual(dict(os.environ), before)

    def test_rejected_ready_caps_or_binding_kills_joins_and_closes_descriptors(self):
        from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as F
        from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
        from .client import Client
        real_receive = W.receive
        for key, value in (("torch_threads", 24), ("bootstrap_sources", {}), ("pid", -1)):
            with self.subTest(key=key), F.fixture() as (root, _kwargs, _radar):
                def corrupted(sock, deadline):
                    response, body = real_receive(sock, deadline)
                    if response.get("status") == "OK":
                        target = response if key == "pid" else response["process_context"]
                        target[key] = value
                    return response, body
                client = Client.__new__(Client)
                descriptor_count = len(os.listdir("/proc/self/fd"))
                try:
                    with mock.patch.object(W, "receive", side_effect=corrupted), \
                            self.assertRaises((ValueError, W.WorkerError)):
                        client.__init__(attempt_dir=root / "child", catalogue_dir=root / "catalogue",
                            max_frames=2, expected_sources=W.sources(), expected_bootstrap_sources=C.sources())
                    self.assertIsNotNone(client.process.poll())
                    self.assertTrue(client.failed and client.closed and client.console.closed)
                    self.assertEqual(client.socket.fileno(), -1)
                    self.assertEqual(len(os.listdir("/proc/self/fd")), descriptor_count)
                    self.assertFalse((root / "edge").exists())
                finally:
                    if getattr(client, "process", None) is not None and client.process.poll() is None:
                        client.abort()

    def test_rejected_ready_prevents_retained_collector_sensor_start(self):
        from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
        from rl_agent.splitfusion_live_parity_runtime_v7.test_worker_integration import RealWorkerCompositionTests
        from rl_agent.splitfusion_live_parity_repair_v1 import composition
        from rl_agent.splitfusion_live_parity_repair_v2 import source_ownership as SO
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        from .client import Client
        from . import runtime as R
        real_receive = W.receive
        clients = []

        def factory(**kwargs):
            client = Client.__new__(Client)
            clients.append(client)
            client.__init__(**kwargs)
            return client

        def corrupted(sock, deadline):
            response, body = real_receive(sock, deadline)
            if response.get("status") == "OK":
                response["process_context"]["environment"]["OMP_NUM_THREADS"] = None
            return response, body

        fixture_owner = RealWorkerCompositionTests()
        with fixture_owner.fixture() as f:
            try:
                with mock.patch.object(W, "receive", side_effect=corrupted), \
                        mock.patch.object(R, "Client", side_effect=factory), self.assertRaises(ValueError):
                    with SO.install_source_ownership(attempt_dir=f.attempt, max_frames=2):
                        with R.install_cpu_gt_worker(attempt_dir=f.attempt, max_frames=2,
                                expected_sources=W.sources(), expected_bootstrap_sources=C.sources()):
                            with composition.ue_scope(attempt_dir=f.attempt, max_frames=2):
                                f.owner = LG.build_live_gt_collector(f.Base)(attempt_dir=f.attempt)
                self.assertNotIn("sensor_fixture_started", f.events)
                self.assertEqual(len(clients), 1)
                self.assertIsNotNone(clients[0].process.poll())
                self.assertTrue(clients[0].failed and clients[0].closed)
                scope = json.loads((f.attempt / "GT_WORKER_SCOPE.json").read_bytes())
                self.assertFalse(scope["complete"])
                self.assertEqual(scope["bootstrap_sources"], C.sources())
                self.assertTrue(scope["effective_caps_required_before_sensor_start"])
            finally:
                for client in clients:
                    if getattr(client, "process", None) is not None and client.process.poll() is None:
                        client.abort()


if __name__ == "__main__":
    if sys.argv[1:] == ["--hook-probe"]:
        print("V5_BOOTSTRAP_PROOF=" + json.dumps(_hook_probe(), sort_keys=True))
    else:
        unittest.main()
