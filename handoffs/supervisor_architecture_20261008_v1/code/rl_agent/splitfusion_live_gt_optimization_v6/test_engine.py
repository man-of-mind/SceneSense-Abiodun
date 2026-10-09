"""Actual complete-GT engines and local CPU child lifecycle, without a route."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType
import unittest
from unittest import mock

import numpy as np

from rl_agent.splitfusion_live_gt_optimization_v3 import test_worker as F
from rl_agent.splitfusion_live_gt_optimization_v3.benchmark_offload import without_avo_clocks
from rl_agent.splitfusion_live_gt_optimization_v4 import snapshot as snapshot
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_gt_optimization_v5 import test_bootstrap as V5
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import gt_bridge as G
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from . import client as client_module
from . import contract as C
from .engine import Engine


def bindings():
    return (S.LiveCompleteTargetProducer, A.build_complete_v010,
            E.prepare_target, E.restore_prepared_target, F.T.write_bundle,
            G.write_source_archive, LG.write_avo_bundle, LG.decode_depth,
            S._original()[3].decode_depth_bgra,
            tuple((key, id(value)) for key, value in S._original()[2].items()))


def v6_copy(function):
    """Retain frozen test code while resolving relative clients/contracts to V6."""
    namespace = dict(function.__globals__, C=C, __package__=__package__, __spec__=__spec__)
    return FunctionType(function.__code__, namespace, function.__name__,
                        function.__defaults__, function.__closure__)


def require_depth_stats(stats, frame):
    expected = dict(frame_id=frame, calls=4, decodes=1, reuses=3, fallbacks=0,
                    completed=True, active_after_request=False, retained_after_request=False)
    if stats != expected:
        raise AssertionError(f"unexpected request-local depth counts: {stats!r}")


def hook_probe():
    """Run the original actual-hook proof and observe authentic V6 responses."""
    observed = []
    original = client_module.Client

    class ObservedClient(original):
        def _exchange(self, meta, body=b"", descriptor=None):
            result = super()._exchange(meta, body=body, descriptor=descriptor)
            payload = result["result"] if meta["command"] == "RUN" else result
            row = dict(command=meta["command"], context=payload.get("process_context"))
            if meta["command"] == "RUN":
                require_depth_stats(payload["depth_reuse"], payload["identity"]["frame_id"])
                row.update(frame_id=payload["identity"]["frame_id"],
                           depth_reuse=payload["depth_reuse"])
            observed.append(row)
            return result

    with mock.patch.object(client_module, "Client", ObservedClient):
        proof = v6_copy(V5._hook_probe)()
    if [row["command"] for row in observed] != ["INIT", "RUN", "RUN", "CLOSE"]:
        raise AssertionError("authentic V6 child lifecycle differs")
    for row in observed:
        C.validate_context(row["context"], C.sources())
    proof["v6_responses"] = observed
    return proof


class EngineTests(unittest.TestCase):
    def test_two_sequential_frames_match_all_payloads_sources_and_avo(self):
        with F.fixture() as (root, kwargs, radar):
            first = snapshot.pack_snapshot(kwargs, radar)
            F.advance(kwargs, replenish=True)
            second = snapshot.pack_snapshot(kwargs, radar)
            requests = (first, second)
            frozen_sources, candidate_sources = W.sources(), C.sources()
            original_bindings = bindings()
            results = {}
            for label, engine_type in (("baseline", W.Engine), ("candidate", Engine)):
                engine = engine_type(attempt_dir=root / label,
                    catalogue_dir=root / "catalogue", max_frames=2,
                    expected_sources=frozen_sources, allow_replay_clock=True,
                    source_attempt_name="independent_depth_engine_parity")
                results[label] = []
                try:
                    for index, packed in enumerate(requests):
                        values, frame_radar = snapshot.unpack_snapshot(packed)
                        result = engine.process(directory=root / (label + "_edge"),
                            kwargs=values, radar=frame_radar, sent_raw_ns=W.raw(),
                            sync_config=F.sync_config(root / label),
                            replay_observed_raw_ns=123 + index)
                        self.assertEqual(result["status"], "OK", result.get("error"))
                        self.assertEqual(result["processed"], index + 1)
                        self.assertEqual(result["sync"]["accepted"], index + 1)
                        self.assertEqual(len(result["rows"]), 5)
                        self.assertTrue(all(row["completed"] for row in result["rows"]))
                        self.assertEqual(snapshot.pack_snapshot(values, frame_radar), packed,
                                         "engine mutated owned request inputs")
                        if label == "candidate":
                            require_depth_stats(result["depth_reuse"], values["identity"]["frame_id"])
                            self.assertFalse(engine.depth.active)
                            self.assertFalse(engine.depth.retained)
                        bundle = F.T.verify_full_inputs(root / (label + "_edge"),
                            expected_identity=values["identity"])
                        path = Path(result["record"]["path"])
                        self.assertEqual(F.sha(path), result["record"]["sha256"])
                        avo = json.loads(path.read_bytes())
                        prescore = avo["prescore_input"]
                        self.assertEqual(F.sha(path.parent / prescore["filename"]), prescore["sha256"])
                        results[label].append(dict(bundle=bundle, avo=without_avo_clocks(avo),
                            sources=F.tree(Path(result["record"]["complete_sources"]["directory"]))))
                finally:
                    closed = engine.close()
                self.assertEqual(closed["records"], 10)
                self.assertEqual(bindings(), original_bindings)
                self.assertEqual(W.sources(), frozen_sources)
                self.assertEqual(C.sources(), candidate_sources)
            for index, (expected, actual) in enumerate(zip(results["baseline"], results["candidate"])):
                with self.subTest(frame_index=index):
                    a, b = expected["bundle"], actual["bundle"]
                    self.assertEqual(a.manifest["payloads"], b.manifest["payloads"])
                    self.assertEqual(a.prepared.content_sha256, b.prepared.content_sha256)
                    self.assertEqual(set(a.prepared.target), set(b.prepared.target))
                    for key, target_value in a.prepared.target.items():
                        if isinstance(target_value, np.ndarray):
                            self.assertEqual(target_value.dtype, b.prepared.target[key].dtype)
                            np.testing.assert_array_equal(target_value, b.prepared.target[key])
                        else:
                            self.assertEqual(target_value, b.prepared.target[key], key)
                    self.assertEqual(expected["sources"], actual["sources"])
                    self.assertEqual(expected["avo"], actual["avo"])

    def test_original_source_failure_clears_depth_and_restores_bindings(self):
        with F.fixture() as (root, kwargs, radar):
            original_bindings = bindings()
            kwargs["frozen_world"].population_evidence["registry_complete"] = False
            engine = Engine(attempt_dir=root / "failed", catalogue_dir=root / "catalogue",
                            max_frames=2, expected_sources=W.sources())
            try:
                result = engine.process(directory=root / "edge", kwargs=kwargs, radar=radar,
                    sent_raw_ns=W.raw(), sync_config=F.sync_config(root / "failed"))
                self.assertEqual(result["status"], "FAILED")
                self.assertTrue(engine.failed)
                self.assertEqual(result["processed"], 0)
                self.assertEqual([row["completed"] for row in result["rows"]], [True, False])
                self.assertEqual(result["depth_reuse"]["decodes"], 1)
                self.assertEqual(result["depth_reuse"]["reuses"], 0)
                self.assertFalse(result["depth_reuse"]["completed"])
                self.assertFalse(engine.depth.active)
                self.assertFalse(engine.depth.retained)
                self.assertIsNotNone(result["error"]["avo_evidence_record"])
            finally:
                closed = engine.close()
            self.assertEqual(closed["records"], 2)
            self.assertEqual(bindings(), original_bindings)


class ChildIntegrationTests(unittest.TestCase):
    def test_real_retained_hook_ready_run_close_caps_depth_stats_and_parent_isolation(self):
        environment = dict(os.environ, **C.ENVIRONMENT, PYTHONDONTWRITEBYTECODE="1")
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run([sys.executable, "-m", __package__ + ".test_engine", "--hook-probe"],
            cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True,
            text=True, timeout=90.)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        proof = json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                                if line.startswith("V6_DEPTH_HOOK_PROOF=")))
        self.assertTrue(proof["parent_unchanged"])
        self.assertTrue(proof["parity_verified"])
        self.assertTrue(proof["same_worker_two_requests"])
        self.assertEqual(proof["ordinary_before"], proof["ordinary_after"])
        self.assertEqual(proof["ready"]["environment"], C.ENVIRONMENT)
        self.assertEqual(proof["ready"]["torch_threads"], 1)
        self.assertEqual([row["command"] for row in proof["v6_responses"]],
                         ["INIT", "RUN", "RUN", "CLOSE"])
        self.assertEqual([row["frame_id"] for row in proof["v6_responses"] if "frame_id" in row], [100, 101])

    def test_bad_ready_caps_binding_or_pid_kills_joins_and_closes_descriptors(self):
        v6_copy(V5.BootstrapTests.test_rejected_ready_caps_or_binding_kills_joins_and_closes_descriptors)(self)


if __name__ == "__main__":
    if sys.argv[1:] == ["--hook-probe"]:
        print("V6_DEPTH_HOOK_PROOF=" + json.dumps(hook_probe(), sort_keys=True))
    else:
        unittest.main()
