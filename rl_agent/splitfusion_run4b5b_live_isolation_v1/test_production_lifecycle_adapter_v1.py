"""CPU-only tests for the B production lifecycle adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_validation_runner_v1 as R
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import live_adapters_v1 as L
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import production_lifecycle_adapter_v1 as P


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class FakeBackend:
    def __init__(self, fail_start=False):
        self.events, self.fail_start = [], fail_start

    def preflight(self, plan): self.events.append("preflight")

    def start(self, plan):
        self.events.append("start")
        if self.fail_start:
            raise RuntimeError("synthetic start failure")

    def execute(self, plan):
        self.events.append("execute")
        return {
            "schema": P.UE_RESULT_SCHEMA, "run_id": self.config.run_id,
            "variant": self.config.variant.value, "transmitted_frames": 300,
            "terminal_status": "COMPLETE", "result_sha256": sha("result"),
            "config_binding_sha256": self.config.binding_sha256(),
            "actor_boundary_sha256": self.manifest.actor_boundary_sha256,
        }

    def stop(self, plan): self.events.append("stop")


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.root = base / "local-repository"
        self.root.mkdir()
        self.settings = P.ProductionLifecycleSettingsV1(
            local_repository=self.root,
            remote_repository=base / "remote-repository",
            remote_attempt_base=base / "remote-attempts")
        order = L.RUN4B_FEATURE_ORDER
        self.manifest = L.BActorManifestV1(
            variant=L.ActorVariant.RUN4B, feature_order=order,
            feature_count=len(order),
            feature_schema_sha256=L.feature_schema_sha256(
                L.ActorVariant.RUN4B, order),
            actor_boundary_sha256=sha("actor"),
            weights_file_sha256=sha("weights"), selected_seed=43,
            selected_update=10_000)
        manifest_path = base / "actor.json"
        manifest_path.write_text(json.dumps(self.manifest.as_dict()))
        self.config = R.BValidationConfigV1.from_mapping({
            "schema": R.CONFIG_SCHEMA, "run_id": "run4b_production_300",
            "variant": L.ActorVariant.RUN4B.value,
            "actor_manifest_path": str(manifest_path),
            "actor_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()).hexdigest(),
            "output_root": str(base / "output"),
            "evidence_root": str(base / "evidence"),
            "transmitted_budget": 300,
            "split_host": {
                "carla_host": P.LOCAL_HOST, "ue_host": P.LOCAL_HOST,
                "cn_host": P.REMOTE_HOST, "edge_host": P.REMOTE_HOST,
                "ext_dn_host": P.REMOTE_HOST,
                "ack_receiver_host": P.LOCAL_HOST,
                "ack_receiver_port": 41070},
            "runner_semantics": R.RUNNER_SEMANTICS,
            "ack_semantics": R.ACK_SEMANTICS,
            "postrun_semantics": R.POSTRUN_SEMANTICS,
            "clock_domain": R.CLOCK_DOMAIN, "deadline_ns": R.DEADLINE_NS})

    def tearDown(self): self.temp.cleanup()

    def materialize_sources(self):
        for module in (P.UE_MODULE, P.EDGE_MODULE, *P.LOCAL_AUTHORITIES):
            path = self.root / P._module_relative_path(module)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# test-only\n")

    def test_missing_entrypoints_fail_before_backend(self):
        backend = FakeBackend()
        lifecycle = P.ProductionSplitHostLifecycleV1(
            self.settings, backend=backend)
        self.assertEqual(P.missing_process_entrypoints(self.settings),
                         (P.UE_MODULE, P.EDGE_MODULE))
        with self.assertRaisesRegex(P.MissingBEntrypointError,
                                    "b_ue_process_v1.*b_edge_process_v1"):
            lifecycle.preflight(self.config, self.manifest)
        self.assertEqual(backend.events, [])

    def test_commands_are_explicit_and_exclude_old_runtime(self):
        plan = P.build_plan(self.config, self.manifest, self.settings)
        rendered = json.dumps(plan.as_dict(), sort_keys=True).lower()
        self.assertIn(P.UE_MODULE.lower(), rendered)
        self.assertIn(P.EDGE_MODULE.lower(), rendered)
        self.assertIn(P.REMOTE_SSH.lower(), rendered)
        for token in P.FORBIDDEN_RUNTIME_TOKENS:
            self.assertNotIn(token, rendered)

    def test_protocol_run_and_cleanup(self):
        self.materialize_sources()
        backend = FakeBackend()
        backend.config, backend.manifest = self.config, self.manifest
        lifecycle = P.ProductionSplitHostLifecycleV1(
            self.settings, backend=backend)
        result = R.run_with_lifecycle(self.config, lifecycle)
        self.assertEqual((result.transmitted_frames, result.terminal_status),
                         (300, "COMPLETE"))
        self.assertEqual(backend.events,
                         ["preflight", "start", "execute", "stop"])

    def test_partial_start_failure_attempts_cleanup(self):
        self.materialize_sources()
        backend = FakeBackend(fail_start=True)
        lifecycle = P.ProductionSplitHostLifecycleV1(
            self.settings, backend=backend)
        with self.assertRaisesRegex(RuntimeError, "synthetic start"):
            R.run_with_lifecycle(self.config, lifecycle)
        self.assertEqual(backend.events, ["preflight", "start", "stop"])


if __name__ == "__main__":
    unittest.main()
