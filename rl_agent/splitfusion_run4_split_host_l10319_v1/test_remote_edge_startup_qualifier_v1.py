"""Offline tests for the bounded L10319 edge startup qualifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from . import contract as C
from . import gt_sender_integration as SENDER
from . import remote_edge_gt_entry_v1 as RGT
from . import remote_edge_lifecycle_v1 as E
from . import remote_edge_startup_qualifier_v1 as Q


SOURCE_ROOT = Path(__file__).resolve().parents[2]


class RepositoryFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repository = root / "repository"
        self.outputs = root / "outputs"
        self.attempt = self.outputs / "startup-attempt"
        self.repository.mkdir()
        self.outputs.mkdir()
        for relative in (Q.BINDING_RELATIVE, Q.CAMPAIGN_RELATIVE):
            destination = self.repository / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((SOURCE_ROOT / relative).read_bytes())
        for artifact in C.ARTIFACTS:
            path = self.repository / artifact.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((artifact.name + "\n").encode())

    def hash_file(self, path: Path) -> str:
        path = Path(path)
        if path.name == Path(Q.CAMPAIGN_RELATIVE).name:
            return Q.CAMPAIGN_SHA256
        for artifact in C.ARTIFACTS:
            if (path == self.repository / artifact.relative_path
                    or path.name == Path(artifact.relative_path).name):
                return artifact.sha256
        return Q._sha256_file(path)

    def prepare(self) -> Q.PreparedStartupAttempt:
        return Q.prepare_attempt(
            repository_root=self.repository, attempt_root=self.attempt,
            attempt_id="startup-smoke-001", run_id="run4-live",
            cell_id="a71__favorable_stable", hash_file=self.hash_file)


def valid_image(binding: C.RemoteRuntimeBinding):
    return {
        "tag": binding.image_tag, "image_id": binding.remote_image_id,
        "manifest_digest": binding.image_manifest_digest,
        "config_digest": binding.image_config_digest,
        "canonical_inspect_fields_sha256": binding.canonical_inspect_fields_sha256,
    }


def gt_ready(prepared: Q.PreparedStartupAttempt) -> dict:
    plan, topology = prepared.plan, C.default_topology()
    return {
        "schema": RGT.SCHEMA, "status": "LISTENING",
        "run_id": plan.invocation.run_id, "cell_id": plan.invocation.cell_id,
        "bind_host": topology.edge_ip,
        "advertised_endpoint": f"{topology.edge_ip}:{E.GT_PORT}",
        "max_tickets": RGT.MAX_GT_TICKETS,
        "socket_timeout_s": E.GT_SOCKET_TIMEOUT_S,
        "expectation_timeout_s": E.GT_EXPECTATION_TIMEOUT_S,
        "cross_host_clock_subtraction": False,
        "policy_deadline_clock_owner": "W10275",
    }


def edge_ready(prepared: Q.PreparedStartupAttempt) -> dict:
    plan, invocation = prepared.plan, prepared.plan.invocation
    topology = C.default_topology()
    return {
        "schema": E.READY_SCHEMA, "architecture": E.READY_ARCHITECTURE,
        "run4_edge": True, "action_id": invocation.action_id,
        "tail_device": "cuda:0", "direct_map_host": topology.map_ip,
        "direct_map_port": topology.map_port,
        "ue_control_host": invocation.ue_control_host,
        "ue_control_port": invocation.ue_control_port,
        "quality_spec_sha256": E.QUALITY_SPEC_SHA256,
        "dense_label_map_on_radio": False, "object_records_on_radio": False,
        "evaluation_evidence_dir": E.EVIDENCE_DESTINATION,
    }


def gt_final(prepared: Q.PreparedStartupAttempt) -> dict:
    plan = prepared.plan
    return {
        "schema": RGT.SCHEMA, "status": "STOPPED",
        "run_id": plan.invocation.run_id, "cell_id": plan.invocation.cell_id,
        "advertised_endpoint": f"{C.default_topology().edge_ip}:{E.GT_PORT}",
        "edge_ready_health_checked": True, "failure": None,
        "counters": {"connections": 0, "authorized": 0},
        "thread_alive": False, "cross_host_clock_subtraction": False,
        "policy_deadline_clock_owner": "W10275",
    }


class FakeRuntime:
    def __init__(self, prepared: Q.PreparedStartupAttempt, *, publish_ready=True,
                 die_before_ready=False, fail_launch=False, fail_logs=False) -> None:
        self.prepared = prepared
        self.publish_ready = publish_ready
        self.die_before_ready = die_before_ready
        self.fail_launch = fail_launch
        self.fail_logs = fail_logs
        self.launched = False
        self.calls: list[tuple[str, ...]] = []

    def container_json(self) -> str:
        service = self.prepared.plan.compose_document["services"][E.SERVICE]
        labels = dict(service["labels"])
        labels["com.docker.compose.project"] = self.prepared.plan.invocation.project_name
        mounts = [{"Destination": row["target"], "Source": row["source"],
                   "RW": not row["read_only"]} for row in service["volumes"]]
        return json.dumps([{
            "Id": "container-123", "Image": C.REMOTE_CONTAINER_IMAGE_ID,
            "Config": {"Labels": labels}, "Mounts": mounts,
        }])

    def write_ready(self) -> None:
        state = self.prepared.plan.paths.state_root
        (state / Path(E.GT_READY_DESTINATION).name).write_text(
            json.dumps(gt_ready(self.prepared)), encoding="utf-8")
        (state / Path(E.READY_DESTINATION).name).write_text(
            json.dumps(edge_ready(self.prepared)), encoding="utf-8")

    def __call__(self, argv, _timeout_s) -> Q.CommandResult:
        words = tuple(argv)
        self.calls.append(words)
        if words == ("hostname", "-s"):
            return Q.CommandResult(0, "L10319\n")
        if words and words[0] == "nvidia-smi":
            gpu = self.prepared.binding.gpu
            return Q.CommandResult(
                0, f"{gpu.model}, {gpu.uuid}, {gpu.memory_total_mib}, "
                   f"{gpu.driver_version}\n")
        if words[:6] == ("sudo", "-n", "docker", "image", "inspect",
                         self.prepared.binding.image_tag):
            return Q.CommandResult(0, "[]")
        if words[:5] == ("sudo", "-n", "docker", "network", "inspect"):
            return Q.CommandResult(0, "[{}]")
        if words[:5] == ("sudo", "-n", "docker", "container", "inspect"):
            return (Q.CommandResult(0, self.container_json()) if self.launched
                    else Q.CommandResult(1, "", "No such container"))
        if words[:5] == ("sudo", "-n", "docker", "inspect", "-f"):
            alive = self.launched and not self.die_before_ready
            return Q.CommandResult(0, "true\n" if alive else "false\n")
        if "compose" in words and "config" in words:
            return Q.CommandResult(0)
        if "compose" in words and "up" in words:
            self.launched = True
            if self.publish_ready:
                self.write_ready()
            return Q.CommandResult(2 if self.fail_launch else 0)
        if words[:4] == ("sudo", "-n", "docker", "logs"):
            if self.fail_logs:
                return Q.CommandResult(3, "", "docker log lookup failed\n")
            return Q.CommandResult(0, "edge bounded startup log\n", "edge stderr\n")
        if "compose" in words and "down" in words:
            final_path = (self.prepared.plan.paths.state_root
                          / Path(E.GT_FINAL_DESTINATION).name)
            if self.publish_ready and not final_path.exists():
                final_path.write_text(json.dumps(gt_final(self.prepared)),
                                      encoding="utf-8")
            self.launched = False
            return Q.CommandResult(0)
        raise AssertionError(f"unexpected command: {words}")


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = RepositoryFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_materializes_create_only_config_checkpoint_and_compose(self) -> None:
        prepared = self.fixture.prepare()
        paths = prepared.plan.paths
        config = json.loads((paths.state_root / E.EDGE_CONFIG_DESTINATION.split("/")[-1])
                            .read_text())
        self.assertEqual(config["schema"], Q.EDGE_CONFIG_SCHEMA)
        self.assertEqual(config["evidence_dir"], E.EVIDENCE_DESTINATION)
        checkpoint = paths.state_root / "hub" / "checkpoints" / paths.fcos_weight_path.name
        self.assertTrue(checkpoint.is_file())
        compose = json.loads(paths.compose_path.read_text())
        self.assertEqual(tuple(compose["services"]), (E.SERVICE,))
        self.assertNotIn("build", compose["services"][E.SERVICE])
        plan = json.loads((paths.attempt_root / Q.PLAN_NAME).read_text())
        self.assertFalse(plan["full_live_run_authorized"])
        self.assertEqual(plan["commands"]["pre_teardown_log_capture"], [
            "sudo", "-n", "docker", "logs", "--timestamps",
            E.CONTAINER,
        ])
        rendered = json.dumps(plan["commands"])
        self.assertNotIn("CARLA", rendered)
        self.assertNotIn("nr-softmodem", rendered)
        with self.assertRaises(FileExistsError):
            self.fixture.prepare()

    def test_registered_campaign_has_the_actual_frozen_sha(self) -> None:
        self.assertEqual(Q._sha256_file(SOURCE_ROOT / Q.CAMPAIGN_RELATIVE),
                         Q.CAMPAIGN_SHA256)

    def test_sender_endpoint_and_bounds_match_listener(self) -> None:
        self.assertEqual((SENDER.REGISTERED_HOST, SENDER.REGISTERED_PORT),
                         (C.default_topology().edge_ip, RGT.REGISTERED_GT_PORT))
        self.assertEqual(SENDER.MAX_PENDING_IDENTITIES, RGT.MAX_GT_TICKETS)
        self.assertLessEqual(SENDER.MAX_COMPONENT_WAIT_S,
                             E.GT_EXPECTATION_TIMEOUT_S)


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = RepositoryFixture(Path(self.temporary.name))
        self.prepared = self.fixture.prepare()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def image_observer(_stdout, binding):
        return valid_image(binding)

    def test_passes_both_ready_gates_then_project_scoped_teardown(self) -> None:
        runtime = FakeRuntime(self.prepared)
        report = Q.qualify_startup(
            self.prepared, runner=runtime, timeout_s=1.0,
            image_observer=self.image_observer)
        self.assertEqual(report["status"],
                         "PASS_STARTUP_READY_AND_PROJECT_TEARDOWN_COMPLETE")
        self.assertEqual(report["gt_endpoint"], "192.168.70.140:51015")
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertTrue(report["cleanup"]["gt_listener_stopped"])
        self.assertEqual(len(report["cleanup"]["gt_final_sha256"]), 64)
        self.assertEqual(report["edge_log_capture"]["status"], "CAPTURED")
        log_path = self.prepared.plan.paths.attempt_root / Q.LOG_NAME
        log_bytes = log_path.read_bytes()
        self.assertIn(b"edge bounded startup log", log_bytes)
        self.assertIn(b"edge stderr", log_bytes)
        self.assertEqual(report["edge_log_capture"]["sha256"],
                         Q._sha256_file(log_path))
        log_index = next(i for i, call in enumerate(runtime.calls)
                         if call[:4] == ("sudo", "-n", "docker", "logs"))
        self.assertFalse(runtime.launched)
        launch = next(call for call in runtime.calls if "up" in call)
        self.assertIn("--no-build", launch)
        self.assertIn("never", launch)
        self.assertIn("--no-deps", launch)
        down = next(call for call in runtime.calls if "down" in call)
        self.assertIn(self.prepared.plan.invocation.project_name, down)
        down_index = runtime.calls.index(down)
        self.assertLess(log_index, down_index)
        rendered = "\n".join(" ".join(call) for call in runtime.calls)
        self.assertNotIn("phase6_live_runner", rendered)
        self.assertNotIn("nr-softmodem", rendered)
        self.assertNotIn("Carla", rendered)
        persisted = json.loads(self.prepared.result_path.read_text())
        self.assertEqual(persisted["status"], report["status"])

    def test_dead_edge_before_ready_fails_and_still_tears_down(self) -> None:
        runtime = FakeRuntime(self.prepared, publish_ready=False,
                              die_before_ready=True)
        report = Q.qualify_startup(
            self.prepared, runner=runtime, timeout_s=1.0,
            image_observer=self.image_observer)
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("edge exited before READY", report["error"])
        self.assertEqual(report["edge_log_capture"]["status"], "CAPTURED")
        self.assertIn(b"edge stderr", (
            self.prepared.plan.paths.attempt_root / Q.LOG_NAME).read_bytes())
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertFalse(runtime.launched)
        self.assertTrue(any("down" in call for call in runtime.calls))
        log_index = next(i for i, call in enumerate(runtime.calls)
                         if call[:4] == ("sudo", "-n", "docker", "logs"))
        down_index = next(i for i, call in enumerate(runtime.calls) if "down" in call)
        self.assertLess(log_index, down_index)

    def test_preflight_refusal_never_launches_or_tears_foreign_resources(self) -> None:
        runtime = FakeRuntime(self.prepared)
        original = runtime.__call__

        def fail_hostname(argv, timeout_s):
            if tuple(argv) == ("hostname", "-s"):
                runtime.calls.append(tuple(argv))
                return Q.CommandResult(0, "FOREIGN\n")
            return original(argv, timeout_s)

        report = Q.qualify_startup(
            self.prepared, runner=fail_hostname, timeout_s=1.0,
            image_observer=self.image_observer)
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(report["cleanup"]["not_launched"])
        self.assertFalse(any("up" in call or "down" in call for call in runtime.calls))
        self.assertEqual(report["edge_log_capture"]["status"],
                         "NOT_REQUIRED_NOT_LAUNCHED")
        self.assertFalse((self.prepared.plan.paths.attempt_root / Q.LOG_NAME).exists())
        self.assertFalse(any(
            call[:4] == ("sudo", "-n", "docker", "logs")
            for call in runtime.calls))

    def test_partial_compose_up_failure_still_tears_down_attempt_project(self) -> None:
        runtime = FakeRuntime(self.prepared, fail_launch=True)
        report = Q.qualify_startup(
            self.prepared, runner=runtime, timeout_s=1.0,
            image_observer=self.image_observer)
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("command failed", report["error"])
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertFalse(runtime.launched)
        self.assertTrue(any("down" in call for call in runtime.calls))
        self.assertEqual(report["edge_log_capture"]["status"], "CAPTURED")
        log_index = next(i for i, call in enumerate(runtime.calls)
                         if call[:4] == ("sudo", "-n", "docker", "logs"))
        down_index = next(i for i, call in enumerate(runtime.calls) if "down" in call)
        self.assertLess(log_index, down_index)

    def test_log_capture_failure_does_not_prevent_teardown(self) -> None:
        runtime = FakeRuntime(self.prepared, fail_logs=True)
        report = Q.qualify_startup(
            self.prepared, runner=runtime, timeout_s=1.0,
            image_observer=self.image_observer)
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("timestamped edge log capture failed", report["error"])
        self.assertEqual(report["edge_log_capture"]["status"], "FAILED")
        self.assertEqual(report["edge_log_capture"]["returncode"], 3)
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertFalse(runtime.launched)
        log_bytes = (self.prepared.plan.paths.attempt_root / Q.LOG_NAME).read_bytes()
        self.assertIn(b"docker log lookup failed", log_bytes)
        log_index = next(i for i, call in enumerate(runtime.calls)
                         if call[:4] == ("sudo", "-n", "docker", "logs"))
        down_index = next(i for i, call in enumerate(runtime.calls) if "down" in call)
        self.assertLess(log_index, down_index)

    def test_existing_log_target_fails_closed_but_teardown_still_runs(self) -> None:
        target = self.prepared.plan.paths.attempt_root / Q.LOG_NAME
        original = b"user-owned prior evidence\n"
        target.write_bytes(original)
        runtime = FakeRuntime(self.prepared)
        report = Q.qualify_startup(
            self.prepared, runner=runtime, timeout_s=1.0,
            image_observer=self.image_observer)
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["edge_log_capture"]["status"], "FAILED")
        self.assertIn("File exists", report["edge_log_capture"]["error"])
        self.assertEqual(target.read_bytes(), original)
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertFalse(runtime.launched)


class InspectParserTests(unittest.TestCase):
    def test_host_only_import_does_not_require_or_import_torch(self) -> None:
        script = r'''
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("host-only startup qualifier imported torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import rl_agent.splitfusion_run4_split_host_l10319_v1.remote_edge_startup_qualifier_v1
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
'''
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=SOURCE_ROOT, env=environment,
            text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_container_parser_keeps_project_labels_mount_modes_and_image(self) -> None:
        raw = json.dumps([{
            "Id": "container", "Image": C.REMOTE_CONTAINER_IMAGE_ID,
            "Config": {"Labels": {"com.docker.compose.project": "project",
                                    "bound": "yes"}},
            "Mounts": [{"Destination": "/target", "Source": "/source", "RW": True}],
        }])
        observed = Q.container_observation_from_inspect(raw)
        self.assertEqual(observed["project"], "project")
        self.assertEqual(observed["mounts"], {"/target": ("/source", True)})
        self.assertEqual(observed["image_id"], C.REMOTE_CONTAINER_IMAGE_ID)

    def test_malformed_inspect_is_refused(self) -> None:
        for value in ("not-json", "[]", "[{},{}]"):
            with self.subTest(value=value):
                with self.assertRaises(Q.RemoteEdgeStartupError):
                    Q.container_observation_from_inspect(value)


if __name__ == "__main__":
    unittest.main()
