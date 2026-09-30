"""Offline tests for the L10319 edge-only lifecycle contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from . import contract as C
from . import remote_edge_gt_entry_v1 as RGT
from . import remote_edge_lifecycle_v1 as E


def measured_binding() -> C.RemoteRuntimeBinding:
    return C.RemoteRuntimeBinding.from_mapping({
        "schema": "scenesense.run4.remote_runtime_binding.v1",
        "hostname": "L10319",
        "host_ipv4": "10.21.16.162",
        "gpu": {
            "model": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "uuid": "GPU-b8c4646c-abb0-5d63-679b-49622ce057b6",
            "memory_total_mib": 24463,
            "driver_version": "610.43.02",
        },
        "image_tag": C.EDGE_IMAGE_TAG,
        "image_manifest_digest": C.EDGE_IMAGE_MANIFEST_DIGEST,
        "image_config_digest": C.EDGE_IMAGE_CONFIG_DIGEST,
        "remote_image_id": C.REMOTE_IMAGE_ID,
        "remote_container_image_id": C.REMOTE_CONTAINER_IMAGE_ID,
        "canonical_inspect_fields_sha256": C.EDGE_IMAGE_CANONICAL_INSPECT_SHA256,
        "artifacts": [
            {"name": value.name, "relative_path": value.relative_path,
             "sha256": value.sha256}
            for value in C.ARTIFACTS
        ],
    })


class Fixture:
    def __init__(self, temporary: str) -> None:
        base = Path(temporary).resolve()
        self.repo = base / "current_worktree"
        self.attempt = base / "attempt"
        fcos = next(value for value in C.ARTIFACTS
                    if value.name == "torchvision_fcos")
        self.paths = E.RemoteEdgePaths(
            repository_root=self.repo,
            attempt_root=self.attempt,
            state_root=self.attempt / "state",
            evidence_root=self.attempt / "evidence",
            compose_path=self.attempt / "remote_edge.compose.json",
            fcos_weight_path=self.repo / fcos.relative_path,
            campaign_config_relative="rl_agent/configs/campaign.json",
        )
        self.invocation = E.RemoteEdgeInvocation(
            attempt_id="split-host-smoke-001",
            run_id="run4-split-host-smoke",
            cell_id="a71__favorable_stable",
            action_id=71,
            allowed_action_ids=(71,),
            edge_receive_port=51002,
            ue_control_host="10.0.0.2",
            ue_control_port=51014,
            edge_compute_cpus="4-7",
            edge_receive_cpus="8",
        )


class PlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.fixture = Fixture(self.temp.name)
        self.plan = E.build_plan(
            binding=measured_binding(), paths=self.fixture.paths,
            invocation=self.fixture.invocation,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_plan_is_edge_only_and_never_authorizes_live_run(self) -> None:
        self.assertEqual(self.plan.purpose, E.PURPOSE)
        self.assertFalse(self.plan.live_run_authorized)
        commands = (*self.plan.preflight, self.plan.launch,
                    *self.plan.post_create, self.plan.teardown)
        rendered = "\n".join(" ".join(value.argv) for value in commands)
        self.assertNotIn("CARLA", rendered)
        self.assertNotIn("nr-softmodem", rendered)
        self.assertNotIn("cn_start", rendered)
        self.assertNotIn("phase6_live_runner", rendered)
        with self.assertRaises(E.RemoteEdgeLifecycleError):
            E.authorize_full_live_run()

    def test_compose_uses_current_worktree_and_per_attempt_mounts(self) -> None:
        service = self.plan.compose_document["services"][E.SERVICE]
        self.assertNotIn("build", service)
        self.assertEqual(service["pull_policy"], "never")
        self.assertEqual(service["networks"]["public_net"]["ipv4_address"],
                         "192.168.70.140")
        mounts = {row["target"]: (row["source"], row["read_only"])
                  for row in service["volumes"]}
        self.assertEqual(mounts[E.REPOSITORY_DESTINATION],
                         (str(self.fixture.repo), True))
        self.assertEqual(mounts[E.STATE_DESTINATION],
                         (str(self.fixture.attempt / "state"), False))
        self.assertEqual(mounts[E.EVIDENCE_DESTINATION],
                         (str(self.fixture.attempt / "evidence"), False))
        self.assertNotIn("../../abiodun", json.dumps(self.plan.compose_document))
        self.assertEqual(service["labels"]["scenesense.gpu_uuid"],
                         measured_binding().gpu.uuid)

    def test_edge_command_has_remote_map_and_no_shell(self) -> None:
        command = self.plan.compose_document["services"][E.SERVICE]["command"]
        self.assertEqual(command[:4], ["python3", "-u", "-m", E.EDGE_MODULE])
        self.assertEqual(command[command.index("--direct-map-host") + 1],
                         "10.21.16.222")
        self.assertEqual(command[command.index("--direct-map-port") + 1], "39320")
        self.assertNotIn("bash", command)
        self.assertNotIn("-lc", command)

    def test_launch_is_no_build_no_pull_no_dependencies_and_one_service(self) -> None:
        argv = self.plan.launch.argv
        self.assertIn("--no-build", argv)
        self.assertEqual(argv[argv.index("--pull") + 1], "never")
        self.assertIn("--no-deps", argv)
        self.assertEqual(argv[-1], E.SERVICE)
        self.assertEqual(tuple(self.plan.compose_document["services"]), (E.SERVICE,))

    def test_measured_gpu_has_no_model_or_vram_default(self) -> None:
        evidence = self.plan.as_evidence()
        self.assertEqual(evidence["remote_gpu"], {
            "model": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "uuid": "GPU-b8c4646c-abb0-5d63-679b-49622ce057b6",
            "memory_total_mib": 24463,
            "driver_version": "610.43.02",
        })
        self.assertNotIn("gpu_default", evidence)

    def test_teardown_is_project_scoped(self) -> None:
        argv = self.plan.teardown.argv
        self.assertIn(self.fixture.invocation.project_name, argv)
        self.assertEqual(argv[-4:], ("down", "--remove-orphans", "--timeout", "30"))
        self.assertNotIn("docker stop", " ".join(argv))

    def test_lifecycle_identity_rule_matches_gt_transport_safety(self) -> None:
        with self.assertRaises(E.RemoteEdgeLifecycleError):
            replace(self.fixture.invocation,
                    run_id="run4/split-host/smoke").validate()
        accepted = replace(
            self.fixture.invocation, run_id="run4+split@v1:smoke").validate()
        self.assertEqual(accepted.run_id, "run4+split@v1:smoke")
        self.assertEqual(E.IDENTITY_RE.pattern, RGT.GT.SAFE_ID_RE.pattern)



class ObservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.fixture = Fixture(self.temp.name)
        self.plan = E.build_plan(binding=measured_binding(), paths=self.fixture.paths,
                                 invocation=self.fixture.invocation)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_image_identity_binds_manifest_config_and_canonical_inspect(self) -> None:
        observation = {
            "tag": C.EDGE_IMAGE_TAG,
            "image_id": C.REMOTE_IMAGE_ID,
            "manifest_digest": C.EDGE_IMAGE_MANIFEST_DIGEST,
            "config_digest": C.EDGE_IMAGE_CONFIG_DIGEST,
            "canonical_inspect_fields_sha256": C.EDGE_IMAGE_CANONICAL_INSPECT_SHA256,
        }
        E.validate_remote_image_observation(observation)
        for field in ("image_id", "manifest_digest", "config_digest",
                      "canonical_inspect_fields_sha256"):
            with self.subTest(field=field):
                bad = dict(observation)
                bad[field] = "0" * 64
                with self.assertRaises(E.RemoteEdgeLifecycleError):
                    E.validate_remote_image_observation(bad)

    def test_selected_inspect_fixture_reproduces_digest_and_drift_refuses(self) -> None:
        fixture = Path(__file__).with_name(
            "PORTABLE_IMAGE_INSPECT_SELECTED_FIELDS_V1.json")
        document = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(set(document), {
            "Architecture", "Created", "Config", "RootFS", "History", "Os",
            "Variant",
        })
        digest = E.canonical_image_inspect_sha256(document)
        self.assertEqual(digest, C.EDGE_IMAGE_CANONICAL_INSPECT_SHA256)

        drifted = json.loads(json.dumps(document))
        drifted["Config"]["WorkingDir"] = "/foreign"
        drifted_digest = E.canonical_image_inspect_sha256(drifted)
        self.assertNotEqual(drifted_digest, digest)
        observation = {
            "tag": C.EDGE_IMAGE_TAG,
            "image_id": C.REMOTE_IMAGE_ID,
            "manifest_digest": C.EDGE_IMAGE_MANIFEST_DIGEST,
            "config_digest": C.EDGE_IMAGE_CONFIG_DIGEST,
            "canonical_inspect_fields_sha256": drifted_digest,
        }
        with self.assertRaises(E.RemoteEdgeLifecycleError):
            E.validate_remote_image_observation(observation)

    def test_container_identity_mounts_and_attempt_ownership(self) -> None:
        service = self.plan.compose_document["services"][E.SERVICE]
        mounts = {row["target"]: (row["source"], not row["read_only"])
                  for row in service["volumes"]}
        observation = {
            "container_id": "container-123",
            "image_id": C.REMOTE_CONTAINER_IMAGE_ID,
            "project": self.fixture.invocation.project_name,
            "labels": dict(service["labels"]),
            "mounts": mounts,
        }
        E.validate_container_observation(observation, plan=self.plan)
        bad = dict(observation, project="someone-elses-project")
        with self.assertRaises(E.RemoteEdgeLifecycleError):
            E.validate_container_observation(bad, plan=self.plan)

    def test_ready_record_binds_map_feedback_evidence_and_action(self) -> None:
        ready = {
            "schema": E.READY_SCHEMA,
            "architecture": E.READY_ARCHITECTURE,
            "run4_edge": True,
            "action_id": 71,
            "tail_device": "cuda:0",
            "direct_map_host": "10.21.16.222",
            "direct_map_port": 39320,
            "ue_control_host": "10.0.0.2",
            "ue_control_port": 51014,
            "quality_spec_sha256": E.QUALITY_SPEC_SHA256,
            "dense_label_map_on_radio": False,
            "object_records_on_radio": False,
            "evaluation_evidence_dir": E.EVIDENCE_DESTINATION,
        }
        E.validate_ready_record(ready, plan=self.plan)
        for field in ("direct_map_host", "ue_control_port", "action_id",
                      "quality_spec_sha256", "evaluation_evidence_dir"):
            with self.subTest(field=field):
                bad = dict(ready)
                bad[field] = "drift"
                with self.assertRaises(E.RemoteEdgeLifecycleError):
                    E.validate_ready_record(bad, plan=self.plan)

    def test_gt_final_record_proves_listener_stopped_for_this_attempt(self) -> None:
        final = {
            "schema": RGT.SCHEMA,
            "status": "STOPPED",
            "run_id": self.plan.invocation.run_id,
            "cell_id": self.plan.invocation.cell_id,
            "advertised_endpoint": "192.168.70.140:51015",
            "edge_ready_health_checked": True,
            "failure": None,
            "counters": {"connections": 0, "authorized": 0},
            "thread_alive": False,
            "cross_host_clock_subtraction": False,
            "policy_deadline_clock_owner": "W10275",
        }
        E.validate_gt_final_record(final, plan=self.plan)
        for field, value in (("status", "FAILED"),
                             ("run_id", "foreign-run"),
                             ("thread_alive", True),
                             ("cross_host_clock_subtraction", True)):
            with self.subTest(field=field):
                bad = dict(final)
                bad[field] = value
                with self.assertRaises(E.RemoteEdgeLifecycleError):
                    E.validate_gt_final_record(bad, plan=self.plan)

    def test_broad_or_foreign_commands_are_refused(self) -> None:
        for argv in (
            ("python3", "-m", "phase6_live_runner"),
            ("nr-softmodem", "-O", "config"),
            ("sudo", "docker", "compose", "up", "oai-perception-rx"),
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(E.RemoteEdgeLifecycleError):
                    E.assert_edge_only_command(argv)


if __name__ == "__main__":
    unittest.main()
