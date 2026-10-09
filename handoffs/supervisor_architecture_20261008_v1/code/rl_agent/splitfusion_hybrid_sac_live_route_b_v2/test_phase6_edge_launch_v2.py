"""Offline tests for setup-repair addendum 3 (no-build, image-bound edge launch).

No Docker command is executed: every docker interaction goes through a fake
runner. The legacy-equivalence test runs the unmodified shared launcher with
a fake ``sudo`` first on ``PATH`` that only records its arguments.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import phase6_edge_launch_v2 as EL
from . import phase6_live_child_nobuild_v2 as NB
from . import phase6_live_child_v2 as C
from . import phase6_live_runner_v2 as RUN

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "rl_agent/splitfusion_hybrid_sac_live_route_b_v2"
BASE_COMMIT = "4f2e68d13dcb9d8aec09a63e7617b722e5be0b9a"
OTHER_ID = "sha256:" + "1" * 64
FROZEN_PACKAGE = (
    # addendum 5 authorizes phase6_decision_engine_v2 / reward_hold_controller_v2 /
    # phase6_ue_runtime_v2 changes; test_phase6_live_path_repair_v2 bounds them.
    # phase6_live_child_v2.py: addendum-6 cycle-boundary stop (bounded elsewhere).
    "live_state_v2.py", "run4_live_wire_v2.py", "run4_map_protocol_v2.py",
    "run4_ue_ledger_v2.py", "continuous_execution_v2.py", "frozen_actor_v2.py",
    # phase6_edge_runtime_v2.py: addendum-4 ready-record repair; bounded by
    # test_phase6_ready_contract_v2.BoundedDiffTest instead.
    "phase6_map_server_v2.py",
    "phase6_result_reporting_v2.py", "phase6_prospective_addendum_2.json",
    "PHASE6_PROSPECTIVE_ADDENDUM_2.md", "live_qualification_300_v2.json",
    "PHASE6_RUNNER_REPORT.md", "ACTOR_BINDING_V2.json",
)
FROZEN_SHARED = (
    "scripts/receiver_container_fusion_back_up.sh", "scripts/config.env",
    "receiver_container/docker-compose.yaml", "receiver_container/docker-compose.fusion-back.yaml",
    "receiver_container/Dockerfile", "receiver_container/entrypoint.sh",
    "rl_agent/splitfusion_direct_edge_map_v1/adapter_direct_v1.py",
    "rl_agent/ue_route_b_split_cell_adapter_v1.py",
)


def _completed(rc=0, stdout=""):
    return subprocess.CompletedProcess([], rc, stdout=stdout, stderr="")


class FakeDocker:
    """Answers the docker calls the launcher makes; records every argv."""

    def __init__(self, *, image_id=EL.ADMITTED_IMAGE_ID, image_present=True,
                 container_image=EL.ADMITTED_IMAGE_ID, mounts=None, compose_rc=0,
                 runtimes="runc\nnvidia\n", network=True) -> None:
        self.calls: list[list[str]] = []
        self.image_id, self.image_present = image_id, image_present
        self.container_image, self.mounts = container_image, mounts
        self.compose_rc, self.runtimes, self.network = compose_rc, runtimes, network

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if argv[-2:] == ["--format", "{{json .}}"] and "image" in argv:
            if not self.image_present:
                return _completed(1)
            return _completed(0, json.dumps({
                "Id": self.image_id, "Created": "2026-09-05T20:55:11Z",
                "RepoTags": [EL.IMAGE_TAG], "RepoDigests": [], "Architecture": "amd64",
                "Os": "linux", "RootFS": {"Type": "layers", "Layers": ["sha256:" + "a" * 64]},
                "Config": {"Entrypoint": ["/entrypoint.sh"]}}))
        if argv[-2:] == ["--format", "{{json .}}"] and "container" in argv:
            return _completed(0, json.dumps({
                "Id": "c" * 64, "Image": self.container_image,
                "Config": {"Image": EL.IMAGE_TAG}, "State": {"Running": True},
                "Mounts": self.mounts}))
        if "network" in argv:
            return _completed(0 if self.network else 1)
        if "info" in argv:
            return _completed(0, self.runtimes)
        if "compose" in argv:
            return _completed(self.compose_rc)
        raise AssertionError(f"unexpected docker call {argv}")


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.state = base / "state"
        self.state.mkdir()
        self.fcos = base / "fcos_resnet50_fpn_coco-99b0c9b7.pth"
        self.fcos.write_bytes(b"fcos-weights")
        self.fcos_sha = hashlib.sha256(b"fcos-weights").hexdigest()
        self.evidence = base / "attempt" / "run4_phase6" / "edge_image_launch.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def mounts(self, *, repo_rw=False, state_rw=True, fcos_rw=False, state=None, fcos=None):
        return [
            {"Destination": EL.REPOSITORY_DESTINATION, "Source": str(ROOT), "RW": repo_rw},
            {"Destination": EL.STATE_DESTINATION, "Source": str(state or self.state),
             "RW": state_rw},
            {"Destination": EL.FCOS_DESTINATION, "Source": str(fcos or self.fcos),
             "RW": fcos_rw},
            {"Destination": "/tmp/.X11-unix", "Source": "/tmp/.X11-unix", "RW": True},
        ]

    def env(self):
        return {"FUSION_BACK_DUAL": "0", "FUSION_BACK_BIND_HOST": "0.0.0.0",
                "FUSION_BACK_REMOTE_HOST": "10.0.0.2", "FUSION_BACK_REMOTE_HOST_1": "10.0.0.2",
                "FUSION_BACK_DEVICE": "cuda", "SPLITFUSION_EDGE_STATE_ROOT": str(self.state),
                "FUSION_BACK_SCRIPT": "-m some.module",
                "FUSION_BACK_EXTRA_ARGS": "--edge --config /work/abiodun/x.json --action-id 71"}

    def launch(self, docker):
        return EL.launch_no_build(self.env(), stdout=io.BytesIO(), timeout=60.0,
                                  evidence_path=self.evidence, fcos_source=self.fcos,
                                  fcos_sha256=self.fcos_sha, run=docker,
                                  interface_exists_fn=lambda name: False,
                                  sleep=lambda s: None)


class CommandTest(Fixture):
    def test_command_is_no_build_pull_never(self) -> None:
        variables = EL.compose_variables(self.env(), config=EL.config_env(),
                                         interface_exists=lambda n: False)
        command = EL.up_command(variables)
        self.assertEqual(command[-6:], ["up", "-d", "--no-build", "--pull", "never",
                                        "--force-recreate"])
        self.assertIn("--no-build", command)
        self.assertEqual(command[command.index("--pull") + 1], "never")
        self.assertNotIn("--build", command)
        self.assertNotIn("pull", [c for c in command if "=" not in c])
        self.assertEqual(EL.forbidden_operations(command), [])
        self.assertEqual(command[:1], ["sudo"])
        self.assertEqual(command[command.index("compose") + 1:command.index("compose") + 5],
                         ["-f", "docker-compose.yaml", "-f", "docker-compose.fusion-back.yaml"])

    def test_forbidden_operations_detects_build_and_pull(self) -> None:
        self.assertIn("build", EL.forbidden_operations(
            ["docker", "compose", "up", "-d", "--build", "--force-recreate"]))
        self.assertIn("pull", EL.forbidden_operations(["docker", "pull", EL.IMAGE_TAG]))
        self.assertIn("pull-policy-not-never", EL.forbidden_operations(
            ["docker", "compose", "up", "--pull", "always"]))
        self.assertIn("retag/push/load", EL.forbidden_operations(
            ["docker", "tag", "a", "b"]))

    def test_variables_equal_legacy_launcher_exactly(self) -> None:
        """Run the unmodified shared launcher with a recording fake ``sudo``."""
        self.assertEqual(EL.sha256_file(EL.LEGACY_LAUNCHER), EL.LEGACY_LAUNCHER_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            bindir, record = Path(tmp) / "bin", Path(tmp) / "calls.jsonl"
            bindir.mkdir()
            fake = bindir / "sudo"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                f"open({str(record)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "if 'info' in sys.argv: print('runc'); print('nvidia')\n",
                encoding="utf-8")
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            env = {"PATH": f"{bindir}:/usr/bin:/bin:/usr/sbin", "HOME": os.environ.get("HOME", "/tmp"),
                   **self.env()}
            done = subprocess.run([str(EL.LEGACY_LAUNCHER)], env=env, cwd=str(ROOT),
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stdout[-500:] + done.stderr[-500:])
            calls = [json.loads(line) for line in record.read_text().splitlines()]
        legacy = [c for c in calls if "compose" in c]
        self.assertEqual(len(legacy), 1)
        legacy = legacy[0]
        ours = EL.up_command(EL.compose_variables(
            self.env(), config=EL.config_env(), interface_exists=EL.interface_exists))[1:]
        split = legacy.index("docker")
        self.assertEqual(legacy[:split], ours[:ours.index("docker")])      # same variables
        self.assertEqual(legacy[split:split + 6], ours[ours.index("docker"):][:6])
        self.assertEqual(legacy[split + 6:], ["up", "-d", "--build", "--force-recreate"])
        self.assertEqual(ours[-6:], list(EL.UP_ARGUMENTS))


class IdentityTest(Fixture):
    def test_missing_image_refused(self) -> None:
        with self.assertRaisesRegex(EL.EdgeImageError, "missing"):
            EL.resolve_admitted_image(FakeDocker(image_present=False))
        docker = FakeDocker(image_present=False)
        with self.assertRaises(EL.EdgeImageError):
            self.launch(docker)
        self.assertFalse(any("compose" in c for c in docker.calls))   # never started
        self.assertEqual(json.loads(self.evidence.read_text())["verdict"], "REFUSED")

    def test_tag_to_id_drift_refused(self) -> None:
        docker = FakeDocker(image_id=OTHER_ID)
        with self.assertRaisesRegex(EL.EdgeImageError, "not the admitted"):
            self.launch(docker)
        self.assertFalse(any("compose" in c for c in docker.calls))

    def test_post_create_container_image_drift_refused(self) -> None:
        docker = FakeDocker(container_image=OTHER_ID, mounts=self.mounts())
        with self.assertRaisesRegex(EL.EdgeImageError, "edge container runs"):
            self.launch(docker)
        evidence = json.loads(self.evidence.read_text())
        self.assertEqual(evidence["verdict"], "REFUSED")
        self.assertEqual(evidence["pre_launch_tag_resolution"]["id"], EL.ADMITTED_IMAGE_ID)

    def test_correct_image_and_mounts_accepted(self) -> None:
        docker = FakeDocker(mounts=self.mounts())
        self.assertEqual(self.launch(docker), 0)
        evidence = json.loads(self.evidence.read_text())
        self.assertEqual(evidence["verdict"], "ADMITTED_IMAGE_LAUNCHED")
        self.assertEqual(evidence["pre_launch_tag_resolution"]["id"], EL.ADMITTED_IMAGE_ID)
        self.assertEqual(evidence["post_create_container"]["image"], EL.ADMITTED_IMAGE_ID)
        self.assertFalse(evidence["legacy_launcher_invoked"])
        compose = [c for c in docker.calls if "compose" in c]
        self.assertEqual(len(compose), 1)
        self.assertEqual(compose[0][-6:], list(EL.UP_ARGUMENTS))
        for call in docker.calls:
            self.assertEqual(EL.forbidden_operations(call), [], call)

    def test_mount_drift_refused(self) -> None:
        other = Path(self.tmp.name) / "other"
        other.mkdir()
        for mounts, message in (
                (self.mounts(repo_rw=True), "repository"),
                (self.mounts(state_rw=False), "state"),
                (self.mounts(state=other), "state"),
                (self.mounts(fcos_rw=True), "FCOS")):
            with self.subTest(message=message), self.assertRaisesRegex(EL.EdgeImageError,
                                                                       message):
                EL.inspect_container(FakeDocker(mounts=mounts), state_root=self.state,
                                     fcos_source=self.fcos, fcos_sha256=self.fcos_sha)
        with self.assertRaisesRegex(EL.EdgeImageError, "content drift"):
            EL.inspect_container(FakeDocker(mounts=self.mounts()), state_root=self.state,
                                 fcos_source=self.fcos, fcos_sha256="0" * 64)

    def test_compose_failure_returns_code_and_nvidia_absent_refused(self) -> None:
        self.assertEqual(self.launch(FakeDocker(compose_rc=3, mounts=self.mounts())), 3)
        self.evidence.unlink()
        with self.assertRaisesRegex(EL.EdgeImageError, "nvidia"):
            self.launch(FakeDocker(runtimes="runc\n"))


class SeamTest(Fixture):
    def test_adapter_seam_reroutes_only_the_legacy_launcher(self) -> None:
        from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D

        original = D.subprocess
        seen = {}
        try:
            campaign = {"deployment": {"fcos_constructor_weights": {
                "path": "experiments/splitfusion_phase15_runtime_cache_v1/torch/hub/"
                        "checkpoints/fcos_resnet50_fpn_coco-99b0c9b7.pth",
                "sha256": "99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7"}}}
            EL.install_adapter_launch_seam(D, campaign, self.evidence)
            real_launch = EL.launch_no_build
            EL.launch_no_build = lambda env, **kw: seen.update(env=env, **kw) or 0
            try:
                result = D.subprocess.run([str(EL.LEGACY_LAUNCHER)], env={"A": "1"},
                                          stdout=io.BytesIO(), timeout=180.0)
            finally:
                EL.launch_no_build = real_launch
            self.assertEqual(result.returncode, 0)
            self.assertEqual(seen["env"], {"A": "1"})
            self.assertEqual(seen["timeout"], 180.0)
            self.assertEqual(seen["fcos_sha256"], campaign["deployment"][
                "fcos_constructor_weights"]["sha256"])
            self.assertIs(D.subprocess.Popen, subprocess.Popen)
            self.assertEqual(D.subprocess.run(["true"]).returncode, 0)   # passthrough
        finally:
            D.subprocess = original

    def test_nobuild_child_wraps_unchanged_child(self) -> None:
        calls = []
        original_install = NB._ORIGINAL_INSTALL
        original_seam = EL.install_adapter_launch_seam
        NB._ORIGINAL_INSTALL = lambda campaign, **kw: calls.append(("install", kw)) or "seams"
        EL.install_adapter_launch_seam = lambda adapter, campaign, path: calls.append(
            ("seam", str(path)))
        try:
            out = NB.install_run4_seams_nobuild({}, attempt_dir=Path("/x"), cell={})
        finally:
            NB._ORIGINAL_INSTALL, EL.install_adapter_launch_seam = original_install, original_seam
        self.assertEqual(out, "seams")
        self.assertEqual([c[0] for c in calls], ["install", "seam"])
        self.assertEqual(calls[1][1], "/x/run4_phase6/edge_image_launch.json")
        self.assertIs(NB._ORIGINAL_INSTALL, C.install_run4_seams)
        self.assertEqual(RUN.CHILD_MODULE,
                         "rl_agent.splitfusion_hybrid_sac_live_route_b_v2."
                         "phase6_live_child_nobuild_v2")

    def test_runner_refuses_before_radio_when_image_drifts(self) -> None:
        calls = []
        cell = SimpleNamespace(cell_id="a71_fav", action_index=71, action_id=71,
                               profile_id="split_ae32_uint4_q9800", model_family="AE32",
                               network_profile_id="FAVORABLE_STABLE", trace_id="t", seed=1)
        supervisor = SimpleNamespace(
            Cell=lambda **kw: SimpleNamespace(**kw), cell_to_dict=lambda c: dict(vars(c)),
            import_lifecycle_helper=lambda cfg: SimpleNamespace(),
            _require_phase15_application_cold=lambda c: calls.append("cold") or {},
            _start_live_radio=lambda *a: calls.append("radio_up") or ("ns", "s", {}),
            _stop_live_radio=lambda *a: calls.append("radio_down") or {},
            _stop_phase15_application=lambda c: calls.append("app_down") or {})

        def drifted():
            raise EL.EdgeImageError("oai-perception-rx:latest resolves to other")

        with tempfile.TemporaryDirectory() as tmp:
            report = RUN.run_one_cell(
                base_config={"runtime": {}}, registered=cell, output_root=Path(tmp),
                run_id="t", transmitted_budget=3, safety_timeout_s=1.0, carla_port=1,
                child_timeout_s=5.0, supervisor=supervisor, capture_class=None,
                image_resolver=drifted)
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("resolves to other", report["error"])
        self.assertNotIn("radio_up", calls)
        self.assertIn("edge_image_post_run_failure", report["cleanup"])
        self.assertFalse(any(k.endswith("_error") for k in report["cleanup"]
                             if k.startswith("edge_image")))


class PreservationTest(unittest.TestCase):
    def test_setup_addendum_binds_the_launcher_image(self) -> None:
        verified = RUN.verify_setup_addendum()
        self.assertEqual(verified["admitted_image_id"], EL.ADMITTED_IMAGE_ID)
        document = json.loads(RUN.SETUP_ADDENDUM_PATH.read_text(encoding="utf-8"))
        self.assertIs(document["scientific_protocol_changed"], False)
        with tempfile.TemporaryDirectory() as tmp:
            forged = Path(tmp) / "a.json"
            forged.write_text(json.dumps({**document, "admitted_image_id": OTHER_ID}))
            with self.assertRaises(RUN.Phase6RunnerError):
                RUN.verify_setup_addendum(forged)

    def test_failed_attempt_is_untouched(self) -> None:
        document = json.loads(RUN.SETUP_ADDENDUM_PATH.read_text(encoding="utf-8"))
        preserved = document["preserved_failed_attempt"]["sha256"]
        self.assertEqual(len(preserved), 15)
        for path, digest in preserved.items():
            target = ROOT / path
            if not target.exists():
                self.skipTest("failed-attempt evidence not present on this host")
            self.assertEqual(EL.sha256_file(target), digest, path)

    def test_frozen_files_unchanged_since_base(self) -> None:
        paths = [f"{PACKAGE}/{name}" for name in FROZEN_PACKAGE] + list(FROZEN_SHARED)
        for path in paths:
            committed = subprocess.run(["git", "show", f"{BASE_COMMIT}:{path}"], cwd=ROOT,
                                       capture_output=True, check=True).stdout
            self.assertEqual((ROOT / path).read_bytes(), committed, path)

    def test_scientific_settings_unchanged(self) -> None:
        plan = json.loads((ROOT / PACKAGE / "live_qualification_300_v2.json").read_text())
        self.assertEqual(plan["run"]["frames"], 300)
        self.assertEqual(plan["run"]["channel_profile"], "FAVORABLE_STABLE")
        self.assertEqual(plan["run"]["hold"], {"k_min": 2, "reward_deadline_ms": 170,
                                               "timeout_reward": -1.0})
        self.assertEqual(plan["run"]["actor"]["seed"], 43)
        self.assertEqual(plan["run"]["actor"]["update"], 10000)
        self.assertEqual(RUN.verify_addendum()["claim_scope"],
                         "SYSTEMS_INTEGRATION_QUALIFICATION_ONLY")

    def test_launch_module_import_is_pure(self) -> None:
        body = Path(EL.__file__).read_text(encoding="utf-8").split('"""', 2)[2]
        top_level = [line for line in body.splitlines()
                     if line and not line.startswith((" ", "#", ")", "]", "}"))]
        for line in top_level:
            self.assertNotIn("subprocess.run(", line)
            self.assertNotIn("open(", line)


if __name__ == "__main__":
    unittest.main()
