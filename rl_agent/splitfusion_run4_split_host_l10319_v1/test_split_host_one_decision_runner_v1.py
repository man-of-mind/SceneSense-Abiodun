from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from . import contract as C
from . import remote_edge_lifecycle_v1 as E
from . import split_host_one_decision_runner_v1 as R
from . import split_host_phase6_coordinator_v1 as CO
from . import remote_edge_held_session_v1 as RH


DIGEST = "a" * 64


def child_evidence() -> tuple[dict, dict]:
    identity = {
        "session_uuid": "s", "decision_seq": 0, "ticket_seq": 0,
        "mode_id": 11, "q_e4": 3000,
    }
    decision = {**identity, "frame_kind": "POLICY_DECISION",
                "reward_requested": True}
    hold = {**identity, "frame_kind": "POLICY_HOLD",
            "reward_requested": False}
    child = {
        "return_code": 0, "error": "", "stop_reason": "DECISION_CYCLE_BOUNDARY",
        "collector": {"collector_cleanup_ok": True, "collector_failures": [],
                      "transmitted_frames": 2},
    }
    ue = {
        "counters": {"policy_decisions": 1, "actor_calls": 1, "policy_holds": 1},
        "faulted": None, "unresolved_tickets_at_close": 0,
        "resolutions": [{"terminal": "SUCCESS"}],
        "frames": [{"reward_requested": True}, {"reward_requested": False}],
        "transmitted_identities": [
            {"run4_identity": decision}, {"run4_identity": hold}],
    }
    return child, ue


class FakeRan:
    def __init__(self, events: list) -> None:
        self.events = events
        self.source_route = {"radio_path": "PASS", "device": "oaitun_ue1",
                             "table": 9999}
        self.policy_ownership = CO.PolicyRoutingOwnershipV1(
            schema=CO.POLICY_OWNERSHIP_SCHEMA,
            before_rules_sha256=DIGEST, before_table_sha256=DIGEST,
            rules=(CO.OwnedPolicyRuleV1(31001, True),),
            routes=(CO.OwnedPolicyRouteV1("192.168.70.140/32", True),))

    def close(self):
        self.events.append("ran_close")
        return SimpleNamespace(ok=True, remote_cn_touched=False)


class FakeContext:
    def __init__(self, ops: "FakeOps") -> None:
        self.ops = ops
        self.sender_ever_connected = False
        self.closed = False

    def __enter__(self):
        self.ops.events.append("context_enter")
        self.ops.cap = 1
        return self

    def close(self):
        if not self.closed:
            self.ops.events.append("context_close")
            self.ops.cap = self.ops.original_cap
            self.closed = True

    def finalize_radio_tensor_path(self, _local, _remote):
        self.ops.events.append("proof_finalize")

    def remote_teardown_release(self):
        self.ops.events.append("teardown_release")
        return {"schema": CO.REMOTE_TEARDOWN_RELEASE_SCHEMA, "proof": DIGEST}

    def remote_abort_release(self):
        self.ops.events.append("abort_release")
        if not self.sender_ever_connected:
            return None
        return {"schema": CO.REMOTE_ABORT_RELEASE_SCHEMA,
                "scientific_pass": False, "proof": DIGEST}


class FakeOps:
    def __init__(self, *, fail: str | None = None,
                 capture_cleanup_failure: bool = False,
                 carla_cleanup_failure: bool = False) -> None:
        self.events: list = []
        self.fail = fail
        self.capture_cleanup_failure = capture_cleanup_failure
        self.carla_cleanup_failure = carla_cleanup_failure
        self.original_cap = 37
        self.cap = self.original_cap
        self.context_value: FakeContext | None = None
        self.evidence = child_evidence()

    def prepare(self, plan):
        self.events.append("prepare")
        retrieval = SimpleNamespace()
        return SimpleNamespace(retrieval=retrieval, child_args=SimpleNamespace())

    def remote_start(self, plan):
        self.events.append("remote_start")
        if self.fail == "remote_start_lost":
            raise RuntimeError("lost remote response")
        return {"status": "HELD_READY_CAPTURE_ACTIVE",
                "prestarted_remote_edge": {
                    "plan_sha256": DIGEST, "attempt_id": plan.attempt_id,
                    "project_name": f"run4-edge-l10319-{plan.attempt_id}",
                    "container_id": "container", "ready_sha256": DIGEST,
                    "gt_ready_sha256": DIGEST, "feedback_route_sha256": DIGEST}}

    def restart_remote_core(self, _plan):
        self.events.append("core_restart")
        if self.fail == "core_restart":
            raise RuntimeError("core reset boom")
        return {"status": "REMOTE_CORE_RESTARTED_HEALTHY",
                "evidence_sha256": DIGEST, "remote_cn_owned": False}

    def start_ran(self, _prepared, _plan):
        self.events.append("ran_start")
        return FakeRan(self.events)

    def start_carla(self, _prepared, _plan):
        self.events.append("carla_start")
        return object()

    def stop_carla(self, _handle):
        self.events.append("carla_stop")
        if self.carla_cleanup_failure:
            raise RuntimeError("carla cleanup boom")
        return {"shutdown_verified": True}

    def start_capture(self, _plan):
        self.events.append("capture_start")
        return object()

    def stop_capture(self, _handle):
        self.events.append("capture_stop")
        if self.capture_cleanup_failure:
            raise RuntimeError("capture cleanup boom")
        return Path("/fake/local.pcap")

    def context(self, _pre, _retrieval):
        self.events.append("context_make")
        self.context_value = FakeContext(self)
        return self.context_value

    def run_child(self, context, _prepared):
        self.events.append("child")
        if self.fail == "before_connect":
            raise RuntimeError("primary child boom")
        context.sender_ever_connected = True
        if self.fail == "after_connect":
            raise RuntimeError("primary child boom")
        return 0

    def child_evidence(self, _plan):
        self.events.append("child_evidence")
        return self.evidence

    def remote_seal(self, _plan):
        self.events.append("remote_seal")
        return {"status": "CAPTURE_SEALED_EDGE_STILL_HELD"}

    def remote_capture(self, _plan):
        self.events.append("remote_capture")
        return Path("/fake/remote.pcap")

    def prove(self, plan, _local, _remote, _ue):
        self.events.append("proof")
        R._write(plan.output_root / R.PROOF_NAME,
                 {"status": "PASS", "cross_host_latency_computed": False})
        return ({"status": "PASS", "cross_host_latency_computed": False},
                object(), object())

    def upload_release(self, _plan, _path, abort):
        self.events.append("upload_abort" if abort else "upload_release")
        return Path("/remote/release")

    def remote_stop(self, _plan, _release):
        self.events.append("remote_stop")
        if self.fail == "remote_stop":
            raise RuntimeError("remote stop response lost")
        return {"status": "STOPPED_PROJECT_ONLY_EVIDENCE_READY"}

    def remote_retrieve(self, _plan):
        self.events.append("remote_retrieve")
        return {"status": "RETRIEVED_CREATE_ONLY"}

    def verify_cold(self, _prepared):
        self.events.append("cold_after")
        return {"cold": True}

    def remote_abort(self, _plan, _primary, connected, release):
        self.events.append(f"remote_abort:{connected}:{release is not None}")
        return {"status": "ABORTED_CLEANLY"}


class RunnerTests(unittest.TestCase):
    def plan(self, root: Path) -> R.OneDecisionPlanV1:
        return R.OneDecisionPlanV1(root / "out", "run", "attempt")

    def test_success_orders_proof_before_release_and_restores_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps()
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertEqual(result["status"], "PASS_ONE_DECISION_SPLIT_HOST")
        self.assertEqual(ops.cap, ops.original_cap)
        order = ops.events
        self.assertLess(order.index("remote_start"), order.index("core_restart"))
        self.assertLess(order.index("core_restart"), order.index("ran_start"))
        self.assertLess(order.index("proof"), order.index("proof_finalize"))
        self.assertLess(order.index("proof_finalize"), order.index("teardown_release"))
        self.assertLess(order.index("teardown_release"), order.index("upload_release"))
        self.assertLess(order.index("upload_release"), order.index("remote_stop"))
        self.assertLess(order.index("remote_stop"), order.index("remote_retrieve"))
        self.assertNotIn("remote_abort:False:False", order)

    def test_lost_remote_start_response_still_attempts_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps(fail="remote_start_lost")
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertIn("lost remote response", result["error"])
        self.assertIn("remote_abort:False:False", ops.events)

    def test_core_gate_failure_starts_no_local_runtime_and_aborts_held_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps(fail="core_restart")
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertIn("core reset boom", result["error"])
        self.assertIn("remote_abort:False:False", ops.events)
        for event in ("ran_start", "carla_start", "capture_start", "child"):
            self.assertNotIn(event, ops.events)
        self.assertFalse(result["remote_cn_owned"])

    def test_abort_before_sender_connect_needs_no_release(self):
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps(fail="before_connect")
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertIn("primary child boom", result["error"])
        self.assertIn("remote_abort:False:False", ops.events)
        self.assertNotIn("upload_abort", ops.events)
        self.assertEqual(ops.cap, ops.original_cap)

    def test_abort_after_sender_connect_is_release_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps(fail="after_connect")
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertIn("primary child boom", result["error"])
        self.assertLess(ops.events.index("context_close"),
                        ops.events.index("abort_release"))
        self.assertLess(ops.events.index("upload_abort"),
                        ops.events.index("remote_abort:True:True"))
        self.assertEqual(ops.cap, ops.original_cap)

    def test_cleanup_failures_do_not_mask_primary_or_skip_other_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps(fail="after_connect", capture_cleanup_failure=True,
                          carla_cleanup_failure=True)
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertIn("primary child boom", result["error"])
        self.assertEqual(len(result["cleanup_errors"]), 2)
        self.assertIn("remote_abort:True:True", ops.events)
        self.assertIn("ran_close", ops.events)

    def test_failed_stop_retries_then_attempts_abort_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps(fail="remote_stop")
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertIn("remote stop response lost", result["error"])
        self.assertEqual(ops.events.count("remote_stop"), 2)
        self.assertIn("upload_abort", ops.events)
        self.assertIn("remote_abort:True:True", ops.events)

    def test_postconditions_refuse_unresolved_ticket(self):
        child, ue = child_evidence()
        ue["unresolved_tickets_at_close"] = 1
        with self.assertRaisesRegex(R.SplitHostOneDecisionError, "unresolved"):
            R.validate_closed_one_decision(child, ue)

    def test_contract_has_fragment_complete_filter_and_no_cross_host_timing(self):
        self.assertIn("udp dst port 51002", R.LOCAL_CAPTURE_FILTER)
        self.assertIn("ip[6:2] & 0x1fff", R.LOCAL_CAPTURE_FILTER)
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertIn("\"-s\", \"0\"", source)
        self.assertIn("\"chown\"", source)
        self.assertIn("attempt / \"ttracer\" / \"ue\"", source)
        self.assertIn("\"/usr/bin/bash\", \"-lc\"", source)
        self.assertIn("\"PYTHONPATH\"", source)
        self.assertIn("_require_phase15_application_cold", source)
        self.assertIn("cold_after_verified", source)
        with tempfile.TemporaryDirectory() as directory:
            ops = FakeOps()
            result = R.run_one_decision(self.plan(Path(directory)), ops=ops,
                                        watchdog=False)
        self.assertFalse(result["cross_host_monotonic_comparison_permitted"])
        self.assertEqual(result["policy_deadline_clock"],
                         "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY")

    def test_primary_plan_is_hard_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            self.assertEqual(plan.outer_runtime_s, 600)
            self.assertEqual(R.SAFETY_TIMEOUT_S, 60)
            self.assertEqual(R.DECISION_CAP, 1)
        self.assertGreaterEqual(
            R.REMOTE_START_TIMEOUT_S
            + RH.DEFAULT_START_ABORT_LOCK_TIMEOUT_S,
            RH.START_OPERATION_TIMEOUT_BUDGET_S)
        self.assertGreater(
            R.REMOTE_ABORT_TIMEOUT_S,
            RH.DEFAULT_START_ABORT_LOCK_TIMEOUT_S
            + RH.ABORT_CLEANUP_TIMEOUT_BUDGET_S)

    def test_plan_refuses_relative_output_and_uses_outside_repo_remote_root(self):
        with self.assertRaisesRegex(R.SplitHostOneDecisionError,
                                    "output root must be absolute"):
            self.plan(Path("relative-output")).validate()
        with tempfile.TemporaryDirectory() as directory:
            bad = R.OneDecisionPlanV1(Path(directory) / "absolute-output",
                                      "run", "Attempt-Uppercase")
            with self.assertRaisesRegex(R.SplitHostOneDecisionError,
                                        "remote-incompatible attempt identity"):
                bad.validate()
        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory) / "absolute-output").validate()
        self.assertEqual(plan.remote_base, R.REMOTE_ATTEMPT_BASE)
        self.assertNotEqual(plan.remote_repository, plan.remote_base)
        self.assertNotIn(plan.remote_repository, plan.remote_attempt.parents)
        self.assertEqual(plan.local_radio_state.parent, R.LOCAL_RADIO_STATE_BASE)
        self.assertEqual(R.LOCAL_RADIO_STATE_BASE,
                         R.LR.ROOT / "experiments/splitfusion_oai_100mhz_4d5u_v1")
        fcos = next(item for item in C.ARTIFACTS
                    if item.name == "torchvision_fcos")
        paths = E.RemoteEdgePaths(
            repository_root=plan.remote_repository,
            attempt_root=plan.remote_attempt,
            state_root=plan.remote_attempt / "state",
            evidence_root=plan.remote_attempt / "evidence",
            compose_path=plan.remote_attempt / "remote_edge.compose.json",
            fcos_weight_path=plan.remote_repository / fcos.relative_path,
            campaign_config_relative="rl_agent/configs/campaign.json",
        )
        self.assertIs(paths.validate(), paths)


    @staticmethod
    def core_inspect(suffix: str, healthy: bool = True):
        return [{
            "Id": service + "-id", "Name": "/" + service,
            "Config": {"Labels": {
                "com.docker.compose.project": R.REMOTE_CN_PROJECT,
                "com.docker.compose.service": service,
                "com.docker.compose.project.working_dir": str(R.REMOTE_CN_DIRECTORY),
                "com.docker.compose.project.config_files": str(R.REMOTE_CN_COMPOSE)}},
            "State": {"StartedAt": suffix, "Running": True,
                      "Health": {"Status": "healthy" if healthy else "unhealthy"}},
        } for service in R.REMOTE_CN_SERVICES]

    def test_remote_core_restart_targets_exact_services_and_proves_health(self):
        remote, calls = R.RemoteCliV1(), []
        inspections = [self.core_inspect("before"), self.core_inspect("after")]
        def ssh(argv, _timeout, data=None):
            calls.append(tuple(argv))
            if argv[0] == "test":
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            if argv[0] == "readlink":
                return SimpleNamespace(returncode=0,
                    stdout=(argv[-1] + "\n").encode(), stderr=b"")
            if argv[:4] == ["sudo", "-n", "docker", "inspect"]:
                return SimpleNamespace(returncode=0,
                    stdout=json.dumps(inspections.pop(0)).encode(), stderr=b"")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        remote._ssh = ssh
        evidence = remote.restart_core_and_wait_healthy(
            repository=R.REMOTE_REPOSITORY, health_timeout_s=0)
        restart = next(c for c in calls if c[:4] ==
                       ("sudo", "-n", "docker", "compose"))
        self.assertEqual(restart[-4:], ("restart", *R.REMOTE_CN_SERVICES))
        self.assertEqual(evidence["services"], list(R.REMOTE_CN_SERVICES))
        self.assertFalse(evidence["remote_cn_owned"])

    def test_remote_core_restart_refuses_missing_or_never_healthy(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                remote = R.RemoteCliV1()
                inspections = [self.core_inspect("before"),
                               self.core_inspect("after", healthy=False)]
                def ssh(argv, _timeout, data=None):
                    if argv[0] == "test":
                        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
                    if argv[0] == "readlink":
                        return SimpleNamespace(returncode=0,
                            stdout=(argv[-1] + "\n").encode(), stderr=b"")
                    if argv[:4] == ["sudo", "-n", "docker", "inspect"]:
                        if missing:
                            return SimpleNamespace(returncode=1, stdout=b"[]",
                                                   stderr=b"missing")
                        return SimpleNamespace(returncode=0,
                            stdout=json.dumps(inspections.pop(0)).encode(), stderr=b"")
                    return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
                remote._ssh = ssh
                with self.assertRaises(R.SplitHostOneDecisionError):
                    remote.restart_core_and_wait_healthy(
                        repository=R.REMOTE_REPOSITORY, health_timeout_s=0,
                        poll_interval_s=0)

    def test_remote_start_creates_only_exact_attempt_parent_first(self):
        events = []

        class RemoteSpy:
            def ensure_attempt_parent(self, *, repository, parent):
                events.append(("ensure", repository, parent))

            def call(self, operation, args, timeout, *, repository):
                events.append(("call", operation, tuple(args), timeout, repository))
                return {"status": "HELD_READY_CAPTURE_ACTIVE"}

        with tempfile.TemporaryDirectory() as directory:
            plan = self.plan(Path(directory))
            result = R.SystemOpsV1(remote=RemoteSpy()).remote_start(plan)
        self.assertEqual(result["status"], "HELD_READY_CAPTURE_ACTIVE")
        self.assertEqual(events[0],
                         ("ensure", plan.remote_repository, plan.remote_base))
        self.assertEqual(events[1][0:2], ("call", "start"))
        self.assertEqual(events[1][3], R.REMOTE_START_TIMEOUT_S)


if __name__ == "__main__":
    unittest.main()
