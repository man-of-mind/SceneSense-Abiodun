"""Offline tests for the durable L10319 held edge-session owner."""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from . import contract as C
from . import remote_edge_held_session_v1 as H
from . import remote_edge_lifecycle_v1 as E
from . import remote_edge_startup_qualifier_v1 as Q
from .test_remote_edge_startup_qualifier_v1 import (
    RepositoryFixture, edge_ready, gt_final, gt_ready, valid_image,
)


SOURCE_ROOT = Path(__file__).resolve().parents[2]


class HeldFakeRuntime:
    def __init__(self, fixture: RepositoryFixture) -> None:
        self.fixture = fixture
        self.calls: list[tuple[str, ...]] = []
        self.launched = False
        self.capture_active = False
        self.foreign_container = False
        self.fail_route = False
        self.fail_capture_health = False
        self.fail_capture_stop = False
        self.fail_logs = False
        self.fail_teardown = False
        self.invalid_pcap_header = False

    @property
    def prepared(self) -> Q.PreparedStartupAttempt:
        return H._rehydrate(self.fixture.attempt).prepared

    def container_json(self) -> str:
        prepared = self.prepared
        service = prepared.plan.compose_document["services"][E.SERVICE]
        labels = dict(service["labels"])
        labels["com.docker.compose.project"] = (
            "foreign-project" if self.foreign_container
            else prepared.plan.invocation.project_name
        )
        mounts = [{
            "Destination": row["target"], "Source": row["source"],
            "RW": not row["read_only"],
        } for row in service["volumes"]]
        return json.dumps([{
            "Id": "held-container", "Image": C.REMOTE_CONTAINER_IMAGE_ID,
            "Config": {"Labels": labels}, "Mounts": mounts,
        }])

    def publish_ready(self) -> None:
        prepared = self.prepared
        state = prepared.plan.paths.state_root
        (state / Path(E.GT_READY_DESTINATION).name).write_text(
            json.dumps(gt_ready(prepared)), encoding="utf-8")
        (state / Path(E.READY_DESTINATION).name).write_text(
            json.dumps(edge_ready(prepared)), encoding="utf-8")

    def finalize_capture(self) -> None:
        evidence = self.prepared.plan.paths.evidence_root
        pcap = evidence / H.CAPTURE_NAME
        if not pcap.exists():
            magic = (b"BAD!" if self.invalid_pcap_header
                     else b"\xd4\xc3\xb2\xa1")
            pcap.write_bytes(magic + b"\x00" * 20 + b"packet")
        log = evidence / H.CAPTURE_LOG_NAME
        if not log.exists():
            log.write_text("1 packet captured\n", encoding="utf-8")

    def __call__(self, argv, _timeout_s) -> Q.CommandResult:
        words = tuple(str(value) for value in argv)
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
            return Q.CommandResult(0, "true\n" if self.launched else "false\n")
        if "compose" in words and "config" in words:
            return Q.CommandResult(0)
        if "compose" in words and "up" in words:
            self.launched = True
            self.publish_ready()
            return Q.CommandResult(0)
        if words[:6] == ("sudo", "-n", "docker", "exec", E.CONTAINER,
                         "ip"):
            if self.fail_route:
                return Q.CommandResult(0, "10.0.0.2 dev eth0 src 192.168.70.140\n")
            return Q.CommandResult(
                0, "10.0.0.2 via 192.168.70.134 dev eth0 "
                   "src 192.168.70.140\n")
        if words[:7] == ("sudo", "-n", "docker", "exec", "-d",
                         E.CONTAINER, "sh"):
            self.capture_active = True
            (self.prepared.plan.paths.state_root / H.CAPTURE_PID_NAME).write_text(
                "4242\n", encoding="utf-8")
            return Q.CommandResult(0)
        if words[:6] == ("sudo", "-n", "docker", "exec", E.CONTAINER, "sh"):
            script = words[7]
            if "kill -INT" in script:
                if self.fail_capture_stop:
                    return Q.CommandResult(9, "", "capture stop failed")
                self.capture_active = False
                self.finalize_capture()
                return Q.CommandResult(0)
            if "kill -0" in script:
                if self.fail_capture_health or not self.capture_active:
                    return Q.CommandResult(1, "", "capture not alive")
                return Q.CommandResult(0)
        if words[:4] == ("sudo", "-n", "docker", "logs"):
            if self.fail_logs:
                return Q.CommandResult(7, "", "log failure")
            return Q.CommandResult(0, "edge held log\n", "edge held stderr\n")
        if "compose" in words and "down" in words:
            if self.fail_teardown:
                return Q.CommandResult(8, "", "teardown failure")
            if self.capture_active:
                self.capture_active = False
                self.finalize_capture()
            final_path = (self.prepared.plan.paths.state_root
                          / Path(E.GT_FINAL_DESTINATION).name)
            if not final_path.exists():
                final_path.write_text(json.dumps(gt_final(self.prepared)),
                                      encoding="utf-8")
            self.launched = False
            return Q.CommandResult(0)
        raise AssertionError(f"unexpected command: {words}")


class HeldSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = RepositoryFixture(Path(self.temporary.name))
        self.runtime = HeldFakeRuntime(self.fixture)
        self.owner = H.RemoteEdgeHeldSessionV1(
            runner=self.runtime,
            image_observer=lambda _stdout, binding: valid_image(binding),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def start(self):
        return self.owner.start(
            repository_root=self.fixture.repository,
            attempt_root=self.fixture.attempt,
            attempt_id="held-edge-001", run_id="run4-live",
            cell_id="a71__favorable_stable", timeout_s=1.0,
            hash_file=self.fixture.hash_file,
        )

    def seal(self):
        return self.owner.seal_capture(attempt_root=self.fixture.attempt)

    def release(self) -> Path:
        path = self.fixture.root / "local_gt_release.json"
        plan = H._rehydrate(self.fixture.attempt).prepared.plan
        path.write_text(json.dumps({
            "schema": H.LOCAL_GT_RELEASE_SCHEMA,
            "local_gt_sender_closed": True,
            "local_gt_sender_final_sha256": "a" * 64,
            "radio_tensor_path_sha256": "b" * 64,
            "remote_teardown_owner": "REMOTE_LIFECYCLE_OWNER_ONLY",
            "remote_cn_teardown_permitted_to_local_coordinator": False,
            "remote_attempt_id": plan.invocation.attempt_id,
            "remote_project_name": plan.invocation.project_name,
            "remote_plan_sha256": H.remote_plan_evidence_sha256(plan),
        }), encoding="utf-8")
        return path

    def abort_release(self) -> Path:
        plan = H._rehydrate(self.fixture.attempt).prepared.plan
        path = self.fixture.root / "local_gt_abort_release.json"
        path.write_text(json.dumps({
            "schema": H.LOCAL_GT_ABORT_RELEASE_SCHEMA,
            "local_gt_sender_was_connected": True,
            "local_gt_sender_closed": True,
            "local_gt_sender_final_sha256": "c" * 64,
            "remote_attempt_id": plan.invocation.attempt_id,
            "remote_project_name": plan.invocation.project_name,
            "remote_plan_sha256": H.remote_plan_evidence_sha256(plan),
            "remote_cn_teardown_permitted_to_local_coordinator": False,
            "scientific_pass": False,
        }), encoding="utf-8")
        return path

    def test_start_holds_verified_edge_and_capture_across_process_boundary(self):
        report = self.start()
        self.assertEqual(report["status"], "HELD_READY_CAPTURE_ACTIVE")
        self.assertTrue(self.runtime.launched)
        self.assertTrue(self.runtime.capture_active)
        remote = report["prestarted_remote_edge"]
        self.assertEqual(remote["container_id"], "held-container")
        self.assertEqual(remote["attempt_id"], "held-edge-001")
        self.assertEqual(remote["project_name"],
                         "run4-edge-l10319-held-edge-001")
        self.assertEqual(remote["plan_sha256"], H.remote_plan_evidence_sha256(
            H._rehydrate(self.fixture.attempt).prepared.plan))
        self.assertFalse(any("down" in call for call in self.runtime.calls))
        route_index = next(i for i, call in enumerate(self.runtime.calls)
                           if "route" in call and "get" in call)
        capture_index = next(i for i, call in enumerate(self.runtime.calls)
                             if "exec" in call and "-d" in call)
        self.assertLess(route_index, capture_index)
        persisted = json.loads(
            (self.fixture.attempt / H.START_NAME).read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], report["status"])
        self.assertFalse(persisted["remote_cn_owned"])
        route = json.loads(
            (self.fixture.attempt / H.FEEDBACK_ROUTE_NAME).read_text())
        self.assertEqual(route["via"], "192.168.70.134")
        self.assertEqual(route["device"], "eth0")

        # A new owner has no in-memory handle, yet rehydrates and revalidates
        # the same attempt/container/capture from durable records.
        second_owner = H.RemoteEdgeHeldSessionV1(runner=self.runtime)
        status = second_owner.capture(attempt_root=self.fixture.attempt)
        self.assertEqual(status["status"], "HELD_CAPTURE_ACTIVE")

    def test_stop_requires_release_then_orders_capture_logs_and_project_down(self):
        self.start()
        before = len(self.runtime.calls)
        with self.assertRaises(H.RemoteEdgeHeldSessionError):
            self.owner.stop(
                attempt_root=self.fixture.attempt,
                local_gt_release=self.fixture.root / "absent.json")
        self.assertEqual(len(self.runtime.calls), before)
        self.assertTrue(self.runtime.launched)

        self.seal()
        report = H.RemoteEdgeHeldSessionV1(runner=self.runtime).stop(
            attempt_root=self.fixture.attempt, local_gt_release=self.release())
        self.assertEqual(report["status"],
                         "STOPPED_PROJECT_ONLY_EVIDENCE_READY")
        self.assertFalse(self.runtime.launched)
        self.assertFalse(self.runtime.capture_active)
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertTrue(report["cleanup"]["gt_listener_stopped"])
        self.assertGreaterEqual(report["pcap_bytes"], 24)
        stop_index = next(i for i, call in enumerate(self.runtime.calls)
                          if "kill -INT" in " ".join(call))
        logs_index = next(i for i, call in enumerate(self.runtime.calls)
                          if call[:4] == ("sudo", "-n", "docker", "logs"))
        down_index = next(i for i, call in enumerate(self.runtime.calls)
                          if "compose" in call and "down" in call)
        self.assertLess(stop_index, logs_index)
        self.assertLess(logs_index, down_index)
        down = self.runtime.calls[down_index]
        self.assertIn("run4-edge-l10319-held-edge-001", down)
        self.assertNotIn("oai-cn5g", " ".join(down))

    def test_stop_refuses_cross_attempt_project_or_plan_release_replay(self):
        self.start()
        self.seal()
        valid = json.loads(self.release().read_text(encoding="utf-8"))
        mutations = {
            "remote_attempt_id": "another-attempt",
            "remote_project_name": "run4-edge-l10319-another-attempt",
            "remote_plan_sha256": "f" * 64,
        }
        for index, (field, value) in enumerate(mutations.items()):
            with self.subTest(field=field):
                document = dict(valid)
                document[field] = value
                path = self.fixture.root / f"foreign-release-{index}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                before = len(self.runtime.calls)
                with self.assertRaises(H.RemoteEdgeHeldSessionError):
                    self.owner.stop(
                        attempt_root=self.fixture.attempt,
                        local_gt_release=path)
                self.assertEqual(len(self.runtime.calls), before)
                self.assertTrue(self.runtime.launched)

    def test_retrieve_is_create_only_hashes_pcap_and_excludes_seeded_cache(self):
        self.start()
        self.seal()
        self.owner.stop(attempt_root=self.fixture.attempt,
                        local_gt_release=self.release())
        destination = self.fixture.outputs / "retrieved"
        result = H.RemoteEdgeHeldSessionV1().retrieve(
            attempt_root=self.fixture.attempt, destination_root=destination)
        self.assertEqual(result["status"], "RETRIEVED_CREATE_ONLY")
        paths = {row["path"] for row in result["files"]}
        self.assertIn("evidence/edge_tensor_ingress.pcap", paths)
        self.assertIn(H.START_NAME, paths)
        self.assertIn(H.STOP_NAME, paths)
        self.assertFalse(any(path.startswith("state/hub/") for path in paths))
        self.assertTrue(result["excluded_seeded_cache"])
        for row in result["files"]:
            target = destination / row["path"]
            self.assertEqual(H._sha256_file(target), row["sha256"])
        with self.assertRaises(H.RemoteEdgeHeldSessionError):
            H.RemoteEdgeHeldSessionV1().retrieve(
                attempt_root=self.fixture.attempt,
                destination_root=destination)

    def test_feedback_route_failure_cleans_failed_start_without_cn_mutation(self):
        self.runtime.fail_route = True
        report = self.start()
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("feedback route", report["error"])
        self.assertFalse(self.runtime.launched)
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertTrue(any("down" in call for call in self.runtime.calls))
        rendered = "\n".join(" ".join(call) for call in self.runtime.calls)
        self.assertNotIn("cn_start", rendered)
        self.assertNotIn("oai-cn5g-basic", rendered)

    def test_capture_health_failure_stops_capture_before_failed_start_teardown(self):
        self.runtime.fail_capture_health = True
        report = self.start()
        self.assertEqual(report["status"], "FAILED")
        self.assertIn("command failed", report["error"])
        self.assertFalse(self.runtime.launched)
        stop_index = next(i for i, call in enumerate(self.runtime.calls)
                          if "kill -INT" in " ".join(call))
        down_index = next(i for i, call in enumerate(self.runtime.calls)
                          if "down" in call)
        self.assertLess(stop_index, down_index)

    def test_stop_preserves_first_failure_and_attempts_remaining_cleanup(self):
        self.start()
        self.seal()
        self.runtime.fail_logs = True
        report = self.owner.stop(
            attempt_root=self.fixture.attempt, local_gt_release=self.release())
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["error"], "timestamped edge log capture failed")
        self.assertTrue(report["cleanup"]["capture_stopped"])
        self.assertTrue(report["cleanup"]["project_teardown_complete"])
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertTrue(report["cleanup"]["gt_listener_stopped"])
        self.assertFalse(self.runtime.launched)

    def test_stop_refuses_foreign_container_without_signalling_or_teardown(self):
        self.start()
        self.seal()
        self.runtime.foreign_container = True
        before = len(self.runtime.calls)
        report = self.owner.stop(
            attempt_root=self.fixture.attempt, local_gt_release=self.release())
        self.assertEqual(report["status"], "FAILED")
        self.assertTrue(
            report["cleanup"]["refused_foreign_or_unproven_container"])
        later = self.runtime.calls[before:]
        self.assertFalse(any("kill -INT" in " ".join(call) for call in later))
        self.assertFalse(any("down" in call for call in later))
        self.assertTrue(self.runtime.launched)

    def test_seal_and_capture_retrieval_keep_edge_and_gt_held(self):
        self.start()
        seal = self.seal()
        self.assertEqual(seal["status"], "CAPTURE_SEALED_EDGE_STILL_HELD")
        self.assertTrue(self.runtime.launched)
        self.assertFalse(self.runtime.capture_active)
        self.assertGreaterEqual(seal["pcap_bytes"], 24)
        destination = self.fixture.outputs / "capture-retrieval"
        result = H.RemoteEdgeHeldSessionV1(runner=self.runtime).retrieve_capture(
            attempt_root=self.fixture.attempt, destination_root=destination)
        self.assertEqual(result["status"],
                         "CAPTURE_RETRIEVED_EDGE_STILL_HELD")
        self.assertTrue(self.runtime.launched)
        self.assertEqual(H._sha256_file(destination / H.CAPTURE_NAME),
                         seal["pcap_sha256"])
        self.assertFalse(any("down" in call for call in self.runtime.calls))

    def test_seal_refuses_invalid_pcap_magic_without_tearing_down_edge(self):
        self.start()
        self.runtime.invalid_pcap_header = True
        with self.assertRaisesRegex(
                H.RemoteEdgeHeldSessionError,
                "pcap header magic is invalid"):
            self.seal()
        self.assertFalse(
            (self.fixture.attempt / H.CAPTURE_SEALED_NAME).exists())
        self.assertTrue(self.runtime.launched)
        self.assertFalse(self.runtime.capture_active)

    def test_stop_without_radio_proof_is_refused_but_abort_cleans_owned_project(self):
        self.start()
        before = len(self.runtime.calls)
        with self.assertRaises(H.RemoteEdgeHeldSessionError):
            self.owner.stop(attempt_root=self.fixture.attempt,
                            local_gt_release=self.release())
        self.assertEqual(len(self.runtime.calls), before)
        aborted = self.owner.abort(
            attempt_root=self.fixture.attempt,
            primary_failure="local child failed before first tensor proof",
            local_gt_sender_connected=False)
        self.assertEqual(aborted["status"], "ABORTED_CLEANLY")
        self.assertFalse(aborted["scientific_pass"])
        self.assertNotIn("PASS", aborted["status"])
        self.assertFalse(self.runtime.launched)

    def test_connected_abort_requires_bound_sender_closed_release(self):
        self.start()
        with self.assertRaises(H.RemoteEdgeHeldSessionError):
            self.owner.abort(attempt_root=self.fixture.attempt,
                primary_failure="child failed", local_gt_sender_connected=True)
        report = self.owner.abort(attempt_root=self.fixture.attempt,
            primary_failure="child failed", local_gt_sender_connected=True,
            local_gt_abort_release=self.abort_release())
        self.assertEqual(report["status"], "ABORTED_CLEANLY")

    def test_lost_pre_start_record_cleans_only_exact_owned_project(self):
        self.start()
        (self.fixture.attempt / H.START_NAME).unlink()
        primary = "SSH response lost before durable START result"
        report = self.owner.abort(
            attempt_root=self.fixture.attempt, primary_failure=primary,
            local_gt_sender_connected=False)
        self.assertEqual(report["status"], "ABORTED_CLEANLY")
        self.assertEqual(report["primary_failure"], primary)
        self.assertTrue(report["cleanup"]["project_teardown_complete"])
        self.assertTrue(report["cleanup"]["container_absent"])
        self.assertFalse(report["scientific_pass"])
        self.assertFalse(report["remote_cn_owned"])
        self.assertFalse(self.runtime.launched)

    def test_start_and_abort_are_serialized_by_bounded_attempt_lock(self):
        entered = threading.Event()
        release = threading.Event()
        result: list[object] = []
        original = self.owner._start_locked

        def blocked_start(**_kwargs):
            entered.set()
            self.assertTrue(release.wait(2.0))
            return {"status": "FIRST_START_OWNER_FINISHED"}

        self.owner._start_locked = blocked_start  # type: ignore[method-assign]
        thread = threading.Thread(target=lambda: result.append(self.owner.start(
            repository_root=self.fixture.repository,
            attempt_root=self.fixture.attempt, attempt_id="held-edge-001",
            run_id="run4-live", cell_id="a71__favorable_stable",
            timeout_s=1.0, hash_file=self.fixture.hash_file,
            lock_timeout_s=1.0)))
        thread.start()
        self.assertTrue(entered.wait(1.0))
        with self.assertRaisesRegex(
                H.RemoteEdgeHeldSessionError,
                "timed out waiting for held-session START/ABORT owner"):
            H.RemoteEdgeHeldSessionV1(runner=self.runtime).start(
                repository_root=self.fixture.repository,
                attempt_root=self.fixture.attempt, attempt_id="held-edge-001",
                run_id="run4-live", cell_id="a71__favorable_stable",
                timeout_s=1.0, hash_file=self.fixture.hash_file,
                lock_timeout_s=0.05)
        release.set()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [{"status": "FIRST_START_OWNER_FINISHED"}])
        self.owner._start_locked = original  # type: ignore[method-assign]

    def test_lost_pre_start_recovery_refuses_foreign_live_container(self):
        self.start()
        (self.fixture.attempt / H.START_NAME).unlink()
        self.runtime.foreign_container = True
        before = len(self.runtime.calls)
        report = self.owner.abort(
            attempt_root=self.fixture.attempt,
            primary_failure="start owner disappeared",
            local_gt_sender_connected=False)
        later = self.runtime.calls[before:]
        self.assertEqual(report["status"], "ABORTED_WITH_CLEANUP_ERRORS")
        self.assertTrue(
            report["cleanup"]["refused_foreign_or_unproven_container"])
        self.assertFalse(any("kill -INT" in " ".join(call) for call in later))
        self.assertFalse(any("down" in call for call in later))
        self.assertTrue(self.runtime.launched)
        self.assertFalse(report["scientific_pass"])
        self.assertFalse(report["remote_cn_owned"])

    def test_abort_refuses_a_successful_stop_without_runtime_mutation(self):
        self.start()
        self.seal()
        stopped = self.owner.stop(
            attempt_root=self.fixture.attempt, local_gt_release=self.release())
        self.assertEqual(stopped["status"],
                         "STOPPED_PROJECT_ONLY_EVIDENCE_READY")
        before = len(self.runtime.calls)
        with self.assertRaisesRegex(
                H.RemoteEdgeHeldSessionError,
                "successful held-session stop already exists"):
            self.owner.abort(
                attempt_root=self.fixture.attempt,
                primary_failure="ambiguous stop response",
                local_gt_sender_connected=False)
        self.assertEqual(len(self.runtime.calls), before)
        self.assertFalse((self.fixture.attempt / H.ABORT_NAME).exists())

    def test_abort_recovers_failed_stop_that_left_owned_container_running(self):
        self.start()
        self.seal()
        self.runtime.fail_teardown = True
        stopped = self.owner.stop(
            attempt_root=self.fixture.attempt, local_gt_release=self.release())
        self.assertEqual(stopped["status"], "FAILED")
        self.assertTrue(self.runtime.launched)

        self.runtime.fail_teardown = False
        primary = "coordinator lost the failed STOP response"
        recovered = self.owner.abort(
            attempt_root=self.fixture.attempt, primary_failure=primary,
            local_gt_sender_connected=False)
        self.assertEqual(recovered["status"], "ABORTED_CLEANLY")
        self.assertEqual(recovered["primary_failure"], primary)
        self.assertEqual(
            recovered["recovery_of_non_success_stop"]["status"], "FAILED")
        self.assertFalse(recovered["scientific_pass"])
        self.assertFalse(recovered["remote_cn_owned"])
        self.assertTrue(recovered["cleanup"]["project_teardown_complete"])
        self.assertFalse(self.runtime.launched)
        self.assertEqual(recovered["edge_log_capture"]["status"],
                         "PRESERVED_FROM_PRIOR_STOP")

    def test_abort_recovers_failed_stop_after_ambiguous_completed_teardown(self):
        self.start()
        self.seal()
        self.runtime.fail_logs = True
        stopped = self.owner.stop(
            attempt_root=self.fixture.attempt, local_gt_release=self.release())
        self.assertEqual(stopped["status"], "FAILED")
        self.assertFalse(self.runtime.launched)
        down_before = sum(
            1 for call in self.runtime.calls if "compose" in call and "down" in call)

        primary = "remote STOP reply was ambiguous"
        recovered = self.owner.abort(
            attempt_root=self.fixture.attempt, primary_failure=primary,
            local_gt_sender_connected=False)
        down_after = sum(
            1 for call in self.runtime.calls if "compose" in call and "down" in call)
        self.assertEqual(recovered["status"], "ABORTED_CLEANLY")
        self.assertEqual(recovered["primary_failure"], primary)
        self.assertTrue(recovered["cleanup"]["container_already_absent"])
        self.assertEqual(down_after, down_before + 1)
        self.assertFalse(recovered["scientific_pass"])
        self.assertFalse(recovered["remote_cn_owned"])

    def test_failed_stop_recovery_refuses_foreign_or_unproven_container(self):
        self.start()
        self.seal()
        self.runtime.foreign_container = True
        stopped = self.owner.stop(
            attempt_root=self.fixture.attempt, local_gt_release=self.release())
        self.assertEqual(stopped["status"], "FAILED")
        before = len(self.runtime.calls)
        primary = "stop refused a foreign live container"
        refused = self.owner.abort(
            attempt_root=self.fixture.attempt, primary_failure=primary,
            local_gt_sender_connected=False)
        later = self.runtime.calls[before:]
        self.assertEqual(refused["status"], "ABORTED_WITH_CLEANUP_ERRORS")
        self.assertEqual(refused["primary_failure"], primary)
        self.assertTrue(
            refused["cleanup"]["refused_foreign_or_unproven_container"])
        self.assertFalse(any("kill -INT" in " ".join(call) for call in later))
        self.assertFalse(any("down" in call for call in later))
        self.assertTrue(self.runtime.launched)
        self.assertFalse(refused["scientific_pass"])
        self.assertFalse(refused["remote_cn_owned"])

    def test_commands_are_edge_only_and_capture_filter_is_exact(self):
        for command in (
            H.feedback_route_command(), H.capture_start_command(),
            H.capture_health_command(), H.capture_stop_command(),
        ):
            E.assert_edge_only_command(command.argv)
        self.assertEqual(
            H.CAPTURE_FILTER,
            "src host 10.0.0.2 and dst host 192.168.70.140 and "
            "(udp dst port 51002 or (ip[6:2] & 0x1fff != 0))")
        start = H.capture_start_command().argv
        self.assertIn("eth0", " ".join(start))
        self.assertIn(H.CAPTURE_FILTER, start)

    def test_cli_has_cross_ssh_operations_and_execute_gate(self):
        for operation in ("start", "capture", "seal-capture", "retrieve-capture",
                          "stop", "abort", "retrieve"):
            with self.subTest(operation=operation):
                self.assertIn(operation, H.build_parser()._subparsers._group_actions[0]
                              .choices)
        with self.assertRaises(H.RemoteEdgeHeldSessionError):
            H.main(["capture", "--attempt-root", str(self.fixture.attempt)])


class ImportPurityTests(unittest.TestCase):
    def test_host_only_import_does_not_import_torch_or_touch_runtime(self):
        script = r'''
import builtins
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("held-session module imported torch")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import rl_agent.splitfusion_run4_split_host_l10319_v1.remote_edge_held_session_v1
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
'''
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=SOURCE_ROOT, env=environment,
            text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
