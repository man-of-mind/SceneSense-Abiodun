"""Durable, project-scoped owner for one held L10319 edge session.

The startup qualifier intentionally tears its container down as soon as READY
is proved.  A split-host child needs the same verified edge to remain alive
across *separate* SSH invocations.  This module composes the qualifier's
preparation and validation primitives into durable cross-SSH operations:

``start``
    create one attempt, verify the registered artifacts/image/GPU/network,
    launch only the edge service, validate both READY records and the feedback
    route, then start an attempt-owned capture on the container's ``eth0``;
``capture``
    revalidate ownership and report that the held capture is still alive;
``seal-capture`` and ``retrieve-capture``
    stop and hash the pcap while edge/GT remain alive, then copy it create-only
    so the local owner can prove the first real tensor before issuing release;
``stop``
    only after a sealed capture and validated local-GT release, persist logs,
    tear down only the matching Compose project, and validate GT FINAL; and
``abort``
    preserve an upstream failure while cleaning the exact owned project,
    without requiring or ever claiming a scientific tensor-path proof;
``retrieve``
    create a new evidence tree and hash every copied file, including the pcap.

Importing this module performs no I/O and starts no process.  It never starts,
stops, or otherwise mutates the remote CN.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

from . import contract as C
from . import remote_edge_entry_v1 as GPU
from . import remote_edge_lifecycle_v1 as E
from . import remote_edge_startup_qualifier_v1 as Q
from . import split_host_phase6_coordinator_v1 as COORD


SCHEMA = "scenesense.run4.remote_edge_held_session.v1"
EXECUTE_TOKEN = "SPLITFUSION_RUN4_REMOTE_EDGE_HELD_SESSION_V1_EXECUTE"
INTENT_NAME = "REMOTE_EDGE_HELD_SESSION_INTENT.json"
START_NAME = "REMOTE_EDGE_HELD_SESSION_START.json"
STOP_NAME = "REMOTE_EDGE_HELD_SESSION_STOP.json"
ABORT_NAME = "REMOTE_EDGE_HELD_SESSION_ABORT.json"
FEEDBACK_ROUTE_NAME = "REMOTE_EDGE_FEEDBACK_ROUTE.json"
CAPTURE_SEALED_NAME = "REMOTE_EDGE_CAPTURE_SEALED.json"
CAPTURE_NAME = "edge_tensor_ingress.pcap"
CAPTURE_LOG_NAME = "edge_tensor_ingress_tcpdump.log"
CAPTURE_PID_NAME = "edge_tensor_ingress_tcpdump.pid"
CAPTURE_RETRIEVAL_MANIFEST_NAME = "REMOTE_EDGE_CAPTURE_RETRIEVAL.json"
RETRIEVAL_MANIFEST_NAME = "REMOTE_EDGE_EVIDENCE_MANIFEST.json"
CAPTURE_FILTER = (
    "src host 10.0.0.2 and dst host 192.168.70.140 and "
    "(udp dst port 51002 or (ip[6:2] & 0x1fff != 0))"
)
REMOTE_PCAP = f"{E.EVIDENCE_DESTINATION}/{CAPTURE_NAME}"
REMOTE_CAPTURE_LOG = f"{E.EVIDENCE_DESTINATION}/{CAPTURE_LOG_NAME}"
REMOTE_CAPTURE_PID = f"{E.STATE_DESTINATION}/{CAPTURE_PID_NAME}"
LOCAL_GT_RELEASE_SCHEMA = "scenesense.run4.split_host_remote_teardown_release.v1"
LOCAL_GT_ABORT_RELEASE_SCHEMA = "scenesense.run4.split_host_remote_abort_release.v1"
DEFAULT_READY_TIMEOUT_S = 300.0
START_OPERATION_TIMEOUT_BUDGET_S = 630.0
DEFAULT_START_ABORT_LOCK_TIMEOUT_S = 330.0
ABORT_CLEANUP_TIMEOUT_BUDGET_S = 140.0
START_ABORT_LOCK_PREFIX = ".splitfusion_remote_edge_start_abort_"


class RemoteEdgeHeldSessionError(RuntimeError):
    """A held-session operation is unsafe, incomplete, or identity-drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RemoteEdgeHeldSessionError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pcap_identity(path: Path) -> tuple[str, int]:
    path = Path(path)
    _require(path.is_file(), "edge tensor pcap is absent")
    size = path.stat().st_size
    _require(size >= 24, "edge tensor pcap is shorter than a pcap header")
    with path.open("rb") as handle:
        magic = handle.read(4)
    _require(magic in {
        b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4",
        b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d",
        b"\x0a\x0d\x0d\x0a",
    }, "edge tensor pcap header magic is invalid")
    return _sha256_file(path), size


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:2000]


def _record_primary(report: dict[str, Any], exc: BaseException | str) -> None:
    message = str(exc) if isinstance(exc, str) else _error(exc)
    if not report.get("error"):
        report["error"] = message[:2000]
    else:
        report.setdefault("secondary_errors", []).append(message[:2000])


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    Q._write_create_only(Path(path), value)


def _load_exact_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RemoteEdgeHeldSessionError(f"{label} is unreadable: {exc}") from exc
    _require(type(value) is dict, f"{label} must be an exact object")
    return value


def _start_abort_lock_path(attempt_root: Path) -> Path:
    attempt = Path(attempt_root).resolve(strict=False)
    digest = hashlib.sha256(str(attempt).encode("utf-8")).hexdigest()[:20]
    return attempt.parent / f"{START_ABORT_LOCK_PREFIX}{digest}.lock"


@contextmanager
def _start_abort_lock(
    attempt_root: Path, *, timeout_s: float,
) -> Iterator[None]:
    """Serialize START/ABORT across processes; the kernel releases dead owners."""
    _require(type(timeout_s) in (int, float)
             and 0.0 < float(timeout_s) <= 720.0,
             "start/abort lock timeout is outside (0,720]")
    path = _start_abort_lock_path(attempt_root)
    _require(path.parent.is_dir(), "attempt parent for operation lock is absent")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    deadline = time.monotonic() + float(timeout_s)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                _require(remaining > 0.0,
                         "timed out waiting for held-session START/ABORT owner")
                time.sleep(min(0.05, remaining))
        yield
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class DurableAttempt:
    prepared: Q.PreparedStartupAttempt
    intent: Mapping[str, Any]


def _intent_document(prepared: Q.PreparedStartupAttempt) -> Mapping[str, Any]:
    plan = prepared.plan
    return {
        "schema": SCHEMA,
        "status": "INTENT_CREATED_BEFORE_LAUNCH",
        "repository_root": str(plan.paths.repository_root),
        "attempt_root": str(plan.paths.attempt_root),
        "attempt_id": plan.invocation.attempt_id,
        "run_id": plan.invocation.run_id,
        "cell_id": plan.invocation.cell_id,
        "action_id": plan.invocation.action_id,
        "allowed_action_ids": list(plan.invocation.allowed_action_ids),
        "edge_receive_port": plan.invocation.edge_receive_port,
        "ue_control_host": plan.invocation.ue_control_host,
        "ue_control_port": plan.invocation.ue_control_port,
        "project_name": plan.invocation.project_name,
        "compose_sha256": plan.compose_sha256,
        "container": E.CONTAINER,
        "remote_cn_owned": False,
    }


def _rehydrate(attempt_root: Path) -> DurableAttempt:
    attempt_root = Path(attempt_root).resolve(strict=True)
    intent = _load_exact_json(attempt_root / INTENT_NAME, "held-session intent")
    expected_fields = {
        "schema", "status", "repository_root", "attempt_root", "attempt_id",
        "run_id", "cell_id", "action_id", "allowed_action_ids",
        "edge_receive_port", "ue_control_host", "ue_control_port",
        "project_name", "compose_sha256", "container", "remote_cn_owned",
    }
    _require(set(intent) == expected_fields, "held-session intent fields drift")
    _require(intent["schema"] == SCHEMA, "held-session intent schema drift")
    _require(intent["status"] == "INTENT_CREATED_BEFORE_LAUNCH",
             "held-session intent status drift")
    _require(Path(str(intent["attempt_root"])).resolve(strict=False) == attempt_root,
             "held-session attempt root drift")
    _require(intent["container"] == E.CONTAINER,
             "held-session container identity drift")
    _require(intent["remote_cn_owned"] is False,
             "held-session must never claim remote CN ownership")

    repository_root = Path(str(intent["repository_root"])).resolve(strict=True)
    binding, _binding_sha = Q.load_binding(repository_root)
    binding.validate()
    fcos = next(item for item in C.ARTIFACTS if item.name == "torchvision_fcos")
    paths = E.RemoteEdgePaths(
        repository_root=repository_root,
        attempt_root=attempt_root,
        state_root=attempt_root / "state",
        evidence_root=attempt_root / "evidence",
        compose_path=attempt_root / "remote_edge.compose.json",
        fcos_weight_path=repository_root / fcos.relative_path,
        campaign_config_relative=Q.CAMPAIGN_RELATIVE,
    )
    invocation = E.RemoteEdgeInvocation(
        attempt_id=str(intent["attempt_id"]),
        run_id=str(intent["run_id"]),
        cell_id=str(intent["cell_id"]),
        action_id=int(intent["action_id"]),
        allowed_action_ids=tuple(int(v) for v in intent["allowed_action_ids"]),
        edge_receive_port=int(intent["edge_receive_port"]),
        ue_control_host=str(intent["ue_control_host"]),
        ue_control_port=int(intent["ue_control_port"]),
    )
    plan = E.build_plan(binding=binding, paths=paths, invocation=invocation)
    _require(plan.invocation.project_name == intent["project_name"],
             "held-session Compose project drift")
    _require(plan.compose_sha256 == intent["compose_sha256"],
             "held-session reconstructed plan drift")
    compose = _load_exact_json(paths.compose_path, "attempt Compose document")
    _require(_canonical_sha256(compose) == plan.compose_sha256,
             "materialized Compose document drift")
    return DurableAttempt(
        prepared=Q.PreparedStartupAttempt(
            binding=binding, plan=plan, result_path=attempt_root / Q.RESULT_NAME),
        intent=intent,
    )


def feedback_route_command() -> E.LifecycleCommand:
    return E.LifecycleCommand(
        "validate remote edge feedback route through the UPF",
        ("sudo", "-n", "docker", "exec", E.CONTAINER,
         "ip", "route", "get", COORD.UE_CONTROL_HOST),
    )


def _feedback_route_observation(stdout: str, *, returncode: int) -> Mapping[str, Any]:
    text = " ".join(str(stdout).split())
    via = re.search(r"(?:^| )via ([0-9.]+)(?: |$)", text)
    device = re.search(r"(?:^| )dev ([^ ]+)(?: |$)", text)
    observation = {
        "schema": COORD.FEEDBACK_ROUTE_SCHEMA,
        "observed_host": C.default_topology().remote_name,
        "container_ip": C.default_topology().edge_ip,
        "destination": COORD.UE_CONTROL_HOST,
        "via": via.group(1) if via else "",
        "device": device.group(1) if device else "",
        "returncode": int(returncode),
        "route_text": text,
    }
    return COORD.validate_remote_feedback_route(observation)


def capture_start_command() -> E.LifecycleCommand:
    script = (
        "set -euC; tcpdump -i eth0 -nn -U -w \"$1\" \"$2\" "
        ">\"$3\" 2>&1 & pid=$!; printf '%s\\n' \"$pid\" >\"$4\"; wait \"$pid\""
    )
    return E.LifecycleCommand(
        "start attempt-owned edge tensor capture",
        ("sudo", "-n", "docker", "exec", "-d", E.CONTAINER,
         "sh", "-c", script, "splitfusion-edge-capture",
         REMOTE_PCAP, CAPTURE_FILTER, REMOTE_CAPTURE_LOG, REMOTE_CAPTURE_PID),
    )


def capture_health_command() -> E.LifecycleCommand:
    script = "test -s \"$1\" && kill -0 \"$(cat \"$1\")\""
    return E.LifecycleCommand(
        "require attempt-owned edge tensor capture alive",
        ("sudo", "-n", "docker", "exec", E.CONTAINER,
         "sh", "-c", script, "splitfusion-edge-capture", REMOTE_CAPTURE_PID),
    )


def capture_stop_command() -> E.LifecycleCommand:
    script = (
        "set -eu; test -s \"$1\"; pid=$(cat \"$1\"); kill -INT \"$pid\"; "
        "n=0; while kill -0 \"$pid\" 2>/dev/null; do "
        "n=$((n+1)); test \"$n\" -le 100; sleep 0.1; done"
    )
    return E.LifecycleCommand(
        "stop attempt-owned edge tensor capture",
        ("sudo", "-n", "docker", "exec", E.CONTAINER,
         "sh", "-c", script, "splitfusion-edge-capture", REMOTE_CAPTURE_PID),
    )


def remote_plan_evidence_sha256(plan: E.RemoteEdgeLifecyclePlan) -> str:
    """Stable identity the local release must echo before remote teardown."""
    return _canonical_sha256(plan.as_evidence())


def _validate_local_gt_release(
    path: Path, *, plan: E.RemoteEdgeLifecyclePlan,
) -> Mapping[str, Any]:
    document = _load_exact_json(path, "local GT release")
    expected = {
        "schema", "local_gt_sender_closed", "local_gt_sender_final_sha256",
        "radio_tensor_path_sha256", "remote_teardown_owner",
        "remote_cn_teardown_permitted_to_local_coordinator",
        "remote_attempt_id", "remote_project_name", "remote_plan_sha256",
    }
    _require(set(document) == expected, "local GT release fields drift")
    _require(document["schema"] == LOCAL_GT_RELEASE_SCHEMA,
             "local GT release schema drift")
    _require(document["local_gt_sender_closed"] is True,
             "local GT sender is not proven closed")
    for name in ("local_gt_sender_final_sha256", "radio_tensor_path_sha256"):
        value = document[name]
        _require(type(value) is str and len(value) == 64
                 and all(c in "0123456789abcdef" for c in value),
                 f"local GT release {name} is invalid")
    _require(document["remote_teardown_owner"] == "REMOTE_LIFECYCLE_OWNER_ONLY",
             "remote teardown ownership drift")
    _require(document["remote_cn_teardown_permitted_to_local_coordinator"] is False,
             "local coordinator may not own remote CN teardown")
    _require(document["remote_attempt_id"] == plan.invocation.attempt_id,
             "local GT release belongs to another remote attempt")
    _require(document["remote_project_name"] == plan.invocation.project_name,
             "local GT release belongs to another Compose project")
    _require(document["remote_plan_sha256"] == remote_plan_evidence_sha256(plan),
             "local GT release belongs to another remote plan")
    return document


def _validate_local_gt_abort_release(
    path: Path, *, plan: E.RemoteEdgeLifecyclePlan,
) -> Mapping[str, Any]:
    document = _load_exact_json(path, "local GT abort release")
    expected = {
        "schema", "local_gt_sender_was_connected", "local_gt_sender_closed",
        "local_gt_sender_final_sha256", "remote_attempt_id",
        "remote_project_name", "remote_plan_sha256",
        "remote_cn_teardown_permitted_to_local_coordinator",
        "scientific_pass",
    }
    _require(set(document) == expected, "local GT abort release fields drift")
    _require(document["schema"] == LOCAL_GT_ABORT_RELEASE_SCHEMA,
             "local GT abort release schema drift")
    _require(document["local_gt_sender_was_connected"] is True
             and document["local_gt_sender_closed"] is True,
             "connected local GT sender is not proven closed")
    digest = document["local_gt_sender_final_sha256"]
    _require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest),
             "local GT abort final digest is invalid")
    _require(document["remote_attempt_id"] == plan.invocation.attempt_id
             and document["remote_project_name"] == plan.invocation.project_name,
             "local GT abort release identity drift")
    _require(document["remote_plan_sha256"] == remote_plan_evidence_sha256(plan),
             "local GT abort release plan drift")
    _require(document["remote_cn_teardown_permitted_to_local_coordinator"] is False,
             "local abort release grants remote CN teardown")
    _require(document["scientific_pass"] is False,
             "abort release cannot make a scientific pass")
    return document


def _validate_capture_seal(
    document: Mapping[str, Any], *, plan: E.RemoteEdgeLifecyclePlan,
) -> Mapping[str, Any]:
    required = {
        "schema", "operation", "status", "attempt_id", "project_name",
        "container_id", "plan_sha256", "pcap_relative", "pcap_sha256",
        "pcap_bytes", "filter", "edge_and_gt_left_running",
        "remote_cn_owned", "commands",
    }
    _require(type(document) is dict and set(document) == required,
             "held capture seal fields drift")
    _require(document["schema"] == SCHEMA
             and document["operation"] == "seal-capture"
             and document["status"] == "CAPTURE_SEALED_EDGE_STILL_HELD",
             "held capture seal status drift")
    _require(document["attempt_id"] == plan.invocation.attempt_id
             and document["project_name"] == plan.invocation.project_name,
             "held capture seal identity drift")
    _require(document["plan_sha256"] == remote_plan_evidence_sha256(plan),
             "held capture seal plan drift")
    _require(document["pcap_relative"] == str(Path("evidence") / CAPTURE_NAME)
             and document["filter"] == CAPTURE_FILTER,
             "held capture seal capture contract drift")
    _require(type(document["pcap_bytes"]) is int and document["pcap_bytes"] >= 24,
             "held capture seal byte count is invalid")
    _require(type(document["pcap_sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", document["pcap_sha256"]) is not None,
             "held capture seal digest is invalid")
    _require(document["edge_and_gt_left_running"] is True
             and document["remote_cn_owned"] is False,
             "held capture seal ownership drift")
    return dict(document)


def _container_observation(
    durable: DurableAttempt, *, runner: Q.CommandRunner,
    ledger: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    command = durable.prepared.plan.post_create[0]
    result = Q._run_checked(command=command, runner=runner, ledger=ledger,
                            timeout_s=20.0)
    observation = Q.container_observation_from_inspect(result.stdout)
    E.validate_container_observation(observation, plan=durable.prepared.plan)
    return observation


def _recoverable_prior_stop(
    attempt_root: Path, *, plan: E.RemoteEdgeLifecyclePlan,
) -> Optional[Mapping[str, Any]]:
    """Return a bound non-success STOP, while making a clean STOP terminal."""
    path = Path(attempt_root) / STOP_NAME
    if not path.exists():
        return None
    document = _load_exact_json(path, "held-session stop record")
    _require(document.get("schema") == SCHEMA
             and document.get("operation") == "stop",
             "held-session stop record identity drift")
    _require(document.get("attempt_id") == plan.invocation.attempt_id
             and document.get("project_name") == plan.invocation.project_name,
             "held-session stop record belongs to another project")
    _require(document.get("remote_cn_owned") is False,
             "held-session stop record claims remote CN ownership")
    status = document.get("status")
    _require(type(status) is str and bool(status),
             "held-session stop status is invalid")
    _require(status != "STOPPED_PROJECT_ONLY_EVIDENCE_READY",
             "successful held-session stop already exists")
    _require(status == "FAILED",
             "held-session stop status is not a recognized non-success")
    return document


@dataclass
class RemoteEdgeHeldSessionV1:
    """Durable operations; no correctness depends on process-local state."""

    runner: Q.CommandRunner = Q.subprocess_runner
    image_observer: Q.ImageObserver = Q.image_observation_from_inspect
    container_observer: Q.ContainerObserver = Q.container_observation_from_inspect

    def start(
        self, *, repository_root: Path, attempt_root: Path, attempt_id: str,
        run_id: str, cell_id: str, timeout_s: float = DEFAULT_READY_TIMEOUT_S,
        hash_file: Callable[[Path], str] = Q._sha256_file,
        lock_timeout_s: float = DEFAULT_START_ABORT_LOCK_TIMEOUT_S,
    ) -> Mapping[str, Any]:
        with _start_abort_lock(attempt_root, timeout_s=lock_timeout_s):
            return self._start_locked(
                repository_root=repository_root, attempt_root=attempt_root,
                attempt_id=attempt_id, run_id=run_id, cell_id=cell_id,
                timeout_s=timeout_s, hash_file=hash_file)

    def _start_locked(
        self, *, repository_root: Path, attempt_root: Path, attempt_id: str,
        run_id: str, cell_id: str, timeout_s: float,
        hash_file: Callable[[Path], str],
    ) -> Mapping[str, Any]:
        _require(1.0 <= float(timeout_s) <= 600.0,
                 "readiness timeout is outside [1,600]")
        prepared = Q.prepare_attempt(
            repository_root=repository_root, attempt_root=attempt_root,
            attempt_id=attempt_id, run_id=run_id, cell_id=cell_id,
            hash_file=hash_file,
        )
        intent = _intent_document(prepared)
        _write_create_only(prepared.plan.paths.attempt_root / INTENT_NAME, intent)
        durable = DurableAttempt(prepared=prepared, intent=intent)
        plan = prepared.plan
        ledger: list[Mapping[str, Any]] = []
        report: dict[str, Any] = {
            "schema": SCHEMA,
            "operation": "start",
            "status": "FAILED",
            "error": "",
            "secondary_errors": [],
            "attempt_id": attempt_id,
            "project_name": plan.invocation.project_name,
            "remote_cn_owned": False,
            "commands": ledger,
            "cleanup": {},
        }
        launch_attempted = False
        ready_achieved = False
        capture_started = False
        try:
            Q.verify_registered_artifacts(repository_root, hash_file=hash_file)
            for command in plan.preflight:
                result = Q._run_checked(
                    command=command, runner=self.runner, ledger=ledger,
                    timeout_s=30.0)
                if command.purpose == "verify remote hostname":
                    _require(result.stdout.strip() == plan.binding.hostname,
                             "remote hostname drift")
                elif command.purpose == "measure the bound GPU":
                    GPU.validate_measured_gpu(
                        GPU.parse_nvidia_smi_row(result.stdout), plan.binding)
                elif command.purpose == "inspect portable edge image":
                    image = self.image_observer(result.stdout, plan.binding)
                    E.validate_remote_image_observation(image)
                    _write_create_only(plan.paths.attempt_root / Q.IMAGE_NAME, image)

            launch_attempted = True
            Q._run_checked(command=plan.launch, runner=self.runner, ledger=ledger,
                           timeout_s=60.0)
            inspected = Q._run_checked(
                command=plan.post_create[0], runner=self.runner, ledger=ledger,
                timeout_s=20.0)
            container = self.container_observer(inspected.stdout)
            E.validate_container_observation(container, plan=plan)
            _write_create_only(plan.paths.attempt_root / Q.CONTAINER_NAME, container)
            gt_ready, edge_ready = Q._wait_for_ready(
                prepared, runner=self.runner, ledger=ledger,
                timeout_s=float(timeout_s))
            ready_achieved = True

            route_command = feedback_route_command()
            route_result = Q._run_checked(
                command=route_command, runner=self.runner, ledger=ledger,
                timeout_s=20.0)
            route = _feedback_route_observation(
                route_result.stdout, returncode=route_result.returncode)
            _write_create_only(plan.paths.attempt_root / FEEDBACK_ROUTE_NAME, route)

            prestarted = COORD.validate_prestarted_remote_edge(
                plan=plan, container_observation=container,
                ready_record=edge_ready, gt_ready_record=gt_ready,
                feedback_route=route)

            Q._run_checked(command=capture_start_command(), runner=self.runner,
                           ledger=ledger, timeout_s=20.0)
            capture_started = True
            Q._run_checked(command=capture_health_command(), runner=self.runner,
                           ledger=ledger, timeout_s=20.0)
            report.update({
                "status": "HELD_READY_CAPTURE_ACTIVE",
                "prestarted_remote_edge": {
                    "plan_sha256": prestarted.plan_sha256,
                    "attempt_id": prestarted.attempt_id,
                    "project_name": prestarted.project_name,
                    "container_id": prestarted.container_id,
                    "ready_sha256": prestarted.ready_sha256,
                    "gt_ready_sha256": prestarted.gt_ready_sha256,
                    "feedback_route_sha256": prestarted.feedback_route_sha256,
                },
                "gt_ready_sha256": _sha256_file(
                    plan.paths.state_root / Path(E.GT_READY_DESTINATION).name),
                "edge_ready_sha256": _sha256_file(
                    plan.paths.state_root / Path(E.READY_DESTINATION).name),
                "feedback_route_sha256": _sha256_file(
                    plan.paths.attempt_root / FEEDBACK_ROUTE_NAME),
                "image_observation_sha256": _sha256_file(
                    plan.paths.attempt_root / Q.IMAGE_NAME),
                "container_observation_sha256": _sha256_file(
                    plan.paths.attempt_root / Q.CONTAINER_NAME),
                "gt_endpoint": gt_ready["advertised_endpoint"],
                "edge_architecture": edge_ready["architecture"],
                "capture": {
                    "interface": "eth0", "filter": CAPTURE_FILTER,
                    "remote_pcap": REMOTE_PCAP,
                },
            })
        except BaseException as exc:
            _record_primary(report, exc)
            self._cleanup_failed_start(
                durable, report=report, ledger=ledger,
                launch_attempted=launch_attempted,
                ready_achieved=ready_achieved,
                capture_started=capture_started,
            )
        _write_create_only(plan.paths.attempt_root / START_NAME, report)
        return report

    def _cleanup_failed_start(
        self, durable: DurableAttempt, *, report: dict[str, Any],
        ledger: list[Mapping[str, Any]], launch_attempted: bool,
        ready_achieved: bool, capture_started: bool,
    ) -> None:
        plan = durable.prepared.plan
        if not launch_attempted:
            report["cleanup"]["not_launched"] = True
            return
        if capture_started:
            try:
                Q._run_checked(command=capture_stop_command(), runner=self.runner,
                               ledger=ledger, timeout_s=20.0)
                report["cleanup"]["capture_stopped"] = True
            except BaseException as exc:
                report["cleanup"]["capture_stopped"] = False
                _record_primary(report, exc)
        try:
            report["edge_log_capture"] = Q._capture_container_logs(
                command=plan.post_create[1], runner=self.runner, ledger=ledger,
                target=plan.paths.attempt_root / Q.LOG_NAME)
            if report["edge_log_capture"].get("status") != "CAPTURED":
                _record_primary(report, "timestamped edge log capture failed")
        except BaseException as exc:
            _record_primary(report, exc)
        try:
            Q._run_checked(command=plan.teardown, runner=self.runner,
                           ledger=ledger, timeout_s=60.0)
            absent = E.LifecycleCommand(
                "verify attempt edge container absent after teardown",
                ("sudo", "-n", "docker", "container", "inspect", E.CONTAINER),
                expected_returncode=1)
            Q._run_checked(command=absent, runner=self.runner, ledger=ledger,
                           timeout_s=20.0)
            report["cleanup"]["container_absent"] = True
            if ready_achieved:
                final_path = (plan.paths.state_root
                              / Path(E.GT_FINAL_DESTINATION).name)
                final = _load_exact_json(final_path, "GT FINAL record")
                E.validate_gt_final_record(final, plan=plan)
                report["cleanup"]["gt_listener_stopped"] = True
        except BaseException as exc:
            report["cleanup"].setdefault("container_absent", False)
            _record_primary(report, exc)

    def capture(self, *, attempt_root: Path) -> Mapping[str, Any]:
        durable = _rehydrate(attempt_root)
        start = _load_exact_json(Path(attempt_root) / START_NAME,
                                 "held-session start record")
        _require(start.get("status") == "HELD_READY_CAPTURE_ACTIVE",
                 "held session did not reach READY")
        _require(not (Path(attempt_root) / STOP_NAME).exists(),
                 "held session is already stopped")
        _require(not (Path(attempt_root) / CAPTURE_SEALED_NAME).exists(),
                 "held capture is already sealed")
        ledger: list[Mapping[str, Any]] = []
        _container_observation(durable, runner=self.runner, ledger=ledger)
        Q._run_checked(command=capture_health_command(), runner=self.runner,
                       ledger=ledger, timeout_s=20.0)
        return {
            "schema": SCHEMA, "operation": "capture",
            "status": "HELD_CAPTURE_ACTIVE",
            "attempt_id": durable.prepared.plan.invocation.attempt_id,
            "project_name": durable.prepared.plan.invocation.project_name,
            "interface": "eth0", "filter": CAPTURE_FILTER,
            "commands": ledger,
        }

    def seal_capture(self, *, attempt_root: Path) -> Mapping[str, Any]:
        """Flush/hash the pcap while deliberately leaving edge and GT alive."""
        durable = _rehydrate(attempt_root)
        plan = durable.prepared.plan
        start = _load_exact_json(Path(attempt_root) / START_NAME,
                                 "held-session start record")
        _require(start.get("status") == "HELD_READY_CAPTURE_ACTIVE",
                 "held session did not reach READY")
        _require(not (Path(attempt_root) / STOP_NAME).exists(),
                 "held session is already stopped")
        target = Path(attempt_root) / CAPTURE_SEALED_NAME
        _require(not target.exists(), "held capture already has a seal record")
        ledger: list[Mapping[str, Any]] = []
        container = _container_observation(
            durable, runner=self.runner, ledger=ledger)
        Q._run_checked(command=capture_health_command(), runner=self.runner,
                       ledger=ledger, timeout_s=20.0)
        Q._run_checked(command=capture_stop_command(), runner=self.runner,
                       ledger=ledger, timeout_s=20.0)
        pcap = plan.paths.evidence_root / CAPTURE_NAME
        pcap_sha256, pcap_bytes = _pcap_identity(pcap)
        record = {
            "schema": SCHEMA, "operation": "seal-capture",
            "status": "CAPTURE_SEALED_EDGE_STILL_HELD",
            "attempt_id": plan.invocation.attempt_id,
            "project_name": plan.invocation.project_name,
            "container_id": container["container_id"],
            "plan_sha256": remote_plan_evidence_sha256(plan),
            "pcap_relative": str(Path("evidence") / CAPTURE_NAME),
            "pcap_sha256": pcap_sha256,
            "pcap_bytes": pcap_bytes,
            "filter": CAPTURE_FILTER,
            "edge_and_gt_left_running": True,
            "remote_cn_owned": False,
            "commands": ledger,
        }
        _validate_capture_seal(record, plan=plan)
        _write_create_only(target, record)
        return record

    def retrieve_capture(
        self, *, attempt_root: Path, destination_root: Path,
    ) -> Mapping[str, Any]:
        """Copy only the sealed capture while the remote edge remains held."""
        durable = _rehydrate(attempt_root)
        plan = durable.prepared.plan
        seal_path = Path(attempt_root) / CAPTURE_SEALED_NAME
        seal = _load_exact_json(seal_path, "held capture seal")
        _validate_capture_seal(seal, plan=plan)
        _require(not (Path(attempt_root) / STOP_NAME).exists(),
                 "capture-only retrieval is only for a still-held edge")
        pcap = plan.paths.evidence_root / CAPTURE_NAME
        _require(pcap.is_file() and pcap.stat().st_size == seal["pcap_bytes"]
                 and _sha256_file(pcap) == seal["pcap_sha256"],
                 "sealed pcap changed before retrieval")
        ledger: list[Mapping[str, Any]] = []
        _container_observation(durable, runner=self.runner, ledger=ledger)
        destination_root = Path(destination_root).resolve(strict=False)
        _require(not destination_root.exists(),
                 "capture retrieval destination already exists")
        destination_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        target = destination_root / CAPTURE_NAME
        with pcap.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        _require(_sha256_file(target) == seal["pcap_sha256"],
                 "capture retrieval pcap hash drift")
        record = {
            "schema": SCHEMA, "operation": "retrieve-capture",
            "status": "CAPTURE_RETRIEVED_EDGE_STILL_HELD",
            "attempt_id": plan.invocation.attempt_id,
            "project_name": plan.invocation.project_name,
            "pcap": CAPTURE_NAME,
            "pcap_sha256": seal["pcap_sha256"],
            "pcap_bytes": seal["pcap_bytes"],
            "capture_seal_sha256": _sha256_file(seal_path),
            "edge_and_gt_left_running": True,
            "remote_cn_owned": False,
            "commands": ledger,
        }
        _write_create_only(destination_root / CAPTURE_RETRIEVAL_MANIFEST_NAME,
                           record)
        return record

    def stop(
        self, *, attempt_root: Path, local_gt_release: Path,
    ) -> Mapping[str, Any]:
        durable = _rehydrate(attempt_root)
        plan = durable.prepared.plan
        start = _load_exact_json(Path(attempt_root) / START_NAME,
                                 "held-session start record")
        _require(start.get("status") == "HELD_READY_CAPTURE_ACTIVE",
                 "held session did not reach READY")
        _require(not (Path(attempt_root) / STOP_NAME).exists(),
                 "held session already has a stop record")

        seal_path = Path(attempt_root) / CAPTURE_SEALED_NAME
        seal = _load_exact_json(seal_path, "held capture seal")
        _validate_capture_seal(seal, plan=plan)
        pcap = plan.paths.evidence_root / CAPTURE_NAME
        _require(pcap.is_file() and pcap.stat().st_size == seal["pcap_bytes"]
                 and _sha256_file(pcap) == seal["pcap_sha256"],
                 "sealed pcap changed before teardown release")

        # This gate deliberately precedes every remote mutation.  Without it,
        # teardown could race a still-connected local GT sender.
        release = _validate_local_gt_release(local_gt_release, plan=plan)
        release_sha = _sha256_file(local_gt_release)
        ledger: list[Mapping[str, Any]] = []
        report: dict[str, Any] = {
            "schema": SCHEMA, "operation": "stop", "status": "FAILED",
            "error": "", "secondary_errors": [],
            "attempt_id": plan.invocation.attempt_id,
            "project_name": plan.invocation.project_name,
            "local_gt_release_sha256": release_sha,
            "local_gt_sender_final_sha256":
                release["local_gt_sender_final_sha256"],
            "radio_tensor_path_sha256": release["radio_tensor_path_sha256"],
            "remote_cn_owned": False,
            "commands": ledger, "cleanup": {},
        }

        # Never send signals or Compose-down until the live container is
        # re-proven to belong to this exact attempt/project/image/mount set.
        try:
            _container_observation(durable, runner=self.runner, ledger=ledger)
        except BaseException as exc:
            _record_primary(report, exc)
            report["cleanup"]["refused_foreign_or_unproven_container"] = True
            _write_create_only(Path(attempt_root) / STOP_NAME, report)
            return report

        report["cleanup"]["capture_stopped"] = True
        report["capture_seal_sha256"] = _sha256_file(seal_path)

        try:
            report["edge_log_capture"] = Q._capture_container_logs(
                command=plan.post_create[1], runner=self.runner, ledger=ledger,
                target=plan.paths.attempt_root / Q.LOG_NAME)
            if report["edge_log_capture"].get("status") != "CAPTURED":
                _record_primary(report, "timestamped edge log capture failed")
        except BaseException as exc:
            _record_primary(report, exc)

        try:
            Q._run_checked(command=plan.teardown, runner=self.runner,
                           ledger=ledger, timeout_s=60.0)
            report["cleanup"]["project_teardown_complete"] = True
        except BaseException as exc:
            report["cleanup"]["project_teardown_complete"] = False
            _record_primary(report, exc)

        try:
            absent = E.LifecycleCommand(
                "verify attempt edge container absent after teardown",
                ("sudo", "-n", "docker", "container", "inspect", E.CONTAINER),
                expected_returncode=1)
            Q._run_checked(command=absent, runner=self.runner, ledger=ledger,
                           timeout_s=20.0)
            report["cleanup"]["container_absent"] = True
        except BaseException as exc:
            report["cleanup"]["container_absent"] = False
            _record_primary(report, exc)

        try:
            final_path = plan.paths.state_root / Path(E.GT_FINAL_DESTINATION).name
            final = _load_exact_json(final_path, "GT FINAL record")
            E.validate_gt_final_record(final, plan=plan)
            report["cleanup"]["gt_listener_stopped"] = True
            report["gt_final_sha256"] = _sha256_file(final_path)
        except BaseException as exc:
            report["cleanup"]["gt_listener_stopped"] = False
            _record_primary(report, exc)

        try:
            _require(pcap.is_file(), "edge tensor pcap is absent")
            _require(pcap.stat().st_size == seal["pcap_bytes"]
                     and _sha256_file(pcap) == seal["pcap_sha256"],
                     "sealed pcap changed during teardown")
            report["pcap_sha256"] = _sha256_file(pcap)
            report["pcap_bytes"] = pcap.stat().st_size
        except BaseException as exc:
            _record_primary(report, exc)

        if (not report["error"]
                and report["cleanup"].get("capture_stopped") is True
                and report["cleanup"].get("project_teardown_complete") is True
                and report["cleanup"].get("container_absent") is True
                and report["cleanup"].get("gt_listener_stopped") is True):
            report["status"] = "STOPPED_PROJECT_ONLY_EVIDENCE_READY"
        _write_create_only(Path(attempt_root) / STOP_NAME, report)
        return report

    def abort(
        self, *, attempt_root: Path, primary_failure: str,
        local_gt_sender_connected: bool,
        local_gt_abort_release: Optional[Path] = None,
        lock_timeout_s: float = DEFAULT_START_ABORT_LOCK_TIMEOUT_S,
    ) -> Mapping[str, Any]:
        """Clean exact owned resources without ever making a science PASS."""
        with _start_abort_lock(attempt_root, timeout_s=lock_timeout_s):
            return self._abort_locked(
                attempt_root=attempt_root, primary_failure=primary_failure,
                local_gt_sender_connected=local_gt_sender_connected,
                local_gt_abort_release=local_gt_abort_release)

    def _abort_locked(
        self, *, attempt_root: Path, primary_failure: str,
        local_gt_sender_connected: bool,
        local_gt_abort_release: Optional[Path],
    ) -> Mapping[str, Any]:
        _require(type(primary_failure) is str and bool(primary_failure.strip()),
                 "abort primary failure is empty")
        _require(type(local_gt_sender_connected) is bool,
                 "abort GT-connected fact must be exact bool")
        durable = _rehydrate(attempt_root)
        plan = durable.prepared.plan
        start_path = Path(attempt_root) / START_NAME
        start: Optional[Mapping[str, Any]] = None
        if start_path.exists():
            start = _load_exact_json(start_path, "held-session start record")
            _require(start.get("schema") == SCHEMA
                     and start.get("operation") == "start"
                     and start.get("attempt_id") == plan.invocation.attempt_id
                     and start.get("project_name") == plan.invocation.project_name
                     and start.get("remote_cn_owned") is False,
                     "held-session start record identity drift")
            _require(start.get("status") in {
                "HELD_READY_CAPTURE_ACTIVE", "FAILED"},
                "held-session start status is not recoverable")
        prior_stop = _recoverable_prior_stop(Path(attempt_root), plan=plan)
        pre_success_start_recovery = (
            start is None or start.get("status") != "HELD_READY_CAPTURE_ACTIVE")
        target = Path(attempt_root) / ABORT_NAME
        _require(not target.exists(), "held session already has an abort record")
        if local_gt_sender_connected:
            _require(local_gt_abort_release is not None,
                     "connected local GT sender lacks abort release")
            release = _validate_local_gt_abort_release(
                Path(local_gt_abort_release), plan=plan)
            release_sha: Optional[str] = _sha256_file(Path(local_gt_abort_release))
        else:
            _require(local_gt_abort_release is None,
                     "abort release supplied for a never-connected GT sender")
            release, release_sha = None, None

        ledger: list[Mapping[str, Any]] = []
        report: dict[str, Any] = {
            "schema": SCHEMA, "operation": "abort",
            "status": "ABORTED_WITH_CLEANUP_ERRORS",
            "primary_failure": primary_failure[:2000],
            "cleanup_errors": [],
            "attempt_id": plan.invocation.attempt_id,
            "project_name": plan.invocation.project_name,
            "local_gt_sender_connected": local_gt_sender_connected,
            "local_gt_abort_release_sha256": release_sha,
            "local_gt_sender_final_sha256": (
                release["local_gt_sender_final_sha256"] if release else None),
            "scientific_pass": False, "remote_cn_owned": False,
            "commands": ledger, "cleanup": {},
        }
        if prior_stop is not None:
            report["recovery_of_non_success_stop"] = {
                "status": prior_stop["status"],
                "error": prior_stop.get("error", ""),
                "sha256": _sha256_file(Path(attempt_root) / STOP_NAME),
            }
        container_absent_after_failed_stop = False
        try:
            container = _container_observation(
                durable, runner=self.runner, ledger=ledger)
        except BaseException as exc:
            # A failed STOP may have completed Compose-down even when its
            # response or a different evidence step failed.  Only in that
            # recovery case may exact absence replace a live ownership proof.
            # A foreign/unproven live container makes this probe return 0 and
            # remains a hard refusal with no signals or Compose mutation.
            if prior_stop is not None or pre_success_start_recovery:
                try:
                    absent = E.LifecycleCommand(
                        "verify attempt edge container already absent after failed stop",
                        ("sudo", "-n", "docker", "container", "inspect",
                         E.CONTAINER), expected_returncode=1)
                    Q._run_checked(command=absent, runner=self.runner,
                                   ledger=ledger, timeout_s=20.0)
                    container = None
                    container_absent_after_failed_stop = True
                    report["cleanup"]["container_already_absent"] = True
                except BaseException:
                    report["cleanup_errors"].append(_error(exc))
                    report["cleanup"][
                        "refused_foreign_or_unproven_container"] = True
                    _write_create_only(target, report)
                    return report
            else:
                report["cleanup_errors"].append(_error(exc))
                report["cleanup"]["refused_foreign_or_unproven_container"] = True
                _write_create_only(target, report)
                return report

        seal_path = Path(attempt_root) / CAPTURE_SEALED_NAME
        if seal_path.exists():
            try:
                _validate_capture_seal(
                    _load_exact_json(seal_path, "held capture seal"), plan=plan)
                report["cleanup"]["capture_stopped"] = True
            except BaseException as exc:
                report["cleanup_errors"].append(_error(exc))
        else:
            capture_pid = plan.paths.state_root / CAPTURE_PID_NAME
            if pre_success_start_recovery and not capture_pid.is_file():
                report["cleanup"]["capture_stopped"] = True
                report["cleanup"]["capture_not_started"] = True
            else:
                try:
                    Q._run_checked(command=capture_stop_command(), runner=self.runner,
                                   ledger=ledger, timeout_s=20.0)
                    report["cleanup"]["capture_stopped"] = True
                except BaseException as exc:
                    report["cleanup"]["capture_stopped"] = False
                    report["cleanup_errors"].append(_error(exc))
            try:
                pcap = plan.paths.evidence_root / CAPTURE_NAME
                if pcap.is_file() and container is None:
                    pcap_sha256, pcap_bytes = _pcap_identity(pcap)
                    report["cleanup"]["unsealed_capture_preserved"] = {
                        "sha256": pcap_sha256, "bytes": pcap_bytes,
                    }
                elif pcap.is_file():
                    pcap_sha256, pcap_bytes = _pcap_identity(pcap)
                    seal = {
                        "schema": SCHEMA, "operation": "seal-capture",
                        "status": "CAPTURE_SEALED_EDGE_STILL_HELD",
                        "attempt_id": plan.invocation.attempt_id,
                        "project_name": plan.invocation.project_name,
                        "container_id": container["container_id"],
                        "plan_sha256": remote_plan_evidence_sha256(plan),
                        "pcap_relative": str(Path("evidence") / CAPTURE_NAME),
                        "pcap_sha256": pcap_sha256,
                        "pcap_bytes": pcap_bytes,
                        "filter": CAPTURE_FILTER,
                        "edge_and_gt_left_running": True,
                        "remote_cn_owned": False, "commands": list(ledger),
                    }
                    _validate_capture_seal(seal, plan=plan)
                    _write_create_only(seal_path, seal)
            except BaseException as exc:
                report["cleanup"]["capture_stopped"] = False
                report["cleanup_errors"].append(_error(exc))

        log_target = plan.paths.attempt_root / Q.LOG_NAME
        if container_absent_after_failed_stop:
            report["edge_log_capture"] = {
                "status": "PRESERVED_OR_UNAVAILABLE_AFTER_PRIOR_STOP",
            }
        elif prior_stop is not None and log_target.is_file():
            report["edge_log_capture"] = {
                "status": "PRESERVED_FROM_PRIOR_STOP",
                "path": log_target.name,
                "bytes": log_target.stat().st_size,
                "sha256": _sha256_file(log_target),
            }
        else:
            try:
                report["edge_log_capture"] = Q._capture_container_logs(
                    command=plan.post_create[1], runner=self.runner, ledger=ledger,
                    target=log_target)
                if report["edge_log_capture"].get("status") != "CAPTURED":
                    report["cleanup_errors"].append(
                        "timestamped edge log capture failed")
            except BaseException as exc:
                report["cleanup_errors"].append(_error(exc))
        try:
            Q._run_checked(command=plan.teardown, runner=self.runner,
                           ledger=ledger, timeout_s=60.0)
            report["cleanup"]["project_teardown_complete"] = True
        except BaseException as exc:
            report["cleanup"]["project_teardown_complete"] = False
            report["cleanup_errors"].append(_error(exc))
        try:
            absent = E.LifecycleCommand(
                "verify attempt edge container absent after abort",
                ("sudo", "-n", "docker", "container", "inspect", E.CONTAINER),
                expected_returncode=1)
            Q._run_checked(command=absent, runner=self.runner, ledger=ledger,
                           timeout_s=20.0)
            report["cleanup"]["container_absent"] = True
        except BaseException as exc:
            report["cleanup"]["container_absent"] = False
            report["cleanup_errors"].append(_error(exc))
        try:
            final_path = plan.paths.state_root / Path(E.GT_FINAL_DESTINATION).name
            final = _load_exact_json(final_path, "GT FINAL record")
            E.validate_gt_final_record(final, plan=plan)
            report["cleanup"]["gt_listener_stopped"] = True
            report["gt_final_sha256"] = _sha256_file(final_path)
        except BaseException as exc:
            report["cleanup"]["gt_listener_stopped"] = False
            if (pre_success_start_recovery
                    and report["cleanup"].get("container_absent") is True):
                report["cleanup"]["gt_listener_quiescent_after_project_down"] = True
            else:
                report["cleanup_errors"].append(_error(exc))
        gt_cleanup_complete = (
            report["cleanup"].get("gt_listener_stopped") is True
            or report["cleanup"].get(
                "gt_listener_quiescent_after_project_down") is True)
        if (not report["cleanup_errors"]
                and report["cleanup"].get("capture_stopped") is True
                and report["cleanup"].get("project_teardown_complete") is True
                and report["cleanup"].get("container_absent") is True
                and gt_cleanup_complete):
            report["status"] = "ABORTED_CLEANLY"
        _write_create_only(target, report)
        return report

    def retrieve(
        self, *, attempt_root: Path, destination_root: Path,
    ) -> Mapping[str, Any]:
        durable = _rehydrate(attempt_root)
        stop = _load_exact_json(Path(attempt_root) / STOP_NAME,
                                "held-session stop record")
        _require(stop.get("status") == "STOPPED_PROJECT_ONLY_EVIDENCE_READY",
                 "remote evidence is not sealed after clean stop")
        source_root = durable.prepared.plan.paths.attempt_root
        destination_root = Path(destination_root).resolve(strict=False)
        _require(not destination_root.exists(), "retrieval destination already exists")
        destination_root.mkdir(mode=0o700, parents=False, exist_ok=False)
        copied: list[Mapping[str, Any]] = []
        excluded: list[str] = []
        try:
            for source in sorted(source_root.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(source_root)
                if relative.parts[:2] == ("state", "hub"):
                    excluded.append(str(relative))
                    continue
                target = destination_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1 << 20)
                    writer.flush()
                    os.fsync(writer.fileno())
                copied.append({
                    "path": str(relative), "bytes": target.stat().st_size,
                    "sha256": _sha256_file(target),
                })
            pcap_rel = str(Path("evidence") / CAPTURE_NAME)
            _require(any(row["path"] == pcap_rel for row in copied),
                     "retrieval omitted the edge tensor pcap")
            manifest = {
                "schema": SCHEMA, "operation": "retrieve",
                "status": "RETRIEVED_CREATE_ONLY",
                "attempt_id": durable.prepared.plan.invocation.attempt_id,
                "project_name": durable.prepared.plan.invocation.project_name,
                "source_root": str(source_root),
                "destination_root": str(destination_root),
                "files": copied,
                "excluded_seeded_cache": excluded,
                "remote_cn_owned": False,
            }
            _write_create_only(destination_root / RETRIEVAL_MANIFEST_NAME, manifest)
            return manifest
        except BaseException:
            # The destination remains as forensic partial evidence.  Never
            # delete a create-only target after a failed retrieval.
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--repository-root", type=Path, required=True)
    start.add_argument("--attempt-root", type=Path, required=True)
    start.add_argument("--attempt-id", required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--cell-id", required=True)
    start.add_argument("--ready-timeout-s", type=float,
                       default=DEFAULT_READY_TIMEOUT_S)
    start.add_argument("--execute", default="")

    capture = subparsers.add_parser("capture")
    capture.add_argument("--attempt-root", type=Path, required=True)
    capture.add_argument("--execute", default="")

    seal = subparsers.add_parser("seal-capture")
    seal.add_argument("--attempt-root", type=Path, required=True)
    seal.add_argument("--execute", default="")

    capture_retrieve = subparsers.add_parser("retrieve-capture")
    capture_retrieve.add_argument("--attempt-root", type=Path, required=True)
    capture_retrieve.add_argument("--destination-root", type=Path, required=True)
    capture_retrieve.add_argument("--execute", default="")

    stop = subparsers.add_parser("stop")
    stop.add_argument("--attempt-root", type=Path, required=True)
    stop.add_argument("--local-gt-release", type=Path, required=True)
    stop.add_argument("--execute", default="")

    abort = subparsers.add_parser("abort")
    abort.add_argument("--attempt-root", type=Path, required=True)
    abort.add_argument("--primary-failure", required=True)
    abort.add_argument("--local-gt-sender-connected", action="store_true")
    abort.add_argument("--local-gt-abort-release", type=Path)
    abort.add_argument("--execute", default="")

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--attempt-root", type=Path, required=True)
    retrieve.add_argument("--destination-root", type=Path, required=True)
    retrieve.add_argument("--execute", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live CLI
    args = build_parser().parse_args(argv)
    _require(args.execute == EXECUTE_TOKEN, "held-session execute token is absent")
    owner = RemoteEdgeHeldSessionV1()
    if args.operation == "start":
        result = owner.start(
            repository_root=args.repository_root,
            attempt_root=args.attempt_root,
            attempt_id=args.attempt_id,
            run_id=args.run_id,
            cell_id=args.cell_id,
            timeout_s=args.ready_timeout_s,
        )
    elif args.operation == "capture":
        result = owner.capture(attempt_root=args.attempt_root)
    elif args.operation == "seal-capture":
        result = owner.seal_capture(attempt_root=args.attempt_root)
    elif args.operation == "retrieve-capture":
        result = owner.retrieve_capture(
            attempt_root=args.attempt_root,
            destination_root=args.destination_root,
        )
    elif args.operation == "stop":
        result = owner.stop(
            attempt_root=args.attempt_root,
            local_gt_release=args.local_gt_release,
        )
    elif args.operation == "abort":
        result = owner.abort(
            attempt_root=args.attempt_root,
            primary_failure=args.primary_failure,
            local_gt_sender_connected=args.local_gt_sender_connected,
            local_gt_abort_release=args.local_gt_abort_release,
        )
    else:
        result = owner.retrieve(
            attempt_root=args.attempt_root,
            destination_root=args.destination_root,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {
        "HELD_READY_CAPTURE_ACTIVE", "HELD_CAPTURE_ACTIVE",
        "CAPTURE_SEALED_EDGE_STILL_HELD",
        "CAPTURE_RETRIEVED_EDGE_STILL_HELD", "ABORTED_CLEANLY",
        "STOPPED_PROJECT_ONLY_EVIDENCE_READY", "RETRIEVED_CREATE_ONLY",
    } else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SCHEMA", "EXECUTE_TOKEN", "INTENT_NAME", "START_NAME", "STOP_NAME",
    "ABORT_NAME", "FEEDBACK_ROUTE_NAME", "CAPTURE_SEALED_NAME",
    "CAPTURE_NAME", "CAPTURE_FILTER", "CAPTURE_RETRIEVAL_MANIFEST_NAME",
    "RETRIEVAL_MANIFEST_NAME", "RemoteEdgeHeldSessionError",
    "RemoteEdgeHeldSessionV1", "feedback_route_command",
    "capture_start_command", "capture_health_command", "capture_stop_command",
    "remote_plan_evidence_sha256", "build_parser", "main",
]
