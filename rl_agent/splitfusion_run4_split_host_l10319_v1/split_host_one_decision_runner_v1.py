#!/usr/bin/env python3
"""Bounded one-decision Run-4 handshake across W10275 and L10319.

No scientific contract is changed. The remote CN is a prerequisite, never an
owned cleanup target. Policy timing remains on W10275 CLOCK_MONOTONIC_RAW;
remote packet timestamps are never exposed or subtracted from local ones.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from . import first_real_tensor_proof_v1 as TP
from . import local_ran_executor_v1 as LX
from . import local_ran_lifecycle_v1 as LR
from . import remote_edge_held_session_v1 as RH
from . import remote_edge_lifecycle_v1 as RE
from . import split_host_phase6_coordinator_v1 as CO

SCHEMA = "scenesense.run4.split_host_one_decision_runner.v1"
EXECUTE_TOKEN = "SPLITFUSION_RUN4_SPLIT_HOST_ONE_DECISION_V1_EXECUTE"
REMOTE_MODULE = ("rl_agent.splitfusion_run4_split_host_l10319_v1."
                 "remote_edge_held_session_v1")
REMOTE_HOST = "shr_aisvcs@L10319.idcc.lab"
REMOTE_REPOSITORY = Path("/home/shr_aisvcs/workarea/carla_0_10_env/"
                         "Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/"
                         "abiodun_run4_l10319")
REMOTE_ATTEMPT_BASE = Path("/home/shr_aisvcs/workarea/carla_0_10_env/"
                           "Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/"
                           "splitfusion_run4_split_host_l10319_v1_attempts")
REMOTE_CN_DIRECTORY = REMOTE_REPOSITORY / "OAI/oai-cn5g"
REMOTE_CN_COMPOSE = REMOTE_CN_DIRECTORY / "docker-compose.yaml"
REMOTE_CN_PROJECT = "oai-cn5g"
REMOTE_CN_SERVICES = ("oai-amf", "oai-smf", "oai-upf")
REMOTE_CN_HEALTH_TIMEOUT_S = 120.0
DEFAULT_CONFIG = (LR.ROOT / "rl_agent/configs/"
                  "splitfusion_direct_edge_map_live_validation_v1.json")
LOCAL_RADIO_STATE_BASE = (LR.ROOT / "experiments/"
                          "splitfusion_oai_100mhz_4d5u_v1")
TRANSMITTED_BUDGET, DECISION_CAP = 40, 1
SAFETY_TIMEOUT_S, OUTER_RUNTIME_S = 60.0, 600.0
REMOTE_START_TIMEOUT_S = 360.0
REMOTE_ABORT_TIMEOUT_S = 540.0
_require_start_coverage = (
    REMOTE_START_TIMEOUT_S + RH.DEFAULT_START_ABORT_LOCK_TIMEOUT_S)
if _require_start_coverage < RH.START_OPERATION_TIMEOUT_BUDGET_S:
    raise RuntimeError("start SSH plus abort lock cannot cover START owner")
_require_abort_timeout = (
    RH.DEFAULT_START_ABORT_LOCK_TIMEOUT_S + RH.ABORT_CLEANUP_TIMEOUT_BUDGET_S)
if REMOTE_ABORT_TIMEOUT_S <= _require_abort_timeout:
    raise RuntimeError("remote abort SSH timeout does not cover lock plus cleanup")
LOCAL_CAPTURE_FILTER = RH.CAPTURE_FILTER
RESULT_NAME = "SPLIT_HOST_ONE_DECISION_RESULT.json"
PROOF_NAME = "FIRST_REAL_TENSOR_PROOF.json"
RELEASE_NAME = "REMOTE_TEARDOWN_RELEASE.json"
ABORT_RELEASE_NAME = "REMOTE_ABORT_RELEASE.json"
REMOTE_CORE_RESET_NAME = "REMOTE_CORE_RESET_EVIDENCE.json"
REMOTE_CORE_RESET_SCHEMA = "scenesense.run4.remote_core_reset_evidence.v1"


class SplitHostOneDecisionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitHostOneDecisionError(message)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, indent=2,
                                default=str, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class OneDecisionPlanV1:
    output_root: Path
    run_id: str
    attempt_id: str
    remote_host: str = REMOTE_HOST
    remote_repository: Path = REMOTE_REPOSITORY
    config_path: Path = DEFAULT_CONFIG
    carla_port: int = 2000
    outer_runtime_s: float = OUTER_RUNTIME_S

    def validate(self) -> "OneDecisionPlanV1":
        supplied_output = Path(self.output_root)
        _require(supplied_output.is_absolute(), "output root must be absolute")
        output = supplied_output.resolve(strict=False)
        _require(str(output) not in {"/", "/tmp", "/home"}
                 and not output.exists(), "unsafe/non-create-only output root")
        _require(re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", self.run_id) is not None
                 and RE.IDENTITY_RE.fullmatch(self.run_id) is not None,
                 "unsafe run identity")
        _require(RE.ATTEMPT_RE.fullmatch(self.attempt_id) is not None,
                 "unsafe or remote-incompatible attempt identity")
        _require(self.remote_host == REMOTE_HOST, "remote SSH identity drift")
        _require(Path(self.remote_repository).is_absolute(), "remote root not absolute")
        _require(Path(self.remote_repository) == REMOTE_REPOSITORY,
                 "remote repository identity drift")
        _require(Path(self.config_path).is_absolute(), "config not absolute")
        _require(1024 <= self.carla_port <= 65535, "CARLA port invalid")
        _require(120 <= float(self.outer_runtime_s) <= OUTER_RUNTIME_S,
                 "outer timeout outside [120,600]")
        return self

    @property
    def remote_base(self) -> Path:
        return REMOTE_ATTEMPT_BASE

    @property
    def remote_attempt(self) -> Path:
        return self.remote_base / f"{self.attempt_id}_held_edge"

    @property
    def remote_capture(self) -> Path:
        return self.remote_base / f"{self.attempt_id}_capture_retrieval"

    @property
    def remote_evidence(self) -> Path:
        return self.remote_base / f"{self.attempt_id}_evidence_retrieval"

    @property
    def local_attempt(self) -> Path:
        return self.output_root / "attempt"

    @property
    def local_radio_state(self) -> Path:
        return LOCAL_RADIO_STATE_BASE / f"split_host_{self.attempt_id}"


@dataclass
class PreparedLocalV1:
    child_args: Any
    lifecycle: Any
    ran_plan: LR.LocalRanPlan
    retrieval: CO.RemoteEvidenceRetrievalPlanV1
    cold_check: Any


class RemoteCliV1:
    """Bounded SSH/SCP access to the durable held-session CLI only."""
    def __init__(self, host: str = REMOTE_HOST) -> None:
        _require(host == REMOTE_HOST, "foreign remote host")
        self.host = host

    def _ssh(self, argv: Sequence[str], timeout: float, data: bytes | None = None
             ) -> subprocess.CompletedProcess[Any]:
        command = " ".join(shlex.quote(item) for item in argv)
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
                               "ConnectTimeout=10", self.host, command],
                              input=data, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, check=False)

    def call(self, operation: str, args: Sequence[str], timeout: float, *,
             repository: Path) -> Mapping[str, Any]:
        inner = ["/usr/bin/env", "-u", "PYTHONPATH", "/usr/bin/python3", "-m",
                 REMOTE_MODULE, operation, *args, "--execute", RH.EXECUTE_TOKEN]
        command = ("cd " + shlex.quote(str(repository)) + " && exec "
                   + " ".join(shlex.quote(item) for item in inner))
        result = self._ssh(["/usr/bin/bash", "-lc", command], timeout)
        lines = [row for row in bytes(result.stdout or b"").decode(
            "utf-8", "replace").splitlines() if row.lstrip().startswith("{")]
        stderr = bytes(result.stderr or b"").decode("utf-8", "replace")[-1500:]
        _require(lines, f"remote {operation} emitted no JSON; stderr={stderr!r}")
        value = json.loads(lines[-1])
        _require(result.returncode == 0 and type(value) is dict,
                 f"remote {operation} failed: {value}")
        return value

    def ensure_attempt_parent(self, *, repository: Path, parent: Path) -> None:
        repository = Path(repository).resolve(strict=False)
        expected = REMOTE_ATTEMPT_BASE.resolve(strict=False)
        _require(parent == expected, "remote attempt parent identity drift")
        _require(repository != expected and repository not in expected.parents,
                 "remote attempt parent must be outside repository")
        result = self._ssh(["mkdir", "-p", "--", str(expected)], 30)
        _require(result.returncode == 0,
                 "remote exact attempt parent creation failed")

    def upload(self, local: Path, remote: Path) -> None:
        data = local.read_bytes()
        code = ("from pathlib import Path;import sys,os;p=Path(sys.argv[1]);"
                "p.parent.mkdir(parents=True,exist_ok=True);f=p.open('xb');"
                "f.write(sys.stdin.buffer.read());f.flush();os.fsync(f.fileno());f.close()")
        result = self._ssh(["python3", "-c", code, str(remote)], 30, data)
        _require(result.returncode == 0, "remote create-only upload failed")

    def download(self, remote: Path, local: Path, *, tree: bool = False) -> None:
        _require(not local.exists(), "download target exists")
        local.parent.mkdir(parents=True, exist_ok=True)
        argv = ["scp", "-q"] + (["-r"] if tree else []) + [
            f"{self.host}:{remote}", str(local)]
        result = subprocess.run(argv, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=180, check=False)
        _require(result.returncode == 0 and (local.is_dir() if tree else local.is_file()),
                 "remote download failed")


    def _inspect_core(self) -> dict[str, dict[str, Any]]:
        result = self._ssh(
            ["sudo", "-n", "docker", "inspect", *REMOTE_CN_SERVICES], 30)
        _require(result.returncode == 0, "remote core inspect failed")
        try:
            rows = json.loads(bytes(result.stdout or b"").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SplitHostOneDecisionError("remote core inspect was not JSON") from exc
        _require(type(rows) is list and len(rows) == len(REMOTE_CN_SERVICES),
                 "remote core inspect service count drift")
        states: dict[str, dict[str, Any]] = {}
        for row in rows:
            _require(type(row) is dict, "remote core inspect row invalid")
            name = str(row.get("Name", "")).removeprefix("/")
            _require(name in REMOTE_CN_SERVICES and name not in states,
                     "remote core service identity drift")
            config, state = row.get("Config"), row.get("State")
            _require(type(config) is dict and type(state) is dict,
                     f"remote core state absent: {name}")
            labels = config.get("Labels")
            _require(type(labels) is dict, f"remote core labels absent: {name}")
            _require(labels.get("com.docker.compose.project") == REMOTE_CN_PROJECT
                     and labels.get("com.docker.compose.service") == name,
                     f"remote core Compose identity drift: {name}")
            working_dir = labels.get("com.docker.compose.project.working_dir")
            config_files = labels.get("com.docker.compose.project.config_files")
            if working_dir is not None:
                _require(working_dir == str(REMOTE_CN_DIRECTORY),
                         f"remote core working-dir drift: {name}")
            if config_files is not None:
                _require(config_files == str(REMOTE_CN_COMPOSE),
                         f"remote core config-file drift: {name}")
            health = state.get("Health")
            states[name] = {
                "container_id": row.get("Id"),
                "started_at": state.get("StartedAt"),
                "running": state.get("Running"),
                "health": health.get("Status") if type(health) is dict else None,
            }
        _require(set(states) == set(REMOTE_CN_SERVICES),
                 "remote core exact service set drift")
        return states

    def restart_core_and_wait_healthy(
            self, *, repository: Path,
            health_timeout_s: float = REMOTE_CN_HEALTH_TIMEOUT_S,
            poll_interval_s: float = 1.0) -> Mapping[str, Any]:
        _require(Path(repository) == REMOTE_REPOSITORY,
                 "remote core repository identity drift")
        for path, kind in ((REMOTE_REPOSITORY, "-d"), (REMOTE_CN_COMPOSE, "-f")):
            present = self._ssh(["test", kind, str(path)], 15)
            resolved = self._ssh(["readlink", "-f", "--", str(path)], 15)
            actual = bytes(resolved.stdout or b"").decode("utf-8", "replace").strip()
            _require(present.returncode == 0 and resolved.returncode == 0
                     and actual == str(path), f"remote core path identity drift: {path}")
        before = self._inspect_core()
        command = ["sudo", "-n", "docker", "compose",
                   "--project-name", REMOTE_CN_PROJECT,
                   "--project-directory", str(REMOTE_CN_DIRECTORY),
                   "-f", str(REMOTE_CN_COMPOSE), "restart", *REMOTE_CN_SERVICES]
        restarted = self._ssh(command, 90)
        _require(restarted.returncode == 0, "remote core restart failed")
        deadline = time.monotonic() + float(health_timeout_s)
        while True:
            after = self._inspect_core()
            if all(after[name]["started_at"] not in {None, ""}
                   and after[name]["started_at"] != before[name]["started_at"]
                   and after[name]["running"] is True
                   and after[name]["health"] == "healthy"
                   for name in REMOTE_CN_SERVICES):
                break
            if time.monotonic() >= deadline:
                raise SplitHostOneDecisionError(
                    "remote core did not reach changed-StartedAt/running/healthy gate")
            time.sleep(float(poll_interval_s))
        return {"schema": REMOTE_CORE_RESET_SCHEMA,
                "status": "REMOTE_CORE_RESTARTED_HEALTHY",
                "remote_host": self.host, "repository": str(REMOTE_REPOSITORY),
                "compose_file": str(REMOTE_CN_COMPOSE),
                "compose_project": REMOTE_CN_PROJECT,
                "services": list(REMOTE_CN_SERVICES),
                "restart_command": command, "before": before, "after": after,
                "remote_cn_owned": False}


class SystemOpsV1:
    """Production effects; tests replace this object completely."""
    def __init__(self, remote: Optional[RemoteCliV1] = None) -> None:
        self.remote = remote or RemoteCliV1()

    def prepare(self, plan: OneDecisionPlanV1) -> PreparedLocalV1:
        from rl_agent import ue_288_campaign_supervisor as supervisor
        from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as P
        config, cells, _ = LP.offline_preflight(
            plan.config_path, action_ids=(71,), profile_ids=("FAVORABLE_STABLE",),
            transmitted_budget=TRANSMITTED_BUDGET,
            safety_timeout_s=SAFETY_TIMEOUT_S, output_root=None)
        _require(len(cells) == 1, "carrier cell count drift")
        selected = cells[0]
        campaign = LP._probe_campaign(config, run_id=plan.run_id)
        campaign["campaign_id"] = f"splitfusion_run4_phase6_v2/{plan.run_id}"
        supervisor._require_phase15_application_cold(campaign)
        cell = supervisor.Cell(
            cell_id=f"run4p6_{plan.run_id}_{selected.cell_id}",
            action_index=selected.action_index, action_id=selected.action_id,
            profile_id=selected.profile_id, model_family=selected.model_family,
            network_profile_id=selected.network_profile_id,
            trace_id=selected.trace_id, seed=selected.seed)
        attempt, service = plan.local_attempt, plan.output_root / "service"
        (attempt / "phase6_artifacts").mkdir(parents=True, exist_ok=False)
        (attempt / "ttracer" / "ue").mkdir(parents=True, exist_ok=False)
        service.mkdir(exist_ok=False)
        campaign_path, cell_path, binding_path = (
            service / "campaign.json", service / "cell.json", service / "bindings.json")
        _write(campaign_path, campaign)
        _write(cell_path, supervisor.cell_to_dict(cell))
        telemetry = P.telemetry_bindings()
        _write(binding_path, {"controller_lineage_sha256": P.controller_lineage_sha256(),
                              "tracer_dir": telemetry["tracer_dir"],
                              "t_messages": telemetry["t_messages"],
                              "ue_relay_port": telemetry["ue_relay_port"]})
        args = SimpleNamespace(
            campaign_json=str(campaign_path), cell_json=str(cell_path),
            attempt_dir=str(attempt), temporary_dir=str(service),
            artifacts_dir=str(attempt / "phase6_artifacts"),
            bindings_json=str(binding_path), carla_port=plan.carla_port,
            map_api_port=35001, spatial_map_port=39310, feedback_port=39401,
            transmitted_budget=TRANSMITTED_BUDGET,
            safety_timeout_s=SAFETY_TIMEOUT_S)
        ran = LR.build_local_ran_plan(plan.local_radio_state, root=LR.ROOT,
                                      raw_path=attempt / "ttracer/ue/ue.raw")
        retrieval = CO.RemoteEvidenceRetrievalPlanV1(
            schema=CO.REMOTE_RETRIEVAL_SCHEMA,
            remote_attempt_root=str(plan.remote_attempt),
            local_destination=str(plan.output_root / "remote_evidence"),
            required_relative_paths=("state/ready.json",
                "state/remote_gt_listener_ready.json",
                "state/remote_gt_listener_final.json",
                "evidence/run4_phase6_edge_report.json", "OUTPUT_MANIFEST.json"),
        ).validate()
        return PreparedLocalV1(args, supervisor.import_lifecycle_helper(config),
                               ran, retrieval,
                               lambda: supervisor._require_phase15_application_cold(campaign))

    def remote_start(self, plan: OneDecisionPlanV1) -> Mapping[str, Any]:
        self.remote.ensure_attempt_parent(
            repository=plan.remote_repository, parent=plan.remote_base)
        return self.remote.call("start", ("--repository-root", str(plan.remote_repository),
            "--attempt-root", str(plan.remote_attempt), "--attempt-id", plan.attempt_id,
            "--run-id", plan.run_id, "--cell-id", "a71__favorable_stable",
            "--ready-timeout-s", "300"), REMOTE_START_TIMEOUT_S,
            repository=plan.remote_repository)

    def restart_remote_core(self, plan: OneDecisionPlanV1) -> Mapping[str, Any]:
        evidence = dict(self.remote.restart_core_and_wait_healthy(
            repository=plan.remote_repository))
        _require(evidence.get("status") == "REMOTE_CORE_RESTARTED_HEALTHY"
                 and evidence.get("remote_cn_owned") is False,
                 "remote core reset evidence invalid")
        path = plan.output_root / REMOTE_CORE_RESET_NAME
        _write(path, evidence)
        return {"status": evidence["status"], "evidence_sha256": _sha(path),
                "remote_cn_owned": False}

    def start_ran(self, prepared: PreparedLocalV1, plan: OneDecisionPlanV1) -> Any:
        return LX.LocalRanExecutorV1(plan=prepared.ran_plan,
            attempt_dir=plan.output_root / "local_ran_executor").start()

    def start_carla(self, prepared: PreparedLocalV1, plan: OneDecisionPlanV1) -> Any:
        server, pgid = prepared.lifecycle.start_carla(
            plan.carla_port, plan.output_root / "carla_server.log")
        _require(prepared.lifecycle.wait_for_rpc(plan.carla_port, 180) is not None,
                 "CARLA RPC not ready")
        return (prepared.lifecycle, server, pgid, plan.carla_port)

    def stop_carla(self, handle: Any) -> Any:
        return handle[0].stop_carla(handle[1], handle[2], handle[3])

    def start_capture(self, plan: OneDecisionPlanV1) -> Any:
        path, log = plan.output_root / "ue_tensor.pcap", plan.output_root / "ue_tensor.log"
        stream = log.open("xb")
        process = subprocess.Popen(["sudo", "-n", "tcpdump", "-i",
            CO.POLICY_INTERFACE, "-nn", "-s", "0", "-U", "-w", str(path), LOCAL_CAPTURE_FILTER],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not path.exists():
            _require(process.poll() is None, "local capture exited")
            time.sleep(.05)
        _require(path.exists(), "local capture not ready")
        return (process, os.getpgid(process.pid), path, stream)

    def stop_capture(self, handle: Any) -> Path:
        process, pgid, path, stream = handle
        try:
            for name, timeout in (("INT", 5), ("TERM", 2), ("KILL", 2)):
                if process.poll() is not None:
                    break
                subprocess.run(["sudo", "-n", "kill", f"-{name}", "--", f"-{pgid}"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5, check=False)
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stream.close()
        owner = f"{os.getuid()}:{os.getgid()}"
        changed = subprocess.run(["sudo", "-n", "chown", owner, str(path)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, timeout=5, check=False)
        _require(changed.returncode == 0 and os.access(path, os.R_OK),
                 "local tensor capture is not caller-readable")
        _sha(path)
        _require(process.poll() is not None and path.stat().st_size >= 24,
                 "local capture did not flush")
        return path

    def context(self, pre: Any, retrieval: Any) -> Any:
        return CO.SplitHostPhase6ChildContextV1(
            prerequisites=pre, retrieval=retrieval, decision_cap=DECISION_CAP)

    def run_child(self, context: Any, prepared: PreparedLocalV1) -> int:
        return int(context.modules.child.run(prepared.child_args))

    def child_evidence(self, plan: OneDecisionPlanV1) -> tuple[dict, dict]:
        return (_read(plan.local_attempt / "phase6_artifacts/child_result.json"),
                _read(plan.local_attempt / "run4_phase6/PHASE6_UE_EVIDENCE.json"))

    def remote_seal(self, plan: OneDecisionPlanV1) -> Mapping[str, Any]:
        return self.remote.call("seal-capture", ("--attempt-root", str(plan.remote_attempt)), 60, repository=plan.remote_repository)

    def remote_capture(self, plan: OneDecisionPlanV1) -> Path:
        self.remote.call("retrieve-capture", ("--attempt-root", str(plan.remote_attempt),
            "--destination-root", str(plan.remote_capture)), 60, repository=plan.remote_repository)
        local = plan.output_root / "remote_tensor.pcap"
        self.remote.download(plan.remote_capture / RH.CAPTURE_NAME, local)
        return local

    def prove(self, plan: OneDecisionPlanV1, local: Path, remote: Path,
              ue: Mapping[str, Any]) -> Any:
        return TP.write_reconciled_proof(plan.output_root / PROOF_NAME,
            local_pcap=local, remote_pcap=remote, ue_evidence=ue)

    def upload_release(self, plan: OneDecisionPlanV1, path: Path, abort: bool) -> Path:
        remote = plan.remote_attempt / (ABORT_RELEASE_NAME if abort else RELEASE_NAME)
        self.remote.upload(path, remote)
        return remote

    def remote_stop(self, plan: OneDecisionPlanV1, release: Path) -> Mapping[str, Any]:
        return self.remote.call("stop", ("--attempt-root", str(plan.remote_attempt),
            "--local-gt-release", str(release)), 120, repository=plan.remote_repository)

    def remote_abort(self, plan: OneDecisionPlanV1, primary: str, connected: bool,
                     release: Optional[Path]) -> Mapping[str, Any]:
        args = ["--attempt-root", str(plan.remote_attempt), "--primary-failure", primary]
        if connected:
            _require(release is not None, "connected abort release absent")
            args += ["--local-gt-sender-connected", "--local-gt-abort-release", str(release)]
        return self.remote.call(
            "abort", args, REMOTE_ABORT_TIMEOUT_S,
            repository=plan.remote_repository)

    def remote_retrieve(self, plan: OneDecisionPlanV1) -> Mapping[str, Any]:
        result = self.remote.call("retrieve", ("--attempt-root", str(plan.remote_attempt),
            "--destination-root", str(plan.remote_evidence)), 120, repository=plan.remote_repository)
        self.remote.download(plan.remote_evidence, plan.output_root / "remote_evidence",
                             tree=True)
        return result


    def verify_cold(self, prepared: PreparedLocalV1) -> Any:
        return prepared.cold_check()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(type(value) is dict, f"non-object evidence: {path}")
    return value


def _prestarted(document: Mapping[str, Any]) -> CO.PrestartedRemoteEdgeV1:
    _require(document.get("status") == "HELD_READY_CAPTURE_ACTIVE",
             "remote edge not held READY")
    value = document.get("prestarted_remote_edge")
    _require(type(value) is dict, "remote start identity absent")
    try:
        return CO.PrestartedRemoteEdgeV1(**value).validate()
    except TypeError as exc:
        raise SplitHostOneDecisionError(f"remote start fields drift: {exc}") from exc


def validate_closed_one_decision(child: Mapping[str, Any],
                                 ue: Mapping[str, Any]) -> Mapping[str, Any]:
    """One actor call, one terminal, and at least one matching held tensor."""
    _require(child.get("return_code") == 0 and not child.get("error"),
             "frozen child failed")
    _require(child.get("stop_reason") == "DECISION_CYCLE_BOUNDARY",
             "child did not stop at decision-cycle boundary")
    collector = child.get("collector")
    _require(type(collector) is dict and collector.get("collector_cleanup_ok") is True
             and not collector.get("collector_failures"), "collector did not drain")
    sent = collector.get("transmitted_frames")
    _require(type(sent) is int and 2 <= sent <= TRANSMITTED_BUDGET,
             "decision plus hold were not both transmitted")
    counters = ue.get("counters")
    _require(type(counters) is dict and counters.get("policy_decisions") == 1
             and counters.get("actor_calls") == 1
             and int(counters.get("policy_holds", 0)) >= 1,
             "not exactly one actor decision plus hold")
    _require(ue.get("faulted") is None and ue.get("unresolved_tickets_at_close") == 0,
             "ticket unresolved or infrastructure faulted")
    _require(type(ue.get("resolutions")) is list and len(ue["resolutions"]) == 1,
             "not exactly one terminal resolution")
    _require(sum(row.get("reward_requested") is True for row in ue.get("frames") or ()) == 1,
             "not exactly one reward-requested frame")
    identities = [row["run4_identity"] for row in ue.get("transmitted_identities") or ()
                  if type(row.get("run4_identity")) is dict]
    decisions = [row for row in identities if row.get("frame_kind") == "POLICY_DECISION"]
    _require(len(decisions) == 1 and decisions[0].get("reward_requested") is True,
             "single transmitted decision identity absent")
    decision = decisions[0]
    holds = [row for row in identities if row.get("frame_kind") == "POLICY_HOLD"
             and (row.get("session_uuid"), row.get("decision_seq"), row.get("ticket_seq"))
             == (decision.get("session_uuid"), decision.get("decision_seq"),
                 decision.get("ticket_seq"))]
    _require(holds, "decision has no matching hold")
    return {"policy_decisions": 1, "actor_calls": 1, "resolutions": 1,
            "unresolved_tickets": 0, "transmitted_frames": sent,
            "policy_holds": len(holds), "stop_reason": "DECISION_CYCLE_BOUNDARY"}


class _Alarm:
    def __init__(self, seconds: float) -> None:
        self.seconds, self.previous = float(seconds), None

    def __enter__(self) -> "_Alarm":
        self.previous = signal.getsignal(signal.SIGALRM)
        def expired(_signum: int, _frame: Any) -> None:
            raise SplitHostOneDecisionError(
                f"split-host attempt exceeded {self.seconds:.0f} seconds")
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


class _NoAlarm:
    def __enter__(self) -> "_NoAlarm": return self
    def __exit__(self, *_args: Any) -> None: return None


def run_one_decision(plan: OneDecisionPlanV1, *, ops: Optional[Any] = None,
                     watchdog: bool = True) -> Mapping[str, Any]:
    """Execute one closed policy cycle; preserve primary and cleanup failures."""
    plan, ops = plan.validate(), (ops or SystemOpsV1())
    plan.output_root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": SCHEMA, "status": "FAILED", "error": "", "cleanup_errors": [],
        "run_id": plan.run_id, "attempt_id": plan.attempt_id,
        "decision_cap": DECISION_CAP, "transmitted_budget": TRANSMITTED_BUDGET,
        "safety_timeout_s": SAFETY_TIMEOUT_S,
        "policy_deadline_clock": "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
        "cross_host_monotonic_comparison_permitted": False,
        "remote_cn_owned": False,
    }
    prepared = ran = carla = capture = context = None
    remote_start_attempted = remote_held = False
    release_uploaded = remote_stopped = False
    uploaded_release: Optional[Path] = None

    def cleanup_failure(label: str, exc: BaseException) -> None:
        report["cleanup_errors"].append(f"{label}: {type(exc).__name__}: {exc}"[:4000])

    try:
        with (_Alarm(plan.outer_runtime_s) if watchdog else _NoAlarm()):
            prepared = ops.prepare(plan)
            remote_start_attempted = True
            start = ops.remote_start(plan)
            remote_held = start.get("status") == "HELD_READY_CAPTURE_ACTIVE"
            remote = _prestarted(start)
            core_reset = ops.restart_remote_core(plan)
            _require(core_reset.get("status") == "REMOTE_CORE_RESTARTED_HEALTHY"
                     and core_reset.get("remote_cn_owned") is False,
                     "remote core reset/readiness gate failed")
            report["remote_core_reset"] = dict(core_reset)
            ran = ops.start_ran(prepared, plan)
            _require(ran.source_route is not None and ran.policy_ownership is not None,
                     "local source-route ownership proof absent")
            pre = CO.SplitHostCoordinatorPreRunV1(
                remote=remote, source_route=ran.source_route,
                policy_ownership=ran.policy_ownership).validate()
            carla = ops.start_carla(prepared, plan)
            capture = ops.start_capture(plan)
            context = ops.context(pre, prepared.retrieval)
            context.__enter__()
            rc = ops.run_child(context, prepared)
            child, ue = ops.child_evidence(plan)
            _require(rc == 0, "frozen child returned nonzero")
            report["one_decision"] = dict(validate_closed_one_decision(child, ue))

            # Flush both observers while the exact edge is still held. The
            # scientific release is impossible until byte proof succeeds.
            local_pcap = ops.stop_capture(capture)
            capture = None
            seal = ops.remote_seal(plan)
            _require(seal.get("status") == "CAPTURE_SEALED_EDGE_STILL_HELD",
                     "remote capture seal failed")
            remote_pcap = ops.remote_capture(plan)
            proof, local_observation, remote_observation = ops.prove(
                plan, local_pcap, remote_pcap, ue)
            _require(proof.get("status") == "PASS"
                     and proof.get("cross_host_latency_computed") is False,
                     "first-real tensor proof failed")
            context.finalize_radio_tensor_path(local_observation, remote_observation)
            release = context.remote_teardown_release()
            release_path = plan.output_root / RELEASE_NAME
            _write(release_path, release)
            context.close()
            remote_release = ops.upload_release(plan, release_path, False)
            uploaded_release = remote_release
            release_uploaded = True
            stopped = ops.remote_stop(plan, remote_release)
            _require(stopped.get("status") == "STOPPED_PROJECT_ONLY_EVIDENCE_READY",
                     "remote exact-project stop failed")
            remote_stopped = True
            retrieved = ops.remote_retrieve(plan)
            _require(retrieved.get("status") == "RETRIEVED_CREATE_ONLY",
                     "remote evidence retrieval failed")
            report.update(status="PASS_ONE_DECISION_SPLIT_HOST",
                          proof_sha256=_sha(plan.output_root / PROOF_NAME),
                          release_sha256=_sha(release_path))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"[:4000]
    finally:
        if capture is not None:
            try: ops.stop_capture(capture)
            except BaseException as exc: cleanup_failure("local capture", exc)
        if context is not None:
            try: context.close()
            except BaseException as exc: cleanup_failure("child context", exc)
        if remote_start_attempted and not remote_stopped and not release_uploaded:
            try:
                connected = bool(context is not None and context.sender_ever_connected)
                abort = context.remote_abort_release() if context is not None else None
                remote_release = None
                if abort is not None:
                    local_release = plan.output_root / ABORT_RELEASE_NAME
                    _write(local_release, abort)
                    remote_release = ops.upload_release(plan, local_release, True)
                result = ops.remote_abort(plan, report["error"] or "cleanup failure",
                                          connected, remote_release)
                _require(result.get("status") == "ABORTED_CLEANLY",
                         "remote abort did not clean exact project")
                report["remote_abort_status"] = result["status"]
            except BaseException as exc: cleanup_failure("remote abort", exc)
        elif remote_held and release_uploaded and not remote_stopped:
            try:
                _require(uploaded_release is not None, "uploaded release path absent")
                stopped = ops.remote_stop(plan, uploaded_release)
                _require(stopped.get("status") == "STOPPED_PROJECT_ONLY_EVIDENCE_READY",
                         "proof-bound remote stop retry did not verify")
                remote_stopped = True
                report["remote_stop_recovered"] = True
            except BaseException as exc:
                cleanup_failure("remote stop retry", exc)
                try:
                    _require(context is not None, "abort context absent")
                    abort = context.remote_abort_release()
                    _require(abort is not None, "connected abort release absent")
                    local_abort = plan.output_root / ABORT_RELEASE_NAME
                    if not local_abort.exists():
                        _write(local_abort, abort)
                    remote_abort_release = ops.upload_release(plan, local_abort, True)
                    aborted = ops.remote_abort(plan, report["error"] or str(exc),
                                               True, remote_abort_release)
                    _require(aborted.get("status") == "ABORTED_CLEANLY",
                             "failed-STOP abort recovery did not verify")
                    report["remote_abort_status"] = aborted["status"]
                except BaseException as abort_exc:
                    cleanup_failure("failed-STOP abort recovery", abort_exc)
        if carla is not None:
            try:
                stopped = ops.stop_carla(carla)
                _require(bool((stopped or {}).get("shutdown_verified", stopped)),
                         "CARLA shutdown not verified")
            except BaseException as exc: cleanup_failure("CARLA", exc)
        if ran is not None:
            try:
                cleaned = ran.close()
                _require(cleaned.ok and cleaned.remote_cn_touched is False,
                         "local RAN cleanup failed")
            except BaseException as exc: cleanup_failure("local RAN", exc)
        if prepared is not None:
            try:
                ops.verify_cold(prepared)
                report["cold_after_verified"] = True
            except BaseException as exc: cleanup_failure("cold after", exc)
        if report["cleanup_errors"]:
            if not report["error"]: report["error"] = report["cleanup_errors"][0]
            report["status"] = "FAILED"
        _write(plan.output_root / RESULT_NAME, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--remote-host", default=REMOTE_HOST)
    parser.add_argument("--remote-repository", type=Path, default=REMOTE_REPOSITORY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--outer-runtime-s", type=float, default=OUTER_RUNTIME_S)
    parser.add_argument("--execute", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    args = build_parser().parse_args(argv)
    _require(args.execute == EXECUTE_TOKEN, "execute token absent")
    plan = OneDecisionPlanV1(args.output_root, args.run_id, args.attempt_id,
                             args.remote_host, args.remote_repository, args.config,
                             args.carla_port, args.outer_runtime_s)
    result = run_one_decision(plan)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS_ONE_DECISION_SPLIT_HOST" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "EXECUTE_TOKEN", "TRANSMITTED_BUDGET", "DECISION_CAP",
           "SAFETY_TIMEOUT_S", "OUTER_RUNTIME_S", "LOCAL_CAPTURE_FILTER",
           "REMOTE_CN_DIRECTORY", "REMOTE_CN_COMPOSE", "REMOTE_CN_PROJECT",
           "REMOTE_CN_SERVICES", "REMOTE_CORE_RESET_NAME",
           "SplitHostOneDecisionError", "OneDecisionPlanV1", "RemoteCliV1",
           "SystemOpsV1", "validate_closed_one_decision", "run_one_decision",
           "build_parser", "main"]
