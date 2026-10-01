"""Production W10275/L10319 lifecycle for the one-frame B handshake.

This module owns orchestration only.  It composes the registered local RAN
executor, CARLA lifecycle, Route-B bridge, UE dependency factory and the
GT-free edge executable.  The edge runs inside the pinned, pre-existing
``oai-perception-rx:latest`` image on L10319 with no build or pull.  The
remote CN is a prerequisite and is never torn down by this lifecycle.

Importing this module performs no I/O and imports neither torch nor CUDA.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from rl_agent.splitfusion_run4_split_host_l10319_v1 import contract as C
from rl_agent.splitfusion_run4_split_host_l10319_v1 import (
    local_ran_lifecycle_v1 as LR,
    local_ran_executor_v1 as LX,
    remote_edge_lifecycle_v1 as RE,
    split_host_one_decision_runner_v1 as OLD,
    split_host_phase6_coordinator_v1 as CO,
)

from . import b_edge_engineering_request_v2 as ER
from . import b_edge_process_v1 as EP
from . import b_edge_runtime_v2 as EDGE
from . import b_one_frame_execution_v1 as ONE
from . import b_one_frame_pipeline_v2 as PIPE
from . import b_production_dependencies_v1 as DEP
from . import b_ue_process_v1 as UE
from . import b_validation_runner_v1 as VALID
from . import branch_evidence_v1 as BE
from . import final_actor_gate_v2 as F
from . import live_adapters_v1 as L
from . import one_frame_engineering_v1 as O
from . import operational_ack_v1 as ACK
from . import production_lifecycle_adapter_v1 as LIFE


FACTORY_SCHEMA = "scenesense.splitfusion.run4b5b.one_frame_factory.v1"
EDGE_PROJECT_PREFIX = "run4b5b-oneframe-"
EDGE_SERVICE = "oai-perception-rx"
EDGE_CONTAINER = "oai-perception-rx"
EDGE_NETWORK = "oai-cn5g-public-net"
EDGE_REPOSITORY_DESTINATION = "/work/abiodun"
EDGE_STATE_DESTINATION = "/work/torch_cache"
EDGE_READY_DESTINATION = f"{EDGE_STATE_DESTINATION}/EDGE_READY.json"
EDGE_RUNTIME_ATTEMPT = f"{EDGE_STATE_DESTINATION}/runtime"
REMOTE_MODULE = (
    "rl_agent.splitfusion_run4b5b_live_isolation_v1.b_edge_runtime_v2"
)
EDGE_SOURCE_FILES = (
    "rl_agent/splitfusion_run4b5b_live_isolation_v1/"
    "b_edge_engineering_request_v2.py",
    "rl_agent/splitfusion_run4b5b_live_isolation_v1/b_edge_runtime_v2.py",
    "rl_agent/splitfusion_run4b5b_live_isolation_v1/b_edge_service_v2.py",
    "rl_agent/splitfusion_run4b5b_live_isolation_v1/b_edge_process_v1.py",
)
ATTEMPT_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,47}")
QUEUE_DEPTH = 64
MAP_API_PORT = 35001
MAP_FEEDBACK_PORT = 39401
SAFETY_TIMEOUT_S = 60.0
OAI_AUTHORITY_ROOT = Path(
    "/home/shr_aisvcs/workarea/carla_0_10_env/"
    "Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/"
    "abiodun/OAI/openairinterface5g"
)
OAI_GITLINK_COMMIT = "7473cdb52e1cf3c40e1e1f189f03b2785bf15610"


class ProductionOneFrameError(O.OneFrameEngineeringError):
    """The composed production lifecycle failed closed."""


# ---------------------------------------------------------------------------
# Attempt-scoped path ownership: exactly one owner per path.
# (name, path relative to the local attempt root, owner, kind, phase)
# ---------------------------------------------------------------------------
ATTEMPT_PATH_OWNERSHIP: tuple[tuple[str, str, str, str, str], ...] = (
    ("legacy_attempt", "attempt",
     "split_host_one_decision_runner_v1.SystemOpsV1.prepare", "dir", "start"),
    ("service", "service",
     "split_host_one_decision_runner_v1.SystemOpsV1.prepare", "dir", "start"),
    ("remote_core_reset", "REMOTE_CORE_RESET_EVIDENCE.json",
     "split_host_one_decision_runner_v1.SystemOpsV1.restart_remote_core",
     "file", "start"),
    ("local_ran_executor", "local_ran_executor",
     "local_ran_executor_v1 (via SystemOpsV1.start_ran)", "dir", "start"),
    ("carla_log", "carla_server.log",
     "carla lifecycle helper start_carla (via SystemOpsV1.start_carla)",
     "file", "start"),
    ("prepared_campaign", "service/campaign.json",
     "split_host_one_decision_runner_v1.SystemOpsV1.prepare", "file", "start"),
    ("prepared_cell", "service/cell.json",
     "split_host_one_decision_runner_v1.SystemOpsV1.prepare", "file", "start"),
    ("prepared_bindings", "service/bindings.json",
     "split_host_one_decision_runner_v1.SystemOpsV1.prepare", "file", "start"),
    ("map_output", "service/map",
     "map-install runtime --output-dir (RealProductionOpsV1._start_map)",
     "dir", "start"),
    ("isolated_campaign", "service/one_frame_campaign.yaml",
     "RealProductionOpsV1.start", "file", "start"),
    ("target_snr_trace", "service/radio_trace.csv",
     "ue_route_b_split_cell_adapter_v1.start_target_snr (moved at stop)",
     "file", "start"),
    ("target_snr_stop_file", "service/stop_target_snr",
     "ue_route_b_split_cell_adapter_v1.stop_target_snr", "file", "stop"),
    ("target_snr_summary", "service/radio_trace.csv.summary.json",
     "ue_route_b_split_cell_adapter_v1.stop_target_snr", "file", "stop"),
    ("target_snr_start", "service/one_frame_target_start",
     "RealProductionOpsV1.start (actuator --start-file)", "file", "start"),
    ("collector_profile_activation", "service/collector_profile_activation",
     "pinned PassiveSplitCollector, first processed frame "
     "(campaign _target_start_file)", "file", "execute"),
    ("ue_telemetry", "ue_telemetry",
     "b_production_dependencies_v1.build_production_dependencies_v1",
     "dir", "start"),
    ("route", "route", "RealProductionOpsV1.start (route attempt_dir)",
     "dir", "start"),
    ("raw_gt_spool", "raw_carla_gt_spool",
     "b_route_bridge_v3.RawGroundTruthSpoolV3 (bridge construction)",
     "dir", "start"),
    ("postrun_remote_prediction", "postrun_remote_prediction",
     "reserved route edge_evidence_dir; never created by the GT-free bridge",
     "dir", "reserved"),
    ("ue_output", "ue_output", "b_one_frame_execution_v1.execute_one",
     "dir", "execute"),
    ("ue_evidence", "ue_evidence", "b_one_frame_execution_v1.execute_one",
     "dir", "execute"),
    ("remote_prediction", "remote_prediction",
     "RealProductionOpsV1._download_prediction", "dir", "execute"),
    ("remote_edge_log", "remote_edge.log",
     "RealProductionOpsV1._stop_remote_edge", "file", "stop"),
    ("radio_restoration_trace", "radio_trace.csv",
     "ue_route_b_split_cell_adapter_v1.stop_target_snr", "file", "stop"),
)


@dataclass(frozen=True, slots=True)
class AttemptPathsV1:
    """The validated ownership map of one local attempt root."""

    root: Path

    def __post_init__(self) -> None:
        names = [row[0] for row in ATTEMPT_PATH_OWNERSHIP]
        relatives = [PurePosixPath(row[1]) for row in ATTEMPT_PATH_OWNERSHIP]
        _require(len(set(names)) == len(names), "duplicate attempt path name")
        _require(len(set(relatives)) == len(relatives),
                 "two owners claim the same attempt path")
        for relative in relatives:
            _require(not relative.is_absolute() and ".." not in relative.parts,
                     "attempt path escapes the attempt root")
        # A path may sit beneath another only under the shared service
        # directory, whose children are listed explicitly above.
        for child in relatives:
            for parent in relatives:
                if child != parent and parent in child.parents:
                    _require(parent == PurePosixPath("service"),
                             f"attempt path {child} is nested in {parent}")
        for left, right in (("ue_telemetry", "ue_evidence"),
                            ("ue_telemetry", "ue_output"),
                            ("ue_output", "ue_evidence")):
            _require(self.get(left) != self.get(right),
                     f"{left} and {right} collide")

    def get(self, name: str) -> Path:
        for row in ATTEMPT_PATH_OWNERSHIP:
            if row[0] == name:
                return self.root / row[1]
        raise ProductionOneFrameError(f"unknown attempt path: {name}")

    def phase(self, phase: str) -> tuple[Path, ...]:
        return tuple(self.root / row[1] for row in ATTEMPT_PATH_OWNERSHIP
                     if row[4] == phase)

    @staticmethod
    def table() -> list[dict[str, str]]:
        return [{"name": n, "path": r, "owner": o, "kind": k, "phase": ph}
                for n, r, o, k, ph in ATTEMPT_PATH_OWNERSHIP]


def attempt_paths(config: O.OneFrameConfigV1) -> AttemptPathsV1:
    return AttemptPathsV1(Path(config.local_attempt_root))


def _require(value: bool, message: str) -> None:
    if not value:
        raise ProductionOneFrameError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProductionOneFrameError("value is not canonical JSON") from exc


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def runtime_variant(scientific_variant: str) -> L.ActorVariant:
    exact = {
        F.RUN4B_VARIANT: L.ActorVariant.RUN4B,
        F.RUN5B_VARIANT: L.ActorVariant.RUN5B,
    }
    try:
        return exact[scientific_variant]
    except KeyError as exc:
        raise ProductionOneFrameError(
            "scientific variant has no exact runtime mapping") from exc


def _split_host_for_ue() -> VALID.SplitHostBindingV1:
    return VALID.SplitHostBindingV1(
        carla_host=O.LOCAL_HOST, ue_host=O.LOCAL_HOST,
        cn_host=O.REMOTE_HOST, edge_host=O.REMOTE_HOST,
        ext_dn_host=O.REMOTE_HOST, ack_receiver_host=O.LOCAL_HOST,
        ack_receiver_port=O.ACK_PORT)


def build_ue_request(config: O.OneFrameConfigV1,
                     actor: F.LoadedFinalActorV2) -> UE.BUEProcessRequestV1:
    """Construct the distinct budget-one UE request without invoking v1's 300 gate."""
    variant = runtime_variant(config.variant)
    expected = L.feature_schema_sha256(
        variant, L.expected_feature_order(variant))
    _require(actor.identity.feature_schema_sha256 == expected,
             "final actor feature schema differs from the live B schema")
    return UE.BUEProcessRequestV1(
        run_id=config.run_id, variant=variant,
        config_binding_sha256=config.binding_sha256(),
        actor_boundary_sha256=actor.identity.actor_tree_sha256,
        feature_schema_sha256=expected, transmitted_budget=1,
        deadline_ns=ACK.ACK_DEADLINE_NS,
        ack_semantics=EP.ACK_SEMANTICS,
        postrun_semantics=EP.POSTRUN_SEMANTICS,
        clock_domain=ACK.CLOCK_DOMAIN, split_host=_split_host_for_ue(),
        output_root=attempt_paths(config).get("ue_output"),
        evidence_root=attempt_paths(config).get("ue_evidence"),
        actor_manifest_path=config.actor_manifest_path,
        required_authority_modules=UE.REQUIRED_AUTHORITIES)


def build_edge_request(config: O.OneFrameConfigV1,
                       actor: F.LoadedFinalActorV2) -> tuple[str, dict[str, Any]]:
    """Create canonical engineering-v2 request and round-trip its validator."""
    ue = build_ue_request(config, actor)
    raw = {
        "schema": ER.SCHEMA, "purpose": ER.PURPOSE,
        "claim_scope": ER.CLAIM_SCOPE, "role": EP.ROLE,
        "run_id": config.run_id, "variant": ue.variant.value,
        "config_binding_sha256": config.binding_sha256(),
        "actor_boundary_sha256": ue.actor_boundary_sha256,
        "feature_schema_sha256": ue.feature_schema_sha256,
        "transmitted_budget": 1, "deadline_ns": EP.DEADLINE_NS,
        # Intentional translation from the user-facing engineering prose to
        # the exact edge wire/post-run constants.
        "ack_semantics": EP.ACK_SEMANTICS,
        "postrun_semantics": EP.POSTRUN_SEMANTICS,
        "clock_domain": EP.CLOCK_DOMAIN,
        "split_host": {
            "carla_host": O.LOCAL_HOST, "ue_host": O.LOCAL_HOST,
            "cn_host": O.REMOTE_HOST, "edge_host": O.REMOTE_HOST,
            "ext_dn_host": O.REMOTE_HOST,
            "ack_receiver_host": O.UE_TUNNEL_IP,
            "ack_receiver_port": O.ACK_PORT,
        },
        "output_root": None, "evidence_root": None,
        "actor_manifest_path": None,
        "remote_attempt_root": EDGE_RUNTIME_ATTEMPT,
        "required_authority_modules": list(EP.REQUIRED_AUTHORITIES),
        "old_live_quality_runtime_permitted": False,
    }
    payload = _canonical(raw)
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    decoded = ER.decode_and_validate(encoded)
    _require(decoded == raw, "edge engineering request round-trip differs")
    return encoded, decoded


@dataclass(frozen=True, slots=True)
class RemoteEdgePlanV1:
    project: str
    compose_path: Path
    state_root: Path
    ready_file: Path
    runtime_root: Path
    request_b64: str
    compose: Mapping[str, Any]


def build_remote_edge_plan(config: O.OneFrameConfigV1,
                           actor: F.LoadedFinalActorV2) -> RemoteEdgePlanV1:
    _require(bool(ATTEMPT_RE.fullmatch(config.run_id)),
             "run_id is not safe for the unique Compose project")
    try:
        campaign_relative = config.edge_campaign_config.relative_to(
            config.remote_repository)
    except ValueError as exc:
        raise ProductionOneFrameError(
            "edge campaign config is not inside the explicit remote repository") from exc
    request_b64, _ = build_edge_request(config, actor)
    state = config.remote_attempt_root / "state"
    compose_path = config.remote_attempt_root / "remote_edge.compose.json"
    project = EDGE_PROJECT_PREFIX + config.run_id
    fcos = config.remote_repository / next(
        item.relative_path for item in C.ARTIFACTS
        if item.name == "torchvision_fcos")
    command = [
        "python3", "-u", "-m", REMOTE_MODULE, "start",
        "--request-b64", request_b64,
        "--campaign-config", str(PurePosixPath(EDGE_REPOSITORY_DESTINATION)
                                  / PurePosixPath(campaign_relative.as_posix())),
        "--ready-file", EDGE_READY_DESTINATION,
        "--cell-id", config.cell_id,
        "--edge-port", str(config.network.edge_feature_port),
        "--direct-map-host", config.network.direct_map_host,
        "--direct-map-port", str(config.network.direct_map_port),
        "--queue-depth", str(QUEUE_DEPTH),
        "--execute", EDGE.EXECUTE_TOKEN,
    ]
    labels = {
        "scenesense.owner": O.PURPOSE,
        "scenesense.run_id": config.run_id,
        "scenesense.config_binding_sha256": config.binding_sha256(),
        "scenesense.image_manifest_digest": C.EDGE_IMAGE_MANIFEST_DIGEST,
    }
    compose: Mapping[str, Any] = {
        "name": project,
        "services": {EDGE_SERVICE: {
            "image": C.EDGE_IMAGE_TAG, "pull_policy": "never",
            "container_name": EDGE_CONTAINER, "privileged": True,
            "init": True, "gpus": "all", "command": command,
            "environment": {
                "PYTHONPATH": (
                    f"{EDGE_REPOSITORY_DESTINATION}:"
                    f"{EDGE_REPOSITORY_DESTINATION}/rl_agent/feature_ae"),
                "TORCH_HOME": EDGE_STATE_DESTINATION,
                "MPLBACKEND": "Agg",
            },
            "volumes": [
                {"type": "bind", "source": str(config.remote_repository),
                 "target": EDGE_REPOSITORY_DESTINATION, "read_only": True},
                {"type": "bind", "source": str(state),
                 "target": EDGE_STATE_DESTINATION, "read_only": False},
                {"type": "bind", "source": str(fcos),
                 "target": RE.FCOS_DESTINATION, "read_only": True},
            ],
            "networks": {"public_net": {"ipv4_address": O.EDGE_IP}},
            "labels": labels,
        }},
        "networks": {"public_net": {
            "external": True, "name": EDGE_NETWORK}},
    }
    service = compose["services"][EDGE_SERVICE]
    _require("build" not in service and service["pull_policy"] == "never",
             "edge compose gained a build or pull path")
    return RemoteEdgePlanV1(
        project=project, compose_path=compose_path, state_root=state,
        ready_file=state / "EDGE_READY.json",
        runtime_root=state / "runtime", request_b64=request_b64,
        compose=compose)


class TargetSnrLeasePumpV1:
    """Forward only observed target-command ACKs into the frozen Run-5B lease."""

    def __init__(self, csv_path: Path, adapter: Any) -> None:
        self.csv_path, self.adapter = Path(csv_path), adapter
        self.stop_event = threading.Event()
        self.first_ack = threading.Event()
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, name="run5b-target-snr-lease-pump", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        seen: set[int] = set()
        active: Optional[str] = None
        try:
            while not self.stop_event.wait(0.01):
                if not self.csv_path.is_file():
                    continue
                with self.csv_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    try:
                        step = int(row["step_index"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if step in seen:
                        continue
                    status = str(row.get("command_timing_status") or "")
                    if status in {"ACK_ON_TIME", "ACK_LATE"}:
                        seen.add(step)
                        active = f"target-snr-step-{step}"
                        self.adapter.record_command_ack(
                            command_id=active, status="ACK", clamped=False,
                            target_snr_db=float(row["target_snr_db"]))
                        self.adapter.record_heartbeat(
                            active_command_id=active)
                        self.first_ack.set()
                    elif status == "SKIP_OBSOLETE_NEVER_BURST":
                        seen.add(step)
                        if active is not None:
                            self.adapter.record_heartbeat(
                                active_command_id=active)
        except BaseException as exc:
            self.error = exc
            self.first_ack.set()

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        _require(self.first_ack.wait(timeout_s),
                 "Run-5B target-SNR lease received no ACK")
        if self.error is not None:
            raise ProductionOneFrameError(
                f"Run-5B target-SNR lease pump failed: {self.error}")

    def close(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=2.0)
        _require(not self._thread.is_alive(),
                 "Run-5B target-SNR lease pump did not stop")
        if self.error is not None:
            raise ProductionOneFrameError(
                f"Run-5B target-SNR lease pump failed: {self.error}")


@dataclass(slots=True)
class StartedOneFrameV1:
    config: O.OneFrameConfigV1
    actor: F.LoadedFinalActorV2
    ue_request: UE.BUEProcessRequestV1
    old_plan: Any = None
    prepared: Any = None
    edge_plan: Optional[RemoteEdgePlanV1] = None
    ran: Any = None
    carla: Any = None
    map_process: Any = None
    target_process: Any = None
    target_output: Optional[Path] = None
    target_stop: Optional[Path] = None
    target_start: Optional[Path] = None
    lease_pump: Optional[TargetSnrLeasePumpV1] = None
    dependencies: Optional[DEP.ProductionDependenciesV1] = None
    pipeline: Any = None
    controller_lineage_sha256: Optional[str] = None
    campaign: Optional[dict[str, Any]] = None
    cell: Optional[dict[str, Any]] = None
    stopped: bool = False


class ProductionOpsV1(Protocol):
    def preflight(self, config: O.OneFrameConfigV1,
                  actor: F.LoadedFinalActorV2) -> None: ...
    def start(self, config: O.OneFrameConfigV1,
              actor: F.LoadedFinalActorV2) -> StartedOneFrameV1: ...
    def execute(self, state: StartedOneFrameV1) -> O.OneFrameExecutionV1: ...
    def stop(self, state: StartedOneFrameV1) -> None: ...


class RealProductionOpsV1:
    """Real effects adapter; injected tests replace this object entirely."""

    def __init__(self, settings: LIFE.ProductionLifecycleSettingsV1) -> None:
        self.settings = settings
        self.remote = OLD.RemoteCliV1()
        self.system = OLD.SystemOpsV1(remote=self.remote)

    @staticmethod
    def _remote_shell(argv: Sequence[str]) -> str:
        return " ".join(shlex.quote(str(item)) for item in argv)

    def _ssh(self, argv: Sequence[str], timeout_s: float = 60.0,
             data: Optional[bytes] = None) -> subprocess.CompletedProcess[bytes]:
        command = self._remote_shell(argv)
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             O.REMOTE_SSH, command], input=data, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout_s, check=False)

    def _checked_ssh(self, argv: Sequence[str], label: str,
                     timeout_s: float = 60.0) -> bytes:
        result = self._ssh(argv, timeout_s)
        _require(result.returncode == 0,
                 f"remote {label} failed: "
                 f"{result.stderr.decode('utf-8', 'replace')[-1200:]}")
        return result.stdout

    @staticmethod
    def _require_local_artifacts(config: O.OneFrameConfigV1) -> None:
        for artifact in C.ARTIFACTS:
            path = config.local_repository / artifact.relative_path
            _require(path.is_file(),
                     f"local retained artifact is absent: {path}")
            _require(_sha_file(path) == artifact.sha256,
                     f"local retained artifact hash differs: {path}")

    @staticmethod
    def _git_object(repository: Path, name: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", name],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=15.0, check=False)
        _require(result.returncode == 0,
                 f"cannot resolve git object {name} in {repository}: "
                 f"{result.stderr[-600:]}")
        value = result.stdout.strip()
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", value)),
                 f"git object is not exact: {repository}:{name}")
        return value

    @staticmethod
    def _mountinfo_path(value: str) -> str:
        for escaped, decoded in ((r"\134", "\\"), (r"\040", " "),
                                 (r"\011", "\t"), (r"\012", "\n")):
            value = value.replace(escaped, decoded)
        return value

    @classmethod
    def _require_local_oai_binding(cls, config: O.OneFrameConfigV1) -> None:
        authority = OAI_AUTHORITY_ROOT.resolve(strict=True)
        target = (config.local_repository / "OAI/openairinterface5g").resolve(
            strict=True)
        expected_target = (config.local_repository.resolve(strict=True)
                           / "OAI/openairinterface5g")
        _require(target == expected_target,
                 "local OAI target resolves away from the moved repository")
        authority_stat, target_stat = authority.stat(), target.stat()
        _require((authority_stat.st_dev, authority_stat.st_ino)
                 == (target_stat.st_dev, target_stat.st_ino),
                 "local OAI bind target is not the authority inode")

        records: list[list[str]] = []
        for line in Path("/proc/self/mountinfo").read_text(
                encoding="utf-8").splitlines():
            fields = line.split()
            if (len(fields) >= 10
                    and cls._mountinfo_path(fields[4]) == str(target)):
                records.append(fields)
        _require(len(records) == 1,
                 "local OAI target is not one exact mountpoint")
        record = records[0]
        _require(cls._mountinfo_path(record[3]) == str(authority),
                 "local OAI bind source differs from authority")
        _require("ro" in set(record[5].split(",")),
                 "local OAI bind mount is not read-only")
        _require(cls._git_object(authority, "HEAD") == OAI_GITLINK_COMMIT,
                 "OAI authority commit differs")
        _require(cls._git_object(
            config.local_repository, "HEAD:OAI/openairinterface5g"
        ) == OAI_GITLINK_COMMIT, "moved-worktree OAI gitlink differs")

        observed = LR.verify_source_pins(config.local_repository)
        _require(len(observed) == len(LR.SOURCE_PINS),
                 "local-RAN source-pin verification is incomplete")
        for name in ("nr_softmodem", "nr_uesoftmodem",
                     "tracer_multi", "tracer_record"):
            pin = next(item for item in LR.SOURCE_PINS if item.name == name)
            _require(os.access(config.local_repository / pin.relative_path,
                               os.X_OK),
                     f"local-RAN executable is not executable: {name}")

        from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as A
        audit = A.reconcile_contract(
            config.local_repository /
            "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json",
            config.local_repository /
            "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json")
        _require(audit.get("status") == "PHASE14A_CPU_RECONCILIATION_PASSED",
                 "Phase14a CPU reconciliation did not pass")
        _require(not (OLD.LOCAL_RADIO_STATE_BASE
                      / f"split_host_{config.run_id}").exists(),
                 "local radio state is not create-only")

    def preflight(self, config: O.OneFrameConfigV1,
                  actor: F.LoadedFinalActorV2) -> None:
        _require(config.local_repository.is_dir(),
                 "local repository is absent")
        _require(config.route_config.is_file(), "local route config is absent")
        self._require_local_artifacts(config)
        self._require_local_oai_binding(config)
        plan = build_remote_edge_plan(config, actor)
        _require(not config.local_attempt_root.exists(),
                 "local attempt root is not create-only")
        hostname = self._checked_ssh(("hostname", "-s"), "hostname").decode().strip()
        _require(hostname == "L10319", "remote hostname differs")
        for path, kind in (
            (config.remote_repository, "-d"),
            (config.edge_campaign_config, "-f"),
            (config.remote_repository / next(
                item.relative_path for item in C.ARTIFACTS
                if item.name == "torchvision_fcos"), "-f"),
        ):
            self._checked_ssh(("test", kind, str(path)), f"path {path}")
        absent = self._ssh(("test", "!", "-e", str(config.remote_attempt_root)), 15)
        _require(absent.returncode == 0, "remote attempt root is not create-only")
        image_raw = self._checked_ssh(
            ("sudo", "-n", "docker", "image", "inspect", C.EDGE_IMAGE_TAG),
            "image inspect")
        image = json.loads(image_raw.decode("utf-8"))
        _require(type(image) is list and len(image) == 1,
                 "remote image inspect count differs")
        _require(image[0].get("Id") == C.REMOTE_IMAGE_ID,
                 "remote edge image identity differs")
        _require(RE.canonical_image_inspect_sha256(image[0])
                 == C.EDGE_IMAGE_CANONICAL_INSPECT_SHA256,
                 "remote edge canonical image identity differs")
        self._checked_ssh(
            ("sudo", "-n", "docker", "network", "inspect", EDGE_NETWORK),
            "CN network inspect")
        existing = self._ssh(
            ("sudo", "-n", "docker", "container", "inspect", EDGE_CONTAINER), 15)
        _require(existing.returncode != 0,
                 "a pre-existing edge container cannot become attempt-owned")
        for relative in EDGE_SOURCE_FILES:
            local = config.local_repository / relative
            _require(local.is_file(), f"local edge source is absent: {relative}")
            remote = self._checked_ssh(
                ("sha256sum", str(config.remote_repository / relative)),
                f"source hash {relative}").decode().split()[0]
            _require(remote == _sha_file(local),
                     f"local/remote edge source differs: {relative}")
        _require(plan.compose["services"][EDGE_SERVICE]["command"][-2:]
                 == ["--execute", EDGE.EXECUTE_TOKEN],
                 "remote edge command authorization drift")

    def _upload_create_only(self, path: Path, payload: bytes) -> None:
        code = (
            "from pathlib import Path;import sys,os;"
            "p=Path(sys.argv[1]);"
            "f=p.open('xb');f.write(sys.stdin.buffer.read());f.flush();"
            "os.fsync(f.fileno());f.close()")
        result = self._ssh(("python3", "-c", code, str(path)), 30, payload)
        _require(result.returncode == 0,
                 "remote create-only compose upload failed")

    def _start_remote_edge(self, state: StartedOneFrameV1) -> None:
        assert state.edge_plan is not None
        plan = state.edge_plan
        self._checked_ssh(
            ("mkdir", "--", str(plan.state_root.parent)),
            "attempt root creation")
        self._checked_ssh(
            ("mkdir", "--", str(plan.state_root)),
            "attempt state creation")
        payload = _canonical(plan.compose) + b"\n"
        self._upload_create_only(plan.compose_path, payload)
        base = ("sudo", "-n", "docker", "compose", "--project-name",
                plan.project, "-f", str(plan.compose_path))
        self._checked_ssh(
            (*base, "config", "--quiet"), "compose validation")
        self._checked_ssh(
            (*base, "up", "-d", "--no-build", "--pull", "never",
             "--force-recreate", "--no-deps", EDGE_SERVICE),
            "edge container launch", 120)
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            result = self._ssh(("test", "-f", str(plan.ready_file)), 10)
            if result.returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise ProductionOneFrameError("remote edge did not become ready")
        ready_raw = self._checked_ssh(
            ("cat", str(plan.ready_file)), "edge ready read")
        ready = json.loads(ready_raw.decode("ascii"))
        expected = {
            "schema": EDGE.READY_SCHEMA, "run_id": state.config.run_id,
            "cell_id": state.config.cell_id,
            "variant": state.ue_request.variant.value,
            "purpose": ER.PURPOSE, "claim_scope": ER.CLAIM_SCOPE,
            "transmitted_budget": 1,
            "edge_port": O.EDGE_FEATURE_PORT,
            "direct_map_host": O.LOCAL_LAN_IP,
            "direct_map_port": O.DIRECT_MAP_PORT,
            "ack_receiver_host": O.UE_TUNNEL_IP,
            "ack_receiver_port": O.ACK_PORT,
            "tcp_listener": False, "gt_ingress": False,
            "qperc_reward_evaluator": False,
            "legacy_quality_feedback": False, "map_wait": False,
            "prewarm_completed": True,
        }
        drift = {key: (ready.get(key), value) for key, value in expected.items()
                 if ready.get(key) != value}
        _require(not drift, f"remote edge ready drift: {drift}")

    @staticmethod
    def _start_map(campaign: Mapping[str, Any], root: Path,
                   carla_port: int) -> subprocess.Popen[Any]:
        from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
        runtime = pinned.repo_path(str(campaign["runtime"]["map_install_runtime"]))
        output = root / "map"
        argv = [
            sys.executable, str(runtime),
            "--api-host", "127.0.0.1", "--api-port", str(MAP_API_PORT),
            "--udp-host", O.LOCAL_LAN_IP,
            "--udp-port", str(O.DIRECT_MAP_PORT),
            "--install-feedback-host", "127.0.0.1",
            "--install-feedback-port", str(MAP_FEEDBACK_PORT),
            "--default-action-id", "71", "--carla-host", "127.0.0.1",
            "--carla-port", str(carla_port), "--output-dir", str(output),
            "--focus-follow-stream-id", "unused",
            "--installed-frame-history-size",
            str(int(campaign["measurement_contract"]
                    ["installed_frame_history_size"])),
        ]
        process = subprocess.Popen(
            argv, cwd=str(pinned.ROOT), stdin=subprocess.DEVNULL)
        try:
            import urllib.request
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                _require(process.poll() is None,
                         "split-host map exited during startup")
                try:
                    with urllib.request.urlopen(
                            f"http://127.0.0.1:{MAP_API_PORT}/healthz",
                            timeout=1.0) as response:
                        if response.status == 200:
                            return process
                except OSError:
                    pass
                time.sleep(0.25)
            raise ProductionOneFrameError("split-host map did not become ready")
        except BaseException:
            pinned.stop_process(process)
            raise

    @staticmethod
    def _read_prepared(prepared: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
        args = prepared.child_args
        campaign = json.loads(Path(args.campaign_json).read_text(encoding="utf-8"))
        cell = json.loads(Path(args.cell_json).read_text(encoding="utf-8"))
        binding = json.loads(Path(args.bindings_json).read_text(encoding="utf-8"))
        lineage = str(binding["controller_lineage_sha256"])
        _require(len(lineage) == 64
                 and all(ch in "0123456789abcdef" for ch in lineage),
                 "prepared controller lineage is invalid")
        return campaign, cell, lineage

    def start(self, config: O.OneFrameConfigV1,
              actor: F.LoadedFinalActorV2) -> StartedOneFrameV1:
        ue_request = build_ue_request(config, actor)
        state = StartedOneFrameV1(config=config, actor=actor,
                                  ue_request=ue_request)
        try:
            state.old_plan = OLD.OneDecisionPlanV1(
                output_root=config.local_attempt_root,
                run_id=config.run_id, attempt_id=config.run_id,
                remote_repository=config.remote_repository,
                config_path=config.route_config,
                carla_port=config.network.carla_rpc_port,
                outer_runtime_s=OLD.OUTER_RUNTIME_S)
            state.old_plan.validate()
            state.prepared = self.system.prepare(state.old_plan)
            state.campaign, state.cell, state.controller_lineage_sha256 = (
                self._read_prepared(state.prepared))
            _require(state.controller_lineage_sha256
                     != ue_request.actor_boundary_sha256,
                     "authoritative controller lineage aliases actor boundary")
            CO.validate_campaign_binding(state.campaign)
            self.system.restart_remote_core(state.old_plan)
            state.edge_plan = build_remote_edge_plan(config, actor)
            self._start_remote_edge(state)
            state.ran = self.system.start_ran(state.prepared, state.old_plan)
            state.carla = self.system.start_carla(state.prepared, state.old_plan)
            paths = attempt_paths(config)
            state.map_process = self._start_map(
                state.campaign, paths.get("service"),
                config.network.carla_rpc_port)
            from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
            import yaml
            service = paths.get("service")
            isolated = paths.get("isolated_campaign")
            target_start = paths.get("target_snr_start")
            # The actuator's start file (created below, before the first
            # decision) and the pinned collector's one-time profile-activation
            # file are distinct create-only paths with separate owners.
            state.campaign["_target_start_file"] = str(
                paths.get("collector_profile_activation"))
            isolated.write_text(
                yaml.safe_dump(state.campaign, sort_keys=False), encoding="utf-8")
            target, output, stop = pinned.start_target_snr(
                state.campaign, campaign_path=isolated,
                profile_id=str(state.cell["network_profile_id"]),
                temporary_dir=service, start_file=target_start)
            state.target_process, state.target_output, state.target_stop = (
                target, output, stop)
            state.target_start = target_start
            telemetry = json.loads(
                Path(state.prepared.child_args.bindings_json).read_text(
                    encoding="utf-8"))
            runtime = state.campaign["runtime"]
            state.dependencies = DEP.build_production_dependencies_v1(
                variant=ue_request.variant,
                tracer_dir=Path(telemetry["tracer_dir"]),
                t_messages=Path(telemetry["t_messages"]),
                ue_relay_port=int(telemetry["ue_relay_port"]),
                telemetry_root=paths.get("ue_telemetry"),
                ue_bind_host=O.UE_TUNNEL_IP,
                edge_remote_host=O.EDGE_IP,
                edge_receive_port=O.EDGE_FEATURE_PORT,
                udp_chunk_bytes=int(runtime["udp_chunk_bytes"]),
                socket_buffer_request_bytes=int(
                    runtime["socket_buffer_request_bytes"]))
            if ue_request.variant is L.ActorVariant.RUN5B:
                _require(state.dependencies.snr_controller_adapter is not None,
                         "Run-5B SNR controller adapter is absent")
                state.lease_pump = TargetSnrLeasePumpV1(
                    output, state.dependencies.snr_controller_adapter)
                state.lease_pump.start()
            else:
                _require(state.dependencies.snr_controller_adapter is None,
                         "Run-4B unexpectedly received an SNR adapter")
            target_start.touch(exist_ok=False)
            if state.lease_pump is not None:
                state.lease_pump.wait_ready()
            else:
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if output.is_file() and len(output.read_text(
                            encoding="utf-8").splitlines()) >= 2:
                        break
                    _require(target.poll() is None,
                             "target-SNR actuator exited during startup")
                    time.sleep(0.05)
                else:
                    raise ProductionOneFrameError(
                        "target-SNR actuator produced no first ACK")
            row = pinned.action_row(state.campaign, int(state.cell["action_id"]))
            route_dir = paths.get("route")
            route_dir.mkdir(exist_ok=False)
            route_kwargs = {
                "campaign": state.campaign, "cell": state.cell, "row": row,
                "binding": {
                    "dispatcher": "run4b5b_gt_free_operational_ack_v1",
                    "controller_lineage_sha256":
                        state.controller_lineage_sha256,
                },
                "attempt_dir": route_dir, "carla_host": "127.0.0.1",
                "carla_port": config.network.carla_rpc_port,
                "map_api_port": MAP_API_PORT,
                "feedback_port": MAP_FEEDBACK_PORT,
                "edge_evidence_dir":
                    paths.get("postrun_remote_prediction"),
                "maximum_loop_sim_s": SAFETY_TIMEOUT_S,
            }
            state.pipeline = PIPE.build_one_frame_pipeline_v2(
                request=ue_request, evidence_root=config.actor_evidence_root,
                controller_lineage_sha256=state.controller_lineage_sha256,
                dependencies=state.dependencies, cell_id=config.cell_id,
                route_kwargs=route_kwargs,
                raw_spool_root=paths.get("raw_gt_spool"),
                postrun_materializer=None)
            for owned in paths.phase("execute"):
                _require(not owned.exists(),
                         f"execution-owned path exists after startup: {owned}")
            return state
        except BaseException:
            try:
                self.stop(state)
            except BaseException:
                pass
            raise

    def _download_prediction(self, state: StartedOneFrameV1,
                             outcome: ACK.OperationalOutcomeV1
                             ) -> BE.PredictionEvidenceRecordV1:
        assert state.edge_plan is not None
        identity = outcome.identity.exact_sha256()
        source = state.edge_plan.runtime_root / "prediction"
        record_remote = source / "records" / f"{identity}.json"
        artifact_remote = source / "artifacts" / f"{identity}.bin"
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._ssh(("test", "-f", str(record_remote)), 10).returncode == 0:
                break
            time.sleep(0.05)
        else:
            raise ProductionOneFrameError("remote prediction evidence is absent")
        local = attempt_paths(state.config).get("remote_prediction")
        (local / "records").mkdir(parents=True, exist_ok=False)
        (local / "artifacts").mkdir(parents=True, exist_ok=False)
        self.remote.download(record_remote, local / "records" / record_remote.name)
        self.remote.download(
            artifact_remote, local / "artifacts" / artifact_remote.name)
        raw = json.loads((local / "records" / record_remote.name).read_text(
            encoding="ascii"))
        record = BE.PredictionEvidenceRecordV1.from_mapping(raw)
        artifact = local / record.artifact_relative_path
        _require(_sha_file(artifact) == record.prediction_sha256,
                 "remote prediction artifact digest differs")
        _require(record.identity == outcome.identity,
                 "UE outcome and edge prediction identity differ")
        _require(record.prediction_sha256 == outcome.tail_output_sha256,
                 "ACK and retained tail-output digests differ")
        return record

    def execute(self, state: StartedOneFrameV1) -> O.OneFrameExecutionV1:
        _require(state.pipeline is not None and state.dependencies is not None,
                 "one-frame lifecycle was not completely started")
        receiver = UE.UdpOperationalAckReceiverV1(
            O.UE_TUNNEL_IP, O.ACK_PORT)
        result = ONE.execute_one(state.ue_request, state.pipeline, receiver)
        store = ACK.OperationalEvidenceStoreV1.open_existing(
            state.ue_request.evidence_root / "operational_evidence")
        snapshot = store.verify_all(require_all_resolved=True)
        _require(len(snapshot.outcomes) == 1,
                 "one-frame operational outcome count differs")
        outcome = snapshot.outcomes[0]
        _require(outcome.success and outcome.observed_latency_ns is not None,
                 "one-frame operational ACK was not timely")
        self._download_prediction(state, outcome)
        return O.OneFrameExecutionV1(
            run_id=state.config.run_id, variant=state.config.variant,
            transmitted_frames=int(result["transmitted_frames"]),
            policy_decisions=1, operational_successes=1,
            operational_timeouts=0,
            observed_latency_ns=int(outcome.observed_latency_ns),
            # This is a source-bound invariant of the hash-matched edge v2
            # dispatcher, never a cross-host timestamp inference.
            ack_before_map_offer=True,
            prediction_evidence_written=True,
            live_qperc_computed=False,
            exact_identity_sha256=outcome.identity.exact_sha256(),
            result_sha256=str(result["result_sha256"]))

    def _stop_remote_edge(self, state: StartedOneFrameV1,
                          errors: list[str]) -> None:
        if state.edge_plan is None:
            return
        plan = state.edge_plan
        base = ("sudo", "-n", "docker", "compose", "--project-name",
                plan.project, "-f", str(plan.compose_path))
        for argv, label, timeout in (
            ((*base, "stop", "--timeout", "30", EDGE_SERVICE),
             "edge stop", 45),
            (("sudo", "-n", "docker", "logs", "--timestamps",
              EDGE_CONTAINER), "edge logs", 30),
            ((*base, "down", "--remove-orphans", "--timeout", "30"),
             "edge down", 60),
        ):
            try:
                output = self._checked_ssh(argv, label, timeout)
                if label == "edge logs":
                    target = attempt_paths(state.config).get("remote_edge_log")
                    target.write_bytes(output)
            except BaseException as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

    def stop(self, state: StartedOneFrameV1) -> None:
        if state.stopped:
            return
        state.stopped = True
        errors: list[str] = []
        from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
        for label, close in (
            ("SNR lease pump", (lambda: state.lease_pump.close())
             if state.lease_pump is not None else None),
            ("dependencies", (lambda: state.dependencies.close())
             if state.dependencies is not None else None),
            ("target SNR", (lambda: _require(bool(pinned.stop_target_snr(
                state.target_process, state.target_output, state.target_stop,
                attempt_paths(state.config).get("radio_restoration_trace"))),
                "target-SNR restore failed"))
             if state.target_process is not None else None),
            ("map", (lambda: _require(bool(pinned.stop_process(
                state.map_process)), "map stop failed"))
             if state.map_process is not None else None),
            ("RAN", (lambda: state.ran.close())
             if state.ran is not None else None),
        ):
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
        self._stop_remote_edge(state, errors)
        # CARLA is stopped last.  The route runs in-process, and stopping the
        # server while its CARLA client/sensor objects are still alive makes
        # the client library abort the process (std::terminate).  Release the
        # finished route's objects first, and order every other service's
        # teardown before this step so an abort here cannot orphan them.
        if state.carla is not None:
            state.pipeline = None
            import gc
            gc.collect()
            try:
                self.system.stop_carla(state.carla)
            except BaseException as exc:
                errors.append(f"CARLA: {type(exc).__name__}: {exc}")
        if errors:
            raise ProductionOneFrameError(
                "one-frame cleanup failed: " + "; ".join(errors))


class ProductionOneFrameLifecycleV1:
    """The lifecycle interface loaded by :mod:`one_frame_engineering_v1`."""

    def __init__(self, settings: LIFE.ProductionLifecycleSettingsV1, *,
                 ops: Optional[ProductionOpsV1] = None) -> None:
        self.settings = settings
        self.ops = ops or RealProductionOpsV1(settings)
        self.state: Optional[StartedOneFrameV1] = None

    def preflight(self, config: O.OneFrameConfigV1,
                  actor: F.LoadedFinalActorV2) -> None:
        self.ops.preflight(config, actor)

    def start(self, config: O.OneFrameConfigV1,
              actor: F.LoadedFinalActorV2) -> None:
        _require(self.state is None, "one-frame lifecycle already started")
        # RealProductionOpsV1.start owns partial-start cleanup because the
        # outer gate calls stop only after start returns.
        self.state = self.ops.start(config, actor)

    def execute(self, config: O.OneFrameConfigV1,
                actor: F.LoadedFinalActorV2) -> O.OneFrameExecutionV1:
        _require(self.state is not None, "one-frame lifecycle is not started")
        return self.ops.execute(self.state)

    def stop(self, config: O.OneFrameConfigV1) -> None:
        state, self.state = self.state, None
        if state is not None:
            self.ops.stop(state)


def build_one_frame_lifecycle_v1(
        config: O.OneFrameConfigV1,
        settings: LIFE.ProductionLifecycleSettingsV1,
        ) -> ProductionOneFrameLifecycleV1:
    _require(type(config) is O.OneFrameConfigV1, "one-frame config is foreign")
    _require(type(settings) is LIFE.ProductionLifecycleSettingsV1,
             "lifecycle settings are foreign")
    return ProductionOneFrameLifecycleV1(settings)


__all__ = [
    "FACTORY_SCHEMA", "ProductionOneFrameError", "runtime_variant",
    "build_ue_request", "build_edge_request", "RemoteEdgePlanV1",
    "build_remote_edge_plan", "TargetSnrLeasePumpV1",
    "ATTEMPT_PATH_OWNERSHIP", "AttemptPathsV1", "attempt_paths",
    "StartedOneFrameV1", "ProductionOneFrameLifecycleV1",
    "build_one_frame_lifecycle_v1",
]
