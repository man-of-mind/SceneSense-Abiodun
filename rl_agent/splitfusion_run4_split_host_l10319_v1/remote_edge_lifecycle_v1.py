"""Pure, edge-only lifecycle contract for L10319.

This module describes the *remote edge startup qualification*; it does not
execute Docker, CUDA, CN, RAN, CARLA, or a live policy run.  The generated
Compose document is standalone on purpose.  In particular it never consumes
``receiver_container/docker-compose.yaml``'s literal ``../../abiodun`` mount:
the repository mounted at ``/work/abiodun`` is the exact worktree supplied in
the request.

The caller remains responsible for collecting the requested observations and
for executing the reviewed argv vectors.  There is deliberately no executor
or ``--execute`` entry point here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from . import contract as C
from . import remote_edge_gt_entry_v1 as RGT


SCHEMA = "scenesense.run4.remote_edge_lifecycle.v1"
PURPOSE = "REMOTE_EDGE_STARTUP_QUALIFICATION_ONLY"
SERVICE = "oai-perception-rx"
CONTAINER = "oai-perception-rx"
NETWORK = "oai-cn5g-public-net"
PROJECT_PREFIX = "run4-edge-l10319-"
REPOSITORY_DESTINATION = "/work/abiodun"
STATE_DESTINATION = "/work/torch_cache"
EVIDENCE_LEAF = "segmentation_evidence"
EVIDENCE_DESTINATION = f"{STATE_DESTINATION}/{EVIDENCE_LEAF}"
READY_DESTINATION = f"{STATE_DESTINATION}/ready.json"
EDGE_CONFIG_DESTINATION = f"{STATE_DESTINATION}/run4_phase6_edge_config.json"
GT_READY_DESTINATION = f"{STATE_DESTINATION}/remote_gt_listener_ready.json"
GT_FINAL_DESTINATION = f"{STATE_DESTINATION}/remote_gt_listener_final.json"
GT_PORT = RGT.REGISTERED_GT_PORT
GT_SOCKET_TIMEOUT_S = 5.0
GT_EXPECTATION_TIMEOUT_S = 1.0
FCOS_DESTINATION = (
    "/home/shr_aisvcs/.cache/torch/hub/checkpoints/"
    "fcos_resnet50_fpn_coco-99b0c9b7.pth"
)
EDGE_MODULE = (
    "rl_agent.splitfusion_run4_split_host_l10319_v1.remote_edge_gt_entry_v1"
)
READY_SCHEMA = "splitfusion_direct_live_edge_ready.v1"
READY_ARCHITECTURE = "DIRECT_EDGE_TO_MAP_V1"
QUALITY_SPEC_SHA256 = (
    "d5d1e0d2d435076dd53c740f8b0e632620144194c32baf7d24db6d5043fc74d9"
)
ATTEMPT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,47}$")
IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,191}$")


class RemoteEdgeLifecycleError(ValueError):
    """The requested remote lifecycle is unsafe, ambiguous, or drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RemoteEdgeLifecycleError(message)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _absolute_non_root(path: Path, label: str) -> Path:
    path = Path(path)
    _require(path.is_absolute(), f"{label} must be absolute")
    normalized = Path(path).resolve(strict=False)
    _require(str(normalized) not in ("/", "/home", "/tmp"),
             f"{label} is too broad")
    return normalized


def _safe_relative(path: str, label: str) -> str:
    parsed = PurePosixPath(str(path))
    _require(not parsed.is_absolute() and ".." not in parsed.parts,
             f"{label} must be repository-relative")
    _require(bool(parsed.parts), f"{label} is empty")
    return str(parsed)


@dataclass(frozen=True)
class RemoteEdgePaths:
    """All host paths are explicit and unique to one attempt."""

    repository_root: Path
    attempt_root: Path
    state_root: Path
    evidence_root: Path
    compose_path: Path
    fcos_weight_path: Path
    campaign_config_relative: str

    def validate(self) -> "RemoteEdgePaths":
        repo = _absolute_non_root(self.repository_root, "repository root")
        attempt = _absolute_non_root(self.attempt_root, "attempt root")
        state = _absolute_non_root(self.state_root, "state root")
        evidence = _absolute_non_root(self.evidence_root, "evidence root")
        compose = _absolute_non_root(self.compose_path, "compose path")
        fcos = _absolute_non_root(self.fcos_weight_path, "FCOS weight path")
        _require(state == attempt / "state", "state root must be <attempt>/state")
        _require(evidence == attempt / "evidence",
                 "evidence root must be <attempt>/evidence")
        _require(compose == attempt / "remote_edge.compose.json",
                 "compose path must be attempt-owned")
        expected_fcos = repo / next(
            item.relative_path for item in C.ARTIFACTS if item.name == "torchvision_fcos"
        )
        _require(fcos == expected_fcos, "FCOS mount is not the registered artifact")
        _require(repo != attempt and repo not in attempt.parents,
                 "attempt output must not be nested inside the repository")
        _safe_relative(self.campaign_config_relative, "campaign config")
        return self


@dataclass(frozen=True)
class RemoteEdgeInvocation:
    """Scientific identity and ports supplied by the W10275 coordinator."""

    attempt_id: str
    run_id: str
    cell_id: str
    action_id: int
    allowed_action_ids: tuple[int, ...]
    edge_receive_port: int
    ue_control_host: str
    ue_control_port: int
    edge_compute_cpus: str = ""
    edge_receive_cpus: str = ""

    def validate(self) -> "RemoteEdgeInvocation":
        _require(bool(ATTEMPT_RE.fullmatch(self.attempt_id)), "unsafe attempt id")
        _require(bool(IDENTITY_RE.fullmatch(self.run_id)), "unsafe run id")
        _require(bool(IDENTITY_RE.fullmatch(self.cell_id)), "unsafe cell id")
        _require(0 <= int(self.action_id) < 72, "action id is outside the catalogue")
        _require(bool(self.allowed_action_ids), "allowed action set is empty")
        _require(tuple(sorted(set(self.allowed_action_ids)))
                 == tuple(self.allowed_action_ids),
                 "allowed action ids must be unique and sorted")
        _require(all(0 <= int(value) < 72 for value in self.allowed_action_ids),
                 "allowed action id is outside the catalogue")
        _require(int(self.action_id) in self.allowed_action_ids,
                 "executed action is not allowed")
        for port in (self.edge_receive_port, self.ue_control_port):
            _require(1 <= int(port) <= 65535, "invalid edge/feedback port")
        ue = ipaddress.ip_address(self.ue_control_host)
        _require(ue.version == 4 and ue in ipaddress.ip_network("10.0.0.0/24"),
                 "UE feedback endpoint is outside the OAI UE subnet")
        for value in (self.edge_compute_cpus, self.edge_receive_cpus):
            _require(not value or bool(re.fullmatch(r"[0-9,-]+", value)),
                     "CPU reservation is not a cpuset")
        return self

    @property
    def project_name(self) -> str:
        return PROJECT_PREFIX + self.attempt_id


@dataclass(frozen=True)
class LifecycleCommand:
    purpose: str
    argv: tuple[str, ...]
    expected_returncode: int = 0

    def __post_init__(self) -> None:
        _require(bool(self.purpose) and bool(self.argv), "empty lifecycle command")
        assert_edge_only_command(self.argv)


@dataclass(frozen=True)
class RemoteEdgeLifecyclePlan:
    schema: str
    purpose: str
    binding: C.RemoteRuntimeBinding
    paths: RemoteEdgePaths
    invocation: RemoteEdgeInvocation
    compose_document: Mapping[str, Any]
    compose_sha256: str
    preflight: tuple[LifecycleCommand, ...]
    launch: LifecycleCommand
    post_create: tuple[LifecycleCommand, ...]
    teardown: LifecycleCommand
    live_run_authorized: bool = False

    def as_evidence(self) -> Mapping[str, Any]:
        """Serializable prospective record; contains no inferred remote fact."""
        return {
            "schema": self.schema,
            "purpose": self.purpose,
            "live_run_authorized": self.live_run_authorized,
            "attempt_id": self.invocation.attempt_id,
            "project_name": self.invocation.project_name,
            "repository_root": str(self.paths.repository_root),
            "attempt_root": str(self.paths.attempt_root),
            "state_root": str(self.paths.state_root),
            "evidence_root": str(self.paths.evidence_root),
            "compose_path": str(self.paths.compose_path),
            "compose_sha256": self.compose_sha256,
            "remote_gpu": {
                "model": self.binding.gpu.model,
                "uuid": self.binding.gpu.uuid,
                "memory_total_mib": self.binding.gpu.memory_total_mib,
                "driver_version": self.binding.gpu.driver_version,
            },
            "image": {
                "tag": self.binding.image_tag,
                "manifest_digest": self.binding.image_manifest_digest,
                "config_digest": self.binding.image_config_digest,
                "canonical_inspect_fields_sha256":
                    self.binding.canonical_inspect_fields_sha256,
            },
            "edge_endpoint": C.default_topology().edge_ip,
            "direct_map_endpoint": (
                f"{C.default_topology().map_ip}:{C.default_topology().map_port}"
            ),
            "gt_ingress": {
                "endpoint": f"{C.default_topology().edge_ip}:{GT_PORT}",
                "persistent_tcp": True,
                "authorization": "EXACT_VERIFIED_REWARD_TICKET_IDENTITY",
                "ready_evidence": GT_READY_DESTINATION,
                "final_evidence": GT_FINAL_DESTINATION,
                "cross_host_clock_subtraction": False,
                "policy_deadline_clock_owner": "W10275",
            },
            "ownership": {
                "container": CONTAINER,
                "compose_project": self.invocation.project_name,
                "teardown_may_remove_only_matching_project": True,
            },
        }


def edge_command(paths: RemoteEdgePaths,
                 invocation: RemoteEdgeInvocation) -> list[str]:
    """Direct argv equivalent of the frozen single-worker edge invocation."""
    paths.validate()
    invocation.validate()
    topology = C.default_topology()
    config_container = str(PurePosixPath(REPOSITORY_DESTINATION)
                           / paths.campaign_config_relative)
    return [
        "python3", "-u", "-m", EDGE_MODULE,
        "--remote-binding",
        "/work/abiodun/rl_agent/splitfusion_run4_split_host_l10319_v1/REMOTE_RUNTIME_BINDING_L10319_V1.json",
        "--remote-gpu-evidence", f"{STATE_DESTINATION}/remote_gpu_entry.json",
        "--remote-gt-bind-host", topology.edge_ip,
        "--remote-gt-advertised-host", topology.edge_ip,
        "--remote-gt-port", str(GT_PORT),
        "--remote-gt-ready-evidence", GT_READY_DESTINATION,
        "--remote-gt-final-evidence", GT_FINAL_DESTINATION,
        "--remote-gt-socket-timeout-s", str(GT_SOCKET_TIMEOUT_S),
        "--remote-gt-expectation-timeout-s", str(GT_EXPECTATION_TIMEOUT_S),
        "--edge",
        "--config", config_container,
        "--action-id", str(invocation.action_id),
        "--allowed-action-ids", ",".join(map(str, invocation.allowed_action_ids)),
        "--ready-file", READY_DESTINATION,
        "--edge-port", str(invocation.edge_receive_port),
        "--direct-map-host", topology.map_ip,
        "--direct-map-port", str(topology.map_port),
        "--ue-control-host", invocation.ue_control_host,
        "--ue-control-port", str(invocation.ue_control_port),
        "--edge-segmentation-evidence-dir", EVIDENCE_DESTINATION,
        "--run-id", invocation.run_id,
        "--cell-id", invocation.cell_id,
        f"--edge-compute-cpus={invocation.edge_compute_cpus}",
        f"--edge-receive-cpus={invocation.edge_receive_cpus}",
        "--run4-config", EDGE_CONFIG_DESTINATION,
    ]


def compose_document(*, binding: C.RemoteRuntimeBinding, paths: RemoteEdgePaths,
                     invocation: RemoteEdgeInvocation) -> Mapping[str, Any]:
    """Standalone Compose configuration with the current worktree mount."""
    binding.validate()
    paths.validate()
    invocation.validate()
    topology = C.default_topology()
    labels = {
        "scenesense.owner": PURPOSE,
        "scenesense.attempt_id": invocation.attempt_id,
        "scenesense.image_manifest_digest": binding.image_manifest_digest,
        "scenesense.image_config_digest": binding.image_config_digest,
        "scenesense.image_inspect_sha256": binding.canonical_inspect_fields_sha256,
        "scenesense.gpu_uuid": binding.gpu.uuid,
        "scenesense.repository_root": str(paths.repository_root),
        "scenesense.gt_endpoint": f"{topology.edge_ip}:{GT_PORT}",
    }
    document: Mapping[str, Any] = {
        "name": invocation.project_name,
        "services": {
            SERVICE: {
                "image": binding.image_tag,
                "pull_policy": "never",
                "container_name": CONTAINER,
                "privileged": True,
                "init": True,
                "environment": {
                    "PYTHONPATH": (
                        "/work/abiodun:/work/abiodun/rl_agent/feature_ae"
                    ),
                    "TORCH_HOME": STATE_DESTINATION,
                    "MPLBACKEND": "Agg",
                },
                "volumes": [
                    {"type": "bind", "source": str(paths.repository_root),
                     "target": REPOSITORY_DESTINATION, "read_only": True},
                    {"type": "bind", "source": str(paths.state_root),
                     "target": STATE_DESTINATION, "read_only": False},
                    {"type": "bind", "source": str(paths.evidence_root),
                     "target": EVIDENCE_DESTINATION, "read_only": False},
                    {"type": "bind", "source": str(paths.fcos_weight_path),
                     "target": FCOS_DESTINATION, "read_only": True},
                ],
                "gpus": "all",
                "command": edge_command(paths, invocation),
                "networks": {"public_net": {"ipv4_address": topology.edge_ip}},
                "labels": labels,
            }
        },
        "networks": {"public_net": {"external": True, "name": NETWORK}},
    }
    serialized = json.dumps(document, sort_keys=True)
    _require("../../abiodun" not in serialized, "legacy repository mount leaked")
    _require("build" not in document["services"][SERVICE], "build stanza is forbidden")
    _require(tuple(document["services"]) == (SERVICE,), "foreign service in edge compose")
    return document


def assert_edge_only_command(argv: Sequence[str]) -> None:
    """Fail closed if a prospective command can start broader infrastructure."""
    words = tuple(str(value) for value in argv)
    rendered = " ".join(words).lower()
    for token in ("carla", "nr-softmodem", "nr-uesoftmodem", "cn_start",
                  "oai-gnb", "oai-nr-ue", "phase6_live_runner", "--transmitted-budget"):
        _require(token not in rendered, f"non-edge lifecycle token is forbidden: {token}")
    if "docker" in words and "compose" in words and "up" in words:
        _require("--no-build" in words, "edge launch must refuse builds")
        _require("--pull" in words and words[words.index("--pull") + 1] == "never",
                 "edge launch must refuse pulls")
        _require("--no-deps" in words, "edge launch may not start dependencies")
        _require(words[-1] == SERVICE, "edge launch must select only the edge service")


def build_plan(*, binding: C.RemoteRuntimeBinding, paths: RemoteEdgePaths,
               invocation: RemoteEdgeInvocation) -> RemoteEdgeLifecyclePlan:
    """Build a non-executing edge-only plan from measured/bound remote facts."""
    binding.validate()
    paths.validate()
    invocation.validate()
    document = compose_document(binding=binding, paths=paths, invocation=invocation)
    digest = _canonical_sha256(document)
    compose = str(paths.compose_path)
    project = invocation.project_name
    base = ("sudo", "-n", "docker", "compose", "--project-name", project,
            "-f", compose)
    preflight = (
        LifecycleCommand("verify remote hostname", ("hostname", "-s")),
        LifecycleCommand(
            "measure the bound GPU",
            ("nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
             "--format=csv,noheader,nounits"),
        ),
        LifecycleCommand(
            "inspect portable edge image",
            ("sudo", "-n", "docker", "image", "inspect", binding.image_tag),
        ),
        LifecycleCommand(
            "require the pre-existing CN network without starting it",
            ("sudo", "-n", "docker", "network", "inspect", NETWORK),
        ),
        LifecycleCommand(
            "require no pre-existing edge container",
            ("sudo", "-n", "docker", "container", "inspect", CONTAINER),
            expected_returncode=1,
        ),
        LifecycleCommand("validate standalone compose", base + ("config", "--quiet")),
    )
    launch = LifecycleCommand(
        "create only the remote edge container",
        base + ("up", "-d", "--no-build", "--pull", "never", "--force-recreate",
                "--no-deps", SERVICE),
    )
    post = (
        LifecycleCommand(
            "bind created container identity and ownership",
            ("sudo", "-n", "docker", "container", "inspect", CONTAINER),
        ),
        LifecycleCommand(
            "capture full timestamped edge logs before teardown",
            ("sudo", "-n", "docker", "logs", "--timestamps", CONTAINER),
        ),
    )
    teardown = LifecycleCommand(
        "remove only this Compose project's edge resources",
        base + ("down", "--remove-orphans", "--timeout", "30"),
    )
    return RemoteEdgeLifecyclePlan(
        schema=SCHEMA, purpose=PURPOSE, binding=binding, paths=paths,
        invocation=invocation, compose_document=document, compose_sha256=digest,
        preflight=preflight, launch=launch, post_create=post, teardown=teardown,
        live_run_authorized=False,
    )


def canonical_image_inspect_sha256(document: Mapping[str, Any]) -> str:
    """Hash only the portable fields registered by the transfer audit."""
    selected = {
        name: document.get(name)
        for name in ("Architecture", "Created", "Config", "RootFS", "History", "Os", "Variant")
    }
    return _canonical_sha256(selected)


def validate_remote_image_observation(observation: Mapping[str, Any]) -> None:
    """Validate independently collected image facts; tag equality is insufficient."""
    required = {
        "tag", "image_id", "manifest_digest", "config_digest",
        "canonical_inspect_fields_sha256",
    }
    _require(set(observation) == required, "remote image observation fields drift")
    _require(observation["tag"] == C.EDGE_IMAGE_TAG, "remote image tag drift")
    _require(observation["image_id"] == C.REMOTE_IMAGE_ID, "remote image ID drift")
    _require(observation["manifest_digest"] == C.EDGE_IMAGE_MANIFEST_DIGEST,
             "remote OCI manifest drift")
    _require(observation["config_digest"] == C.EDGE_IMAGE_CONFIG_DIGEST,
             "remote OCI config drift")
    _require(observation["canonical_inspect_fields_sha256"]
             == C.EDGE_IMAGE_CANONICAL_INSPECT_SHA256,
             "remote canonical inspect drift")


def validate_container_observation(observation: Mapping[str, Any], *,
                                   plan: RemoteEdgeLifecyclePlan) -> None:
    """Prove the created container belongs to this attempt before use/teardown."""
    required = {"container_id", "image_id", "project", "labels", "mounts"}
    _require(set(observation) == required, "container observation fields drift")
    _require(bool(str(observation["container_id"])), "container ID is empty")
    _require(observation["image_id"] == C.REMOTE_CONTAINER_IMAGE_ID,
             "created container image drift")
    _require(observation["project"] == plan.invocation.project_name,
             "created container belongs to another Compose project")
    labels = observation["labels"]
    _require(isinstance(labels, Mapping), "container labels are absent")
    expected_labels = plan.compose_document["services"][SERVICE]["labels"]
    _require(all(labels.get(key) == value for key, value in expected_labels.items()),
             "container ownership/binding labels drift")
    mounts = observation["mounts"]
    expected = {
        REPOSITORY_DESTINATION: (str(plan.paths.repository_root), False),
        STATE_DESTINATION: (str(plan.paths.state_root), True),
        EVIDENCE_DESTINATION: (str(plan.paths.evidence_root), True),
        FCOS_DESTINATION: (str(plan.paths.fcos_weight_path), False),
    }
    _require(mounts == expected, "container mount source/mode drift")


def validate_ready_record(document: Mapping[str, Any], *,
                          plan: RemoteEdgeLifecyclePlan) -> None:
    """Validate readiness without treating it as live-run authorization."""
    topology = C.default_topology()
    invocation = plan.invocation
    checks = {
        "schema": document.get("schema") == READY_SCHEMA,
        "architecture": document.get("architecture") == READY_ARCHITECTURE,
        "run4_edge": document.get("run4_edge") is True,
        "action_id": document.get("action_id") == invocation.action_id,
        "tail_device": document.get("tail_device") == "cuda:0",
        "direct_map_host": document.get("direct_map_host") == topology.map_ip,
        "direct_map_port": document.get("direct_map_port") == topology.map_port,
        "ue_control_host": document.get("ue_control_host") == invocation.ue_control_host,
        "ue_control_port": document.get("ue_control_port") == invocation.ue_control_port,
        "quality_spec": document.get("quality_spec_sha256") == QUALITY_SPEC_SHA256,
        "dense_map_off_radio": document.get("dense_label_map_on_radio") is False,
        "objects_off_radio": document.get("object_records_on_radio") is False,
        "evidence": document.get("evaluation_evidence_dir") == EVIDENCE_DESTINATION,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    _require(not failed, f"remote edge ready record drift: {failed}")


def validate_gt_ready_record(document: Mapping[str, Any], *,
                             plan: RemoteEdgeLifecyclePlan) -> None:
    """Bind listener readiness to the same attempt before accepting edge READY."""
    topology = C.default_topology()
    expected = {
        "schema": RGT.SCHEMA,
        "status": "LISTENING",
        "run_id": plan.invocation.run_id,
        "cell_id": plan.invocation.cell_id,
        "bind_host": topology.edge_ip,
        "advertised_endpoint": f"{topology.edge_ip}:{GT_PORT}",
        "max_tickets": RGT.MAX_GT_TICKETS,
        "socket_timeout_s": GT_SOCKET_TIMEOUT_S,
        "expectation_timeout_s": GT_EXPECTATION_TIMEOUT_S,
        "cross_host_clock_subtraction": False,
        "policy_deadline_clock_owner": "W10275",
    }
    _require(dict(document) == expected, "remote GT listener ready record drift")


def validate_gt_final_record(document: Mapping[str, Any], *,
                             plan: RemoteEdgeLifecyclePlan) -> None:
    """Prove the attempt-owned listener stopped cleanly during project teardown."""
    required = {
        "schema", "status", "run_id", "cell_id", "advertised_endpoint",
        "edge_ready_health_checked", "failure", "counters", "thread_alive",
        "cross_host_clock_subtraction", "policy_deadline_clock_owner",
    }
    topology = C.default_topology()
    checks = {
        "fields": set(document) == required,
        "schema": document.get("schema") == RGT.SCHEMA,
        "status": document.get("status") == "STOPPED",
        "run": document.get("run_id") == plan.invocation.run_id,
        "cell": document.get("cell_id") == plan.invocation.cell_id,
        "endpoint": document.get("advertised_endpoint")
                    == f"{topology.edge_ip}:{GT_PORT}",
        "ready_checked": document.get("edge_ready_health_checked") is True,
        "failure": document.get("failure") is None,
        "counters": isinstance(document.get("counters"), Mapping),
        "thread_stopped": document.get("thread_alive") is False,
        "clock": document.get("cross_host_clock_subtraction") is False
                 and document.get("policy_deadline_clock_owner") == "W10275",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    _require(not failed, f"remote GT listener final record drift: {failed}")


def authorize_full_live_run(*_args: Any, **_kwargs: Any) -> None:
    """This package cannot authorize a frame-producing qualification."""
    raise RemoteEdgeLifecycleError(
        "remote edge lifecycle is startup-only; full live run requires a separate gate"
    )


__all__ = [
    "SCHEMA", "PURPOSE", "SERVICE", "CONTAINER", "NETWORK",
    "RemoteEdgeLifecycleError", "RemoteEdgePaths", "RemoteEdgeInvocation",
    "LifecycleCommand", "RemoteEdgeLifecyclePlan", "edge_command",
    "compose_document", "build_plan", "canonical_image_inspect_sha256",
    "validate_remote_image_observation", "validate_container_observation",
    "validate_ready_record", "validate_gt_ready_record", "validate_gt_final_record",
    "authorize_full_live_run",
]
