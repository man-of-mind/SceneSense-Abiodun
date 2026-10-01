"""Fail-closed orchestration boundary for isolated Run-4B/Run-5B validation.

This module does not launch CARLA, OAI, Docker, CUDA, or a socket.  It binds a
sealed run configuration to one exact B-variant actor manifest and drives an
injected lifecycle adapter.  The production lifecycle adapter is deliberately
outside this module; the same abstraction can therefore be exercised with
offline fakes before it is connected to the already-proven split-host
lifecycle.

The accepted semantics are intentionally narrower than the old Phase-6 path:
the edge emits a GT-free operational ACK at usable tail-output readiness,
before the independent map and post-run evidence branches.  Live Q_perc,
quality ACKs, reward tickets, map-install waits, and GT-gated feedback are
unrepresentable in the exact configuration schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Protocol

from .live_adapters_v1 import ActorVariant, BActorManifestV1


CONFIG_SCHEMA = "scenesense.splitfusion.run4b5b.validation_config.v1"
CONFIG_SEAL_SCHEMA = "scenesense.splitfusion.run4b5b.validation_config_seal.v1"
RUNNER_SEMANTICS = "SPLIT_HOST_B_OPERATIONAL_ACK_V1"
ACK_SEMANTICS = "TAIL_OUTPUT_READY__GT_FREE__BEFORE_MAP_AND_EVALUATION"
POSTRUN_SEMANTICS = "CARLA_GT_AND_QPERC_POSTRUN_ONLY__NEVER_LIVE_FEEDBACK"
CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
TRANSMITTED_BUDGET = 300
DEADLINE_NS = 170_000_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HOST_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")
_RUN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_FORBIDDEN_TEXT = (
    "phase6", "quality_ack", "qperc_ack", "live_qperc", "gt_feedback",
    "reward_ticket", "map_install_ack", "evaluation_ack",
)


class ValidationRunnerError(RuntimeError):
    """The B validation orchestration contract was violated."""


class ValidationConfigError(ValidationRunnerError):
    """The sealed run configuration is invalid, foreign, or unsealed."""


class LifecycleError(ValidationRunnerError):
    """The injected lifecycle did not satisfy its orchestration contract."""


def _require(condition: bool, message: str,
             error: type[ValidationRunnerError] = ValidationConfigError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValidationConfigError("configuration is not canonicalizable") from exc


def _sha(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_SHA256_RE.fullmatch(value)),
             f"{field} is not a lowercase SHA-256")
    return value


def _host(value: Any, field: str) -> str:
    _require(type(value) is str and bool(_HOST_RE.fullmatch(value)),
             f"{field} is not a safe host name")
    return value


def _absolute_path(value: Any, field: str) -> str:
    _require(type(value) is str and bool(value), f"{field} is empty")
    path = PurePath(value)
    _require(path.is_absolute() and ".." not in path.parts,
             f"{field} must be an absolute path without parent traversal")
    _require(str(path) not in ("/", "/tmp"), f"{field} is too broad")
    return str(path)


def _reject_legacy_text(raw: Mapping[str, Any]) -> None:
    lowered = _canonical(raw).decode("ascii").lower()
    hit = next((token for token in _FORBIDDEN_TEXT if token in lowered), None)
    _require(hit is None,
             f"legacy GT/quality Phase-6 semantics are forbidden: {hit}")


@dataclass(frozen=True, slots=True)
class SplitHostBindingV1:
    carla_host: str
    ue_host: str
    cn_host: str
    edge_host: str
    ext_dn_host: str
    ack_receiver_host: str
    ack_receiver_port: int

    def __post_init__(self) -> None:
        for field in ("carla_host", "ue_host", "cn_host", "edge_host",
                      "ext_dn_host", "ack_receiver_host"):
            _host(getattr(self, field), field)
        _require(self.carla_host == self.ue_host,
                 "CARLA and UE must share the W10275-side host")
        _require(self.cn_host == self.edge_host == self.ext_dn_host,
                 "CN, edge, and ext-DN must share the L10319-side host")
        _require(self.ue_host != self.edge_host,
                 "UE/front and CN/edge must be isolated on different hosts")
        _require(self.ack_receiver_host == self.ue_host,
                 "the operational ACK receiver must be UE-side")
        _require(type(self.ack_receiver_port) is int
                 and 1024 <= self.ack_receiver_port <= 65535,
                 "ack_receiver_port must be an unprivileged TCP/UDP port")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SplitHostBindingV1":
        fields = set(cls.__dataclass_fields__)
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "split-host binding fields are incomplete or foreign")
        return cls(**{field: raw[field] for field in fields})

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BValidationConfigV1:
    run_id: str
    variant: ActorVariant
    actor_manifest_path: str
    actor_manifest_sha256: str
    output_root: str
    evidence_root: str
    transmitted_budget: int
    split_host: SplitHostBindingV1
    runner_semantics: str
    ack_semantics: str
    postrun_semantics: str
    clock_domain: str
    deadline_ns: int

    def __post_init__(self) -> None:
        _require(type(self.run_id) is str and bool(_RUN_RE.fullmatch(self.run_id)),
                 "run_id is unsafe")
        _require(type(self.variant) is ActorVariant,
                 "variant must be an exact ActorVariant")
        for field in ("actor_manifest_path", "output_root", "evidence_root"):
            object.__setattr__(self, field,
                               _absolute_path(getattr(self, field), field))
        _sha(self.actor_manifest_sha256, "actor_manifest_sha256")
        _require(self.output_root != self.evidence_root,
                 "output and evidence roots must be distinct")
        _require(type(self.transmitted_budget) is int
                 and self.transmitted_budget == TRANSMITTED_BUDGET,
                 "the qualification budget must be exactly 300 transmitted frames")
        _require(type(self.split_host) is SplitHostBindingV1,
                 "split_host must be exactly SplitHostBindingV1")
        _require(self.runner_semantics == RUNNER_SEMANTICS,
                 "runner semantics are not the B operational-ACK path")
        _require(self.ack_semantics == ACK_SEMANTICS,
                 "ACK must be GT-free and precede map/evaluation")
        _require(self.postrun_semantics == POSTRUN_SEMANTICS,
                 "quality evaluation must be post-run only")
        _require(self.clock_domain == CLOCK_DOMAIN,
                 "operational timing must use CLOCK_MONOTONIC_RAW")
        _require(type(self.deadline_ns) is int and self.deadline_ns == DEADLINE_NS,
                 "operational ACK deadline must be exactly 170 ms inclusive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BValidationConfigV1":
        fields = {
            "schema", "run_id", "variant", "actor_manifest_path",
            "actor_manifest_sha256", "output_root", "evidence_root",
            "transmitted_budget", "split_host", "runner_semantics",
            "ack_semantics", "postrun_semantics", "clock_domain",
            "deadline_ns",
        }
        _require(isinstance(raw, Mapping) and set(raw) == fields,
                 "validation config fields are incomplete or foreign")
        _reject_legacy_text(raw)
        _require(raw["schema"] == CONFIG_SCHEMA,
                 "validation config schema drift")
        try:
            variant = ActorVariant(raw["variant"])
        except (TypeError, ValueError) as exc:
            raise ValidationConfigError("unknown B actor variant") from exc
        return cls(
            run_id=raw["run_id"], variant=variant,
            actor_manifest_path=raw["actor_manifest_path"],
            actor_manifest_sha256=raw["actor_manifest_sha256"],
            output_root=raw["output_root"], evidence_root=raw["evidence_root"],
            transmitted_budget=raw["transmitted_budget"],
            split_host=SplitHostBindingV1.from_mapping(raw["split_host"]),
            runner_semantics=raw["runner_semantics"],
            ack_semantics=raw["ack_semantics"],
            postrun_semantics=raw["postrun_semantics"],
            clock_domain=raw["clock_domain"], deadline_ns=raw["deadline_ns"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CONFIG_SCHEMA, "run_id": self.run_id,
            "variant": self.variant.value,
            "actor_manifest_path": self.actor_manifest_path,
            "actor_manifest_sha256": self.actor_manifest_sha256,
            "output_root": self.output_root, "evidence_root": self.evidence_root,
            "transmitted_budget": self.transmitted_budget,
            "split_host": self.split_host.as_dict(),
            "runner_semantics": self.runner_semantics,
            "ack_semantics": self.ack_semantics,
            "postrun_semantics": self.postrun_semantics,
            "clock_domain": self.clock_domain, "deadline_ns": self.deadline_ns,
        }

    def binding_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()


def seal_mapping(config: BValidationConfigV1) -> dict[str, Any]:
    _require(type(config) is BValidationConfigV1,
             "config must be exactly BValidationConfigV1")
    return {
        "schema": CONFIG_SEAL_SCHEMA,
        "binding_sha256": config.binding_sha256(),
        "config": config.as_dict(),
    }


def load_sealed_config(path: Path) -> BValidationConfigV1:
    _require(isinstance(path, Path) and path.is_file(),
             "sealed configuration file does not exist")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationConfigError("sealed configuration is unreadable") from exc
    _require(isinstance(raw, Mapping)
             and set(raw) == {"schema", "binding_sha256", "config"},
             "sealed configuration envelope is incomplete or foreign")
    _require(raw["schema"] == CONFIG_SEAL_SCHEMA,
             "sealed configuration envelope schema drift")
    config = BValidationConfigV1.from_mapping(raw["config"])
    _require(raw["binding_sha256"] == config.binding_sha256(),
             "sealed configuration binding digest differs")
    return config


def load_actor_manifest(config: BValidationConfigV1) -> BActorManifestV1:
    path = Path(config.actor_manifest_path)
    _require(path.is_file(), "actor manifest file does not exist")
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == config.actor_manifest_sha256,
             "actor manifest file digest differs")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValidationConfigError("actor manifest is not JSON") from exc
    manifest = BActorManifestV1.from_mapping(raw)
    _require(manifest.variant is config.variant,
             "actor manifest variant differs from selected run variant")
    return manifest


@dataclass(frozen=True, slots=True)
class LifecycleExecutionV1:
    transmitted_frames: int
    terminal_status: str
    result_sha256: str

    def __post_init__(self) -> None:
        _require(type(self.transmitted_frames) is int
                 and self.transmitted_frames >= 0,
                 "transmitted frame count is invalid", LifecycleError)
        _require(type(self.terminal_status) is str
                 and bool(_RUN_RE.fullmatch(self.terminal_status)),
                 "terminal status is unsafe", LifecycleError)
        _sha(self.result_sha256, "result_sha256")


class SplitHostLifecycleV1(Protocol):
    """Injected seam around the existing proven split-host lifecycle."""

    def preflight(self, config: BValidationConfigV1,
                  manifest: BActorManifestV1) -> None: ...
    def start(self, config: BValidationConfigV1,
              manifest: BActorManifestV1) -> None: ...
    def execute(self, config: BValidationConfigV1,
                manifest: BActorManifestV1) -> LifecycleExecutionV1: ...
    def stop(self, config: BValidationConfigV1) -> None: ...


@dataclass(frozen=True, slots=True)
class OrchestrationResultV1:
    config_binding_sha256: str
    actor_boundary_sha256: str
    transmitted_frames: int
    terminal_status: str
    lifecycle_result_sha256: str


def run_with_lifecycle(config: BValidationConfigV1,
                       lifecycle: SplitHostLifecycleV1) -> OrchestrationResultV1:
    """Validate, run once, and always stop an injected lifecycle after start."""
    _require(type(config) is BValidationConfigV1,
             "config must be exactly BValidationConfigV1")
    for method in ("preflight", "start", "execute", "stop"):
        _require(callable(getattr(lifecycle, method, None)),
                 f"lifecycle lacks {method}", LifecycleError)
    manifest = load_actor_manifest(config)
    lifecycle.preflight(config, manifest)
    started = False
    try:
        lifecycle.start(config, manifest)
        started = True
        execution = lifecycle.execute(config, manifest)
        _require(type(execution) is LifecycleExecutionV1,
                 "lifecycle execute returned a foreign record", LifecycleError)
        _require(execution.transmitted_frames == config.transmitted_budget,
                 "lifecycle did not stop at exactly 300 transmitted frames",
                 LifecycleError)
        _require(execution.terminal_status == "COMPLETE",
                 "lifecycle did not complete", LifecycleError)
        return OrchestrationResultV1(
            config_binding_sha256=config.binding_sha256(),
            actor_boundary_sha256=manifest.actor_boundary_sha256,
            transmitted_frames=execution.transmitted_frames,
            terminal_status=execution.terminal_status,
            lifecycle_result_sha256=execution.result_sha256,
        )
    finally:
        if started:
            lifecycle.stop(config)


class _OfflineDryRunLifecycle:
    """CLI-only proof of orchestration; deliberately launches no service."""

    def __init__(self) -> None:
        self.started = False

    def preflight(self, config: BValidationConfigV1,
                  manifest: BActorManifestV1) -> None:
        return None

    def start(self, config: BValidationConfigV1,
              manifest: BActorManifestV1) -> None:
        self.started = True

    def execute(self, config: BValidationConfigV1,
                manifest: BActorManifestV1) -> LifecycleExecutionV1:
        _require(self.started, "offline lifecycle was not started", LifecycleError)
        digest = hashlib.sha256(_canonical({
            "config": config.binding_sha256(),
            "actor": manifest.actor_boundary_sha256,
            "budget": config.transmitted_budget,
            "offline": True,
        })).hexdigest()
        return LifecycleExecutionV1(config.transmitted_budget, "COMPLETE", digest)

    def stop(self, config: BValidationConfigV1) -> None:
        self.started = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--offline-dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_sealed_config(args.config)
    manifest = load_actor_manifest(config)
    if args.preflight:
        print(json.dumps({
            "status": "B_VALIDATION_PREFLIGHT_PASS",
            "variant": config.variant.value,
            "budget": config.transmitted_budget,
            "config_binding_sha256": config.binding_sha256(),
            "actor_boundary_sha256": manifest.actor_boundary_sha256,
            "services_launched": False,
        }, sort_keys=True))
        return 0
    result = run_with_lifecycle(config, _OfflineDryRunLifecycle())
    print(json.dumps({
        "status": "B_VALIDATION_OFFLINE_DRY_RUN_PASS",
        **{field: getattr(result, field)
           for field in result.__dataclass_fields__},
        "services_launched": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
