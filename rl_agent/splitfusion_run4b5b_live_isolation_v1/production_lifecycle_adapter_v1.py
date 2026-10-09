"""Production split-host lifecycle seam for Run-4B/Run-5B validation.

This module is deliberately an orchestrator, not a third implementation of
the W10275/L10319 lifecycle.  It constructs exact commands for two B-specific
process entry points.  Those entry points must compose the already-qualified
local RAN/CARLA and remote CN/routing/edge authorities.

The entry points do not exist yet.  Consequently :meth:`preflight` fails
before running a command on the current tree.  In particular this adapter
never falls back to the old live-quality child or its GT-gated ACK path.

Importing this module performs no I/O and starts no process.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shlex
import subprocess
from typing import Any, Mapping, Protocol, Sequence

from . import b_validation_runner_v1 as V
from .live_adapters_v1 import BActorManifestV1


ADAPTER_SCHEMA = "scenesense.splitfusion.run4b5b.production_lifecycle_adapter.v1"
REQUEST_SCHEMA = "scenesense.splitfusion.run4b5b.process_request.v1"
UE_RESULT_SCHEMA = "scenesense.splitfusion.run4b5b.b_ue_process_result.v1"
EXECUTE_TOKEN = "SPLITFUSION_RUN4B5B_LIVE_VALIDATION_V1_EXECUTE"

LOCAL_HOST = "W10275.idcc.lab"
REMOTE_HOST = "L10319.idcc.lab"
REMOTE_SSH = "shr_aisvcs@L10319.idcc.lab"

UE_MODULE = (
    "rl_agent.splitfusion_run4b5b_live_isolation_v1.b_ue_process_v1"
)
EDGE_MODULE = (
    "rl_agent.splitfusion_run4b5b_live_isolation_v1.b_edge_process_v1"
)

# These are authorities the missing entry points must compose, rather than
# copy.  They are checked by path without importing them during preflight.
LOCAL_AUTHORITIES = (
    "rl_agent.splitfusion_run4_split_host_l10319_v1.contract",
    "rl_agent.splitfusion_run4_split_host_l10319_v1.local_ran_lifecycle_v1",
    "rl_agent.splitfusion_run4_split_host_l10319_v1.local_ran_executor_v1",
)
REMOTE_AUTHORITIES = (
    "rl_agent.splitfusion_run4_split_host_l10319_v1.contract",
)

# No generated argv or request is allowed to invoke these historical paths.
FORBIDDEN_RUNTIME_TOKENS = (
    "phase6_live_child",
    "remote_edge_gt_entry",
    "quality_feedback_probe",
    "quality_ack",
    "gt_feedback",
    "reward_ticket",
)


class ProductionLifecycleError(V.LifecycleError):
    """The production lifecycle plan or execution is unsafe."""


class MissingBEntrypointError(ProductionLifecycleError):
    """One or both B-specific runnable process entry points are absent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionLifecycleError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ProductionLifecycleError("process request is not canonical") from exc


def _module_relative_path(module: str) -> Path:
    _require(type(module) is str and module.startswith("rl_agent."),
             "process module is outside rl_agent")
    parts = module.split(".")
    _require(all(part and part.replace("_", "a").isalnum() for part in parts),
             "process module name is unsafe")
    return Path(*parts).with_suffix(".py")


def _absolute_nonroot(path: Path, label: str) -> Path:
    path = Path(path)
    _require(path.is_absolute(), f"{label} must be absolute")
    normalized = path.resolve(strict=False)
    _require(str(normalized) not in {"/", "/home", "/tmp"},
             f"{label} is too broad")
    return normalized


def _reject_legacy(value: Any) -> None:
    lowered = _canonical(value).decode("ascii").lower()
    hit = next((token for token in FORBIDDEN_RUNTIME_TOKENS
                if token in lowered), None)
    _require(hit is None, f"legacy live-quality runtime is forbidden: {hit}")


@dataclass(frozen=True, slots=True)
class CommandV1:
    host: str
    purpose: str
    argv: tuple[str, ...]
    timeout_s: float

    def __post_init__(self) -> None:
        _require(self.host in {LOCAL_HOST, REMOTE_HOST}, "command host drift")
        _require(type(self.purpose) is str and bool(self.purpose),
                 "command purpose is empty")
        _require(type(self.argv) is tuple and bool(self.argv)
                 and all(type(item) is str and item for item in self.argv),
                 "command argv is invalid")
        _require(type(self.timeout_s) in {int, float}
                 and 1 <= float(self.timeout_s) <= 1200,
                 "command timeout is outside [1,1200]")
        _reject_legacy({"purpose": self.purpose, "argv": self.argv})


@dataclass(frozen=True, slots=True)
class ProductionLifecycleSettingsV1:
    local_repository: Path
    remote_repository: Path
    remote_attempt_base: Path
    python_executable: str = "/usr/bin/python3"
    ssh_executable: str = "/usr/bin/ssh"
    remote_ssh: str = REMOTE_SSH
    execution_timeout_s: float = 900.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_repository",
                           _absolute_nonroot(self.local_repository,
                                             "local repository"))
        object.__setattr__(self, "remote_repository",
                           _absolute_nonroot(self.remote_repository,
                                             "remote repository"))
        object.__setattr__(self, "remote_attempt_base",
                           _absolute_nonroot(self.remote_attempt_base,
                                             "remote attempt base"))
        _require(self.remote_repository != self.remote_attempt_base
                 and self.remote_repository not in self.remote_attempt_base.parents,
                 "remote attempt output must be outside the repository")
        for value, label in ((self.python_executable, "python executable"),
                             (self.ssh_executable, "ssh executable")):
            parsed = Path(value)
            _require(parsed.is_absolute() and ".." not in parsed.parts,
                     f"{label} must be an absolute path")
        _require(self.remote_ssh == REMOTE_SSH, "remote SSH identity drift")
        _require(type(self.execution_timeout_s) in {int, float}
                 and 120 <= float(self.execution_timeout_s) <= 1200,
                 "execution timeout is outside [120,1200]")


@dataclass(frozen=True, slots=True)
class ProductionLifecyclePlanV1:
    schema: str
    run_id: str
    local_request_sha256: str
    remote_request_sha256: str
    preflight_commands: tuple[CommandV1, ...]
    remote_start: CommandV1
    local_run: CommandV1
    local_stop: CommandV1
    remote_stop: CommandV1

    def __post_init__(self) -> None:
        _require(self.schema == ADAPTER_SCHEMA, "lifecycle plan schema drift")
        _require(bool(self.preflight_commands), "preflight command set is empty")
        for value in (self.local_request_sha256, self.remote_request_sha256):
            _require(type(value) is str and len(value) == 64
                     and all(ch in "0123456789abcdef" for ch in value),
                     "request digest is invalid")
        _require(self.remote_start.host == REMOTE_HOST,
                 "remote start command is not remote")
        _require(self.local_run.host == LOCAL_HOST,
                 "UE run command is not local")
        _require(self.local_stop.host == LOCAL_HOST
                 and self.remote_stop.host == REMOTE_HOST,
                 "teardown host order drift")
        _reject_legacy(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        def command(value: CommandV1) -> dict[str, Any]:
            return {"host": value.host, "purpose": value.purpose,
                    "argv": list(value.argv), "timeout_s": value.timeout_s}
        return {
            "schema": self.schema, "run_id": self.run_id,
            "local_request_sha256": self.local_request_sha256,
            "remote_request_sha256": self.remote_request_sha256,
            "preflight_commands": [command(row)
                                   for row in self.preflight_commands],
            "remote_start": command(self.remote_start),
            "local_run": command(self.local_run),
            "local_stop": command(self.local_stop),
            "remote_stop": command(self.remote_stop),
        }


def _encoded_request(*, role: str, config: V.BValidationConfigV1,
                     manifest: BActorManifestV1,
                     settings: ProductionLifecycleSettingsV1) -> tuple[str, str]:
    _require(role in {"UE_FRONT", "CN_EDGE"}, "unknown process role")
    authorities = LOCAL_AUTHORITIES if role == "UE_FRONT" else REMOTE_AUTHORITIES
    value = {
        "schema": REQUEST_SCHEMA,
        "role": role,
        "run_id": config.run_id,
        "variant": config.variant.value,
        "config_binding_sha256": config.binding_sha256(),
        "actor_boundary_sha256": manifest.actor_boundary_sha256,
        "feature_schema_sha256": manifest.feature_schema_sha256,
        "transmitted_budget": config.transmitted_budget,
        "deadline_ns": config.deadline_ns,
        "ack_semantics": config.ack_semantics,
        "postrun_semantics": config.postrun_semantics,
        "clock_domain": config.clock_domain,
        "split_host": config.split_host.as_dict(),
        "output_root": config.output_root if role == "UE_FRONT" else None,
        "evidence_root": config.evidence_root if role == "UE_FRONT" else None,
        "actor_manifest_path": (config.actor_manifest_path
                                if role == "UE_FRONT" else None),
        "remote_attempt_root": (
            str(settings.remote_attempt_base / config.run_id)
            if role == "CN_EDGE" else None
        ),
        "required_authority_modules": list(authorities),
        "old_live_quality_runtime_permitted": False,
    }
    _reject_legacy(value)
    payload = _canonical(value)
    return (base64.urlsafe_b64encode(payload).decode("ascii"),
            hashlib.sha256(payload).hexdigest())


def _remote_argv(settings: ProductionLifecycleSettingsV1,
                 inner: Sequence[str]) -> tuple[str, ...]:
    command = ("cd " + shlex.quote(str(settings.remote_repository))
               + " && exec "
               + " ".join(shlex.quote(item) for item in inner))
    return (settings.ssh_executable, "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", settings.remote_ssh,
            "/usr/bin/bash", "-lc", command)


def build_plan(config: V.BValidationConfigV1,
               manifest: BActorManifestV1,
               settings: ProductionLifecycleSettingsV1,
               ) -> ProductionLifecyclePlanV1:
    """Construct commands only; do not inspect a host or launch a process."""
    _require(type(config) is V.BValidationConfigV1, "config type is foreign")
    _require(type(manifest) is BActorManifestV1, "actor manifest type is foreign")
    _require(type(settings) is ProductionLifecycleSettingsV1,
             "lifecycle settings type is foreign")
    _require(config.split_host.carla_host == LOCAL_HOST
             and config.split_host.ue_host == LOCAL_HOST
             and config.split_host.ack_receiver_host == LOCAL_HOST,
             "local host binding differs from W10275")
    _require(config.split_host.cn_host == REMOTE_HOST
             and config.split_host.edge_host == REMOTE_HOST
             and config.split_host.ext_dn_host == REMOTE_HOST,
             "remote host binding differs from L10319")
    _require(manifest.variant is config.variant, "manifest variant drift")

    local_request, local_digest = _encoded_request(
        role="UE_FRONT", config=config, manifest=manifest, settings=settings)
    remote_request, remote_digest = _encoded_request(
        role="CN_EDGE", config=config, manifest=manifest, settings=settings)

    local_module_path = settings.local_repository / _module_relative_path(UE_MODULE)
    local_authority_paths = tuple(
        settings.local_repository / _module_relative_path(module)
        for module in LOCAL_AUTHORITIES
    )
    remote_paths = (_module_relative_path(EDGE_MODULE), *(
        _module_relative_path(module) for module in REMOTE_AUTHORITIES))

    preflight: list[CommandV1] = [
        CommandV1(LOCAL_HOST, f"verify local source {path}",
                  ("/usr/bin/test", "-f", str(path)), 10.0)
        for path in (local_module_path, *local_authority_paths)
    ]
    for relative in remote_paths:
        preflight.append(CommandV1(
            REMOTE_HOST, f"verify remote source {relative}",
            _remote_argv(settings, ("/usr/bin/test", "-f",
                                    str(settings.remote_repository / relative))),
            20.0))

    local_prefix = ("/usr/bin/env", "-u", "PYTHONPATH",
                    settings.python_executable, "-m", UE_MODULE)
    remote_prefix = ("/usr/bin/env", "-u", "PYTHONPATH",
                     settings.python_executable, "-m", EDGE_MODULE)
    preflight.extend((
        CommandV1(LOCAL_HOST, "preflight B UE/front process",
                  (*local_prefix, "preflight", "--request-b64", local_request),
                  120.0),
        CommandV1(REMOTE_HOST, "preflight B CN/edge process",
                  _remote_argv(settings, (*remote_prefix, "preflight",
                                         "--request-b64", remote_request)),
                  180.0),
    ))
    return ProductionLifecyclePlanV1(
        schema=ADAPTER_SCHEMA, run_id=config.run_id,
        local_request_sha256=local_digest,
        remote_request_sha256=remote_digest,
        preflight_commands=tuple(preflight),
        remote_start=CommandV1(
            REMOTE_HOST, "start held B CN/edge process",
            _remote_argv(settings, (*remote_prefix, "start",
                                    "--request-b64", remote_request,
                                    "--execute", EXECUTE_TOKEN)), 420.0),
        local_run=CommandV1(
            LOCAL_HOST, "run B UE/CARLA/RAN process",
            (*local_prefix, "run", "--request-b64", local_request,
             "--execute", EXECUTE_TOKEN), settings.execution_timeout_s),
        local_stop=CommandV1(
            LOCAL_HOST, "stop B UE/CARLA/RAN process",
            (*local_prefix, "stop", "--request-b64", local_request,
             "--execute", EXECUTE_TOKEN), 180.0),
        remote_stop=CommandV1(
            REMOTE_HOST, "stop held B CN/edge process",
            _remote_argv(settings, (*remote_prefix, "stop",
                                    "--request-b64", remote_request,
                                    "--execute", EXECUTE_TOKEN)), 420.0),
    )


class LifecycleBackendV1(Protocol):
    def preflight(self, plan: ProductionLifecyclePlanV1) -> None: ...
    def start(self, plan: ProductionLifecyclePlanV1) -> None: ...
    def execute(self, plan: ProductionLifecyclePlanV1) -> Mapping[str, Any]: ...
    def stop(self, plan: ProductionLifecyclePlanV1) -> None: ...


class SubprocessLifecycleBackendV1:
    """Bounded command executor; B entry points own host-specific cleanup."""

    def __init__(self) -> None:
        self._plan: ProductionLifecyclePlanV1 | None = None

    @staticmethod
    def _run(command: CommandV1) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            command.argv, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=command.timeout_s, check=False,
        )
        _require(result.returncode == 0,
                 f"{command.purpose} failed with rc={result.returncode}: "
                 f"{result.stderr.decode('utf-8', 'replace')[-1200:]}")
        return result

    def preflight(self, plan: ProductionLifecyclePlanV1) -> None:
        for command in plan.preflight_commands:
            self._run(command)

    def start(self, plan: ProductionLifecyclePlanV1) -> None:
        self._run(plan.remote_start)
        self._plan = plan

    def execute(self, plan: ProductionLifecyclePlanV1) -> Mapping[str, Any]:
        _require(self._plan is plan, "backend was not started for this plan")
        result = self._run(plan.local_run)
        lines = [line for line in result.stdout.decode("utf-8", "replace").splitlines()
                 if line.lstrip().startswith("{")]
        _require(bool(lines), "B UE process emitted no result JSON")
        try:
            value = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise ProductionLifecycleError("B UE result is not JSON") from exc
        _require(type(value) is dict, "B UE result is not an object")
        return value

    def stop(self, plan: ProductionLifecyclePlanV1) -> None:
        errors: list[str] = []
        for command in (plan.local_stop, plan.remote_stop):
            try:
                self._run(command)
            except BaseException as exc:  # preserve both cleanup attempts
                errors.append(f"{command.purpose}: {type(exc).__name__}: {exc}")
        self._plan = None
        if errors:
            raise ProductionLifecycleError("; ".join(errors))


class ProductionSplitHostLifecycleV1:
    """Implementation of :class:`b_validation_runner_v1.SplitHostLifecycleV1`."""

    def __init__(self, settings: ProductionLifecycleSettingsV1, *,
                 backend: LifecycleBackendV1 | None = None) -> None:
        _require(type(settings) is ProductionLifecycleSettingsV1,
                 "settings type is foreign")
        self.settings = settings
        self.backend = backend or SubprocessLifecycleBackendV1()
        self._plan: ProductionLifecyclePlanV1 | None = None
        self._started = False

    def _missing_local_entrypoints(self) -> tuple[str, ...]:
        missing: list[str] = []
        for module in (UE_MODULE, EDGE_MODULE):
            path = self.settings.local_repository / _module_relative_path(module)
            if not path.is_file():
                missing.append(module)
        return tuple(missing)

    def preflight(self, config: V.BValidationConfigV1,
                  manifest: BActorManifestV1) -> None:
        missing = self._missing_local_entrypoints()
        if missing:
            raise MissingBEntrypointError(
                "missing B process entrypoints: " + ", ".join(missing))
        _require(not Path(config.output_root).exists(),
                 "validation output root must be create-only")
        _require(not Path(config.evidence_root).exists(),
                 "validation evidence root must be create-only")
        plan = build_plan(config, manifest, self.settings)
        self.backend.preflight(plan)
        self._plan = plan

    def start(self, config: V.BValidationConfigV1,
              manifest: BActorManifestV1) -> None:
        _require(self._plan is not None and not self._started,
                 "lifecycle was not preflighted or is already started")
        try:
            self.backend.start(self._plan)
            self._started = True
        except BaseException:
            # start() may have made the remote edge live before failing.  The
            # runner only calls stop after a successful start, so clean here.
            try:
                self.backend.stop(self._plan)
            except BaseException:
                pass
            raise

    def execute(self, config: V.BValidationConfigV1,
                manifest: BActorManifestV1) -> V.LifecycleExecutionV1:
        _require(self._plan is not None and self._started,
                 "lifecycle execute called before start")
        value = self.backend.execute(self._plan)
        fields = {"schema", "run_id", "variant", "transmitted_frames",
                  "terminal_status", "result_sha256",
                  "config_binding_sha256", "actor_boundary_sha256"}
        _require(set(value) == fields, "B UE result fields are incomplete or foreign")
        _require(value["schema"] == UE_RESULT_SCHEMA, "B UE result schema drift")
        _require(value["run_id"] == config.run_id, "B UE result run drift")
        _require(value["variant"] == config.variant.value,
                 "B UE result variant drift")
        _require(value["config_binding_sha256"] == config.binding_sha256(),
                 "B UE result config binding drift")
        _require(value["actor_boundary_sha256"]
                 == manifest.actor_boundary_sha256,
                 "B UE result actor boundary drift")
        return V.LifecycleExecutionV1(
            transmitted_frames=value["transmitted_frames"],
            terminal_status=value["terminal_status"],
            result_sha256=value["result_sha256"],
        )

    def stop(self, config: V.BValidationConfigV1) -> None:
        if self._plan is None:
            return
        try:
            self.backend.stop(self._plan)
        finally:
            self._started = False
            self._plan = None


def missing_process_entrypoints(settings: ProductionLifecycleSettingsV1,
                                ) -> tuple[str, ...]:
    """Read-only status helper used by readiness reporting and tests."""
    lifecycle = ProductionSplitHostLifecycleV1(settings)
    return lifecycle._missing_local_entrypoints()


__all__ = [
    "ADAPTER_SCHEMA", "REQUEST_SCHEMA", "UE_RESULT_SCHEMA", "EXECUTE_TOKEN",
    "UE_MODULE", "EDGE_MODULE", "LOCAL_AUTHORITIES", "REMOTE_AUTHORITIES",
    "ProductionLifecycleError", "MissingBEntrypointError", "CommandV1",
    "ProductionLifecycleSettingsV1", "ProductionLifecyclePlanV1",
    "LifecycleBackendV1", "SubprocessLifecycleBackendV1",
    "ProductionSplitHostLifecycleV1", "build_plan",
    "missing_process_entrypoints",
]
