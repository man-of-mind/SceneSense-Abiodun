"""Create-only L10319 edge startup qualification; never a live run.

This module runs *on L10319*.  It materializes one fresh attempt, reuses the
pure :mod:`remote_edge_lifecycle_v1` plan, starts only its one edge service,
validates both the GT listener and frozen edge READY records, captures bounded
evidence, and immediately tears down only the attempt's Compose project.

No command in this module starts CN, RAN, CARLA, a map service, or a frame
producer.  Importing it performs no I/O and starts no process.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from . import contract as C
from . import remote_edge_entry_v1 as GPU
from . import remote_edge_gt_entry_v1 as RGT
from . import remote_edge_lifecycle_v1 as E


SCHEMA = "scenesense.run4.remote_edge_startup_qualification.v1"
EXECUTE_TOKEN = "SPLITFUSION_RUN4_REMOTE_EDGE_STARTUP_ONLY_V1_EXECUTE"
CAMPAIGN_RELATIVE = (
    "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json"
)
CAMPAIGN_SHA256 = "fd9bdae329ca7e275520f13bce8737d4bdc23e285bff7fa0b23867a14ac4d6e2"
BINDING_RELATIVE = (
    "rl_agent/splitfusion_run4_split_host_l10319_v1/"
    "REMOTE_RUNTIME_BINDING_L10319_V1.json"
)
EDGE_CONFIG_SCHEMA = "scenesense.run4_live_v2.phase6_edge_config.v1"
RESULT_NAME = "REMOTE_EDGE_STARTUP_QUALIFICATION.json"
PLAN_NAME = "REMOTE_EDGE_STARTUP_PLAN.json"
IMAGE_NAME = "REMOTE_EDGE_IMAGE_OBSERVATION.json"
CONTAINER_NAME = "REMOTE_EDGE_CONTAINER_OBSERVATION.json"
LOG_NAME = "remote_edge_logs_tail.txt"
DEFAULT_READY_TIMEOUT_S = 300.0
FIXED_ACTION_ID = 71
FIXED_ALLOWED_ACTIONS = (71,)


class RemoteEdgeStartupError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RemoteEdgeStartupError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _write_create_only(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, indent=1,
                                allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_create_only(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=False)
    with Path(source).open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, destination, follow_symlinks=True)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[[Sequence[str], float], CommandResult]


def subprocess_runner(argv: Sequence[str], timeout_s: float) -> CommandResult:
    completed = subprocess.run(
        list(argv), check=False, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=float(timeout_s))
    return CommandResult(int(completed.returncode), completed.stdout or "",
                         completed.stderr or "")


@dataclass(frozen=True)
class PreparedStartupAttempt:
    binding: C.RemoteRuntimeBinding
    plan: E.RemoteEdgeLifecyclePlan
    result_path: Path


def load_binding(repository_root: Path) -> tuple[C.RemoteRuntimeBinding, str]:
    payload = (Path(repository_root) / BINDING_RELATIVE).read_bytes()
    return C.RemoteRuntimeBinding.from_mapping(json.loads(payload)), hashlib.sha256(
        payload).hexdigest()


def verify_registered_artifacts(
        repository_root: Path, *, hash_file: Callable[[Path], str] = _sha256_file) -> None:
    for artifact in C.ARTIFACTS:
        path = Path(repository_root) / artifact.relative_path
        _require(path.is_file(), f"registered artifact is missing: {artifact.name}")
        _require(hash_file(path) == artifact.sha256,
                 f"registered artifact hash drift: {artifact.name}")


def prepare_attempt(
    *, repository_root: Path, attempt_root: Path, attempt_id: str,
    run_id: str, cell_id: str,
    hash_file: Callable[[Path], str] = _sha256_file,
) -> PreparedStartupAttempt:
    """Create exactly one attempt and seed the frozen edge inputs."""
    repository_root = Path(repository_root).resolve(strict=True)
    attempt_root = Path(attempt_root).resolve(strict=False)
    _require(attempt_root.parent.is_dir(), "attempt parent must already exist")
    binding, binding_sha256 = load_binding(repository_root)
    binding.validate()
    verify_registered_artifacts(repository_root, hash_file=hash_file)

    campaign_path = repository_root / CAMPAIGN_RELATIVE
    _require(campaign_path.is_file(), "frozen campaign config is missing")
    _require(hash_file(campaign_path) == CAMPAIGN_SHA256,
             "frozen campaign config hash drift")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    runtime = campaign.get("runtime") or {}
    measurement = campaign.get("measurement_contract") or {}
    _require(runtime.get("edge_remote_host") == C.default_topology().edge_ip,
             "campaign edge address drift")
    _require(runtime.get("direct_map_ingest_port") == C.default_topology().map_port,
             "campaign map port drift")

    fcos = next(item for item in C.ARTIFACTS if item.name == "torchvision_fcos")
    paths = E.RemoteEdgePaths(
        repository_root=repository_root, attempt_root=attempt_root,
        state_root=attempt_root / "state", evidence_root=attempt_root / "evidence",
        compose_path=attempt_root / "remote_edge.compose.json",
        fcos_weight_path=repository_root / fcos.relative_path,
        campaign_config_relative=CAMPAIGN_RELATIVE)
    invocation = E.RemoteEdgeInvocation(
        attempt_id=attempt_id, run_id=run_id, cell_id=cell_id,
        action_id=FIXED_ACTION_ID, allowed_action_ids=FIXED_ALLOWED_ACTIONS,
        edge_receive_port=int(runtime["edge_receive_port"]),
        ue_control_host=str(runtime["ue_bind_host"]),
        ue_control_port=int(runtime["ue_control_port"]))
    plan = E.build_plan(binding=binding, paths=paths, invocation=invocation)

    attempt_root.mkdir(mode=0o700, exist_ok=False)
    paths.state_root.mkdir(mode=0o755)
    paths.evidence_root.mkdir(mode=0o777)
    os.chmod(paths.evidence_root, 0o777)

    checkpoint = paths.state_root / "hub" / "checkpoints" / paths.fcos_weight_path.name
    _copy_create_only(paths.fcos_weight_path, checkpoint)
    _require(hash_file(checkpoint) == fcos.sha256, "seeded FCOS checkpoint hash drift")

    edge_config = {
        "schema": EDGE_CONFIG_SCHEMA,
        "run_id": run_id, "cell_id": cell_id,
        "evidence_dir": E.EVIDENCE_DESTINATION,
        "report_path": f"{E.EVIDENCE_DESTINATION}/run4_phase6_edge_report.json",
        "match_distance_m": float(measurement["match_distance_m"]),
        "gt_timeout_s": 2.0, "queue_depth": 64,
    }
    _write_create_only(paths.state_root / "run4_phase6_edge_config.json", edge_config)
    compose_bytes = (json.dumps(plan.compose_document, sort_keys=True, indent=1,
                                allow_nan=False) + "\n").encode("utf-8")
    _write_bytes_create_only(paths.compose_path, compose_bytes)
    _require(_canonical_sha256(json.loads(compose_bytes)) == plan.compose_sha256,
             "materialized Compose digest drift")

    plan_record = {
        "schema": SCHEMA, "status": "PREPARED_NOT_EXECUTED",
        "binding_sha256": binding_sha256,
        "campaign_relative": CAMPAIGN_RELATIVE,
        "campaign_sha256": CAMPAIGN_SHA256,
        "seeded_checkpoint": str(checkpoint),
        "seeded_checkpoint_sha256": fcos.sha256,
        "plan": plan.as_evidence(),
        "commands": {
            "preflight": [{"purpose": c.purpose, "argv": list(c.argv),
                            "expected_returncode": c.expected_returncode}
                           for c in plan.preflight],
            "launch": list(plan.launch.argv),
            "post_create": [list(c.argv) for c in plan.post_create],
            "teardown": list(plan.teardown.argv),
        },
        "full_live_run_authorized": False,
    }
    _write_create_only(attempt_root / PLAN_NAME, plan_record)
    return PreparedStartupAttempt(binding=binding, plan=plan,
                                  result_path=attempt_root / RESULT_NAME)


def image_observation_from_inspect(
        stdout: str, binding: C.RemoteRuntimeBinding) -> Mapping[str, Any]:
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RemoteEdgeStartupError("image inspect is not JSON") from exc
    _require(isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], Mapping),
             "image inspect must contain exactly one object")
    row = rows[0]
    observation = {
        "tag": binding.image_tag,
        "image_id": row.get("Id"),
        "manifest_digest": row.get("Id"),
        "config_digest": binding.image_config_digest,
        "canonical_inspect_fields_sha256": E.canonical_image_inspect_sha256(row),
    }
    E.validate_remote_image_observation(observation)
    return observation


def container_observation_from_inspect(stdout: str) -> Mapping[str, Any]:
    try:
        rows = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RemoteEdgeStartupError("container inspect is not JSON") from exc
    _require(isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], Mapping),
             "container inspect must contain exactly one object")
    row = rows[0]
    config = row.get("Config") or {}
    labels = config.get("Labels") or {}
    mounts = {
        str(item.get("Destination")): (str(item.get("Source")), bool(item.get("RW")))
        for item in (row.get("Mounts") or ())
    }
    return {
        "container_id": str(row.get("Id") or ""),
        "image_id": str(row.get("Image") or ""),
        "project": labels.get("com.docker.compose.project"),
        "labels": dict(labels), "mounts": mounts,
    }


def _command_record(purpose: str, argv: Sequence[str], result: CommandResult,
                    expected: int) -> Mapping[str, Any]:
    return {
        "purpose": purpose, "argv": list(argv), "returncode": result.returncode,
        "expected_returncode": expected,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "stdout_tail": result.stdout[-4096:], "stderr_tail": result.stderr[-4096:],
    }


def _run_checked(*, command: E.LifecycleCommand, runner: CommandRunner,
                 ledger: list[Mapping[str, Any]], timeout_s: float) -> CommandResult:
    result = runner(command.argv, timeout_s)
    ledger.append(_command_record(command.purpose, command.argv, result,
                                  command.expected_returncode))
    _require(result.returncode == command.expected_returncode,
             f"command failed: {command.purpose}; rc={result.returncode}")
    return result


def _wait_for_ready(
    prepared: PreparedStartupAttempt, *, runner: CommandRunner,
    ledger: list[Mapping[str, Any]], timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    state = prepared.plan.paths.state_root
    gt_path = state / Path(E.GT_READY_DESTINATION).name
    edge_path = state / Path(E.READY_DESTINATION).name
    deadline, next_liveness = clock() + float(timeout_s), 0.0
    last_json_error = ""
    while clock() < deadline:
        if gt_path.is_file() and edge_path.is_file():
            try:
                gt = json.loads(gt_path.read_text(encoding="utf-8"))
                edge = json.loads(edge_path.read_text(encoding="utf-8"))
                E.validate_gt_ready_record(gt, plan=prepared.plan)
                E.validate_ready_record(edge, plan=prepared.plan)
                _require(gt_path.stat().st_mtime_ns <= edge_path.stat().st_mtime_ns,
                         "edge READY predates GT-listener READY")
                return gt, edge
            except (json.JSONDecodeError, OSError) as exc:
                last_json_error = f"{type(exc).__name__}: {exc}"
        now = clock()
        if now >= next_liveness:
            command = E.LifecycleCommand(
                "require edge container alive during readiness",
                ("sudo", "-n", "docker", "inspect", "-f", "{{.State.Running}}",
                 E.CONTAINER))
            result = _run_checked(command=command, runner=runner, ledger=ledger,
                                  timeout_s=10.0)
            _require(result.stdout.strip() == "true", "edge exited before READY")
            next_liveness = now + 2.0
        sleep(0.1)
    raise RemoteEdgeStartupError(
        f"remote edge did not publish both READY records in {timeout_s}s; "
        f"last_json_error={last_json_error!r}")


ImageObserver = Callable[[str, C.RemoteRuntimeBinding], Mapping[str, Any]]
ContainerObserver = Callable[[str], Mapping[str, Any]]


def qualify_startup(
    prepared: PreparedStartupAttempt, *, runner: CommandRunner = subprocess_runner,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    image_observer: ImageObserver = image_observation_from_inspect,
    container_observer: ContainerObserver = container_observation_from_inspect,
) -> Mapping[str, Any]:
    """Launch, verify READY, and always project-scope teardown the edge."""
    _require(1.0 <= float(timeout_s) <= 600.0, "readiness timeout is outside [1,600]")
    plan = prepared.plan
    ledger: list[Mapping[str, Any]] = []
    report: dict[str, Any] = {
        "schema": SCHEMA, "status": "FAILED", "error": "",
        "attempt_id": plan.invocation.attempt_id,
        "project_name": plan.invocation.project_name,
        "live_run_authorized": False, "frame_producer_started": False,
        "commands": ledger, "cleanup": {},
    }
    launch_attempted = False
    ready_achieved = False
    try:
        preflight_results = []
        for command in plan.preflight:
            result = _run_checked(command=command, runner=runner, ledger=ledger,
                                  timeout_s=30.0)
            preflight_results.append((command, result))
            if command.purpose == "verify remote hostname":
                _require(result.stdout.strip() == plan.binding.hostname,
                         "remote hostname drift")
            elif command.purpose == "measure the bound GPU":
                GPU.validate_measured_gpu(GPU.parse_nvidia_smi_row(result.stdout),
                                          plan.binding)
            elif command.purpose == "inspect portable edge image":
                image = image_observer(result.stdout, plan.binding)
                E.validate_remote_image_observation(image)
                _write_create_only(plan.paths.attempt_root / IMAGE_NAME, image)

        # Compose may create the project before returning an error.  From this
        # point onward teardown is mandatory even when ``up`` itself fails.
        launch_attempted = True
        _run_checked(command=plan.launch, runner=runner, ledger=ledger,
                     timeout_s=60.0)
        inspect_command, logs_command = plan.post_create
        inspected = _run_checked(command=inspect_command, runner=runner,
                                 ledger=ledger, timeout_s=20.0)
        container = container_observer(inspected.stdout)
        E.validate_container_observation(container, plan=plan)
        _write_create_only(plan.paths.attempt_root / CONTAINER_NAME, container)

        gt_ready, edge_ready = _wait_for_ready(
            prepared, runner=runner, ledger=ledger, timeout_s=float(timeout_s))
        logs = _run_checked(command=logs_command, runner=runner, ledger=ledger,
                            timeout_s=20.0)
        _write_bytes_create_only(plan.paths.attempt_root / LOG_NAME,
                                 logs.stdout[-65536:].encode("utf-8"))
        report.update({
            "status": "STARTUP_READY_PENDING_TEARDOWN",
            "image_observation_sha256": _sha256_file(plan.paths.attempt_root / IMAGE_NAME),
            "container_observation_sha256": _sha256_file(
                plan.paths.attempt_root / CONTAINER_NAME),
            "gt_ready_sha256": _sha256_file(
                plan.paths.state_root / Path(E.GT_READY_DESTINATION).name),
            "edge_ready_sha256": _sha256_file(
                plan.paths.state_root / Path(E.READY_DESTINATION).name),
            "gt_endpoint": gt_ready["advertised_endpoint"],
            "edge_architecture": edge_ready["architecture"],
        })
        ready_achieved = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"[:1000]
    finally:
        if launch_attempted:
            try:
                result = _run_checked(command=plan.teardown, runner=runner,
                                      ledger=ledger, timeout_s=60.0)
                report["cleanup"]["project_teardown_returncode"] = result.returncode
                absent = E.LifecycleCommand(
                    "verify attempt edge container absent after teardown",
                    ("sudo", "-n", "docker", "container", "inspect", E.CONTAINER),
                    expected_returncode=1)
                _run_checked(command=absent, runner=runner, ledger=ledger,
                             timeout_s=20.0)
                report["cleanup"]["container_absent"] = True
                if ready_achieved:
                    final_path = (plan.paths.state_root
                                  / Path(E.GT_FINAL_DESTINATION).name)
                    _require(final_path.is_file(),
                             "GT listener final record missing after teardown")
                    final = json.loads(final_path.read_text(encoding="utf-8"))
                    E.validate_gt_final_record(final, plan=plan)
                    report["cleanup"]["gt_listener_stopped"] = True
                    report["cleanup"]["gt_final_sha256"] = _sha256_file(final_path)
                else:
                    report["cleanup"]["gt_listener_stopped"] = (
                        "NOT_REQUIRED_BEFORE_BOTH_READY_RECORDS")
            except BaseException as exc:
                report["cleanup"].setdefault("container_absent", False)
                cleanup_error = f"{type(exc).__name__}: {exc}"[:1000]
                report["cleanup"]["error"] = cleanup_error
                report["error"] = report["error"] or cleanup_error
        else:
            report["cleanup"]["not_launched"] = True
        if (report["status"] == "STARTUP_READY_PENDING_TEARDOWN"
                and report["cleanup"].get("container_absent") is True
                and report["cleanup"].get("gt_listener_stopped") is True):
            report["status"] = "PASS_STARTUP_READY_AND_PROJECT_TEARDOWN_COMPLETE"
        else:
            report["status"] = "FAILED"
        _write_create_only(prepared.result_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--ready-timeout-s", type=float, default=DEFAULT_READY_TIMEOUT_S)
    parser.add_argument("--execute", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live CLI
    args = build_parser().parse_args(argv)
    prepared = prepare_attempt(
        repository_root=args.repository_root, attempt_root=args.attempt_root,
        attempt_id=args.attempt_id, run_id=args.run_id, cell_id=args.cell_id)
    if args.execute != EXECUTE_TOKEN:
        print(json.dumps({"status": "PREPARED_NOT_EXECUTED",
                          "plan": str(prepared.plan.paths.attempt_root / PLAN_NAME),
                          "execute_token_required": EXECUTE_TOKEN}, sort_keys=True))
        return 0
    report = qualify_startup(prepared, timeout_s=args.ready_timeout_s)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
