from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .common import (
    CARLA_BIN,
    DEFAULT_CONFIG,
    NEU_COLLAB_ROOT,
    PROJECT_PYTHON,
    WORKFLOW_ROOT,
    append_jsonl,
    create_experiment_dir,
    load_config,
    read_manifest,
    run_subprocess,
    save_json,
    setup_logger,
    subprocess_env,
    utc_iso,
    write_yaml,
)
from .evaluate_fusion import find_best_checkpoint


STOP_REQUESTED = False


def _handle_signal(signum, frame) -> None:  # type: ignore[no-untyped-def]
    global STOP_REQUESTED
    STOP_REQUESTED = True


def update_manifest(exp_dir: Path, updates: Dict) -> None:
    path = exp_dir / "manifest.json"
    manifest: Dict = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.update(updates)
    manifest["updated_at"] = utc_iso()
    save_json(path, manifest)


def emit_status(exp_dir: Path, stage: str, status: str, **extra: object) -> None:
    append_jsonl(
        exp_dir / "status.jsonl",
        {"timestamp": utc_iso(), "stage": stage, "status": status, **extra},
    )


def check_stop(exp_dir: Path) -> None:
    if STOP_REQUESTED or (exp_dir / "stop_requested").exists():
        update_manifest(exp_dir, {"status": "stopping", "stopping_at": utc_iso()})
        emit_status(exp_dir, "supervisor", "stop_requested")
        raise KeyboardInterrupt("Stop requested.")


def import_carla():
    import carla  # type: ignore

    return carla


def check_carla(config: Dict, timeout_s: float = 3.0) -> Tuple[bool, str]:
    try:
        carla = import_carla()
        client = carla.Client(config["carla"].get("host", "127.0.0.1"), int(config["carla"].get("port", 2000)))
        client.set_timeout(float(timeout_s))
        world = client.get_world()
        return True, str(world.get_map().name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def start_carla_server(config: Dict, exp_dir: Path, log) -> Optional[subprocess.Popen]:
    command = str(config["carla"].get("server_command", "")).strip()
    if not command:
        log("No CARLA server command configured; waiting for an external CARLA server.")
        return None
    path = Path(command).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CARLA server command not found: {path}")
    log_path = exp_dir / "logs" / "carla_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Starting CARLA server with {path}; log={log_path}")
    log_fh = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [str(path)],
        cwd=str(path.parent),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    append_jsonl(exp_dir / "carla_restart_events.jsonl", {"timestamp": utc_iso(), "event": "carla_start", "pid": proc.pid})
    return proc


def terminate_carla(carla_proc: Optional[subprocess.Popen], exp_dir: Path, log) -> None:
    if carla_proc is None or carla_proc.poll() is not None:
        return
    log("Stopping CARLA server started by this supervisor.")
    append_jsonl(exp_dir / "carla_restart_events.jsonl", {"timestamp": utc_iso(), "event": "carla_stop", "pid": carla_proc.pid})
    try:
        os.killpg(carla_proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if carla_proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(carla_proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def ensure_carla(config: Dict, exp_dir: Path, log, carla_proc: Optional[subprocess.Popen]) -> Optional[subprocess.Popen]:
    ok, detail = check_carla(config)
    if ok:
        log(f"CARLA reachable: {detail}")
        emit_status(exp_dir, "carla", "reachable", detail=detail)
        return carla_proc
    append_jsonl(exp_dir / "carla_restart_events.jsonl", {"timestamp": utc_iso(), "event": "carla_unreachable", "detail": detail})
    log(f"CARLA unreachable ({detail}); starting/restarting server.")
    if carla_proc is None or carla_proc.poll() is not None:
        carla_proc = start_carla_server(config, exp_dir, log)
    timeout_s = float(config["carla"].get("startup_timeout_s", 180))
    deadline = time.monotonic() + timeout_s
    last_detail = detail
    while time.monotonic() < deadline:
        check_stop(exp_dir)
        time.sleep(5.0)
        ok, last_detail = check_carla(config)
        if ok:
            append_jsonl(exp_dir / "carla_restart_events.jsonl", {"timestamp": utc_iso(), "event": "carla_reachable_after_restart", "detail": last_detail})
            log(f"CARLA initialized: {last_detail}")
            emit_status(exp_dir, "carla", "reachable_after_restart", detail=last_detail)
            return carla_proc
    append_jsonl(exp_dir / "carla_restart_events.jsonl", {"timestamp": utc_iso(), "event": "carla_restart_timeout", "detail": last_detail})
    raise RuntimeError(f"CARLA did not become reachable within {timeout_s:.0f}s: {last_detail}")


def run_stage(
    name: str,
    command: Sequence[str],
    exp_dir: Path,
    env: Dict[str, str],
    log,
) -> int:
    check_stop(exp_dir)
    stage_log = exp_dir / "logs" / f"{name}.log"
    log(f"Starting stage {name}.")
    emit_status(exp_dir, name, "started", log=str(stage_log))
    update_manifest(exp_dir, {"stage": name, f"{name}_started_at": utc_iso()})
    code = run_subprocess(command, cwd=WORKFLOW_ROOT, log_path=stage_log, env=env, stop_file=exp_dir / "stop_requested")
    update_manifest(exp_dir, {f"{name}_finished_at": utc_iso(), f"{name}_return_code": code})
    emit_status(exp_dir, name, "finished", return_code=code, log=str(stage_log))
    log(f"Stage {name} finished with return code {code}; log={stage_log}")
    if (exp_dir / "stop_requested").exists():
        raise KeyboardInterrupt("Stop requested.")
    return code


def run_collection_with_recovery(config: Dict, config_path: Path, exp_dir: Path, env: Dict[str, str], log) -> Optional[subprocess.Popen]:
    carla_proc: Optional[subprocess.Popen] = None
    max_restarts = int(config["carla"].get("max_restarts", 12))
    attempts = 0
    command = [
        str(PROJECT_PYTHON),
        "-m",
        "pole_lraspp_multimodal_fusion.collect_dataset",
        "--config",
        str(config_path),
        "--experiment-dir",
        str(exp_dir),
    ]
    while True:
        check_stop(exp_dir)
        carla_proc = ensure_carla(config, exp_dir, log, carla_proc)
        code = run_stage("collection", command, exp_dir, env, log)
        if code == 0:
            return carla_proc
        attempts += 1
        append_jsonl(
            exp_dir / "carla_restart_events.jsonl",
            {"timestamp": utc_iso(), "event": "collection_stage_failed", "attempt": attempts, "return_code": code},
        )
        if attempts > max_restarts:
            raise RuntimeError(f"Collection failed after {attempts} attempts.")
        cooldown = float(config["carla"].get("restart_cooldown_s", 20))
        log(f"Collection failed; pausing {cooldown:.0f}s before CARLA recovery attempt {attempts}/{max_restarts}.")
        emit_status(exp_dir, "collection", "retry_wait", attempt=attempts, cooldown_s=cooldown)
        time.sleep(cooldown)


def trial_command(config_path: Path, exp_dir: Path, trial: Dict, budget_hours: float) -> List[str]:
    return [
        str(PROJECT_PYTHON),
        "-m",
        "pole_lraspp_multimodal_fusion.train_fusion",
        "--config",
        str(config_path),
        "--experiment-dir",
        str(exp_dir),
        "--trial-json",
        json.dumps(trial, sort_keys=True),
        "--training-budget-hours",
        str(float(budget_hours)),
    ]


def run_training_sweep(config: Dict, config_path: Path, exp_dir: Path, env: Dict[str, str], log) -> None:
    trials = list(config.get("training", {}).get("trials", []))
    if not trials:
        raise RuntimeError("No training trials configured.")
    total_budget = float(config.get("training_budget_hours", 3.25))
    per_trial_budget = max(0.1, total_budget / max(1, len(trials)))
    for trial in trials:
        check_stop(exp_dir)
        name = str(trial.get("name", "trial"))
        code = run_stage(f"train_{name}", trial_command(config_path, exp_dir, trial, per_trial_budget), exp_dir, env, log)
        if code == 0:
            continue
        batch_size = int(trial.get("batch_size", config["training"].get("batch_size", 6)))
        if batch_size <= 1:
            raise RuntimeError(f"Training trial {name} failed and cannot reduce batch size further.")
        retry_trial = dict(trial)
        retry_trial["batch_size"] = max(1, batch_size // 2)
        retry_trial["name"] = f"{name}_bs{retry_trial['batch_size']}_retry"
        append_jsonl(
            exp_dir / "training_retry_events.jsonl",
            {
                "timestamp": utc_iso(),
                "event": "retry_with_smaller_batch",
                "trial": name,
                "return_code": code,
                "retry_trial": retry_trial,
            },
        )
        retry_code = run_stage(f"train_{retry_trial['name']}", trial_command(config_path, exp_dir, retry_trial, per_trial_budget), exp_dir, env, log)
        if retry_code != 0:
            raise RuntimeError(f"Training trial {name} failed after smaller-batch retry.")


def run_evaluation(config_path: Path, exp_dir: Path, env: Dict[str, str], log) -> Path:
    best_checkpoint = find_best_checkpoint(exp_dir)
    log(f"Selected best fusion checkpoint: {best_checkpoint}")
    for split in ("val", "test"):
        command = [
            str(PROJECT_PYTHON),
            "-m",
            "pole_lraspp_multimodal_fusion.evaluate_fusion",
            "--config",
            str(config_path),
            "--experiment-dir",
            str(exp_dir),
            "--checkpoint",
            str(best_checkpoint),
            "--split",
            split,
        ]
        code = run_stage(f"evaluate_{split}", command, exp_dir, env, log)
        if code != 0:
            raise RuntimeError(f"Evaluation failed for split={split}")
    return best_checkpoint


def load_metric_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def generate_final_report(exp_dir: Path, best_checkpoint: Path) -> Path:
    manifest_rows = read_manifest(exp_dir / "dataset" / "manifest.csv")
    split_counts: Dict[str, int] = {}
    for row in manifest_rows:
        split = row.get("split", "unknown")
        split_counts[split] = split_counts.get(split, 0) + 1
    val_metrics = load_metric_json(exp_dir / "metrics" / "val_fusion_evaluation_metrics.json")
    test_metrics = load_metric_json(exp_dir / "metrics" / "test_fusion_evaluation_metrics.json")
    summaries = []
    for path in sorted((exp_dir / "checkpoints").glob("*/trial_summary.json")):
        summary = load_metric_json(path)
        if summary:
            summaries.append(summary)
    summaries.sort(key=lambda item: float(item.get("best_selection_score", item.get("best_miou", -math.inf))), reverse=True)
    report_path = exp_dir / "final_report.txt"
    lines = [
        "Traffic-Light-Pole RGB+Radar LR-ASPP Learned Localization Training Report",
        f"generated_at: {utc_iso()}",
        f"experiment_dir: {exp_dir}",
        f"dataset_samples: {len(manifest_rows)}",
        f"split_counts: {json.dumps(split_counts, sort_keys=True)}",
        f"best_checkpoint: {best_checkpoint}",
        "",
        "Best Trials:",
    ]
    for summary in summaries[:5]:
        trial = summary.get("trial", {})
        lines.append(
            f"- {trial.get('name', 'trial')}: "
            f"best_selection_score={float(summary.get('best_selection_score', float('nan'))):.4f}, "
            f"best_val_miou={float(summary.get('best_miou', float('nan'))):.4f}"
        )
    lines.extend(
        [
            "",
            "Validation Metrics:",
            json.dumps(val_metrics, indent=2, sort_keys=True),
            "",
            "Test Metrics:",
            json.dumps(test_metrics, indent=2, sort_keys=True),
            "",
            "Key artifact locations:",
            f"- dataset manifest: {exp_dir / 'dataset' / 'manifest.csv'}",
            f"- object boxes: {exp_dir / 'dataset' / 'object_boxes.csv'}",
            f"- radar tensors: {exp_dir / 'dataset' / 'radar_tensor'}",
            f"- radar points: {exp_dir / 'dataset' / 'radar_points'}",
            f"- checkpoints: {exp_dir / 'checkpoints'}",
            f"- metrics: {exp_dir / 'metrics'}",
            f"- learned object metrics: {exp_dir / 'metrics' / 'test_learned_object_metrics.csv'}",
            f"- figures: {exp_dir / 'figures'}",
            f"- supervisor log: {exp_dir / 'supervisor.log'}",
            f"- status stream: {exp_dir / 'status.jsonl'}",
            f"- CARLA restart events: {exp_dir / 'carla_restart_events.jsonl'}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_pipeline(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    config = load_config(args.config)
    if args.runtime_budget_hours > 0:
        config["runtime_budget_hours"] = float(args.runtime_budget_hours)
    if args.carla_bin:
        config.setdefault("carla", {})["server_command"] = str(Path(args.carla_bin).expanduser())
    exp_dir = create_experiment_dir(config, args.experiment_dir)
    log = setup_logger(exp_dir / "supervisor.log")
    (exp_dir / "supervisor.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
    config_path = exp_dir / "resolved_config.yaml"
    write_yaml(config_path, config)
    save_json(exp_dir / "resolved_config.json", config)
    update_manifest(
        exp_dir,
        {
            "experiment_id": exp_dir.name,
            "experiment_dir": str(exp_dir),
            "status": "running",
            "started_at": utc_iso(),
            "resolved_config": str(config_path),
            "python": str(PROJECT_PYTHON),
            "execution_boundary": "long launcher must be started from user host shell, not Codex sandbox",
        },
    )
    emit_status(exp_dir, "supervisor", "started", experiment_dir=str(exp_dir))
    env = subprocess_env()
    log(f"Experiment directory: {exp_dir}")
    log(f"Resolved config: {config_path}")

    if args.dry_run:
        update_manifest(exp_dir, {"status": "dry_run_complete"})
        emit_status(exp_dir, "supervisor", "dry_run_complete")
        log("Dry run complete; no CARLA collection or training stages were launched.")
        return 0

    carla_proc: Optional[subprocess.Popen] = None
    try:
        carla_proc = run_collection_with_recovery(config, config_path, exp_dir, env, log)
        check_stop(exp_dir)
        run_training_sweep(config, config_path, exp_dir, env, log)
        check_stop(exp_dir)
        best_checkpoint = run_evaluation(config_path, exp_dir, env, log)
        report_path = generate_final_report(exp_dir, best_checkpoint)
        update_manifest(
            exp_dir,
            {
                "status": "complete",
                "completed_at": utc_iso(),
                "best_checkpoint": str(best_checkpoint),
                "final_report": str(report_path),
                "artifacts": {
                    "dataset_manifest": str(exp_dir / "dataset" / "manifest.csv"),
                    "object_boxes": str(exp_dir / "dataset" / "object_boxes.csv"),
                    "radar_tensors": str(exp_dir / "dataset" / "radar_tensor"),
                    "radar_points": str(exp_dir / "dataset" / "radar_points"),
                    "checkpoints": str(exp_dir / "checkpoints"),
                    "metrics": str(exp_dir / "metrics"),
                    "figures": str(exp_dir / "figures"),
                },
            },
        )
        emit_status(exp_dir, "supervisor", "complete", best_checkpoint=str(best_checkpoint), final_report=str(report_path))
        log(f"Pipeline complete. Report: {report_path}")
        return 0
    except KeyboardInterrupt as exc:
        update_manifest(exp_dir, {"status": "stopped", "stopped_at": utc_iso(), "stop_reason": str(exc)})
        emit_status(exp_dir, "supervisor", "stopped", reason=str(exc))
        log(f"Pipeline stopped: {exc}")
        return 130
    except Exception as exc:
        update_manifest(exp_dir, {"status": "failed", "failed_at": utc_iso(), "failure": f"{type(exc).__name__}: {exc}"})
        emit_status(exp_dir, "supervisor", "failed", failure=f"{type(exc).__name__}: {exc}")
        log(f"Pipeline failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        if bool(config.get("carla", {}).get("stop_server_on_exit", False)):
            terminate_carla(carla_proc, exp_dir, log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", default="")
    parser.add_argument("--runtime-budget-hours", type=float, default=0.0)
    parser.add_argument("--carla-bin", default=str(CARLA_BIN))
    parser.add_argument("--resume", choices=("auto", "off"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_pipeline(args))


if __name__ == "__main__":
    main()
