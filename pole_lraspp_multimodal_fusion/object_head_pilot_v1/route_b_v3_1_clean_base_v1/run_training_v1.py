#!/usr/bin/env python3
"""Run and hard-gate the single authorized 25-epoch v3.1 training job."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from runtime_v1 import install, trainer


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()
    install()
    experiment = args.experiment.resolve()
    trial = json.loads(args.trial.read_text(encoding="utf-8"))
    started = time.monotonic()
    write_json_x(experiment / "TRAINING_STARTED.json", {
        "schema": "route_b_v3_1_training_started_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trial": trial, "config": str(args.config.resolve()),
        "single_authorized_training_launch": True,
    })
    try:
        code = trainer.train(argparse.Namespace(
            config=str(args.config.resolve()), experiment_dir=str(experiment),
            trial_json=json.dumps(trial, sort_keys=True), training_budget_hours=0.0,
        ))
        if code != 0:
            raise RuntimeError(f"trainer returned {code}")
        checkpoint_dir = experiment / "checkpoints" / trial["name"]
        expected = [checkpoint_dir / f"epoch_{epoch:03d}.pt" for epoch in range(1, 26)]
        missing = [str(path) for path in expected if not path.is_file()]
        metrics_path = experiment / "metrics" / f"{trial['name']}_metrics.csv"
        with metrics_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        finite = all(
            all(str(row[key]).lower() not in {"nan", "inf", "-inf"} for key in ("train_loss", "val_loss", "object_loss", "seg_loss"))
            for row in rows
        )
        gates = {
            "epochs_exact_1_through_25": epochs == list(range(1, 26)),
            "all_epoch_checkpoints_present": not missing,
            "losses_finite": finite,
            "q_clean_only": float(trial["feature_drop_max"]) == 0.0 and float(trial["feature_drop_val"]) == 0.0,
            "ae_disabled": int(trial["ae_bottleneck"]) == 0,
            "batch_size_16": int(trial["batch_size"]) == 16,
            "workers_8": int(trial["num_workers"]) == 8,
        }
        if not all(gates.values()):
            raise RuntimeError(f"training completion gate failure: gates={gates} missing={missing}")
        result = {
            "schema": "route_b_v3_1_training_complete_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "gates": gates,
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_mib": max(float(row["cuda_max_memory_allocated_mib"]) for row in rows),
            "peak_reserved_mib": max(float(row["cuda_max_memory_reserved_mib"]) for row in rows),
            "loss_best_epoch": int(min(rows, key=lambda row: float(row["val_loss"]))["epoch"]),
            "final_epoch": epochs[-1], "training_rows": len(rows),
        }
        write_json_x(experiment / "TRAINING_COMPLETE.json", result)
        (experiment / "PHASE3_TRAINING_COMPLETE").write_text("PHASE3_TRAINING_COMPLETE\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        (experiment / "TERMINAL_VERDICT.txt").write_text("LRASPP_V3_1_RUNTIME_FAILURE\n", encoding="utf-8")
        write_json_x(experiment / "training_failure.json", {
            "terminal": "LRASPP_V3_1_RUNTIME_FAILURE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}", "wall_seconds": time.monotonic() - started,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
