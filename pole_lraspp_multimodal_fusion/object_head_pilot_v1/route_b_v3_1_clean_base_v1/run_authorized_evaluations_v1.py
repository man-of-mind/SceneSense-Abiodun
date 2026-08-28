#!/usr/bin/env python3
"""Run exactly the preregistered decoded checkpoint inference set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


EPOCHS = (5, 10, 15, 20, 25)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_x(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--infer-script", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    checkpoint_dir = experiment / "checkpoints/route_b_v3_1_clean_noae_stage2_v1"
    started = time.monotonic()
    records = []
    try:
        for epoch in EPOCHS:
            checkpoint = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            checkpoint_hash = sha256(checkpoint)
            tag = f"trained_epoch_{epoch:03d}"
            command = [
                sys.executable, str(args.infer_script.resolve()),
                "--experiment", str(experiment), "--checkpoint", str(checkpoint),
                "--checkpoint-sha256", checkpoint_hash, "--tag", tag,
            ]
            print(f"[authorized evaluation] epoch={epoch}", flush=True)
            result = subprocess.run(command)
            if result.returncode != 0:
                raise RuntimeError(f"inference failed for epoch {epoch}: rc={result.returncode}")
            manifest = json.loads((experiment / "predictions" / tag / "inference_manifest.json").read_text(encoding="utf-8"))
            records.append({
                "epoch": epoch, "tag": tag, "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "prediction_set_sha256": manifest["prediction_set_sha256"],
                "wall_seconds": manifest["wall_seconds"],
                "peak_allocated_mib": manifest["peak_allocated_mib"],
                "peak_reserved_mib": manifest["peak_reserved_mib"],
            })
        result = {
            "schema": "route_b_v3_1_authorized_evaluations_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated_epochs": list(EPOCHS), "loss_best_epoch": 5,
            "loss_best_was_additional": False, "records": records,
            "inference_passes": len(records), "wall_seconds": time.monotonic() - started,
        }
        write_json_x(experiment / "AUTHORIZED_EVALUATIONS_COMPLETE.json", result)
        (experiment / "AUTHORIZED_EVALUATIONS_COMPLETE").write_text("AUTHORIZED_EVALUATIONS_COMPLETE\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        (experiment / "TERMINAL_VERDICT.txt").write_text("LRASPP_V3_1_RUNTIME_FAILURE\n", encoding="utf-8")
        write_json_x(experiment / "evaluation_failure.json", {
            "terminal": "LRASPP_V3_1_RUNTIME_FAILURE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}", "wall_seconds": time.monotonic() - started,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
