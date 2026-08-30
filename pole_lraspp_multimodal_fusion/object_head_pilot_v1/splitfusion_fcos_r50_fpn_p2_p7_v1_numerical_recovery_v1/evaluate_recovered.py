from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .base_runtime import load_base
from .contracts import (atomic_json, atomic_text, load_recovery_config, require_qualified, resolve_repo_path,
                        verify_original_provenance)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen evaluator over original 3/8 and recovered 16/22/26")
    parser.add_argument("--recovered-experiment", required=True, type=Path)
    parser.add_argument("--qualification-dir", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--execute-recovered-evaluation", required=True, choices=("ALL_FIVE_INFERENCE_PASSES_COMPLETE",))
    args = parser.parse_args()
    require_qualified(args.qualification_dir, args.authorization)
    verify_original_provenance(checkpoint_metadata=False)
    recovered = args.recovered_experiment.resolve(strict=True)
    if not (recovered / "TRAINING_COMPLETE").is_file():
        raise RuntimeError("recovered training incomplete")
    immutable = load_recovery_config(); original = resolve_repo_path(immutable["original"]["experiment"])
    staging = recovered / "evaluation_original_003_008_recovered_016_022_026"
    staging.mkdir(parents=True, exist_ok=False); (staging / "predictions").mkdir(); (staging / "checkpoints").mkdir()
    atomic_text(staging / "TRAINING_COMPLETE", "COMBINED_EVALUATION_INPUT_ONLY\n")
    sources = {}
    for epoch in (3, 8, 16, 22, 26):
        source = (original if epoch in (3, 8) else recovered) / f"predictions/epoch_{epoch:03d}"
        if not (source / "INFERENCE_COMPLETE").is_file():
            raise RuntimeError(f"required inference pass incomplete: {source}")
        destination = staging / f"predictions/epoch_{epoch:03d}"
        destination.symlink_to(source, target_is_directory=True)
        checkpoint_source = (original if epoch in (3, 8) else recovered) / f"checkpoints/epoch_{epoch:03d}.pt"
        (staging / f"checkpoints/epoch_{epoch:03d}.pt").symlink_to(checkpoint_source)
        sources[str(epoch)] = {"kind": "original_healthy" if epoch in (3, 8) else "recovered",
                               "path": str(source), "checkpoint": str(checkpoint_source)}
    atomic_json(staging / "CHECKPOINT_ORIGIN_LABELS.json", {"epochs": sources,
        "excluded_original_epochs_10_26": list(range(10, 27)),
        "excluded_label": "CORRUPTED_FINITE_GRADIENT_TRAJECTORY_DO_NOT_USE",
        "same_frozen_evaluator_settings_gates_selection": True, "sensitivity_contract": "v025_selected_only"})
    base = load_base(); previous = list(sys.argv)
    try:
        sys.argv = [str(Path(base.evaluate.__file__)), "--experiment", str(staging)]
        result = int(base.evaluate.main())
    finally:
        sys.argv = previous
    atomic_text(staging / "RECOVERED_COMPARISON_COMPLETE", "ORIGINAL_003_008_VS_RECOVERED_016_022_026\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
