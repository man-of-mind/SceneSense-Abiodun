"""Phase-10A AE64/AE32 train-holdout evaluation and preregistered selection.

A separate command from Phase-10A training, deliberately: nothing here can run
while the trainer runs, and the trainer refuses to start if this module is even
imported or if its output directory already exists.

    python3 -m ...ae_v1.ae_phase10_holdout_selection \\
      --execute SPLITFUSION_AE64_PHASE10_HOLDOUT_SELECTION \\
      --bottleneck 64 --training <completed_ae64_run>

    python3 -m ...ae_v1.ae_phase10_holdout_selection \\
      --execute SPLITFUSION_AE32_PHASE10_HOLDOUT_SELECTION \\
      --bottleneck 32 --training <completed_ae32_run>

After a separately authorized completed training run for one family this
evaluates exactly the three candidate epochs {4, 8, 12} at exactly
q = {0, 0.30, 0.50, 0.70}, one holdout inference/evaluation pass per
checkpoint/q pair, over the 3,284 reserved train-holdout frames. Validation and
test are never opened, and no fit teacher shard is deserialized.

Transport at this phase is **FP32 AE latent reconstruction only**: no UINT8, no
zstd, no UINT6/UINT4. The single-pass encode/drop/decode/serve routine, the
frozen noAE same-q reference, the registered preservation gates, the scorers,
the thresholds, the batching-independent float64 frame-summed reconstruction
tie-breaker and the preregistered ranking key are the completed AE128 ones,
imported and reused rather than restated; only the family labels differ.

Selecting a checkpoint is not a service-ready claim, and deployment-path
UINT8+zstd validation is a later, separately authorized phase.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    contract,
    continuous_q,
    guards,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    build_train_dataset,
    load_frozen_perception,
    sha256_file,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
    build_contract_alias,
    load_frozen_scorers,
    load_holdout_person_truth,
    load_p025_qualification,
    source_delta,
)
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_holdout import score_pass
from ..splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.teacher_cache import (
    build_split_partition,
)
from . import ae_phase10_common as family
from . import ae_training_common as common

# The AE128 selection machinery that carries no family label: the FP32
# encode/drop/decode/serve holdout pass, the per-checkpoint aggregation and the
# preregistered ordering key. Reusing the objects themselves is what makes
# "the same rule" checkable rather than merely claimed.
from .ae_holdout_selection import (
    GATE_COUNT,
    INFERENCE_BATCH,
    RANKING_CRITERIA,
    RANKING_RULE,
    aggregate_by_checkpoint,
    ranking_key,
    run_pass,
)

DATALOADER_WORKERS = 8

# Registered gate scoring, shared with AE128 and unchanged.
evaluate_same_q_gates = common.evaluate_same_q_gates


# ---------------------------------------------------------------------------
# Preregistered checkpoint ranking (the AE128 rule, family-labelled output)
# ---------------------------------------------------------------------------


def rank_checkpoints(records: Sequence[Mapping[str, Any]], bottleneck: int) -> dict[str, Any]:
    """Apply the preregistered AE128 rule verbatim to one Phase-10A family.

    `aggregate_by_checkpoint` and `ranking_key` are the completed AE128 objects,
    so the criteria, their order, the normalized-degradation definition and the
    batching-independent reconstruction tie-breaker cannot drift here. Only the
    reported family identity and purpose text are this phase's own.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    aggregated = aggregate_by_checkpoint(records)
    if not aggregated:
        raise guards.HybridQConfigError("checkpoint ranking needs at least one candidate")
    ordered = sorted(aggregated, key=ranking_key)
    best, *rest = ordered
    decided_at = RANKING_CRITERIA[-1]
    if rest:
        best_key = ranking_key(best)
        runner_key = ranking_key(rest[0])
        for position, name in enumerate(RANKING_CRITERIA):
            if best_key[position] != runner_key[position]:
                decided_at = name
                break
    label = family.family_label(size)
    return {
        "rule": RANKING_RULE,
        "rule_source": "phase9c_preregistered_ranking_reused_unchanged",
        "criteria_in_order": list(RANKING_CRITERIA),
        "reconstruction_loss_definition": (
            "criterion 4 uses global_total_loss: the plain and "
            "combined-importance ratios are each formed from float64 numerator "
            "and denominator sums accumulated over every holdout frame, so the "
            "value is independent of the inference batch grouping. The per-batch "
            "means are retained as diagnostics only"
        ),
        "normalized_degradation_definition": (
            "signed protected-metric degradation divided by that metric's "
            "registered preservation-gate bound, so metrics with different "
            "bounds are comparable"
        ),
        "ranking": ordered,
        "selected_epoch": int(best["epoch"]),
        "selected": dict(best),
        "decided_at_criterion": decided_at,
        "selection_is_a_service_ready_claim": False,
        "selection_purpose": (
            f"chooses the one {label} checkpoint a later, separately authorized "
            f"deployment-path UINT8+zstd validation would use; it does not "
            f"accept {label} for service"
        ),
        **family.family_fields(size),
    }


# ---------------------------------------------------------------------------
# The completed training run this command consumes
# ---------------------------------------------------------------------------


def load_training_run(training_dir: Path, bottleneck: int) -> dict[str, Any]:
    """Verify a completed same-family training run and return its report."""
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_label(size)
    terminal = family.training_terminal(size)
    if not (training_dir / terminal).is_file():
        raise guards.HybridQConfigError(
            f"holdout selection requires a separately authorized completed {label} "
            f"training run declaring {terminal}"
        )
    report_path = training_dir / family.training_report_filename(size)
    if not report_path.is_file():
        raise guards.HybridQConfigError(
            f"{report_path.name} is missing; this is not a completed {label} run"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["schema"] != family.training_schema(size):
        raise guards.HybridQConfigError("training report schema drift")
    if report["terminal"] != terminal:
        raise guards.HybridQConfigError("the training run did not complete")
    family.require_family_fields(report["scope"], size, what="training report scope")
    if report["configuration"] != family.training_configuration(size):
        raise guards.HybridQConfigError(
            "the training run used a different locked configuration"
        )
    expected = {
        family.candidate_filename(size, epoch) for epoch in common.AE_CANDIDATE_EPOCHS
    }
    if set(report["candidate_checkpoints"]) != expected:
        raise guards.HybridQConfigError("candidate checkpoint set drift")
    for name, digest in report["candidate_checkpoints"].items():
        if sha256_file(training_dir / "checkpoints" / name) != digest:
            raise guards.HybridQConfigError(f"candidate checkpoint {name} sha256 drift")
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-10A AE64/AE32 train-holdout evaluation and checkpoint selection"
        )
    )
    parser.add_argument("--execute", required=True, choices=family.HOLDOUT_EXECUTE_TOKENS)
    parser.add_argument(
        "--bottleneck", required=True, type=int, choices=family.AE_PHASE10_BOTTLENECKS
    )
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--keep-segmentation", action="store_true")
    parser.add_argument("--smoke-batches", type=int, default=0)
    args = parser.parse_args()

    bottleneck = family.require_token_agrees_with_bottleneck(
        args.execute, args.bottleneck, kind="holdout"
    )
    family.bind_process_family(bottleneck)
    label = family.family_label(bottleneck)
    terminal = family.holdout_terminal(bottleneck)

    training_dir = args.training.resolve(strict=True)
    training_report = load_training_run(training_dir, bottleneck)
    output = family.holdout_selection_dir(training_dir, bottleneck)
    if output.exists():
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError(f"Phase-10A {label} holdout selection requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(family.process_seed(bottleneck))

    binding = common.bind_frozen_inputs()
    delta = source_delta(binding)
    # Exactly the frozen noAE same-q holdout reference AE128 was scored against.
    reference = common.load_noae_holdout_reference()

    model, base, perception = load_frozen_perception(device)
    common.freeze(model)
    ranker = common.load_stable_ranker(device)
    guards.require_frozen_perception([model, ranker])
    guards.require_eval_mode([model, ranker])
    frozen_model_state = guards.snapshot_module_state(model)
    frozen_ranker_state = guards.snapshot_module_state(ranker)

    route = build_train_dataset(base)
    partition = build_split_partition(route)
    frame_ids = list(partition.holdout_sample_ids)
    store = common.load_ae_teacher_store(partition, "holdout")
    if store.split != "holdout" or store.frames != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError("the selection teacher store is not the holdout")

    root = contract.repository_root()
    config = json.loads(
        (
            root
            / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json"
        ).read_text(encoding="utf-8")
    )
    dataset_root = (root / config["dataset_root"]).resolve(strict=True)
    inference = base.data.InferenceDataset(dataset_root, "train")
    position_by_id = {row["sample_id"]: index for index, row in enumerate(inference.rows)}
    if len(position_by_id) != contract.TRAIN_TOTAL_FRAMES:
        raise guards.HybridQConfigError("inference dataset train frame count drift")
    positions = [position_by_id[sample_id] for sample_id in frame_ids]
    if len(positions) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError("holdout position count drift")

    output.mkdir(parents=True, exist_ok=False)
    alias_root = build_contract_alias(dataset_root, output / "train_contract_alias")
    scorers = load_frozen_scorers()
    qualification = load_p025_qualification()
    truth = load_holdout_person_truth(frame_ids)
    gt, _gt_states = scorers.load_gt(alias_root, contract.PRIMARY_CONTRACT)
    holdout_gt = {sample_id: gt.get(sample_id, []) for sample_id in frame_ids}
    ignore_cache: dict[str, Any] = {}

    limit = args.smoke_batches or None
    scored_ids = frame_ids if limit is None else frame_ids[: limit * INFERENCE_BATCH]
    predictions_root = output / "predictions"
    predictions_root.mkdir()

    started = time.time()
    evaluations: list[dict[str, Any]] = []
    for epoch in common.AE_CANDIDATE_EPOCHS:
        name = family.candidate_filename(bottleneck, epoch)
        path = training_dir / "checkpoints" / name
        digest = sha256_file(path)
        autoencoder, metadata = family.load_candidate(
            path, bottleneck, epoch, device, binding
        )
        for q in common.AE_HOLDOUT_Q_VALUES:
            plan = continuous_q.quantize_q(float(q))
            raw = run_pass(
                model=model,
                base=base,
                ranker=ranker,
                autoencoder=autoencoder,
                q=plan.wire_q,
                dataset=inference,
                positions=positions,
                frame_ids=frame_ids,
                store=store,
                device=device,
                output=predictions_root
                / f"{family.family_slug(bottleneck)}_epoch{epoch:02d}_q{plan.q_e4:04d}",
                workers=int(args.workers),
                limit=limit,
            )
            guards.require_module_state_unchanged(model, frozen_model_state)
            guards.require_module_state_unchanged(ranker, frozen_ranker_state)
            scored = score_pass(
                result=raw,
                scorers=scorers,
                qualification=qualification,
                truth=truth,
                alias_root=alias_root,
                frame_ids=scored_ids,
                gt=holdout_gt,
                ignore_cache=ignore_cache,
                cross_check=False,
                require_defined=limit is None,
            )
            scored["configuration"] = (
                f"{family.family_slug(bottleneck)}_epoch{epoch:02d}_q{plan.wire_q:.2f}"
            )
            scored.update(family.family_fields(bottleneck))
            scored["epoch"] = epoch
            scored["checkpoint"] = name
            scored["checkpoint_sha256"] = digest
            scored["checkpoint_metadata"] = {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }
            scored["noae_same_q_reference"] = reference[plan.wire_q]
            scored["gate_result"] = evaluate_same_q_gates(
                reference[plan.wire_q]["metrics"], scored["metrics"]
            )
            evaluations.append(scored)
            print(
                json.dumps(
                    {
                        "pass": scored["configuration"],
                        "gates_passed": scored["gate_result"]["gates_passed"],
                        "of": GATE_COUNT,
                        "failed": scored["gate_result"]["failed"],
                        "worst_normalized_degradation": round(
                            scored["gate_result"]["worst_normalized_degradation"], 4
                        ),
                        "global_reconstruction_loss": round(
                            scored["reconstruction"]["global_total_loss"], 6
                        ),
                    }
                ),
                flush=True,
            )
        del autoencoder

    decision = rank_checkpoints(evaluations, bottleneck)

    if not args.keep_segmentation:
        for entry in evaluations:
            directory = Path(entry["prediction_root"]) / "segmentation"
            if directory.is_dir():
                shutil.rmtree(directory)
            entry["segmentation_masks_removed_after_scoring"] = True

    report = {
        "schema": family.holdout_schema(bottleneck),
        "terminal": terminal,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.time() - started,
        "scope": {
            **family.family_fields(bottleneck),
            "ae128_touched": False,
            "evaluated_split": "reserved train-holdout episodes",
            "holdout_episodes": list(contract.TRAIN_HOLDOUT_EPISODES),
            "holdout_frames": contract.TRAIN_HOLDOUT_FRAMES,
            "evaluated_epochs": list(common.AE_CANDIDATE_EPOCHS),
            "evaluated_q_values": [float(q) for q in common.AE_HOLDOUT_Q_VALUES],
            "passes": len(evaluations),
            "one_pass_per_checkpoint_q_pair": True,
            "transport": common.AE_HOLDOUT_QUANTIZER,
            "uint8_zstd_uint6_uint4_run": False,
            "deployment_validation_performed_here": False,
            "stress_q_values_not_evaluated": list(common.AE_EXCLUDED_Q),
            "validation_or_test_accessed": False,
            "fit_teacher_shards_deserialized": 0,
            "training_run_here": False,
            "threshold_calibration_nms_or_visibility_changed": False,
            "carla_launched": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "binding": binding,
        "perception_binding": perception,
        "hybrid_q_source_delta_since_phase4": delta,
        "teacher_store": store.provenance(),
        "training_run": {
            "path": str(training_dir),
            "candidate_checkpoints": dict(training_report["candidate_checkpoints"]),
            "configuration": training_report["configuration"],
            "configuration_delta_from_ae128": training_report.get(
                "configuration_delta_from_ae128"
            ),
        },
        "noae_reference": {
            "path": common.NOAE_HOLDOUT_RELPATH,
            "sha256": common.NOAE_HOLDOUT_SHA256,
            "ranker_epoch_for_q_above_zero": common.NOAE_REFERENCE_RANKER_EPOCH,
            "identical_reference_as_ae128": True,
            "per_q": reference,
        },
        "preservation_gates": [
            {"metric": name, "direction": direction, "bound": bound}
            for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES
        ],
        "service_pipeline": {
            "vehicle_score_threshold": contract.VEHICLE_SCORE_THRESHOLD,
            "person_service_score_threshold": contract.PERSON_SERVICE_SCORE_THRESHOLD,
            "person_avo_threshold": contract.PERSON_AVO_THRESHOLD,
            "scorer_sha256": scorers.sha256,
            "unchanged": True,
        },
        "checkpoint_q_evaluations": evaluations,
        "selection": decision,
        "frozen_state_unchanged_at_end": True,
    }
    report_path = output / family.holdout_report_filename(bottleneck)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (output / terminal).write_text(f"{sha256_file(report_path)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "family": label,
                "selected_epoch": decision["selected_epoch"],
                "selected_checkpoint": family.candidate_filename(
                    bottleneck, decision["selected_epoch"]
                ),
                "decided_at_criterion": decision["decided_at_criterion"],
                "report_sha256": sha256_file(report_path),
            }
        )
    )
    print(terminal)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
