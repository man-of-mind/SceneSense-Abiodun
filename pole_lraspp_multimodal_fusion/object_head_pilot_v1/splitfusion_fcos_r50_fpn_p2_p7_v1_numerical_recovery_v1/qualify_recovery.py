from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .base_runtime import load_base
from .contracts import (atomic_json, atomic_text, canonical_hash, current_commit, load_json, load_recovery_config,
                        package_hashes, resolve_repo_path, sha256, verify_original_provenance)
from .envelope import build_healthy_envelope
from .guards import PreStepBreaker
from .precision_compare import compare_registered_precisions
from .replay import prepare_runtime, run_replay_once
from .runner import run_guarded_epoch


def _violations(metrics: Mapping[str, Any], ceilings: Mapping[str, Any]) -> list[str]:
    result = []
    for family in ("gradient_norm", "momentum_norm", "proposed_sgd_update_norm"):
        for group, value in metrics[family].items():
            if float(value) > float(ceilings[family][group]):
                result.append(f"{family}.{group}")
    if float(metrics["max_parameter_relative_update"]) > float(ceilings["max_parameter_relative_update"]):
        result.append("max_parameter_relative_update")
    return result


def _reproductions_agree(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    a, b = left["boundary"]["pre_step"], right["boundary"]["pre_step"]
    names = []
    for family in ("gradient_norm", "momentum_norm", "proposed_sgd_update_norm"):
        names.extend((float(a[family][group]), float(b[family][group])) for group in a[family])
    names.append((float(a["max_parameter_relative_update"]), float(b["max_parameter_relative_update"])))
    return all(abs(x - y) <= max(1e-5, 5e-3 * max(abs(x), abs(y))) for x, y in names)


def _original_telemetry_context(paths: Sequence[Path]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    update_count = 0
    for path in paths:
        epoch = int(path.stem.split("_")[-1]); payload = load_json(path)
        if int(payload.get("epoch", -1)) != epoch or not payload.get("all_updates_finite", False):
            raise RuntimeError(f"original healthy telemetry contract failed: {path}")
        for row in payload.get("update_boundary_records", []):
            if not row.get("finite", False):
                raise RuntimeError(f"nonfinite row in declared healthy telemetry: {path}")
            update_count += 1
            for name in ("loss", "radar_stem_gradient_norm", "rgb_stem_gradient_norm"):
                values.setdefault(name, []).append(float(row[name]))
            for name, evidence in row.get("required_gradient_evidence", {}).items():
                values.setdefault(f"required_gradient_l2.{name}", []).append(float(evidence["l2"]))
    statistics = {}
    for name, rows in values.items():
        array = np.asarray(rows, dtype=np.float64)
        if not len(array) or not bool(np.isfinite(array).all()):
            raise RuntimeError(f"invalid original telemetry values for {name}")
        statistics[name] = {"count": len(rows), "min": float(array.min()),
                            "median": float(np.percentile(array, 50)), "p99": float(np.percentile(array, 99)),
                            "max": float(array.max())}
    return {"epochs": [4, 5, 6, 7, 8, 9], "update_count": update_count,
            "statistics": statistics, "role": "healthy corroboration; breaker ceilings use compatible replay metrics"}


def _audit_coverage(replay: Mapping[str, Any]) -> dict[str, Any]:
    keys: list[str] = []
    for microbatch in replay["boundary"]["physical_microbatches"]:
        keys.extend(microbatch.get("tensor_audit", {}).keys())
        keys.extend(microbatch.get("score_and_box_decode_audit", {}).keys())
        for row in microbatch.get("geometry", {}).get("numerical_tensor_audits", []):
            keys.extend(row.get("records", {}).keys())
    required = ("batch.input", "outputs.c2", "outputs.features.p2", "outputs.features.p3",
                "outputs.features.p4", "outputs.features.p5", "outputs.features.p6", "outputs.features.p7",
                "cls_logits", "bbox_regression", "bbox_ctrness", "semantic_logits", "dense_depth",
                "depth_bin_logits", "depth_bin_probabilities", "bounded_depth_residuals", "decoded_depth",
                "physical_ray_offsets", "physical_uv", "local_xyz", "world_xyz", "log_dimensions",
                "exponentiated_dimensions", "raw_yaw", "raw_yaw_norm", "normalized_yaw", ".scores", ".boxes")
    missing = [name for name in required if not any(name in key for key in keys)]
    return {"required": list(required), "missing": missing, "complete": not missing,
            "audited_tensor_records": len(keys)}


def _full_disposable_range(output: Path, tau: float, ceilings: Mapping[str, Any]) -> dict[str, Any]:
    base, config, calibration, model, optimizer, _loader, _loss = prepare_runtime(tau, "candidate")
    dataset_root = (base.common.ROOT / config["dataset_root"]).resolve(strict=True)
    rows = base.data.load_split_rows(dataset_root, "train")
    cache = base.data.DepthCache((base.common.ROOT / config["train_depth_cache"]).resolve(strict=True), rows)
    dataset = base.data.RouteBDataset(dataset_root, "train", int(config["scientific_seed"]), cache, augment=True)
    breaker = PreStepBreaker(ceilings, output / "numerical_failures")
    dataset.set_epoch(10); loader10, _sampler10 = base.train.dataloader(dataset, int(config["scientific_seed"]), 10, 4,
                                                                       int(config["training"]["workers"]))
    global_update, epoch10 = run_guarded_epoch(base, model, optimizer, loader10, config, calibration["multipliers"],
                                                10, 9468, 4, breaker, enforce_required_nonzero=True)
    dataset.set_epoch(11); loader11, _sampler11 = base.train.dataloader(dataset, int(config["scientific_seed"]), 11, 4,
                                                                       int(config["training"]["workers"]))
    global_update, epoch11 = run_guarded_epoch(base, model, optimizer, loader11, config, calibration["multipliers"],
                                                11, global_update, 4, breaker, maximum_updates=32,
                                                enforce_required_nonzero=True)
    return {"epoch10": epoch10, "epoch11_first32": epoch11, "global_update": global_update,
            "repaired_update447_and_successor_included": epoch10["updates"] >= 448,
            "full_epoch10": epoch10["updates"] == 1052, "first32_epoch11": epoch11["updates"] == 32,
            "pre_step_guard_checks": epoch10["updates"] + epoch11["updates"],
            "circuit_breaker_events": 0, "all_guard_checks_passed": True,
            "required_reachability_nonzero_every_update": True,
            "peak_allocated_mib": max(epoch10["peak_allocated_mib"], epoch11["peak_allocated_mib"]),
            "vram_cap_mib": 12288, "disposable": True, "checkpoint_written": False,
            "validation_accessed": False, "original_epochs10_26_used": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run preregistered disposable recovery qualification")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-update447", required=True, type=Path)
    parser.add_argument("--independent-review", required=True, type=Path)
    parser.add_argument("--execute-qualification", required=True, choices=("DISPOSABLE_NO_SCIENTIFIC_STATE",))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    expected_record = load_json(args.expected_update447)
    expected = expected_record.get("sample_ids", [])
    if expected_record.get("preregistered") is not True or len(expected) != 16:
        raise RuntimeError("exact reviewed update-447 identities must be preregistered")
    review = load_json(args.independent_review); commit = current_commit(); source_hash = canonical_hash(package_hashes())
    expected_update447_sha256 = sha256(args.expected_update447)
    if not (review.get("approved") is True and review.get("source_commit") == commit
            and review.get("source_files_sha256") == source_hash
            and review.get("expected_update447_sha256") == expected_update447_sha256):
        raise RuntimeError("independent source review does not bind this exact implementation")
    verify_original_provenance(checkpoint_metadata=True)
    output.mkdir(parents=True, exist_ok=False); (output / "replays").mkdir(); (output / "candidates").mkdir()
    shutil.copy2(args.independent_review, output / "INDEPENDENT_SOURCE_REVIEW.json")
    original_a = run_replay_once(output / "replays/original_a", expected, normalization="original")
    original_b = run_replay_once(output / "replays/original_b", expected, normalization="original")
    if not _reproductions_agree(original_a, original_b):
        raise RuntimeError("original failure did not reproduce numerically twice")
    telemetry_root = resolve_repo_path(load_recovery_config()["original"]["experiment"])
    telemetry_files = [telemetry_root / "training_metrics" / f"epoch_{epoch:03d}.json" for epoch in range(4, 10)]
    telemetry_hashes = {str(path): sha256(path) for path in telemetry_files}
    envelope = build_healthy_envelope(original_a["healthy_records"])
    envelope["original_healthy_epoch4_9_telemetry"] = {"hashes": telemetry_hashes,
                                                         **_original_telemetry_context(telemetry_files)}
    envelope["threshold_values_derived_from"] = "compatible optimizer/group metrics from explicit epoch10 updates1-446; epoch4-9 loss/reachability/stem telemetry included as healthy corroboration"
    ceilings = envelope["ceilings"]
    original_violations = _violations(original_a["boundary"]["pre_step"], ceilings)
    if not original_violations:
        raise RuntimeError("original update447 did not violate the preregistered healthy envelope")
    atomic_json(output / "original_failure_audit.json", {"pass": True, "two_reproductions": True,
                "numeric_agreement": True, "first_failure": {"epoch": 10, "update": 447, "global": 9915},
                "violations": original_violations, "expected_sample_ids": expected,
                "expected_update447_sha256": expected_update447_sha256,
                "update446_healthy": True, "optimizer_step_at_447": False})
    atomic_json(output / "healthy_envelope.json", envelope)
    candidate_reports = []
    for tau in load_recovery_config()["yaw"]["candidate_tau"]:
        root = output / "candidates" / f"tau_{float(tau):.0e}"; root.mkdir()
        left = run_replay_once(root / "reproduction_a", expected, normalization="candidate", tau=float(tau))
        right = run_replay_once(root / "reproduction_b", expected, normalization="candidate", tau=float(tau))
        violations = _violations(left["boundary"]["pre_step"], ceilings)
        identities = [identity for micro in left["boundary"]["physical_microbatches"]
                      for identity in micro.get("geometry", {}).get("carrier_identities", [])]
        affected = sum(bool(row["below_tau"]) for row in identities)
        coverage = _audit_coverage(left)
        ordinary_errors = []
        for row in identities:
            if float(row["raw_yaw_norm"]) >= float(tau):
                raw = row["raw_yaw"]; expected_yaw = [raw[0] / row["raw_yaw_norm"], raw[1] / row["raw_yaw_norm"]]
                ordinary_errors.extend(abs(float(actual) - float(expected_value))
                                       for actual, expected_value in zip(row["normalized_yaw"], expected_yaw))
        ordinary_max_error = max(ordinary_errors, default=0.0)
        report = {"tau": tau, "two_reproductions_agree": _reproductions_agree(left, right),
                  "guard_violations": violations, "affected_carriers": affected,
                  "ordinary_norm_ge_tau_count": len(ordinary_errors) // 2,
                  "ordinary_norm_ge_tau_max_equation_error": ordinary_max_error,
                  "ordinary_norm_ge_tau_equation_unchanged": ordinary_max_error <= 2e-7,
                  "raw_decoded_output_audit": coverage,
                  "passes_update447": not violations and _reproductions_agree(left, right)
                      and affected > 0 and bool(ordinary_errors) and ordinary_max_error <= 2e-7
                      and coverage["complete"]}
        candidate_reports.append(report); atomic_json(root / "candidate.json", report)
    passing = [row for row in candidate_reports if row["passes_update447"]]
    if not passing:
        atomic_text(output / "CONTRACT_INVALID_NO_CANDIDATE", "NO_PREREGISTERED_TAU_SATISFIED_THE_GUARDS\n")
        atomic_text(output / "TERMINAL_VERDICT.txt",
                    "SPLITFUSION_FCOS_RECOVERY_CONTRACT_INVALID_NO_PREREGISTERED_TAU\n")
        raise RuntimeError("contract-invalid: no preregistered yaw tau satisfied qualification guards")
    selected_report = min(passing, key=lambda row: (row["affected_carriers"], float(row["tau"])))
    selected = float(selected_report["tau"])
    range_report = _full_disposable_range(output / "selected_disposable", selected, ceilings)
    if not (range_report["full_epoch10"] and range_report["first32_epoch11"]
            and range_report["repaired_update447_and_successor_included"]
            and range_report["peak_allocated_mib"] <= 12288):
        raise RuntimeError("selected candidate failed full disposable range or VRAM guard")
    base, config, calibration, model, optimizer, _loader, _loss = prepare_runtime(selected, "candidate")
    immutable = load_recovery_config(); original = Path(base.common.ROOT / immutable["original"]["experiment"])
    registration = load_json(original / "SCIENTIFIC_REGISTRATION.json")
    dataset_root = (base.common.ROOT / config["dataset_root"]).resolve(strict=True)
    rows = base.data.load_split_rows(dataset_root, "train")
    cache = base.data.DepthCache((base.common.ROOT / config["train_depth_cache"]).resolve(strict=True), rows)
    dataset = base.data.RouteBDataset(dataset_root, "train", int(config["scientific_seed"]), cache, augment=True); dataset.set_epoch(10)
    precision = compare_registered_precisions(base, model, optimizer, dataset, registration["calibration_batches"],
                                               calibration["multipliers"], 4)
    atomic_json(output / "bf16_fp32_train_only.json", precision)
    atomic_json(output / "candidate_selection.json", {"rule": "minimum affected carriers, then smallest preregistered passing tau",
                                                       "selected_tau": selected, "candidates": candidate_reports})
    marker = output / "QUALIFIED_TO_TRAIN"; marker_contents = "QUALIFIED_TO_TRAIN\n"
    marker_sha256 = hashlib.sha256(marker_contents.encode()).hexdigest()
    qualified = {"schema": "splitfusion_fcos_numerical_recovery_qualified_config_v1", "state": "QUALIFIED_TO_TRAIN",
                 "selected_tau": selected, "ceilings": ceilings, "threshold_formula": "10 * maximum_healthy_value",
                 "source_commit": commit, "source_files_sha256": source_hash, "marker_sha256": marker_sha256,
                 "original_checkpoint_sha256": load_recovery_config()["original"]["checkpoint_sha256"],
                 "original_source_canonical_sha256": load_recovery_config()["original"]["source_canonical_sha256"],
                 "original_checkpoint_source_canonical_sha256": load_recovery_config()["original"]["checkpoint_source_canonical_sha256"],
                 "original_config_sha256": load_recovery_config()["original"]["config_sha256"],
                 "original_registration_sha256": load_recovery_config()["original"]["registration_sha256"],
                 "expected_update447_sha256": expected_update447_sha256,
                 "selected_tau_and_ceilings_hash": canonical_hash({"selected_tau": selected, "ceilings": ceilings}),
                 "candidate_set": load_recovery_config()["yaw"]["candidate_tau"]}
    qualification = {"schema": "splitfusion_fcos_numerical_recovery_qualification_v1", "pass": True,
        "source_commit": commit, "source_files_sha256": source_hash, "original_failure_reproduced_twice": True,
        "selected_tau": selected, "candidate_selection": "minimum affected carriers then smallest passing", "range": range_report,
        "causal_repair_evidence": "only shared yaw normalization changed; original boundary violated; repaired boundary and ordinary vectors passed",
        "precision_comparison": "bf16_fp32_train_only.json", "validation_accessed": False,
        "original_checkpoint_sha256": load_recovery_config()["original"]["checkpoint_sha256"],
        "original_registration_sha256": load_recovery_config()["original"]["registration_sha256"],
        "expected_update447_sha256": expected_update447_sha256,
        "selected_tau_and_ceilings_hash": canonical_hash({"selected_tau": selected, "ceilings": ceilings}),
        "scientific_optimizer_steps": 0,
        "disposable_optimizer_steps": (2 + 2 * len(load_recovery_config()["yaw"]["candidate_tau"])) * 446 + 1052 + 32,
        "disposable_state_discarded": True, "independent_review": True}
    atomic_json(output / "QUALIFIED_RECOVERY_CONFIG.json", qualified)
    atomic_json(output / "RECOVERY_QUALIFICATION.json", qualification)
    atomic_text(output / "DISPOSABLE_STATE_DISCARDED", "NO_QUALIFICATION_CHECKPOINT_IS_A_SCIENTIFIC_INPUT\n")
    atomic_text(marker, marker_contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
