from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_relational_selector_v1.provenance import (
    CONSOLIDATION_MANIFEST_SHA256,
    FROZEN_CHECKPOINT_SHA256,
    HOLDOUT_EXPERIMENT_IDS,
    ROI_MANIFEST_SHA256,
    load_locked_caches,
    sha256,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_relational_selector_v1.selector import (
    ARCHITECTURE,
    PersonRelationalSelector,
    refined_person_scores,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_relational_selector_v1.train_selector import (
    _holdout_outputs,
    _rematched_metrics_at_threshold,
)

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
LOCKED_CONFIG_PATH = PACKAGE / "locked_config.json"
HOLDOUT_VERIFICATION_PATH = PACKAGE / "HOLDOUT_VERIFICATION.json"

SELECTOR_CHECKPOINT_SHA256 = "af7e8016dbbab41b4edf9ef30f3780bc504b07efe6772a45ac04f5f10df4555a"
RAW_RELATIONAL_THRESHOLD = 0.6632936000823975
DEPLOYMENT_LOGIT_BIAS = -2.064300755242339
CANONICAL_PERSON_THRESHOLD = 0.20
EXPECTED_SCORE_BOUNDARIES = 218_742
HISTORICAL_STATUS = "train_infeasible"

BASE_CHECKPOINT_RELATIVE = Path(
    "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/"
    "20260830_recovered_epoch10_gate_v1/checkpoints/epoch_026.pt"
)
SELECTOR_CHECKPOINT_RELATIVE = Path(
    "experiments/person_relational_selector_v1/person_relational_selector.pt"
)

EXPECTED_COUNTS = {
    "aggregate": {"tp": 2879, "fp": 735, "fn": 777, "ignored": 429},
    "episodes": {
        "canonical_v3_03_train_30_30_s503_tm1503": {
            "tp": 898, "fp": 259, "fn": 200, "ignored": 175,
        },
        "canonical_v3_04_train_50_50_s504_tm1504": {
            "tp": 1981, "fp": 476, "fn": 577, "ignored": 254,
        },
    },
}


@dataclass(frozen=True)
class RevisedSelectorRuntime:
    selector: PersonRelationalSelector
    deployment_bias: float
    selector_checkpoint: Path
    selector_checkpoint_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    historical_status: str


def _expected_metrics(counts: Mapping[str, int]) -> tuple[float, float]:
    tp, fp, fn = (int(counts[name]) for name in ("tp", "fp", "fn"))
    return tp / (tp + fp), tp / (tp + fn)


def _verify_locked_config(config: Mapping[str, Any]) -> None:
    calibration = config.get("calibration", {})
    objective = config.get("objective", {})
    holdout = config.get("holdout", {})
    runtime = config.get("runtime", {})
    if (config.get("schema") != "splitfusion_fcos_relational_p070_contract_v1"
            or config.get("base_checkpoint") != {
                "path": str(BASE_CHECKPOINT_RELATIVE), "sha256": FROZEN_CHECKPOINT_SHA256,
            }
            or config.get("selector_checkpoint") != {
                "path": str(SELECTOR_CHECKPOINT_RELATIVE), "sha256": SELECTOR_CHECKPOINT_SHA256,
            }
            or objective != {
                "kind": "post_hoc_revised_train_holdout_gate",
                "precision_minimum": 0.7,
                "recall_minimum": 0.7,
                "historical_0_80_status_preserved": HISTORICAL_STATUS,
            }
            or calibration != {
                "raw_relational_threshold": RAW_RELATIONAL_THRESHOLD,
                "deployment_logit_bias": DEPLOYMENT_LOGIT_BIAS,
                "deployment_threshold": CANONICAL_PERSON_THRESHOLD,
                "arithmetic": "FP32",
            }
            or int(holdout.get("score_boundaries", -1)) != EXPECTED_SCORE_BOUNDARIES
            or holdout.get("tie_processing")
            != "all_equal_scores_selected_together_before_canonical_frame_rematching"
            or not math.isclose(float(holdout.get("maximin_minimum", -1.0)),
                                0.77443315089914, rel_tol=0.0, abs_tol=1e-15)
            or runtime != {
                "candidate_creation": False,
                "consolidation_is_feature_only": True,
                "geometry_changed": False,
                "nms_rerun": False,
                "segmentation_changed": False,
                "vehicle_policy": "bit_exact_service_candidate_v1",
            }):
        raise RuntimeError("revised relational-p070 configuration drift")
    for scope, counts in EXPECTED_COUNTS.items():
        reports = {"aggregate": holdout.get("aggregate", {})} if scope == "aggregate" else holdout.get("episodes", {})
        expected_reports = {"aggregate": counts} if scope == "aggregate" else counts
        if set(reports) != set(expected_reports):
            raise RuntimeError("revised holdout report scope drift")
        for name, expected in expected_reports.items():
            report = reports[name]
            if any(int(report.get(field, -1)) != value for field, value in expected.items()):
                raise RuntimeError("revised holdout count drift")
            precision, recall = _expected_metrics(expected)
            if (not math.isclose(float(report.get("precision", -1.0)), precision,
                                 rel_tol=0.0, abs_tol=1e-15)
                    or not math.isclose(float(report.get("recall", -1.0)), recall,
                                        rel_tol=0.0, abs_tol=1e-15)):
                raise RuntimeError("revised holdout metric arithmetic drift")
    minimum = min(
        metric
        for report in (holdout["aggregate"], *holdout["episodes"].values())
        for metric in (float(report["precision"]), float(report["recall"]))
    )
    if not math.isclose(minimum, float(holdout["maximin_minimum"]), rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("revised maximin summary drift")


def load_locked_config() -> dict[str, Any]:
    config = json.loads(LOCKED_CONFIG_PATH.read_text(encoding="utf-8"))
    _verify_locked_config(config)
    return config


def calibration_at_raw_threshold() -> torch.Tensor:
    """Evaluate the exact deployment FP32 arithmetic at the selected boundary."""
    raw = torch.tensor([RAW_RELATIONAL_THRESHOLD], dtype=torch.float32)
    zero = torch.zeros_like(raw)
    return refined_person_scores(raw, zero, DEPLOYMENT_LOGIT_BIAS)


def _validate_historical_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    training = checkpoint.get("training", {})
    holdout = checkpoint.get("holdout", {})
    selector_state = checkpoint.get("selector")
    if (checkpoint.get("schema") != "splitfusion_fcos_person_relational_selector_v1"
            or checkpoint.get("base_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256
            or checkpoint.get("roi_manifest_sha256") != ROI_MANIFEST_SHA256
            or checkpoint.get("consolidation_manifest_sha256") != CONSOLIDATION_MANIFEST_SHA256
            or checkpoint.get("architecture") != ARCHITECTURE
            or checkpoint.get("status") != HISTORICAL_STATUS
            or checkpoint.get("validation_allowed") is not False
            or checkpoint.get("validation_or_test_accessed") is not False
            or training.get("epochs") != 5
            or training.get("selected_epoch") != 5
            or training.get("seed") != 20260831
            or holdout.get("before_calibration", {}).get(
                "joint_precision_recall_0_80_exists",
            ) is not False
            or int(holdout.get("before_calibration", {}).get(
                "score_boundaries", -1,
            )) != EXPECTED_SCORE_BOUNDARIES
            or not isinstance(selector_state, Mapping)):
        raise RuntimeError("historical selector contract drift; revised wrapper fails closed")


def load_revised_selector(
    device: torch.device, *, require_holdout_verification: bool = True,
) -> RevisedSelectorRuntime:
    """Load unchanged frozen inputs under only the new post-hoc 0.70 contract."""
    load_locked_config()
    base_path = (ROOT / BASE_CHECKPOINT_RELATIVE).resolve(strict=True)
    selector_path = (ROOT / SELECTOR_CHECKPOINT_RELATIVE).resolve(strict=True)
    if sha256(base_path) != FROZEN_CHECKPOINT_SHA256:
        raise RuntimeError("frozen FCOS checkpoint SHA-256 mismatch")
    if sha256(selector_path) != SELECTOR_CHECKPOINT_SHA256:
        raise RuntimeError("frozen relational-selector checkpoint SHA-256 mismatch")
    checkpoint = torch.load(selector_path, map_location="cpu", weights_only=True)
    _validate_historical_checkpoint(checkpoint)
    selector = PersonRelationalSelector()
    selector.load_state_dict(checkpoint["selector"], strict=True)
    selector.to(device).eval()
    for parameter in selector.parameters():
        parameter.requires_grad_(False)
    if any(not bool(torch.isfinite(parameter).all()) for parameter in selector.parameters()):
        raise FloatingPointError("non-finite frozen relational-selector parameter")
    if any(parameter.requires_grad for parameter in selector.parameters()):
        raise RuntimeError("relational-selector parameters are not frozen")
    mapped = calibration_at_raw_threshold()
    if mapped.dtype != torch.float32 or not torch.equal(
        mapped, torch.tensor([CANONICAL_PERSON_THRESHOLD], dtype=torch.float32),
    ):
        raise RuntimeError("locked calibration does not map the raw boundary to FP32 0.20")
    runtime = RevisedSelectorRuntime(
        selector=selector,
        deployment_bias=DEPLOYMENT_LOGIT_BIAS,
        selector_checkpoint=selector_path,
        selector_checkpoint_sha256=SELECTOR_CHECKPOINT_SHA256,
        base_checkpoint=base_path,
        base_checkpoint_sha256=FROZEN_CHECKPOINT_SHA256,
        historical_status=str(checkpoint["status"]),
    )
    if require_holdout_verification:
        if not HOLDOUT_VERIFICATION_PATH.is_file():
            raise RuntimeError("reviewed train-holdout verification is missing; inference prohibited")
        report = json.loads(HOLDOUT_VERIFICATION_PATH.read_text(encoding="utf-8"))
        _validate_holdout_verification_report(report)
    return runtime


def _verify_metric_counts(report: Mapping[str, Any], expected: Mapping[str, int]) -> None:
    if any(int(report.get(name, -1)) != int(value) for name, value in expected.items()):
        raise RuntimeError("recomputed revised holdout counts do not match the locked frontier point")
    precision, recall = _expected_metrics(expected)
    if (not math.isclose(float(report.get("precision", -1.0)), precision,
                         rel_tol=0.0, abs_tol=1e-15)
            or not math.isclose(float(report.get("recall", -1.0)), recall,
                                rel_tol=0.0, abs_tol=1e-15)):
        raise RuntimeError("recomputed revised holdout metric arithmetic drift")


def _validate_holdout_verification_report(report: Mapping[str, Any]) -> None:
    if (report.get("schema") != "splitfusion_fcos_relational_p070_holdout_verification_v1"
            or report.get("source_split") != "train_holdout_only"
            or report.get("validation_or_test_accessed") is not False
            or report.get("historical_status_unchanged") != HISTORICAL_STATUS
            or int(report.get("historical_score_boundaries_verified", -1))
            != EXPECTED_SCORE_BOUNDARIES
            or float(report.get("raw_threshold", -1.0)) != RAW_RELATIONAL_THRESHOLD
            or float(report.get("deployment_bias", 0.0)) != DEPLOYMENT_LOGIT_BIAS
            or float(report.get("deployment_threshold", -1.0)) != CANONICAL_PERSON_THRESHOLD
            or report.get("revised_0_70_gate_passed") is not True
            or not math.isclose(float(report.get("maximin_minimum", -1.0)),
                                0.77443315089914, rel_tol=0.0, abs_tol=1e-15)):
        raise RuntimeError("reviewed train-holdout verification drift; inference prohibited")
    raw, deployment = report.get("raw", {}), report.get("deployment", {})
    _verify_metric_counts(raw.get("aggregate", {}), EXPECTED_COUNTS["aggregate"])
    _verify_metric_counts(deployment.get("aggregate", {}), EXPECTED_COUNTS["aggregate"])
    for episode in HOLDOUT_EXPERIMENT_IDS:
        _verify_metric_counts(raw.get("episodes", {}).get(episode, {}),
                              EXPECTED_COUNTS["episodes"][episode])
        _verify_metric_counts(deployment.get("episodes", {}).get(episode, {}),
                              EXPECTED_COUNTS["episodes"][episode])
    for raw_report, deployment_report in (
        (raw["aggregate"], deployment["aggregate"]),
        *((raw["episodes"][episode], deployment["episodes"][episode])
          for episode in HOLDOUT_EXPERIMENT_IDS),
    ):
        if (float(raw_report.get("threshold", -1.0)) != RAW_RELATIONAL_THRESHOLD
                or float(deployment_report.get("threshold", -1.0))
                != CANONICAL_PERSON_THRESHOLD):
            raise RuntimeError("reviewed holdout threshold arithmetic drift; inference prohibited")


def verify_revised_holdout(runtime: RevisedSelectorRuntime, device: torch.device) -> dict[str, Any]:
    """Reproduce the selected train-only frontier point with canonical rematching."""
    caches = load_locked_caches()
    frames = _holdout_outputs(runtime.selector, caches, device)
    raw_scores = [frame.refined_scores for frame in frames]
    flat_scores = torch.cat(raw_scores)
    raw_value = torch.tensor(RAW_RELATIONAL_THRESHOLD, dtype=torch.float32)
    if not bool(flat_scores.eq(raw_value).any()):
        raise RuntimeError("selected raw threshold is not an observed tied-score boundary")
    raw = _rematched_metrics_at_threshold(frames, raw_scores, RAW_RELATIONAL_THRESHOLD)
    calibrated_scores = [
        refined_person_scores(frame.base_scores, frame.residual_logits, DEPLOYMENT_LOGIT_BIAS)
        for frame in frames
    ]
    deployed = _rematched_metrics_at_threshold(
        frames, calibrated_scores, CANONICAL_PERSON_THRESHOLD,
    )
    _verify_metric_counts(raw["aggregate"], EXPECTED_COUNTS["aggregate"])
    _verify_metric_counts(deployed["aggregate"], EXPECTED_COUNTS["aggregate"])
    for episode in HOLDOUT_EXPERIMENT_IDS:
        _verify_metric_counts(raw["episodes"][episode], EXPECTED_COUNTS["episodes"][episode])
        _verify_metric_counts(deployed["episodes"][episode], EXPECTED_COUNTS["episodes"][episode])
    scopes = ((raw["aggregate"], deployed["aggregate"]), *(
        (raw["episodes"][episode], deployed["episodes"][episode])
        for episode in HOLDOUT_EXPERIMENT_IDS
    ))
    if any(any(int(left[name]) != int(right[name]) for name in ("tp", "fp", "fn", "ignored"))
           for left, right in scopes):
        raise RuntimeError("raw-threshold and deployment-threshold rematching disagree")
    minimum = min(
        metric
        for report in (raw["aggregate"], *raw["episodes"].values())
        for metric in (float(report["precision"]), float(report["recall"]))
    )
    if not math.isclose(minimum, 0.77443315089914, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("recomputed revised maximin value drift")
    return {
        "schema": "splitfusion_fcos_relational_p070_holdout_verification_v1",
        "source_split": "train_holdout_only",
        "validation_or_test_accessed": False,
        "historical_status_unchanged": runtime.historical_status,
        "historical_score_boundaries_verified": EXPECTED_SCORE_BOUNDARIES,
        "raw_threshold": RAW_RELATIONAL_THRESHOLD,
        "deployment_bias": DEPLOYMENT_LOGIT_BIAS,
        "deployment_threshold": CANONICAL_PERSON_THRESHOLD,
        "raw": raw,
        "deployment": deployed,
        "maximin_minimum": minimum,
        "revised_0_70_gate_passed": minimum >= 0.70,
    }
