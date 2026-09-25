"""Seal the held-out acceptance of the Run-4 d=2 UL-MCS generator.

This is an offline dynamics qualification, not a live/deployment claim.  The
model is fitted exclusively from the registered FIT transitions; the
contiguous internal-validation transitions are opened only by the scoring
function.  Acceptance requires finite proper scores and a strictly better
top-1 successor prediction than the no-change persistence comparator.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import dynamic_mcs_273prb_evidence as evidence_module
from . import mcs_transition_provider as provider


SCHEMA = "splitfusion.run4.mcs_transition_acceptance.v1"
MANIFEST_SCHEMA = "splitfusion.run4.mcs_transition_acceptance_manifest.v1"
EVIDENCE_CLASS = "HELD_OUT_INTERNAL_MCS_DYNAMICS_QUALIFICATION_NOT_DEPLOYMENT"
SEALED_DIR = Path(__file__).with_name("sealed_mcs_transition_v1")
REPORT_NAME = "MCS_TRANSITION_ACCEPTANCE.json"
REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256 = (
    "2a96b27d13b816dfdafe3c6369589d436a0a37fec5234320739e7acef9240be2"
)


class McsAcceptanceError(RuntimeError):
    """The MCS acceptance artifact is missing, changed, or contradictory."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_report() -> dict[str, Any]:
    """Recompute the immutable held-out report from pinned source evidence."""

    evidence = evidence_module.load_dynamic_mcs_273prb_evidence()
    model = provider.fit_mcs_markov_model(evidence)
    metrics = provider.evaluate_internal_validation(model, evidence)
    finite_scores = all(
        math.isfinite(float(metrics[name]))
        for name in ("brier_mean", "mean_negative_log_likelihood")
    )
    top1_beats_persistence = (
        float(metrics["top1_accuracy"])
        > float(metrics["persistence_accuracy"])
    )
    identities_close = (
        metrics["model_binding_sha256"] == model.binding_sha256
        and metrics["fit_transitions"]
        == len(evidence.fit_transitions)
        and metrics["validation_transitions"]
        == len(evidence.internal_validation_transitions)
    )
    accepted = bool(finite_scores and top1_beats_persistence and identities_close)
    return {
        "schema": SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "accepted_for_offline_run4_mcs_dynamics": accepted,
        "production_or_deployment_authorized": False,
        "source_evidence_sha256": evidence.canonical_evidence_sha256,
        "model_binding_sha256": model.binding_sha256,
        "fit_transition_count": len(evidence.fit_transitions),
        "validation_transition_count": len(evidence.internal_validation_transitions),
        "validation_metrics": metrics,
        "acceptance_checks": {
            "finite_brier_and_nll": finite_scores,
            "top1_strictly_beats_persistence": top1_beats_persistence,
            "source_and_counts_close_exactly": identities_close,
        },
        "causal_scope": {
            "duration_tensors": 2,
            "fit_only_model_construction": True,
            "hidden_profile_is_not_a_model_input": True,
            "action_payload_backlog_reward_are_not_model_inputs": True,
            "validation_is_not_used_for_fit_or_hyperparameter_selection": True,
        },
        "limitations": [
            "internal validation uses held-out contiguous portions of one realization per dynamic profile",
            "this qualifies the initial offline MCS generator, not unseen-channel generalization",
            "live missing or stale UE MCS still requires the external fallback",
        ],
    }


def write_create_only(output_dir: Path) -> tuple[Path, Path]:
    """Create a report and manifest without overwriting prior evidence."""

    output_dir.mkdir(parents=True, exist_ok=False)
    report = build_report()
    if not report["accepted_for_offline_run4_mcs_dynamics"]:
        raise McsAcceptanceError("held-out MCS dynamics acceptance failed")
    result_path = output_dir / REPORT_NAME
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "result": {"path": REPORT_NAME, "sha256": _sha256(result_path)},
        "sources": {
            "acceptance": {
                "path": str(Path(__file__).relative_to(Path(__file__).parents[2])),
                "sha256": _sha256(Path(__file__)),
            },
            "provider": {
                "path": str(Path(provider.__file__).relative_to(Path(__file__).parents[2])),
                "sha256": _sha256(Path(provider.__file__)),
            },
            "evidence_loader": {
                "path": str(Path(evidence_module.__file__).relative_to(Path(__file__).parents[2])),
                "sha256": _sha256(Path(evidence_module.__file__)),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result_path, manifest_path


def load_registered_acceptance(
    directory: Path = SEALED_DIR,
) -> dict[str, Any]:
    """Hash-verify and semantically recompute the registered acceptance."""

    manifest_path = directory / "manifest.json"
    report_path = directory / REPORT_NAME
    if not manifest_path.is_file() or not report_path.is_file():
        raise McsAcceptanceError("sealed MCS acceptance files are absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise McsAcceptanceError("MCS acceptance manifest schema drifted")
    expected_source_paths = {
        "acceptance": Path(__file__),
        "provider": Path(provider.__file__),
        "evidence_loader": Path(evidence_module.__file__),
    }
    sources = manifest.get("sources")
    if type(sources) is not dict or set(sources) != set(expected_source_paths):
        raise McsAcceptanceError("MCS acceptance source inventory drifted")
    for name, path in expected_source_paths.items():
        expected = {
            "path": str(path.relative_to(path.parents[2])),
            "sha256": _sha256(path),
        }
        if sources.get(name) != expected:
            raise McsAcceptanceError(f"MCS acceptance source {name} drifted")
    if manifest.get("result") != {
        "path": REPORT_NAME,
        "sha256": _sha256(report_path),
    }:
        raise McsAcceptanceError("MCS acceptance result hash drifted")
    if _sha256(report_path) != REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256:
        raise McsAcceptanceError("MCS acceptance is not the registered result")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report != build_report():
        raise McsAcceptanceError("MCS acceptance no longer recomputes exactly")
    if report.get("accepted_for_offline_run4_mcs_dynamics") is not True:
        raise McsAcceptanceError("MCS dynamics are not accepted")
    return report


__all__ = [
    "EVIDENCE_CLASS",
    "McsAcceptanceError",
    "REGISTERED_MCS_ACCEPTANCE_RESULT_SHA256",
    "build_report",
    "load_registered_acceptance",
    "write_create_only",
]
