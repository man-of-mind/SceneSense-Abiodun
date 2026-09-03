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

There is no bounded or partial mode on this command. Every pass evaluates all
3,284 reserved frames, and no partial execution can emit a holdout-selection
report, a checkpoint decision or the completion terminal. `ae_holdout_selection.run_pass`
still takes an internal `limit`, which CPU tests may use, but it is not reachable
from this CLI and `main` always passes `limit=None`.

Recovery is per pass and durable. Before the first pass the run writes an atomic
manifest binding the family, the training run, the three candidate hashes, the
frozen bindings, the noAE reference, the scorer identity, the exact epochs, q
values and frame count, and the runner source identity. After each complete
inference-and-scoring pass one compact setting record is written atomically.
`--resume` requires a manifest whose identity is bit-identical to the live one,
reuses only fully validated setting records, runs only the missing
checkpoint/q pairs, and refuses -- rather than overwriting or re-measuring --
any record that fails validation. The selection is emitted only once exactly
twelve valid records exist.

Selecting a checkpoint is not a service-ready claim, and deployment-path
UINT8+zstd validation is a later, separately authorized phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    RANKING_CRITERIA,
    RANKING_RULE,
    aggregate_by_checkpoint,
    ranking_key,
    run_pass,
)

DATALOADER_WORKERS = 8

# One completed inference-and-scoring pass per file, under the run directory.
SETTINGS_DIRNAME = "settings"

# The AE package modules whose identity this runner is pinned to by name. The
# whole package source map is bound as well (through the frozen binding); these
# are called out so a reviewer can see which files define the runner itself.
RUNNER_SOURCES = (
    "ae_phase10_holdout_selection.py",
    "ae_phase10_common.py",
    "ae_holdout_selection.py",
    "ae_training_common.py",
    "ae_model.py",
    "ae_composition.py",
    "ae_loss.py",
    "ae_contract.py",
)

# Registered gate scoring, shared with AE128 and unchanged.
evaluate_same_q_gates = common.evaluate_same_q_gates


def registered_plans() -> list[Any]:
    """The four registered selection q, quantized once, in registered order."""
    return [continuous_q.quantize_q(float(q)) for q in common.AE_HOLDOUT_Q_VALUES]


def expected_pairs() -> list[tuple[int, int]]:
    """Every (epoch, q_e4) pair this command must produce, in run order."""
    return [
        (int(epoch), plan.q_e4)
        for epoch in common.AE_CANDIDATE_EPOCHS
        for plan in registered_plans()
    ]


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
# Durable per-pass recovery: one run manifest, then one record per pass
# ---------------------------------------------------------------------------


def run_identity(
    *,
    bottleneck: int,
    training_dir: Path,
    training_report_sha256: str,
    candidate_hashes: Mapping[str, str],
    binding: Mapping[str, Any],
    scorer_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Everything a resumed run must be measuring the same thing under.

    Nothing derived, nothing optional: the family, the exact scientific scope,
    the training run and its three candidate hashes, every frozen binding (which
    includes the whole AE package and hybrid-q source maps), the frozen noAE
    reference, the scorer identity and the named runner sources. Any change to
    any of them changes the digest, and a resume against a different digest is
    refused rather than mixed.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    package = dict(binding["ae_package_source_sha256"])
    missing = [name for name in RUNNER_SOURCES if name not in package]
    if missing:
        raise guards.HybridQConfigError(
            f"the AE package source map does not cover {missing}"
        )
    return {
        **family.family_fields(size),
        "command": family.holdout_token(size),
        "epochs": [int(epoch) for epoch in common.AE_CANDIDATE_EPOCHS],
        "q_values": [float(plan.wire_q) for plan in registered_plans()],
        "q_e4_values": [plan.q_e4 for plan in registered_plans()],
        "holdout_frames": contract.TRAIN_HOLDOUT_FRAMES,
        "transport": common.AE_HOLDOUT_QUANTIZER,
        "bounded_or_partial_passes_possible": False,
        "training_run_path": str(training_dir),
        "training_report": family.training_report_filename(size),
        "training_report_sha256": str(training_report_sha256),
        "candidate_checkpoints": {
            str(name): str(digest) for name, digest in sorted(candidate_hashes.items())
        },
        "binding": common.binding_fields(binding),
        "noae_holdout_reference": {
            "path": common.NOAE_HOLDOUT_RELPATH,
            "sha256": common.NOAE_HOLDOUT_SHA256,
        },
        "scorer_sha256": {str(k): str(v) for k, v in sorted(scorer_sha256.items())},
        "runner_sources": {name: package[name] for name in RUNNER_SOURCES},
    }


def identity_digest(identity: Mapping[str, Any]) -> str:
    """One sha256 over the canonical identity document."""
    canonical = json.dumps(
        dict(identity), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_document(
    identity: Mapping[str, Any], bottleneck: int
) -> dict[str, Any]:
    size = family.require_phase10_bottleneck(bottleneck)
    return {
        "schema": family.holdout_manifest_schema(size),
        **family.family_fields(size),
        "terminal_when_complete": family.holdout_terminal(size),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_identity_sha256": identity_digest(identity),
        "identity": dict(identity),
        "settings_directory": SETTINGS_DIRNAME,
        "expected_setting_count": len(expected_pairs()),
        "expected_settings": [
            family.setting_record_filename(size, epoch, q_e4)
            for epoch, q_e4 in expected_pairs()
        ],
        "policy": (
            "every pass evaluates all holdout frames; a partial run emits no "
            "selection, no decision and no terminal, and --resume reuses only "
            "fully validated setting records"
        ),
    }


def write_run_manifest(output: Path, identity: Mapping[str, Any], bottleneck: int) -> str:
    """Write the manifest atomically, before the first pass."""
    size = family.require_phase10_bottleneck(bottleneck)
    return common.atomic_write_json(
        manifest_document(identity, size),
        Path(output) / family.holdout_manifest_filename(size),
    )


def load_run_manifest(
    output: Path, identity: Mapping[str, Any], bottleneck: int
) -> dict[str, Any]:
    """Require an existing manifest that binds exactly this run identity."""
    size = family.require_phase10_bottleneck(bottleneck)
    path = Path(output) / family.holdout_manifest_filename(size)
    if not path.is_file():
        raise guards.HybridQConfigError(
            f"--resume requires an existing run manifest: {path} does not exist"
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != family.holdout_manifest_schema(size):
        raise guards.HybridQConfigError("holdout run manifest schema drift")
    family.require_family_fields(document, size, what=path.name)
    expected = manifest_document(identity, size)
    if document.get("run_identity_sha256") != expected["run_identity_sha256"]:
        raise guards.HybridQConfigError(
            "the run manifest binds a different run identity; refusing to resume "
            "into a run measured under different inputs"
        )
    if dict(document.get("identity", {})) != dict(identity):
        raise guards.HybridQConfigError("run manifest identity drift")
    if list(document.get("expected_settings", [])) != expected["expected_settings"]:
        raise guards.HybridQConfigError("run manifest expected-setting drift")
    return document


def _require_finite_number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise guards.HybridQPayloadError(f"{what} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise guards.HybridQNumericalError(f"{what} is not finite")
    return number


PASS_DECLARATIONS = {
    "uint8_zstd_uint6_uint4_run": False,
    "bounded_or_partial_pass": False,
    "training_run_here": False,
    "validation_or_test_accessed": False,
    "fit_teacher_shards_deserialized": 0,
    "frozen_state_unchanged_after_pass": True,
    "carla_launched": False,
}

RECONSTRUCTION_TOTALS = (
    "plain_squared_error_numerator",
    "plain_reference_energy_denominator",
    "combined_importance_numerator",
    "combined_importance_reference_energy_denominator",
    "global_plain_reconstruction",
    "global_combined_importance_reconstruction",
    "global_total_loss",
)


def build_setting_record(
    *,
    bottleneck: int,
    run_identity_sha256: str,
    epoch: int,
    plan: Any,
    checkpoint: str,
    checkpoint_sha256: str,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """One compact, self-describing record of one completed pass."""
    size = family.require_phase10_bottleneck(bottleneck)
    return {
        "schema": family.holdout_setting_schema(size),
        **family.family_fields(size),
        "run_identity_sha256": str(run_identity_sha256),
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": int(epoch),
        "q": float(plan.wire_q),
        "q_e4": int(plan.q_e4),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": str(checkpoint_sha256),
        "frames": int(evaluation["frames"]),
        "inference_passes": 1,
        "retained_cells": int(evaluation["retained_cells"]),
        "dropped_cells": int(evaluation["dropped_cells"]),
        "ranker_invoked": bool(evaluation["ranker_invoked"]),
        "ranker_invocations": int(evaluation["ranker_invocations"]),
        "transport": str(evaluation["transport"]),
        "declarations": dict(PASS_DECLARATIONS),
        "evaluation": dict(evaluation),
    }


def validate_setting_record(
    record: Mapping[str, Any],
    *,
    bottleneck: int,
    run_identity_sha256: str,
    epoch: int,
    plan: Any,
    checkpoint: str,
    checkpoint_sha256: str,
    what: str,
) -> dict[str, Any]:
    """Fail closed unless this record is a complete pass of exactly this setting.

    A record that does not validate is refused outright: it is never overwritten
    and never silently re-measured, because either answer would hide which of the
    two measurements the report is actually built from.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    if record.get("schema") != family.holdout_setting_schema(size):
        raise guards.HybridQConfigError(f"{what} setting schema drift")
    family.require_family_fields(record, size, what=what)
    if record.get("run_identity_sha256") != str(run_identity_sha256):
        raise guards.HybridQConfigError(f"{what} belongs to a different run identity")
    if int(record["epoch"]) != int(epoch):
        raise guards.HybridQConfigError(f"{what} epoch drift")
    if int(record["q_e4"]) != int(plan.q_e4):
        raise guards.HybridQConfigError(f"{what} q_e4 drift")
    if continuous_q.quantize_q(float(record["q"])).q_e4 != int(plan.q_e4):
        raise guards.HybridQConfigError(f"{what} q disagrees with its own q_e4")
    if str(record["checkpoint"]) != str(checkpoint):
        raise guards.HybridQConfigError(f"{what} checkpoint name drift")
    if str(record["checkpoint_sha256"]) != str(checkpoint_sha256):
        raise guards.HybridQConfigError(f"{what} checkpoint sha256 drift")
    if int(record["frames"]) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError(
            f"{what} covers {record['frames']} frames, the holdout has "
            f"{contract.TRAIN_HOLDOUT_FRAMES}"
        )
    if int(record["inference_passes"]) != 1:
        raise guards.HybridQConfigError(f"{what} is not exactly one inference pass")
    if int(record["retained_cells"]) != contract.keep_count(plan.wire_q):
        raise guards.HybridQConfigError(f"{what} keep-count drift")
    if int(record["dropped_cells"]) != contract.drop_count(plan.wire_q):
        raise guards.HybridQConfigError(f"{what} drop-count drift")
    if bool(record["ranker_invoked"]) != (not plan.is_bypass):
        raise guards.HybridQConfigError(f"{what} ranker-use drift")
    expected_invocations = 0 if plan.is_bypass else contract.TRAIN_HOLDOUT_FRAMES
    if int(record["ranker_invocations"]) != expected_invocations:
        raise guards.HybridQConfigError(f"{what} ranker-invocation drift")
    if str(record["transport"]) != common.AE_HOLDOUT_QUANTIZER:
        raise guards.HybridQConfigError(f"{what} transport drift")
    if dict(record["declarations"]) != PASS_DECLARATIONS:
        raise guards.HybridQConfigError(f"{what} scope declaration drift")

    evaluation = dict(record["evaluation"])
    family.require_family_fields(evaluation, size, what=f"{what} evaluation")
    if int(evaluation["epoch"]) != int(epoch):
        raise guards.HybridQConfigError(f"{what} evaluation epoch drift")
    if continuous_q.quantize_q(float(evaluation["q"])).q_e4 != int(plan.q_e4):
        raise guards.HybridQConfigError(f"{what} evaluation q drift")
    if int(evaluation["frames"]) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError(f"{what} evaluation frame-count drift")
    if str(evaluation["checkpoint_sha256"]) != str(checkpoint_sha256):
        raise guards.HybridQConfigError(f"{what} evaluation checkpoint drift")

    metrics = dict(evaluation["metrics"])
    if set(metrics) != set(contract.PROTECTED_METRICS):
        raise guards.HybridQConfigError(f"{what} protected metric set drift")
    for name, value in metrics.items():
        _require_finite_number(value, f"{what} metric {name}")

    reconstruction = dict(evaluation["reconstruction"])
    if int(reconstruction["frames"]) != contract.TRAIN_HOLDOUT_FRAMES:
        raise guards.HybridQConfigError(f"{what} reconstruction frame-count drift")
    for name in RECONSTRUCTION_TOTALS:
        if name not in reconstruction:
            raise guards.HybridQConfigError(f"{what} reconstruction is missing {name}")
        _require_finite_number(reconstruction[name], f"{what} reconstruction {name}")

    gate_result = dict(evaluation["gate_result"])
    gates = dict(gate_result["gates"])
    registered = {name for name, _direction, _bound in contract.HOLDOUT_PRESERVATION_GATES}
    if set(gates) != registered or len(gates) != GATE_COUNT:
        raise guards.HybridQConfigError(f"{what} preservation-gate set drift")
    if int(gate_result["gates_total"]) != GATE_COUNT:
        raise guards.HybridQConfigError(f"{what} gate count drift")
    for name, row in gates.items():
        _require_finite_number(row["degradation"], f"{what} gate {name} degradation")
        _require_finite_number(
            row["normalized_degradation"], f"{what} gate {name} normalized degradation"
        )
        if not isinstance(row["passed"], bool):
            raise guards.HybridQConfigError(f"{what} gate {name} verdict is not boolean")
    passed = sum(1 for row in gates.values() if row["passed"])
    if int(gate_result["gates_passed"]) != passed:
        raise guards.HybridQConfigError(f"{what} gate pass count disagrees with its gates")
    if bool(gate_result["all_passed"]) != (passed == GATE_COUNT):
        raise guards.HybridQConfigError(f"{what} all-passed flag disagrees with its gates")
    _require_finite_number(
        gate_result["worst_normalized_degradation"],
        f"{what} worst normalized degradation",
    )
    return evaluation


def collect_completed_settings(
    settings_dir: Path,
    *,
    bottleneck: int,
    run_identity_sha256: str,
    candidate_hashes: Mapping[str, str],
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[tuple[int, int]]]:
    """Split the twelve settings into validated-reusable and still-missing.

    Every record found is validated in full; an invalid one raises here rather
    than being re-measured. A file in the settings directory that is not one of
    the twelve registered records is refused too.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    settings_dir = Path(settings_dir)
    expected = {
        family.setting_record_filename(size, epoch, q_e4): (epoch, q_e4)
        for epoch, q_e4 in expected_pairs()
    }
    if settings_dir.is_dir():
        foreign = sorted(
            child.name
            for child in settings_dir.iterdir()
            if child.name not in expected
        )
        if foreign:
            raise guards.HybridQOwnershipError(
                f"{settings_dir} holds file(s) that are not registered "
                f"{family.family_label(size)} setting records: {foreign}"
            )

    plans = {plan.q_e4: plan for plan in registered_plans()}
    reusable: dict[tuple[int, int], dict[str, Any]] = {}
    pending: list[tuple[int, int]] = []
    for name, (epoch, q_e4) in expected.items():
        path = settings_dir / name
        if not path.is_file():
            pending.append((epoch, q_e4))
            continue
        checkpoint = family.candidate_filename(size, epoch)
        if checkpoint not in candidate_hashes:
            raise guards.HybridQConfigError(
                f"the training run declares no {checkpoint} for {name}"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        reusable[(epoch, q_e4)] = validate_setting_record(
            record,
            bottleneck=size,
            run_identity_sha256=run_identity_sha256,
            epoch=epoch,
            plan=plans[q_e4],
            checkpoint=checkpoint,
            checkpoint_sha256=candidate_hashes[checkpoint],
            what=name,
        )
    pending.sort(key=lambda pair: expected_pairs().index(pair))
    return reusable, pending


def contract_alias(dataset_root: Path, alias_root: Path) -> Path:
    """Build the read-only train-contract alias, or reuse an identical one.

    The frozen builder refuses to overwrite an existing alias, which is correct;
    on resume the alias from the interrupted run is still there, so it is
    re-resolved and checked instead of rebuilt.
    """
    alias_root = Path(alias_root)
    links = {name: alias_root / f"contracts/{name}/val" for name in ("v010", "v025")}
    if not any(link.is_symlink() or link.exists() for link in links.values()):
        return build_contract_alias(dataset_root, alias_root)
    for name, link in links.items():
        if not link.is_symlink():
            raise guards.HybridQConfigError(f"{link} is not the contract alias symlink")
        if link.resolve(strict=True) != (
            dataset_root / f"contracts/{name}/train"
        ).resolve(strict=True):
            raise guards.HybridQConfigError(
                f"the existing {name} contract alias does not resolve to the train split"
            )
    return alias_root


# ---------------------------------------------------------------------------
# The completed training run this command consumes
# ---------------------------------------------------------------------------


def load_training_run(training_dir: Path, bottleneck: int) -> tuple[dict[str, Any], str]:
    """Verify a completed same-family training run; return its report and hash.

    The trainer writes its terminal marker last, holding the sha256 of the
    report it had just written. That digest is therefore the trainer's own
    statement of what it produced, and it must still equal the report's current
    hash: if the report has been edited, replaced or truncated since the run
    ended, this command refuses rather than selecting a checkpoint against a
    document the trainer never signed.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    label = family.family_label(size)
    terminal = family.training_terminal(size)
    marker_path = training_dir / terminal
    if not marker_path.is_file():
        raise guards.HybridQConfigError(
            f"holdout selection requires a separately authorized completed {label} "
            f"training run declaring {terminal}"
        )
    report_path = training_dir / family.training_report_filename(size)
    if not report_path.is_file():
        raise guards.HybridQConfigError(
            f"{report_path.name} is missing; this is not a completed {label} run"
        )
    recorded = marker_path.read_text(encoding="utf-8").strip()
    observed = sha256_file(report_path)
    if recorded != observed:
        raise guards.HybridQConfigError(
            f"{terminal} records report sha256 {recorded!r}, but "
            f"{report_path.name} currently hashes to {observed!r}"
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
    return report, observed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The public command surface.

    There is deliberately no bounded, smoke or frame-limiting option here: the
    scientific holdout command always evaluates all
    3,284 frames for all twelve checkpoint/q pairs.
    """
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
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    bottleneck = family.require_token_agrees_with_bottleneck(
        args.execute, args.bottleneck, kind="holdout"
    )
    family.bind_process_family(bottleneck)
    label = family.family_label(bottleneck)
    slug = family.family_slug(bottleneck)
    terminal = family.holdout_terminal(bottleneck)

    training_dir = args.training.resolve(strict=True)
    training_report, training_report_sha256 = load_training_run(training_dir, bottleneck)
    candidate_hashes = {
        str(name): str(digest)
        for name, digest in training_report["candidate_checkpoints"].items()
    }
    output = family.holdout_selection_dir(training_dir, bottleneck)
    if args.resume:
        if not output.is_dir():
            raise guards.HybridQConfigError(
                f"--resume requires an existing selection directory: {output} "
                "does not exist"
            )
        if (output / terminal).is_file():
            raise guards.HybridQConfigError(
                f"{output / terminal} already exists: this selection completed, "
                "and a completed decision is not rewritten"
            )
    elif output.exists():
        raise guards.HybridQConfigError(f"create-only: {output} already exists")
    if not torch.cuda.is_available():
        raise RuntimeError(f"Phase-10A {label} holdout selection requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(family.process_seed(bottleneck))

    binding = common.bind_frozen_inputs()
    delta = source_delta(binding)
    # Exactly the frozen noAE same-q holdout reference AE128 was scored against.
    reference = common.load_noae_holdout_reference()
    scorers = load_frozen_scorers()

    identity = run_identity(
        bottleneck=bottleneck,
        training_dir=training_dir,
        training_report_sha256=training_report_sha256,
        candidate_hashes=candidate_hashes,
        binding=binding,
        scorer_sha256=scorers.sha256,
    )
    run_sha256 = identity_digest(identity)
    output.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output / family.holdout_manifest_filename(bottleneck)
    if args.resume:
        manifest = load_run_manifest(output, identity, bottleneck)
    else:
        # Written before the first pass, atomically, then read back and checked
        # through exactly the validator a later --resume would use.
        write_run_manifest(output, identity, bottleneck)
        manifest = load_run_manifest(output, identity, bottleneck)
    manifest_sha256 = sha256_file(manifest_path)

    settings_dir = output / SETTINGS_DIRNAME
    settings_dir.mkdir(exist_ok=True)
    reusable, pending = collect_completed_settings(
        settings_dir,
        bottleneck=bottleneck,
        run_identity_sha256=run_sha256,
        candidate_hashes=candidate_hashes,
    )
    if reusable and not args.resume:
        raise guards.HybridQConfigError(
            "a fresh run cannot already hold completed setting records"
        )
    print(
        json.dumps(
            {
                "family": label,
                "run_identity_sha256": run_sha256,
                "reusing_completed_settings": len(reusable),
                "settings_to_run": len(pending),
            }
        ),
        flush=True,
    )

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

    alias_root = contract_alias(dataset_root, output / "train_contract_alias")
    qualification = load_p025_qualification()
    truth = load_holdout_person_truth(frame_ids)
    gt, _gt_states = scorers.load_gt(alias_root, contract.PRIMARY_CONTRACT)
    holdout_gt = {sample_id: gt.get(sample_id, []) for sample_id in frame_ids}
    ignore_cache: dict[str, Any] = {}

    predictions_root = output / "predictions"
    predictions_root.mkdir(exist_ok=True)

    started = time.time()
    plans = {plan.q_e4: plan for plan in registered_plans()}
    completed: dict[tuple[int, int], dict[str, Any]] = dict(reusable)
    executed: list[str] = []
    for epoch in common.AE_CANDIDATE_EPOCHS:
        missing = [q_e4 for (candidate_epoch, q_e4) in pending if candidate_epoch == epoch]
        if not missing:
            continue
        name = family.candidate_filename(bottleneck, epoch)
        path = training_dir / "checkpoints" / name
        digest = sha256_file(path)
        if digest != candidate_hashes[name]:
            raise guards.HybridQConfigError(f"candidate checkpoint {name} sha256 drift")
        autoencoder, metadata = family.load_candidate(
            path, bottleneck, epoch, device, binding
        )
        for q_e4 in missing:
            plan = plans[q_e4]
            prediction_dir = (
                predictions_root / f"{slug}_epoch{epoch:02d}_q{plan.q_e4:04d}"
            )
            if prediction_dir.exists():
                # Scratch output of a pass that never produced a setting record.
                # No record references it, so it is discarded and re-measured in
                # full rather than being partially reused.
                print(
                    f"[{slug}] discarding incomplete prediction scratch "
                    f"{prediction_dir.name} from an interrupted pass",
                    flush=True,
                )
                shutil.rmtree(prediction_dir)
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
                output=prediction_dir,
                workers=int(args.workers),
                # The scientific command never runs a bounded pass.
                limit=None,
            )
            guards.require_module_state_unchanged(model, frozen_model_state)
            guards.require_module_state_unchanged(ranker, frozen_ranker_state)
            scored = score_pass(
                result=raw,
                scorers=scorers,
                qualification=qualification,
                truth=truth,
                alias_root=alias_root,
                frame_ids=frame_ids,
                gt=holdout_gt,
                ignore_cache=ignore_cache,
                cross_check=False,
                require_defined=True,
            )
            scored["configuration"] = f"{slug}_epoch{epoch:02d}_q{plan.wire_q:.2f}"
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

            record_name = family.setting_record_filename(bottleneck, epoch, plan.q_e4)
            record = build_setting_record(
                bottleneck=bottleneck,
                run_identity_sha256=run_sha256,
                epoch=epoch,
                plan=plan,
                checkpoint=name,
                checkpoint_sha256=digest,
                evaluation=scored,
            )
            # Written atomically, and immediately read back through exactly the
            # validator a later --resume would use.
            common.atomic_write_json(record, settings_dir / record_name)
            validate_setting_record(
                json.loads(
                    (settings_dir / record_name).read_text(encoding="utf-8")
                ),
                bottleneck=bottleneck,
                run_identity_sha256=run_sha256,
                epoch=epoch,
                plan=plan,
                checkpoint=name,
                checkpoint_sha256=digest,
                what=record_name,
            )
            completed[(epoch, plan.q_e4)] = scored
            executed.append(record_name)
            print(
                json.dumps(
                    {
                        "pass": scored["configuration"],
                        "setting_record": record_name,
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

    # The selection is emitted only when all twelve settings exist, each one a
    # complete pass that validated under this run identity.
    final, still_missing = collect_completed_settings(
        settings_dir,
        bottleneck=bottleneck,
        run_identity_sha256=run_sha256,
        candidate_hashes=candidate_hashes,
    )
    if still_missing:
        raise guards.HybridQConfigError(
            f"{len(still_missing)} checkpoint/q setting(s) are still missing: "
            f"{still_missing}"
        )
    if sorted(final) != sorted(expected_pairs()) or len(final) != len(expected_pairs()):
        raise guards.HybridQConfigError("completed setting inventory drift")
    if sorted(completed) != sorted(final):
        raise guards.HybridQConfigError(
            "the in-memory pass inventory disagrees with the durable records"
        )
    evaluations = [final[pair] for pair in expected_pairs()]

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
            "bounded_or_partial_passes_possible": False,
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
        "run_recovery": {
            "manifest": family.holdout_manifest_filename(bottleneck),
            "manifest_sha256": manifest_sha256,
            "run_identity_sha256": run_sha256,
            "manifest_run_identity_sha256": manifest["run_identity_sha256"],
            "settings_directory": SETTINGS_DIRNAME,
            "settings_total": len(final),
            "resumed": bool(args.resume),
            "reused_settings": sorted(
                family.setting_record_filename(bottleneck, epoch, q_e4)
                for epoch, q_e4 in reusable
            ),
            "executed_settings": executed,
            "policy": (
                "one atomic record per completed checkpoint/q pass; a resumed "
                "run reuses only records that revalidate under this identity and "
                "re-measures nothing else"
            ),
        },
        "binding": binding,
        "perception_binding": perception,
        "hybrid_q_source_delta_since_phase4": delta,
        "teacher_store": store.provenance(),
        "training_run": {
            "path": str(training_dir),
            "report_sha256": training_report_sha256,
            "candidate_checkpoints": dict(candidate_hashes),
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
                "settings_reused": len(reusable),
                "settings_executed": len(executed),
                "report_sha256": sha256_file(report_path),
            }
        )
    )
    print(terminal)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
