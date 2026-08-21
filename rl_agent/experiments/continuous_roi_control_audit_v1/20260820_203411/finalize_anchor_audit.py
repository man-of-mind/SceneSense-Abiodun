#!/usr/bin/env python3
"""Create only the terminal files after completed anchor analysis.

The initial one-shot analyzer completed every CSV and figure but stopped while
rendering REPORT.md because a Markdown literal used unescaped f-string braces.
This resume path reads those immutable outputs and creates only files that do not
already exist. It performs no inference and overwrites nothing.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_existing_q_anchors as audit


OUT = Path(__file__).resolve().parent


def main() -> int:
    terminal_names = ("REPORT.md", "RESULTS_SUMMARY.json", "REVIEW_REQUIRED.json", "manifest.json")
    collisions = [name for name in terminal_names if (OUT / name).exists()]
    if collisions:
        raise FileExistsError(f"Create-only terminal collision: {collisions}")

    preflight = json.loads((OUT / "preflight.json").read_text(encoding="utf-8"))
    if not preflight.get("stop_rule_triggered") or preflight.get("dense_inference_started"):
        raise AssertionError("Expected stopped preflight with no dense inference")

    action_summary = json.loads((OUT / "q_action_structural_summary.json").read_text(encoding="utf-8"))
    split_manifest = pd.read_csv(OUT / "audit_split_manifest.csv")
    per_q = pd.read_csv(OUT / "per_q_results.csv")
    payload_pairs = pd.read_csv(OUT / "payload_frame_monotonicity.csv")
    interpolation = pd.read_csv(OUT / "coarse_anchor_interpolation.csv")
    crossing_table = pd.read_csv(OUT / "branch_crossings.csv")

    test_q = per_q.loc[per_q.split == "audit_test"]
    test_id = test_q.loc[test_q.q <= audit.ID_Q_MAX]
    payload_branch_monotone = (
        test_id.sort_values("q")
        .groupby(["family", "quantizer"])
        .payload_bytes_mean.apply(lambda series: bool(np.all(np.diff(series.to_numpy()) <= 0)))
    )
    paired_id = payload_pairs.loc[(payload_pairs.split == "audit_test") & (payload_pairs.q1 <= audit.ID_Q_MAX)]
    min_frame_monotonic = float(paired_id.nonincreasing_or_tied_fraction.min())
    interpolation_test = interpolation.loc[interpolation.split == "audit_test"]
    interpolation_max = (
        interpolation_test.groupby("metric").absolute_error.max().reset_index().sort_values("metric")
    )
    crossing_id_count = int((crossing_table.interval_role == "in_distribution").sum()) if not crossing_table.empty else 0
    split_counts = split_manifest.audit_split.value_counts().to_dict()
    branch_snapshot = test_q.loc[np.isclose(test_q.q, 0.7), [
        "family", "quantizer", "payload_bytes_mean", "veh_precision", "veh_recall",
        "ped_precision", "ped_recall", "fp_per_frame", "xy_mae_m", "xy_rmse_m", "miou",
    ]].sort_values(["family", "quantizer"])

    report = f"""# Continuous ROI-Drop Control Audit

Audit: `continuous_roi_control_audit_v1/20260820_203411`  
Terminal: **`INSUFFICIENT_EVIDENCE`**  
Controller recommendation: **keep measured discrete q anchors** pending dense-q evidence.

## Answer

The current 72-profile evidence does **not** establish q as a valid continuous
control in any of the 12 fixed family/quantizer branches. It contains only four
in-distribution q anchors (`0,.3,.5,.7`), so it cannot resolve local behavior at
the planned 0.05 scale or score held-out midpoint q values. This is not a finding
that q is intrinsically discrete-only: the preregistered `DISCRETE_ONLY` terminal
also requires a complete dense-q run. The defensible status is
`INSUFFICIENT_EVIDENCE`, with the controller remaining discrete.

## Preflight and stop rule

The frozen plan was written before inference. Preflight then triggered its stop
rule because the recorded dataset manifest/object file are absent and PyTorch
reports CUDA unavailable. No CPU full-run fallback was allowed. Dense inference,
CARLA, and OAI were not started; all results below derive from immutable existing
anchor CSVs.

- Planned grid: 19 q values; 228 profiles; 492,936 profile-frame evaluations.
- Audit validation: {int(split_counts.get('audit_validation', 0)):,} unique frames.
- Frozen audit test: {int(split_counts.get('audit_test', 0)):,} unique frames.
- Identifier and trajectory-block overlap: zero.

## Production q semantics and piecewise action

Production and evaluator code both use rank drop: independently for the native
low/high feature maps, compute objectness ordering and zero the
`round(q*N)` lowest-ranked cells. q is **not** a score threshold. With 5,184 low
cells and 1,296 high cells, `[0,.8]` has
**{action_summary['joint_plateau_count']:,} joint mask-count plateaus** separated
by {action_summary['joint_transition_count_open_interval']:,} transitions. Plateau
widths range from {action_summary['minimum_plateau_width_q']:.7f} to
{action_summary['maximum_plateau_width_q']:.7f} in q. Thus a float-valued API
produces a fine but integer, piecewise-constant actuator. The planned 0.05 grid
tests macro smoothness; it does not prove single-cell smoothness.

## Existing-anchor evidence

All 72 profiles are complete: 2,162 unique frames at six q anchors for every
branch. Aggregate audit-test payload is non-increasing over the measured
in-distribution anchors in **{int(payload_branch_monotone.sum())}/12 branches**.
The worst frame-paired non-increasing/tied rate over those large anchor gaps is
**{min_frame_monotonic:.2%}**. These are coarse payload facts, not continuous
quality validation.

At q=0.7, the 12 separate branch outcomes are:

{audit.markdown_table(branch_snapshot, list(branch_snapshot.columns), 4)}

Object quality is not assumed monotonic. `anchor_curve_diagnostics.csv` records
slope reversals per branch/metric, while `paired_anchor_bootstrap.csv` provides
2,000 trajectory-block paired confidence intervals. The audit found
**{crossing_id_count} in-distribution ordering crossings** across family or
quantizer comparisons, reinforcing that branch factors are not safely separable.

## Interpolation and smoothness limit

Only a coarse leave-one-current-anchor-out diagnostic is possible: predict q=.3
from q=0/.5 and q=.5 from q=.3/.7. Its audit-test maximum errors are:

{audit.markdown_table(interpolation_max, ['metric', 'absolute_error'], 5)}

Those targets are existing anchors and the gaps are 0.2--0.3 wide. They are not
the preregistered held-out `.05,.15,...,.75` midpoint test and cannot earn a
continuous terminal. No dense local discontinuity test was run. `.9/.98` remain
measured extrapolation references and are excluded from the continuous decision.

## Latency

q-gating and serialization latency is `NOT_MEASURED`. Historical end-to-end or
technical-smoke numbers do not isolate objectness/ranking, mask application,
integrated AE, quantization, serialization, and zstd-3 as required by the frozen
plan, so they were not substituted.

## Hybrid-action implication

Even if q later passes, the action is not plain continuous SAC/TD3: it is a
categorical choice among 12 `{{family, quantizer}}` branches plus a conditional
bounded `q in [0,.8]`. It needs an explicit hierarchy, parameterized-action
critic, or categorical branch policy with a conditional q actor. This audit adds
no q-selection reward and implements no RL agent.

## Smallest next experiment

Restore the exact hashed dataset and a CUDA device, then run **AE64/uint6 on the
{int(split_counts.get('audit_validation', 0))}-frame audit-validation split across
all 19 q values**: {int(split_counts.get('audit_validation', 0)) * 19:,}
profile-frame evaluations, reusing backbone features/ranking per frame. Stop
there if midpoint interpolation or local-jump criteria fail.

Before promotion, evidence must expand to all 12 branches on the frozen
{int(split_counts.get('audit_test', 0))}-frame audit test, pass the preregistered
branch criteria with trajectory-block paired uncertainty, include
production-equivalent gate/serialization timing, and show 12/12 branch support.
Until then: no q promotion, no registry change, and no continuous/hybrid policy
implementation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    summary = {
        "audit_id": "continuous_roi_control_audit_v1/20260820_203411",
        "terminal": "INSUFFICIENT_EVIDENCE",
        "controller_recommendation": "KEEP_MEASURED_DISCRETE_Q_ANCHORS",
        "dense_inference_started": False,
        "measured_profiles_complete": 72,
        "planned_dense_profiles": 228,
        "planned_dense_profile_frames": 492936,
        "structural_action": action_summary,
        "audit_split": {"validation_frames": int(split_counts.get("audit_validation", 0)),
                        "test_frames": int(split_counts.get("audit_test", 0)),
                        "identifier_overlap": 0, "block_overlap": 0},
        "existing_anchor_diagnostics": {
            "payload_monotone_branches": int(payload_branch_monotone.sum()),
            "branches": 12,
            "minimum_frame_paired_payload_nonincrease_fraction": min_frame_monotonic,
            "in_distribution_ordering_crossings": crossing_id_count,
        },
        "missing_evidence": ["13 unmeasured planned q points", "dense midpoint interpolation",
                             "0.05-step local discontinuity tests", "q gate and serialization latency",
                             "original dataset files", "CUDA device"],
        "next_experiment": {"branch": "ae64/uint6", "split": "audit_validation",
                            "frames": int(split_counts.get("audit_validation", 0)), "q_points": 19,
                            "profile_frame_evaluations": int(split_counts.get("audit_validation", 0)) * 19},
        "promotion_allowed": False,
    }
    (OUT / "RESULTS_SUMMARY.json").write_text(
        json.dumps(audit.sanitize(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    review = {
        "schema": "scenesense.continuous_roi_control_review_required.v1",
        "terminal": "REVIEW_REQUIRED",
        "analysis_conclusion": "INSUFFICIENT_EVIDENCE",
        "controller_action": "KEEP_MEASURED_DISCRETE_Q_ANCHORS",
        "promotion_allowed": False,
        "registry_changed": False, "runtime_changed": False, "controller_changed": False,
        "carla_run": False, "oai_run": False, "rl_agent_implemented": False,
        "review_questions": [
            "Restore the exact dataset path or approve an equivalently hashed immutable cache.",
            "Provide CUDA capacity for the AE64/uint6 19-q validation pilot.",
            "Review the preregistered interpolation and local-jump tolerances before any promotion run.",
        ],
    }
    (OUT / "REVIEW_REQUIRED.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output_files = sorted(
        path for path in OUT.rglob("*")
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts
    )
    manifest = {
        "schema": "scenesense.continuous_roi_control_audit_manifest.v1",
        "audit_id": "continuous_roi_control_audit_v1/20260820_203411",
        "created_at_unix": time.time(),
        "repository_commit": audit.git_value("rev-parse", "HEAD"),
        "repository_branch": audit.git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "create_only": True,
        "resume_note": "Initial analysis completed; terminal rendering resumed after escaping a Markdown f-string literal. No computed artifact was overwritten.",
        "terminal": "REVIEW_REQUIRED",
        "analysis_conclusion": "INSUFFICIENT_EVIDENCE",
        "inputs": audit.input_entries(),
        "outputs": [
            {"path": str(path.relative_to(OUT)), "size_bytes": path.stat().st_size, "sha256": audit.sha256_file(path)}
            for path in output_files
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"terminal": "REVIEW_REQUIRED", "conclusion": "INSUFFICIENT_EVIDENCE",
                      "output_dir": str(OUT), "outputs": len(output_files)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
