"""Structured output, summaries, manifests, and figures for Track A runs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .config import REPO_ROOT, public_config


def new_run_directory(kind: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPO_ROOT / "rl_agent" / "policy" / "experiments" / kind / stamp
    path.mkdir(parents=True, exist_ok=False)
    (path / "figures").mkdir()
    return path


def summarize_frames(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["scenario", "controller"] if "scenario" in frame.columns else ["episode_id", "controller"]
    for group_values, group in frame.groupby(keys, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(keys, group_values))
        attempts = group["actual_delivery"].notna()
        deliveries = group.loc[attempts, "actual_delivery"].astype("boolean").fillna(False).astype(bool)
        row.update(
            {
                "frames": int(len(group)),
                "split_pct": 100.0 * float((group["mode"] == "SPLIT").mean()),
                "skip_pct": 100.0 * float((group["mode"] == "SKIP").mean()),
                "degraded_tier_pct": 100.0 * float(group["degraded_tier_used"].mean()),
                "over_budget_pct": 100.0 * float(group["shield_over_budget"].mean()),
                "shield_ood_pct": 100.0 * float(group["shield_ood"].mean()),
                "selected_true_safe_pct": 100.0 * float(group["selected_true_safe"].mean()),
                "selected_matched_true_safe_pct": 100.0
                * float(group["selected_matched_true_safe"].mean()),
                "false_admit_selected_pct": 100.0 * float(group["false_admit_selected"].mean()),
                "false_admit_selected_matched_pct": 100.0
                * float(group["false_admit_selected_matched"].mean()),
                "false_reject_frame_pct": 100.0 * float(group["false_reject_frame"].mean()),
                "mean_bound_m": float(group["shield_bound_m"].replace([np.inf, -np.inf], np.nan).mean()),
                "p95_true_risk_m": float(group["true_risk_p95_m"].replace([np.inf, -np.inf], np.nan).quantile(0.95)),
                "p95_matched_true_risk_m": float(
                    group["matched_true_risk_p95_m"].replace([np.inf, -np.inf], np.nan).quantile(0.95)
                ),
                "mean_reward": float(group["shield_expected_reward"].mean()),
                "mean_prb_cost": float(group["shield_prb_cost"].mean()),
                "mean_oracle_reward_gap_safe_only": float(group["oracle_reward_gap_safe_only"].mean()),
                "oracle_action_set_mismatch_pct": 100.0 * float(group["oracle_action_set_mismatch"].mean()),
                "attempts": int(attempts.sum()),
                "capture_attempt_pct": 100.0 * float(attempts.mean()),
                "delivery_pct_attempted": 100.0 * float(deliveries.mean()) if len(deliveries) else np.nan,
                "c1_estimate_miss_count": int(group["c1_estimate_miss"].sum()),
                "c1_estimate_miss_pct_attempted": (
                    100.0 * float(group.loc[attempts, "c1_estimate_miss"].mean()) if attempts.any() else np.nan
                ),
                "truth_objects": int(group["truth_object_count"].sum()),
                "observed_objects": int(group["observed_object_count"].sum()),
                "unobserved_gt_objects": int(group["unobserved_gt_object_count"].sum()),
            }
        )
        row["observation_coverage_pct"] = (
            100.0 * row["observed_objects"] / row["truth_objects"] if row["truth_objects"] else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_mode_and_risk_figure(frame: pd.DataFrame, output_prefix: Path, title: str) -> None:
    grouped = frame.groupby("controller")
    controllers = list(grouped.groups)
    split = [100.0 * float((grouped.get_group(name)["mode"] == "SPLIT").mean()) for name in controllers]
    skip = [100.0 - value for value in split]
    # Plot the tracked-object C2 population that the shield can actually observe.
    # End-to-end exposure (including unobserved ground-truth objects) remains in
    # the CSV/report as a separate perception-coverage diagnostic.
    risk_column = "matched_true_risk_p95_m" if "matched_true_risk_p95_m" in frame.columns else "true_risk_p95_m"
    risk = [float(grouped.get_group(name)[risk_column].quantile(0.95)) for name in controllers]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(controllers, split, label="SPLIT", color="#4472C4")
    axes[0].bar(controllers, skip, bottom=split, label="SKIP", color="#A5A5A5")
    axes[0].set_ylabel("Selected frames (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].set_title("Mode mix")
    axes[1].bar(controllers, risk, color="#ED7D31")
    axes[1].axhline(2.0, color="#333333", linestyle="--", linewidth=1.2, label="epsilon = 2 m")
    axes[1].set_ylabel("Tracked true-risk p95 (m)")
    axes[1].set_title("Tail localization risk")
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_prefix.with_suffix(".png"), dpi=300)
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)


def write_run_files(
    run_dir: Path,
    config: Mapping[str, object],
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    manifest_extra: Mapping[str, object],
) -> None:
    metrics_path = run_dir / "per_frame_metrics.csv"
    summary_path = run_dir / "summary.csv"
    config_path = run_dir / "resolved_config.yaml"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(yaml.safe_dump(public_config(dict(config)), sort_keys=False), encoding="utf-8")
    files = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(run_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir.relative_to(REPO_ROOT)),
        "files": files,
        **dict(manifest_extra),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
