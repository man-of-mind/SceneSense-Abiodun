"""Paired Task-B ablation for observed vulnerable-object shield rules."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .catalog import flatten_actions, load_profile_catalog
from .channel import ChannelSurface
from .config import REPO_ROOT, load_controller_ladder_config
from .controllers import GreedyController
from .ladder import run_deployable_controller
from .replay import discover_trace_registry, registry_frame
from .reporting import new_run_directory, write_run_files
from .run_controller_ladder import (
    _apply_corpus_overrides,
    _load_split,
    _manifest_record,
    _new_env,
    _verify_corpus_contract,
)


VARIANTS = {
    "disabled": None,
    "enabled_conf_0.20": 0.20,
    "enabled_conf_0.30_primary": 0.30,
    "enabled_conf_0.40": 0.40,
}


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in metrics.groupby("guardrail_variant", sort=False):
        vulnerable = group["observed_vulnerable_count"] > 0
        low_confidence = group["observed_low_confidence_vulnerable_count"] > 0
        finite = ~group["matched_true_unobserved_sentinel"].astype(bool)
        rows.append(
            {
                "guardrail_variant": variant,
                "frames": len(group),
                "vulnerable_opportunity_frames": int(vulnerable.sum()),
                "low_confidence_opportunity_frames": int(low_confidence.sum()),
                "mean_matched_reward_finite": float(
                    group.loc[finite, "matched_true_expected_reward"].mean()
                ),
                "mean_offered_mbps": float(group["offered_mbps"].mean()),
                "mean_payload_kib_selected": float(group["payload_kib"].mean()),
                "split_pct": 100.0 * float((group["mode"] == "SPLIT").mean()),
                "shield_feasible_pct": 100.0 * float(group["shield_feasible"].mean()),
                "matched_safe_pct": 100.0 * float(group["selected_matched_true_safe"].mean()),
                "skip_on_observed_vulnerable_count": int(
                    (vulnerable & (group["mode"] == "SKIP")).sum()
                ),
                "roi_drop_on_low_confidence_vulnerable_count": int(
                    (low_confidence & (group["mode"] == "SPLIT") & (group["roi_q"] > 0)).sum()
                ),
                "guardrail_applied_frames": int(group["vulnerable_guardrail_applied"].sum()),
                "guardrail_unachievable_frames": int(
                    group["vulnerable_guardrail_unachievable"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _cluster_ci(frame: pd.DataFrame, column: str, seed: int = 20260814) -> tuple[float, float]:
    by_episode = frame.groupby("episode_id")[column].mean()
    if len(by_episode) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = by_episode.to_numpy(dtype=float)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def _paired_primary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    keys = ["episode_id", "step_index"]
    columns = [
        "matched_true_expected_reward",
        "matched_true_unobserved_sentinel",
        "offered_mbps",
        "payload_kib",
        "action_id",
        "selected_matched_true_safe",
    ]
    off = metrics[metrics["guardrail_variant"] == "disabled"].set_index(keys)[columns]
    on = metrics[
        metrics["guardrail_variant"] == "enabled_conf_0.30_primary"
    ].set_index(keys)[columns]
    if not off.index.equals(on.index):
        raise RuntimeError("paired guardrail variants do not contain identical evaluation ticks")
    paired = pd.DataFrame(index=off.index).reset_index()
    finite = ~(
        off["matched_true_unobserved_sentinel"].astype(bool)
        | on["matched_true_unobserved_sentinel"].astype(bool)
    )
    paired["reward_delta_on_minus_off"] = (
        on["matched_true_expected_reward"] - off["matched_true_expected_reward"]
    ).where(finite).to_numpy()
    paired["offered_mbps_delta_on_minus_off"] = (
        on["offered_mbps"] - off["offered_mbps"]
    ).to_numpy()
    paired["payload_kib_delta_on_minus_off"] = (
        on["payload_kib"] - off["payload_kib"]
    ).to_numpy()
    paired["action_changed"] = (on["action_id"] != off["action_id"]).to_numpy()
    paired["matched_safe_delta"] = (
        on["selected_matched_true_safe"].astype(int)
        - off["selected_matched_true_safe"].astype(int)
    ).to_numpy()
    reward_ci = _cluster_ci(paired.dropna(subset=["reward_delta_on_minus_off"]), "reward_delta_on_minus_off")
    stats = {
        "action_change_pct": 100.0 * float(paired["action_changed"].mean()),
        "mean_reward_delta": float(paired["reward_delta_on_minus_off"].mean()),
        "reward_delta_ci95_low": reward_ci[0],
        "reward_delta_ci95_high": reward_ci[1],
        "mean_offered_mbps_delta": float(paired["offered_mbps_delta_on_minus_off"].mean()),
        "mean_payload_kib_delta": float(paired["payload_kib_delta_on_minus_off"].mean()),
        "matched_safe_pp_delta": 100.0 * float(paired["matched_safe_delta"].mean()),
    }
    return paired, stats


def _save_figure(summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = summary["guardrail_variant"].str.replace("enabled_conf_", "", regex=False)
    axes[0].bar(labels, summary["mean_matched_reward_finite"], color="#4472C4")
    axes[0].set_ylabel("Mean finite matched reward")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(labels, summary["mean_offered_mbps"], color="#ED7D31")
    axes[1].set_ylabel("Mean offered load (Mbps)")
    axes[1].tick_params(axis="x", rotation=20)
    figure.suptitle("Observed-vulnerable-object guardrail cost")
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def run(
    config_path: Path,
    replay_roots: Sequence[Path],
    split_manifest: Path,
    verification_manifest: Path,
) -> Path:
    config = _apply_corpus_overrides(
        load_controller_ladder_config(config_path),
        replay_roots,
        split_manifest,
        verification_manifest,
    )
    _verify_corpus_contract(config)
    profiles = load_profile_catalog(config["actions"]["catalog_csv"])
    actions = flatten_actions(
        profiles, config["actions"]["fps"], config["actions"]["preferred_core_kib"]
    )
    roi_by_action = {action.action_id: action.roi_q for action in actions}
    surface = ChannelSurface(config)
    registry = discover_trace_registry(config)
    evaluation = _load_split(
        config,
        registry,
        config["controller_ladder"]["evaluation_split"],
        config["controller_ladder"]["evaluation_episode_count"],
    )
    seeds = [int(value) for value in config["controller_ladder"]["channel_seeds"]]
    rows = []
    for episode_index, (record, frames) in enumerate(evaluation):
        channel_seed = seeds[episode_index % len(seeds)]
        for variant, threshold in VARIANTS.items():
            variant_config = copy.deepcopy(config)
            guardrail = variant_config["safety"]["vulnerable_object_guardrails"]
            guardrail["enabled"] = threshold is not None
            if threshold is not None:
                guardrail["low_confidence_threshold"] = threshold
            result = run_deployable_controller(
                _new_env(variant_config, frames, actions, surface, channel_seed),
                GreedyController(),
                training=False,
            )
            for row in result.rows:
                row.update(
                    {
                        "guardrail_variant": variant,
                        "guardrail_confidence_threshold": threshold,
                        "scenario": record.run_group,
                        "scenario_family": record.scenario_family,
                        "replay_split": record.split,
                        "roi_q": roi_by_action[row["action_id"]],
                    }
                )
                rows.append(row)
    metrics = pd.DataFrame(rows)
    summary = _summarize(metrics)
    paired, stats = _paired_primary(metrics)
    run_dir = new_run_directory("vulnerable_guardrail")
    registry_frame(registry).to_csv(run_dir / "replay_registry.csv", index=False)
    paired.to_csv(run_dir / "paired_primary_deltas.csv", index=False)
    _save_figure(summary, run_dir / "figures" / "guardrail_cost")
    report = "\n".join(
        [
            "# Task B — observed vulnerable-object guardrail ablation",
            "",
            "The hard rules prevent SKIP whenever an observed pedestrian/cyclist is active and clamp low-confidence vulnerable-object frames to ROI0. They cannot protect detector misses or unrepresented hidden hazards.",
            "",
            "C1 remains dominant: if no C1-admitted action satisfies the vulnerable rule, the least-risk C1 action is used and `vulnerable_guardrail_unachievable` is raised.",
            "",
            "## Primary paired cost (confidence < 0.30)",
            "",
            f"- Action changes: {stats['action_change_pct']:.2f}% of held-out ticks.",
            f"- Finite matched-reward delta (on - off): {stats['mean_reward_delta']:+.6f}, trajectory-cluster 95% CI [{stats['reward_delta_ci95_low']:+.6f}, {stats['reward_delta_ci95_high']:+.6f}].",
            f"- Offered-load delta: {stats['mean_offered_mbps_delta']:+.4f} Mbps; selected-payload delta: {stats['mean_payload_kib_delta']:+.3f} KiB.",
            f"- Matched-safe-rate delta: {stats['matched_safe_pp_delta']:+.3f} percentage points.",
            "",
            "## Threshold sensitivity",
            "",
            "```text",
            summary.to_string(index=False),
            "```",
            "",
            "Cyclists are supported by the rule but absent from the accepted corpus, so cyclist protection is contract-tested rather than empirically costed here.",
            "",
        ]
    )
    report_path = run_dir / "TASK_B_RESULTS.md"
    report_path.write_text(report, encoding="utf-8")
    selected_records = [item for item in evaluation]
    write_run_files(
        run_dir,
        config,
        metrics,
        summary,
        {
            "run_type": "vulnerable_object_guardrail_ablation",
            "implementation_status": "completed_surrogate_guardrail_evaluation",
            "scope": "observed pedestrians/cyclists only",
            "variants": VARIANTS,
            "primary_paired_statistics": stats,
            "selected_replays": [
                _manifest_record(record, frames) for record, frames in selected_records
            ],
            "source_hashes": {
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "shield_sha256": hashlib.sha256(
                    (REPO_ROOT / "rl_agent/policy/shield.py").read_bytes()
                ).hexdigest(),
            },
            "limitations": [
                "guardrail sees only objects emitted by the deployed detector/tracker",
                "accepted replay has pedestrians but no cyclists or hidden-hazard flag",
                "confidence thresholds are engineering sensitivity points, not calibrated risk probabilities",
                "table-driven surrogate evaluation only",
            ],
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("rl_agent/policy/configs/controller_ladder_advisor_rich_v5.yaml"),
    )
    parser.add_argument("--replay-root", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--verification-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        run(args.config, args.replay_root, args.split_manifest, args.verification_manifest)
    )


if __name__ == "__main__":
    main()
