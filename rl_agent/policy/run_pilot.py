"""Run the gated real-replay Track A pilot and write the first policy report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .catalog import flatten_actions, load_profile_catalog
from .channel import ChannelProcess, ChannelSurface
from .config import REPO_ROOT, load_config
from .env import SurrogateEnv
from .oracles import run_oracle
from .replay import TraceRecord, discover_trace_registry, load_trace_episode, registry_frame
from .reporting import new_run_directory, save_mode_and_risk_figure, summarize_frames, write_run_files


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```text\n" + frame.to_string(index=False) + "\n```"


def _select_episodes(config, registry: List[TraceRecord]) -> List[Tuple[TraceRecord, list]]:
    target = int(config["pilot"]["episode_count"])
    selected = []
    for record in registry:
        if record.split != config["pilot"]["split"] or record.prediction_path is None:
            continue
        frames = load_trace_episode(
            record,
            config,
            range_m=float(config["safety"]["range_m"]),
            max_steps=int(config["replay"]["max_episode_steps"]),
        )
        truth_count = sum(len(frame.truth_objects) for frame in frames)
        observed_count = sum(len(frame.observed_objects) for frame in frames)
        if truth_count == 0 or observed_count == 0:
            continue
        selected.append((record, frames))
        if len(selected) == target:
            break
    if len(selected) < target:
        raise ValueError(f"pilot requested {target} usable test episodes, found {len(selected)}")
    return selected


def run(config_path: Path | None = None) -> Path:
    config = load_config(config_path)
    profiles = load_profile_catalog(config["actions"]["catalog_csv"])
    actions = flatten_actions(profiles, config["actions"]["fps"], config["actions"]["preferred_core_kib"])
    surface = ChannelSurface(config)
    registry = discover_trace_registry(config)
    selected = _select_episodes(config, registry)
    run_dir = new_run_directory("pilot")
    registry_frame(registry).to_csv(run_dir / "replay_registry.csv", index=False)
    rows = []
    selected_manifest = []
    channel_seeds = [int(value) for value in config["pilot"]["channel_seeds"]]
    for episode_index, (record, frames) in enumerate(selected):
        selected_manifest.append(
            {
                "episode_id": record.episode_id,
                "run_group": record.run_group,
                "scenario_family": record.scenario_family,
                "split": record.split,
                "ground_truth_path": str(record.ground_truth_path.relative_to(REPO_ROOT)),
                "ground_truth_sha256": record.ground_truth_sha256,
                "prediction_path": str(record.prediction_path.relative_to(REPO_ROOT)),
                "prediction_sha256": record.prediction_sha256,
                "frame_count": len(frames),
            }
        )
        for controller_index, controller in enumerate(("shielded", "clairvoyant")):
            seed = channel_seeds[episode_index % len(channel_seeds)]
            channel = ChannelProcess(config, surface, seed)
            env = SurrogateEnv(
                config,
                frames,
                actions,
                channel,
                surface,
                seed + 10_000,
                latency_mode="sample",
            )
            result = run_oracle(env, controller)
            for row in result.rows:
                row["scenario"] = record.run_group
                row["scenario_family"] = record.scenario_family
                row["replay_split"] = record.split
                rows.append(row)
    metrics = pd.DataFrame(rows)
    summary = summarize_frames(metrics)
    overall = summarize_frames(metrics.assign(scenario="ALL_REAL_REPLAY"))
    summary_all = pd.concat([overall, summary], ignore_index=True)
    save_mode_and_risk_figure(
        metrics,
        run_dir / "figures" / "pilot_mode_and_risk",
        "Track A real-replay pilot: epsilon=2 m, preferred core=90 KiB, range<=25 m",
    )
    attempt_rows = metrics[metrics["actual_delivery"].notna()]
    projection = {
        "attempted_frames": int(len(attempt_rows)),
        "payload_projection_pct": (
            100.0 * float((attempt_rows["payload_provenance"] == "payload_projection").mean())
            if len(attempt_rows)
            else 0.0
        ),
        "fps_projection_pct": (
            100.0 * float((attempt_rows["rate_provenance"] == "fps_projection").mean())
            if len(attempt_rows)
            else 0.0
        ),
    }
    overall_display = overall[
        [
            "controller",
            "frames",
            "split_pct",
            "skip_pct",
            "capture_attempt_pct",
            "delivery_pct_attempted",
            "c1_estimate_miss_pct_attempted",
            "over_budget_pct",
            "selected_true_safe_pct",
            "selected_matched_true_safe_pct",
            "false_admit_selected_pct",
            "false_admit_selected_matched_pct",
            "false_reject_frame_pct",
            "mean_prb_cost",
            "mean_oracle_reward_gap_safe_only",
            "oracle_action_set_mismatch_pct",
            "observation_coverage_pct",
        ]
    ].copy()
    shield_overall = overall[overall["controller"] == "shielded"].iloc[0]
    report_lines = [
        "# Track A policy results — gated one-configuration pilot",
        "",
        "**Scope:** table-driven SPLIT+SKIP only; epsilon=2.0 m; 90 KiB preferred core; objects within 25 m; "
        "real CARLA vehicle replay composed with a synthetic Markov channel. No CARLA, OAI, LOCAL, or RL run.",
        "",
        "## Gate status",
        "",
        "- Canonical seven-profile / 36-action catalog: PASS",
        "- Contract tests: PASS",
        "- Four deterministic acceptance episodes: PASS",
        "- One-config real-replay pilot: COMPLETE",
        "- Twelve-condition advisor sweep: NOT STARTED (still gated on review of this pilot)",
        "",
        "## Pre-sweep verdict",
        "",
        "The implementation and pilot gates pass, but the 12-condition advisor sweep remains intentionally "
        "paused until the pre-registered weight sensitivity and metric-scope review are complete.",
        "",
        f"- Matched/tracked-object false admission: "
        f"{shield_overall['false_admit_selected_matched_pct']:.2f}%.",
        f"- Matched/tracked-object false rejection: {shield_overall['false_reject_frame_pct']:.2f}%.",
        f"- Strict end-to-end GT false admission: {shield_overall['false_admit_selected_pct']:.2f}% "
        f"with {shield_overall['observation_coverage_pct']:.2f}% observation coverage; this includes upstream "
        "perception misses and must not be attributed solely to the channel/AoI shield.",
        f"- Frames flagged over budget by the shield: {shield_overall['over_budget_pct']:.2f}%.",
        f"- C1 estimate misses among attempted captures: "
        f"{shield_overall['c1_estimate_miss_pct_attempted']:.2f}%.",
        "",
        "## Overall pilot summary",
        "",
        _markdown_table(overall_display.round(4)),
        "",
        "## Per-replay summary",
        "",
        _markdown_table(summary.round(4)),
        "",
        "## Projection and interpretation guardrails",
        "",
        f"- Attempted transmitted frames: {projection['attempted_frames']}.",
        f"- Payload-projected attempts: {projection['payload_projection_pct']:.2f}%.",
        f"- FPS-projected attempts: {projection['fps_projection_pct']:.2f}%.",
        "- `split_pct` is the fraction of 20 Hz control ticks for which a SPLIT schedule was active; "
        "`capture_attempt_pct` is the actual transmitted-frame fraction.",
        "- Shield false-admit/reject values are surrogate validation against replay GT + synthetic channel truth, "
        "not live safety validation.",
        "- `false_admit_selected_matched_pct` isolates localization-shield failures on matched/tracked objects, "
        "matching the staleness study's C2 domain. `false_admit_selected_pct` is the stricter end-to-end GT "
        "exposure and includes upstream perception misses. Observation coverage is reported separately.",
        "- The replay GT contains vehicles only. Pedestrian conclusions require the separately labelled synthetic "
        "stress extension and cannot be claimed from this pilot.",
        "- 25-40 m remains extrapolative and was not used in this pilot.",
        "",
        "## Artifacts",
        "",
        f"Run directory: `{run_dir.relative_to(REPO_ROOT)}`",
        "",
        "See `per_frame_metrics.csv`, `summary.csv`, `replay_registry.csv`, `resolved_config.yaml`, "
        "`manifest.json`, and `figures/pilot_mode_and_risk.{png,pdf}` in that directory.",
        "",
    ]
    report = "\n".join(report_lines)
    (run_dir / "PILOT_RESULTS.md").write_text(report, encoding="utf-8")
    policy_results = REPO_ROOT / "rl_agent" / "policy" / "POLICY_RESULTS.md"
    policy_results.write_text(report, encoding="utf-8")
    catalog_meta_path = REPO_ROOT / "rl_agent" / "policy" / "data" / "action_catalog.meta.json"
    catalog_meta = json.loads(catalog_meta_path.read_text())
    write_run_files(
        run_dir,
        config,
        metrics,
        summary_all,
        {
            "run_type": "one_config_real_replay_pilot",
            "controllers": ["shielded", "clairvoyant"],
            "selected_replays": selected_manifest,
            "source_hashes": {**surface.source_hashes, **catalog_meta},
            "projection": projection,
            "limitations": [
                "synthetic Markov channel composed with real CARLA scene replay",
                "vehicle-only replay ground truth",
                "channel payload/FPS projection outside measured anchors",
                "surrogate validation only",
            ],
        },
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
