"""Run the four deterministic Track A acceptance episodes."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from .catalog import flatten_actions, load_profile_catalog
from .channel import ChannelProcess, ChannelSurface
from .config import load_config
from .env import SurrogateEnv
from .oracles import run_oracle
from .replay import synthetic_episode
from .reporting import new_run_directory, save_mode_and_risk_figure, summarize_frames, write_run_files


def _scenario_specs():
    return {
        "empty_clear": {
            "frames": synthetic_episode("empty_clear", [], 120),
            "rungs": ["clear"] * 120,
            "capacity_multiplier": 1.0,
        },
        "slow_clear": {
            "frames": synthetic_episode("slow_clear", [2.0], 240),
            "rungs": ["clear"] * 240,
            "capacity_multiplier": 1.0,
        },
        "fast_strong": {
            "frames": synthetic_episode("fast_strong", [14.3], 240),
            "rungs": ["strong"] * 240,
            "capacity_multiplier": 1.0,
        },
        "clear_to_fade": {
            "frames": synthetic_episode("clear_to_fade", [8.0], 240),
            "rungs": ["clear"] * 80 + ["mild"] * 40 + ["mid"] * 40 + ["strong"] * 80,
            "capacity_multiplier": 1.0,
        },
    }


def run(config_path: Path | None = None) -> Path:
    config = load_config(config_path)
    profiles = load_profile_catalog(config["actions"]["catalog_csv"])
    actions = flatten_actions(profiles, config["actions"]["fps"], config["actions"]["preferred_core_kib"])
    surface = ChannelSurface(config)
    run_dir = new_run_directory("deterministic_acceptance")
    rows = []
    for scenario_index, (scenario, spec) in enumerate(_scenario_specs().items()):
        for controller_index, controller in enumerate(("shielded", "clairvoyant")):
            seed = int(config["seed"]) + scenario_index * 100 + controller_index
            channel = ChannelProcess(
                config,
                surface,
                seed,
                fixed_rungs=spec["rungs"],
                fixed_capacity_multiplier=spec["capacity_multiplier"],
            )
            env = SurrogateEnv(
                config,
                spec["frames"],
                actions,
                channel,
                surface,
                seed + 10_000,
                latency_mode="p50",
            )
            result = run_oracle(env, controller)
            for row in result.rows:
                row["scenario"] = scenario
                rows.append(row)
    metrics = pd.DataFrame(rows)
    summary = summarize_frames(metrics)
    empty_shielded = metrics[(metrics.scenario == "empty_clear") & (metrics.controller == "shielded")]
    slow_shielded = metrics[(metrics.scenario == "slow_clear") & (metrics.controller == "shielded")]
    fast_shielded = metrics[(metrics.scenario == "fast_strong") & (metrics.controller == "shielded")]
    fade = metrics[metrics.scenario == "clear_to_fade"]
    checks = {
        "empty_scene_all_skip": bool((empty_shielded["mode"] == "SKIP").all()),
        "slow_scene_attempts_updates": bool(slow_shielded["actual_delivery"].notna().any()),
        "fast_strong_flags_infeasibility": bool(fast_shielded["shield_over_budget"].any()),
        "fade_visits_all_four_rungs": int(fade["channel_rung_true"].nunique()) == 4,
        "no_ood_in_supported_scenarios": not bool(metrics["shield_ood"].any()),
        "safe_only_clairvoyant_gap_nonnegative": bool(
            (metrics["oracle_reward_gap_safe_only"].dropna() >= -1e-9).all()
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"deterministic acceptance failed: {failed}")
    figure_prefix = run_dir / "figures" / "acceptance_mode_and_risk"
    save_mode_and_risk_figure(metrics, figure_prefix, "Track A deterministic acceptance")
    report = [
        "# Track A deterministic acceptance",
        "",
        "All structural acceptance checks passed.",
        "",
        "## Checks",
        "",
    ]
    report.extend(f"- `{name}`: **PASS**" for name in checks)
    report.extend(["", "## Summary", "", summary.to_markdown(index=False), ""])
    (run_dir / "ACCEPTANCE_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    write_run_files(
        run_dir,
        config,
        metrics,
        summary,
        {
            "run_type": "deterministic_acceptance",
            "controllers": ["shielded", "clairvoyant"],
            "checks": checks,
            "source_hashes": surface.source_hashes,
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
