"""CLI for the pre-registered desk-only expanded-action gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from .channel import ChannelSurface
from .config import REPO_ROOT
from .expanded_gate import (
    CONTROLLERS,
    canonical_json,
    compute_feasibility_frontier,
    decide_outcome,
    load_accepted_config,
    load_actions,
    load_frozen_registry,
    load_gate_spec,
    run_expanded_evaluation,
    sha256_file,
    verify_frozen_sources,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "expanded_action_gate_v3.yaml"


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _plot_frontier(frame: pd.DataFrame, output: Path) -> None:
    focus = frame[
        (frame["share_envelope"] == "equal_c1")
        & (frame["target_fps"] == 5)
        & (frame["deadline_s"] == 0.5)
    ].copy()
    payloads = sorted(focus["payload_kib"].unique())
    rungs = ["clear", "mild", "mid", "strong"]
    matrix = []
    labels = []
    for payload in payloads:
        subset = focus[focus["payload_kib"] == payload]
        labels.append(f"{payload:g}")
        matrix.append(
            [
                int(
                    subset[
                        (subset["ue_count"] == 2) & (subset["rung"] == rung)
                    ]["joint_necessary_feasible"].iloc[0]
                )
                for rung in rungs
            ]
        )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(rungs)), rungs)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Measured channel rung")
    ax.set_ylabel("Payload (KiB)")
    ax.set_title("Necessary feasibility: N=2, 5 FPS, 500 ms, equal C1 share")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, "feasible" if value else "no", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, ticks=[0, 1], label="necessary-feasible indicator")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _plot_rewards(summary: pd.DataFrame, output: Path) -> None:
    grouped = summary.groupby(["group_id", "controller"], as_index=False)["mean_reward_v5"].mean()
    pivot = grouped.pivot(index="group_id", columns="controller", values="mean_reward_v5")
    pivot = pivot.loc[sorted(pivot.index)]
    positions = list(range(len(pivot)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(
        [value - width / 2 for value in positions],
        pivot[CONTROLLERS[0]],
        width,
        label="decentralized greedy",
        color="#4C78A8",
    )
    ax.bar(
        [value + width / 2 for value in positions],
        pivot[CONTROLLERS[1]],
        width,
        label="joint true-state one-step oracle",
        color="#F2CF5B",
    )
    ax.set_xticks(positions, pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Mean matched-truth reward v5")
    ax.set_title("Expanded SPLIT+SKIP gate (three paired channel seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def _write_report(
    output_dir: Path,
    decision: Dict[str, object],
    frontier: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    equal = frontier[frontier["share_envelope"] == "equal_c1"]
    count = int(equal["joint_necessary_feasible"].sum())
    total = len(equal)
    by_payload = (
        equal.groupby("payload_kib")["joint_necessary_feasible"]
        .agg(["sum", "count"])
        .reset_index()
    )
    payload_lines = "\n".join(
        f"- {row.payload_kib:g} KiB: {int(row['sum'])}/{int(row['count'])} necessary-feasible cells"
        for _, row in by_payload.iterrows()
    )
    text = f"""# Expanded action gate results ({decision.get('gate_schema', 'schema unavailable')})

Verdict: **`{decision['verdict']}`**.

This was a desk-only run over immutable accepted reward-v5 replay. It launched no OAI or CARLA and includes
neither LOCAL, MPC, nor RL. The oracle is a joint true-state **one-step** upper bound. The replay comparison is
queue-free and must not be presented as a shared-queue or real-radio result.

## Registered reward gate

- Expanded decentralized greedy: {decision['group_equal_greedy_mean_reward_v5']:.6f}
- Expanded joint oracle: {decision['group_equal_oracle_mean_reward_v5']:.6f}
- Absolute lift: {decision['absolute_reward_lift']:.6f}
- Relative lift: {100.0 * decision['relative_reward_lift']:.3f}%
- Group-cluster bootstrap 95% interval: [{decision['cluster_bootstrap_95ci'][0]:.6f}, {decision['cluster_bootstrap_95ci'][1]:.6f}]
- Lift by UE count: {decision['mean_reward_lift_by_ue_count']}
- Minimum paired worst-UE lift: {decision['minimum_paired_worst_ue_reward_lift']:.6f}
- Maximum decentralized aggregate true-C1 miss fraction: {100.0 * decision['maximum_greedy_aggregate_c1_miss_fraction']:.3f}%

Frozen checks: `{json.dumps(decision['checks'], sort_keys=True)}`.

## Deadline-feasibility frontier

Across the equal-C1-share rows, {count}/{total} payload × N × rung × deadline × FPS cells meet both the
on-wire rate condition and the queue-free p95 necessary condition. Feasible does **not** mean queue-sufficient.

{payload_lines}

Detailed per-cell, per-frame, and per-group outputs are in `feasibility_frontier.csv`,
`per_frame_metrics.csv`, and `group_seed_summary.csv`. Source/code hashes are in `manifest.json`.
"""
    (output_dir / "EXPANDED_ACTION_GATE_RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--detached-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    spec = load_gate_spec(args.config)
    source_hashes_before = verify_frozen_sources(spec)
    config = load_accepted_config(spec)
    actions = load_actions(config, spec)
    registry = load_frozen_registry(spec)
    selected = {
        str(episode_id)
        for group in spec["evaluation"]["groups"]
        for episode_id in group["episodes"]
    }
    if not selected.issubset(registry):
        raise ValueError("one or more frozen evaluation episodes are absent")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "config_sha256": spec["_meta"]["config_sha256"],
                    "actions": len(actions),
                    "selected_episodes": len(selected),
                    "verified_sources": len(source_hashes_before),
                },
                sort_keys=True,
            )
        )
        return

    if args.detach:
        if args.detached_child:
            raise ValueError("--detach and --detached-child are mutually exclusive")
        if args.output_dir:
            detached_output = Path(args.output_dir).resolve()
        else:
            stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S_pdt")
            detached_output = (
                REPO_ROOT / "rl_agent" / "policy" / "experiments" / "expanded_action_gate" / stamp
            )
        if detached_output.exists():
            raise FileExistsError(f"refusing to mutate existing output directory: {detached_output}")
        detached_output.mkdir(parents=True)
        log_path = detached_output / "run.log"
        command = [
            sys.executable,
            "-m",
            "rl_agent.policy.run_expanded_action_gate",
            "--config",
            str(args.config),
            "--output-dir",
            str(detached_output),
            "--detached-child",
        ]
        with log_path.open("ab", buffering=0) as log_stream:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched = {
            "schema": "scenesense.policy.expanded_action_gate.launched.v1",
            "status": "RUNNING",
            "pid": process.pid,
            "command": command,
            "completion_sentinel": "COMPLETED.json",
            "failure_sentinel": "FAILED.json",
            "progress_log": "progress.jsonl",
        }
        (detached_output / "LAUNCHED.json").write_text(
            json.dumps(launched, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "RUNNING", "output_dir": _relative(detached_output), "pid": process.pid}))
        return

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        stamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S_pdt")
        output_dir = REPO_ROOT / "rl_agent" / "policy" / "experiments" / "expanded_action_gate" / stamp
    if output_dir.exists():
        allowed = {"LAUNCHED.json", "run.log"}
        present = {path.name for path in output_dir.iterdir()}
        if not args.detached_child or not present.issubset(allowed):
            raise FileExistsError(f"refusing to mutate existing output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    (output_dir / "figures").mkdir()
    (output_dir / "resolved_gate_config.yaml").write_text(
        yaml.safe_dump({key: value for key, value in spec.items() if key != "_meta"}, sort_keys=False),
        encoding="utf-8",
    )

    progress_path = output_dir / "progress.jsonl"

    def progress(event: Dict[str, object]) -> None:
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    progress({"event": "started", "status": "RUNNING"})
    try:
        surface = ChannelSurface(config)
        frontier = compute_feasibility_frontier(config, spec)
        progress({"event": "frontier_complete", "rows": len(frontier)})
        per_frame, summary = run_expanded_evaluation(
            config, spec, actions, surface, registry, progress_callback=progress
        )
        decision = decide_outcome(summary, spec)
    except Exception as error:
        failure = {
            "schema": "scenesense.policy.expanded_action_gate.failed.v1",
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (output_dir / "FAILED.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        progress({"event": "failed", "status": "FAILED", "error": str(error)})
        raise
    frontier.to_csv(output_dir / "feasibility_frontier.csv", index=False)
    per_frame.to_csv(output_dir / "per_frame_metrics.csv", index=False)
    summary.to_csv(output_dir / "group_seed_summary.csv", index=False)
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_frontier(frontier, output_dir / "figures" / "feasibility_frontier_n2")
    _plot_rewards(summary, output_dir / "figures" / "expanded_reward_comparison")
    _write_report(output_dir, decision, frontier, summary)

    source_hashes_after = verify_frozen_sources(spec)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("frozen source hashes changed during the desk-only run")
    artifact_paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name not in {"manifest.json", "COMPLETED.json"}
    )
    config_input_path = Path(args.config)
    if not config_input_path.is_absolute():
        config_input_path = REPO_ROOT / config_input_path
    spec_document = {
        "scenesense.policy.expanded_action_gate.v1": "EXPANDED_ACTION_GATE_SPEC.md",
        "scenesense.policy.expanded_action_gate.v2": "EXPANDED_ACTION_GATE_V2_SPEC.md",
        "scenesense.policy.expanded_action_gate.v3": "EXPANDED_ACTION_GATE_V3_SPEC.md",
    }[spec["schema"]]
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("expanded_gate.py"),
        config_input_path,
        REPO_ROOT / "rl_agent" / "policy" / spec_document,
    ]
    manifest = {
        "schema": "scenesense.policy.expanded_action_gate.manifest.v1",
        "implementation_status": "completed_desk_only_expanded_surrogate_gate",
        "output_dir": _relative(output_dir),
        "verdict": decision["verdict"],
        "authorization": spec["authorization"],
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "code_hashes": {_relative(path): sha256_file(path) for path in code_paths},
        "artifact_hashes": {_relative(path): sha256_file(path) for path in artifact_paths},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completed = {
        "schema": "scenesense.policy.expanded_action_gate.completed.v1",
        "status": "COMPLETED",
        "verdict": decision["verdict"],
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "decision_sha256": sha256_file(output_dir / "decision.json"),
    }
    (output_dir / "COMPLETED.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress({"event": "completed", "status": "COMPLETED", "verdict": decision["verdict"]})
    print(canonical_json({"status": "COMPLETED", "output_dir": _relative(output_dir), **decision}))


if __name__ == "__main__":
    main()
