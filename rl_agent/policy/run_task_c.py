"""Task C: static 36-profile RDO audit plus held-out surrogate baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .catalog import RETAINED_PROFILES
from .config import REPO_ROOT
from .rdo import lagrangian_dual_bound, supported_upper_hull
from .run_controller_ladder import run as run_controller_ladder


PROFILE_RE = re.compile(r"^(noae|ae32|ae64|ae128)__uint(4|6|8)__roi(0\.0|0\.3|0\.5)$")
MATRIX = REPO_ROOT / "rl_agent" / "PERMODEL_KNOB_MATRIX_ZSTD.md"
EVAL_ROOT = REPO_ROOT / "experiments" / "ae_integrated_20260710" / "sweeps_permodel_zstd"


def _parse_float(value: str) -> float:
    return float(value.strip().replace("~", ""))


def load_measured_profiles(reward: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 17 or PROFILE_RE.match(cells[0]) is None:
            continue
        metrics_path = EVAL_ROOT / cells[0] / "metrics" / "test_fusion_evaluation_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "profile_id": cells[0],
                "payload_kib": _parse_float(cells[5]),
                "miou": _parse_float(cells[7]),
                "pedestrian_recall": _parse_float(cells[9]),
                "vehicle_recall": float(metrics["learned_vehicle_object_recall"]),
                "metrics_source": str(metrics_path.relative_to(REPO_ROOT)),
                "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["payload_kib", "profile_id"]).reset_index(drop=True)
    if len(frame) != 36 or frame["profile_id"].nunique() != 36:
        raise ValueError(f"expected exactly 36 measured profiles, found {len(frame)}")
    weights = reward["task_metric_weights"]
    references = reward["task_metric_references"]
    frame["utility_v5"] = sum(
        float(weights[name]) * frame[name] / float(references[name])
        for name in ("miou", "pedestrian_recall", "vehicle_recall")
    )
    return frame


def static_rdo_analysis(profiles: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    points = [
        (str(row.profile_id), float(row.payload_kib), float(row.utility_v5))
        for row in profiles.itertuples(index=False)
    ]
    supported = supported_upper_hull(points)
    supported_set = set(supported)
    rows = []
    for budget in sorted(profiles["payload_kib"].unique()):
        feasible = [item for item in points if item[1] <= budget + 1e-12]
        exact = max(feasible, key=lambda item: (item[2], -item[1], item[0]))
        supported_feasible = [item for item in feasible if item[0] in supported_set]
        lookup = max(supported_feasible, key=lambda item: (item[2], -item[1], item[0]))
        dual_bound, multiplier = lagrangian_dual_bound(points, float(budget))
        rows.append(
            {
                "budget_kib": budget,
                "feasible_profile_count": len(feasible),
                "exact_profile_id": exact[0],
                "exact_utility_v5": exact[2],
                "exact_profile_in_retained7": exact[0] in RETAINED_PROFILES,
                "lambda_rdo_profile_id": lookup[0],
                "lambda_rdo_utility_v5": lookup[2],
                "action_agreement": exact[0] == lookup[0],
                "utility_gap_exact_minus_lambda": exact[2] - lookup[2],
                "dual_bound": dual_bound,
                "dual_minimizing_lambda_per_kib": multiplier,
                "lagrangian_duality_gap": max(0.0, dual_bound - exact[2]),
            }
        )
    return pd.DataFrame(rows), supported


def _cluster_ci(frame: pd.DataFrame, column: str, seed: int) -> tuple[float, float]:
    values = frame.groupby("episode_id")[column].mean().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def runtime_comparison(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["episode_id", "step_index"]
    reference = metrics[metrics["controller"] == "budgeted_enumerator"].set_index(keys)
    rows = []
    paired_frames = []
    for controller in ("lambda_rdo", "aoi_index"):
        candidate = metrics[metrics["controller"] == controller].set_index(keys)
        if not reference.index.equals(candidate.index):
            raise RuntimeError(f"{controller} does not share the exact enumerator's replay ticks")
        paired = pd.DataFrame(index=reference.index).reset_index()
        paired["controller"] = controller
        paired["cross_trajectory_action_agreement"] = (
            reference["action_id"] == candidate["action_id"]
        ).to_numpy()
        finite = ~(
            reference["matched_true_unobserved_sentinel"].astype(bool)
            | candidate["matched_true_unobserved_sentinel"].astype(bool)
        )
        paired["matched_reward_delta_vs_exact"] = (
            candidate["matched_true_expected_reward"]
            - reference["matched_true_expected_reward"]
        ).where(finite).to_numpy()
        paired_frames.append(paired)
        ci = _cluster_ci(
            paired.dropna(subset=["matched_reward_delta_vs_exact"]),
            "matched_reward_delta_vs_exact",
            20260814 + len(rows),
        )
        own_state_agreement = 100.0 * float(
            (~candidate["selection_changed_from_greedy"].astype(bool)).mean()
        )
        row = {
            "controller": controller,
            "frames": len(candidate),
            "own_state_exact_action_agreement_pct": own_state_agreement,
            "cross_trajectory_action_agreement_pct": 100.0
            * float(paired["cross_trajectory_action_agreement"].mean()),
            "mean_matched_reward_delta_vs_exact": float(
                paired["matched_reward_delta_vs_exact"].mean()
            ),
            "reward_delta_ci95_low": ci[0],
            "reward_delta_ci95_high": ci[1],
            "matched_safe_pct": 100.0 * float(candidate["selected_matched_true_safe"].mean()),
            "mean_offered_mbps": float(candidate["offered_mbps"].mean()),
        }
        if controller == "lambda_rdo":
            row["mean_own_state_predicted_reward_gap"] = float(
                candidate["lambda_rdo_full_enumerator_reward_gap"].mean()
            )
            row["max_own_state_predicted_reward_gap"] = float(
                candidate["lambda_rdo_full_enumerator_reward_gap"].max()
            )
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(paired_frames, ignore_index=True)


def _save_figure(profiles: pd.DataFrame, static: pd.DataFrame, runtime: pd.DataFrame, path: Path) -> None:
    supported = set(static["lambda_rdo_profile_id"])
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    colors = ["#4472C4" if value in supported else "#A5A5A5" for value in profiles["profile_id"]]
    axes[0].scatter(profiles["payload_kib"], profiles["utility_v5"], c=colors, s=30)
    hull = profiles[profiles["profile_id"].isin(supported)].sort_values("payload_kib")
    axes[0].plot(hull["payload_kib"], hull["utility_v5"], color="#4472C4", linewidth=1.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Measured payload (KiB/frame, log scale)")
    axes[0].set_ylabel("Reward-v5 static task utility")
    axes[0].set_title("36-profile supported hull")
    axes[1].bar(runtime["controller"], runtime["mean_matched_reward_delta_vs_exact"], color=["#ED7D31", "#70AD47"])
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Matched reward delta vs exact enumerator")
    axes[1].set_title("Held-out sequential replay")
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
    ladder_dir = run_controller_ladder(
        config_path,
        replay_roots=replay_roots,
        split_manifest=split_manifest,
        verification_manifest=verification_manifest,
    )
    resolved = yaml.safe_load((ladder_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    profiles = load_measured_profiles(resolved["reward"])
    static, supported = static_rdo_analysis(profiles)
    metrics = pd.read_csv(ladder_dir / "per_frame_metrics.csv")
    runtime, paired = runtime_comparison(metrics)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "rl_agent" / "policy" / "experiments" / "task_c" / stamp
    (run_dir / "figures").mkdir(parents=True)
    profiles.to_csv(run_dir / "measured_profiles_36.csv", index=False)
    static.to_csv(run_dir / "static_budget_rdo.csv", index=False)
    runtime.to_csv(run_dir / "runtime_baseline_comparison.csv", index=False)
    paired.to_csv(run_dir / "runtime_paired_deltas.csv", index=False)
    _save_figure(profiles, static, runtime, run_dir / "figures" / "task_c_rdo_and_runtime")

    agreement = 100.0 * float(static["action_agreement"].mean())
    positive_gap = int((static["utility_gap_exact_minus_lambda"] > 1e-12).sum())
    retained = 100.0 * float(static["exact_profile_in_retained7"].mean())
    lambda_row = runtime.set_index("controller").loc["lambda_rdo"]
    aoi_row = runtime.set_index("controller").loc["aoi_index"]
    report = "\n".join(
        [
            "# Task C — measured-table enumerator, lambda-RDO, and freshness heuristic",
            "",
            "This separates two questions that must not be conflated: the static 36-profile scalar rate-distortion problem, and the stateful retained-catalog SPLIT+SKIP surrogate.",
            "",
            "## Static 36-profile H2 test",
            "",
            f"The lambda sweep supports {len(supported)}/36 profiles: `{', '.join(supported)}`.",
            f"Across all {len(static)} measured payload breakpoints, supported-hull lookup agrees with exact budgeted enumeration on {agreement:.2f}% and loses utility at {positive_gap} breakpoints.",
            f"Mean/max exact-minus-lambda utility gap: {static['utility_gap_exact_minus_lambda'].mean():.6f} / {static['utility_gap_exact_minus_lambda'].max():.6f}.",
            f"Mean/max Lagrangian duality gap: {static['lagrangian_duality_gap'].mean():.6f} / {static['lagrangian_duality_gap'].max():.6f}.",
            f"The exact static winner belongs to the prior retained-seven catalog at {retained:.2f}% of breakpoints.",
            "",
            "## Held-out retained-catalog ladder",
            "",
            f"Lambda-RDO own-state agreement with full enumeration is {lambda_row['own_state_exact_action_agreement_pct']:.2f}%; its mean own-state predicted reward gap is {lambda_row['mean_own_state_predicted_reward_gap']:.6f}.",
            f"Its independent rollout matched-reward delta is {lambda_row['mean_matched_reward_delta_vs_exact']:+.6f}, trajectory-cluster CI [{lambda_row['reward_delta_ci95_low']:+.6f}, {lambda_row['reward_delta_ci95_high']:+.6f}].",
            f"The AoI-index-inspired heuristic (not Whittle) has {aoi_row['own_state_exact_action_agreement_pct']:.2f}% own-state agreement and matched-reward delta {aoi_row['mean_matched_reward_delta_vs_exact']:+.6f}, CI [{aoi_row['reward_delta_ci95_low']:+.6f}, {aoi_row['reward_delta_ci95_high']:+.6f}].",
            "",
            "## Verdict boundary",
            "",
            "H1/H2 are tested rather than assumed. Static hull agreement speaks only to the scalar profile problem after FPS/budget are fixed. Runtime agreement cannot prove the full controller collapses to one scalar because AoI, speed, FPS, latency, prior map state, pending frames, safety, and switching remain active.",
            "",
            f"Linked ladder artifact: `{ladder_dir.relative_to(REPO_ROOT)}`.",
            "Genuine Whittle-index evaluation remains deferred to Phase-2 object-selective sharing, where per-object arms and indexability can be defined.",
            "",
        ]
    )
    (run_dir / "TASK_C_RESULTS.md").write_text(report, encoding="utf-8")
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
        "run_type": "task_c_rdo_structural_and_runtime",
        "implementation_status": "completed_table_driven_evaluation",
        "linked_controller_ladder": str(ladder_dir.relative_to(REPO_ROOT)),
        "source_hashes": {
            "matrix": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rdo": hashlib.sha256((REPO_ROOT / "rl_agent/policy/rdo.py").read_bytes()).hexdigest(),
        },
        "supported_profiles": list(supported),
        "files": files,
        "limitations": [
            "static duality gap is normalized task utility, not full sequential reward",
            "runtime ladder uses the epsilon-dominance-pruned seven-profile catalog",
            "lambda-RDO uses measured mean profile utility and has no scene conditioning",
            "AoI heuristic is not a Whittle index and carries no indexability claim",
            "table-driven surrogate only; no CARLA, OAI, OTA, or RL",
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("rl_agent/policy/configs/controller_ladder_task_c_v1.yaml"),
    )
    parser.add_argument("--replay-root", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--verification-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config, args.replay_root, args.split_manifest, args.verification_manifest))


if __name__ == "__main__":
    main()
