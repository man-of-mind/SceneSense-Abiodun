#!/usr/bin/env python3
"""Quality-control and plot a valid Experiment-3 FOV diagnostic run.

Localization error is summarized only for gated target matches. Match rate is
reported separately so edge failures cannot disappear through match censoring.
This script writes a QC report, not a research conclusion.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-samples-per-offset", type=int, default=10)
    parser.add_argument("--min-center-match-rate", type=float, default=0.5)
    parser.add_argument("--center-sanity-max-m", type=float, default=3.0)
    parser.add_argument(
        "--expected-offsets",
        default="-8,-6,-4,-2,0,2,4,6,8",
        help="Comma-separated requested sweep positions; missing positions fail QC.",
    )
    return parser.parse_args()


def as_float(row: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        value = str(row.get(key, "")).strip()
        return float(value) if value else default
    except (TypeError, ValueError):
        return default


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        value = str(row.get(key, "")).strip()
        return int(float(value)) if value else default
    except (TypeError, ValueError):
        return default


def finite(values: List[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def fmt(value: float, digits: int = 3) -> str:
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    results_csv = args.results_csv.resolve()
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    output_dir = (args.output_dir or results_csv.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {results_csv}")

    sweep_mode = str(rows[0].get("sweep_mode", "lateral") or "lateral")
    grouped: Dict[float, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        sweep_value = as_float(row, "sweep_value")
        if not math.isfinite(sweep_value):
            sweep_value = as_float(row, "offset_m")
        grouped[sweep_value].append(row)

    summaries = []
    qc_failures = []
    minimum = max(1, int(args.min_samples_per_offset))
    expected_offsets = [float(value.strip()) for value in args.expected_offsets.split(",") if value.strip()]
    for expected in expected_offsets:
        if not any(abs(observed - expected) < 1e-6 for observed in grouped):
            qc_failures.append(f"requested sweep position {expected:+g} is missing")
    for offset in sorted(grouped):
        group = grouped[offset]
        in_fov = [row for row in group if as_int(row, "target_in_fov") == 1]
        matched = [row for row in in_fov if as_int(row, "matched") == 1]
        errors = finite([as_float(row, "error_m") for row in matched])
        target_pixels = finite([as_float(row, "target_pixel_x_from_center") for row in in_fov])
        radar_counts = finite([as_float(row, "raw_radar_support_count") for row in in_fov])
        radar_scores = finite([as_float(row, "radar_support_score") for row in matched])
        scores = finite([as_float(row, "score") for row in matched])
        summary = {
            "sweep_value": offset,
            "samples": len(group),
            "target_in_fov": len(in_fov),
            "matched": len(matched),
            "match_rate": len(matched) / len(in_fov) if in_fov else float("nan"),
            "target_pixel_x_from_center_mean": float(np.mean(target_pixels)) if target_pixels.size else float("nan"),
            "error_mean_m": float(np.mean(errors)) if errors.size else float("nan"),
            "error_std_m": float(np.std(errors)) if errors.size else float("nan"),
            "error_median_m": float(np.median(errors)) if errors.size else float("nan"),
            "error_p90_m": float(np.percentile(errors, 90)) if errors.size else float("nan"),
            "raw_radar_support_count_mean": float(np.mean(radar_counts)) if radar_counts.size else float("nan"),
            "radar_support_score_mean": float(np.mean(radar_scores)) if radar_scores.size else float("nan"),
            "detection_score_mean": float(np.mean(scores)) if scores.size else float("nan"),
        }
        summaries.append(summary)
        if len(group) < minimum:
            qc_failures.append(f"sweep position {offset:+g} has only {len(group)} samples (<{minimum})")

    center = next((row for row in summaries if abs(float(row["sweep_value"])) < 1e-6), None)
    if center is None:
        qc_failures.append("center offset is missing")
    else:
        center_rate = float(center["match_rate"])
        center_median = float(center["error_median_m"])
        if not math.isfinite(center_rate) or center_rate < float(args.min_center_match_rate):
            qc_failures.append(
                f"center match rate {center_rate:.1%} is below {float(args.min_center_match_rate):.1%}"
            )
        if not math.isfinite(center_median) or center_median > float(args.center_sanity_max_m):
            qc_failures.append(
                f"center median error {center_median:.2f} m exceeds {float(args.center_sanity_max_m):.2f} m"
            )

    summary_csv = output_dir / "summary_by_offset.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    x = np.asarray([float(row["target_pixel_x_from_center_mean"]) for row in summaries])
    error_mean = np.asarray([float(row["error_mean_m"]) for row in summaries])
    error_std = np.asarray([float(row["error_std_m"]) for row in summaries])
    match_rate = np.asarray([float(row["match_rate"]) for row in summaries])
    radar_count = np.asarray([float(row["raw_radar_support_count_mean"]) for row in summaries])
    radar_score = np.asarray([float(row["radar_support_score_mean"]) for row in summaries])

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    valid = np.isfinite(x) & np.isfinite(error_mean)
    ax.errorbar(x[valid], error_mean[valid], yerr=error_std[valid], marker="o", capsize=4)
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel("GT target pixel x from image center (model pixels)")
    ax.set_ylabel("Gated target localization error (m), mean ± std")
    ax.set_title("FOV position vs conditional localization error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fov_error_vs_position.png", dpi=220)
    fig.savefig(output_dir / "fov_error_vs_position.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    valid = np.isfinite(x) & np.isfinite(match_rate)
    ax.plot(x[valid], match_rate[valid], marker="o")
    ax.axvline(0.0, color="0.6", linestyle="--", linewidth=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("GT target pixel x from image center (model pixels)")
    ax.set_ylabel("Target match rate")
    ax.set_title("FOV position vs target availability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fov_match_rate_vs_position.png", dpi=220)
    fig.savefig(output_dir / "fov_match_rate_vs_position.pdf")
    plt.close(fig)

    fig, ax_left = plt.subplots(figsize=(7.2, 4.5))
    valid_count = np.isfinite(x) & np.isfinite(radar_count)
    count_line = ax_left.plot(x[valid_count], radar_count[valid_count], marker="o", label="Raw radar count")
    ax_left.set_xlabel("GT target pixel x from image center (model pixels)")
    ax_left.set_ylabel("Raw radar points inside target box")
    ax_left.grid(alpha=0.25)
    ax_right = ax_left.twinx()
    valid_score = np.isfinite(x) & np.isfinite(radar_score)
    score_line = ax_right.plot(
        x[valid_score], radar_score[valid_score], marker="s", color="tab:orange", label="Predicted radar support"
    )
    ax_right.set_ylabel("Predicted radar-support score")
    ax_left.legend(count_line + score_line, [line.get_label() for line in count_line + score_line], loc="best")
    ax_left.set_title("FOV position vs radar support")
    fig.tight_layout()
    fig.savefig(output_dir / "fov_radar_support_vs_position.png", dpi=220)
    fig.savefig(output_dir / "fov_radar_support_vs_position.pdf")
    plt.close(fig)

    report = [
        "# Experiment 3 FOV Diagnostic — QC Report",
        "",
        f"- Input: `{results_csv}`",
        f"- Rows: `{len(rows)}`",
        f"- QC status: **{'FAIL — DO NOT INTERPRET' if qc_failures else 'PASS'}**",
        "",
        "## Checks",
        "",
    ]
    if qc_failures:
        report.extend(f"- FAIL: {failure}" for failure in qc_failures)
    else:
        report.append(
            "- All sweep positions meet the sample floor; center match-rate and localization sanity checks pass."
        )
    report.extend(
        [
            "",
            "## Per-offset measurements",
            "",
            (
                "| target angle deg | n | in FOV | matched | match rate | error mean±std m | median m | raw radar count | radar score |"
                if sweep_mode == "target-angle"
                else "| offset m | n | in FOV | matched | match rate | error mean±std m | median m | raw radar count | radar score |"
            ),
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        report.append(
            f"| {float(row['sweep_value']):+g} | {int(row['samples'])} | {int(row['target_in_fov'])} | "
            f"{int(row['matched'])} | {fmt(float(row['match_rate']))} | "
            f"{fmt(float(row['error_mean_m']))}±{fmt(float(row['error_std_m']))} | "
            f"{fmt(float(row['error_median_m']))} | {fmt(float(row['raw_radar_support_count_mean']))} | "
            f"{fmt(float(row['radar_support_score_mean']))} |"
        )
    report.extend(
        [
            "",
            "Conditional error and match rate must be interpreted together. A falling error curve caused by "
            "low edge match rate is selection bias, not improved edge localization.",
            "",
        ]
    )
    (output_dir / "QC_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {summary_csv} and QC plots to {output_dir}")
    if qc_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
