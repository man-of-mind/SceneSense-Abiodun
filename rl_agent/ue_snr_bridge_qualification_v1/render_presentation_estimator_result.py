#!/usr/bin/env python3
"""Render the UE-local uplink-SNR estimator result for a presentation slide."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def render(result_path: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads(result_path.read_text(encoding="utf-8"))
    plateaus = [abs(int(value)) for value in result["plateau_order"]]
    measured = [float(value) for value in result["measured_by_plateau"]]
    predicted = [float(value) for value in result["predicted_by_plateau"]]

    fig, axis = plt.subplots(figsize=(10.8, 6.0))
    for index, step in enumerate(plateaus):
        role = "HELDOUT" if str(-step) in result["heldout_by_plateau"] else "FIT"
        if role == "HELDOUT":
            axis.axvspan(
                step - 0.55,
                step + 0.55,
                color="#9AA0A6",
                alpha=0.15,
                zorder=0,
                label="Held-out channel plateau" if index == 1 else None,
            )

    axis.plot(
        plateaus,
        measured,
        "o-",
        color="#1F6FEB",
        linewidth=2.8,
        markersize=8,
        label="Measured at gNB (uplink PUSCH SNR)",
    )
    axis.plot(
        plateaus,
        predicted,
        "s--",
        color="#D2544F",
        linewidth=2.8,
        markersize=8,
        label="Estimated locally at UE",
    )
    axis.set_title(
        "UE-Local Estimate Does Not Track Uplink SNR",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    axis.set_xlabel(
        "Controlled RFsim channel step (dB)",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_ylabel("SNR (dB)", fontsize=13, fontweight="bold")
    axis.set_xticks(plateaus)
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=10.5, frameon=True)
    axis.text(
        0.98,
        0.08,
        "gNB measurement: 20 → 32 dB\nUE estimate: approximately 26 dB",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "alpha": 0.9},
    )
    axis.set_ylim(min(measured + predicted) - 1.5, max(measured + predicted) + 1.5)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    render(args.result.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
