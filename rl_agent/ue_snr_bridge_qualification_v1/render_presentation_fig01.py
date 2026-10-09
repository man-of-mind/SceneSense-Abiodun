#!/usr/bin/env python3
"""Render a slide-friendly copy of the UE/gNB SNR time-series figure.

The scientific evidence figure is deliberately left unchanged.  This renderer
only changes presentation typography, layout, and human-readable profile names.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rl_agent.ue_snr_bridge_qualification_v1 import (
    ue_snr_bridge_analyzer_v1 as analysis,
)


DISPLAY_NAMES = {
    "FAVORABLE_STABLE": "Favorable",
    "MID_VARIABLE": "Mid-variable",
    "ADVERSE_STABLE": "Adverse",
    "FADE_RECOVERY": "Fade recovery",
}


def render(run_dir: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = analysis.analyze(run_dir, bin_ms=100)
    present = [
        name
        for name in analysis.PROFILE_ORDER
        if name in report["per_profile"]
    ]
    if len(present) != 4:
        raise RuntimeError(
            f"expected four registered profiles, found {present!r}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.2), sharex=False)
    for axis, name in zip(axes.flat, present):
        block = report["per_profile"][name]
        axis.plot(
            block["gnb_series"],
            block["gnb_values"],
            linewidth=1.2,
            alpha=0.72,
            color="#4A4A4A",
            label="gNB received uplink PUSCH SNR",
        )
        axis.plot(
            block["ue_series"],
            block["ue_values"],
            linewidth=1.6,
            marker=".",
            markersize=2.8,
            color=analysis.PROFILE_COLORS[name],
            label="UE downlink SNR",
        )
        axis.set_title(
            DISPLAY_NAMES[name], fontsize=14, fontweight="bold", pad=8
        )
        axis.grid(alpha=0.25)
        axis.tick_params(axis="both", labelsize=10)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        fontsize=10,
        frameon=True,
    )

    fig.supxlabel(
        "Time within measured window (s)", fontsize=13, fontweight="bold"
    )
    fig.supylabel("SNR (dB)", fontsize=13, fontweight="bold")
    fig.suptitle(
        "UE Downlink SNR and gNB Uplink PUSCH SNR Across Network Profiles",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.025, 0.025, 1.0, 0.885))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    render(args.run_dir.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
