#!/usr/bin/env python3
"""Extend the 288-cell presentation pack with the final live edge result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_BASE_MANIFEST_SHA256 = (
    "67a9cfa1cb787d37bb49c5eb01ec7e173be138c7a222f053145a0e704b2ae00e"
)
EXPECTED_LIVE_MANIFEST_SHA256 = (
    "96fffa9d0f821af6fea571425f8190d482651d294d4354beaac0ca04057c675d"
)
EXPECTED_ACTION15_FAILURE_SHA256 = (
    "aca5a6eb5ea1e6e26156fd95ba739a4dd17a0f706c4075f27e415290a001c352"
)
V2 = "V2_PREDICTED_INSTALL_HORIZON"
V3 = "V3_OVERLAPPED_FINAL"
ACTIONS = (30, 50, 71)
ACTION_LABELS = {
    30: "A30\nAE128 / UINT4 / q=0",
    50: "A50\nAE64 / UINT4 / q=.50",
    71: "A71\nAE32 / UINT4 / q=.98",
}
VARIANT_LABELS = {V2: "Before final optimization", V3: "Final v3"}
VARIANT_COLORS = {V2: "#9CA3AF", V3: "#176B87"}


class FinalPresentationError(RuntimeError):
    """Raised when bound presentation evidence is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve(strict=True)
    observed = sha256_file(resolved)
    if observed != expected:
        raise FinalPresentationError(
            f"{label} SHA-256 mismatch: {observed} != {expected}"
        )
    return resolved


def verify_artifact_manifest(root: Path, manifest: dict[str, Any]) -> None:
    entries = manifest.get("sha256")
    if not isinstance(entries, dict) or not entries:
        raise FinalPresentationError("live artifact manifest has no SHA-256 inventory")
    problems = []
    for relative, expected in sorted(entries.items()):
        path = root / relative
        if not path.is_file():
            problems.append(f"missing:{relative}")
            continue
        observed = sha256_file(path)
        if observed != expected:
            problems.append(f"hash:{relative}:{observed}")
    if problems:
        raise FinalPresentationError(
            "live artifact verification failed: " + "; ".join(problems[:8])
        )
    if int(manifest.get("artifact_count", -1)) != len(entries):
        raise FinalPresentationError("live artifact count disagrees with inventory")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.15,
        }
    )


def bold_axes(axis: plt.Axes) -> None:
    axis.xaxis.label.set_fontweight("bold")
    axis.yaxis.label.set_fontweight("bold")
    axis.title.set_fontweight("bold")
    axis.tick_params(axis="both", width=1.1)
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontweight("bold")


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for axis in fig.axes:
        bold_axes(axis)
    metadata = {"Creator": "SplitFusion final optimization presentation pack"}
    fig.savefig(
        output / f"{stem}.png", bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        output / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={**metadata, "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def indexed_cells(document: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    if document.get("status") != "COMPLETE":
        raise FinalPresentationError("live comparison is not COMPLETE")
    if tuple(document.get("actions", ())) != ACTIONS:
        raise FinalPresentationError("live action inventory is not 30/50/71")
    cells = document.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise FinalPresentationError("live comparison must contain six cells")
    indexed = {(int(row["action_id"]), str(row["variant"])): row for row in cells}
    expected = {(action, variant) for action in ACTIONS for variant in (V2, V3)}
    if set(indexed) != expected:
        raise FinalPresentationError("live action/variant matrix is incomplete")
    if document.get("timing_semantics", {}).get(
        "per_installed_frame_components_reconcile_exactly"
    ) is not True:
        raise FinalPresentationError("live timing reconciliation was not asserted")
    return indexed


def build_summary(indexed: dict[tuple[int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in ACTIONS:
        before = indexed[(action, V2)]
        after = indexed[(action, V3)]
        service_before = float(before["edge_service_to_result_ms_median"])
        service_after = float(after["edge_service_to_result_ms_median"])
        aoi_before = float(before["install_aoi_ms_median"])
        aoi_after = float(after["install_aoi_ms_median"])
        rows.append(
            {
                "action_id": action,
                "profile_id": after["profile_id"],
                "payload_kib_v3": float(after["payload_bytes_median"]) / 1024.0,
                "edge_service_v2_ms": service_before,
                "edge_service_v3_ms": service_after,
                "edge_service_saving_ms": service_before - service_after,
                "edge_service_saving_percent": 100.0
                * (service_before - service_after)
                / service_before,
                "install_aoi_v2_ms": aoi_before,
                "install_aoi_v3_ms": aoi_after,
                "install_aoi_difference_ms": aoi_before - aoi_after,
                "installed_v2": int(before["ack_installed_frames"]),
                "installed_v3": int(after["ack_installed_frames"]),
                "installed_within_100ms_v2": int(before["installed_within_100ms"]),
                "installed_within_100ms_v3": int(after["installed_within_100ms"]),
            }
        )
    return rows


def plot_edge_service(
    indexed: dict[tuple[int, str], dict[str, Any]], output: Path
) -> None:
    x = np.arange(len(ACTIONS))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), constrained_layout=True)
    for offset, variant in ((-width / 2, V2), (width / 2, V3)):
        values = [
            float(indexed[(action, variant)]["edge_service_to_result_ms_median"])
            for action in ACTIONS
        ]
        bars = axes[0].bar(
            x + offset,
            values,
            width,
            label=VARIANT_LABELS[variant],
            color=VARIANT_COLORS[variant],
        )
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontweight="bold")
    savings = [
        float(indexed[(action, V2)]["edge_service_to_result_ms_median"])
        - float(indexed[(action, V3)]["edge_service_to_result_ms_median"])
        for action in ACTIONS
    ]
    percents = [
        100.0
        * saving
        / float(indexed[(action, V2)]["edge_service_to_result_ms_median"])
        for action, saving in zip(ACTIONS, savings)
    ]
    saving_bars = axes[1].bar(x, savings, color="#54A24B", width=0.55)
    for bar, value, percent in zip(saving_bars, savings, percents):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{value:.1f} ms\n({percent:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    labels = [ACTION_LABELS[action] for action in ACTIONS]
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("Split action")
    axes[0].set_ylim(0, 90)
    axes[0].set_yticks(np.arange(0, 91, 10))
    axes[0].set_ylabel("Median edge service (ms)")
    axes[0].set_title("Worker start to compact-result send")
    axes[0].legend(loc="upper right", frameon=True)
    axes[1].set_ylim(0, 11)
    axes[1].set_yticks(np.arange(0, 11, 2))
    axes[1].set_ylabel("Median edge-service saving (ms)")
    axes[1].set_title("Measured incremental saving")
    fig.suptitle(
        "Final output-preserving edge optimization — live CARLA + OAI",
        fontsize=15,
        fontweight="bold",
    )
    save(fig, output, "11_final_edge_service_v2_vs_v3")


def plot_live_stage_medians(
    indexed: dict[tuple[int, str], dict[str, Any]], output: Path
) -> None:
    stages = (
        ("Capture/preparation\nresidual", "preparation_to_front_residual_ms_median"),
        ("UE split\ndispatch", "ue_front_dispatch_ms_median"),
        ("Feature\nuplink", "feature_uplink_ms_median"),
        ("Edge\nqueue", "edge_queue_wait_ms_median"),
        ("Edge\nservice", "edge_service_to_result_ms_median"),
        ("Result to map\ninstallation", "result_to_map_install_ms_median"),
        ("Capture to map\ninstallation", "install_aoi_ms_median"),
    )
    x = np.arange(len(stages))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.5), constrained_layout=True)
    for axis, action in zip(axes, ACTIONS):
        for offset, variant in ((-width / 2, V2), (width / 2, V3)):
            values = [float(indexed[(action, variant)][key]) for _, key in stages]
            axis.bar(
                x + offset,
                values,
                width,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
            )
        axis.set_xticks(x, [label for label, _ in stages], rotation=28, ha="right")
        axis.set_ylim(0, 320)
        axis.set_yticks(np.arange(0, 301, 50))
        axis.set_xlabel("Measured interval")
        axis.set_ylabel("Median latency (ms)")
        axis.set_title(ACTION_LABELS[action].replace("\n", " — "))
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle(
        "Live latency medians by causal interval — bars are not an additive stack",
        fontsize=15,
        fontweight="bold",
    )
    save(fig, output, "12_final_live_latency_intervals")


def append_talking_points(
    base: str,
    rows: list[dict[str, Any]],
    action15_failure: dict[str, Any],
) -> str:
    savings = [float(row["edge_service_saving_ms"]) for row in rows]
    percentages = [float(row["edge_service_saving_percent"]) for row in rows]
    table = "\n".join(
        "| {action_id} | {edge_service_v2_ms:.1f} | {edge_service_v3_ms:.1f} | "
        "{edge_service_saving_ms:.1f} ({edge_service_saving_percent:.1f}%) | "
        "{install_aoi_v2_ms:.1f} | {install_aoi_v3_ms:.1f} |".format(**row)
        for row in rows
    )
    return (
        base.rstrip()
        + "\n\n## Figure 11 — final live edge optimization\n\n"
        + "- Both bars use the same predicted-install scheduler and fresh live "
        "CARLA/OAI lifecycles; only the edge implementation changes.\n"
        + f"- Median edge service fell by {min(savings):.1f}–{max(savings):.1f} ms "
        f"({min(percentages):.1f}–{max(percentages):.1f}%). This is a real, "
        "repeatable direction of improvement, but it is below the hoped-for "
        "10–20 ms.\n"
        + "- The changes overlap independent CPU/GPU work, remove one duplicate "
        "full-tensor finite scan, and avoid an immediate JSON serialize/parse cycle.\n"
        + "- Pure FCOS `decode_tail` remains about 21 ms. The displayed edge "
        "service ends only after the compact result has been sent.\n"
        + "- Agent connection: the optimized service distribution belongs in the "
        "simulator. Do not subtract one constant from all 288 historical samples.\n\n"
        + "| Action | v2 edge (ms) | v3 edge (ms) | Saving | v2 E2E AoI (ms) | v3 E2E AoI (ms) |\n"
        + "|---:|---:|---:|---:|---:|---:|\n"
        + table
        + "\n\n## Figure 12 — final live latency intervals\n\n"
        + "- The six causal intervals are shown separately, followed by the "
        "authoritative capture-to-install AoI. They are not stacked because "
        "medians from different installed frames do not add exactly.\n"
        + "- Feature uplink includes the live OAI application path from first UE "
        "datagram send to complete edge reassembly. It is unaffected by the edge "
        "code change in design; small run-to-run differences are expected.\n"
        + "- Edge queue medians are near zero under the predicted-install/latest-only "
        "scheduler. The remaining latency is distributed across UE preparation/"
        "dispatch, radio transfer, edge service and result installation.\n"
        + "- One action-71 frame installed within 100 ms in the v3 run, but a single "
        "event is not 100-ms service qualification.\n"
        + "- Action 15 is deliberately absent: two v3 live attempts failed closed "
        "after a non-finite camera-aware geometry output. It is a numerical "
        "reliability finding, not usable latency evidence, and must not be hidden "
        "or averaged into the successful actions.\n\n"
        + "## Honest meeting conclusion\n\n"
        + "The complete optimization program substantially reduced the original "
        "edge bottleneck, and the final pass adds another measured 6–9 ms. End-to-end "
        "latency is still mostly 160–290 ms for these three examples, so the system "
        "is suitable for cooperative map awareness and early warning—not as the sole "
        "hard real-time emergency-braking authority. The recurrent policy should "
        "optimize installed-map utility and freshness under this measured frontier.\n\n"
        + "Action-15 retained failure statement: `"
        + str(action15_failure.get("failure", "missing"))
        + "`.\n"
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=root
        / "experiments/splitfusion_rl_policy_design_v1/20260910_288_results_presentation_pack_v6",
    )
    parser.add_argument(
        "--live",
        type=Path,
        default=root
        / "experiments/splitfusion_edge_optimization_v3/20260910_live_v2_vs_v3_actions30_50_71_retry1",
    )
    parser.add_argument(
        "--action15-failure",
        type=Path,
        default=root
        / "experiments/splitfusion_edge_optimization_v3/20260910_action15_v3_failure_reproduction_retry1/action15__v3_overlapped_final/NO_SCIENTIFIC_ROWS.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base.resolve(strict=True)
    live = args.live.resolve(strict=True)
    require_sha256(
        base / "artifact_manifest.json",
        EXPECTED_BASE_MANIFEST_SHA256,
        "base presentation manifest",
    )
    live_manifest_path = require_sha256(
        live / "ARTIFACT_MANIFEST.json",
        EXPECTED_LIVE_MANIFEST_SHA256,
        "live artifact manifest",
    )
    action15_path = require_sha256(
        args.action15_failure,
        EXPECTED_ACTION15_FAILURE_SHA256,
        "action-15 failure record",
    )
    base_manifest = json.loads((base / "artifact_manifest.json").read_text())
    for relative, metadata in base_manifest["artifacts"].items():
        path = base / relative
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise FinalPresentationError(f"base artifact failed verification: {relative}")
    live_manifest = json.loads(live_manifest_path.read_text())
    verify_artifact_manifest(live, live_manifest)
    live_document = json.loads((live / "LIVE_PAYOFF_RESULTS.json").read_text())
    indexed = indexed_cells(live_document)
    action15_failure = json.loads(action15_path.read_text())
    if "non-finite" not in str(action15_failure.get("failure", "")):
        raise FinalPresentationError("action-15 failure classification drift")

    output = args.output.resolve(strict=False)
    if output.exists():
        raise FinalPresentationError(f"create-only output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for source in sorted(base.iterdir()):
        if source.is_file() and source.name not in {
            "artifact_manifest.json",
            "TALKING_POINTS.md",
        }:
            shutil.copy2(source, output / source.name)

    configure_style()
    plot_edge_service(indexed, output)
    plot_live_stage_medians(indexed, output)
    rows = build_summary(indexed)
    with (output / "final_edge_v2_v3_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    base_talking = (base / "TALKING_POINTS.md").read_text(encoding="utf-8")
    (output / "TALKING_POINTS.md").write_text(
        append_talking_points(base_talking, rows, action15_failure),
        encoding="utf-8",
    )

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.name == "artifact_manifest.json" or not path.is_file():
            continue
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": "scenesense.splitfusion.288_results_presentation_pack.v3",
        "status": "COMPLETE",
        "base_manifest_sha256": sha256_file(base / "artifact_manifest.json"),
        "live_manifest_sha256": sha256_file(live_manifest_path),
        "live_result_sha256": sha256_file(live / "LIVE_PAYOFF_RESULTS.json"),
        "action15_failure_sha256": sha256_file(action15_path),
        "final_live_contract": {
            "actions": list(ACTIONS),
            "variants": [V2, V3],
            "frames_per_cell": 300,
            "network_profile": "FAVORABLE_STABLE",
            "claims_100ms_service_ready": False,
            "run_to_run_variance_characterized": False,
        },
        "artifacts": artifacts,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("SPLITFUSION_FINAL_OPTIMIZATION_PRESENTATION_PACK_COMPLETE")
    print(json.dumps({"output": str(output), "artifacts": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
