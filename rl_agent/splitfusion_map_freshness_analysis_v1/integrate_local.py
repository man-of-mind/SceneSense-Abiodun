#!/usr/bin/env python3
"""Integrate the measured LOCAL baseline with the 288-cell SPLIT surface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260913_action_network_freshness_v2"
)
LOCAL_ROOT = ROOT / (
    "experiments/splitfusion_local_action_baseline_v1/"
    "20260913_local_compact_four_profile_retry1"
)
PREP_ROOT = ROOT / (
    "experiments/splitfusion_edge_freshness_scheduler_v1/"
    "20260910_live_actions50_71_two_policies"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/splitfusion_map_freshness_policy_analysis_v1/"
    "20260913_policy_decision_with_local_v3"
)
BUDGETS = (150, 200, 250)
PROFILES = (
    "FAVORABLE_STABLE",
    "MID_VARIABLE",
    "ADVERSE_STABLE",
    "FADE_RECOVERY",
)
OBSERVATION_TAIL_NS = 500_000_000
TERMINAL = "SPLITFUSION_MAP_FRESHNESS_POLICY_SYNTHESIS_COMPLETE"


class SynthesisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SynthesisError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def verify_manifest(root: Path, name: str) -> dict[str, str]:
    manifest = load_json(root / name)
    records = manifest.get("sha256") or manifest.get("files")
    require(isinstance(records, dict) and records, f"invalid manifest: {root/name}")
    for relative, expected in records.items():
        path = root / str(relative)
        require(path.is_file(), f"manifest artifact absent: {path}")
        require(sha256(path) == expected, f"manifest hash drift: {path}")
    return {str(key): str(value) for key, value in records.items()}


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    require(bool(ordered), "empty percentile input")
    return ordered[max(0, math.ceil(len(ordered) * probability) - 1)]


def optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def map_metrics(rows: Sequence[Mapping[str, str]], preparation_offset_ms: float) -> dict[str, float]:
    offset_ns = int(round(float(preparation_offset_ms) * 1e6))
    events = sorted(
        (
            int(row["edge_install_raw_ns"]),
            int(row["capture_raw_ns"]) - offset_ns,
        )
        for row in rows
        if row["ack_status"] == "ACK_INSTALLED"
    )
    require(len(events) == 300, "LOCAL map series is incomplete")
    require(all(events[i][1] < events[i + 1][1] for i in range(len(events) - 1)), "LOCAL capture order drift")
    require(all(events[i][0] < events[i + 1][0] for i in range(len(events) - 1)), "LOCAL install order drift")
    start = min(capture for _, capture in events)
    end = max(capture for _, capture in events) + OBSERVATION_TAIL_NS
    duration = end - start
    first_install = events[0][0]
    available = max(0, end - first_install)
    area = 0.0
    fresh = {budget: 0 for budget in BUDGETS}
    for index, (install, capture) in enumerate(events):
        interval_start = max(start, install)
        interval_end = min(end, events[index + 1][0]) if index + 1 < len(events) else end
        if interval_end <= interval_start:
            continue
        age_start = interval_start - capture
        interval = interval_end - interval_start
        area += float(age_start) * interval + 0.5 * float(interval) ** 2
        for budget in BUDGETS:
            fresh[budget] += max(0, min(interval_end, capture + budget * 1_000_000) - interval_start)
    result = {
        "observation_duration_s": duration / 1e9,
        "map_available_fraction": available / duration,
        "time_weighted_map_aoi_ms_when_available": area / available / 1e6,
        "preparation_offset_ms": float(preparation_offset_ms),
    }
    for budget in BUDGETS:
        result[f"fresh_map_time_ms_le_{budget}_fraction"] = fresh[budget] / duration
    return result


def preparation_summary() -> dict[str, Any]:
    manifest = verify_manifest(PREP_ROOT, "ARTIFACT_MANIFEST.json")
    paths = (
        PREP_ROOT / "action50__latest_only_no_expiry/per_frame/action_50_split_ae64_uint4_q5000.csv",
        PREP_ROOT / "action71__latest_only_no_expiry/per_frame/action_71_split_ae32_uint4_q9800.csv",
    )
    values = [
        float(row["prep_queue_wait_ms"])
        for path in paths
        for row in read_csv(path)
        if row.get("prep_queue_wait_ms") not in (None, "")
    ]
    compute = [
        float(row["prep_pre_front_compute_ms"])
        for path in paths
        for row in read_csv(path)
        if row.get("prep_pre_front_compute_ms") not in (None, "")
    ]
    require(len(values) == len(compute) == 600, "preparation sensitivity sample drift")
    return {
        "source_manifest_sha256": sha256(PREP_ROOT / "ARTIFACT_MANIFEST.json"),
        "source_artifacts_verified": len(manifest),
        "samples": len(values),
        "total_sensor_sync_and_preparation_ms": {
            "p50": statistics.median(values),
            "p95": percentile(values, 0.95),
        },
        "compute_only_pre_front_ms": {
            "p50": statistics.median(compute),
            "p95": percentile(compute, 0.95),
        },
        "use": "CONSTANT_P50_AND_P95_SENSITIVITY_OFFSETS_NOT_A_MATCHED_LOCAL_REMEASUREMENT",
    }


def split_rows() -> list[dict[str, str]]:
    rows = read_csv(SPLIT_ROOT / "action_network_policy_freshness.csv")
    selected = [row for row in rows if row["queue_policy"] == "LATEST_ONLY_NO_EXPIRY"]
    require(len(selected) == 288, "strict-latest split surface is not 72 x 4")
    require({row["network_profile"] for row in selected} == set(PROFILES), "split profile inventory drift")
    return selected


def local_rows(preparation: Mapping[str, Any], qualification: Mapping[str, Any]) -> list[dict[str, Any]]:
    quality = qualification["quality"]["full_local_fp32_q0"]
    p50 = float(preparation["total_sensor_sync_and_preparation_ms"]["p50"])
    p95 = float(preparation["total_sensor_sync_and_preparation_ms"]["p95"])
    result = []
    for index, profile in enumerate(PROFILES):
        directory = LOCAL_ROOT / "profiles" / f"{index:02d}_{profile}"
        rows = read_csv(directory / "local_transport_frames.csv")
        direct = map_metrics(rows, 0.0)
        sensor_p50 = map_metrics(rows, p50)
        sensor_p95 = map_metrics(rows, p95)
        payload = [float(row["payload_bytes"]) for row in rows]
        summary = load_json(directory / "profile_summary.json")
        item: dict[str, Any] = {
            "action_kind": "LOCAL_PROXY",
            "action_id": "LOCAL",
            "profile_id": "local_fcos_r50_fpn_p2_p7_p025_v1",
            "network_profile": profile,
            "family": "FULL_LOCAL_FCOS",
            "quantizer": "FP32_LOCAL",
            "q": 0.0,
            "uplink_payload_semantics": "COMPACT_P025_OBJECT_RESULT",
            "median_uplink_bytes": statistics.median(payload),
            "p95_uplink_bytes": percentile(payload, 0.95),
            "installation_rate_per_sent": float(summary["installation_rate_per_sent"]),
            "full_local_compute_ms_p50": qualification["compute"]["timing_ms"]["local_result_available_ms"]["p50"],
            "full_local_compute_ms_p95": qualification["compute"]["timing_ms"]["local_result_available_ms"]["p95"],
            "vehicle_f1": float(quality["vehicle_f1"]),
            "person_f1": float(quality["person_avo_f1"]),
            "vehicle_xy_mae_m": float(quality["vehicle_xy_mae_m"]),
            "person_xy_mae_m": float(quality["person_avo_xy_mae_m"]),
            "measured_prepared_input_to_map_ms_p50": summary["timing_ms"]["capture_to_map_install_ms"]["p50"],
            "measured_prepared_input_to_map_ms_p95": summary["timing_ms"]["capture_to_map_install_ms"]["p95"],
            "prepared_boundary_map_aoi_ms": direct["time_weighted_map_aoi_ms_when_available"],
            "sensor_p50_sensitivity_map_aoi_ms": sensor_p50["time_weighted_map_aoi_ms_when_available"],
            "sensor_p95_sensitivity_map_aoi_ms": sensor_p95["time_weighted_map_aoi_ms_when_available"],
        }
        for budget in BUDGETS:
            for label, source in (
                ("prepared_boundary", direct),
                ("sensor_p50_sensitivity", sensor_p50),
                ("sensor_p95_sensitivity", sensor_p95),
            ):
                fraction = source[f"fresh_map_time_ms_le_{budget}_fraction"]
                item[f"{label}_fresh_map_ms_le_{budget}_fraction"] = fraction
                item[f"{label}_person_f1_x_freshness_ms_le_{budget}"] = fraction * item["person_f1"]
                item[f"{label}_vehicle_f1_x_freshness_ms_le_{budget}"] = fraction * item["vehicle_f1"]
        result.append(item)
    return result


def compact_split_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item: dict[str, Any] = {
            "action_kind": "SPLIT",
            "action_id": int(row["action_id"]),
            "profile_id": row["profile_id"],
            "network_profile": row["network_profile"],
            "family": row["family"],
            "quantizer": row["quantizer"],
            "q": float(row["q"]),
            "uplink_payload_semantics": "SPLIT_FEATURE_PAYLOAD",
            "median_uplink_bytes": float(row["median_feature_bytes"]),
            "installation_rate_per_sent": float(row["rate_useful_install_per_sent"]),
            "vehicle_f1": float(row["val_vehicle_f1"]),
            "person_f1": float(row["val_canonical_person_f1"]),
            "vehicle_xy_mae_m": float(row["val_vehicle_xy_mae_m"]),
            "person_xy_mae_m": float(row["val_canonical_person_xy_mae_m"]),
            "map_aoi_ms": optional_float(row["time_weighted_map_aoi_ms_when_available"]),
            "queue_wait_ms_p95": optional_float(row["queue_wait_ms_p95"]),
            "superseded_pending_compute": int(row["reason_SUPERSEDED_PENDING_COMPUTE"]),
        }
        for budget in BUDGETS:
            fraction = float(row[f"fresh_map_time_ms_le_{budget}_fraction"])
            item[f"fresh_map_ms_le_{budget}_fraction"] = fraction
            item[f"person_f1_x_freshness_ms_le_{budget}"] = fraction * item["person_f1"]
            item[f"vehicle_f1_x_freshness_ms_le_{budget}"] = fraction * item["vehicle_f1"]
        result.append(item)
    return result


def best_table(split: Sequence[Mapping[str, Any]], local: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for profile in PROFILES:
        candidates = [row for row in split if row["network_profile"] == profile]
        local_row = next(row for row in local if row["network_profile"] == profile)
        for budget in BUDGETS:
            raw_key = f"fresh_map_ms_le_{budget}_fraction"
            person_key = f"person_f1_x_freshness_ms_le_{budget}"
            vehicle_key = f"vehicle_f1_x_freshness_ms_le_{budget}"
            best_raw = max(candidates, key=lambda row: row[raw_key])
            best_person = max(candidates, key=lambda row: row[person_key])
            best_vehicle = max(candidates, key=lambda row: row[vehicle_key])
            local_raw = local_row[f"sensor_p50_sensitivity_fresh_map_ms_le_{budget}_fraction"]
            local_person = local_row[f"sensor_p50_sensitivity_person_f1_x_freshness_ms_le_{budget}"]
            local_vehicle = local_row[f"sensor_p50_sensitivity_vehicle_f1_x_freshness_ms_le_{budget}"]
            result.append({
                "network_profile": profile,
                "budget_ms": budget,
                "best_raw_split_action": best_raw["action_id"],
                "best_raw_split_fraction": best_raw[raw_key],
                "best_person_split_action": best_person["action_id"],
                "best_person_split_score": best_person[person_key],
                "best_vehicle_split_action": best_vehicle["action_id"],
                "best_vehicle_split_score": best_vehicle[vehicle_key],
                "local_sensor_p50_fraction": local_raw,
                "local_sensor_p95_stress_fraction": local_row[f"sensor_p95_sensitivity_fresh_map_ms_le_{budget}_fraction"],
                "local_person_score_sensor_p50": local_person,
                "local_vehicle_score_sensor_p50": local_vehicle,
                "local_minus_best_raw_split": local_raw - best_raw[raw_key],
                "local_minus_best_person_split": local_person - best_person[person_key],
                "local_minus_best_vehicle_split": local_vehicle - best_vehicle[vehicle_key],
            })
    return result


def render_report(
    preparation: Mapping[str, Any],
    local: Sequence[Mapping[str, Any]],
    best: Sequence[Mapping[str, Any]],
) -> str:
    prep50 = preparation["total_sensor_sync_and_preparation_ms"]["p50"]
    prep95 = preparation["total_sensor_sync_and_preparation_ms"]["p95"]
    lines = [
        "# SplitFusion map-freshness policy synthesis",
        "",
        "## Scientific decision",
        "",
        "Strict latest-only is the edge scheduling policy. It never interrupts active CUDA work; when the worker becomes free it processes only the newest pending frame and records older pending frames as `SUPERSEDED_PENDING`. There is no 25-ms expiry and no FIFO fallback.",
        "",
        "The current SPLIT surface does not reliably satisfy 150, 200 or 250 ms. The measured LOCAL proxy changes that boundary, but it does not remove the need for an explicit vehicle-compute cost or hardware-availability state.",
        "",
        "## LOCAL result",
        "",
        "The current FCOS full-local path takes 26.55 ms at p50 and 29.06 ms at p95 on the RTX 5090 proxy. Its compact p025 object result is about 10 KiB median. All 1,200 one-shot messages installed across the four OAI profiles; there were no duplicates, reassembly expirations or rejected identities.",
        "",
        "The direct LOCAL clock starts when the normalized seven-channel tensor is ready. It is not physical sensor-capture AoI. To expose that boundary honestly, the table below adds constant offsets from 600 retained live preparation rows: %.2f ms p50 and %.2f ms p95. These are sensitivity cases, not matched LOCAL remeasurements." % (prep50, prep95),
        "",
        "| Profile | Budget | Best SPLIT fresh map | LOCAL + prep p50 | LOCAL + prep p95 stress |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            "| %s | %d ms | %.1f%% (a%s) | %.1f%% | %.1f%% |" % (
                str(row["network_profile"]).replace("_", " ").title(),
                row["budget_ms"],
                100 * row["best_raw_split_fraction"],
                row["best_raw_split_action"],
                100 * row["local_sensor_p50_fraction"],
                100 * row["local_sensor_p95_stress_fraction"],
            )
        )
    lines += [
        "",
        "The p50-preparation LOCAL sensitivity exceeds the best raw SPLIT result in every profile and budget. At the p95 preparation stress, LOCAL is effectively unable to meet 150 ms, is similar to the best SPLIT region around 200 ms, and remains clearly stronger at 250 ms. The preparation path is therefore part of the policy boundary, not a harmless constant.",
        "",
        "## What this means for PPO",
        "",
        "If the reward contains only quality and freshness, `LOCAL` is nearly dominant: it has full-FP32 quality, a small compact uplink and much fresher map state. A useful controller must also charge measured vehicle compute/energy and expose local accelerator availability, occupancy, temperature or battery state. Otherwise PPO will simply learn `always LOCAL`.",
        "",
        "For SPLIT, the network profile label is never a policy input. It is an analysis stratum. The recurrent policy observes causal SNR/MCS/capacity trends, current map AoI, time since useful install, edge busy/pending state, previous action and terminal outcome. The LSTM can infer the latent profile and its direction without seeing future trace values.",
        "",
        "Use separate person and vehicle utilities. For budget `B`, a suitable starting reward is:",
        "",
        "```math",
        "r_t = (w_p Q_p(a_t)+w_v Q_v(a_t))F_B(\\mathrm{AoI}_{t+1})",
        "      -\\lambda_b b(a_t)/B_{\\max}",
        "      -\\lambda_v C_{\\mathrm{vehicle}}(a_t)",
        "      -\\lambda_e C_{\\mathrm{edge}}(a_t)",
        "      -\\lambda_s \\mathbf{1}[a_t\\ne a_{t-1}]",
        "```",
        "",
        "where `F_B` is evaluated on the map state, not merely on an ACK. A superseded SPLIT frame earns no new-map utility, while already spent radio/compute cost remains charged; it receives no extra arbitrary discard penalty.",
        "",
        "## Training-data consequence",
        "",
        "The simulator should contain two empirical branches:",
        "",
        "1. 72 SPLIT actions × four measured network strata under strict latest-only scheduling.",
        "2. One `LOCAL_PROXY` branch with measured local compute and four compact-result transport distributions, plus a sampled shared sensor-preparation term.",
        "",
        "The budgets 150/200/250 ms may be separate experiments or a context variable in the state. Start with separate fixed-budget experiments for interpretability. Do not train from profile names, average network profiles together, or subtract a single latency constant from the 288 cells.",
        "",
        "## Limits",
        "",
        "- LOCAL compute was measured on a desktop RTX 5090, not production vehicle hardware.",
        "- LOCAL transport has one 300-sample run per profile; it establishes feasibility, not run-to-run variance.",
        "- The LOCAL sink is the versioned compact object-map boundary, not the future multi-UE fusion server.",
        "- Dense segmentation is not transported and earns no map utility.",
        "- Person localization aging remains unavailable in the 288-cell retained data; only vehicle localization has aligned secondary evidence.",
        "",
    ]
    return "\n".join(lines)


def plot(output: Path, best: Sequence[Mapping[str, Any]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    labels = ["Favorable", "Mid-variable", "Adverse", "Fade/recovery"]
    generated: list[str] = []
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    x = np.arange(len(PROFILES))
    width = 0.25
    for axis, budget in zip(axes, BUDGETS):
        rows = [next(row for row in best if row["network_profile"] == profile and row["budget_ms"] == budget) for profile in PROFILES]
        axis.bar(x - width, [100 * row["best_raw_split_fraction"] for row in rows], width, label="Best SPLIT", color="#4472C4")
        axis.bar(x, [100 * row["local_sensor_p50_fraction"] for row in rows], width, label="LOCAL + prep p50", color="#70AD47")
        axis.bar(x + width, [100 * row["local_sensor_p95_stress_fraction"] for row in rows], width, label="LOCAL + prep p95 stress", color="#ED7D31")
        axis.set_title(f"{budget} ms budget")
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlabel("Network profile")
    axes[0].set_ylabel("Fresh-map time (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.07, right=0.99, wspace=0.15)
    for suffix in ("png", "pdf"):
        name = f"01_split_vs_local_map_freshness.{suffix}"
        fig.savefig(output / name, dpi=220 if suffix == "png" else None)
        generated.append(name)
    plt.close(fig)

    for metric, title, filename in (
        ("person", "Person quality × fresh-map time", "02_person_quality_weighted_freshness"),
        ("vehicle", "Vehicle quality × fresh-map time", "03_vehicle_quality_weighted_freshness"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
        for axis, budget in zip(axes, BUDGETS):
            rows = [next(row for row in best if row["network_profile"] == profile and row["budget_ms"] == budget) for profile in PROFILES]
            split_key = f"best_{metric}_split_score"
            local_key = f"local_{metric}_score_sensor_p50"
            axis.bar(x - width / 2, [row[split_key] for row in rows], width, label="Best SPLIT", color="#4472C4")
            axis.bar(x + width / 2, [row[local_key] for row in rows], width, label="LOCAL + prep p50", color="#70AD47")
            axis.set_title(f"{budget} ms budget")
            axis.set_xticks(x, labels, rotation=24, ha="right")
            axis.grid(axis="y", alpha=0.25)
            axis.set_xlabel("Network profile")
        axes[0].set_ylabel(title)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
        fig.subplots_adjust(top=0.82, bottom=0.25, left=0.07, right=0.99, wspace=0.15)
        for suffix in ("png", "pdf"):
            name = f"{filename}.{suffix}"
            fig.savefig(output / name, dpi=220 if suffix == "png" else None)
            generated.append(name)
        plt.close(fig)
    return generated


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), f"create-only output exists: {output}")
    split_manifest = verify_manifest(SPLIT_ROOT, "artifact_manifest.json")
    local_manifest = verify_manifest(LOCAL_ROOT, "artifact_manifest.json")
    require((LOCAL_ROOT / "SPLITFUSION_LOCAL_ACTION_BASELINE_COMPLETE").is_file(), "LOCAL terminal absent")
    qualification = load_json(LOCAL_ROOT / "qualification.json")
    require(qualification.get("status") == "COMPLETE", "LOCAL qualification is incomplete")
    prep = preparation_summary()
    split = compact_split_rows(split_rows())
    local = local_rows(prep, qualification)
    require(len(split) == 288 and len(local) == 4, "policy candidate inventory drift")
    best = best_table(split, local)
    require(len(best) == 12, "profile-budget table drift")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "split_latest_action_profile.csv", split)
    write_csv(output / "local_profile_measurement_and_sensitivity.csv", local)
    write_csv(output / "best_split_vs_local_by_profile_budget.csv", best)
    atomic_text(output / "SCIENTIFIC_SYNTHESIS.md", render_report(prep, local, best))
    figures = plot(output, best)
    document = {
        "schema": "scenesense.splitfusion.map_freshness_policy_synthesis.v3",
        "status": "COMPLETE",
        "budgets_ms": list(BUDGETS),
        "split_queue_policy": "LATEST_ONLY_NO_EXPIRY",
        "split_candidates": 288,
        "local_candidates": 4,
        "local_action_status": "MEASURED_DESKTOP_PROXY_POLICY_CANDIDATE_NOT_VEHICLE_HARDWARE_QUALIFIED",
        "preparation_sensitivity": prep,
        "all_local_sensor_p50_freshness_above_best_split": all(row["local_minus_best_raw_split"] > 0 for row in best),
        "policy_conclusion": "INCLUDE_LOCAL_PROXY_ONLY_WITH_VEHICLE_COMPUTE_COST_AND_AVAILABILITY_STATE",
        "provenance": {
            "split_manifest_sha256": sha256(SPLIT_ROOT / "artifact_manifest.json"),
            "split_artifacts_verified": len(split_manifest),
            "local_manifest_sha256": sha256(LOCAL_ROOT / "artifact_manifest.json"),
            "local_artifacts_verified": len(local_manifest),
            "local_qualification_sha256": sha256(LOCAL_ROOT / "qualification.json"),
        },
        "presentation_artifacts": figures,
    }
    atomic_json(output / "policy_synthesis.json", document)
    primary = [
        "split_latest_action_profile.csv",
        "local_profile_measurement_and_sensitivity.csv",
        "best_split_vs_local_by_profile_budget.csv",
        "SCIENTIFIC_SYNTHESIS.md",
        "policy_synthesis.json",
        *figures,
    ]
    hashes = {name: sha256(output / name) for name in primary}
    atomic_json(output / "artifact_manifest.json", {"schema": f"{document['schema']}.artifacts", "sha256": hashes})
    atomic_text(output / TERMINAL, TERMINAL + "\n")
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    print(TERMINAL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
