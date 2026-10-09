"""Read-only original Run-5 environment audit; no model imports or inference.

Source inspection is restricted to named files. SQLite is opened mode=ro.
The only output is a create-only JSON report, or verification of that report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

PREREG_SHA = "2270baa0cf025b5e64a85644a28b8fd98e1f9004b11dcd6cb39fb5f61c5d18cf"
R5 = "rl_agent/splitfusion_hybrid_sac_run5_v1/"
CV = "rl_agent/ue_production_transport_model_v2/collector_v1.py"
GRID = "experiments/splitfusion_hybrid_sac_quality_grid_v1/20260918_exact_continuous_q_grid_a1b_full/"
PROBE = "experiments/splitfusion_quality_feedback_probe_v1/20260916_action50_favorable_adverse_retry4/cells/"
CRITICAL = [R5 + "run5_collector.py", R5 + "run5_channel.py", R5 + "run5_training.py",
            CV, "rl_agent/ue_production_transport_model_v2/scene_source.py",
            "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
            "rl_agent/splitfusion_hybrid_sac_run4_v1/environment.py"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def stats(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("empty/nonfinite statistical population")
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "maximum": float(x.max()), "minimum": float(x.min())}


def audit(main, original):
    sources = {}

    def bind(path):
        path = Path(path).resolve()
        digest = sha(path)
        sources[str(path)] = digest
        return path

    def read(path):
        return json.loads(bind(path).read_text())

    prereg_path = original / (R5 + "RUN5_TRAINING_PREREGISTRATION.json")
    prereg = read(prereg_path)
    if sha(prereg_path) != PREREG_SHA:
        raise ValueError("not the original trained Run-5 registration")
    pins = {}
    for rel in CRITICAL:
        got = sha(bind(original / rel))
        expected = prereg["source_sha256"][rel]
        if got != expected:
            raise ValueError(f"original registered source drift: {rel}")
        pins[rel] = {"sha256": got, "matches_original_training_preregistration": True}

    anchors = {
        CV: ["residual = total - (first - enqueued)",
             "composed_ns = (context.retained_residual_ns + send_span_ns",
             "+ transport_ns + collector.actor_reserve_ns)",
             "episode_boundary=contract.EpisodeBoundary.CONTINUES)",
             "self._scene_rng.randrange(self.catalog.scene_count)",
             "self._held_scene_rng.randrange(self.catalog.scene_count)"],
        R5 + "run5_collector.py": ["class Run5ModeledCollectorV1(CV.RealModeledTransitionCollectorV1):",
                                   "self._channel.advance(J.SUPPORTED_DURATION_TENSORS)"],
        R5 + "run5_channel.py": ["probabilities = self._kernel.probabilities(self._mcs, self._snr)"],
    }
    code_evidence = []
    for rel, snippets in anchors.items():
        lines = (original / rel).read_text().splitlines()
        for snippet in snippets:
            matching = [i + 1 for i, line in enumerate(lines) if snippet in line]
            if len(matching) != 1:
                raise ValueError(f"ambiguous/missing source anchor: {rel}: {snippet}")
            code_evidence.append({"source": str((original / rel).resolve()),
                                  "line": matching[0], "text": snippet})

    # Exact same source-row acceptance as the registered residual loader.
    pools = {k: [] for k in ("residual_ms", "gt_wait_ms", "gt_scoring_ms",
                             "residual_without_gt_stages_ms", "removed_first_send_to_enqueue_ms")}
    cells = {}
    retained_rows = []
    for cell in ("a50__favorable_stable", "a50__adverse_stable"):
        path = bind(main / PROBE / cell / "quality_feedback_timing_join.csv")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames)
            rows = list(reader)
        accepted = 0
        for index, row in enumerate(rows):
            try:
                a, b, e = [float(row[key]) for key in (
                    "model_prepare_start_to_ue_receive_ms",
                    "first_feature_datagram_send_to_ue_receive_ms",
                    "evaluation_enqueued_to_ue_receive_ms")]
            except (KeyError, ValueError, TypeError):
                continue
            raw = a - (b - e)
            if not math.isfinite(raw) or raw <= 0:
                continue
            # The runtime stores rounded integer nanoseconds.
            residual = int(round(raw * 1e6)) / 1e6
            started, completed = [float(row[key]) for key in (
                "evaluation_started_to_ue_receive_ms", "evaluation_completed_to_ue_receive_ms")]
            values = (residual, e - started, started - completed,
                      residual - (e - completed), b - e)
            for key, value in zip(pools, values):
                pools[key].append(value)
            retained_rows.append({"cell": cell, "csv_row_index": index,
                                  "frame_id": int(row["frame_id"]), "residual_ms": residual})
            accepted += 1
        cells[cell] = {"total_rows": len(rows), "accepted": accepted,
                       "columns": columns,
                       "complete_reassembly_timestamp_column_present":
                           any("reassembly" in key for key in columns)}
    pooled = {key: stats(value) for key, value in pools.items()}
    pooled["residual_alone_over_170_ms"] = sum(x > 170 for x in pools["residual_ms"])
    pooled["without_gt_alone_over_170_ms"] = sum(x > 170 for x in pools["residual_without_gt_stages_ms"])
    pooled["gt_wait_plus_scoring_fraction_of_summed_residual"] = (
        sum(pools["gt_wait_ms"] + pools["gt_scoring_ms"]) / sum(pools["residual_ms"]))
    actor_timing = read(main / "rl_agent/splitfusion_hybrid_sac_run4_v1/sealed_actor_timing_v1/ACTOR_TIMING_QUALIFICATION.json")
    transport_rel = "rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json"
    model = read(main / transport_rel)
    if sha(main / transport_rel) != prereg["source_sha256"][transport_rel]:
        raise ValueError("registered transport artifact differs")
    boundary_source = bind(original / "rl_agent/ue_production_queue_capture_v1/contract.py")

    # Demonstrate the boundary identity, not a latency simulation or repair.
    synthetic = {"prepare": 0, "first_send": 11, "last_send": 12,
                 "reassembly": 42, "enqueue": 94, "feedback": 102}
    residual_demo = (synthetic["feedback"] - synthetic["prepare"]) - (
        (synthetic["feedback"] - synthetic["first_send"]) -
        (synthetic["feedback"] - synthetic["enqueue"]))
    rebuilt_demo = residual_demo + (synthetic["last_send"] - synthetic["first_send"]) + (
        synthetic["reassembly"] - synthetic["last_send"])
    if synthetic["feedback"] - rebuilt_demo != synthetic["enqueue"] - synthetic["reassembly"]:
        raise AssertionError("boundary algebra failed")

    manifest = read(main / GRID / "run_manifest.json")
    database = bind(main / GRID / "quality_rows.sqlite3")
    if sha(database) != manifest["artifact_sha256"]["quality_rows.sqlite3"]:
        raise ValueError("registered quality database hash differs")
    sidecar = read(main / "experiments/splitfusion_hybrid_sac_p40_sidecar_v1/20260921_exact_grid_corrected_p40_v1/corrected_p40_sidecar.json")
    p40 = {f"{r['episode_id']}|{r['sample_id']}|{r['frame_id']}": r["corrected_p40"]
           for r in sidecar["records"] if r["grid_split"] == "fit"}
    scenes = {}
    conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        for (blob,) in conn.execute("SELECT row_json FROM quality_rows WHERE grid_split='fit'"):
            row = json.loads(blob)
            key = f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"
            entry = scenes.setdefault(key, {"camera_si": row.get("camera_si"), "grid": {}})
            entry["grid"][(int(row["mode_id"]), int(row["q_e4"]))] = row.get("q_perc")
    finally:
        conn.close()
    eligible = {key: row for key, row in scenes.items()
                if row["camera_si"] is not None and p40.get(key) is not None
                and len(row["grid"]) == 132 and all(x is not None for x in row["grid"].values())}
    quality_spec = read(main / GRID / "reward_spec.json")
    # These source files define arithmetic/domain differences, never executed here.
    supplementary = [
        "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/quality.py",
        "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/executor.py",
        "rl_agent/splitfusion_hybrid_sac_v1/protocol_v2_contract.py",
        "rl_agent/splitfusion_hybrid_sac_run4_v1/scientific_basis.py",
        "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py",
        "rl_agent/splitfusion_run5_live_avo_v1/quality_adapter.py",
        "rl_agent/splitfusion_run5_live_avo_v1/edge_quality.py",
        "rl_agent/splitfusion_run5_live_avo_v1/actor_export/ACTOR_BINDING.json",
    ]
    for rel in supplementary:
        bind(main / rel)
    bind(Path(__file__))
    findings = [
        {"id": "LATENCY_BOUNDARY_GAP", "status": "CONFIRMED_CODE_DEFECT",
         "finding": "The removed first-send→evaluation-enqueue interval is larger than the declared uplink replacement. Send-span + transport restores only first-send→complete-reassembly; complete-reassembly→evaluation-enqueue is absent.",
         "unmeasured": "Old probe has no complete-reassembly column; missing queue/decode/tail/publication quantity cannot be recovered from these CSVs. Do not substitute today's edge timing or claim net bias direction."},
        {"id": "GT_INCLUDED_ONCE", "status": "CONFIRMED_NOT_DOUBLE_COUNTING",
         "finding": "GT wait + scoring + ACK path occur once in the retained enqueue→receipt residual. Inclusion matches the current GT-inclusive deadline. The issue is old residual calibration and action independence, not permission to remove GT."},
        {"id": "UNCONDITIONAL_OLD_RESIDUAL", "status": "CONFIRMED_MODEL_ASSUMPTION",
         "finding": "One pooled action-50 residual is sampled independently of scene, compression family, q, MCS and backlog. It omits mode-specific front/edge compute response and preserves old GT delays. 66/555 draws force timeout before any added transport."},
        {"id": "QUALITY_ARITHMETIC", "status": "CONFIRMED_NO_MEAN_MEDIAN_FORMULA_DRIFT",
         "finding": "Both grid and live call evaluate_exact_quality / _derive_exact_quality. Per-class matched error is median, utilities sqrt(recall*exp(-median/tau)), weighted geometric Q_loc and normalized weighted geometric Q_seg; Q_perc=Q_loc*(.7+.3*Q_seg). Missing classes are excluded as specified. The scalar RL reward uses170ms/.25, not inert grid reward_spec200ms/zero-latency metadata."},
        {"id": "QUALITY_INPUT_DOMAIN", "status": "CONFIRMED_DECLARED_DOMAIN_MISMATCH",
         "finding": "The original quality grid uses the frozen offline v010 segmentation/ignore/eligibility pipeline and filtered offline vehicle scorer. Current live explicitly declares raw corrected CARLA masks NOT v010 equivalence; person AVO is restored, vehicle live matcher remains unchanged. Equal arithmetic does not establish equivalent quality labels or previous-Q distribution."},
        {"id": "NO_RANDOM_SCENE_ROUTE_REPLAY", "status": "CONFIRMED_MODEL_ASSUMPTION",
         "finding": "Training samples reward and held FIT scenes independently and uniformly from eligible scenes. Two100ms tensor slots are modeled; no live CARLA/OAI process executes, and this is not chronological Route-B10FPS replay."},
        {"id": "PREVIOUS_QUALITY_AND_RESETS", "status": "CONFIRMED_MODEL_LIVE_COVERAGE_GAP",
         "finding": "Previous success carries exact prior Q and latency; timeout/failure encodes both0 with prev_present1/prev_success0; genesis allprevious0 with prev_present0. Excluded evaluator events require fresh session. Training has one continuing session and only all-anchor-defined-Q scenes, hence no undefined-GT rollover examples."},
        {"id": "SNR_CHANNEL_EFFECT", "status": "CONFIRMED_REGISTERED_MODEL_ASSUMPTION",
         "finding": "Immediate transport depends on bytes/backlog/MCS only. Current SNR tilts successor MCS; SNR/MCS path has no action parameter. Action influences quality/payload and successor backlog, not the channel process. Favorable/adverse-stable kernel transfer is registered unvalidated. Model channel advances exactly2ticks/decision rather than measured live elapsed cadence."},
        {"id": "RANKING_CAUSALITY", "status": "UNRESOLVED",
         "finding": "These confirmed boundary/domain defects and assumptions undermine live calibration and could change action values. They do not prove a Bellman critic defect or explain a particular quality/latency deficit causally. Current live preprocessing/GT faults must be resolved before considering retraining or reward changes."},
    ]
    result = {
        "schema": "splitfusion.original_run5_environment_audit.v1",
        "scope": "READ_ONLY_ORIGINAL_22D_RUN5_NOT_RUN5B_NO_INFERENCE_OR_SERVICES",
        "original_preregistration_sha256": PREREG_SHA,
        "registered_sources": pins, "code_evidence": code_evidence,
        "latency": {"formula": "R_old + .513047ns*wire_bytes + T_v2b(B,wire,MCS) +1ms_actor_reserve",
                    "residual_identity": "prepare→first_send + evaluation_enqueue→UE_feedback_receipt",
                    "transport_boundary": model.get("boundary"),
                    "declared_replacement": model.get("replaces_288_component"),
                    "boundary_source": str(boundary_source),
                    "actor_reserve_ns": actor_timing["modeled_actor_reserve_ns"],
                    "cells": cells, "statistics": pooled,
                    "quantile_method": "numpy linear, not nearest-rank; actor reserve not added to residual statistics",
                    "accepted_pool_identity_sha256": hashlib.sha256(json.dumps(retained_rows, sort_keys=True).encode()).hexdigest(),
                    "symbolic_boundary_check_passed": True},
        "quality": {"fit_scenes_total": len(scenes), "eligible_fit_scenes": len(eligible),
                    "eligible_scene_anchor_rows": sum(len(row["grid"]) for row in eligible.values()),
                    "anchor_q_perc_unweighted": stats([q for row in eligible.values() for q in row["grid"].values()]),
                    "anchor_statistics_are_not_training_policy_performance": True,
                    "matched_error_statistic": "per-class median of matched XY errors, not mean",
                    "quality_spec_sha256": sha(main / GRID / "reward_spec.json"),
                    "quality_parameters": {k: quality_spec[k] for k in (
                        "tau_person_m", "tau_vehicle_m", "w_loc_person", "w_loc_vehicle",
                        "seg_reference_person_iou", "seg_reference_vehicle_iou", "segmentation_modulation_beta")},
                    "training_undefined_quality_scene_excluded": True,
                    "off_anchor_quality": "linear interpolation between same-scene/same-mode registered q anchors; exploratory"},
        "registered_design": {k: prereg["design"][k] for k in ("state", "reward", "channel", "scenes")},
        "findings": findings,
        "source_sha256": sources,
        "source_integrity_verified_after_analysis": all(sha(p) == h for p, h in sources.items()),
    }
    if not result["source_integrity_verified_after_analysis"]:
        raise ValueError("source changed during read-only audit")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, default=Path.cwd())
    parser.add_argument("--original-run5-root", type=Path, default=Path("../abiodun_run5_wt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = audit(args.main_root.resolve(), args.original_run5_root.resolve())
    if args.verify_only:
        if json.loads(args.output.read_text()) != result:
            raise ValueError("audit result does not reproduce")
        print("ENVIRONMENT_AUDIT_REPRODUCED")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as f:
            json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
        print(json.dumps({"status": "ENVIRONMENT_AUDIT_COMPLETE", "output": str(args.output),
                          "sha256": sha(args.output), "findings": len(result["findings"])}))


if __name__ == "__main__":
    main()
