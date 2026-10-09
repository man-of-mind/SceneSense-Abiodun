"""Read-only Run-5 outcome/state audit; no actor inference or environment replay.

Run from repository root with ``python -m ...audit_outcomes --output DIR``.
All statistical bins are descriptive, not causal action comparisons. Source
files are hashed before/after, and the sealed capture hashes are checked.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STORY = ROOT / "experiments/splitfusion_run5_story_v1/20261002_v2"
RESULTS = ROOT / "experiments/splitfusion_run5_live_avo_results_v1/20261002_1000_frame_budget_v1"
CAPTURE = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261002T224700Z_run5_live_avo_capture_1000_v1"
CELL = CAPTURE / "capture/cells/a71__favorable_stable"
TRAIN = ROOT.parent / "abiodun_run5_wt/rl_agent/splitfusion_hybrid_sac_run5_v1/campaign_runs/run5_three_seed_10000_v1/seed_43"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    a = np.asarray(list(values), dtype=float)
    if not len(a):
        return {"n": 0}
    assert np.isfinite(a).all()
    qs = np.percentile(a, [0, 5, 10, 50, 90, 95, 99, 100], method="linear")
    return {"n": len(a), "mean": float(a.mean()), **dict(zip(
        ("min", "p05", "p10", "p50", "p90", "p95", "p99", "max"), map(float, qs)))}


def population(rows):
    result = {"decisions": len(rows), "timely": sum(r["timely"] for r in rows),
              "terminal_counts": dict(Counter(r["terminal"] for r in rows)),
              "latency_ms": stats(r["latency_ms"] for r in rows),
              "payload_bytes": stats(r["payload_bytes"] for r in rows),
              "q_e4": stats(r["q_e4"] for r in rows)}
    for label in ("person", "vehicle"):
        errors = [e for r in rows for e in r[label]["errors_m"]]
        counts = {key: sum(r[label][key] for r in rows)
                  for key in ("eligible_gt", "tp", "fp", "fn")}
        tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
        result[label] = {**counts,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "matched_xy_errors_m": stats(errors),
            "within_frame_median_errors_m": stats(r[label]["xy_median_m"] for r in rows
                                                    if r[label]["xy_median_m"] is not None),
            "matched_error_threshold_exceedance": {
                str(th): {"count": sum(e > th for e in errors), "denominator": len(errors),
                          "fraction": sum(e > th for e in errors) / len(errors) if errors else None}
                for th in (1., 1.2, 2., 2.5)}}
    return result


def prior_stats(features):
    a = np.asarray(features, dtype=float)
    success = a[:, 20] == 1
    return {"n": len(a), "genesis": int((a[:, 19] == 0).sum()),
        "previous_success": int(success.sum()), "previous_quality_all": stats(a[:, 17]),
        "previous_quality_success_only": stats(a[success, 17]),
        "previous_latency_success_only_ms": stats(a[success, 18] * 170),
        "mode_counts": dict(Counter(int(np.argmax(row[4:16])) for row in a if row[19] == 1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    targets = [args.output / f"outcomes_{name}" for name in
               ("audit.json", "bins.json", "policy_features.csv", "sha256.json")]
    if any(path.exists() for path in targets):
        raise RuntimeError("create-only audit outputs already exist")
    sources = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            assert digest == expected, f"hash mismatch: {path}"
        sources[str(path)] = digest
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    provenance = read(STORY / "PROVENANCE.json")["source_inputs_sha256"]
    joined_path = bind(STORY / "JOINED_POLICY_ROWS.json")
    rows = json.loads(joined_path.read_text())
    canonical_path = RESULTS / "POLICY_ROWS.json"
    canonical = read(canonical_path, provenance[str(canonical_path)])
    assert len(rows) == len(canonical) == 497
    for joined, old in zip(rows, canonical):
        assert all(joined[k] == v for k, v in old.items())
    capture_manifest = read(CAPTURE / "RUN5_POSTFLIGHT_INPUTS.json",
        "4b9c463f1cb48f43b05551f87200fc1338ab08965bc88faeef5a7532c9e965ef")["files"]
    ue_path = CELL / "run4_phase6/PHASE6_UE_EVIDENCE.json"
    ue = read(ue_path, capture_manifest[str(ue_path.relative_to(CAPTURE))])
    policy = [r for r in ue["decisions"] if r["kind"] == "POLICY_DECISION"]
    by_frame = {r["frame_id"]: r for r in policy}
    assert len(by_frame) == len(rows)
    resolutions = {(r["identity"]["session_uuid"], r["identity"]["decision_seq"]): r
                   for r in ue["resolutions"]}
    assert len(resolutions) == 497
    bookkeeping_errors, resets, features = [], [], []
    previous = None
    for r in rows:
        d = by_frame[r["frame_id"]]
        key = (r["session_uuid"], r["decision_seq"])
        assert d["decision_session_uuid"] == r["session_uuid"]
        assert (d["mode_id"], d["q_e4"]) == (r["mode_id"], r["q_e4"])
        f = d["features"]
        assert len(f) == 22
        expected = np.zeros(17)
        if previous is not None and previous[0] == key[0]:
            prev = previous[1]
            expected[prev["action"]["mode_id"]] = 1
            expected[12] = prev["action"]["q_e4"] / 9800
            expected[15] = 1
            if prev["terminal"] == "SUCCESS":
                expected[13:15] = [prev["q_perc"], prev["latency_ms"] / 170]
                expected[16] = 1
        elif previous is not None:
            resets.append({"frame_id": r["frame_id"], "elapsed_s": r["action_elapsed_raw_s"],
                           "preceding_terminal": previous[1]["terminal"],
                           "new_session": key[0]})
        if not np.array_equal(np.asarray(f[4:21]), expected):
            bookkeeping_errors.append(r["frame_id"])
        previous = (key[0], resolutions[key])
        features.append(f)
    assert not bookkeeping_errors, bookkeeping_errors
    assert len(resets) == len(ue["sessions"]) - 1 == 42
    assert all(r["preceding_terminal"] == "EVALUATOR_FAULT" for r in resets)
    holds = [r for r in ue["frames"] if not r["reward_requested"]]
    group = Counter((r["session_uuid"], r["decision_seq"]) for r in ue["frames"])
    hold_identity_errors = []
    for h in holds:
        source = resolutions[(h["session_uuid"], h["decision_seq"])]["action"]
        if (h["mode_id"], h["q_e4"]) != (source["mode_id"], source["q_e4"]):
            hold_identity_errors.append(h["frame_id"])

    checkpoint = TRAIN / "checkpoints/checkpoint_010000"
    manifest = read(checkpoint / "manifest.json")
    event = read(checkpoint / "event.json", manifest["files"]["event.json"]["sha256"])
    ledger_path = bind(TRAIN / "decisions.jsonl")
    training = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert len(training) == event["decision_count"] == 40288
    assert [r["digest"] for r in training] == [r["digest"] for r in event["ledger"]]
    # No model/replay reconstruction: these are exact prior fields from the
    # saved outcome ledger, with the collector's single-session semantics.
    train_features = [np.zeros(22)]
    for r in training[:-1]:
        f = np.zeros(22)
        f[4 + r["mode_id"]] = 1
        f[16] = r["q_e4"] / 9800
        f[19] = 1
        if r["terminal"] == "SUCCESS":
            f[17:19] = [r["q_perc"], r["latency_ms"] / 170]
            f[20] = 1
        train_features.append(f)

    from rl_agent.ue_production_transport_model_v2 import scene_source as ss
    from rl_agent.ue_production_queue_capture_v1 import contract as capture_contract
    bind(Path(ss.__file__))
    bind(ROOT / capture_contract.PAYLOAD_AUTHORITY_RELPATH)
    bind(ROOT / ss.P40_SIDECAR_RELPATH, ss.P40_SIDECAR_SHA256)
    catalog = ss.FitSceneCatalog(ROOT)
    support = np.array([catalog.scene_descriptors(key) for key in catalog.keys])
    live_scene = np.array([[r["camera_si"], r["radar_p40"]] for r in rows])
    outside = (live_scene < support.min(axis=0)) | (live_scene > support.max(axis=0))
    support_report = {}
    for index, label in enumerate(("camera_si", "radar_p40")):
        support_report[label] = {"eligible_training_catalog": stats(support[:, index]),
            "live_policy_decisions": stats(live_scene[:, index]),
            "outside_empirical_training_minmax": int(outside[:, index].sum())}

    # GT population only: missing matched-object IDs forbid error-vs-distance.
    distances, eligibility_per_frame, people_per_frame = [], {}, []
    selected_avo = [name for name in capture_manifest if
                    "/gt_scratch_preserved/" in name and name.endswith(".avo.json")]
    assert len(selected_avo) == 497
    for name in selected_avo:
        gt = read(CAPTURE / name, capture_manifest[name])
        frame_id = gt["source_frame_id"]
        assert frame_id in by_frame
        raw = {str(r["gt_actor_id"]): r for r in gt["raw_pedestrian_rows"]}
        eligible = [r for r in gt["avo"]["qualified"] if r["avo"] >= .65]
        eligibility_per_frame[frame_id] = len(eligible)
        people_per_frame.append(len(raw))
        distances.extend(raw[str(r["gt_actor_id"])]["gt_distance_m"] for r in eligible)
    assert sum(eligibility_per_frame.values()) == 697
    assert all(eligibility_per_frame[r["frame_id"]] == r["person"]["eligible_gt"] for r in rows)

    bins = {}
    for label, key_fn in {
        "mode": lambda r: str(r["mode_id"]),
        "q_e4_2000_bins": lambda r: str(2000 * (r["q_e4"] // 2000)),
        "payload_decimal_100kb_bins": lambda r: str(100000 * (r["payload_bytes"] // 100000)),
        "snr_4db_bins": lambda r: str(4 * int(r["snr_db"] // 4)),
        "wall_elapsed_10s_bins": lambda r: str(10 * int(r["action_elapsed_raw_s"] // 10)),
        "previous_present": lambda r: str(int(by_frame[r["frame_id"]]["features"][19])),
        "scene_outside_empirical_support": lambda r: str(bool(outside[rows.index(r)].any())),
    }.items():
        groups = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r)
        bins[label] = {k: population(v) for k, v in sorted(groups.items())}

    baseline_path = ROOT / "experiments/splitfusion_fcos_person_p025_calibration_v1/validation_confirmation.json"
    baseline = read(baseline_path)["validation"]["avo_gte_0_65"]
    report = {"schema": "run5_readonly_outcome_audit_v1", "overall": population(rows),
        "state_bookkeeping": {"errors": bookkeeping_errors, "session_count": len(ue["sessions"]),
            "reset_count": len(resets), "resets": resets, "hold_count": len(holds),
            "hold_action_identity_errors": hold_identity_errors,
            "tensors_per_decision": dict(Counter(group.values())),
            "fallback_count": len(ue["fallback_log"]), "live_prior": prior_stats(features),
            "training_seed43_prior": prior_stats(train_features),
            "training_final2000_prior": prior_stats(train_features[-2000:]),
            "training_action_mode_counts": dict(Counter(r["mode_id"] for r in training)),
            "training_final2000_action_mode_counts": dict(Counter(r["mode_id"] for r in training[-2000:]))},
        "scene_support": support_report,
        "gt_population": {"retained_avo_frames": len(selected_avo),
            "raw_people_per_frame": stats(people_per_frame),
            "eligible_actor_frame_distance_m": stats(distances),
            "eligible_distance_bins": {f"{lo}_{lo+10}m": sum(lo <= d < lo+10 for d in distances)
                                       for lo in (0, 10, 20, 30)},
            "matched_error_distance_unavailable": True},
        "historical_uncompressed_fcos_p025": baseline,
        "source_verification": {"live_inputs_match_seals": True,
            "training_event_matches_checkpoint_manifest": True,
            "training_ledger_digests_match_event": True,
            "source_hashes_before": sources},
        "limitations": [
            "Repeated actor-frame observations; not independent unique objects.",
            "Matched XY errors are bounded by the 3m association gate; misses excluded and reported via recall.",
            "Vehicle live GT does not include complete static-vehicle reconciliation.",
            "Live Q_perc has provisional raw-semantic mask semantics unlike training; distributions are not accuracy-equivalent.",
            "Binned action/channel/time outcomes are associations, not controlled action effects.",
            "Training prior fields derived only from saved outcomes and single-session collector; no state replay or inference.",
            "Catalog support is empirical min/max, not a claim of contract admission failure.",
            "Historical quality baseline has different scenes/action pipeline; not an actor-only comparison.",
            "Predictions/matched target identities absent; current localization error cannot be stratified by object distance."]}
    for name, digest in sources.items():
        assert sha(name) == digest, f"source modified: {name}"
    report["source_verification"]["source_hashes_after_identical"] = True
    targets[0].write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    targets[1].write_text(json.dumps(bins, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with targets[2].open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "session_uuid", "decision_seq", *[f"feature_{i}" for i in range(22)]])
        for row, fs in zip(rows, features):
            writer.writerow([row["frame_id"], row["session_uuid"], row["decision_seq"], *fs])
    outputs = {path.name: sha(path) for path in targets[:3]}
    outputs["audit_script"] = sha(__file__)
    targets[3].write_text(json.dumps(outputs, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": "READ_ONLY_AUDIT_COMPLETE", "outputs": outputs,
                      "verified_inputs": len(sources), "bookkeeping_errors": bookkeeping_errors}))


if __name__ == "__main__":
    main()
