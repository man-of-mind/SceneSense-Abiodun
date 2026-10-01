"""Strictly post-run report for one single-host Run-4B 300-frame attempt.

Reads only durable evidence written by the finished run and writes create-only
outputs to a new directory.  It never touches the live harness, so nothing it
computes can reach the actor, the ACK path or the hold controller.

Localization, segmentation and Q_perc use the registered Phase-6 functions
(``run4_live_wire_v2.live_measurement`` and ``offline_quality_grid.quality
.evaluate_exact_quality`` with the pinned Run-4 quality spec).  The population
is every policy decision; missing predictions or GT are counted, never dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "scenesense.splitfusion.run4b_single_host.postrun_report.v1"
DEADLINE_MS = 170.0
STARTUP_FRAMES = 30


def pct(values: Sequence[float], q: float) -> Optional[float]:
    """Nearest-rank percentile."""
    data = sorted(float(v) for v in values)
    if not data:
        return None
    rank = max(1, math.ceil(q / 100.0 * len(data)))
    return data[rank - 1]


def summary(values: Iterable[Optional[float]]) -> dict[str, Any]:
    data = [float(v) for v in values if v is not None]
    return {"n": len(data), "p50": pct(data, 50), "p90": pct(data, 90),
            "p95": pct(data, 95), "p99": pct(data, 99),
            "max": max(data) if data else None,
            "mean": (sum(data) / len(data)) if data else None}


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ns(record: Mapping[str, Any], name: str) -> Optional[int]:
    value = record.get(name)
    if isinstance(value, Mapping):
        value = value.get("ns")
    return None if value is None else int(value)


def _ms(later: Optional[float], earlier: Optional[float], scale: float) -> Optional[float]:
    if later is None or earlier is None:
        return None
    return (float(later) - float(earlier)) / scale


def frame_table(ue: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per transmitted frame in transmit order (decision/hold/fallback)."""
    rows = []
    for order, record in enumerate(sorted(ue.get("decisions") or (),
                                          key=lambda r: int(r["tensor_seq"]))):
        stages = dict(record.get("stages") or {})
        rows.append({"frame_order": order + 1, "frame_id": int(record["frame_id"]),
                     "tensor_seq": int(record["tensor_seq"]), "kind": record["kind"],
                     "mode_id": record.get("mode_id"), "q_e4": record.get("q_e4"),
                     "action_open_raw_ns": _ns(record, "action_open"),
                     "state_commit_raw_ns": _ns(record, "state_commit"),
                     "features": record.get("features"),
                     "fallback_reasons": record.get("fallback_reasons") or [],
                     "stages": stages})
    return rows


def invert_features(features: Sequence[float]) -> dict[str, Any]:
    from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4B

    from .ue_v1 import SCALING

    return {"prior_ul_mcs": int(round(features[2] * (C4B.UL_MCS_MAX - C4B.UL_MCS_MIN)
                                      + C4B.UL_MCS_MIN)),
            "pre_action_rlc_backlog_bytes": int(round(math.expm1(
                features[3] * SCALING.backlog_log1p_scale))),
            "camera_si": features[0] * SCALING.camera_si_scale + SCALING.camera_si_center,
            "radar_p40": features[1]}


def offline_quality(*, decisions: Sequence[Mapping[str, Any]], prediction_dir: Path,
                    gt_dir: Path, match_distance_m: float) -> dict[str, Any]:
    """Registered measurement and Q_perc per decision; misses are counted."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
    from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid import quality as Q
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence

    spec = W.load_run4_quality_spec(ROOT)
    by_frame = {}
    for path in sorted(Path(prediction_dir).glob("*.records.json")):
        document = _json(path)
        by_frame[(document["session_uuid"], int(document["decision_seq"]))] = (path, document)
    rows = []
    for decision in decisions:
        key = (decision["session_uuid"], int(decision["decision_seq"]))
        row = {**decision, "prediction": "MISSING", "ground_truth": "MISSING",
               "q_perc": None}
        found = by_frame.get(key)
        if found is not None:
            import numpy as np

            path, document = found
            mask = np.load(path.with_name(document["mask_file"]), allow_pickle=False)
            row["prediction"] = "PRESENT"
            row["tail_output_sha256"] = document["tail_output_sha256"]
            try:
                gt = gt_evidence.read_ground_truth(
                    gt_dir, expected_identity=document["gt_identity"], timeout_s=0.01)
            except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                row["ground_truth"] = f"MISSING:{type(exc).__name__}"
            else:
                row["ground_truth"] = "PRESENT"
                measurement = W.live_measurement(
                    frame_id=int(document["frame_id"]), predicted_mask=mask,
                    ground_truth_mask=gt["semantic"], predictions=document["records"],
                    ground_truth_objects=list(gt["objects"]),
                    match_distance_m=float(match_distance_m))
                row["measurement"] = measurement
                exact = Q.evaluate_exact_quality(spec, measurement)
                row["q_perc"] = None if exact is None else float(exact.q_perc)
        rows.append(row)

    def iou(name: str) -> list[float]:
        out = []
        for row in rows:
            m = row.get("measurement")
            if m and m[f"seg_{name}_union_pixels"] > 0:
                out.append(m[f"seg_{name}_intersection_pixels"]
                           / m[f"seg_{name}_union_pixels"])
        return out

    def loc(name: str) -> dict[str, Any]:
        measured = [r["measurement"] for r in rows if r.get("measurement")]
        tp = sum(m[f"loc_{name}_tp"] for m in measured)
        fp = sum(m[f"loc_{name}_fp"] for m in measured)
        fn = sum(m[f"loc_{name}_fn"] for m in measured)
        errors = [e for m in measured for e in m[f"loc_{name}_matched_xy_errors_m"]]
        return {"frames": len(measured), "tp": tp, "fp": fp, "fn": fn,
                "eligible_gt": sum(m[f"loc_{name}_eligible_gt"] for m in measured),
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "matched_xy_error_m": summary(errors)}

    qvals = [r["q_perc"] for r in rows if r["q_perc"] is not None]
    return {
        "population": "EVERY_POLICY_DECISION",
        "decisions": len(rows),
        "with_prediction": sum(r["prediction"] == "PRESENT" for r in rows),
        "with_ground_truth": sum(r["ground_truth"] == "PRESENT" for r in rows),
        "missing_prediction": sum(r["prediction"] == "MISSING" for r in rows),
        "missing_ground_truth": sum(r["ground_truth"] != "PRESENT" for r in rows),
        "q_perc_defined": len(qvals),
        "q_perc_undefined_no_eligible_gt": sum(
            1 for r in rows if r.get("measurement") and r["q_perc"] is None),
        "q_perc": summary(qvals),
        "segmentation_iou": {name: summary(iou(name)) for name in ("vehicle", "person")},
        "localization": {name: loc(name) for name in ("vehicle", "person")},
        "rows": [{k: v for k, v in r.items() if k != "measurement"} for r in rows],
    }


def gpu_trace(path: Path, start_wall_s: float, end_wall_s: float) -> dict[str, Any]:
    """Passive nvidia-smi CSV restricted to the live window."""
    from datetime import datetime

    rows = _csv(path)
    if not rows:
        return {"samples": 0}
    keys = {name.strip(): name for name in rows[0]}

    def value(row, prefix, unit):
        for name, raw in keys.items():
            if name.startswith(prefix):
                text = str(row[raw]).replace(unit, "").strip()
                try:
                    return float(text)
                except ValueError:
                    return None
        return None

    samples = []
    for row in rows:
        stamp = str(row[keys["timestamp"]]).strip()
        try:
            wall = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        if start_wall_s <= wall <= end_wall_s:
            samples.append({"util": value(row, "utilization.gpu", "%"),
                            "sm": value(row, "clocks.current.sm", "MHz")
                            if any(n.startswith("clocks.current.sm") for n in keys)
                            else value(row, "clocks.sm", "MHz"),
                            "power": value(row, "power.draw", "W"),
                            "mem": value(row, "memory.used", "MiB"),
                            "pstate": str(row[keys["pstate"]]).strip()
                            if "pstate" in keys else None})
    return {"samples": len(samples),
            "utilization_pct": summary(s["util"] for s in samples),
            "sm_clock_mhz": summary(s["sm"] for s in samples),
            "power_w": summary(s["power"] for s in samples),
            "memory_mib": summary(s["mem"] for s in samples),
            "pstates": dict(Counter(s["pstate"] for s in samples))}


def payload_summary(path: Path, frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-kind payload/datagram distributions from the collector's metrics."""
    rows = _csv(path)
    kind = {int(f["frame_id"]): f["kind"] for f in frames}
    out: dict[str, Any] = {"rows": len(rows)}
    columns = [c for c in (rows[0] if rows else {})
               if any(t in c for t in ("payload", "datagram", "chunk", "sfd4"))]
    for column in columns:
        for name in ("POLICY_DECISION", "POLICY_HOLD", "FALLBACK"):
            values = []
            for row in rows:
                try:
                    if kind.get(int(row["frame_id"])) == name:
                        values.append(float(row[column]))
                except (KeyError, TypeError, ValueError):
                    continue
            if values:
                out[f"{column}:{name}"] = summary(values)
    return out


def _overlap(a: Sequence[tuple[float, float]], b: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for start, end in a:
        for other_start, other_end in b:
            total += max(0.0, min(end, other_end) - max(start, other_start))
    return total


def build_report(attempt: Path, *, gpu_csv: Optional[Path] = None,
                 match_distance_m: Optional[float] = None) -> dict[str, Any]:
    attempt = Path(attempt)
    evidence = attempt / "run4_phase6"
    ue = _json(evidence / "PHASE6_UE_EVIDENCE.json")
    edge = _json(evidence / "edge_report.json")
    cell = _json(attempt / "CELL_RESULT.json") if (attempt / "CELL_RESULT.json").is_file() else {}
    frames = frame_table(ue)
    resolutions = list(ue.get("resolutions") or ())
    ack_rows = [r for r in ue.get("feedback_rows") or ()
                if r.get("kind") == "OPERATIONAL_ACK"]
    decisions_by_key = {}
    for res in resolutions:
        ident = res["identity"]
        decisions_by_key[(ident["session_uuid"], int(ident["decision_seq"]))] = res
    first_ack = {}
    for row in ack_rows:
        key = (row["session_uuid"], int(row["decision_seq"]))
        first_ack.setdefault(key, row)
    # raw<->wall offset from ACK receipts (both clocks are stamped together)
    offsets = sorted(int(round(r["receipt_wall_s"] * 1e9)) - int(r["receipt_raw_ns"])
                     for r in ack_rows if r.get("receipt_wall_s"))
    offset = offsets[len(offsets) // 2] if offsets else None

    def raw_to_wall_s(raw: Optional[int]) -> Optional[float]:
        return None if raw is None or offset is None else (raw + offset) / 1e9

    edge_ack = {(r["session_uuid"], int(r["decision_seq"])): r
                for r in edge.get("evaluations") or () if r.get("ack_sha256")}
    decision_frames = [f for f in frames if f["kind"] == "POLICY_DECISION"]
    # Exact join: the controller ledger maps every tensor to its decision.
    tensor_key = {int(f["tensor_seq"]): (f["session_uuid"], int(f["decision_seq"]))
                  for f in ue.get("frames") or () if f.get("reward_requested")}
    rows = []
    for frame in decision_frames:
        key = tensor_key.get(int(frame["tensor_seq"]))
        res = decisions_by_key.get(key) if key is not None else None
        ack = first_ack.get(key)
        eack = edge_ack.get(key)
        opened = frame["action_open_raw_ns"]
        st = frame["stages"]
        last_send_wall = raw_to_wall_s(st.get("last_packet_send_raw_ns"))
        row = {"frame_order": frame["frame_order"], "frame_id": frame["frame_id"],
               "tensor_seq": frame["tensor_seq"], "mode_id": frame["mode_id"],
               "q_e4": frame["q_e4"],
               "session_uuid": None if res is None else res["identity"]["session_uuid"],
               "decision_seq": None if res is None else res["identity"]["decision_seq"],
               "terminal": None if res is None else res["terminal"],
               "ack_latency_ms": (None if ack is None or opened is None
                                  else (int(ack["receipt_raw_ns"]) - opened) / 1e6),
               "ack_class": None if ack is None else ack["class"],
               "ue_prep_ms": _ms(opened, st.get("rgb_receipt_raw_ns"), 1e6),
               "state_actor_ms": _ms(st.get("input_7ch_start_raw_ns"), opened, 1e6),
               "input_7ch_ms": _ms(st.get("front_start_raw_ns"),
                                   st.get("input_7ch_start_raw_ns"), 1e6),
               "front_codec_ms": _ms(st.get("front_end_raw_ns"),
                                     st.get("front_start_raw_ns"), 1e6),
               "send_ms": _ms(st.get("last_packet_send_raw_ns"),
                              st.get("first_packet_send_raw_ns"), 1e6),
               "open_to_last_send_ms": _ms(st.get("last_packet_send_raw_ns"), opened, 1e6),
               "uplink_ms": (None if eack is None or last_send_wall is None else
                             (eack["reassembly_complete_wall_s"] - last_send_wall) * 1e3),
               "edge_queue_ms": (None if eack is None else
                                 (eack["compute_start_wall_s"]
                                  - eack["reassembly_complete_wall_s"]) * 1e3),
               "edge_tail_ms": (None if eack is None else
                                eack["tail_ready_wall_ns"] / 1e6
                                - eack["compute_start_wall_s"] * 1e3),
               "edge_ack_send_ms": (None if eack is None else
                                    (eack["ack_send_wall_ns"]
                                     - eack["tail_ready_wall_ns"]) / 1e6),
               "ack_downlink_ms": (None if eack is None or ack is None else
                                   ack["receipt_wall_s"] * 1e3
                                   - eack["ack_send_wall_ns"] / 1e6),
               **({} if not frame["features"] else invert_features(frame["features"]))}
        row["timely"] = row["terminal"] == "SUCCESS"
        rows.append(row)

    def latency_block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        measured = [r["ack_latency_ms"] for r in subset if r["ack_latency_ms"] is not None]
        return {"decisions": len(subset),
                "success": sum(r["terminal"] == "SUCCESS" for r in subset),
                "timeout": sum(r["terminal"] == "TIMEOUT" for r in subset),
                "service_failure": sum(r["terminal"] == "SERVICE_FAILURE" for r in subset),
                "unresolved": sum(r["terminal"] is None for r in subset),
                "ack_never_received": sum(r["ack_latency_ms"] is None for r in subset),
                "fraction_le_170ms": (sum(r["terminal"] == "SUCCESS" for r in subset)
                                      / len(subset)) if subset else None,
                "action_open_to_ack_ms_received": summary(measured)}

    stage_names = ("ue_prep_ms", "state_actor_ms", "input_7ch_ms", "front_codec_ms",
                   "send_ms", "open_to_last_send_ms", "uplink_ms", "edge_queue_ms",
                   "edge_tail_ms", "edge_ack_send_ms", "ack_downlink_ms")
    startup = [r for r in rows if r["frame_order"] <= STARTUP_FRAMES]
    sustained = [r for r in rows if r["frame_order"] > STARTUP_FRAMES]
    kinds = Counter(f["kind"] for f in frames)
    map_rows = _csv(attempt / "direct_edge_map" / "direct_map_ingest.csv")
    map_outcomes = Counter(r.get("outcome") for r in map_rows)
    map_frames = {int(r["frame_id"]) for r in map_rows if r.get("frame_id")}
    hold_runs = []
    for prev, cur in zip(frames, frames[1:]):
        if cur["kind"] == "POLICY_HOLD":
            hold_runs.append((cur["mode_id"], cur["q_e4"]) == (prev["mode_id"], prev["q_e4"])
                             or prev["kind"] == "POLICY_HOLD")
    after_timeout = []
    for prev, cur in zip(rows, rows[1:]):
        if prev["terminal"] == "TIMEOUT":
            after_timeout.append((cur["mode_id"], cur["q_e4"]) != (prev["mode_id"],
                                                                   prev["q_e4"]))
    gpu = None
    span = [raw_to_wall_s(f["stages"].get("rgb_receipt_raw_ns")) for f in frames]
    span = [s for s in span if s is not None]
    if gpu_csv is not None and span:
        gpu = gpu_trace(gpu_csv, min(span) - 1.0, max(span) + 1.0)
    front = [(raw_to_wall_s(f["stages"].get("front_start_raw_ns")),
              raw_to_wall_s(f["stages"].get("front_end_raw_ns"))) for f in frames]
    front = [(a, b) for a, b in front if a is not None and b is not None]
    tail = [(r["compute_start_wall_s"], r["tail_ready_wall_ns"] / 1e9)
            for r in edge.get("evaluations") or () if r.get("ack_sha256")]
    tail_total = sum(b - a for a, b in tail)
    gt_dir = evidence / "gt_scratch_preserved"
    edge_config = _json(evidence / "run4_phase6_edge_config.json") if (
        evidence / "run4_phase6_edge_config.json").is_file() else {}
    quality = offline_quality(
        decisions=[r for r in rows if r["session_uuid"] is not None],
        prediction_dir=gt_dir / "run4b_predictions", gt_dir=gt_dir,
        match_distance_m=float(match_distance_m or edge_config.get("match_distance_m")
                               or 3.0))
    from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C4B

    offline_reward = []
    for row, qrow in zip([r for r in rows if r["session_uuid"] is not None],
                         quality["rows"]):
        if row["terminal"] == "SUCCESS" and qrow["q_perc"] is not None:
            offline_reward.append(qrow["q_perc"] - C4B.LATENCY_WEIGHT
                                  * row["ack_latency_ms"] / DEADLINE_MS)
        elif row["terminal"] in ("TIMEOUT", "SERVICE_FAILURE"):
            offline_reward.append(C4B.FAILURE_REWARD)
    return {
        "schema": SCHEMA, "attempt": str(attempt),
        "classification": cell.get("status"), "cell_error": cell.get("error"),
        "reconciliation": {
            "transmitted_frames": len(frames),
            "transmitted_identities": len(ue.get("transmitted_identities") or ()),
            "frame_kinds": dict(kinds),
            "policy_decisions": (ue.get("counters") or {}).get("policy_decisions"),
            "resolutions": len(resolutions),
            "terminals": dict(Counter(r["terminal"] for r in resolutions)),
            "unresolved_tickets_at_close": ue.get("unresolved_tickets_at_close"),
            "ack_rows": dict(Counter(r["class"] for r in ack_rows)),
            "holds_reuse_active_action": all(hold_runs) if hold_runs else None,
            "faulted": ue.get("faulted")},
        "latency_all": latency_block(rows),
        "latency_startup_frames_1_30": latency_block(startup),
        "latency_sustained_frames_31_300": latency_block(sustained),
        "stages_ms": {name: summary(r[name] for r in rows) for name in stage_names},
        "stages_ms_startup": {name: summary(r[name] for r in startup) for name in stage_names},
        "stages_ms_sustained": {name: summary(r[name] for r in sustained)
                                for name in stage_names},
        "actions": {
            "modes": dict(Counter(r["mode_id"] for r in rows)),
            "q_e4": summary(r["q_e4"] for r in rows),
            "action_changed_after_timeout": (sum(after_timeout), len(after_timeout)),
            "by_frame_order": [[r["frame_order"], r["mode_id"], r["q_e4"], r["terminal"]]
                               for r in rows]},
        "radio": {"prior_ul_mcs": dict(Counter(r.get("prior_ul_mcs") for r in rows)),
                  "pre_action_rlc_backlog_bytes":
                      summary(r.get("pre_action_rlc_backlog_bytes") for r in rows)},
        "payload": payload_summary(attempt / "per_frame_metrics.csv", frames),
        "fallbacks": dict(Counter(tuple(f["fallback_reasons"]) and "; ".join(
            f["fallback_reasons"]) for f in frames if f["kind"] == "FALLBACK")),
        "map": {"rows": len(map_rows), "outcomes": dict(map_outcomes),
                "missing_frames": len({f["frame_id"] for f in frames} - map_frames)},
        "gpu": {"trace": gpu,
                "edge_tail_total_s": tail_total,
                "ue_front_total_s": sum(b - a for a, b in front),
                "edge_tail_overlapping_ue_front_s": _overlap(tail, front)},
        "offline_quality": {k: v for k, v in quality.items() if k != "rows"},
        "offline_reward_label": "OFFLINE_ONLY__NEVER_FED_BACK",
        "offline_reward": summary(offline_reward),
        "edge": {"counters": edge.get("counters"), "evaluator": edge.get("evaluator"),
                 "failures": edge.get("failures")},
        "decision_rows": rows,
        "quality_rows": quality["rows"],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_report(args.attempt, gpu_csv=args.gpu_csv)
    with Path(args.output).open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, sort_keys=True, default=str)
    print(json.dumps({k: report[k] for k in ("reconciliation", "latency_all")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
