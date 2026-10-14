"""Elapsed-time-binned M3.7 v18 (= v11 definitions) report: GT accounting cost, FREEZE/BIND, GT readiness, the two latency boundaries,
timeliness with explicit denominators, calibrated radio telemetry, payload, scene complexity and map freshness.

Everything is binned on ONE axis: seconds since the first capture of the route (``t_since_first_capture_s`` in the
registered analyzer's ``DECISION_TIMELINE.csv``).  The GT prep worker's trace and the accounting timer record
CLOCK_MONOTONIC_RAW only, so they are placed on that axis through the telemetry collector's own
(wall, RAW) anchor plus the first capture's wall clock -- both system-wide clocks, read in the same session.

Denominators are never implicit: every fraction carries the count it was taken over, and decisions with undefined
quality are reported separately from late ones.
"""
import csv
import json
from pathlib import Path
import statistics

from . import telemetry as T

BIN_S = 30.0


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _q(values, p):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))]


def _stats(values, scale=1.0):
    values = [v * scale for v in values if v is not None]
    if not values:
        return dict(n=0)
    return dict(n=len(values), p50=statistics.median(values), p95=_q(values, 0.95), p99=_q(values, 0.99),
                mean=statistics.fmean(values), max=max(values))


def load_decisions(analysis_dir):
    with (Path(analysis_dir) / "DECISION_TIMELINE.csv").open(newline="") as handle:
        return [row for row in csv.DictReader(handle)]


def load_scene(route_layers_dir):
    """camera SI / radar P40 per decision, from the route-overlay join (empty if the overlays were not rendered)."""
    path = Path(route_layers_dir) / "ROUTE_LAYERS_JOINED.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return {row["frame_id"]: row for row in csv.DictReader(handle)}


def radio_dir(root):
    """``m3_radio`` is written under the capture directory; the GT worker traces sit at the prospective root."""
    return Path(root) / "capture/m3_radio"


def trace_dir(root):
    return Path(root) / "gt_gc_trace"


def clock_anchor(root):
    """(wall_ns, raw_ns) read in the runner parent when the telemetry collectors started."""
    document = json.loads((radio_dir(root) / "RADIO_TELEMETRY.json").read_bytes())
    anchor = document.get("clock_anchor") or {}
    if not anchor.get("wall_ns") or not anchor.get("raw_ns"):
        raise ValueError("radio telemetry record carries no clock anchor")
    return int(anchor["wall_ns"]), int(anchor["raw_ns"])


def raw_to_elapsed(raw_ns, anchor, first_capture_wall_s):
    wall_ns, anchor_raw = anchor
    return (wall_ns + (int(raw_ns) - anchor_raw)) / 1e9 - first_capture_wall_s


def gt_prep_ops(root):
    document = json.loads((trace_dir(root) / "prep_gc_trace.json").read_bytes())
    return document.get("ops") or [], document


def accounting_calls(root):
    path = trace_dir(root) / "storage_accounting_timing.json"
    return json.loads(path.read_bytes()) if path.is_file() else None


def build(root, analysis_dir, route_layers_dir=None, bin_s=BIN_S):
    """``root`` is the prospective root (it holds ``capture/`` and ``gt_gc_trace/``)."""
    root = Path(root)
    decisions = load_decisions(analysis_dir)
    scene = load_scene(route_layers_dir) if route_layers_dir else {}
    first_capture = min(_f(row["capture_wall_s"]) for row in decisions)
    anchor = clock_anchor(root)
    ops, prep_trace = gt_prep_ops(root)
    timing = accounting_calls(root)
    telemetry = T.load(radio_dir(root) / "ttracer")

    def bucket(elapsed):
        return None if elapsed is None else int(elapsed // bin_s)

    bins = {}

    def row_for(index):
        return bins.setdefault(index, dict(
            bin_index=index, t_start_s=index * bin_s, t_end_s=(index + 1) * bin_s,
            decisions=[], freeze=[], bind=[], freeze_wall=[], bind_wall=[], accounting=[], accounting_by_caller={},
            pusch=[], avg_snr=[], mcs=[], scene_si=[], scene_radar=[]))

    for row in decisions:
        index = bucket(_f(row["t_since_first_capture_s"]))
        if index is None:
            continue
        item = row_for(index)
        item["decisions"].append(row)
        joined = scene.get(row["frame_id"])
        if joined:
            for key, target in (("camera_si", "scene_si"), ("radar_p40", "scene_radar")):
                value = _f(joined.get(key))
                if value is not None:
                    item[target].append(value)

    for op in ops:
        index = bucket(raw_to_elapsed(op["begin_raw_ns"], anchor, first_capture))
        if index is None or op.get("op") not in ("FREEZE", "BIND"):
            continue
        item = row_for(index)
        item["freeze" if op["op"] == "FREEZE" else "bind"].append(op["cpu_ns"] / 1e6)
        item["freeze_wall" if op["op"] == "FREEZE" else "bind_wall"].append(
            (op["end_raw_ns"] - op["begin_raw_ns"]) / 1e6)

    if timing:
        names = timing.get("caller_names") or []
        for begin, cpu, caller in zip(timing["begin_raw_ns"], timing["cpu_ns"], timing["caller"]):
            index = bucket(raw_to_elapsed(begin, anchor, first_capture))
            if index is None:
                continue
            item = row_for(index)
            item["accounting"].append(cpu / 1e6)
            name = names[caller] if caller < len(names) else str(caller)
            item["accounting_by_caller"].setdefault(name, []).append(cpu / 1e6)

    for name, key, field in (("pusch", "pusch", "pusch_snr_db"), ("mcs", "avg_snr", "scheduler_avg_snr_db")):
        for entry in telemetry[name]["rows"]:
            index = bucket((entry["wall_ns"] / 1e9) - first_capture)
            if index is None:
                continue
            item = row_for(index)
            item[key].append(entry[field])
            if name == "mcs":
                item["mcs"].append(entry["final_mcs"])

    out = []
    for index in sorted(bins):
        item = bins[index]
        rows = item["decisions"]
        policy = [r for r in rows if r.get("kind") == "policy"] or rows
        timely = [r for r in rows if r.get("timely_170") in ("1", "True", "true")]
        defined = [r for r in rows if r.get("quality_defined") in ("1", "True", "true")]
        outcomes = {}
        for r in rows:
            outcomes[r.get("outcome_class") or ""] = outcomes.get(r.get("outcome_class") or "", 0) + 1
        record = dict(
            bin_index=index, t_start_s=item["t_start_s"], t_end_s=item["t_end_s"],
            decisions_n=len(rows),
            timely_n=len(timely), timely_fraction=(len(timely) / len(rows)) if rows else None,
            timely_denominator=len(rows), outcome_counts=outcomes,
            quality_defined_n=len(defined),
            prediction_completion_ms=_stats([_f(r["prediction_completion_ms"]) for r in rows]),
            feedback_receipt_ms=_stats([_f(r["feedback_receipt_ms"]) for r in rows]),
            gt_published_ms=_stats([_f(r["gt_published_ms"]) for r in rows]),
            freeze_cpu_ms=_stats(item["freeze"]), bind_cpu_ms=_stats(item["bind"]),
            freeze_wall_ms=_stats(item["freeze_wall"]), bind_wall_ms=_stats(item["bind_wall"]),
            accounting_cpu_ms=_stats(item["accounting"]),
            accounting_calls_n=len(item["accounting"]),
            accounting_cpu_ms_by_caller={k: _stats(v) for k, v in sorted(item["accounting_by_caller"].items())},
            accounting_ms_per_decision=(sum(item["accounting"]) / len(rows)) if rows and item["accounting"] else None,
            pusch_snr_db=T.summarise(item["pusch"]), scheduler_avg_snr_db=T.summarise(item["avg_snr"]),
            mcs=T.summarise(item["mcs"]),
            commanded_target_db=T.summarise([_f(r["snr_proxy_db"]) for r in rows if _f(r["snr_proxy_db"]) is not None]),
            prior_ul_mcs=T.summarise([_f(r["prior_ul_mcs"]) for r in rows if _f(r["prior_ul_mcs"]) is not None]),
            backlog_bytes=T.summarise([_f(r["pre_action_rlc_backlog_bytes"]) for r in rows
                                       if _f(r["pre_action_rlc_backlog_bytes"]) is not None]),
            payload_l2_bytes=T.summarise([_f(r["payload_l2_bytes"]) for r in rows if _f(r["payload_l2_bytes"])]),
            q_e4=T.summarise([_f(r["q_e4"]) for r in policy if _f(r["q_e4"]) is not None]),
            camera_si=T.summarise(item["scene_si"]), radar_p40=T.summarise(item["scene_radar"]),
            map_age_at_install_ms=_stats([_f(r["map_age_at_install_ms"]) for r in rows]))
        out.append(record)
    return dict(schema="splitfusion.m3_route_b_v18.binned_report.v1", bin_seconds=bin_s,
                first_capture_wall_s=first_capture, clock_anchor=dict(wall_ns=anchor[0], raw_ns=anchor[1]),
                decisions_total=len(decisions), bins=out,
                accounting_timing=None if not timing else {k: timing[k] for k in
                    ("calls", "recorded", "dropped", "errors", "foreign_frames", "capacity", "caller_names")},
                prep_trace=dict(ops=len(ops), gen2=len(prep_trace.get("gen2") or []),
                                safety_stop=prep_trace.get("safety_stop"), rss_final=prep_trace.get("rss_final"),
                                dropped=prep_trace.get("dropped")),
                telemetry_rows={k: len(v["rows"]) for k, v in telemetry.items()},
                telemetry_skipped={k: v["skipped_lines"] for k, v in telemetry.items()},
                labels=dict(T.LABELS))
