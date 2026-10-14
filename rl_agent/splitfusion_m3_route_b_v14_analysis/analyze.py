"""M3.4 BAD_ALL copy of the registered v3 analyzer (``splitfusion_m3_route_b_v3_analysis.analyze``): identical metric
definitions and outputs; the ONLY changes are presentation labels in figure 01 (the commanded schedule name is read
from the radio trace rows instead of the hard-coded GOOD_ALL, and the MAC-log series is labelled as the power-control
statistic it is, not achieved channel SNR) and the absolute import of ``route_geometry``.  Original header follows.

M3.1 GOOD_ALL analysis: integration-independent performance summaries, joinable CSV/JSON and verification figures.

    python3 -B -m rl_agent.splitfusion_m3_route_b_v3_analysis.analyze --root <capture root> --out <dir>

Read-only on the capture; create-only outputs. Definitions are the registered ones (ACCEPTANCE.json
analysis_definitions); per-decision boundaries/quality come from the retained row builder
(``splitfusion_run5_full_live_analysis_v1.report.rows``), unchanged. Basic verification figures only.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_m3_route_b_v3 import contract as C
from rl_agent.splitfusion_run5_full_live_analysis_v1 import report as LIVE
from rl_agent.splitfusion_m3_route_b_v3_analysis import route_geometry as RG

MAIN = Path(__file__).resolve().parents[2]
TOL = {"vehicle": 1.0, "person": 1.2}
DEADLINE_MS = 170.0
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STATS = re.compile(r"ulsch_rounds .*?BLER ([0-9.]+) MCS \(\d+\) (\d+).*?SNR ([\-0-9.]+) \(")


def q(values, ps=(50, 95, 99)):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    return dict(n=int(v.size), **({f"p{p}": None for p in ps} if not v.size else
                                   {f"p{p}": float(np.percentile(v, p)) for p in ps}),
                **({} if not v.size else dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()))))


def ratio(a, b):
    return None if not b else a / b


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def rows_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})


def pooled(rows, cls):
    tol = TOL[cls]
    eligible = sum(r[f"{cls}_eligible_gt"] or 0 for r in rows)
    tp = sum(r[f"{cls}_tp"] or 0 for r in rows)
    fp = sum(r[f"{cls}_fp"] or 0 for r in rows)
    errors = [float(e) for r in rows for e in (r[f"{cls}_matched_xy_errors_m"] or [])]
    within = sum(e <= tol for e in errors)
    inter = sum(r[f"seg_{cls}_intersection_pixels"] or 0 for r in rows)
    union = sum(r[f"seg_{cls}_union_pixels"] or 0 for r in rows)
    return dict(tolerance_m=tol, eligible_gt=eligible, matched_3m=tp, misses=eligible - tp, false_positives=fp,
                within_tolerance=within, recall_within_tolerance=ratio(within, eligible),
                precision_3m_gate=ratio(tp, tp + fp), seg_iou_pooled=ratio(inter, union),
                matched_error_m=q(errors, (50, 90, 95, 99)), errors=errors)


def frame_poses(cell, route):
    poses = RG.poses_from_archive(cell / "pose_archive/pose_evidence.jsonl")
    series = RG.progress_index_series([(x, y) for _, x, y, _ in poses], route)
    out = {}
    for (fid, x, y, ts), idx in zip(poses, series):
        out[fid] = dict(ego_x=x, ego_y=y, carla_timestamp=ts, progress_index=idx, **RG.sections(idx, route))
    return out, poses


def nearest_pose(index, fid):
    if fid in index:
        return index[fid]
    keys = [k for k in (fid - 1, fid + 1, fid - 2, fid + 2) if k in index]
    return index[keys[0]] if keys else {}


def map_tables(cell):
    ingest = rows_csv(cell / "direct_edge_map/direct_map_ingest.csv")
    identity = {int(r["frame_id"]): r for r in rows_csv(cell / "direct_edge_map/run4_map_identity.csv")}
    installs = []
    for r in ingest:
        kind = json.loads(identity.get(int(r["frame_id"]), {}).get("run4_identity_json") or "{}").get("frame_kind")
        installs.append(dict(frame_id=int(r["frame_id"]), frame_kind=kind, outcome=r["outcome"], agent_credit=r["agent_credit"],
                             capture_wall_s=float(r["capture_timestamp"]),
                             install_wall_s=float(r["install_timestamp"]) if r["install_timestamp"] else None,
                             map_age_at_install_ms=float(r["map_age_at_install_ms"]) if r["map_age_at_install_ms"] else None,
                             record_count=int(r["record_count"] or 0), update_bytes=int(r["direct_update_bytes"] or 0),
                             rejection_reason=r["rejection_reason"]))
    return installs


def aoi_series(installs, step_s=0.1):
    ok = sorted((r["install_wall_s"], r["capture_wall_s"]) for r in installs
                if r["outcome"] == "RESULT_INSTALLED" and r["install_wall_s"] is not None)
    if not ok:
        return [], []
    t, latest, i, out = ok[0][0], None, 0, []
    while t <= ok[-1][0] + 1e-9:
        while i < len(ok) and ok[i][0] <= t:
            latest = ok[i][1] if latest is None else max(latest, ok[i][1])
            i += 1
        out.append(dict(wall_s=t, aoi_ms=(t - latest) * 1e3))
        t += step_s
    gaps = [(b[0] - a[0]) * 1e3 for a, b in zip(ok, ok[1:])]
    return out, gaps


def radio_tables(cell, capture_root):
    commanded = [dict(step=int(r["step_index"]), wall_s=int(r["command_ack_wall_ns"]) / 1e9 if r["command_ack_wall_ns"] else
                      int(r["scheduled_monotonic_ns"]) / 1e9, target_snr_db=float(r["target_snr_db"]),
                      rfsim_noise_power_db=float(r["mapped_rfsim_command_db"]), status=r["command_timing_status"],
                      profile_id=r.get("profile_id"),
                      ack_latency_ms=float(r["command_latency_ms"]) if r["command_latency_ms"] else None,
                      clock="command_ack_wall_ns (host wall)") for r in rows_csv(cell / "radio_trace.csv")]
    achieved = []
    path = capture_root / "m3_radio/gnb_stats_observed.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            m = STATS.search(row["line"])
            if m:
                achieved.append(dict(observed_wall_s=row["observed_wall_ns"] / 1e9, ul_bler=float(m.group(1)),
                                     ul_mcs=int(m.group(2)), ul_pusch_snr_db=float(m.group(3)),
                                     semantics="ACHIEVED gNB MAC stats; observation time upper bound (2-Hz poll)"))
    return commanded, achieved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verdict", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args(argv)
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    arm, rows, evidence = LIVE.rows(a.root)
    cell = arm.cell
    capture_root = arm.dir / "capture"
    route = RG.load_route(MAIN)
    pose_index, poses = frame_poses(cell, route)
    decisions = evidence["decisions"]
    per_frame = rows_csv(cell / "per_frame_metrics.csv")
    sent = [r for r in per_frame if r.get("prepare_status") == "SENT"]
    installs = map_tables(cell)
    aoi, gaps = aoi_series(installs)
    commanded, achieved = radio_tables(cell, capture_root)
    result = json.loads((cell / "CELL_RESULT.json").read_bytes())
    child = result.get("child") or {}
    summary_route = json.loads((cell / "route_metrics_summary.json").read_bytes()) if (cell / "route_metrics_summary.json").exists() else {}
    edge_report = json.loads((cell / "run4_phase6/edge_report.json").read_bytes())
    coverage = RG.ordered_coverage([(x, y) for _, x, y, _ in poses], route)

    # ---- per-decision joined timeline
    first_capture = min((r["capture_timestamp_ns"] for r in rows if r.get("capture_timestamp_ns")), default=None)
    install_by_frame = {r["frame_id"]: r for r in installs}
    timeline = []
    for r in rows:
        p = nearest_pose(pose_index, int(r["frame_id"]))
        m = install_by_frame.get(int(r["frame_id"])) or {}
        completed = r["prediction_materialized"] is not None and r["edge_compute_end"] is not None
        timeline.append(dict(
            frame_id=r["frame_id"], decision_index=r["decision_index"], kind=r["executed_action"]["kind"],
            capture_wall_s=None if r.get("capture_timestamp_ns") is None else r["capture_timestamp_ns"] / 1e9,
            t_since_first_capture_s=None if first_capture is None or r.get("capture_timestamp_ns") is None
            else (r["capture_timestamp_ns"] - first_capture) / 1e9,
            carla_timestamp=r.get("carla_timestamp"), progress_index=p.get("progress_index"), leg=p.get("leg"),
            region=p.get("region"), decile=p.get("decile"), ego_x=p.get("ego_x"), ego_y=p.get("ego_y"),
            snr_proxy_db=r.get("snr_db"), prior_ul_mcs=r.get("prior_ul_mcs"),
            pre_action_rlc_backlog_bytes=r.get("pre_action_rlc_backlog_bytes"),
            mode_id=r["executed_action"]["mode_id"], q_e4=r["executed_action"]["q_e4"],
            payload_l2_bytes=r.get("payload_bytes"), payload_chunks=r.get("payload_chunks"),
            prediction_completion_ms=r["edge_compute_end"] if completed else None, prediction_completed=completed,
            gt_published_ms=r.get("gt_published"), scoring_end_ms=r.get("scoring_worker_end"), feedback_receipt_ms=r["ue_receipt"],
            timely_170=r["timely"], outcome_class=r["outcome_class"], controller_resolution=r["controller_resolution"],
            service_outcome=r["service_outcome"], quality_defined=r["quality_defined"], q_perc=r["q_perc"],
            map_outcome=m.get("outcome"), map_age_at_install_ms=m.get("map_age_at_install_ms"),
            vehicle_eligible=r["vehicle_eligible_gt"], vehicle_within_1p0=sum(1 for e in (r["vehicle_matched_xy_errors_m"] or []) if e <= 1.0),
            person_eligible=r["person_eligible_gt"], person_within_1p2=sum(1 for e in (r["person_matched_xy_errors_m"] or []) if e <= 1.2)))
    write_csv(out / "DECISION_TIMELINE.csv", timeline)

    # ---- per-frame (all transmitted frames incl. holds/fallbacks) map + payload table
    dec_by_frame = {int(d["frame_id"]): d for d in decisions}
    frames = []
    for r in per_frame:
        fid = int(r["frame_id"])
        p = nearest_pose(pose_index, fid)
        m = install_by_frame.get(fid) or {}
        d = dec_by_frame.get(fid) or {}
        frames.append(dict(frame_id=fid, prepare_status=r.get("prepare_status"), kind=d.get("kind"), mode_id=d.get("mode_id"),
                           q_e4=d.get("q_e4"), payload_l2_bytes=r.get("payload_bytes"), payload_chunks=r.get("payload_chunks"),
                           progress_index=p.get("progress_index"), leg=p.get("leg"), region=p.get("region"),
                           decile=p.get("decile"), map_outcome=m.get("outcome"), capture_wall_s=m.get("capture_wall_s"),
                           install_wall_s=m.get("install_wall_s"), map_age_at_install_ms=m.get("map_age_at_install_ms"),
                           record_count=m.get("record_count")))
    write_csv(out / "FRAME_MAP_PAYLOAD.csv", frames)
    write_csv(out / "MAP_AOI_TIMELINE.csv", aoi, ["wall_s", "aoi_ms"])
    write_csv(out / "RADIO_COMMANDED.csv", commanded)
    write_csv(out / "RADIO_ACHIEVED_GNB_STATS.csv", achieved,
              ["observed_wall_s", "ul_pusch_snr_db", "ul_mcs", "ul_bler", "semantics"])

    # ---- metrics
    scored = [r for r in rows if r["service_outcome"] == "SCORED"]
    defined = [r for r in scored if r["quality_defined"]]
    timely_defined = [r for r in rows if r["outcome_class"] == "TIMELY_DEFINED"]
    payload = [int(r["payload_bytes"]) for r in sent if r.get("payload_bytes")]
    completion = [t["prediction_completion_ms"] for t in timeline]
    receipts = [t["feedback_receipt_ms"] for t in timeline]

    def cdf(values, grid):
        n = len(values)
        return [sum(1 for v in values if v is not None and v <= g) / n if n else None for g in grid]
    grid = list(np.arange(0, 400.5, 1.0))
    with open(out / "LATENCY_CDF.csv", "x", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["ms", "prediction_completion_fraction_of_all_decisions", "feedback_receipt_fraction_of_all_decisions",
                    "denominator_decisions"])
        for g, a1, a2 in zip(grid, cdf(completion, grid), cdf(receipts, grid)):
            w.writerow([g, a1, a2, len(rows)])
    with open(out / "LOCALIZATION_CDF.csv", "x", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["class", "threshold_m", "fraction_of_all_eligible_gt", "eligible_gt", "matched_3m", "misses"])
        for cls in TOL:
            p = pooled(defined, cls)
            for x in np.linspace(0, 3.0, 301):
                w.writerow([cls, round(float(x), 3), ratio(sum(e <= x for e in p["errors"]), p["eligible_gt"]),
                            p["eligible_gt"], p["matched_3m"], p["misses"]])

    def section_summary(key):
        groups = collections.defaultdict(list)
        for t, r in zip(timeline, rows):
            groups[t[key]].append((t, r))
        out_rows = []
        for name in sorted(groups, key=lambda x: (x is None, str(x))):
            items = groups[name]
            ts = [t for t, _ in items]
            rs = [r for _, r in items if r["service_outcome"] == "SCORED" and r["quality_defined"]]
            fm = [f for f in frames if f[key] == name]
            v, pr = pooled(rs, "vehicle"), pooled(rs, "person")
            out_rows.append(dict(section=name, decisions=len(items), timely_170=sum(t["timely_170"] for t in ts),
                                 timely_fraction=ratio(sum(t["timely_170"] for t in ts), len(ts)),
                                 timely_defined=sum(t["outcome_class"] == "TIMELY_DEFINED" for t in ts),
                                 feedback_p50_ms=q([t["feedback_receipt_ms"] for t in ts])["p50"],
                                 completion_p50_ms=q([t["prediction_completion_ms"] for t in ts])["p50"],
                                 payload_l2_p50=q([t["payload_l2_bytes"] for t in ts])["p50"],
                                 vehicle_eligible=v["eligible_gt"], vehicle_recall_1p0=v["recall_within_tolerance"],
                                 person_eligible=pr["eligible_gt"], person_recall_1p2=pr["recall_within_tolerance"],
                                 frames=len(fm), map_installed=sum(f["map_outcome"] == "RESULT_INSTALLED" for f in fm),
                                 map_age_p50_ms=q([f["map_age_at_install_ms"] for f in fm])["p50"],
                                 mode_11_share=ratio(sum(t["mode_id"] == 11 for t in ts), len(ts))))
        return out_rows
    sections = {key: section_summary(key) for key in ("leg", "region", "decile")}
    for key, table in sections.items():
        write_csv(out / f"SECTIONS_{key.upper()}.csv", table)

    fallbacks = [d for d in decisions if d["kind"] == "FALLBACK"]
    counters = evidence["counters"]
    map_outcomes = collections.Counter(r["outcome"] for r in installs)
    journal = json.loads((cell / C.MAP_JOURNAL_DIR / "JOURNAL_SUMMARY.json").read_bytes()) \
        if (cell / C.MAP_JOURNAL_DIR / "JOURNAL_SUMMARY.json").exists() else {}
    profile = [json.loads(l) for l in (cell / "profiler/edge/edge.profile.jsonl").read_text().splitlines() if l.strip()]
    compute = [x for x in profile if x.get("r") == "span" and x.get("stage") == "edge.compute" and x.get("ph") == "live"]
    modes = collections.Counter((t["mode_id"]) for t in timeline if t["kind"] == "POLICY_DECISION")
    snr_values = sorted({t["snr_proxy_db"] for t in timeline if t["snr_proxy_db"] is not None})
    metrics = dict(
        schema="splitfusion.m3_route_b_v3.metrics.v1",
        route=dict(completed=child.get("stop_reason") == "ROUTE_COMPLETED", completion=child.get("m3_route_completion"),
                   summary={k: summary_route.get(k) for k in ("status", "loops_completed", "planned_route_length_m",
                            "driven_distance_m_median", "simulation_duration_s_median", "wall_clock_duration_s_total",
                            "return_position_error_m_max", "return_heading_error_deg_max", "intervention_count_total",
                            "collision_incident_count_total", "watchdog_aborted_any", "ego_overtakes_total",
                            "npc_roadblocks_cleared_total")},
                   ordered_progress_coverage={k: coverage[k] for k in ("points", "matched", "coverage", "radius_m")},
                   cell_wall_s=result.get("finished_at_unix_s", 0) - result.get("started_at_unix_s", 0),
                   capture_span_s=None if not installs else max(r["capture_wall_s"] for r in installs) - min(r["capture_wall_s"] for r in installs)),
        counts=dict(prepare_status=dict(collections.Counter(r.get("prepare_status") for r in per_frame)),
                    transmitted=len(evidence["transmitted_identities"]), edge_compute_spans=len(compute),
                    edge_compute_ok=sum(x.get("ok") is True for x in compute),
                    edge_publications=edge_report.get("publications"), map_ingest_rows=len(installs),
                    map_outcomes=dict(map_outcomes), map_journal=journal.get("counts"),
                    edge_terminals={k: v for k, v in (edge_report.get("counters") or {}).items() if k.startswith("edge_terminal_")}),
        decisions=dict(policy_decisions=counters.get("policy_decisions"), holds=counters.get("policy_holds"),
                       fallbacks=len(fallbacks), actor_calls=counters.get("actor_calls"),
                       fallback_reasons=dict(collections.Counter(x.split(":")[0] for d in fallbacks
                                                                 for x in d.get("fallback_reasons", []))),
                       controller_resolutions=dict(collections.Counter(r["controller_resolution"] for r in rows)),
                       service_outcomes=dict(collections.Counter(r["service_outcome"] for r in rows)),
                       outcome_classes=dict(collections.Counter(r["outcome_class"] for r in rows)),
                       decision_rows=len(rows)),
        latency_ms=dict(prediction_completion=q(completion), prediction_completed=sum(c is not None for c in completion),
                        prediction_completed_by_170=sum(1 for c in completion if c is not None and c <= DEADLINE_MS),
                        feedback_receipt=q(receipts), feedback_received=sum(c is not None for c in receipts),
                        timely_170=sum(t["timely_170"] for t in timeline),
                        timely_wilson95=wilson(sum(t["timely_170"] for t in timeline), len(timeline)),
                        timely_defined_quality=len(timely_defined), denominator_decisions=len(rows)),
        quality=dict(scored=len(scored), defined=len(defined), undefined_scored=len(scored) - len(defined),
                     **{cls: {k: v for k, v in pooled(defined, cls).items() if k != "errors"} for cls in TOL},
                     timely_defined={cls: {k: v for k, v in pooled(timely_defined, cls).items() if k != "errors"} for cls in TOL}),
        payload=dict(layer="L2 complete SFD4 application frame (per_frame_metrics payload_bytes) over every SENT frame",
                     l2_bytes=q(payload), l2_total_bytes=int(sum(payload)), sent_frames=len(payload),
                     datagrams=q([int(r["payload_chunks"]) for r in sent if r.get("payload_chunks")]),
                     not_measured="L3 header bytes, GTP-U/PDCP/RLC/MAC/radio overhead"),
        actions=dict(mode_share_policy_decisions=dict(sorted(modes.items())),
                     q_e4_mode11=q([t["q_e4"] for t in timeline if t["mode_id"] == 11 and t["kind"] == "POLICY_DECISION"])),
        channel=dict(snr_proxy_values_db=snr_values, commanded_rows=len(commanded),
                     commanded_values_db=sorted({c["target_snr_db"] for c in commanded}),
                     commanded_noise_power_db=sorted({c["rfsim_noise_power_db"] for c in commanded}),
                     command_status=dict(collections.Counter(c["status"] for c in commanded)),
                     command_ack_latency_ms=q([c["ack_latency_ms"] for c in commanded]),
                     achieved_ul_pusch_snr_db=q([x["ul_pusch_snr_db"] for x in achieved]),
                     achieved_ul_mcs=q([x["ul_mcs"] for x in achieved]), achieved_samples=len(achieved),
                     prior_ul_mcs_state=q([t["prior_ul_mcs"] for t in timeline]),
                     backlog_state_bytes=q([t["pre_action_rlc_backlog_bytes"] for t in timeline])),
        map=dict(age_at_install_ms=q([r["map_age_at_install_ms"] for r in installs if r["outcome"] == "RESULT_INSTALLED"]),
                 aoi_ms=q([x["aoi_ms"] for x in aoi]), update_gap_ms=q(gaps, (50, 95, 99)),
                 gaps_over_500ms=sum(g > 500 for g in gaps), horizon_ms=500.0,
                 horizon_boundary="capture_timestamp (UE host wall clock at the CARLA capture tick) vs map ingest time",
                 late_feedback_but_installed=sum(1 for t in timeline if not t["timely_170"] and t["map_outcome"] == "RESULT_INSTALLED"),
                 decisions_not_installed=dict(collections.Counter(t["map_outcome"] for t in timeline
                                                                  if t["map_outcome"] != "RESULT_INSTALLED"))),
        sections={k: v for k, v in sections.items()})
    (out / "METRICS.json").write_text(json.dumps(metrics, indent=1, sort_keys=True, default=str) + "\n")
    (out / "CLOCK_MAPPING.json").write_text(json.dumps(dict(
        action_open_relative_ms="UE/edge/service CLOCK_MONOTONIC_RAW; edge profiler bridged by the retained profiler clock map "
                                "(splitfusion_fixed_action_screen_analysis_v2.report.rows_for); all *_ms columns = t - action_open",
        capture_timestamp="host wall clock (time.time_ns) at the CARLA capture tick, stamped by the UE; edge/map share the host clock",
        map_install="map process time.time() (host wall); age = install - capture",
        radio_command="radio_trace command_ack_wall_ns (host wall) and CLOCK_MONOTONIC send/ack",
        snr_proxy="controller publication CLOCK_MONOTONIC_RAW; admitted iff effective before state commit and heartbeat <= 200 ms",
        gnb_stats="runner observation CLOCK_MONOTONIC_RAW + wall; upper bound of print time (2-Hz poll)",
        route_progress="pose archive OWNED_SOURCE ego_world per CARLA frame id (simulator); joined by frame_id"),
        indent=1) + "\n")
    figures(out, timeline, frames, aoi, commanded, achieved, metrics, route, poses)
    snapshots(out, cell, rows, timeline)
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                        for p in sorted(out.iterdir()) if p.is_file()}, indent=1))
    print(json.dumps(dict(route=metrics["route"]["completed"], decisions=len(rows), timely=metrics["latency_ms"]["timely_170"],
                          installed=map_outcomes.get("RESULT_INSTALLED")), default=str))


def figures(out, timeline, frames, aoi, commanded, achieved, metrics, route, poses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = min(t["capture_wall_s"] for t in timeline if t["capture_wall_s"])
    tt = [None if t["capture_wall_s"] is None else t["capture_wall_s"] - t0 for t in timeline]

    def save(fig, name):
        fig.savefig(out / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
    # 01 aligned latency / SNR timeline (stacked, shared x; one y-axis per panel)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ax = axes[0]
    ax.plot([c["wall_s"] - t0 for c in commanded], [c["target_snr_db"] for c in commanded], lw=2, color=COLORS[0],
            label="commanded target (" + "/".join(sorted({c.get("profile_id") or "?" for c in commanded})) + ")")
    ax.plot([x["observed_wall_s"] - t0 for x in achieved], [x["ul_pusch_snr_db"] for x in achieved], lw=1, marker="o", ms=3,
            color=COLORS[1], label="MAC-log power-control SNR (avg_snr+tpc; NOT channel SNR)")
    ax.set_ylabel("SNR (dB)"); ax.legend(loc="lower left", fontsize=8); ax.set_title("Channel: commanded vs achieved")
    ax = axes[1]
    ax.plot(tt, [t["prior_ul_mcs"] for t in timeline], lw=1, color=COLORS[2]); ax.set_ylabel("prior UL MCS (state)")
    ax = axes[2]
    ax.scatter(tt, [t["prediction_completion_ms"] for t in timeline], s=6, color=COLORS[0], label="prediction completion")
    ax.scatter(tt, [t["feedback_receipt_ms"] for t in timeline], s=6, color=COLORS[1], label="feedback receipt")
    ax.axhline(DEADLINE_MS, color="#555555", lw=1, ls="--"); ax.set_ylabel("ms after action-open"); ax.legend(fontsize=8)
    ax = axes[3]
    ax.plot([a["wall_s"] - t0 for a in aoi], [a["aoi_ms"] for a in aoi], lw=1, color=COLORS[3])
    ax.set_ylabel("map AoI (ms)"); ax.set_xlabel("time since first capture (s, host wall)")
    save(fig, "01_timeline_time.png")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    pi = [t["progress_index"] for t in timeline]
    axes[0].scatter(pi, [t["feedback_receipt_ms"] for t in timeline], s=6, color=COLORS[1]); axes[0].axhline(DEADLINE_MS, color="#555555", lw=1, ls="--")
    axes[0].set_ylabel("feedback (ms)"); axes[0].set_title("Along route progress (4-m progress points; legs marked)")
    axes[1].scatter(pi, [t["payload_l2_bytes"] for t in timeline], s=6, color=COLORS[0]); axes[1].set_ylabel("payload L2 (B)")
    axes[2].scatter(pi, [t["map_age_at_install_ms"] for t in timeline], s=6, color=COLORS[3]); axes[2].set_ylabel("map age at install (ms)")
    for ax in axes:
        for m in RG.waypoint_progress_index(route):
            ax.axvline(m, color="#cccccc", lw=0.6)
    axes[2].set_xlabel("progress index")
    save(fig, "02_timeline_progress.png")
    # 03 mode / q / payload over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].scatter(tt, [t["mode_id"] for t in timeline], s=6, color=COLORS[0]); axes[0].set_ylabel("mode"); axes[0].set_yticks(range(12))
    axes[1].scatter(tt, [t["q_e4"] for t in timeline], s=6, color=COLORS[2]); axes[1].set_ylabel("q (e4)")
    axes[2].scatter(tt, [t["payload_l2_bytes"] for t in timeline], s=6, color=COLORS[1]); axes[2].set_ylabel("payload L2 (B)")
    axes[2].set_xlabel("time since first capture (s)")
    axes[0].set_title("Executed action (policy decisions, holds, fallbacks)")
    save(fig, "03_mode_q_payload.png")
    # 04 CDFs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n = len(timeline)
    for i, (key, label) in enumerate((("prediction_completion_ms", "prediction completion"), ("feedback_receipt_ms", "feedback receipt"))):
        v = np.sort([t[key] for t in timeline if t[key] is not None])
        axes[0].step(v, np.arange(1, len(v) + 1) / n, where="post", color=COLORS[i], lw=2, label=f"{label} ({len(v)}/{n})")
    axes[0].axvline(DEADLINE_MS, color="#555555", lw=1, ls="--"); axes[0].set_xlabel("ms after action-open")
    axes[0].set_ylabel("fraction of ALL decisions"); axes[0].legend(fontsize=8); axes[0].set_xlim(0, 400)
    rows = list(csv.DictReader(open(out / "LOCALIZATION_CDF.csv")))
    for i, cls in enumerate(TOL):
        sub = [r for r in rows if r["class"] == cls]
        axes[1].plot([float(r["threshold_m"]) for r in sub], [float(r["fraction_of_all_eligible_gt"] or 0) for r in sub],
                     lw=2, color=COLORS[i], label=f"{cls} (eligible {sub[0]['eligible_gt']}, misses in denominator)")
        axes[1].axvline(TOL[cls], color=COLORS[i], lw=0.8, ls=":")
    axes[1].set_xlabel("xy localisation error (m)"); axes[1].set_ylabel("fraction of eligible GT"); axes[1].legend(fontsize=8)
    save(fig, "04_cdfs.png")
    # 05 route map coloured by feedback timeliness
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([p[0] for p in route["progress"]], [p[1] for p in route["progress"]], color="#cccccc", lw=6, label="planned loop")
    ax.plot([x for _, x, _, _ in poses], [y for _, _, y, _ in poses], color="#555555", lw=1, label="measured ego")
    for ok, color, label in ((True, COLORS[0], "decision timely <=170 ms"), (False, COLORS[1], "decision late / no feedback")):
        sel = [t for t in timeline if bool(t["timely_170"]) == ok and t["ego_x"] is not None]
        ax.scatter([t["ego_x"] for t in sel], [t["ego_y"] for t in sel], s=22, color=color, label=f"{label} ({len(sel)})",
                   zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title("Route B full loop (CARLA world x/y, m)")
    save(fig, "05_route_timeliness.png")


def join_map_to_predictions(cell, journal, frame_ids):
    """Installed map objects (class-less 'Unknown' type) joined to the exact scorer prediction records of the same frame
    by score (1e-6) and world x/y (1e-3 m); returns per-frame (joined, objects, predictions) and the class per object."""
    result, classes = {}, {}
    for fid in frame_ids:
        path = cell / f"numeric_gt/service/archive/frame_{fid}/scorer/numerical_score.json"
        if fid not in journal or not path.exists():
            continue
        preds = json.loads(path.read_bytes()).get("prediction_records") or []
        objs = journal[fid].get("objects", [])
        names = []
        for o in objs:
            hit = [p for p in preds if abs(p["score"] - o["score"]) < 1e-6 and abs(p["world_x"] - o["location"]["x"]) < 1e-3
                   and abs(p["world_y"] - o["location"]["y"]) < 1e-3]
            names.append(hit[0]["class_name"] if len(hit) == 1 else None)
        result[fid] = (sum(n is not None for n in names), len(objs), len(preds))
        classes[fid] = names
    return result, classes


def snapshots(out, cell, rows, timeline):
    """Best/worst 3 scored decision frames (>= 3 eligible objects) by within-tolerance F1; map record vs GT boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    journal = {}
    path = cell / C.MAP_JOURNAL_DIR / "installed_map_records.jsonl"
    with path.open() as handle:
        for line in handle:
            r = json.loads(line)
            if r["outcome"] == "OK" and r.get("map_document"):
                journal[r["frame_id"]] = r["map_document"]["record"]
    candidates = []
    for r in rows:
        if r["service_outcome"] != "SCORED" or not r["quality_defined"]:
            continue
        el = (r["vehicle_eligible_gt"] or 0) + (r["person_eligible_gt"] or 0)
        if el < 3 or r["frame_id"] not in journal:
            continue
        within = sum(1 for c in TOL for e in (r[f"{c}_matched_xy_errors_m"] or []) if e <= TOL[c])
        fp = (r["vehicle_fp"] or 0) + (r["person_fp"] or 0)
        recall, prec = within / el, within / max(1, within + fp)
        f1 = 0.0 if within == 0 else 2 * recall * prec / (recall + prec)
        candidates.append((f1, r["frame_id"], el))
    candidates.sort()
    picks = [("worst", c) for c in candidates[:3]] + [("best", c) for c in candidates[-3:]]
    scored_ids = [r["frame_id"] for r in rows if r["service_outcome"] == "SCORED"]
    joins, classes = join_map_to_predictions(cell, journal, scored_ids)
    exact = sum(1 for j, o, p in joins.values() if j == o == p)
    (out / "MAP_PREDICTION_JOIN.json").write_text(json.dumps(dict(
        scored_frames=len(scored_ids), frames_checked=len(joins), frames_exact=exact,
        objects=sum(o for _, o, _ in joins.values()), objects_joined=sum(j for j, _, _ in joins.values()),
        rule="installed map object == exact scorer prediction record (score 1e-6, world x/y 1e-3 m); map object type is "
             "'Unknown' in the Run-4 map update, class taken from the joined prediction record"), indent=1) + "\n")
    table = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (label, (f1, fid, el)) in zip(axes.ravel(), picks):
        meta_path = cell / f"numeric_gt/service/archive/frame_{fid}/producer/input_metadata.json"
        meta = json.loads(meta_path.read_bytes()) if meta_path.exists() else {}
        boxes = meta.get("actor_boxes") or []
        record = journal[fid]
        ego = next((t for t in timeline if t["frame_id"] == fid), {})
        ex, ey = ego.get("ego_x") or 0.0, ego.get("ego_y") or 0.0
        for i, cls in enumerate(("vehicle", "person")):
            gt = [b for b in boxes if str(b.get("label", "")).lower().startswith(cls[:3]) and (b.get("gt_distance_m") or 0) <= 60]
            ax.scatter([b["object_world_x"] for b in gt], [b["object_world_y"] for b in gt], s=40, facecolors="none",
                       edgecolors=COLORS[i], label=f"GT {cls}")
            names = classes.get(fid) or [None] * len(record.get("objects", []))
            pred = [o for o, n in zip(record.get("objects", []), names) if n == cls]
            for strong, alpha in ((True, 1.0), (False, 0.25)):
                sel = [o for o in pred if (o["score"] >= 0.3) == strong]
                ax.scatter([o["location"]["x"] for o in sel], [o["location"]["y"] for o in sel], s=26, marker="x",
                           color=COLORS[i], alpha=alpha, label=f"map {cls} (score {'>=' if strong else '<'} 0.3)")
        ax.scatter([ex], [ey], marker="^", s=60, color="#333333", label="ego")
        ax.set_xlim(ex - 50, ex + 50); ax.set_ylim(ey - 50, ey + 50); ax.set_aspect("equal")
        ax.set_title(f"{label}: frame {fid}, F1 {f1:.2f}, eligible {el}", fontsize=9)
        table.append(dict(selection=label, frame_id=fid, f1_within_tolerance=f1, eligible=el,
                          map_objects=len(record.get("objects", [])), gt_actors_in_archive=len(boxes)))
    if picks:
        axes.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Map snapshots: installed (measured) records vs exact GT actor positions (<= 60 m); selection by "
                 "within-tolerance F1 of the exact scorer; map class from the exact prediction join; faint = score < 0.3 (display only)",
                 fontsize=9)
    fig.savefig(out / "06_map_snapshots.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    write_csv(out / "MAP_SNAPSHOTS.csv", table)
    write_csv(out / "MAP_SNAPSHOT_CANDIDATES.csv", [dict(f1=c[0], frame_id=c[1], eligible=c[2]) for c in candidates])


if __name__ == "__main__":
    main()
