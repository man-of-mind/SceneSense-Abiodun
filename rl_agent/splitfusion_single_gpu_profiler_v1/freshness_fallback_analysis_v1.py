#!/usr/bin/env python3
"""Freshness fallback accounting over retained evidence (read-only).

For one profiled capture this script:

1. accounts for every fallback by reason class;
2. decomposes each **scene-stale** frame on the raw clock from the RGB
   callback to the state commit -- scheduling/worker wait, sensor wait, radar
   window, rasterization, RGB conversion, scene snapshot, submit preamble,
   SI/P40, engine-lock wait, telemetry snapshot and state build -- marks the
   first milestone at which the camera-SI age exceeded the 100-ms rule, and
   verifies that the descriptors belong to the exact image/radar inputs that
   were encoded;
3. decomposes each **MCS/backlog-stale** frame from the retained T-tracer
   streams using only rows the reader had received before the decision
   boundary: no recent eligible source event, source-to-reader delay,
   reader-to-publication delay, or a selection/timestamp discrepancy;
4. separates startup readiness (clock bridge warming, first telemetry) from
   steady-state failures;
5. reports, per capture, how many scene-stale frames were already over-age
   at worker start (cannot be rescued by any reordering inside the worker)
   and how many would have met the rule had SI/P40 been computed before the
   radar rasterization (a pure reordering of existing steps; the camera
   source timestamp stays the RGB receipt).

Nothing here changes a limit, restamps an observation or substitutes a frame.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

SCHEMA = "scenesense.single_gpu_profiler.freshness_fallback_analysis.v1"
RULE_NS = 100_000_000
STALE_RE = re.compile(r"^(camera_si|radar_p40) is stale \((\d+) ns\)$")
SPAN_STAGES = ("ue.worker_frame", "ue.sensor_wait", "ue.radar_window", "ue.radar_rasterize",
               "ue.rgb_convert", "ue.scene_snapshot", "ue.submit", "ue.plan",
               "agent.telemetry_snapshot", "agent.state_build")


def pct(values: Sequence[float], p: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(1, -(-int(p * len(ordered)) // 100)) - 1]


def dist(values: Iterable[Optional[float]]) -> dict[str, Any]:
    present = [float(v) for v in values if v is not None]
    return {"n": len(present), "mean": statistics.fmean(present) if present else None,
            "p50": pct(present, 50), "p95": pct(present, 95), "max": max(present) if present else None}


def ms(a: Optional[int], b: Optional[int]) -> Optional[float]:
    return None if a is None or b is None else (int(b) - int(a)) / 1e6


# ---------------------------------------------------------------------------
# telemetry streams
# ---------------------------------------------------------------------------


def _tod_to_wall_ns(tod: str, near_wall_ns: int) -> Optional[int]:
    """T-tracer local time-of-day -> host realtime ns, on the day nearest the receipt."""
    try:
        hh, mm, rest = tod.split(":")
        seconds = float(rest)
    except ValueError:
        return None
    local = dt.datetime.fromtimestamp(near_wall_ns / 1e9)
    candidates = []
    for day in (-1, 0, 1):
        base = (local.replace(hour=0, minute=0, second=0, microsecond=0)
                + dt.timedelta(days=day))
        stamp = base + dt.timedelta(hours=int(hh), minutes=int(mm), seconds=seconds)
        candidates.append(int(stamp.timestamp() * 1e9))
    return min(candidates, key=lambda w: abs(w - near_wall_ns))


def load_stream(path: Path) -> list[dict[str, Any]]:
    """Rows of one ``*_live.csv``: receipt raw/wall, event wall (from tod), fields."""
    rows: list[dict[str, Any]] = []
    header: Optional[list[str]] = None
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            try:
                receipt_raw, receipt_wall = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            payload = parts[2:]
            if payload[0] == "time":
                header = payload
                continue
            if header is None or len(payload) != len(header):
                continue
            row = dict(zip(header, payload))
            event_wall = _tod_to_wall_ns(row["time"], receipt_wall)
            rows.append({"receipt_raw": receipt_raw, "receipt_wall": receipt_wall,
                         "event_wall": event_wall,
                         "event_raw": None if event_wall is None
                         else receipt_raw - (receipt_wall - event_wall), **row})
    return rows


def dci_eligible(row: Mapping[str, Any]) -> bool:
    """The provider's new-data uplink grant rule (direction 1, first round)."""
    try:
        return int(row["direction"]) == 1 and int(row["round"]) == 0
    except (KeyError, ValueError):
        return False


def classify_radio_stale(boundary_raw: int, reported_age_ns: int, rows: Sequence[Mapping[str, Any]],
                         eligible: Any, label: str) -> dict[str, Any]:
    """Why the freshest eligible event was older than the rule at the boundary."""
    before = [r for r in rows if r["receipt_raw"] <= boundary_raw and r["event_raw"] is not None]
    window_lo = boundary_raw - RULE_NS
    in_window_any = [r for r in before if r["event_raw"] > window_lo]
    in_window_eligible = [r for r in in_window_any if eligible(r)]
    latest_eligible = max((r for r in before if eligible(r)), key=lambda r: r["event_raw"],
                          default=None)
    # events that happened in the window but reached the reader only after the boundary
    arrived_late = [r for r in rows if r["event_raw"] is not None and window_lo < r["event_raw"] <= boundary_raw
                    and r["receipt_raw"] > boundary_raw and eligible(r)]
    receipts = sorted(r["receipt_raw"] for r in rows if window_lo - RULE_NS <= r["receipt_raw"] <= boundary_raw)
    max_gap = max((b - a for a, b in zip(receipts, receipts[1:])), default=None)
    source_to_reader = [(r["receipt_raw"] - r["event_raw"]) / 1e6 for r in before[-200:]]
    out = {"slot": label, "boundary_raw_ns": boundary_raw, "reported_age_ms": reported_age_ns / 1e6,
           "latest_eligible_event_age_ms": None if latest_eligible is None
           else (boundary_raw - latest_eligible["event_raw"]) / 1e6,
           "latest_eligible_receipt_before_boundary_ms": None if latest_eligible is None
           else (boundary_raw - latest_eligible["receipt_raw"]) / 1e6,
           "rows_in_window_received_before_boundary": len(in_window_any),
           "eligible_rows_in_window_received_before_boundary": len(in_window_eligible),
           "eligible_events_in_window_received_after_boundary": len(arrived_late),
           "max_receipt_gap_in_preceding_200ms_ms": None if max_gap is None else max_gap / 1e6,
           "source_to_reader_delay_ms_recent": dist(source_to_reader)}
    if in_window_eligible:
        out["cause"] = "SELECTION_OR_TIMESTAMP_DISCREPANCY"
        out["detail"] = ("an eligible event inside the 100-ms window had already reached the reader; "
                         "the provider's age differs from the stream's (bridge/eligibility detail)")
    elif arrived_late:
        out["cause"] = "SOURCE_TO_READER_DELAY"
        out["detail"] = "an eligible event occurred inside the window but reached the reader after the boundary"
    elif in_window_any:
        out["cause"] = "NO_ELIGIBLE_SOURCE_EVENT__ONLY_INELIGIBLE_IN_WINDOW"
        out["detail"] = "only ineligible rows (other direction or retransmission rounds) in the window"
    else:
        out["cause"] = ("NO_SOURCE_EVENT_IN_WINDOW__READER_GAP" if max_gap and max_gap > RULE_NS
                        else "NO_SOURCE_EVENT_IN_WINDOW")
        out["detail"] = "no row of this stream with an event inside the window reached the reader"
    agreement = (out["latest_eligible_event_age_ms"] is not None
                 and abs(out["latest_eligible_event_age_ms"] - out["reported_age_ms"]) < 5.0)
    out["stream_age_agrees_with_provider_within_5ms"] = bool(agreement)
    return out


# ---------------------------------------------------------------------------
# per capture
# ---------------------------------------------------------------------------


class Capture:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        cells = sorted((self.root / "cells").iterdir())
        self.cell = cells[0]
        self.ue = json.loads((self.cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_text())
        self.decisions = sorted(self.ue["decisions"], key=lambda d: int(d["frame_id"]))
        self.by_frame = {int(d["frame_id"]): d for d in self.decisions}
        self.transmitted = {int(x["frame_id"]) for x in self.ue.get("transmitted_identities") or ()}
        self.spans: dict[int, dict[str, tuple[int, int]]] = collections.defaultdict(dict)
        self.rgb_marks: dict[int, int] = {}
        profile = self.cell / "profiler/ue.profile.jsonl"
        with profile.open(encoding="utf-8") as handle:
            for line in handle:
                if '"r":"span"' in line:
                    r = json.loads(line)
                    if r.get("stage") in SPAN_STAGES and r.get("frame_id") is not None:
                        self.spans[int(r["frame_id"])].setdefault(
                            r["stage"], (int(r["start_raw_ns"]), int(r["end_raw_ns"])))
                elif '"carla.rgb_receipt"' in line:
                    r = json.loads(line)
                    self.rgb_marks.setdefault(int(r["carla_frame"]), int(r["t_raw_ns"]))
        trace = self.ue.get("carla_trace") or {}
        self.trace_rgb = {int(f): int(raw) for f, raw in trace.get("rgb", [])}
        self.ticks = sorted((int(raw), int(f)) for f, _t, raw in trace.get("ticks", []))
        self.tick_by_frame = {f: raw for raw, f in self.ticks}
        telemetry = self.cell / "run4_phase6/telemetry_live"
        self.dci = load_stream(telemetry / "NRUE_MAC_DCI_GRANT_live.csv")
        self.rlc = load_stream(telemetry / "NRUE_MAC_RLC_BUFFER_STATUS_live.csv")
        self.pdcp = load_stream(telemetry / "NR_PDCP_TX_SDU_live.csv")

    # -- classification ---------------------------------------------------------
    @staticmethod
    def reason_class(reasons: Sequence[str]) -> str:
        if any(r == "CLOCK_BRIDGE_INVALID_OR_WARMING" for r in reasons):
            return "STARTUP_CLOCK_BRIDGE_WARMING"
        if any(STALE_RE.match(r) for r in reasons):
            return "SCENE_STALE"
        if any(r.startswith("UE_PRIOR_NEW_DATA_UL_MCS_INDEX:STALE") for r in reasons):
            return "MCS_STALE"
        if any(r.startswith("UE_PRE_ACTION_RLC_BACKLOG_BYTES:STALE") for r in reasons):
            return "BACKLOG_STALE"
        if any(":MISSING" in r for r in reasons):
            return "RADIO_MISSING_STEADY_STATE"
        return "OTHER:" + "|".join(sorted(reasons))[:80]

    def _boundary(self, d: Mapping[str, Any]) -> Optional[int]:
        commit = d.get("state_commit")
        return int(commit["ns"]) if isinstance(commit, Mapping) else None

    # -- scene-stale decomposition ------------------------------------------------
    def scene_row(self, d: Mapping[str, Any]) -> dict[str, Any]:
        frame = int(d["frame_id"])
        st = dict(d.get("stages") or {})
        sp = self.spans.get(frame, {})
        rgb = st.get("rgb_receipt_raw_ns")
        commit = self._boundary(d) or st.get("si_p40_end_raw_ns")
        reported = next((int(m.group(2)) for r in d.get("fallback_reasons") or ()
                         for m in [STALE_RE.match(r)] if m), None)
        g = lambda name, i: (sp[name][i] if name in sp else None)  # noqa: E731
        milestones = {
            "scheduled_tick": self.tick_by_frame.get(frame), "worker_start": g("ue.worker_frame", 0),
            "sensor_ready": g("ue.sensor_wait", 1), "radar_window_end": g("ue.radar_window", 1),
            "rasterize_start": st.get("radar_tensor_start_raw_ns"),
            "rasterize_end": st.get("radar_tensor_end_raw_ns"),
            "rgb_convert_end": g("ue.rgb_convert", 1), "scene_snapshot_end": g("ue.scene_snapshot", 1),
            "submit_start": g("ue.submit", 0), "sensor_prepared": st.get("sensor_prepared_raw_ns"),
            "si_p40_start": st.get("si_p40_start_raw_ns"), "si_p40_end": st.get("si_p40_end_raw_ns"),
            "plan_start_lock_acquired": g("ue.plan", 0),
            "telemetry_snapshot_end": g("agent.telemetry_snapshot", 1),
            "state_build_end": g("agent.state_build", 1), "state_commit": commit}
        ages = {k: ms(rgb, v) for k, v in milestones.items()}
        first_over = next((k for k, v in ages.items() if v is not None and v > RULE_NS / 1e6
                           and k not in ("scheduled_tick",)), None)
        segments = {
            "wait_callback_to_worker_start": ms(rgb, milestones["worker_start"]),
            "wait_sensor_ready": ms(milestones["worker_start"], milestones["sensor_ready"]),
            "compute_radar_window": ms(g("ue.radar_window", 0), g("ue.radar_window", 1)),
            "compute_rasterize": ms(milestones["rasterize_start"], milestones["rasterize_end"]),
            "compute_rgb_convert": ms(g("ue.rgb_convert", 0), g("ue.rgb_convert", 1)),
            "sync_scene_snapshot": ms(g("ue.scene_snapshot", 0), g("ue.scene_snapshot", 1)),
            "submit_preamble": ms(milestones["submit_start"], milestones["si_p40_start"]),
            "compute_si_p40": ms(milestones["si_p40_start"], milestones["si_p40_end"]),
            "sync_engine_lock_wait": ms(milestones["si_p40_end"], milestones["plan_start_lock_acquired"]),
            "compute_telemetry_and_state": ms(milestones["plan_start_lock_acquired"], commit),
        }
        rasterize = segments["compute_rasterize"] or 0.0
        age_commit = ages["state_commit"]
        return {"frame_id": frame, "kind": d.get("kind"), "reported_age_ms": None if reported is None
                else reported / 1e6, "age_at_commit_ms": age_commit,
                "age_at_worker_start_ms": ages["worker_start"], "age_at_si_p40_start_ms": ages["si_p40_start"],
                "first_milestone_over_rule": first_over, "ages_ms": ages, "segments_ms": segments,
                "over_age_before_si_p40": (ages["si_p40_start"] or 0) > RULE_NS / 1e6,
                "over_age_at_worker_start": (ages["worker_start"] or 0) > RULE_NS / 1e6,
                "rescued_if_si_p40_before_rasterize": (age_commit is not None
                                                       and age_commit - rasterize <= RULE_NS / 1e6),
                "rasterize_ms": rasterize,
                "rgb_receipt_matches_profiler_mark_ms": ms(self.rgb_marks.get(frame), rgb),
                "rgb_receipt_equals_trace": (self.trace_rgb.get(frame) == rgb) if rgb is not None else None,
                "radar_window_sha256": d.get("radar_window_sha256"),
                "carla_timestamp": d.get("carla_timestamp")}

    # -- aggregate ----------------------------------------------------------------
    def analyze(self) -> dict[str, Any]:
        fallbacks = [d for d in self.decisions if d.get("kind") == "FALLBACK"]
        classes = collections.Counter(self.reason_class(d.get("fallback_reasons") or ()) for d in fallbacks)
        # startup
        first_tick = self.ticks[0][0] if self.ticks else None
        startup = [d for d in fallbacks if self.reason_class(d.get("fallback_reasons") or ())
                   == "STARTUP_CLOCK_BRIDGE_WARMING"]
        first_admitted = next((d for d in self.decisions if d.get("kind") == "POLICY_DECISION"), None)
        first_pdcp = min((r["receipt_raw"] for r in self.pdcp), default=None)
        startup_report = {
            "frames": [int(d["frame_id"]) for d in startup],
            "first_route_tick_to_first_pdcp_row_ms": ms(first_tick, first_pdcp),
            "first_route_tick_to_first_admitted_decision_ms": ms(first_tick, self._boundary(first_admitted)
                                                                 if first_admitted else None),
            "first_admitted_frame": None if first_admitted is None else int(first_admitted["frame_id"]),
            "note": "the causal clock bridge needs PDCP anchors, which exist only once uplink traffic flows; "
                    "startup frames are therefore structural, not steady-state failures"}
        # scene
        scene_rows = [self.scene_row(d) for d in fallbacks
                      if self.reason_class(d.get("fallback_reasons") or ()) == "SCENE_STALE"]
        fresh_rows = [self.scene_row(d) for d in self.decisions if d.get("kind") == "POLICY_DECISION"]
        hold_rows = [self.scene_row(d) for d in self.decisions if d.get("kind") == "POLICY_HOLD"]

        def seg_dist(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            keys = rows[0]["segments_ms"].keys() if rows else ()
            return {k: dist(r["segments_ms"][k] for r in rows) for k in keys}

        def age_dist(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            keys = rows[0]["ages_ms"].keys() if rows else ()
            return {k: dist(r["ages_ms"][k] for r in rows) for k in keys}

        identity = {
            "frames_checked": len(fresh_rows) + len(hold_rows) + len(scene_rows),
            "rgb_receipt_equals_collector_trace": sum(1 for r in fresh_rows + hold_rows + scene_rows
                                                     if r["rgb_receipt_equals_trace"] is True),
            "rgb_receipt_within_2ms_of_profiler_mark": sum(
                1 for r in fresh_rows + hold_rows + scene_rows
                if r["rgb_receipt_matches_profiler_mark_ms"] is not None
                and 0 <= r["rgb_receipt_matches_profiler_mark_ms"] <= 2.0),
            "radar_window_digests_unique_over_sent_frames": len({
                d.get("radar_window_sha256") for d in self.decisions
                if int(d["frame_id"]) in self.transmitted}) == len(self.transmitted),
            "binding": "SensorFirstPlannerV2 refuses a plan unless the rasterized window digest equals the "
                       "P40 window digest; PlannedRun4FrameV2.verify refuses materialization on any "
                       "frame/capture/CARLA-timestamp/window drift (code path, fail-closed)"}
        scene_report = {
            "count": len(scene_rows),
            "reported_age_ms": dist(r["reported_age_ms"] for r in scene_rows),
            "age_at_commit_ms": dist(r["age_at_commit_ms"] for r in scene_rows),
            "first_milestone_over_rule": dict(collections.Counter(r["first_milestone_over_rule"]
                                                                   for r in scene_rows)),
            "over_age_before_si_p40": sum(1 for r in scene_rows if r["over_age_before_si_p40"]),
            "over_age_at_worker_start": sum(1 for r in scene_rows if r["over_age_at_worker_start"]),
            "rescued_if_si_p40_before_rasterize": sum(1 for r in scene_rows
                                                      if r["rescued_if_si_p40_before_rasterize"]),
            "ages_ms_stale": age_dist(scene_rows), "segments_ms_stale": seg_dist(scene_rows),
            "ages_ms_admitted_decisions": age_dist(fresh_rows),
            "segments_ms_admitted_decisions": seg_dist(fresh_rows),
            "rows": scene_rows}
        # radio
        radio_rows = []
        for d in fallbacks:
            klass = self.reason_class(d.get("fallback_reasons") or ())
            if klass not in ("MCS_STALE", "BACKLOG_STALE"):
                continue
            boundary = self._boundary(d)
            if boundary is None:
                continue
            for reason in d.get("fallback_reasons") or ():
                if reason.startswith("UE_PRIOR_NEW_DATA_UL_MCS_INDEX:STALE"):
                    radio_rows.append({"frame_id": int(d["frame_id"]), **classify_radio_stale(
                        boundary, int(reason.rsplit(":", 1)[1]), self.dci, dci_eligible, "prior_ul_mcs")})
                elif reason.startswith("UE_PRE_ACTION_RLC_BACKLOG_BYTES:STALE"):
                    radio_rows.append({"frame_id": int(d["frame_id"]), **classify_radio_stale(
                        boundary, int(reason.rsplit(":", 1)[1]), self.rlc, lambda r: True,
                        "pre_action_rlc_backlog")})
        radio_report = {"count": len(radio_rows),
                        "causes": dict(collections.Counter(r["cause"] for r in radio_rows)),
                        "stream_agrees_with_provider": sum(1 for r in radio_rows
                                                           if r["stream_age_agrees_with_provider_within_5ms"]),
                        "rows": radio_rows,
                        "dci_rows_total": len(self.dci), "dci_eligible_total": sum(1 for r in self.dci
                                                                                  if dci_eligible(r)),
                        "dci_source_to_reader_ms": dist((r["receipt_raw"] - r["event_raw"]) / 1e6
                                                        for r in self.dci if r["event_raw"] is not None)}
        return {"schema": SCHEMA, "root": str(self.root), "frames_planned": len(self.decisions),
                "frames_sent": len(self.transmitted),
                "kinds": dict(collections.Counter(d.get("kind") for d in self.decisions)),
                "fallbacks_total": len(fallbacks), "fallback_classes": dict(classes),
                "startup": startup_report, "scene_stale": scene_report, "radio_stale": radio_report,
                "identity": identity}


def summary_line(name: str, a: Mapping[str, Any]) -> str:
    s, r = a["scene_stale"], a["radio_stale"]
    return (f"| {name} | {a['fallbacks_total']} | {a['fallback_classes']} | "
            f"{s['count']} | {s['over_age_at_worker_start']} | {s['over_age_before_si_p40']} | "
            f"{s['rescued_if_si_p40_before_rasterize']} | {r['count']} | {r['causes']} |")


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", help="name=output_root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for item in args.captures:
        name, root = item.split("=", 1)
        results[name] = Capture(Path(root)).analyze()
        with (args.out / f"FRESHNESS_{name}.json").open("x", encoding="utf-8") as handle:
            json.dump(results[name], handle, indent=1, sort_keys=True, default=str)
    lines = ["| capture | fallbacks | classes | scene-stale | over-age at worker start | over-age before SI/P40 | "
             "rescued if SI/P40 before rasterize | radio-stale | radio causes |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [summary_line(n, a) for n, a in results.items()]
    text = "\n".join(lines) + "\n"
    (args.out / "SUMMARY.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
