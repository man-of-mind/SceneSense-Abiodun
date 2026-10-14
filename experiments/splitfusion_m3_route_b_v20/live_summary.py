"""Summarise the live-viewer side of the V20 run (display evidence only; the integration verdict is separate).

Inputs: live_view_01/VIEWER_EVENTS.jsonl (viewer frame/connection/report/flag log), live_view_01/recording/
(STATE_1HZ.jsonl, RECORDING.json), live_view_01/VIEWER_CPU.jsonl, and the capture's live_view tap status files.
Writes live_view_01/LIVE_SUMMARY.json (create-only).
"""
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
LIVE = HERE / "live_view_01"


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).is_file() else []


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return round(values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))], 2)


def number(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def dist(values):
    values = [number(v) for v in values]
    values = [v for v in values if v is not None]
    return dict(n=len(values), p50=pct(values, .5), p95=pct(values, .95), p99=pct(values, .99),
                max=round(max(values), 2) if values else None)


def main(cell):
    events = rows(LIVE / "VIEWER_EVENTS.jsonl")
    frames = [e for e in events if e["kind"] == "frame"]
    reports = [e for e in events if e["kind"] == "client_report"]
    stats = [r for r in reports if r.get("report_kind") == "stats"]
    by_client = {}
    for r in stats:
        by_client.setdefault(r.get("client", "browser"), []).append(r)
    states = [r for r in reports if r.get("report_kind") == "state"]
    flags = [r for r in reports if r.get("report_kind") == "flag"]
    gaps = []
    for a, b in zip(frames, frames[1:]):
        gap = (b["capture_ns"] - a["capture_ns"]) / 1e6
        if gap > 500:
            gaps.append(dict(after_frame=a["frame_id"], next_frame=b["frame_id"], capture_gap_ms=round(gap, 1)))
    clients = {}
    for r in stats:
        c = clients.setdefault(r.get("client", "browser"), dict(reports=0))
        c["reports"] += 1
    summary = dict(
        schema="scenesense.live_view.run_summary.v1",
        frames_emitted=len(frames),
        pose_bound=sum(1 for f in frames if f["pose_error"] is None),
        pose_errors={k: sum(1 for f in frames if f["pose_error"] == k) for k in {f["pose_error"] for f in frames} if k},
        heading_join={k: sum(1 for f in frames if f["heading_join"] == k) for k in {f["heading_join"] for f in frames}},
        capture_to_install_ms=dist([f["capture_to_install_ms"] for f in frames]),
        install_to_tap_written_ms=dist([f["install_to_tap_written_ms"] for f in frames]),
        capture_to_viewer_read_ms=dist([f["capture_to_viewer_read_ms"] for f in frames]),
        capture_to_viewer_emit_ms=dist([f["capture_to_emit_ms"] for f in frames]),
        capture_gaps_over_500ms=gaps,
        refused={k: sum(1 for e in events if e["kind"] == "refused" and e.get("reason") == k)
                 for k in {e.get("reason") for e in events if e["kind"] == "refused"}},
        connections=[dict(kind=e["kind"], client=e.get("client"), wall_ns=e["wall_ns"], sent=e.get("sent"),
                          dropped=e.get("dropped"), duration_s=e.get("duration_s"))
                     for e in events if e["kind"] in ("client_connect", "client_disconnect")],
        browser_stats={client: dict(
            reports=len(v),
            frame_interval_p50_ms=dist([r.get("frame_interval_p50") for r in v]),
            frame_interval_p95_ms=dist([r.get("frame_interval_p95") for r in v]),
            frame_interval_max_ms=dist([r.get("frame_interval_max") for r in v]),
            draw_p95_ms=dist([r.get("draw_p95") for r in v]),
            capture_to_display_p50_ms=dist([r.get("display_age_p50") for r in v]),
            capture_to_display_p95_ms=dist([r.get("display_age_p95") for r in v]),
            reconnects_max=max((r.get("reconnects") or 0) for r in v),
            hidden_reports=sum(1 for r in v if r.get("hidden")),
            states_reported=sorted({r.get("state") for r in v})) for client, v in by_client.items()},
        state_changes=[dict(client=r.get("client"), state=r.get("state"), wall_ns=r["wall_ns"], frame_id=r.get("frame_id"))
                       for r in states],
        flags=[dict(n=r.get("n"), client=r.get("client"), wall_ns=r["wall_ns"], frame_id=r.get("frame_id"),
                    capture_ns=r.get("capture_ns"), displayed_capture_age_ms=r.get("displayed_capture_age_ms"),
                    shown=r.get("shown")) for r in flags],
        feed_stop=[e for e in events if e["kind"] == "feed_stop"],
        feed_errors=[e for e in events if e["kind"] == "feed_error"])
    cpu = rows(LIVE / "VIEWER_CPU.jsonl")
    summary["viewer_cpu"] = dict(samples=len(cpu), total_cpu_pct=dist([c["total_cpu_pct"] for c in cpu]),
                                 rss_mb_max=max((p["rss_mb"] for c in cpu for p in c["processes"].values()), default=None),
                                 cpus=sorted({tuple(p["cpus"]) for c in cpu for p in c["processes"].values()}))
    recording = LIVE / "recording" / "RECORDING.json"
    summary["recording"] = json.loads(recording.read_text()) if recording.is_file() else None
    state_rows = rows(LIVE / "recording" / "STATE_1HZ.jsonl")
    summary["recorder_states"] = {s: sum(1 for r in state_rows if r["state"] == s) for s in {r["state"] for r in state_rows}}
    taps = {}
    for name in ("map_tap_status.json", "ue_pose_tap_status.json"):
        path = Path(cell) / "live_view" / name
        taps[name] = json.loads(path.read_text()) if path.is_file() else None
    summary["tap_status"] = taps
    out = LIVE / "LIVE_SUMMARY.json"
    with out.open("x") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    print(json.dumps({k: summary[k] for k in ("frames_emitted", "pose_bound", "heading_join", "capture_to_viewer_emit_ms",
                                               "refused", "recorder_states")}, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
