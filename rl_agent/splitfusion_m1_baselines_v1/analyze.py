"""Failure-aware M1 handoff: exactly three PNGs plus tables/report/evidence hashes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_run5_full_live_analysis_v1 import report as LIVE
from rl_agent.splitfusion_run5_quality_reference_v1 import summarize as QUALITY

METHODS = ("Full local", "Static noAE", "Run-5 agent", "Full remote")
COLORS = ("#4c78a8", "#f58518", "#54a24b", "#b279a2")
SURFACE, INK, MUTED, GRID = "#fbfbfa", "#171717", "#5d5d5d", "#dedede"
RGB_BYTES = 2_764_800
RADAR_BYTES_PER_RETURN = 16
RADAR_PPS = 200_000
RADAR_SWEEP_MAX = 20_000       # 200k points/s at 10 sweep/s


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(values, ps=(50, 95, 99)):
    a = np.asarray(values, float)
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def style(ax):
    ax.set_facecolor(SURFACE); ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def load(agent_root, static_root, replay_path):
    arm, live_rows, ue = LIVE.rows(agent_root)
    live = LIVE.summarize(arm, live_rows, ue)
    replay = json.loads(Path(replay_path).read_bytes())
    paired = [row for row in replay["rows"] if row["admission"] == "ADMITTED"]
    paired_quality = QUALITY.analyze(paired)
    cell = next((Path(agent_root) / "capture/cells").iterdir())
    per_frame = list(csv.DictReader(open(cell / "per_frame_metrics.csv")))
    sent = [row for row in per_frame if row["prepare_status"] == "SENT"]
    wire = np.asarray([int(row["payload_bytes"]) for row in sent])
    windows = [int(row["window_returns"]) for row in sent]

    # Consecutive two-sweep windows are w_i=s_i+s_(i+1). The registered 200k
    # returns/s sensor bounds each 0.1-s sweep at 20k. Propagating that bound
    # leaves only two adjacent integer solutions for s_1; report the envelope.
    feasible = []
    for first in range(RADAR_SWEEP_MAX + 1):
        sweeps = [first]
        ok = True
        for window in windows:
            value = window - sweeps[-1]
            if not 0 <= value <= RADAR_SWEEP_MAX:
                ok = False; break
            sweeps.append(value)
        if ok: feasible.append(sweeps)
    if not feasible:
        raise ValueError("raw-radar overlapping-window counts have no bounded streaming solution")
    stream_payloads = [RGB_BYTES + RADAR_BYTES_PER_RETURN * np.asarray(sweeps[2:], int)
                       for sweeps in feasible]
    remote = {q: [float(np.percentile(a, q)) for a in stream_payloads] for q in (50, 99)}
    evidence_path = cell / "run4_phase6/PHASE6_UE_EVIDENCE.json"
    fixed_path = next((Path(static_root) / "capture").rglob("FIXED_ACTION_SUMMARY.json"))
    edge_log = next((Path(static_root) / "capture").rglob("edge_container.log"))
    fixed = json.loads(fixed_path.read_bytes())
    static_failure = dict(
        classification="PRE_FRAME_IMPLEMENTATION_STARTUP_FAULT__NOT_ZERO_COMPLETION",
        launch_returncode=1, offered=0, transmitted=0, completed=0,
        decisions=fixed["counters"]["decisions"], holds=fixed["counters"]["holds"],
        fallbacks=fixed["counters"]["fallbacks"], actor_calls=fixed["counters"]["actor_calls"],
        reason="edge preload exited: registration verifier imported SciPy in the slim edge image; SciPy absent",
        no_retry=True, live_quality="UNAVAILABLE", latency="UNAVAILABLE", payload="UNAVAILABLE",
        edge_log=str(edge_log), edge_log_sha256=sha(edge_log), fixed_summary_sha256=sha(fixed_path))
    return dict(arm=arm, rows=live_rows, ue=ue, live=live, replay=replay, paired=paired,
                paired_quality=paired_quality, sent=sent, wire=wire, feasible=feasible,
                remote=remote, cell=cell, evidence_path=evidence_path,
                static_failure=static_failure)


def quality_figure(data, out):
    q = data["paired_quality"]
    specs = (("vehicle", "recall_within_tol", "Vehicle recall ≤1 m"),
             ("vehicle", "precision", "Vehicle precision"),
             ("vehicle", "seg_iou_pooled", "Vehicle seg. IoU"),
             ("person", "recall_within_tol", "Person recall ≤1.2 m"),
             ("person", "precision", "Person precision"),
             ("person", "seg_iou_pooled", "Person seg. IoU"))
    x = np.arange(len(specs)); width = .34
    full = [q[c]["branches"]["full_local"][m] for c, m, _ in specs]
    agent = [q[c]["branches"]["agent_selected"][m] for c, m, _ in specs]
    fig, ax = plt.subplots(figsize=(13.2, 5.8), facecolor=SURFACE)
    b1 = ax.bar(x - width / 2, full, width, color=COLORS[0], label="Full local — offline, same retained inputs")
    b2 = ax.bar(x + width / 2, agent, width, color=COLORS[2], label="Run-5 agent — archived live outputs")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .018,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [s[2] for s in specs], rotation=18, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("pooled metric"); style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.set_title("Quality on 99 outcome-independent retained decisions (exact replay parity)", loc="left", color=INK)
    fig.text(.5, .015, "Static noAE: unavailable (pre-frame startup fault; no live prediction).  Full remote: no inference by design.",
             ha="center", color="#9c3f00", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(out / "01_quality.png", dpi=160, facecolor=SURFACE); plt.close(fig)


def latency_figure(data, out):
    rows = [row for row in data["rows"] if row.get("edge_compute_end") is not None]
    primary = [row["edge_compute_end"] for row in rows]
    capture = [row["edge_compute_end"] - row["capture_world_tick_end"] for row in rows
               if row.get("capture_world_tick_end") is not None]
    feedback = [row["ue_receipt"] for row in rows if row.get("ue_receipt") is not None]
    p = pct(primary)
    counts = data["ue"]["counters"]
    offered = len(data["ue"]["decisions"])
    transmitted = len(data["ue"]["transmitted_identities"])
    completed = json.loads((data["cell"] / "run4_phase6/edge_report.json").read_bytes())["counters"]["direct_map_publications"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor=SURFACE)
    ax = axes[0]
    bars = ax.bar([0, 1, 2], [p["p50"], p["p95"], p["p99"]], color=COLORS[2], width=.62)
    for bar, name in zip(bars, ("P50", "P95", "P99")):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4, f"{bar.get_height():.1f} ms",
                ha="center", fontsize=9)
    ax.set_xticks([0, 1, 2], ["P50", "P95", "P99"]); ax.set_ylabel("ms from action-open")
    ax.axhline(170, color="#9c3f00", ls="--", lw=1, label="170 ms feedback contract (context only)")
    ax.set_ylim(0, max(230, p["p99"] * 1.18)); style(ax); ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"Run-5 agent edge-compute completion (n={len(primary)})", loc="left")
    ax.text(.02, .94, "Static noAE: N/A — no frame reached edge READY", transform=ax.transAxes,
            va="top", color="#9c3f00", fontsize=9)

    ax = axes[1]; x = np.arange(2); width=.22
    series = (("offered", [offered, 0]), ("transmitted", [transmitted, 0]), ("completed", [completed, 0]))
    for i, (label, values) in enumerate(series):
        bars=ax.bar(x+(i-1)*width, values, width, label=label,
                    color=("#9ecae9", "#4292c6", "#08519c")[i])
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+18, str(int(bar.get_height())),
                    ha="center", fontsize=8)
    ax.set_xticks(x, ["Run-5 agent", "Static noAE\n(startup fault)"]); ax.set_ylim(0, 1140)
    ax.set_ylabel("input frames"); style(ax); ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title("Frame accounting and completion", loc="left")
    ax.text(.02, .76, f"agent: {counts['policy_decisions']} decisions / {counts['policy_holds']} holds / "
            f"{counts['fallbacks']} fallbacks\ncompleted fraction: {completed}/{transmitted} = {completed/transmitted:.3f}",
            transform=ax.transAxes, fontsize=9, color=INK)
    secondary = dict(capture_to_prediction_ms=pct(capture), feedback_receipt_ms=pct(feedback))
    fig.suptitle("Latency and counts — primary latency excludes stale/timeout/zero substitutions", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out / "02_latency_and_counts.png", dpi=160, facecolor=SURFACE); plt.close(fig)
    return dict(primary_action_open_to_edge_compute_ms=dict(n=len(primary), **p), secondary=secondary,
                offered=offered, transmitted=transmitted, completed=completed,
                decisions=counts["policy_decisions"], holds=counts["policy_holds"], fallbacks=counts["fallbacks"])


def payload_figure(data, out):
    wire = data["wire"]
    agent = {50: float(np.percentile(wire, 50)), 99: float(np.percentile(wire, 99))}
    remote = {q: max(data["remote"][q]) for q in (50, 99)}   # 16-byte conservative envelope at P50
    values = np.asarray([[0, 0], [np.nan, np.nan], [agent[50], agent[99]], [remote[50], remote[99]]]) / 1e6
    fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor=SURFACE)
    x=np.arange(4); width=.34
    for j, (q, hatch) in enumerate(((50, ""), (99, "//"))):
        bars=ax.bar(x+(j-.5)*width, values[:,j], width, color=COLORS, hatch=hatch,
                    edgecolor="white", label=f"P{q}")
        for i,bar in enumerate(bars):
            if np.isfinite(values[i,j]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.06,
                        f"{values[i,j]:.3f}", ha="center", fontsize=8)
    ax.text(1, .18, "N/A\npre-frame fault", ha="center", color="#9c3f00", fontsize=9)
    ax.set_xticks(x, METHODS); ax.set_ylabel("MB per transmitted input (10⁶ bytes)")
    ax.set_ylim(0, 3.55); style(ax); ax.legend(frameon=False, ncol=2)
    ax.set_title("Application payload P50/P99 — measured agent, calculated raw full remote", loc="left")
    ax.text(.01, .98, "Full local: 0 input uplink. Agent: SFD4 wire, all 1,000 frames (455 B framing/sample).\n"
            "Full remote: RGB 2,764,800 B + 16 B per unique new radar return; overlapping sweeps deduplicated; no network/inference.",
            transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)
    fig.tight_layout(); fig.savefig(out / "03_payload.png", dpi=160, facecolor=SURFACE); plt.close(fig)
    return dict(
        full_local=dict(p50=0, p99=0, kind="calculated", representation_bytes=0, framing_bytes=0),
        static_noae=dict(p50=None, p99=None, kind="unavailable_pre_frame_fault"),
        agent=dict(p50=agent[50], p99=agent[99], n=len(wire), kind="measured_SFD4_application_wire",
                   representation_p50=agent[50]-455, representation_p99=agent[99]-455,
                   sfd4_framing_bytes=455, transport_overhead="not included"),
        full_remote=dict(p50=remote[50], p99=remote[99], kind="calculated_streaming_payload_only",
            rgb_bytes=RGB_BYTES, radar_bytes_per_unique_return=RADAR_BYTES_PER_RETURN,
            unique_sweep_solutions=len(data["feasible"]), p50_envelope=data["remote"][50],
            p99_envelope=data["remote"][99], framing_bytes=0, inference=False, network=False))


def run(agent_root, static_root, replay, campaign, out):
    out=Path(out); out.mkdir(parents=False, exist_ok=False)
    data=load(agent_root, static_root, replay)
    quality_figure(data,out); latency=latency_figure(data,out); payload=payload_figure(data,out)
    q=data["paired_quality"]
    quality=dict(n=99, source="outcome-independent retained agent decisions",
        full_local={c:{m:q[c]["branches"]["full_local"][m] for m in
                      ("recall_within_tol","precision","seg_iou_pooled")} for c in ("vehicle","person")},
        agent_live_archived={c:{m:q[c]["branches"]["agent_selected"][m] for m in
                               ("recall_within_tol","precision","seg_iou_pooled")} for c in ("vehicle","person")},
        static_live="UNAVAILABLE_PRE_FRAME_STARTUP_FAULT", full_remote="NO_INFERENCE_BY_DESIGN",
        agent_all_completed_predictions=data["live"]["quality_all_scored"])
    summary=dict(schema="splitfusion.m1_baselines_v1.failure_aware_summary.v1",
        status="M1_INCOMPLETE_STATIC_ARM_PRE_FRAME_FAULT_NO_RETRY",
        actor_weights_sha256="bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce",
        sessions="separate; matched configuration/route; not identical scenes",
        quality=quality, latency_and_counts=latency, payload=payload,
        agent_outcomes=data["live"]["outcome_classes"],
        agent_feedback_timely=dict(count=data["live"]["timely"], decisions=data["live"]["decisions"],
                                   fraction=data["live"]["timely"]/data["live"]["decisions"]),
        static_failure=data["static_failure"],
        no_substitution="old 100-frame static and offline noAE predictions are not used as static live evidence")
    (out/"SUMMARY.json").write_text(json.dumps(summary,indent=1,sort_keys=True,default=str))
    (out/"STATIC_FAILURE.json").write_text(json.dumps(data["static_failure"],indent=1,sort_keys=True))
    table=("| Method | Quality | Primary latency P50/P95/P99 | Offered / transmitted / completed | Payload P50 / P99 |\n"
           "|---|---|---:|---:|---:|\n"
           f"| Full local | Offline retained n=99 | N/A | N/A | 0 / 0 B |\n"
           f"| Static noAE | **Unavailable: pre-frame startup fault** | N/A | 0 / 0 / 0 | N/A |\n"
           f"| Run-5 agent | Live archived, retained n=99 (all completed n=495 separately) | "
           f"{latency['primary_action_open_to_edge_compute_ms']['p50']:.1f} / "
           f"{latency['primary_action_open_to_edge_compute_ms']['p95']:.1f} / "
           f"{latency['primary_action_open_to_edge_compute_ms']['p99']:.1f} ms | "
           f"{latency['offered']} / {latency['transmitted']} / {latency['completed']} | "
           f"{payload['agent']['p50']/1e3:.1f} / {payload['agent']['p99']/1e3:.1f} kB |\n"
           f"| Full remote | No inference by design | N/A | Calculated only | "
           f"{payload['full_remote']['p50']/1e6:.3f} / {payload['full_remote']['p99']/1e6:.3f} MB |\n")
    (out/"TABLE.md").write_text(table)
    report=("# M1 baseline handoff\n\n"
        "M1 is incomplete: the fresh agent arm is valid, but the sole authorized static arm failed before edge READY "
        "and before any frame. The failure is an implementation/startup fault, not a zero-completion outcome. No retry "
        "was made, and no old or offline result was substituted for static live evidence.\n\n"
        "The agent arm completed 1,000/1,000 transmitted inputs and edge publications: 495 policy decisions, 495 holds, "
        "10 typed fallbacks. All 495 policy decisions produced scored predictions; 410 feedback replies were within the "
        "170 ms feedback contract (370 quality-defined, 40 quality-undefined) and 85 were late. Primary action-open to "
        f"edge-compute completion was {latency['primary_action_open_to_edge_compute_ms']['p50']:.1f}/"
        f"{latency['primary_action_open_to_edge_compute_ms']['p95']:.1f}/"
        f"{latency['primary_action_open_to_edge_compute_ms']['p99']:.1f} ms at P50/P95/P99.\n\n"
        "All 99 reached retained samples resolved to exact GT and passed three-branch replay parity. The quality figure "
        "uses full-local offline output and the corresponding archived live agent output on those same samples. Static "
        "quality and latency remain unavailable. Full remote is payload arithmetic only.\n\n"+table+"\n"
        "## Decision handoff\n\n"
        "- **M2:** keep closed. The static startup failure is an evidence-collection fault, not evidence for or against "
        "learned action selection, and it does not justify reopening Run-5 policy work.\n"
        "- **M3:** do not start the full-route phase from this incomplete four-configuration comparison. If a complete "
        "M1 is still required, explicitly authorize one new static attempt after replacing the edge-side verifier with "
        "an import-safe check, then seal that attempt as a new session. If the incomplete result is accepted instead, "
        "M3 may consume only the agent, full-local and calculated full-remote facts while carrying static as unavailable. "
        "Neither branch was executed here.\n")
    (out/"REPORT.md").write_text(report)
    inputs=[Path(agent_root)/"registration.json",Path(static_root)/"registration.json",Path(campaign)/"FREEZE.json",
            Path(campaign)/"VERDICT_agent.json",Path(replay),data["evidence_path"],Path(__file__)]
    (out/"INPUT_SHA256.json").write_text(json.dumps({str(p):sha(p) for p in inputs},indent=1,sort_keys=True))
    (out/"OUTPUT_SHA256.json").write_text(json.dumps({p.name:sha(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name!="OUTPUT_SHA256.json"},indent=1,sort_keys=True))
    print(json.dumps(dict(status=summary["status"],pngs=sorted(p.name for p in out.glob("*.png")),
                          agent_frames=latency["transmitted"],static_frames=0),indent=1))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent-root",required=True);p.add_argument("--static-root",required=True)
    p.add_argument("--replay",required=True);p.add_argument("--campaign",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();run(a.agent_root,a.static_root,a.replay,a.campaign,a.out)
