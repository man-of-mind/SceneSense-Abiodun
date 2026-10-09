"""M1 revision 2: completed static arm and denominator-honest CDFs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_run5_full_live_analysis_v1 import report as LIVE
from rl_agent.splitfusion_run5_quality_reference_v1 import summarize as QUALITY


SURFACE = "#fbfbfa"
INK = "#171717"
MUTED = "#5d5d5d"
GRID = "#dedede"
LOCAL = "#4c78a8"
STATIC = "#f58518"
AGENT = "#54a24b"
REMOTE = "#b279a2"
RGB_BYTES = 2_764_800
RADAR_BYTES_PER_RETURN = 16
RADAR_SWEEP_MAX = 20_000
SFD4_FRAMING_BYTES = 455


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def find_one(root, name):
    values = list(Path(root).rglob(name))
    if len(values) != 1:
        raise ValueError(f"expected one {name} below {root}, found {len(values)}")
    return values[0]


def percentiles(values, ps=(50, 95, 99)):
    values = np.asarray(values, dtype=float)
    return {f"p{p}": float(np.percentile(values, p)) for p in ps}


def style(ax, xgrid=False):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="both" if xgrid else "y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def sent_rows(root):
    path = find_one(Path(root) / "capture", "per_frame_metrics.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return path, rows, [row for row in rows if row["prepare_status"] == "SENT"]


def remote_payload_envelope(agent_sent):
    windows = [int(row["window_returns"]) for row in agent_sent]
    feasible = []
    for first in range(RADAR_SWEEP_MAX + 1):
        sweeps = [first]
        for window in windows:
            value = window - sweeps[-1]
            if not 0 <= value <= RADAR_SWEEP_MAX:
                break
            sweeps.append(value)
        else:
            feasible.append(sweeps)
    if not feasible:
        raise ValueError("raw-radar overlapping-window counts have no bounded streaming solution")
    payloads = [RGB_BYTES + RADAR_BYTES_PER_RETURN * np.asarray(sweeps[2:], dtype=int)
                for sweeps in feasible]
    envelope = {q: [float(np.percentile(values, q)) for values in payloads]
                for q in (50, 99)}
    return feasible, payloads, envelope


def load(agent_root, static_root, replay_path, static_capture, preflight_path):
    agent_root = Path(agent_root)
    static_root = Path(static_root)
    static_capture = Path(static_capture)
    arm, agent_rows, agent_ue = LIVE.rows(agent_root)
    agent_live = LIVE.summarize(arm, agent_rows, agent_ue)
    static_ue_path = find_one(static_root / "capture", "PHASE6_UE_EVIDENCE.json")
    static_ue = read_json(static_ue_path)
    agent_edge_path = find_one(agent_root / "capture", "edge_report.json")
    static_edge_path = find_one(static_root / "capture", "edge_report.json")
    agent_edge = read_json(agent_edge_path)
    static_edge = read_json(static_edge_path)
    fixed_path = find_one(static_root / "capture", "FIXED_ACTION_SUMMARY.json")
    fixed = read_json(fixed_path)
    agent_csv, agent_all_route, agent_sent = sent_rows(agent_root)
    static_csv, static_all_route, static_sent = sent_rows(static_root)
    replay = read_json(replay_path)
    paired = [row for row in replay["rows"] if row["admission"] == "ADMITTED"]
    quality = QUALITY.analyze(paired)
    feasible, remote_payloads, remote_envelope = remote_payload_envelope(agent_sent)
    return dict(
        agent_root=agent_root, static_root=static_root, arm=arm,
        agent_rows=agent_rows, agent_ue=agent_ue, agent_live=agent_live,
        static_ue=static_ue, agent_edge=agent_edge, static_edge=static_edge,
        fixed=fixed, quality=quality, replay=replay, paired=paired,
        agent_csv=agent_csv, static_csv=static_csv,
        agent_all_route=agent_all_route, static_all_route=static_all_route,
        agent_sent=agent_sent, static_sent=static_sent,
        feasible=feasible, remote_payloads=remote_payloads,
        remote_envelope=remote_envelope,
        agent_ue_path=find_one(agent_root / "capture", "PHASE6_UE_EVIDENCE.json"),
        static_ue_path=static_ue_path, agent_edge_path=agent_edge_path,
        static_edge_path=static_edge_path, fixed_path=fixed_path,
        static_raw_verdict=read_json(static_capture / "VERDICT_static.json"),
        static_verdict=read_json(static_capture / "VERDICT_static_v2.json"),
        static_retention=read_json(static_root / "RETENTION_RESULT.json"),
        preflight=read_json(preflight_path),
    )


def conservative_cdf(branch):
    errors = np.asarray(sorted(branch["errors"]), dtype=float)
    x = np.concatenate(([0.0], errors))
    y = np.concatenate(([0.0], np.arange(1, len(errors) + 1) / branch["eligible"]))
    return x, y


def quality_figure(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), facecolor=SURFACE)
    specs = (("vehicle", 1.0, "Vehicles"), ("person", 1.2, "Pedestrians"))
    for ax, (category, threshold, label) in zip(axes, specs):
        for branch_name, branch_label, color in (
                ("full_local", "Full local — offline reference", LOCAL),
                ("agent_selected", "Run-5 agent — archived live output", AGENT)):
            branch = data["quality"][category]["branches"][branch_name]
            x, y = conservative_cdf(branch)
            ax.step(x, y, where="post", linewidth=2.2, color=color, label=branch_label)
            ax.plot([3.0], [len(branch["errors"]) / branch["eligible"]],
                    marker="o", markersize=4, color=color)
        local = data["quality"][category]["branches"]["full_local"]
        agent = data["quality"][category]["branches"]["agent_selected"]
        ax.axvline(threshold, color="#9c3f00", linestyle="--", linewidth=1.3,
                   label=f"{threshold:.1f} m reporting threshold")
        ax.set_xlim(0, 3.05)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("matched XY localization error t (m)")
        ax.set_ylabel("eligible-GT completion fraction")
        ax.set_title(f"{label}: all eligible GT in denominator (n={local['eligible']})",
                     loc="left", fontsize=11, color=INK)
        ax.text(.98, .08,
                f"at {threshold:.1f} m: local {local['within_tol']}/{local['eligible']} = "
                f"{local['recall_within_tol']:.3f}\n"
                f"agent {agent['within_tol']}/{agent['eligible']} = "
                f"{agent['recall_within_tol']:.3f}\n"
                f"misses: local {local['misses']}; agent {agent['misses']}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color=INK, bbox=dict(facecolor=SURFACE, edgecolor=GRID, alpha=.92))
        style(ax, xgrid=True)
        ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    fig.suptitle("Conservative localization CDF — same-input paired comparison, 99 retained decisions",
                 fontsize=13, color=INK)
    fig.text(.5, .035,
             "CDF(t) = matched eligible GT with error ≤ t / all eligible GT; misses remain in the denominator.\n"
             "Static is a different live session (500 decisions, 0 completed predictions): no static quality CDF.  "
             "Full remote has no inference.",
             ha="center", fontsize=9.2, color=MUTED)
    fig.tight_layout(rect=(0, .105, 1, .94))
    fig.savefig(out / "01_quality_cdf.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)


def latency_data(data):
    agent_values = np.asarray([row["edge_compute_end"] for row in data["agent_rows"]
                               if row.get("edge_compute_end") is not None], dtype=float)
    agent_decisions = int(data["agent_ue"]["counters"]["policy_decisions"])
    static_decisions = int(data["static_ue"]["counters"]["policy_decisions"])
    if len(agent_values) > agent_decisions:
        raise ValueError("agent completions exceed decision-frame denominator")
    if int(data["static_edge"].get("publications", 0)) != 0:
        raise ValueError("revision-2 zero-completion rendering only accepts zero static publications")
    p = percentiles(agent_values)
    return dict(
        agent_values=agent_values, agent_decisions=agent_decisions,
        static_decisions=static_decisions, agent_completed=len(agent_values),
        static_completed=0, agent_quantiles=p,
        agent_by_170=int(np.sum(agent_values <= 170.0)),
    )


def latency_figure(data, out):
    latency = latency_data(data)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": (1.25, 1)})
    ax = axes[0]
    values = np.sort(latency["agent_values"])
    x = np.concatenate(([0.0], values))
    y = np.concatenate(([0.0], np.arange(1, len(values) + 1) /
                        latency["agent_decisions"]))
    xmax = max(220.0, math.ceil(float(values.max()) / 10) * 10)
    ax.step(x, y, where="post", color=AGENT, linewidth=2.3,
            label=f"Run-5 agent: {len(values)}/{latency['agent_decisions']} completed")
    ax.plot([0, xmax], [0, 0], color=STATIC, linewidth=2.3,
            label=f"Static noAE: 0/{latency['static_decisions']} completed")
    ax.axvline(170, color="#9c3f00", linestyle="--", linewidth=1.4,
               label="170 ms — original feedback deadline")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-.025, 1.025)
    ax.set_xlabel("time from decision action-open (ms)")
    ax.set_ylabel("fraction of all decision frames completed by t")
    ax.set_title("Prediction-completion CDF", loc="left", fontsize=11, color=INK)
    ax.text(.97, .12,
            f"agent P50/P95/P99: {latency['agent_quantiles']['p50']:.1f} / "
            f"{latency['agent_quantiles']['p95']:.1f} / "
            f"{latency['agent_quantiles']['p99']:.1f} ms\n"
            f"by 170 ms: {latency['agent_by_170']}/{latency['agent_decisions']} = "
            f"{latency['agent_by_170']/latency['agent_decisions']:.3f}\n"
            "static zero line is an outcome fraction,\nnot a measured latency distribution",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color=INK, bbox=dict(facecolor=SURFACE, edgecolor=GRID, alpha=.94))
    style(ax, xgrid=True)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")

    ax = axes[1]
    labels = ["Run-5 agent", "Static noAE"]
    completed = np.asarray([latency["agent_completed"], 0])
    incomplete = np.asarray([0, latency["static_decisions"]])
    xpos = np.arange(2)
    ax.bar(xpos, completed, color=AGENT, width=.58, label="completed prediction")
    ax.bar(xpos, incomplete, bottom=completed, color="#d9d9d9", width=.58,
           label="incomplete / timeout")
    for i, total in enumerate(completed + incomplete):
        ax.text(i, total + 12, f"{completed[i]}/{total} completed",
                ha="center", fontsize=9, color=INK)
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 575)
    ax.set_ylabel("decision frames")
    ax.set_title("Decision outcomes (different live sessions)", loc="left",
                 fontsize=11, color=INK)
    ax.text(.5, .73,
            "AGENT SESSION\n1,000 offered / 1,000 transmitted / 1,000 maps\n"
            "495 decisions / 495 holds / 10 fallbacks\n\n"
            "STATIC SESSION\n1,000 offered / 1,000 transmitted / 0 maps\n"
            "500 decisions / 500 holds / 0 fallbacks\n"
            "500 TIMEOUT resolutions; 21 stale-before-edge +\n"
            "1 stale-before-map terminals; no published prediction",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.4,
            color=INK, bbox=dict(facecolor=SURFACE, edgecolor=GRID, alpha=.94))
    style(ax)
    ax.legend(frameon=False, fontsize=8.2, loc="upper center")
    fig.suptitle("Completion latency and outcomes — incomplete decisions remain in each denominator",
                 fontsize=13, color=INK)
    fig.text(.5, .022,
             "Agent and static are separate sessions, not paired scenes. STALE receipts, timeout timestamps, and "
             "the static stale-before-map compute are excluded from prediction latency. Full remote has no latency CDF.",
             ha="center", fontsize=9.1, color=MUTED)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(out / "02_prediction_completion_cdf.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return latency


def payload_data(data):
    agent_wire = np.asarray([int(row["payload_bytes"]) for row in data["agent_sent"]], dtype=float)
    static_wire = np.asarray([int(row["payload_bytes"]) for row in data["static_sent"]], dtype=float)
    agent_dgrams = np.asarray([int(row["datagrams"]) for row in data["agent_sent"]], dtype=float)
    static_dgrams = np.asarray([int(row["datagrams"]) for row in data["static_sent"]], dtype=float)
    agent = percentiles(agent_wire, (50, 99))
    static = percentiles(static_wire, (50, 99))
    remote = {f"p{q}": max(data["remote_envelope"][q]) for q in (50, 99)}
    return dict(agent_wire=agent_wire, static_wire=static_wire,
                agent=agent, static=static, remote=remote,
                agent_datagrams=percentiles(agent_dgrams, (50, 99)),
                static_datagrams=percentiles(static_dgrams, (50, 99)))


def payload_figure(data, out):
    payload = payload_data(data)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": (1.08, 1)})
    ax = axes[0]
    labels = ("Full local", "Static noAE", "Run-5 agent", "Full remote")
    colors = (LOCAL, STATIC, AGENT, REMOTE)
    values = np.asarray([
        [0.0, 0.0],
        [payload["static"]["p50"], payload["static"]["p99"]],
        [payload["agent"]["p50"], payload["agent"]["p99"]],
        [payload["remote"]["p50"], payload["remote"]["p99"]],
    ]) / 1e6
    xpos = np.arange(4)
    width = .34
    for j, (quantile, hatch) in enumerate((("P50", ""), ("P99", "//"))):
        bars = ax.bar(xpos + (j - .5) * width, values[:, j], width,
                      color=colors, hatch=hatch, edgecolor="white", label=quantile)
        for i, bar in enumerate(bars):
            if values[i, j] == 0:
                if j == 0:
                    ax.text(bar.get_x() + bar.get_width()/2, .055, "0",
                            ha="center", fontsize=8)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .075,
                        f"{values[i, j]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("application payload per transmitted input (MB, 10⁶ bytes)")
    ax.set_ylim(0, 4.05)
    ax.set_title("P50 / P99 payload", loc="left", fontsize=11, color=INK)
    ax.text(.02, .97,
            "Measured: static + agent, n=1,000 each\n"
            "Calculated: local + full remote\n"
            "Full remote = raw RGB + unique radar returns only",
            transform=ax.transAxes, va="top", fontsize=8.4, color=MUTED)
    style(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8.5)

    ax = axes[1]
    for values_, label, color in (
            (payload["agent_wire"], "Run-5 agent measured (n=1,000)", AGENT),
            (payload["static_wire"], "Static noAE measured (n=1,000)", STATIC)):
        values_ = np.sort(values_) / 1e6
        y = np.arange(1, len(values_) + 1) / len(values_)
        ax.step(values_, y, where="post", color=color, linewidth=2.2, label=label)
    ax.set_xscale("log")
    ax.set_xlim(.05, 4.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("SFD4 application payload (MB, log scale)")
    ax.set_ylabel("fraction of all transmitted inputs")
    ax.set_title("Empirical payload CDF (different sessions)", loc="left",
                 fontsize=11, color=INK)
    ax.text(.03, .16,
            "SFD4 wire = representation + 455 B framing\n"
            "UDP/IP/radio overhead excluded\n"
            "Full remote: calculated bars only; no empirical CDF",
            transform=ax.transAxes, fontsize=8.4, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, alpha=.94))
    style(ax, xgrid=True)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left")
    fig.suptitle("Input payload — separately measured live sessions and calculated references",
                 fontsize=13, color=INK)
    fig.text(.5, .023,
             "Payload denominator: every transmitted input frame. Representation P50/P99 equals each measured "
             "SFD4 bar minus 455 B. Full remote is payload arithmetic only—no latency or quality result.",
             ha="center", fontsize=9.1, color=MUTED)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(out / "03_payload.png", dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return payload


def route_counts(rows):
    counts = {}
    for row in rows:
        counts[row["prepare_status"]] = counts.get(row["prepare_status"], 0) + 1
    return counts


def summarized_quality(data):
    result = dict(
        comparison="SAME_INPUT_PAIRED__99_OUTCOME_INDEPENDENT_RETAINED_AGENT_DECISIONS",
        cdf_denominator="all eligible GT objects; misses remain in denominator",
        static="UNAVAILABLE__SEPARATE_SESSION__ZERO_COMPLETED_PREDICTIONS",
        full_remote="NO_INFERENCE_BY_DESIGN",
    )
    for category in ("vehicle", "person"):
        result[category] = {}
        for branch in ("full_local", "agent_selected"):
            values = data["quality"][category]["branches"][branch]
            result[category][branch] = {key: values[key] for key in (
                "samples", "eligible", "matched_3m", "misses", "false_positives",
                "within_tol", "recall_within_tol", "precision", "seg_iou_pooled",
                "err_p50", "err_p90", "err_p95", "err_max")}
    return result


def build_summary(data, latency, payload):
    agent_counts = data["agent_ue"]["counters"]
    static_counts = data["static_ue"]["counters"]
    agent_maps = int(data["agent_edge"]["counters"]["direct_map_publications"])
    static_maps = int(data["static_edge"].get("publications", 0))
    return dict(
        schema="splitfusion.m1_baselines_v1.completed_summary.v2",
        status="M1_COMPLETE_STATIC_VALID_ZERO_COMPLETION_OVERLOAD_OUTCOME",
        session_comparability=dict(
            quality_full_local_vs_agent="same-input paired, 99 retained agent-session decisions",
            agent_vs_static="different live sessions; matched registered budget/route/cadence/transport",
        ),
        frame_accounting=dict(
            definition=("offered = registered experiment frames entering decision/hold/fallback selection; "
                        "completed = installed map publications"),
            agent=dict(offered=1000, transmitted=len(data["agent_ue"]["transmitted_identities"]),
                       completed=agent_maps, decisions=agent_counts["policy_decisions"],
                       holds=agent_counts["policy_holds"], fallbacks=agent_counts["fallbacks"],
                       actor_calls=agent_counts["actor_calls"],
                       route_observations=route_counts(data["agent_all_route"])),
            static=dict(offered=1000, transmitted=len(data["static_ue"]["transmitted_identities"]),
                        completed=static_maps, decisions=static_counts["policy_decisions"],
                        holds=static_counts["policy_holds"], fallbacks=static_counts["fallbacks"],
                        actor_calls=static_counts["actor_calls"],
                        action_0=data["fixed"]["fixed_action"],
                        fixed_counters=data["fixed"]["counters"],
                        route_observations=route_counts(data["static_all_route"])),
        ),
        prediction_completion=dict(
            definition=("fraction by t = decision frames with a published prediction whose action-open-to-"
                        "edge-compute time is <=t / all decision frames; incomplete frames remain denominator"),
            deadline=dict(value_ms=170, label="original feedback deadline"),
            agent=dict(decisions=latency["agent_decisions"], completed=latency["agent_completed"],
                       completion_fraction=latency["agent_completed"]/latency["agent_decisions"],
                       completed_by_170=latency["agent_by_170"],
                       fraction_by_170=latency["agent_by_170"]/latency["agent_decisions"],
                       action_open_to_edge_compute_ms=latency["agent_quantiles"]),
            static=dict(decisions=latency["static_decisions"], completed=0,
                        completion_fraction=0.0, latency_distribution="NOT_MEASURED",
                        timeout_resolutions=sum(row.get("terminal") == "TIMEOUT"
                                                for row in data["static_ue"]["resolutions"]),
                        excluded_from_latency=("STALE terminal/receipt and timeout timestamps, including one "
                                               "stale-before-map edge compute")),
            full_remote="NO_INFERENCE__NO_LATENCY_CDF",
        ),
        quality=summarized_quality(data),
        payload=dict(
            denominator="all transmitted input frames; bytes are application payload",
            full_local=dict(p50_bytes=0, p99_bytes=0, kind="calculated_no_input_uplink"),
            static=dict(n=len(payload["static_wire"]), p50_bytes=payload["static"]["p50"],
                        p99_bytes=payload["static"]["p99"],
                        representation_p50_bytes=payload["static"]["p50"]-SFD4_FRAMING_BYTES,
                        representation_p99_bytes=payload["static"]["p99"]-SFD4_FRAMING_BYTES,
                        datagrams=payload["static_datagrams"], kind="measured_SFD4_application_wire"),
            agent=dict(n=len(payload["agent_wire"]), p50_bytes=payload["agent"]["p50"],
                       p99_bytes=payload["agent"]["p99"],
                       representation_p50_bytes=payload["agent"]["p50"]-SFD4_FRAMING_BYTES,
                       representation_p99_bytes=payload["agent"]["p99"]-SFD4_FRAMING_BYTES,
                       datagrams=payload["agent_datagrams"], kind="measured_SFD4_application_wire"),
            full_remote=dict(p50_bytes=payload["remote"]["p50"],
                             p99_bytes=payload["remote"]["p99"],
                             p50_envelope=data["remote_envelope"][50],
                             p99_envelope=data["remote_envelope"][99],
                             rgb_bytes=RGB_BYTES,
                             radar_bytes_per_unique_return=RADAR_BYTES_PER_RETURN,
                             kind="calculated_raw_input_payload_only",
                             inference=False, latency=False, quality=False),
            sfd4_framing_bytes=SFD4_FRAMING_BYTES,
            excluded_overhead="UDP/IP/radio",
        ),
        static_outcome=dict(
            classification=data["static_verdict"]["verdict"],
            passed=data["static_verdict"]["passed"],
            launch_returncode=data["static_verdict"]["launch_returncode"],
            raw_launcher_verdict=data["static_raw_verdict"].get("verdict"),
            raw_launcher_preserved=True,
            startup="EDGE_READY_AND_1000_FRAMES_TRANSMITTED",
            edge_outcomes=data["static_verdict"]["edge_outcomes"],
            protocol_faults=data["static_verdict"]["protocol_faults"],
            interpretation=data["static_verdict"]["generic_launcher_gate_status"]["interpretation"],
        ),
        static_retention=dict(
            observed_policy_decisions=data["static_retention"].get("observed_policy_decisions"),
            status_counts=data["static_retention"].get("status_counts"),
            storage_bytes=data["static_retention"].get("storage_bytes"),
            interpretation=("all 100 planned samples exceeded the retained 2 MiB wire budget; "
                            "no static retained-input quality replay exists"),
        ),
        verification=dict(
            preflight_status=data["preflight"].get("status"),
            valid_registration_passed=data["preflight"]["checks"]["valid_registration_passed"],
            deliberately_mismatched_source_rejected=
                data["preflight"]["checks"]["mismatched_source_rejected"],
            scipy_absent=data["preflight"]["checks"]["scipy_absent"],
            action_0_noae_uint8_no_drop=all(data["preflight"]["checks"][key] for key in
                ("action_0", "noae_uint8_q0", "no_spatial_drop")),
            actor_admission_or_action_71_substitution=not all(
                data["preflight"]["checks"][key] for key in
                ("actor_independent", "no_action71_substitution")),
        ),
        boundaries=dict(agent_rerun=False, automatic_retry=False, training_or_optimization=False,
                        offline_noae_substituted=False, m2="UNCHANGED", m3="NOT_STARTED"),
    )


def markdown_table(summary):
    latency = summary["prediction_completion"]["agent"]["action_open_to_edge_compute_ms"]
    payload = summary["payload"]
    return (
        "| Method / session | Offered / transmitted / completed maps | Decisions / holds / fallbacks | "
        "Prediction completion | Payload P50 / P99 | Quality scope |\n"
        "|---|---:|---:|---|---:|---|\n"
        "| Full local (offline reference) | N/A | N/A | No live latency | 0 / 0 B | "
        "Paired retained inputs, n=99 |\n"
        f"| Run-5 agent (retained valid live session) | 1,000 / 1,000 / 1,000 | 495 / 495 / 10 | "
        f"495/495 decisions; {latency['p50']:.1f}/{latency['p95']:.1f}/{latency['p99']:.1f} ms "
        f"P50/P95/P99 | {payload['agent']['p50_bytes']/1e3:.1f} / "
        f"{payload['agent']['p99_bytes']/1e3:.1f} kB | Archived live output on paired n=99 |\n"
        "| Static noAE action 0 (fresh live session) | 1,000 / 1,000 / 0 | 500 / 500 / 0 | "
        f"0/500 decisions; **no latency distribution** | "
        f"{payload['static']['p50_bytes']/1e6:.3f} / "
        f"{payload['static']['p99_bytes']/1e6:.3f} MB | None: zero completed predictions |\n"
        f"| Full remote (calculated only) | N/A | N/A | No inference / latency CDF | "
        f"{payload['full_remote']['p50_bytes']/1e6:.3f} / "
        f"{payload['full_remote']['p99_bytes']/1e6:.3f} MB | No inference / quality CDF |\n"
    )


def report_text(summary, table):
    agent = summary["prediction_completion"]["agent"]
    static_payload = summary["payload"]["static"]
    agent_payload = summary["payload"]["agent"]
    q = summary["quality"]
    return f"""# M1 baseline completion — revision 2

M1 is complete with an honestly labelled static result. The fresh static session passed edge READY, transmitted all 1,000 registered inputs under fixed action 0, and produced **zero completed predictions**. This is a measured overload/zero-completion outcome, not the preserved earlier pre-frame startup failure. No retry followed.

The slim-image preflight used the actual registration. The import-safe verifier accepted every valid registered source and rejected a deliberately mismatched digest while SciPy remained absent. Action 0 remained noAE / UINT8 / q=0 / zero spatial drop; there were no actor calls, admission-dependent substitutions, fallbacks, or action-71 substitutions.

## Static outcome and classification

The static session recorded 1,000 offered and transmitted experiment frames, 500 decisions, 500 holds, zero fallbacks, and zero map publications. All 500 decision resolutions were TIMEOUT. Edge evidence recorded 21 `STALE_BEFORE_EDGE` terminals and one `STALE_BEFORE_MAP` terminal; the latter completed edge compute only after its action window had expired. It is not a completed published prediction and its stale/timeout timing is not treated as inference latency.

The retained generic launcher returned 1 and its original verdict is preserved because its actor-oriented G8/P7 feedback checks cannot pass with zero feedback. The revision-2 evidence classifier applies the preregistered zero-completion rule only after retaining the mandatory identity, frame-accounting, instrumentation, protocol-fault, GC, infrastructure, and restoration checks. Those checks pass, so the result is `VALID_STATIC_ZERO_COMPLETION_MEASUREMENT`; the raw launcher result is not overwritten.

The retained sampling observer saw 500 static decisions, but all 100 planned samples ended `INCOMPLETE_WIRE_BUDGET`: the existing 2 MiB/sample retention cap was below the approximately 3.55 MB noAE wire payload. Consequently there is no retained static replay. More importantly, there were no completed live static predictions to score, so no static quality CDF or latency quantiles are reported and no offline noAE prediction is substituted.

## Comparison

{table}

The agent and static arms are different live sessions, with the same registered 1,000-frame budget, favourable route/channel, cadence, and transport limits; they are not paired scenes. The only same-input comparison is full local versus the archived agent output on the 99 outcome-independent retained agent decisions.

For localization, the CDF denominator is every eligible GT object, not only matches: misses remain in the denominator. Vehicles have n={q['vehicle']['full_local']['eligible']} eligible GT; recall at 1.0 m is {q['vehicle']['full_local']['recall_within_tol']:.3f} full local versus {q['vehicle']['agent_selected']['recall_within_tol']:.3f} agent. Pedestrians have n={q['person']['full_local']['eligible']} eligible GT; recall at 1.2 m is {q['person']['full_local']['recall_within_tol']:.3f} versus {q['person']['agent_selected']['recall_within_tol']:.3f}.

For latency, the curve is the number of decision frames with a completed prediction by t divided by all decision frames. Incomplete decisions remain in the denominator. The agent completed {agent['completed']}/{agent['decisions']} decisions; {agent['completed_by_170']}/{agent['decisions']} ({agent['fraction_by_170']:.3f}) completed by the 170 ms **original feedback deadline**. Static is shown only as a zero observed completion fraction (0/500), not as a latency distribution. Full remote has no latency or quality CDF.

Payload bars use all 1,000 transmitted inputs per live session. Agent SFD4 application-wire P50/P99 are {agent_payload['p50_bytes']:,.1f}/{agent_payload['p99_bytes']:,.1f} B; static P50/P99 are {static_payload['p50_bytes']:,.1f}/{static_payload['p99_bytes']:,.1f} B. Each includes 455 B of SFD4 framing; representation values are reported separately in `SUMMARY.json`, and UDP/IP/radio overhead is excluded. Full remote is raw-input arithmetic only (RGB plus deduplicated new radar returns), not a measured transport, inference, latency, or quality result.

## Scope closure

The valid agent capture and both static attempts remain preserved as separate sessions. No agent rerun, automatic retry, training, or runtime optimization occurred. M2 is unchanged. M3 was not started. No further experiment follows this M1 completion.
"""


def run(agent_root, static_root, replay, agent_campaign, static_capture,
        preflight, acceptance, out):
    out = Path(out)
    out.mkdir(parents=False, exist_ok=False)
    data = load(agent_root, static_root, replay, static_capture, preflight)
    quality_figure(data, out)
    latency = latency_figure(data, out)
    payload = payload_figure(data, out)
    summary = build_summary(data, latency, payload)
    table = markdown_table(summary)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    (out / "TABLE.md").write_text(table)
    (out / "REPORT.md").write_text(report_text(summary, table))

    inputs = [
        Path(agent_root) / "registration.json",
        Path(static_root) / "registration.json",
        Path(agent_campaign) / "FREEZE.json",
        Path(agent_campaign) / "VERDICT_agent.json",
        Path(static_capture) / "FREEZE.json",
        Path(static_capture) / "VERDICT_static.json",
        Path(static_capture) / "VERDICT_static_v2.json",
        Path(preflight), Path(acceptance), Path(replay),
        data["agent_ue_path"], data["static_ue_path"],
        data["agent_edge_path"], data["static_edge_path"],
        data["agent_csv"], data["static_csv"],
        Path(static_root) / "RETENTION_RESULT.json",
        Path(__file__),
    ]
    (out / "INPUT_SHA256.json").write_text(json.dumps(
        {str(path): sha(path) for path in inputs}, indent=1, sort_keys=True) + "\n")
    pngs = sorted(path.name for path in out.glob("*.png"))
    if pngs != ["01_quality_cdf.png", "02_prediction_completion_cdf.png", "03_payload.png"]:
        raise ValueError(f"expected exactly three registered PNGs, found {pngs}")
    outputs = {path.name: sha(path) for path in sorted(out.iterdir())
               if path.is_file() and path.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(outputs, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(status=summary["status"], pngs=pngs,
                          static_completed=0, static_decisions=latency["static_decisions"]),
                     indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-root", required=True)
    parser.add_argument("--static-root", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--agent-campaign", required=True)
    parser.add_argument("--static-capture", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(args.agent_root, args.static_root, args.replay, args.agent_campaign,
        args.static_capture, args.preflight, args.acceptance, args.out)
