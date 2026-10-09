"""M1 final revision 3: four live sessions, paired references and 4K slides."""
from __future__ import annotations

import argparse
import collections
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


MAIN = Path(__file__).resolve().parents[2]
SURFACE, INK, MUTED, GRID = "#fbfbfa", "#151515", "#565656", "#d9d9d9"
COLORS = {
    "agent": "#2E8B57", "action0": "#D97706", "arm_a": "#1677B8",
    "arm_b": "#7A4FB3", "full_local": "#173B57", "remote": "#B23A73",
    "incomplete": "#D7D7D7",
}
LABELS = {
    "agent": "Run-5 agent", "action0": "noAE overload", "arm_a": "AE32 UINT8 · a54",
    "arm_b": "Mode 11 UINT4 · q6000",
}
ORDER = ("agent", "action0", "arm_a", "arm_b")
PAIR_ORDER = ("agent", "arm_a", "arm_b")
TOL = {"vehicle": 1.0, "person": 1.2}
NATIVE_BGRA_BYTES = 1280 * 720 * 4
RADAR_RETURN_BYTES = 16
SFD4_FRAMING_BYTES = 455
FIGSIZE, DPI = (16, 9), 240


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def find_one(root, name):
    values = list(Path(root).rglob(name))
    if len(values) != 1:
        raise ValueError(f"expected one {name} below {root}, found {len(values)}")
    return values[0]


def q(values, percentiles=(50, 95, 99)):
    values = np.asarray(values, dtype=float)
    return ({f"p{p}": float(np.percentile(values, p)) for p in percentiles}
            if len(values) else {f"p{p}": None for p in percentiles})


def ratio(num, den):
    return float(num / den) if den else None


def style(ax, xgrid=False):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="both" if xgrid else "y", color=GRID, linewidth=1.2, alpha=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=16)


def save(fig, path, *, bottom=.105, top=.90):
    fig.tight_layout(rect=(.04, bottom, .98, top))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def profile_counts(root):
    rows = [json.loads(line) for line in find_one(Path(root) / "capture", "edge.profile.jsonl")
            .read_text().splitlines() if line.strip()]
    compute = [row for row in rows if row.get("r") == "span"
               and row.get("stage") == "edge.compute" and row.get("ph") == "live"]
    return dict(
        successful_processed=sum(row.get("ok") is True for row in compute),
        successful_prediction=sum(row.get("ok") is True
                                  and row.get("reward_requested") is True for row in compute),
        failed_prediction_spans=sum(row.get("ok") is not True
                                    and row.get("reward_requested") is True for row in compute),
        compute_spans=len(compute))


def trajectory(rows):
    xy = np.asarray([row["ego_world"][:2] for row in rows if row.get("ego_world")], float)
    if not len(xy):
        return dict(points=0, observed_path_m=None, x_extent_m=None, y_extent_m=None)
    distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    return dict(points=len(xy), observed_path_m=distance,
                x_extent_m=float(np.ptp(xy[:, 0])), y_extent_m=float(np.ptp(xy[:, 1])),
                x_min=float(xy[:, 0].min()), x_max=float(xy[:, 0].max()),
                y_min=float(xy[:, 1].min()), y_max=float(xy[:, 1].max()),
                definition="ordered policy-decision ego positions; durable route summary unavailable for fixed arms")


def load_session(key, root):
    root = Path(root)
    _, rows, ue = LIVE.rows(root)
    edge = read_json(find_one(root / "capture", "edge_report.json"))
    route_path = find_one(root / "capture", "per_frame_metrics.csv")
    route_rows = csv_rows(route_path)
    sent = [row for row in route_rows if row["prepare_status"] == "SENT"]
    cell = read_json(find_one(root / "capture", "CELL_RESULT.json"))
    counts = ue["counters"]
    profile = profile_counts(root)
    completed_rows = [row for row in rows if row.get("prediction_materialized") is not None]
    if len(completed_rows) != profile["successful_prediction"]:
        raise ValueError(f"{key}: materialized row/profile completion mismatch")
    return dict(
        key=key, root=root, rows=rows, ue=ue, edge=edge, route_path=route_path,
        route_rows=route_rows, sent=sent, cell=cell, profile=profile,
        completed_rows=completed_rows,
        latency_ms=np.asarray([row["edge_compute_end"] for row in completed_rows], float),
        feedback_ms=np.asarray([row["ue_receipt"] for row in rows
                                if row.get("ue_receipt") is not None], float),
        payload_bytes=np.asarray([int(row["payload_bytes"]) for row in sent], float),
        datagrams=np.asarray([int(row["datagrams"]) for row in sent], float),
        frame=dict(
            offered=1000, transmitted=len(ue["transmitted_identities"]),
            successfully_processed=profile["successful_processed"],
            published=int(edge.get("publications", 0)),
            decisions=int(counts["policy_decisions"]), holds=int(counts["policy_holds"]),
            fallbacks=int(counts["fallbacks"]), actor_calls=int(counts["actor_calls"]),
            route_input_outcomes=dict(collections.Counter(
                row["prepare_status"] for row in route_rows)),
            elapsed_s=float((cell["finished_at_unix_s"] - cell["started_at_unix_s"])),
            route_coverage=trajectory(rows)),
        controller_outcomes=dict(collections.Counter(
            row.get("terminal") for row in ue.get("resolutions", []))),
        edge_typed_outcomes={name: value for name, value in edge.get("counters", {}).items()
                             if name.startswith("edge_terminal_")},
    )


def measurement_summary(measurements, cls, samples):
    prefix = f"loc_{cls}_"
    errors = [float(error) for m in measurements
              for error in m[prefix + "matched_xy_errors_m"]]
    eligible = sum(int(m[prefix + "eligible_gt"]) for m in measurements)
    tp = sum(int(m[prefix + "tp"]) for m in measurements)
    fp = sum(int(m[prefix + "fp"]) for m in measurements)
    fn = sum(int(m[prefix + "fn"]) for m in measurements)
    within = sum(error <= TOL[cls] for error in errors)
    inter = sum(int(m[f"seg_{cls}_intersection_pixels"]) for m in measurements)
    union = sum(int(m[f"seg_{cls}_union_pixels"]) for m in measurements)
    return dict(
        samples=samples, eligible=eligible, matched_3m=tp, misses=fn,
        false_positives=fp, within_tol=within,
        recall_within_tol=ratio(within, eligible), recall_3m=ratio(tp, eligible),
        precision=ratio(tp, tp + fp), seg_iou_pooled=ratio(inter, union),
        samples_with_eligible=sum(int(m[prefix + "eligible_gt"]) > 0 for m in measurements),
        errors=sorted(errors), error_quantiles_m=q(errors) if errors else q([]))


def replay_quality(report):
    rows = [row for row in report["rows"] if row["admission"] == "ADMITTED"]
    result = dict(samples=len(rows), missing=[
        dict(ordinal=row["ordinal"], admission=row["admission"], fault=row.get("fault"))
        for row in report["rows"] if row["admission"] != "ADMITTED"], classes={})
    for cls in TOL:
        result["classes"][cls] = {}
        for branch in ("full_local", "live_selected"):
            measurements = [row[branch]["measurement"] for row in rows]
            result["classes"][cls][branch] = measurement_summary(
                measurements, cls, len(rows))
    return result


def live_quality(session):
    result = dict(samples=len(session["completed_rows"]), classes={})
    for cls in TOL:
        ms = []
        for row in session["completed_rows"]:
            ms.append(dict(
                **{f"loc_{cls}_{name}": row[f"{cls}_{name}"] for name in
                   ("eligible_gt", "tp", "fn", "fp", "matched_xy_errors_m")},
                **{f"seg_{cls}_{name}": row[f"seg_{cls}_{name}"] for name in
                   ("gt_pixels", "pred_pixels", "intersection_pixels", "union_pixels")}))
        result["classes"][cls] = measurement_summary(ms, cls, len(ms))
    return result


def strip_errors(value):
    if isinstance(value, dict):
        return {key: strip_errors(item) for key, item in value.items() if key != "errors"}
    if isinstance(value, list):
        return [strip_errors(item) for item in value]
    return value


def conservative_xy(metric):
    errors = np.asarray(metric["errors"], float)
    x = np.linspace(0, 3.0, 301)
    y = np.asarray([(errors <= value).sum() / metric["eligible"]
                    if metric["eligible"] else np.nan for value in x])
    return x, y


def write_cdf_csv(out, paired):
    path = out / "LOCALIZATION_CDF.csv"
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["session", "branch", "class", "threshold_m", "fraction_all_eligible_gt",
                         "eligible_gt_denominator", "matched_errors", "misses", "same_input_pair"])
        for key in PAIR_ORDER:
            for cls in TOL:
                for branch in ("full_local", "live_selected"):
                    metric = paired[key]["classes"][cls][branch]
                    x, y = conservative_xy(metric)
                    for xx, yy in zip(x, y):
                        writer.writerow([key, branch, cls, xx, yy, metric["eligible"],
                                         len(metric["errors"]), metric["misses"], True])
    return path


def fig_localization(out, paired):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=SURFACE)
    for ax, cls in zip(axes, TOL):
        for key in PAIR_ORDER:
            for branch, linestyle, alpha, suffix in (
                    ("live_selected", "-", 1.0, "live"),
                    ("full_local", "--", .76, "full-local")):
                metric = paired[key]["classes"][cls][branch]
                x, y = conservative_xy(metric)
                ax.plot(x, y, linestyle=linestyle, linewidth=3.5,
                        color=COLORS[key], alpha=alpha,
                        label=(f"{LABELS[key]} · {suffix}  "
                               f"{metric['within_tol']}/{metric['eligible']}"))
        ax.axvline(TOL[cls], color="#9B2F2F", linestyle=":", linewidth=3,
                   label=f"{TOL[cls]:.1f} m threshold")
        ax.set_xlim(0, 3.0); ax.set_ylim(0, 1.02)
        ax.set_xlabel("XY localization error threshold t (m)", fontsize=21, weight="bold")
        ax.set_ylabel("eligible GT localized within t", fontsize=21, weight="bold")
        name = "Vehicles" if cls == "vehicle" else "Pedestrians"
        ns = [paired[k]["classes"][cls]["live_selected"]["eligible"]
              for k in PAIR_ORDER]
        denoms = f"GT n = {ns[0]} / {ns[1]} / {ns[2]} (agent / a54 / m11)"
        ax.set_title(f"{name} · eligible-GT denominator\n{denoms}",
                     fontsize=18, weight="bold", loc="left")
        style(ax, xgrid=True)
        ax.legend(frameon=False, fontsize=13.5, loc="lower right", ncol=1)
    fig.suptitle("Localization CDF · misses remain in the denominator",
                 fontsize=29, weight="bold", color=INK)
    fig.text(.5, .034,
             "Solid/dashed lines share inputs within a session · colors denote different live sessions.",
             ha="center", fontsize=17, color=MUTED, weight="bold")
    save(fig, out / "01_localization_cdf.png", bottom=.105, top=.88)


def fig_completion(out, sessions):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": (1.35, 1)})
    ax = axes[0]
    max_latency = max(float(sessions[key]["latency_ms"].max())
                      for key in ORDER if len(sessions[key]["latency_ms"]))
    xmax = math.ceil(max_latency / 50) * 50
    cdf_rows = []
    for key in ORDER:
        values = np.sort(sessions[key]["latency_ms"])
        den = sessions[key]["frame"]["decisions"]
        if len(values):
            x = np.concatenate(([0.0], values, [xmax]))
            y = np.concatenate(([0.0], np.arange(1, len(values)+1)/den, [len(values)/den]))
        else:
            x, y = np.asarray([0.0, xmax]), np.asarray([0.0, 0.0])
        ax.step(x, y, where="post", linewidth=4, color=COLORS[key],
                label=f"{LABELS[key]} · {len(values)}/{den}")
        for xx, yy in zip(x, y):
            cdf_rows.append([key, xx, yy, den, len(values)])
    ax.axvline(170, color="#9B2F2F", linestyle="--", linewidth=3,
               label="170 ms · original feedback deadline")
    ax.set_xlim(0, xmax); ax.set_ylim(-.02, 1.03)
    ax.set_xlabel("decision open → successful edge.compute end (ms)", fontsize=19, weight="bold")
    ax.set_ylabel("fraction of all decisions completed by t", fontsize=19, weight="bold")
    ax.set_title("Prediction-completion curve", fontsize=24, weight="bold", loc="left")
    style(ax, xgrid=True); ax.legend(frameon=False, fontsize=15, loc="lower right")
    with (out / "PREDICTION_COMPLETION_CDF.csv").open("x", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(
            ["session", "time_ms", "fraction_all_policy_decisions", "decision_denominator", "completed"])
        writer.writerows(cdf_rows)

    ax = axes[1]
    ypos = np.arange(len(ORDER))
    complete = np.asarray([len(sessions[key]["latency_ms"]) for key in ORDER])
    total = np.asarray([sessions[key]["frame"]["decisions"] for key in ORDER])
    ax.barh(ypos, complete, color=[COLORS[key] for key in ORDER], height=.62,
            label="successful prediction")
    ax.barh(ypos, total-complete, left=complete, color=COLORS["incomplete"],
            height=.62, label="incomplete")
    for i, key in enumerate(ORDER):
        by170 = int(np.sum(sessions[key]["latency_ms"] <= 170))
        ax.text(8, i, f"{complete[i]}/{total[i]} complete\n{by170} by 170 ms",
                va="center", fontsize=13.5, weight="bold",
                color="white" if complete[i] > 250 else INK)
    ax.set_yticks(ypos, [LABELS[key] for key in ORDER], fontsize=17)
    ax.invert_yaxis(); ax.set_xlim(0, 540)
    ax.set_xlabel("policy decisions", fontsize=20, weight="bold")
    ax.set_title("Outcome counts", fontsize=24, weight="bold", loc="left")
    style(ax)
    ax.text(.98, .98, "colored = complete | gray = incomplete",
            transform=ax.transAxes, ha="right", va="top", fontsize=14,
            color=MUTED, weight="bold")
    fig.suptitle("Prediction completion · decision-frame denominator",
                 fontsize=29, weight="bold", color=INK)
    fig.text(.5, .034,
             "CPU prediction materialized only · TIMEOUT/STALE and map publication are not latency samples.",
             ha="center", fontsize=16.5, color=MUTED, weight="bold")
    save(fig, out / "02_prediction_completion.png", bottom=.105, top=.89)


def fmt_size(value):
    if value == 0: return "0"
    if value < 1e6: return f"{value/1e3:.1f} kB"
    return f"{value/1e6:.3f} MB"


def fig_data_size(out, sessions, full_local_sizes, remote):
    pooled_local = np.concatenate([full_local_sizes[key] for key in PAIR_ORDER])
    labels = ["Full-local\ninput uplink", "Full-local\ndetection output",
              "Mode 11\nUINT4 q6000", "Run-5\nagent", "AE32 UINT8\naction 54",
              "noAE UINT8\noverload", "Full remote\nnative bundle"]
    values = [np.asarray([0]), pooled_local, sessions["arm_b"]["payload_bytes"],
              sessions["agent"]["payload_bytes"], sessions["arm_a"]["payload_bytes"],
              sessions["action0"]["payload_bytes"], remote]
    p50 = np.asarray([np.percentile(v, 50) for v in values]) / 1e6
    p99 = np.asarray([np.percentile(v, 99) for v in values]) / 1e6
    colors = [COLORS["full_local"], COLORS["full_local"], COLORS["arm_b"],
              COLORS["agent"], COLORS["arm_a"], COLORS["action0"], COLORS["remote"]]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": (1.2, 1)})
    ypos = np.arange(len(labels)); height = .34
    ax = axes[0]
    b1 = ax.barh(ypos-height/2, p50, height, color=colors,
                 edgecolor="white", label="P50")
    b2 = ax.barh(ypos+height/2, p99, height, color=colors,
                 edgecolor="white", hatch="//", label="P99")
    for bars, raw in ((b1, p50), (b2, p99)):
        for bar, value in zip(bars, raw):
            ax.text(max(value+.045, .055), bar.get_y()+bar.get_height()/2,
                    fmt_size(value*1e6), ha="left", va="center",
                    fontsize=12.5, weight="bold")
    ax.set_yticks(ypos, labels, fontsize=13.5, weight="bold")
    ax.set_xlim(0, 4.85); ax.invert_yaxis()
    ax.set_xlabel("bytes per item (MB, 10⁶ bytes)", fontsize=20, weight="bold")
    ax.set_title("Full scale", fontsize=24, weight="bold", loc="left")
    style(ax, xgrid=True); ax.legend(frameon=False, fontsize=16,
                                     loc="upper right", ncol=2)

    ax = axes[1]; small = np.arange(5)
    ax.barh(small-height/2, p50[:5], height, color=colors[:5], edgecolor="white")
    ax.barh(small+height/2, p99[:5], height, color=colors[:5],
            edgecolor="white", hatch="//")
    for i in range(5):
        ax.text(p99[i]+.012, i+height/2, fmt_size(p99[i]*1e6),
                ha="left", va="center", fontsize=13, weight="bold")
    ax.set_yticks(small, labels[:5], fontsize=13.5, weight="bold")
    ax.set_xlim(0, .72); ax.invert_yaxis()
    ax.set_xlabel("bytes per item (MB, 10⁶ bytes)", fontsize=20, weight="bold")
    ax.set_title("Detail below 0.7 MB", fontsize=24, weight="bold", loc="left")
    style(ax, xgrid=True)
    fig.suptitle("Data size by computation placement · P50 / P99",
                 fontsize=31,weight="bold",color=INK)
    fig.text(.5,.034,
             "Measured SFD4 for split arms · full-local output and full-remote native input are calculated.",
             ha="center",fontsize=16,color=MUTED,weight="bold")
    save(fig,out/"03_data_size.png",bottom=.105,top=.89)


def paired_bar_metrics(out, paired):
    metrics=(("vehicle","seg_iou_pooled","Vehicle segmentation IoU"),
             ("person","seg_iou_pooled","Pedestrian segmentation IoU"),
             ("vehicle","precision","Vehicle detection precision"),
             ("person","precision","Pedestrian detection precision"))
    fig,axes=plt.subplots(2,2,figsize=FIGSIZE,facecolor=SURFACE)
    for ax,(cls,metric,title) in zip(axes.flat,metrics):
        x=np.arange(len(PAIR_ORDER)); w=.34
        local=[paired[k]["classes"][cls]["full_local"][metric] or 0 for k in PAIR_ORDER]
        live=[paired[k]["classes"][cls]["live_selected"][metric] or 0 for k in PAIR_ORDER]
        ax.bar(x-w/2,local,w,color=[COLORS[k] for k in PAIR_ORDER],alpha=.35,
               edgecolor=[COLORS[k] for k in PAIR_ORDER],linewidth=2,hatch="//",label="full-local")
        ax.bar(x+w/2,live,w,color=[COLORS[k] for k in PAIR_ORDER],label="archived live")
        for xx,v in zip(x-w/2,local): ax.text(xx,v+.018,f"{v:.3f}",ha="center",fontsize=13,weight="bold")
        for xx,v in zip(x+w/2,live): ax.text(xx,v+.018,f"{v:.3f}",ha="center",fontsize=13,weight="bold")
        short = {"agent": "Agent", "arm_a": "Action 54", "arm_b": "Mode 11"}
        ax.set_xticks(x,[f"{short[k]}\nn={paired[k]['samples']}" for k in PAIR_ORDER],fontsize=13,weight="bold")
        ax.set_ylim(0,1.08);ax.set_title(title,fontsize=21,weight="bold",loc="left");style(ax)
    axes[0,0].legend(frameon=False,fontsize=15,ncol=2)
    fig.suptitle("Same-input quality · segmentation and precision",fontsize=31,weight="bold")
    fig.text(.5,.035,"Each color is a separate live session; paired bars share retained input and GT within that session.",
             ha="center",fontsize=16.5,color=MUTED,weight="bold")
    save(fig,out/"04_segmentation_precision.png",bottom=.105,top=.89)


def fig_delivered(out, paired, live_quality_all):
    fig,axes=plt.subplots(1,2,figsize=FIGSIZE,facecolor=SURFACE)
    for ax,cls in zip(axes,TOL):
        x=np.arange(len(PAIR_ORDER));w=.23
        local=[paired[k]["classes"][cls]["full_local"]["recall_within_tol"] or 0 for k in PAIR_ORDER]
        delivered=[paired[k]["classes"][cls]["live_selected"]["recall_within_tol"] or 0 for k in PAIR_ORDER]
        conditional=[live_quality_all[k]["classes"][cls]["recall_within_tol"] or 0 for k in PAIR_ORDER]
        ax.bar(x-w,local,w,color=[COLORS[k] for k in PAIR_ORDER],alpha=.30,hatch="//",label="full-local paired")
        ax.bar(x,delivered,w,color=[COLORS[k] for k in PAIR_ORDER],label="live delivered · known-GT retained")
        ax.bar(x+w,conditional,w,color=[COLORS[k] for k in PAIR_ORDER],alpha=.62,hatch="..",label="live quality · all completions")
        for offset,vals in ((-w,local),(0,delivered),(w,conditional)):
            for xx,v in zip(x+offset,vals): ax.text(xx,v+.02,f"{v:.3f}",ha="center",fontsize=12.5,rotation=90,weight="bold")
        short = {"agent": "Agent", "arm_a": "Action 54", "arm_b": "Mode 11"}
        ax.set_xticks(x,[f"{short[k]}\nknown GT: {paired[k]['samples']}" for k in PAIR_ORDER],fontsize=14,weight="bold")
        ax.set_ylim(0,1.12);ax.set_ylabel("within-tolerance recall",fontsize=20,weight="bold")
        ax.set_title(f"{'Vehicles' if cls=='vehicle' else 'Pedestrians'} · {TOL[cls]:.1f} m",
                     fontsize=24,weight="bold",loc="left");style(ax)
    axes[0].legend(frameon=False,fontsize=14,loc="upper left")
    fig.suptitle("Delivered localization · paired and completion-conditional",fontsize=29,weight="bold")
    fig.text(.5,.034,"Known GT: agent 99/100 · action 54 94/100 · mode 11 100/100. Missing GT is excluded and reported.",
             ha="center",fontsize=16,color=MUTED,weight="bold")
    save(fig,out/"05_delivered_localization.png",bottom=.105,top=.89)


def fig_feedback(out,sessions):
    fig,ax=plt.subplots(figsize=FIGSIZE,facecolor=SURFACE)
    xmax=math.ceil(max(float(sessions[k]["feedback_ms"].max()) for k in ORDER if len(sessions[k]["feedback_ms"]))/50)*50
    rows=[]
    for key in ORDER:
        values=np.sort(sessions[key]["feedback_ms"]);den=sessions[key]["frame"]["decisions"]
        if len(values):
            x=np.concatenate(([0],values,[xmax]));y=np.concatenate(([0],np.arange(1,len(values)+1)/den,[len(values)/den]))
        else:x=np.asarray([0,xmax]);y=np.asarray([0,0])
        by170=int(np.sum(values<=170))
        ax.step(x,y,where="post",linewidth=4,color=COLORS[key],label=f"{LABELS[key]} · {by170}/{den} by 170 ms")
        rows += [[key,xx,yy,den,len(values)] for xx,yy in zip(x,y)]
    ax.axvline(170,color="#9B2F2F",linestyle="--",linewidth=3,label="170 ms · original feedback deadline")
    ax.set_xlim(0,xmax);ax.set_ylim(-.02,1.03)
    ax.set_xlabel("action-open → UE feedback receipt (ms)",fontsize=21,weight="bold")
    ax.set_ylabel("fraction of policy decisions with reply by t",fontsize=21,weight="bold")
    ax.set_title("Feedback timeliness · separate from prediction completion",fontsize=30,weight="bold",loc="left")
    style(ax,xgrid=True);ax.legend(frameon=False,fontsize=17,loc="lower right")
    fig.text(.5,.035,"Agent reference: 410/495 feedback replies by 170 ms; its prediction-completion count is separately 476/495.",
             ha="center",fontsize=17,color=MUTED,weight="bold")
    save(fig,out/"06_feedback_timeliness.png",bottom=.105,top=.91)
    with (out/"FEEDBACK_CDF.csv").open("x",newline="") as stream:
        w=csv.writer(stream);w.writerow(["session","time_ms","fraction_policy_decisions","decision_denominator","observed_replies"]);w.writerows(rows)


def fig_payload_cdf(out,sessions,full_local_sizes,remote):
    fig,axes=plt.subplots(1,2,figsize=FIGSIZE,facecolor=SURFACE,gridspec_kw={"width_ratios":(1.1,1.25)})
    ax=axes[0];payload_rows=[]
    for key in ORDER:
        values=np.sort(sessions[key]["payload_bytes"])/1e6;y=np.arange(1,len(values)+1)/len(values)
        ax.step(values,y,where="post",linewidth=3.5,color=COLORS[key],label=f"{LABELS[key]} · measured n={len(values)}")
        payload_rows += [[key,"MEASURED_SFD4_INPUT",v*1e6,yy] for v,yy in zip(values,y)]
    rv=np.sort(remote)/1e6;ry=np.arange(1,len(rv)+1)/len(rv)
    ax.step(rv,ry,where="post",linewidth=3.5,linestyle="--",color=COLORS["remote"],label="Full remote · calculated n=1,000")
    payload_rows += [["full_remote","CALCULATED_NATIVE_INPUT",v*1e6,yy] for v,yy in zip(rv,ry)]
    local=np.sort(np.concatenate([full_local_sizes[k] for k in PAIR_ORDER]))/1e6;ly=np.arange(1,len(local)+1)/len(local)
    ax.step(local,ly,where="post",linewidth=3,linestyle=":",color=COLORS["full_local"],label=f"Full-local detection output · serialized n={len(local)}")
    payload_rows += [["full_local","SERIALIZED_DETECTION_OUTPUT",v*1e6,yy] for v,yy in zip(local,ly)]
    ax.set_xscale("log");ax.set_xlim(.02,5);ax.set_ylim(0,1.02)
    ax.set_xlabel("bytes per item (MB, log scale)",fontsize=20,weight="bold");ax.set_ylabel("empirical fraction",fontsize=20,weight="bold")
    ax.set_title("Payload / output-size CDF",fontsize=24,weight="bold",loc="left");style(ax,xgrid=True);ax.legend(frameon=False,fontsize=13.5,loc="upper left")
    with (out/"PAYLOAD_CDF.csv").open("x",newline="") as stream:
        w=csv.writer(stream);w.writerow(["session","byte_layer","bytes","empirical_fraction"]);w.writerows(payload_rows)

    ax=axes[1];ax.axis("off")
    columns=["Session","Sent","Proc.","Maps","Pred.","Reply\n≤170","Timeout","Eval\nfault"]
    table=[]
    short = {"agent": "Agent", "action0": "noAE", "arm_a": "Action 54", "arm_b": "Mode 11"}
    for key in ORDER:
        s=sessions[key];outcomes=s["controller_outcomes"]
        table.append([short[key],s["frame"]["transmitted"],s["frame"]["successfully_processed"],s["frame"]["published"],
                      f"{len(s['latency_ms'])}/{s['frame']['decisions']}",int(np.sum(s["feedback_ms"]<=170)),
                      outcomes.get("TIMEOUT",0),outcomes.get("EVALUATOR_FAULT",0)])
    tab=ax.table(cellText=table,colLabels=columns,loc="center",cellLoc="center",colLoc="center",
                 colWidths=[.22,.10,.10,.10,.13,.13,.11,.11])
    tab.auto_set_font_size(False);tab.set_fontsize(11.5);tab.scale(1,2.45)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("white")
        if r==0:cell.set_facecolor("#333333");cell.get_text().set_color("white");cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f0f0ee" if r%2 else "#e3e3df")
            if c==0:cell.get_text().set_weight("bold");cell.get_text().set_color(COLORS[ORDER[r-1]])
    ax.set_title("Live outcome accounting",fontsize=24,weight="bold",loc="left")
    fig.suptitle("Payload distribution · live outcome accounting",fontsize=29,weight="bold")
    fig.text(.5,.034,"SFD4/native exclude protocol overhead · full-local JSON is output size, not traffic.",
             ha="center",fontsize=16,color=MUTED,weight="bold")
    save(fig,out/"07_payload_cdf_outcomes.png",bottom=.105,top=.89)


def fig_table(out,sessions,payload,full_local_stats,paired):
    fig,ax=plt.subplots(figsize=FIGSIZE,facecolor=SURFACE);ax.axis("off")
    columns=["Session","Offered / sent","Proc. / maps","Dec. / hold / FB","Pred. complete","By 170 ms","SFD4 P50 / P99","Paired GT"]
    table=[]
    short = {"agent": "Agent", "action0": "noAE overload", "arm_a": "Action 54", "arm_b": "Mode 11"}
    for key in ORDER:
        s=sessions[key];den=s["frame"]["decisions"];by=int(np.sum(s["latency_ms"]<=170))
        paired_gt=("—" if key=="action0" else f"n={paired[key]['samples']}")
        table.append([short[key],f"1,000 / {s['frame']['transmitted']}",f"{s['frame']['successfully_processed']} / {s['frame']['published']}",
                      f"{den} / {s['frame']['holds']} / {s['frame']['fallbacks']}",f"{len(s['latency_ms'])}/{den}",f"{by}/{den}",
                      f"{payload[key]['p50']/1e3:.1f} / {payload[key]['p99']/1e3:.1f} kB",paired_gt])
    table += [["Full local", "input uplink = 0", "offline reference", "—", "—", "—",
               f"output {full_local_stats['pooled']['p50']/1e3:.1f} / {full_local_stats['pooled']['p99']/1e3:.1f} kB", "n=293"],
              ["Full remote", "calculated only", "no inference", "—", "—", "—",
               f"native {payload['full_remote']['p50']/1e6:.3f} / {payload['full_remote']['p99']/1e6:.3f} MB", "no quality"]]
    tab=ax.table(cellText=table,colLabels=columns,loc="center",cellLoc="center",colLoc="center",
                 colWidths=[.16,.115,.115,.15,.10,.10,.16,.10])
    tab.auto_set_font_size(False);tab.set_fontsize(12.5);tab.scale(1,2.65)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("white")
        if r==0:cell.set_facecolor("#333333");cell.get_text().set_color("white");cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f2f2ef" if r%2 else "#e4e4df")
            if c==0:cell.get_text().set_weight("bold")
    ax.set_title("M1 comparison · matched profiles, separate live sessions",fontsize=28,weight="bold",loc="left",pad=28)
    fig.text(.5,.07,"Prediction completion = successful edge.compute + CPU materialization · endpoint sizes are calculated.",
             ha="center",fontsize=16,color=MUTED,weight="bold")
    save(fig,out/"08_comparison_table.png",bottom=.10,top=.90)


def build_table(summary):
    lines=["# M1 comparison table","",("All live rows are different sessions. `processed` is a successful live `edge.compute` frame; "
           "`prediction` is a successful policy-decision compute with CPU materialization. Maps are separate."),"",
           "| Session | Offered | Sent | Processed | Maps | Decisions | Holds | Fallbacks | Prediction complete | By 170 ms | Payload P50/P99 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in ORDER:
        s=summary["sessions"][key];p=summary["payload"][key];c=s["prediction_completion"]
        lines.append(f"| {LABELS[key]} | 1,000 | {s['frame']['transmitted']} | {s['frame']['successfully_processed']} | {s['frame']['published']} | {s['frame']['decisions']} | {s['frame']['holds']} | {s['frame']['fallbacks']} | {c['completed']}/{c['denominator']} | {c['completed_by_170']}/{c['denominator']} | {p['p50']/1e3:.1f}/{p['p99']/1e3:.1f} kB |")
    lines += ["", "| Calculated reference | P50 | P99 | Meaning |", "|---|---:|---:|---|",
              f"| Full-local input uplink | 0 B | 0 B | No sensor input transmitted |",
              f"| Full-local serialized detections, pooled n=293 | {summary['full_local_detection_output']['pooled']['p50']/1e3:.1f} kB | {summary['full_local_detection_output']['pooled']['p99']/1e3:.1f} kB | Canonical JSON, no compression; not transmitted |",
              f"| Full remote native bundle, calculated n=1,000 | {summary['payload']['full_remote']['p50']/1e6:.3f} MB | {summary['payload']['full_remote']['p99']/1e6:.3f} MB | 1280×720×4 BGRA + 16×two-sweep raw radar returns |",
              "", "Full remote excludes metadata/framing overhead and is not a minimum-streaming claim; a stream may cache the previous radar sweep.", ""]
    return "\n".join(lines)


def build_report(summary):
    agent=summary["sessions"]["agent"]
    overload=summary["sessions"]["action0"]
    a,b=summary["sessions"]["arm_a"],summary["sessions"]["arm_b"]

    def latency_text(session):
        values=session["prediction_completion"]["completion_latency_ms"]
        if values["p50"] is None:
            return "—"
        return f"{values['p50']:.1f} / {values['p95']:.1f} / {values['p99']:.1f}"

    def payload_text(key):
        values=summary["payload"][key]
        return f"{values['p50']/1e3:.1f} / {values['p99']/1e3:.1f} kB"

    def session_row(key, session):
        frame=session["frame"]
        complete=session["prediction_completion"]
        return (f"| {session['label']} | 1,000 / {frame['transmitted']} | "
                f"{frame['successfully_processed']} / {frame['published']} | "
                f"{frame['decisions']} / {frame['holds']} / {frame['fallbacks']} | "
                f"{complete['completed']}/{complete['denominator']} | "
                f"{complete['completed_by_170']}/{complete['denominator']} | "
                f"{latency_text(session)} | {payload_text(key)} |")

    qa=a["live_quality"]["classes"]
    qb=b["live_quality"]["classes"]
    return f"""# M1 close-out — fixed baselines and presentation revision 3

M1 is complete. The retained Run-5 agent capture is unchanged, and exactly two new preregistered fixed live attempts were run once each, in order, with no retry. M2 was not reopened and M3 was not started.

## Outcome

- AE32/UINT8 action 54: 1,000/1,000 frames transmitted; {a['frame']['successfully_processed']} processed and {a['frame']['published']} maps published; {a['prediction_completion']['completed']}/500 policy predictions completed. All 100 retention slots completed; 94 GT references resolved and six remain explicit missing-GT exclusions.
- Mode 11 AE32/UINT4 q=6000: 1,000/1,000 transmitted; {b['frame']['successfully_processed']} processed and {b['frame']['published']} maps published; {b['prediction_completion']['completed']}/500 policy predictions completed. All 100 retention/GT references resolved.
- Both fixed arms made 500 decisions and 500 holds, with zero actor calls, fallbacks, substitutions, or identity mismatches. Their outcome-aware M1 verdict is `VALID_M1_EXTERNAL_FIXED_MEASUREMENT`. The preserved generic actor-driven qualification is non-pass solely because its actor-replay/actor-call gates do not apply to the preregistered actor-free selector; all retained non-actor integrity, accounting, reward, transport, GC and cleanup checks pass.
- The earlier noAE overload remains a separate valid overload reference: 1,000 transmitted, 500 decisions, and zero successful predictions. Its earlier pre-frame registration failure remains a separate failed attempt.

Each row below is a different live session. Completion latency is reported only for successful `edge.compute` plus CPU materialization; values are P50/P95/P99 in milliseconds.

| Session | Offered / sent | Processed / maps | Decisions / holds / fallbacks | Prediction complete | By 170 ms | Completion latency (ms) | Payload P50 / P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
{session_row('agent',agent)}
{session_row('action0',overload)}
{session_row('arm_a',a)}
{session_row('arm_b',b)}

Controller outcome labels are separate from prediction completion: agent SUCCESS/TIMEOUT/EVALUATOR_FAULT = 370/85/40; noAE = 0/500/0; action 54 = 9/489/2; mode 11 = 372/107/21.

Live fixed-arm quality below covers all materialized prediction frames and keeps all eligible GT objects, including misses, in the localization denominator. Cross-row comparison is descriptive because these are different sessions.

| Static session | Prediction frames | Vehicle recall ≤1.0 m | Pedestrian recall ≤1.2 m | Vehicle / pedestrian precision | Vehicle / pedestrian segmentation IoU |
|---|---:|---:|---:|---:|---:|
| AE32/UINT8 action 54 | {a['live_quality']['samples']} | {qa['vehicle']['recall_within_tol']:.3f} | {qa['person']['recall_within_tol']:.3f} | {qa['vehicle']['precision']:.3f} / {qa['person']['precision']:.3f} | {qa['vehicle']['seg_iou_pooled']:.3f} / {qa['person']['seg_iou_pooled']:.3f} |
| Mode 11 AE32/UINT4 q=6000 | {b['live_quality']['samples']} | {qb['vehicle']['recall_within_tol']:.3f} | {qb['person']['recall_within_tol']:.3f} | {qb['vehicle']['precision']:.3f} / {qb['person']['precision']:.3f} | {qb['vehicle']['seg_iou_pooled']:.3f} / {qb['person']['seg_iou_pooled']:.3f} |

## Corrections and definitions

Frame 1338 in the noAE session did **not** complete inference. It expired at `EDGE_BEFORE_TAIL`; its `edge.compute` and frozen-tail spans ended unsuccessfully and no CPU prediction materialized. Map publication is not a prediction-completion shortcut.

Primary latency is action-open to the end of a successful live `edge.compute` span with CPU prediction materialized. Incomplete decisions remain in the completion-CDF denominator. The 170 ms line is the original feedback deadline, not a requalified prediction deadline. TIMEOUT/STALE timestamps are never latency samples. Feedback receipt is plotted separately.

Localisation CDFs divide detections within *t* by every eligible GT object; misses remain in the denominator. Solid/dashed pairs share retained input and GT within one session. Comparisons across session colors are descriptive, not paired scenes. Missing GT is excluded and counted separately.

## Size references

Measured split payload uses the SFD4 application byte layer for all 1,000 transmitted inputs per session. Full remote is calculated as native BGRA plus the complete two-sweep radar bundle: `1280×720×4 + 16×raw_return_count`; P50/P99 are {summary['payload']['full_remote']['p50']/1e6:.3f}/{summary['payload']['full_remote']['p99']/1e6:.3f} MB. It excludes metadata/framing and is not a minimum-streaming claim because a stream may cache the prior sweep.

Full-local input uplink remains zero. Full-local detection size is a separate calculated output: uncompressed canonical JSON from `DetachedOptimizedTailAdapterV3.serialize_product`, including all emitted detections and identity metadata, excluding GT/debug logs, label rasters and dense segmentation. The exported records reproduce all 99 sealed agent hashes; all admitted fixed-arm selected outputs also reproduce their archived live records exactly.

## Limitations / M3 handoff

Sessions share registered route/profile/cadence/transport limits but not identical scenes or duration. The retained route runner reaches the 1,000-frame budget and then emits its known post-budget `bool` assignment exception, so per-frame decision positions/input outcomes are retained but no durable route-summary claim is made for the new fixed arms. Live quality is available only where prediction and GT evidence exist. Full-local is an offline quality/output-size reference; full remote has payload arithmetic only—no remote inference, latency or quality measurement.

No further experiment follows this close-out.
"""


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-root",type=Path,required=True)
    parser.add_argument("--action0-root",type=Path,required=True)
    parser.add_argument("--arm-a-root",type=Path,required=True)
    parser.add_argument("--arm-b-root",type=Path,required=True)
    parser.add_argument("--agent-replay",type=Path,required=True)
    parser.add_argument("--arm-a-replay",type=Path,required=True)
    parser.add_argument("--arm-b-replay",type=Path,required=True)
    parser.add_argument("--fixed-capture",type=Path,required=True)
    parser.add_argument("--preflight",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv);out=args.out;out.mkdir(parents=False,exist_ok=False)
    roots=dict(agent=args.agent_root,action0=args.action0_root,arm_a=args.arm_a_root,arm_b=args.arm_b_root)
    sessions={key:load_session(key,root) for key,root in roots.items()}
    replays={"agent":read_json(args.agent_replay),"arm_a":read_json(args.arm_a_replay),"arm_b":read_json(args.arm_b_replay)}
    paired={key:replay_quality(report) for key,report in replays.items()}
    live_all={key:live_quality(sessions[key]) for key in PAIR_ORDER}
    full_local_sizes={key:np.asarray([row["full_local_export"]["bytes"] for row in replays[key]["rows"] if row["admission"]=="ADMITTED"],float) for key in PAIR_ORDER}
    remote=np.asarray([NATIVE_BGRA_BYTES+RADAR_RETURN_BYTES*int(row["window_returns"]) for row in sessions["agent"]["sent"]],float)
    if not (4.29e6 < np.percentile(remote,50) < 4.31e6 and 4.31e6 < np.percentile(remote,99) < 4.34e6):
        raise ValueError("native full-remote independent check outside expected envelope")
    payload={key:{**q(sessions[key]["payload_bytes"]),"n":len(sessions[key]["payload_bytes"]),"kind":"MEASURED_SFD4_APPLICATION_PAYLOAD","representation_subtract_framing_bytes":SFD4_FRAMING_BYTES,"datagrams":q(sessions[key]["datagrams"],(50,99))} for key in ORDER}
    payload["full_remote"]={**q(remote,(50,99)),"n":len(remote),"kind":"CALCULATED_NATIVE_BGRA_PLUS_COMPLETE_TWO_SWEEP_RAW_RADAR","formula":"1280*720*4 + 16*two_sweep_raw_radar_return_count","rgb_bytes":NATIVE_BGRA_BYTES,"radar_bytes_per_return":RADAR_RETURN_BYTES,"metadata_and_framing":"EXCLUDED","remote_inference":False,"minimum_streaming_claim":False}
    pooled=np.concatenate(list(full_local_sizes.values()))
    local_stats={key:{**q(values,(50,99)),"n":len(values)} for key,values in full_local_sizes.items()}
    local_stats["pooled"]={**q(pooled,(50,99)),"n":len(pooled)}
    local_manifest=read_json(args.agent_replay.parent/"FULL_LOCAL_EXPORT_MANIFEST.json")

    session_summary={}
    for key,s in sessions.items():
        den=s["frame"]["decisions"]
        session_summary[key]=dict(
            label=LABELS[key],frame=s["frame"],controller_outcomes=s["controller_outcomes"],
            edge_typed_outcomes=s["edge_typed_outcomes"],
            prediction_completion=dict(
                definition="successful live edge.compute plus CPU prediction materialized",
                denominator=den,completed=len(s["latency_ms"]),incomplete=den-len(s["latency_ms"]),
                fraction=ratio(len(s["latency_ms"]),den),completed_by_170=int(np.sum(s["latency_ms"]<=170)),
                fraction_by_170=ratio(int(np.sum(s["latency_ms"]<=170)),den),
                completion_latency_ms=q(s["latency_ms"])),
            feedback=dict(observed=len(s["feedback_ms"]),received_by_170=int(np.sum(s["feedback_ms"]<=170)),
                          denominator=den,receipt_latency_ms=q(s["feedback_ms"])),
            live_quality=(strip_errors(live_all[key]) if key in live_all else "UNAVAILABLE_ZERO_COMPLETION"))
    verdicts={
        "arm_a":read_json(args.fixed_capture/"VERDICT_fixed_ae32_uint8_a54_v3.json"),
        "arm_b":read_json(args.fixed_capture/"VERDICT_fixed_mode11_q6000_v3.json"),
        "action0":read_json(MAIN/"experiments/splitfusion_m1_baselines_v1/static_capture_02/VERDICT_static_v2.json")}
    summary=dict(
        schema="splitfusion.m1_baselines_v1.completed_summary.v3",status="M1_COMPLETE_NO_FURTHER_EXPERIMENT",
        actor_boundary_sha256="bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce",
        session_comparability="all live methods are separate sessions; same-input pairing exists only within each retained live/full-local pair",
        deadline=dict(value_ms=170,label="original feedback deadline"),sessions=session_summary,
        paired_quality={key:strip_errors(value) for key,value in paired.items()},
        payload=payload,
        full_local_detection_output=dict(
            **local_stats,serializer=local_manifest["serializer"],encoding=local_manifest["encoding"],
            compression=local_manifest["compression"],included=local_manifest["included"],excluded=local_manifest["excluded"],
            fields=local_manifest["record_fields"],network_transmission=False,input_uplink_bytes=0),
        verdicts=verdicts,preflight=read_json(args.preflight),
        correction_action0_frame1338=dict(
            prior_claim="STALE_BEFORE_MAP frame completed inference",corrected=False,
            actual="expired EDGE_BEFORE_TAIL; edge.compute/frozen-tail spans ok=false; no CPU prediction materialized",
            successful_predictions=0,latency_sample=False,map_publication_is_completion=False),
        limitations=dict(
            route_summary="new fixed arms lack durable route summary after known post-budget runner exception; per-frame trajectory retained",
            full_local="offline quality/output-size reference only",full_remote="payload arithmetic only; no inference/latency/quality",
            action0_quality="unavailable; no live predictions and no offline substitution"),
        source_sha256={"analyze_v3.py":sha(__file__),"fixed_verdict_v3.py":sha(Path(__file__).with_name("fixed_verdict_v3.py")),"replay_export_v3.py":sha(Path(__file__).with_name("replay_export_v3.py"))})
    (out/"SUMMARY.json").write_text(json.dumps(summary,indent=1,sort_keys=True,allow_nan=False)+"\n")
    write_cdf_csv(out,paired)
    with (out/"METRICS.csv").open("x",newline="") as stream:
        fields=["session","offered","transmitted","processed","published","decisions","holds","fallbacks","actor_calls","prediction_completed","prediction_incomplete","prediction_by_170","feedback_by_170","payload_p50_bytes","payload_p95_bytes","payload_p99_bytes","elapsed_s","dropped_replaced"]
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for key in ORDER:
            s=session_summary[key];f=s["frame"];p=payload[key]
            w.writerow(dict(session=key,offered=f["offered"],transmitted=f["transmitted"],processed=f["successfully_processed"],published=f["published"],decisions=f["decisions"],holds=f["holds"],fallbacks=f["fallbacks"],actor_calls=f["actor_calls"],prediction_completed=s["prediction_completion"]["completed"],prediction_incomplete=s["prediction_completion"]["incomplete"],prediction_by_170=s["prediction_completion"]["completed_by_170"],feedback_by_170=s["feedback"]["received_by_170"],payload_p50_bytes=p["p50"],payload_p95_bytes=p["p95"],payload_p99_bytes=p["p99"],elapsed_s=f["elapsed_s"],dropped_replaced=f["route_input_outcomes"].get("DROPPED_REPLACED_BY_NEWER_FRAME",0)))
    with (out/"FULL_LOCAL_OUTPUT_SIZES.csv").open("x",newline="") as stream:
        w=csv.writer(stream);w.writerow(["session","ordinal","frame_id","bytes","records","sha256","compression","transmitted"])
        for key in PAIR_ORDER:
            for row in replays[key]["rows"]:
                if row["admission"]=="ADMITTED":
                    e=row["full_local_export"];w.writerow([key,row["ordinal"],row["frame_id"],e["bytes"],e["records"],e["sha256"],"NONE",False])
    fig_localization(out,paired);fig_completion(out,sessions);fig_data_size(out,sessions,full_local_sizes,remote)
    paired_bar_metrics(out,paired);fig_delivered(out,paired,live_all);fig_feedback(out,sessions)
    fig_payload_cdf(out,sessions,full_local_sizes,remote);fig_table(out,sessions,payload,local_stats,paired)
    (out/"TABLE.md").write_text(build_table(summary))
    (out/"REPORT.md").write_text(build_report(summary))
    command=("python3 -B -m rl_agent.splitfusion_m1_baselines_v1.analyze_v3 "
             f"--agent-root {args.agent_root} --action0-root {args.action0_root} "
             f"--arm-a-root {args.arm_a_root} --arm-b-root {args.arm_b_root} "
             f"--agent-replay {args.agent_replay} --arm-a-replay {args.arm_a_replay} "
             f"--arm-b-replay {args.arm_b_replay} --fixed-capture {args.fixed_capture} "
             f"--preflight {args.preflight} --out {args.out}")
    (out/"REPRODUCE.md").write_text("# Reproduce analysis\n\n```bash\n"+command+"\n```\n\nThis is analysis only; it starts no live service or experiment.\n")
    inputs={str(path):sha(path) for path in [
        args.agent_replay,args.arm_a_replay,args.arm_b_replay,args.preflight,
        args.fixed_capture/"FREEZE.json",args.fixed_capture/"CAMPAIGN_RESULT.json",
        args.fixed_capture/"VERDICT_fixed_ae32_uint8_a54_v3.json",
        args.fixed_capture/"VERDICT_fixed_mode11_q6000_v3.json",
        *[roots[key]/"registration.json" for key in ORDER],
        *[sessions[key]["route_path"] for key in ORDER],
        *[find_one(roots[key]/"capture","PHASE6_UE_EVIDENCE.json") for key in ORDER],
        *[find_one(roots[key]/"capture","edge_report.json") for key in ORDER]]}
    (out/"INPUT_SHA256.json").write_text(json.dumps(inputs,indent=1,sort_keys=True)+"\n")
    outputs={path.name:sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.name!="OUTPUT_SHA256.json"}
    (out/"OUTPUT_SHA256.json").write_text(json.dumps(outputs,indent=1,sort_keys=True)+"\n")
    print(json.dumps(dict(status=summary["status"],output=str(out),payload=payload,
                          full_local=local_stats),indent=1))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
