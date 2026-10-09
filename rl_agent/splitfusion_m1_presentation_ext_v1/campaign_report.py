"""Cumulative static-mode v13 campaign report (analysis only; batch outputs are never edited).

For every completed arm: the pre-registered assessment verdict and raw rc, completion outcomes and curves (all
decisions in the denominator; zero completions are a flat zero curve with no latency percentiles), payload P50/P99,
feedback receipt (separate), and deadline-aware quality with explicit GT-coverage labels (missing GT is never zero:
values with unknown GT on failed decisions are upper bounds; segmentation without retained GT is reported as
UNMEASURED).  Precision is reported both over all completions and over timely completions only.
Batch-1 rows are recomputed from the same evidence and must equal the sealed batch-1 report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from rl_agent.splitfusion_m1_baselines_v1 import analyze_v3 as A3
from rl_agent.splitfusion_m1_presentation_ext_v1 import metrics as MX
from rl_agent.splitfusion_m1_presentation_ext_v1 import render as R
from rl_agent.splitfusion_static_modes_v13_offline_v1 import completion as CP

WT = MX.MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
CAPTURES = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_static_modes_v13"
ORDER = ("agent", "mode08", "mode11", "mode03", "mode02", "mode01", "mode10", "mode07", "mode00", "mode06", "mode04",
         "mode09", "mode05")
FAMILY = {m: ("noAE", "AE128", "AE64", "AE32")[m // 3] for m in range(12)}
QUANT = {m: ("UINT8", "UINT6", "UINT4")[m % 3] for m in range(12)}
BATCH1 = MX.MAIN / "experiments/splitfusion_static_modes_v13/batch1_report_01/BATCH1_REPORT.json"


def label(arm):
    if arm == "agent":
        return "Agent (deployed actor)"
    m = int(arm[4:])
    return f"Mode {m} · {FAMILY[m]}/{QUANT[m]} · q=0"


def completed_arms():
    return [a for a in ORDER if (CAPTURES / a / "ASSESSMENT_V13.json").exists()]


def collect(arms):
    eligibility = MX.EligibilityFromTarget()
    A3.LABELS.update({a: label(a) for a in arms})
    out = {}
    for arm in arms:
        root = CAPTURES / arm
        assessment = json.loads((root / "ASSESSMENT_V13.json").read_bytes())
        acct = CP.account(root)
        row = dict(arm=arm, label=label(arm), verdict=assessment["verdict"], subtype=assessment["subtype"],
                   launch_rc=int((root / "campaign_launch.rc").read_text()), frames=assessment["frames"],
                   payload=assessment["payload_bytes"], latencies=sorted(acct["latency_ms"]),
                   completion={k: assessment["prediction_completion"][k] for k in (
                       "decisions", "completed", "completed_by_170", "completed_later", "no_completed_prediction",
                       "no_completion_subcategories", "latency_ms")},
                   non_pass_checks={k: v for k, v in assessment["checks"].items() if v != "PASS"},
                   gt_capacity_refusals=assessment["gt_capacity_refusals_reported"])
        if assessment["verdict"] == "VALID_MEASUREMENT":
            session = A3.load_session(arm, root)
            opps, validated = MX.opportunities(arm, session, eligibility)
            summary = MX.session_summary(arm, session, opps, validated)
            if (summary["completed"], summary["completed_by_170"]) != (row["completion"]["completed"],
                                                                       row["completion"]["completed_by_170"]):
                raise ValueError(f"{arm}: quality rows disagree with the registered assessment")
            quality = {}
            for c in MX.CLASSES:
                loc, seg = summary["localisation"][c], summary["segmentation"][c]
                quality[c] = dict(
                    recall_all=dict(n=loc["all_completion"]["numerator"], d=loc["all_completion"]["denominator"],
                                    value=loc["all_completion"]["recall"]),
                    recall_deadline=dict(n=loc["deadline_aware"]["numerator"], d=loc["deadline_aware"]["denominator"],
                                         value=loc["deadline_aware"]["recall"], bound=loc["deadline_aware"]["bound"]),
                    precision_all=loc["all_completion"]["precision"],
                    precision_timely=loc["deadline_aware"]["precision_timely"],
                    seg_iou_all=seg["all_completion"]["iou"], seg_iou_deadline=seg["deadline_aware"]["iou"],
                    seg_bound=seg["deadline_aware"]["bound"],
                    seg_evaluable=f"{seg['deadline_aware']['evaluable_opportunities']}/"
                                  f"{seg['deadline_aware']['opportunities']}")
            row.update(feedback_by_170=summary["feedback_by_170"], gt_coverage=summary["gt_coverage"],
                       quality=quality)
        else:
            row.update(feedback_by_170=None, gt_coverage=None, quality=None,
                       quality_note="not computed: the registered assessment did not classify this arm as valid")
        out[arm] = row
    return out


SERIES, LIGHT, REF = "#2a78d6", "#9ec5f4", "#8a8f98"     # one hue (reference palette slot 1) + neutral reference


def fig_completion(plots, rows):
    """Small multiples: one panel per arm (single hue), the agent curve as a grey reference in every panel."""
    arms = list(rows)
    fig = R.canvas("Static-mode campaign · prediction completion (cumulative)",
                   "FAVORABLE_STABLE · 1,000 transmitted frames per arm · separate live sessions in the frozen order",
                   "Completion: action-open → successful edge.compute with CPU prediction materialized; all decisions in "
                   "the denominator. Grey = agent reference.\n170 ms = original feedback-deadline reference. Zero "
                   "completions = flat zero curve, not a latency distribution. Counts: ≤170 / later / none.")
    cols = 3 if len(arms) <= 6 else 5
    nrows = -(-len(arms) // cols)
    grid = fig.add_gridspec(nrows, cols, left=.07, right=.97, top=.80, bottom=.235, hspace=.75, wspace=.12)
    xmax = 600
    agent = rows.get("agent")

    def curve(row):
        lat, n = row["latencies"], row["completion"]["decisions"]
        x = np.concatenate(([0.0], lat, [xmax]))
        y = np.concatenate(([0.0], np.arange(1, len(lat) + 1) / n, [len(lat) / n]))
        return x, y
    for i, arm in enumerate(arms):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        if agent is not None and arm != "agent":
            ax.step(*curve(agent), where="post", color=REF, lw=1.8)
        ax.step(*curve(rows[arm]), where="post", color=SERIES, lw=3.0)
        ax.axvline(170, color=R.INK, ls=":", lw=1.5)
        c = rows[arm]["completion"]
        ax.set_title(f"{rows[arm]['label']}\n{c['completed_by_170']} / {c['completed_later']} / "
                     f"{c['no_completed_prediction']}", loc="left", fontsize=13)
        ax.set(xlim=(0, xmax), ylim=(-.03, 1.05))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_yticks([0, .5, 1])
        ax.set_xticks([0, 170, 400])
        ax.tick_params(labelsize=12, labelleft=i % cols == 0)
        ax.grid(True)
    fig.text(.52, .145, "Time after action-open (ms)", ha="center", fontsize=16)
    R.save(fig, plots / "01_campaign_prediction_completion.png")


def fig_outcomes(plots, rows):
    arms = list(rows)
    fig = R.canvas("Static-mode campaign · prediction outcomes per policy decision",
                   "Completed ≤170 ms · completed later · no completed prediction · every decision in the denominator",
                   "No completed prediction keeps its terminal subcategories in CAMPAIGN_REPORT.json; it is never "
                   "relabelled as packet loss.")
    ax = fig.add_axes([.27, .26, .62, .52])
    k = len(arms)
    for i, arm in enumerate(arms):
        c = rows[arm]["completion"]
        n = c["decisions"]
        t, l, none = c["completed_by_170"], c["completed_later"], c["no_completed_prediction"]
        y = k - 1 - i
        ax.barh(y, t / n, height=.6, color=SERIES)
        ax.barh(y, l / n, left=t / n, height=.6, color=LIGHT)
        ax.barh(y, none / n, left=(t + l) / n, height=.6, color="#EDF0F3", hatch="///", edgecolor="#8895A1")
        ax.text(1.01, y, f"{t} / {l} / {none}", va="center", fontsize=12.5)
    ax.set(yticks=list(range(k - 1, -1, -1)), yticklabels=[rows[a]["label"] for a in arms], xlim=(0, 1),
           ylim=(-.6, k - .4), xlabel="Share of policy decisions")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0, labelsize=12)
    fig.legend(handles=[Patch(facecolor=SERIES, label="Completed ≤170 ms"), Patch(facecolor=LIGHT, label="Completed later"),
                        Patch(facecolor="#EDF0F3", edgecolor="#8895A1", hatch="///", label="No completed prediction")],
               loc="center", bbox_to_anchor=(.55, .135), ncol=3, fontsize=14)
    R.save(fig, plots / "02_campaign_outcomes.png")


def fig_payload(plots, rows):
    arms = list(rows)
    fig = R.canvas("Static-mode campaign · payload per transmitted frame",
                   "P50 (solid) and P99 (hatched) · measured SFD4 application-frame bytes · n = 1,000 frames per arm",
                   "SFD4 application layer (zstd payload + 455-byte framing); IP/GTP overhead not included. Log scale; "
                   "1 kB = 1,000 B.")
    ax = fig.add_axes([.27, .22, .60, .56])
    k = len(arms)
    for i, arm in enumerate(arms):
        p = rows[arm]["payload"]
        y = k - 1 - i
        for off, stat in ((.18, "p50"), (-.18, "p99")):
            ax.barh(y + off, p[stat] / 1e3, height=.32, color=SERIES if stat == "p50" else "white",
                    edgecolor=SERIES, hatch="///" if stat == "p99" else None, linewidth=1.4)
            ax.text(p[stat] / 1e3 * 1.08, y + off, f"{p[stat] / 1e3:,.0f}", va="center", fontsize=12)
    ax.set_xscale("log")
    ax.set(yticks=list(range(k - 1, -1, -1)), yticklabels=[rows[a]["label"] for a in arms], xlim=(50, 6000),
           ylim=(-.6, k - .4), xlabel="kB per transmitted frame (log)")
    from matplotlib.ticker import NullFormatter, NullLocator
    ax.set_xticks([100, 300, 1000, 3000], ["100", "300", "1,000", "3,000"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0, labelsize=12)
    R.save(fig, plots / "03_campaign_payload.png")


def fmt(q, key):
    v = q.get(key)
    return "n/a" if v is None else f"{v:.3f}"


def table(rows):
    lines = ["| Arm | rc | Verdict | Completed / dec. | ≤170 | Later | None | P50/P99 ms | Payload P50/P99 kB | Feedback ≤170 |",
             "|---|---:|---|---:|---:|---:|---:|---|---|---:|"]
    for arm, r in rows.items():
        c = r["completion"]
        lat = c["latency_ms"]
        latency = "— (no completion)" if lat.get("p50") is None else f"{lat['p50']:.1f} / {lat['p99']:.1f}"
        fb = "—" if r["feedback_by_170"] is None else f"{r['feedback_by_170']}/{c['decisions']}"
        lines.append(f"| {r['label']} | {r['launch_rc']} | {r['verdict']}{' · ' + r['subtype'] if r['subtype'] else ''} "
                     f"| {c['completed']}/{c['decisions']} | {c['completed_by_170']} | {c['completed_later']} | "
                     f"{c['no_completed_prediction']} | {latency} | {r['payload']['p50'] / 1e3:.1f} / "
                     f"{r['payload']['p99'] / 1e3:.1f} | {fb} |")
    lines += ["", "≥ = denominator counts only decisions with GT objects; decisions without GT (NO_GT_PRODUCER_CAPACITY) "
              "are listed in gt_coverage and never counted as zero objects.", "",
              "| Arm | Class | Recall all | Recall ≤170 (bound) | Precision all / timely | Seg IoU all → ≤170 (evaluable) |",
              "|---|---|---|---|---|---|"]
    for arm, r in rows.items():
        if not r["quality"]:
            lines.append(f"| {r['label']} | — | {r.get('quality_note')} | | | |")
            continue
        missing_gt = any(k.endswith("MISSING") for k in (r["gt_coverage"] or {}))
        no_timely = r["completion"]["completed_by_170"] == 0
        for cls, q in r["quality"].items():
            ra, rd = q["recall_all"], q["recall_deadline"]
            if ra["value"] is None:
                all_txt = ("n/a (0 completed)" if r["completion"]["completed"] == 0
                           else "n/a (no eligible GT on the completed frames)")
            else:
                all_txt = f"{ra['n']}/{ra['d']} = {ra['value']:.3f}"
            ge = "≥" if missing_gt else ""
            dl = f"{rd['n']}/{ge}{rd['d']} = {rd['value']:.3f} ({rd['bound']})"
            if no_timely:
                seg_dl = ("0 by construction (no timely prediction; intersection 0)" if q["seg_iou_deadline"] is not None
                          else "0 wherever GT foreground exists (no timely prediction); union UNMEASURED")
            elif q["seg_iou_deadline"] is None:
                seg_dl = "UNMEASURED (GT seg not retained)"
            else:
                seg_dl = f"{q['seg_iou_deadline']:.3f} ({q['seg_bound']}, {q['seg_evaluable']})"
            lines.append(f"| {r['label']} | {cls} | {all_txt} | {dl} | {fmt(q, 'precision_all')} / "
                         f"{fmt(q, 'precision_timely')} | {fmt(q, 'seg_iou_all')} → {seg_dl} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=False)
    arms = completed_arms()
    rows = collect(arms)
    sealed = json.loads(BATCH1.read_bytes())["arms"]
    batch1_equal = {}
    for arm, old in sealed.items():
        new = rows[arm]
        batch1_equal[arm] = (old["completion"]["completed"] == new["completion"]["completed"]
                             and old["completion"]["completed_by_170"] == new["completion"]["completed_by_170"]
                             and old["payload"] == new["payload"] and old["launch_rc"] == new["launch_rc"]
                             and all(old["quality"][c]["recall_deadline"][:2] == [new["quality"][c]["recall_deadline"]["n"],
                                                                                 new["quality"][c]["recall_deadline"]["d"]]
                                     for c in MX.CLASSES))
    if not all(batch1_equal.values()):
        raise ValueError("cumulative recomputation differs from the sealed batch-1 report: " + repr(batch1_equal))
    fig_completion(plots, rows)
    fig_outcomes(plots, rows)
    fig_payload(plots, rows)
    (args.out / "CAMPAIGN_TABLE.md").write_text(table(rows))
    (args.out / "CAMPAIGN_REPORT.json").write_text(json.dumps(dict(
        schema="splitfusion.static_modes_v13.campaign_report.v1", arms={a: {k: v for k, v in r.items() if k != "latencies"}
                                                                    for a, r in rows.items()},
        batch1_rows_equal_sealed=batch1_equal, order=list(ORDER)), indent=1, sort_keys=True, default=str) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
