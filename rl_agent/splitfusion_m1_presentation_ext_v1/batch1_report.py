"""Report for static-mode v13 batch 1 (agent, mode08, mode11): completion outcomes and curves, payload P50/P99,
available deadline-aware quality, cleanup and disk.  Analysis only: reads the three capture roots and their
pre-registered assessments; recomputes nothing the assessment decides.  Kept outside the v13 runtime packages so their
sealed source inventory is untouched.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from rl_agent.splitfusion_m1_baselines_v1 import analyze_v3 as A3
from rl_agent.splitfusion_m1_presentation_ext_v1 import metrics as MX
from rl_agent.splitfusion_m1_presentation_ext_v1 import render as R
from rl_agent.splitfusion_static_modes_v13_offline_v1 import completion as CP

WT = MX.MAIN.parent / "abiodun_run4_single_gpu_profiler_v1"
CAPTURES = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_static_modes_v13"
ARMS = ("agent", "mode08", "mode11")
LABEL = {"agent": "Agent (deployed actor)", "mode08": "Mode 8 · AE64/UINT4 · q=0",
         "mode11": "Mode 11 · AE32/UINT4 · q=0"}
COLOR = {"agent": "#087F5B", "mode08": "#B88209", "mode11": "#7651A8"}


def collect():
    eligibility = MX.EligibilityFromTarget()
    A3.LABELS.update(LABEL)                         # analyze_v3 labels sessions by key
    out = {}
    for arm in ARMS:
        root = CAPTURES / arm
        assessment = json.loads((root / "ASSESSMENT_V13.json").read_bytes())
        session = A3.load_session(arm, root)
        opps, validated = MX.opportunities(arm, session, eligibility)
        summary = MX.session_summary(arm, session, opps, validated)
        acct = CP.account(root)
        pc = assessment["prediction_completion"]
        if (summary["completed"], summary["completed_by_170"]) != (pc["completed"], pc["completed_by_170"]):
            raise ValueError(f"{arm}: quality rows disagree with the registered assessment counts")
        out[arm] = dict(assessment=assessment, summary=summary, latencies=sorted(acct["latency_ms"]),
                        decisions=acct["decisions"], launch_rc=int((root / "campaign_launch.rc").read_text()))
    return out


def fig_completion(plots, data):
    fig = R.canvas("Batch 1 · prediction completion",
                   "FAVORABLE_STABLE · 1,000 transmitted frames per arm · separate live sessions, run back to back",
                   "Completion: action-open → successful edge.compute with CPU prediction materialized; all decisions in "
                   "the denominator.\n170 ms = original feedback-deadline reference. TIMEOUT/STALE, map publication and "
                   "feedback receipt are never completion times.")
    ax = fig.add_axes([.075, .30, .48, .48])
    xmax = 450
    for arm in ARMS:
        lat, n = data[arm]["latencies"], data[arm]["decisions"]
        x = np.concatenate(([0.0], lat, [xmax]))
        y = np.concatenate(([0.0], np.arange(1, len(lat) + 1) / n, [len(lat) / n]))
        ax.step(x, y, where="post", color=COLOR[arm], lw=3.4, label=LABEL[arm])
    ax.axvline(170, color=R.INK, ls=":", lw=2)
    ax.text(174, .06, "170 ms", fontsize=15)
    ax.set(xlim=(0, xmax), ylim=(0, 1.03), xlabel="Time after action-open (ms)",
           ylabel="Decisions with completed prediction")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True)
    ax.set_title("Completion curve", loc="left", fontsize=20)
    ax.legend(loc="lower right", fontsize=14)
    bx = fig.add_axes([.68, .30, .29, .48])
    for i, arm in enumerate(ARMS):
        s = data[arm]["summary"]["prediction_states"]
        n = data[arm]["decisions"]
        t, l, none = s.get("TIMELY", 0), s.get("LATE", 0), s.get("NONE", 0)
        y = 2 - i
        bx.barh(y, t / n, height=.5, color=COLOR[arm])
        bx.barh(y, l / n, left=t / n, height=.5, color=COLOR[arm], alpha=.35)
        bx.barh(y, none / n, left=(t + l) / n, height=.5, color="#EDF0F3", hatch="///", edgecolor="#8895A1")
        bx.text(.02, y + .36, f"{t}/{n} ≤170 · {l} later · {none} none", fontsize=13)
    bx.set(yticks=[2, 1, 0], yticklabels=["Agent", "Mode 8", "Mode 11"], xlim=(0, 1), ylim=(-.5, 2.8),
           xlabel="Share of policy decisions")
    bx.xaxis.set_major_formatter(PercentFormatter(1))
    bx.grid(axis="x")
    bx.set_title("Outcomes", loc="left", fontsize=20)
    fig.legend(handles=[Patch(facecolor=R.INK, label="Completed ≤170 ms"),
                        Patch(facecolor="#B5C0C9", label="Completed later"),
                        Patch(facecolor="#EDF0F3", edgecolor="#8895A1", hatch="///", label="No completed prediction")],
               loc="center", bbox_to_anchor=(.5, .175), ncol=3, fontsize=14)
    R.save(fig, plots / "01_batch1_prediction_completion.png")


def fig_payload(plots, data):
    fig = R.canvas("Batch 1 · application payload per transmitted frame",
                   "P50 and P99 · measured SFD4 application-frame bytes · n = 1,000 transmitted frames per arm",
                   "SFD4 application layer (zstd payload + 455-byte framing); IP/GTP overhead not included. 1 kB = 1,000 B.")
    ax = fig.add_axes([.27, .22, .60, .55])
    for i, arm in enumerate(ARMS):
        p = data[arm]["assessment"]["payload_bytes"]
        y = 2 - i
        for off, stat in ((.17, "p50"), (-.17, "p99")):
            ax.barh(y + off, p[stat] / 1e3, height=.30, color=COLOR[arm] if stat == "p50" else "white",
                    edgecolor=COLOR[arm], hatch="///" if stat == "p99" else None, linewidth=1.6)
            ax.text(p[stat] / 1e3 + 6, y + off, f"{stat.upper()} {p[stat] / 1e3:.1f} kB", va="center", fontsize=15)
    ax.set(yticks=[2, 1, 0], yticklabels=[LABEL[a] for a in ARMS], xlim=(0, 620), ylim=(-.6, 2.6),
           xlabel="kB per transmitted frame", xticks=[0, 100, 200, 300, 400, 500, 600])
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    R.save(fig, plots / "02_batch1_payload.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plots = args.out / "plots"
    plots.mkdir(parents=True, exist_ok=False)
    data = collect()
    fig_completion(plots, data)
    fig_payload(plots, data)
    stage = sorted(glob.glob(str(WT / "experiments/splitfusion_static_modes_v13/campaign_stage_*.json")))[-1]
    report = {}
    for arm in ARMS:
        a, s = data[arm]["assessment"], data[arm]["summary"]
        quality = {}
        for c in MX.CLASSES:
            loc, seg = s["localisation"][c], s["segmentation"][c]
            quality[c] = dict(
                recall_all=[loc["all_completion"]["numerator"], loc["all_completion"]["denominator"],
                            loc["all_completion"]["recall"]],
                recall_deadline=[loc["deadline_aware"]["numerator"], loc["deadline_aware"]["denominator"],
                                 loc["deadline_aware"]["recall"], loc["deadline_aware"]["bound"]],
                precision_all=loc["all_completion"]["precision"], seg_iou_all=seg["all_completion"]["iou"],
                seg_iou_deadline=seg["deadline_aware"]["iou"], seg_bound=seg["deadline_aware"]["bound"])
        report[arm] = dict(verdict=a["verdict"], subtype=a["subtype"], launch_rc=data[arm]["launch_rc"],
                           frames=a["frames"], completion={k: a["prediction_completion"][k] for k in (
                               "decisions", "completed", "completed_by_170", "completed_later",
                               "no_completed_prediction", "latency_ms")},
                           feedback_by_170=s["feedback_by_170"], payload=a["payload_bytes"],
                           gt_coverage=s["gt_coverage"], quality=quality,
                           non_pass_checks={k: v for k, v in a["checks"].items() if v != "PASS"})
    document = dict(schema="splitfusion.static_modes_v13.batch1_report.v1", arms=report, stage_record=stage,
                    stage=json.loads(Path(stage).read_bytes()), tolerances_m=MX.TOL, deadline_ms=MX.DEADLINE_MS)
    (args.out / "BATCH1_REPORT.json").write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({a: dict(verdict=r["verdict"], rc=r["launch_rc"], completion=r["completion"],
                              payload=r["payload"], quality=r["quality"], feedback=r["feedback_by_170"])
                      for a, r in report.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
