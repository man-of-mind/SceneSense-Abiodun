"""Meeting figures for the pilot-actor 1,000-frame validation (valid_02), in the Run-4 presentation style.

Inputs (read-only): valid_02/plots/DECISIONS_valid1000.jsonl and the capture's PHASE6_UE_EVIDENCE.json
(transmitted_identities, in transmission order -> transmitted-frame number = position + 1).
Outputs (create-only): experiments/splitfusion_run5_pilot_live_v1/presentation_01/.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "experiments/splitfusion_run5_pilot_live_v1"
CAPTURE_ROOT = (ROOT.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                "20261006_run5_pilot_live_v1")
RUN = dict(evidence="valid_02", tag="valid1000", root="valid_b1000_02", label="")     # set by main()
DEADLINE = 170.0
MODES = {5: "AE128 / UINT4", 8: "AE64 / UINT4", 11: "AE32 / UINT4"}
BRACKETS = {5: (0.70, 0.98), 8: (0.50, 0.98), 11: (0.30, 0.80)}
TEAL, RED, GREY, PURPLE = "#2a7f74", "#c0392b", "#7f7f7f", "#6b5b8a"
CONTEXT = ("Restricted pilot actor (CANDIDATE seed 43, update 10,000), frozen • Favorable stable • "
           "999 transmitted frames • 493 policy decisions")
PIPELINE_NOTE = ("Not like-for-like with Run-4: corrected 7-channel perception input, exact-GT early service and "
                 "v2 evaluator differ from the Run-4 capture.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    global CONTEXT
    decisions = EV / f"{RUN['evidence']}/plots/DECISIONS_{RUN['tag']}.jsonl"
    capture = CAPTURE_ROOT / RUN["root"] / "capture/cells/a71__favorable_stable"
    rows = [json.loads(line) for line in open(decisions)]
    evidence = json.loads((capture / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_bytes())
    order = [int(x["frame_id"]) for x in evidence["transmitted_identities"]]
    if len(order) != len(set(order)) or order != sorted(order):
        raise ValueError("transmitted identities must be unique and in frame order")
    number = {fid: i + 1 for i, fid in enumerate(order)}
    kinds = {int(x["frame_id"]): x["run4_identity"]["frame_kind"] for x in evidence["transmitted_identities"]}
    for r in rows:
        r["frame_number"] = number[int(r["frame_id"])]
        r["q"] = r["executed_action"]["q_e4"] / 1e4
        r["mode"] = r["executed_action"]["mode_id"]
    rows.sort(key=lambda r: r["frame_number"])
    CONTEXT = (f"Restricted pilot actor (CANDIDATE seed 43, update 10,000), frozen{RUN['label']} • Favorable stable • "
               f"{len(order)} transmitted frames • {len(rows)} policy decisions")
    inputs = dict(decisions=str(decisions), decisions_sha256=sha(decisions),
                  phase6=str(capture / "run4_phase6/PHASE6_UE_EVIDENCE.json"),
                  phase6_sha256=sha(capture / "run4_phase6/PHASE6_UE_EVIDENCE.json"),
                  transmitted=len(order), frame_kinds={k: sum(v == k for v in kinds.values()) for k in set(kinds.values())})
    return rows, inputs


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "font.weight": "bold",
        "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelsize": 14, "axes.labelweight": "bold",
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 100})


def finish(fig, out, name, title, foot, top=.86):
    fig.suptitle(title, fontsize=22, fontweight="bold", y=.985)
    fig.text(.5, .935, CONTEXT, ha="center", fontsize=13, fontweight="bold")
    fig.text(.04, .012, foot, fontsize=11.5, color="#404040", va="bottom", fontweight="bold")
    fig.subplots_adjust(top=top)
    fig.savefig(out / name, dpi=133, facecolor="white")
    plt.close(fig)


def fig_latency_snr(rows, out):
    x = np.array([r["frame_number"] for r in rows]); y = np.array([r["ue_receipt"] for r in rows])
    late = y > DEADLINE
    fig, (a, b) = plt.subplots(2, 1, figsize=(20, 12.9), sharex=True, gridspec_kw=dict(height_ratios=[2.1, 1], hspace=.28))
    a.plot(x, y, color="#1f5f99", lw=1.3, label="Measured feedback latency per decision")
    a.scatter(x[late], y[late], s=22, color=RED, zorder=3, label=f"Late feedback ({int(late.sum())})")
    a.axhline(DEADLINE, color="#b22222", ls="--", lw=2, label="Deadline: 170 ms (inclusive)")
    a.set_ylabel("Feedback latency (ms)"); a.set_ylim(0, max(450, y.max() + 20))
    a.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0, 1.13))
    b.plot(x, [r["snr_db"] for r in rows], color="#1d7f6e", lw=1.5)
    b.set_title("External SNR target (latest causal lease before each decision)", loc="left")
    b.set_ylabel("External SNR\ntarget (dB)"); b.set_ylim(0, 28)
    b.set_xlabel("Transmitted frame number"); b.set_xlim(0, 1000)
    timely = int((~late).sum())
    finish(fig, out, "01_feedback_latency_and_snr.png", "Feedback latency and network profile",
           f"All {len(rows)} measured replies (no smoothing); the line connects policy decisions, not held frames. "
           f"Timely: {timely}/{len(rows)} = {timely / len(rows):.1%} (includes "
           f"{sum(r['outcome_class'] == 'TIMELY_UNDEFINED' for r in rows)} undefined-quality replies).\n"
           "SNR is the external channel target used by the causal lease, not a UE measurement. "
           "Single-run descriptive results; no causal claim.\n" + PIPELINE_NOTE, top=.83)


def fig_quality(rows, out):
    fig, (a, b) = plt.subplots(2, 1, figsize=(20, 10.3), sharex=True, gridspec_kw=dict(height_ratios=[4, 1], hspace=.1))
    td = [r for r in rows if r["outcome_class"] == "TIMELY_DEFINED"]
    ld = [r for r in rows if r["outcome_class"] == "LATE" and r["quality_defined"]]
    tu = [r for r in rows if r["outcome_class"] == "TIMELY_UNDEFINED"]
    lu = [r for r in rows if r["outcome_class"] == "LATE" and not r["quality_defined"]]
    a.scatter([r["frame_number"] for r in td], [r["q_perc"] for r in td], s=34, color=TEAL, alpha=.9,
              label=f"Timely, defined quality ({len(td)})")
    a.scatter([r["frame_number"] for r in ld], [r["q_perc"] for r in ld], s=38, facecolors="none", edgecolors=RED,
              linewidths=1.6, label=f"Late, measured quality ({len(ld)}; timeout reward remains −1)")
    a.set_ylim(-.03, 1.0); a.set_ylabel("Perception quality Q_perc")
    a.legend(loc="upper left", ncol=2, frameon=True)
    b.scatter([r["frame_number"] for r in tu], [0] * len(tu), marker="|", s=260, color=PURPLE, linewidths=2,
              label=f"Undefined quality, timely ({len(tu)})")
    b.scatter([r["frame_number"] for r in lu], [0] * len(lu), marker="x", s=70, color=RED, linewidths=2,
              label=f"Undefined quality, late ({len(lu)})")
    b.set_yticks([0]); b.set_yticklabels(["No eligible GT"]); b.set_ylim(-1, 1)
    b.set_xlabel("Transmitted frame number"); b.set_xlim(0, 1000)
    b.legend(loc="upper right", ncol=2, frameon=True)
    finish(fig, out, "02_quality_over_frames.png", "Perception quality of the agent-selected actions",
           f"Q_perc is a composite perception-quality score, not segmentation or localization alone. "
           f"{len(td) + len(ld)} measured values; {len(tu) + len(lu)} undefined remain missing; "
           "no quality is imputed for held/fallback frames.\nLate quality is shown descriptively only. " + PIPELINE_NOTE,
           top=.87)


def fig_heatmap(rows, out):
    edges = np.round(np.arange(0, 1.0001, .1), 2); edges[-1] = .98
    modes = list(range(12))
    names = {0: "noAE / UINT8", 1: "noAE / UINT6", 2: "noAE / UINT4", 3: "AE128 / UINT8", 4: "AE128 / UINT6",
             5: "AE128 / UINT4", 6: "AE64 / UINT8", 7: "AE64 / UINT6", 8: "AE64 / UINT4", 9: "AE32 / UINT8",
             10: "AE32 / UINT6", 11: "AE32 / UINT4"}
    counts = np.zeros((12, len(edges) - 1), int)
    for r in rows:
        counts[r["mode"], min(np.searchsorted(edges, r["q"], side="right") - 1, len(edges) - 2)] += 1
    fig = plt.figure(figsize=(20, 10.6))
    a = fig.add_axes([.17, .17, .52, .64]); c = fig.add_axes([.70, .17, .012, .64]); t = fig.add_axes([.80, .17, .16, .64])
    im = a.imshow(counts, aspect="auto", cmap="Blues", vmin=0, origin="upper")
    for m in modes:
        for j in range(counts.shape[1]):
            if counts[m, j]:
                a.text(j, m, str(counts[m, j]), ha="center", va="center", fontsize=13, fontweight="bold",
                       color="white" if counts[m, j] > counts.max() * .55 else "black")
        if m not in MODES:
            a.axhspan(m - .5, m + .5, color="#e6e6e6", zorder=0, alpha=.9)
            a.text(len(edges) - 1.6, m, "masked", ha="right", va="center", fontsize=11, color=GREY, fontstyle="italic")
    def pos(q):                      # continuous q -> heatmap x (bin j spans [j-0.5, j+0.5]; last bin is 0.90-0.98)
        return q / .1 - .5 if q <= .9 else 8.5 + (q - .9) / .08
    for m, (lo, hi) in BRACKETS.items():
        a.plot([pos(lo), pos(hi)], [m + .38, m + .38], color="#333", lw=2.6, ls=":")
    a.set_yticks(modes); a.set_yticklabels([f"{m}: {names[m]}" for m in modes])
    a.set_xticks(range(len(edges) - 1)); a.set_xticklabels([f"{edges[j]:.1f}–{edges[j+1]:.2f}".replace("0.98", ".98") for j in range(len(edges) - 1)],
                                                        rotation=35, ha="right")
    a.set_xlabel("Selected drop fraction q (higher q = more cells dropped)"); a.set_ylabel("Selected mode"); a.grid(False)
    fig.colorbar(im, cax=c).set_label("Policy decision count", fontweight="bold")
    totals = counts.sum(1)
    t.barh(modes, totals, color="#2f6aa3"); t.invert_yaxis(); t.set_yticks(modes)
    for m in modes:
        t.text(totals[m] + 4, m, str(totals[m]), va="center", fontsize=12, fontweight="bold")
    t.set_title("Total per mode"); t.set_xlabel("Decisions"); t.set_xlim(0, max(totals) * 1.25)
    finish(fig, out, "03_action_heatmap.png", "Which actions did the agent select?",
           f"{len(rows)} new policy decisions only; startup/stale-input fallbacks (mode 11, q 0.98) and held frames are "
           "not counted. Grey rows: modes masked by the restricted support (only 5, 8, 11 allowed).\n"
           "Dotted line: registered q bracket per allowed mode (5: 0.70–0.98, 8: 0.50–0.98, 11: 0.30–0.80). "
           "Continuous q is binned for display only.\n" + PIPELINE_NOTE, top=.86)


def fig_latency_q(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10.2), sharey=True)
    norm = Normalize(0, 1)
    sc = None
    for a, m in zip(axes, (5, 8, 11)):
        sub = [r for r in rows if r["mode"] == m]
        lo, hi = BRACKETS[m]
        a.axvspan(lo, hi, color="#eef3f8", zorder=0)
        a.axhline(DEADLINE, color="#b22222", ls="--", lw=2)
        d = [r for r in sub if r["quality_defined"]]; u = [r for r in sub if not r["quality_defined"]]
        if d:
            sc = a.scatter([r["q"] for r in d], [r["ue_receipt"] for r in d], c=[r["q_perc"] for r in d], cmap="viridis",
                           norm=norm, s=34, alpha=.85, edgecolors="none")
        if u:
            a.scatter([r["q"] for r in u], [r["ue_receipt"] for r in u], marker="x", color=GREY, s=50, linewidths=1.8)
        if not sub:
            a.text(.5, 300, "Not selected", ha="center", color=GREY, fontsize=15)
        late = sum(r["ue_receipt"] > DEADLINE for r in sub)
        a.set_title(f"{m}: {MODES[m]}\nn={len(sub)}, late={late}")
        a.set_xlim(0, .98); a.set_ylim(0, 450)
    axes[0].set_ylabel("Action open → feedback receipt (ms)")
    fig.supxlabel("Selected drop fraction q (shaded: registered bracket)", fontweight="bold", y=.125)
    cax = fig.add_axes([.925, .2, .012, .6])
    fig.colorbar(sc, cax=cax).set_label("Measured Q_perc", fontweight="bold")
    fig.subplots_adjust(left=.07, right=.9, bottom=.2, wspace=.12)
    finish(fig, out, "04_latency_by_selected_q.png", "Latency of each selected (mode, q) action",
           "One dot per policy decision; colour = measured Q_perc; grey crosses = undefined quality (no eligible GT). "
           "Dashed line: 170-ms deadline.\nObserved outcomes, not causal action comparisons: scene, q and route position are "
           "confounded in one run.\n" + PIPELINE_NOTE, top=.8)


def main(name="presentation_01", evidence="valid_02", tag="valid1000", root="valid_b1000_02", label=""):
    RUN.update(evidence=evidence, tag=tag, root=root, label=label)
    out = EV / name
    out.mkdir(parents=False, exist_ok=False)
    style()
    rows, inputs = load()
    fig_latency_snr(rows, out); fig_quality(rows, out); fig_heatmap(rows, out); fig_latency_q(rows, out)
    backup = out / "backup"
    backup.mkdir()
    for n in ("04_si_p40_latency_quality", "05_localization", "06_segmentation", "07_freshness", "08_cdf_exceedance",
              "03_snr_latency_quality"):
        shutil.copy2(EV / f"{evidence}/plots/{n}_{tag}.png", backup / f"B_{n}_{tag}.png")
    (out / "INPUT_VERIFICATION.json").write_text(json.dumps(dict(inputs, x_axis="transmitted-frame number = position in "
        "transmitted_identities (frame order) + 1; markers at policy decisions only",
        backup_slides=f"copies of the {evidence} plots (unchanged)", pipeline_note=PIPELINE_NOTE), indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1, sort_keys=True))
    print("ok", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
