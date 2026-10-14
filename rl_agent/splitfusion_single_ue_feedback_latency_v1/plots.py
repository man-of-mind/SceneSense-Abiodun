"""Two presentation figures (PNG, light surface) from the joined per-decision rows.

01_stage_contributions.png  top: mean stage contribution per outcome group (means are additive -> bar length = mean
                            UE receipt); bottom: mean excess of each stage over the on-time group (= which interval
                            expands). Columns = captures.
02_representative_timeline.png  aligned per-decision timelines (t = 0 at UE action-open, 170-ms deadline line).
"""
import collections

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# reference categorical palette, fixed order (validated: scripts/validate_palette.js, light)
CATEGORIES = (
    ("UE preparation", "#2a78d6", ("st_ue_actor", "st_ue_pre_input", "st_ue_input", "st_ue_encode_send")),
    ("Uplink", "#eb6834", ("st_uplink",)),
    ("Edge queue + compute", "#1baf7a", ("st_edge",)),
    ("Edge <-> GT-service handoffs", "#eda100", ("st_pred_to_service", "st_result_return", "st_feedback_prep")),
    ("Extra wait for GT", "#e87ba4", ("st_gt_extra_wait",)),
    ("Scoring (channel wait + op)", "#008300", ("st_score_wait", "st_score_exec")),
    ("Edge send -> UE tunnel (core, gNB DL, air, UE stack)", "#6250d6", ("st_send_to_tun",)),
    ("UE tunnel -> app receive (kernel + Python thread)", "#e34948", ("st_tun_to_app",)),
)
GROUP_LABEL = {"ON_TIME": "on time (<=170 ms)", "LATE_170_200": "late (170-200 ms]", "EXTREME_GT200": "extreme (>200 ms)"}
DEADLINE = 170.0


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def cat_value(r, cols):
    return sum(r[c] for c in cols)


def stage_contributions(rows, report, path):
    caps = list(report["captures"])
    fig, axes = plt.subplots(2, len(caps), figsize=(16, 9.2), facecolor=SURFACE,
                             gridspec_kw=dict(height_ratios=[1, 1.45], hspace=0.38, wspace=0.08, top=0.86))
    groups = ("ON_TIME", "LATE_170_200", "EXTREME_GT200")
    global XMAX_EXCESS
    XMAX_EXCESS = 15 + max(np.mean([cat_value(r, cols) for r in rows if r["capture"] == c and r["group"] == g])
                           - np.mean([cat_value(r, cols) for r in rows if r["capture"] == c and r["group"] == "ON_TIME"])
                           for c in caps for g in groups[1:] for _, _, cols in CATEGORIES)
    for j, cap in enumerate(caps):
        sub = [r for r in rows if r["capture"] == cap]
        by = {g: [r for r in sub if r["group"] == g] for g in groups}
        means = {g: [np.mean([cat_value(r, cols) for r in by[g]]) for _, _, cols in CATEGORIES] for g in groups}
        ax = axes[0, j]
        _style(ax)
        for i, g in enumerate(groups):
            left = 0.0
            for k, (name, color, _) in enumerate(CATEGORIES):
                w = means[g][k]
                ax.barh(i, w, left=left, height=0.6, color=color, edgecolor=SURFACE, lw=1.5)
                left += w
            ax.text(left + 2, i, f"{left:.0f} ms (n={len(by[g])})", va="center", fontsize=9, color=INK)
        ax.axvline(DEADLINE, color=INK2, lw=1.2, ls="--")
        ax.text(DEADLINE + 2, 2.42, "170-ms deadline", color=INK2, fontsize=8.5, ha="left", va="top")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([GROUP_LABEL[g] for g in groups] if j == 0 else [], fontsize=9.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 305)
        ax.set_xlabel("mean ms after UE action-open (stage means add up to the mean UE receipt)", fontsize=9, color=INK2)
        ax.set_title(f"{cap}  ({len(sub)} decisions, {len(by['ON_TIME'])} on time)", fontsize=11.5, color=INK, loc="left")
        # bottom: excess over on-time
        ax = axes[1, j]
        _style(ax)
        y = np.arange(len(CATEGORIES))
        for off, g, alpha, hatch in ((-0.19, "LATE_170_200", 0.45, None), (0.19, "EXTREME_GT200", 1.0, None)):
            ex = [means[g][k] - means["ON_TIME"][k] for k in range(len(CATEGORIES))]
            ax.barh(y + off, ex, height=0.36, color=[c for _, c, _ in CATEGORIES], alpha=alpha, edgecolor="none")
            for k, v in enumerate(ex):
                ax.text(v + (1 if v >= 0 else -1), y[k] + off, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                        fontsize=8.5, color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _, _ in CATEGORIES] if j == 0 else [], fontsize=9.5, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(-8, XMAX_EXCESS)
        ax.set_xlabel("mean excess over the on-time group, ms   (pale bar = late 170-200;  solid bar = extreme >200)",
                      fontsize=9, color=INK2)
        ax.set_title("Which interval expands", fontsize=10.5, color=INK, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in CATEGORIES]
    fig.legend(handles, [n for n, _, _ in CATEGORIES], loc="upper center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, 0.955), labelcolor=INK)
    fig.suptitle("Single-UE policy feedback: where the time goes on late and extreme decisions "
                 "(paired per-decision intervals; GOOD_ALL 23.5 dB)", y=0.995, fontsize=12.5, color=INK, x=0.5)
    fig.text(0.01, 0.005, "'Edge send -> UE tunnel' = pre-send edge stamp -> kernel arrival on oaitun_ue1 (core + gNB DL queue/"
             "scheduling + RFsim air + UE stack); it is NOT pure over-the-air latency. "
             "V20_DEMO_03 was watched off-host (viewer server on this host, CPUs 22-23).", fontsize=8, color=INK2)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def pick_representatives(rows):
    def nearest(sub, target):
        return min(sub, key=lambda r: abs(r["ue_receipt"] - target)) if sub else None
    picks = []
    for cap in dict.fromkeys(r["capture"] for r in rows):
        sub = [r for r in rows if r["capture"] == cap]
        on = [r for r in sub if r["group"] == "ON_TIME"]
        picks.append((cap, "on time, median receipt", nearest(on, float(np.median([r["ue_receipt"] for r in on])))))
        diffuse = [r for r in sub if r["spike_class"] == "DIFFUSE_ACCUMULATION"]
        if diffuse:
            picks.append((cap, "late, diffuse (median of class)",
                          nearest(diffuse, float(np.median([r["ue_receipt"] for r in diffuse])))))
        seen = set()
        for cls, label in (("st_ue_pre_input", "UE wait before input"), ("st_ue_input", "UE input-build stall"),
                           ("st_gt_extra_wait", "GT publication late"), ("st_score_exec", "score-worker stall"),
                           ("st_send_to_tun", "downlink-path outlier"), ("st_tun_to_app", "UE receive-thread stall")):
            cand = [r for r in sub if r["spike_class"] == f"SINGLE_STAGE:{cls}"]
            if cand and label not in seen:
                r = max(cand, key=lambda r: r["dominant_excess_ms"])
                picks.append((cap, label + (" + UE GT-freeze stall" if r["ue_freeze_stall"] else ""), r))
                seen.add(label)
    return picks


def timeline(rows, path):
    picks = pick_representatives(rows)
    fig, ax = plt.subplots(figsize=(16, 0.62 * len(picks) + 2.2), facecolor=SURFACE)
    _style(ax)
    for i, (cap, label, r) in enumerate(picks):
        left = 0.0
        for name, color, cols in CATEGORIES:
            for c in cols:
                w = r[c]
                if w > 0:
                    ax.barh(i, w, left=left, height=0.42, color=color, edgecolor=SURFACE, lw=1.0)
                left += w
        # GT branch (concurrent): service freeze received -> GT published
        if r["service_freeze_received"] is not None and r["gt_published"] is not None:
            ax.barh(i + 0.33, r["gt_published"] - r["service_freeze_received"], left=r["service_freeze_received"],
                    height=0.13, color="#8a8984", edgecolor="none")
        # UE GT freeze (concurrent, UE side)
        if r["ue_freeze_start"] is not None and r["ue_freeze_frozen"] is not None:
            ax.barh(i - 0.32, r["ue_freeze_frozen"] - r["ue_freeze_start"], left=r["ue_freeze_start"], height=0.1,
                    color="#2a78d6", alpha=0.45, edgecolor="none")
        ax.plot([r["edge_compute_end"]] * 2, [i - 0.24, i + 0.24], color=INK, lw=1.4)
        ax.text(r["ue_receipt"] + 3, i, f"{r['ue_receipt']:.0f} ms", va="center", fontsize=9, color=INK)
        ax.text(-6, i, f"{cap}  d{r['decision_index']}\n{label}", va="center", ha="right", fontsize=8.8, color=INK)
    ax.axvline(DEADLINE, color=INK2, lw=1.2, ls="--")
    ax.text(DEADLINE + 1, -0.85, "170-ms deadline", color=INK2, fontsize=8.5)
    ax.set_yticks([])
    ax.set_ylim(len(picks) - 0.4, -1.1)
    ax.set_xlim(0, 420)
    ax.set_xlabel("ms after UE action-open (CLOCK_MONOTONIC_RAW)", fontsize=9.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in CATEGORIES]
    handles += [plt.Rectangle((0, 0), 1, 1, color="#8a8984"), plt.Rectangle((0, 0), 1, 1, color="#2a78d6", alpha=0.45),
                plt.Line2D([0], [0], color=INK, lw=1.4)]
    names = [n for n, _, _ in CATEGORIES] + ["GT branch: freeze received -> GT published (concurrent)",
                                             "UE GT freeze start -> frozen (concurrent)", "prediction completed (edge)"]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4,
              frameon=False, fontsize=8.8, labelcolor=INK)
    ax.set_title("Representative decisions on one clock: critical path to UE receipt (thick) with concurrent GT work (thin)",
                 fontsize=12.5, color=INK, loc="left", pad=14)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return [dict(capture=c, label=l, decision_index=r["decision_index"], frame_id=r["frame_id"],
                 ue_receipt=round(r["ue_receipt"], 2)) for c, l, r in picks]


def render(rows, report, out):
    stage_contributions(rows, report, out / "01_stage_contributions.png")
    report["representatives"] = timeline(rows, out / "02_representative_timeline.png")
    import json
    (out / "SUMMARY.json").write_text(json.dumps(report, indent=1, default=str))
