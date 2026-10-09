"""Tables, paired per-frame CSV and presentation plots from REPLAY.json (no inference)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BR = ("full_local", "fixed_uint8", "historical_selected")
LABEL = {"full_local": "Full-local (no AE / no quant / no drop)",
         "fixed_uint8": "Fixed baseline: noAE UINT8 q=0",
         "historical_selected": "Historical v11 policy actions (AE64/UINT4 x29, AE128/UINT4 x15, AE32/UINT6 x3, AE64/UINT6 x2) - NOT the full Run-5 actor"}
SHORT = {"full_local": "Full-local", "fixed_uint8": "UINT8 q=0", "historical_selected": "Historical action"}
COLOR = {"full_local": "#2a78d6", "fixed_uint8": "#eb6834", "historical_selected": "#1baf7a"}
STYLE = {"full_local": "-", "fixed_uint8": "--", "historical_selected": ":"}
TOL = {"vehicle": 1.0, "person": 1.2}
WEIGHT = {"person": 0.6, "vehicle": 0.4}


def within(c, cls):
    return sum(e <= TOL[cls] for e in c["matched_xy_errors_m"])


def pooled(rows, b, cls):
    cs = [r[b]["classes"][cls] for r in rows]
    t = {k: sum(c[k] for c in cs) for k in ("eligible_gt", "tp", "fp", "fn")}
    errs = [e for c in cs for e in c["matched_xy_errors_m"]]
    w = sum(e <= TOL[cls] for e in errs)
    seg = {k: sum(c["segmentation_surrogate"][k] for c in cs) for k in ("intersection_pixels", "union_pixels")}
    return dict(t, within_tol=w, recall_within_tol=w / t["eligible_gt"] if t["eligible_gt"] else None,
                recall_3m_gate=t["tp"] / t["eligible_gt"] if t["eligible_gt"] else None,
                precision=t["tp"] / (t["tp"] + t["fp"]) if t["tp"] + t["fp"] else None,
                matched_err_median_m=float(np.median(errs)) if errs else None, errors=errs,
                seg_iou_pooled=seg["intersection_pixels"] / seg["union_pixels"] if seg["union_pixels"] else None)


def boot_diff(rows, a, b, cls, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    el = np.array([r[a]["classes"][cls]["eligible_gt"] for r in rows], float)
    wa = np.array([within(r[a]["classes"][cls], cls) for r in rows], float)
    wb = np.array([within(r[b]["classes"][cls], cls) for r in rows], float)
    idx = rng.integers(0, len(rows), (n, len(rows)))
    d = (wb[idx].sum(1) - wa[idx].sum(1)) / np.maximum(el[idx].sum(1), 1)
    return float((wb.sum() - wa.sum()) / el.sum()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main(replay_dir: Path, out: Path):
    rep = json.loads((replay_dir / "REPLAY.json").read_text())
    rows = [r for r in rep["rows"] if r["status"] == "PAIRED"]
    assert len(rows) == len(rep["rows"])
    assert all(r["parity"]["selected_inner_payload_byte_identical"] and r["parity"]["selected_native_labels_identical"]
               and r["parity"]["selected_complete_records_identical"] for r in rows)
    orig_eq = sum(r["selected_matches_original_evaluator"] is True for r in rows)
    plt.rcParams.update({"font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                         "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    summary = dict(frames=len(rows), selected_parity_frames=len(rows), selected_equals_original_evaluator=orig_eq,
                   timely_frames=sum(r["timely_feedback"] == "True" for r in rows),
                   historical_actions=sorted({(r["historical_mode_id"], r["historical_q_e4"]) for r in rows}),
                   branches={})
    for subset_name, subset in (("all", rows), ("timely", [r for r in rows if r["timely_feedback"] == "True"]),
                                ("late", [r for r in rows if r["timely_feedback"] != "True"])):
        for b in BR:
            q = [r[b]["quality"] for r in subset if r[b]["quality"]["defined"]]
            v = [r[b]["run5_agreed_value_if_timely"] for r in subset if r[b]["run5_agreed_value_if_timely"] is not None]
            summary["branches"].setdefault(subset_name, {})[b] = dict(
                frames=len(subset), q_defined_frames=len(q), q_undefined_frames=len(subset) - len(q),
                q_perc_mean=float(np.mean([x["q_perc"] for x in q])) if q else None,
                q_seg_mean=float(np.mean([x["q_seg"] for x in q if x["q_seg"] is not None])) if q else None,
                run5_value_mean_if_timely=float(np.mean(v)) if v else None,
                classes={c: {k: x for k, x in pooled(subset, b, c).items() if k != "errors"} for c in TOL})
    summary["paired_bootstrap_within_tol_recall_diff"] = {
        f"{b}_minus_full_local": {c: boot_diff(rows, "full_local", b, c) for c in TOL} for b in BR[1:]}
    summary["paired_bootstrap_within_tol_recall_diff"]["historical_minus_uint8"] = {
        c: boot_diff(rows, "fixed_uint8", "historical_selected", c) for c in TOL}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    # table
    lines = ["| Branch | Class | Eligible GT | Matched (3 m gate) | Misses | FP | Precision | Recall @3 m gate | "
             "**Recall within tol (misses incl.)** | Matched err P50 (m) | Seg IoU (pooled) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    A = summary["branches"]["all"]
    for c in ("vehicle", "person"):
        for b in BR:
            x = A[b]["classes"][c]
            lines.append(f"| {SHORT[b]} | {c} (≤{TOL[c]} m) | {x['eligible_gt']} | {x['tp']} | {x['fn']} | {x['fp']} | "
                         f"{x['precision']:.3f} | {x['recall_3m_gate']:.3f} | **{x['within_tol']}/{x['eligible_gt']} = "
                         f"{x['recall_within_tol']:.3f}** | {x['matched_err_median_m']:.2f} | {x['seg_iou_pooled']:.3f} |")
    lines += ["", "| Branch | Q defined / undefined frames | Mean Q_perc | Mean Q_seg | Mean Run-5 value Y(0.7+0.3Q_seg) if timely |",
              "|---|---|---|---|---|"]
    for b in BR:
        x = A[b]
        lines.append(f"| {SHORT[b]} | {x['q_defined_frames']} / {x['q_undefined_frames']} | {x['q_perc_mean']:.3f} | "
                     f"{x['q_seg_mean']:.3f} | {x['run5_value_mean_if_timely']:.3f} |")
    (out / "QUALITY_TABLE.md").write_text("\n".join(lines) + "\n")
    # per-frame CSV
    with (out / "PAIRED_FRAMES.csv").open("x", newline="") as s:
        cols = ["index", "frame_id", "timely_feedback", "historical_mode_id", "historical_q_e4",
                "historical_payload_bytes", "uint8_payload_bytes"]
        for b in BR:
            for c in TOL:
                cols += [f"{b}_{c}_{k}" for k in ("eligible", "tp", "fn", "fp", "within_tol", "seg_iou")]
            cols += [f"{b}_q_perc", f"{b}_q_seg", f"{b}_run5_value"]
        w = csv.DictWriter(s, fieldnames=cols)
        w.writeheader()
        for r in rows:
            d = dict(index=r["index"], frame_id=r["frame_id"], timely_feedback=r["timely_feedback"],
                     historical_mode_id=r["historical_mode_id"], historical_q_e4=r["historical_q_e4"],
                     historical_payload_bytes=r["historical_payload_bytes"], uint8_payload_bytes=r["baseline"]["payload_bytes"])
            for b in BR:
                for c in TOL:
                    k = r[b]["classes"][c]
                    d.update({f"{b}_{c}_eligible": k["eligible_gt"], f"{b}_{c}_tp": k["tp"], f"{b}_{c}_fn": k["fn"],
                              f"{b}_{c}_fp": k["fp"], f"{b}_{c}_within_tol": within(k, c),
                              f"{b}_{c}_seg_iou": k["segmentation_surrogate"]["iou"]})
                d.update({f"{b}_q_perc": r[b]["quality"]["q_perc"], f"{b}_q_seg": r[b]["quality"]["q_seg"],
                          f"{b}_run5_value": r[b]["run5_agreed_value_if_timely"]})
            w.writerow(d)
    # Plot 1: matched-object localisation CDFs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, c in zip(axes, ("vehicle", "person")):
        for b in BR:
            x = pooled(rows, b, c)
            e = np.sort(x["errors"])
            ax.step(np.r_[0, e], np.r_[0, np.arange(1, len(e) + 1) / len(e)], where="post", color=COLOR[b],
                    ls=STYLE[b], lw=2.2, label=f"{SHORT[b]}: {x['tp']} matched, {x['fn']} missed of {x['eligible_gt']}")
        ax.axvline(TOL[c], color="#555555", lw=1.5)
        ax.text(TOL[c] + 0.04, 0.55, f"{TOL[c]} m\ntolerance", color="#333333", fontweight="bold")
        ax.set_xlim(0, 3); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
        ax.set_xlabel("Centroid XY localisation error (m)"); ax.set_ylabel("Fraction of MATCHED objects")
        ax.set_title(f"{c.capitalize()}: MATCHED objects only")
        ax.legend(loc="lower right", fontsize=10.5, frameon=False)
    fig.suptitle(f"v11 treatment capture, {len(rows)} paired frames, identical inputs/weights/GT "
                 "\nCDF over matched objects only - missed detections are NOT in this curve (see recall figure)", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(out / "fig1_localization_cdf_matched_objects.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    # Plot 2: misses-inclusive within-tolerance recall
    fig, ax = plt.subplots(figsize=(12, 6.2))
    xs = np.arange(2); width = 0.26
    for i, b in enumerate(BR):
        vals = [pooled(rows, b, c) for c in ("vehicle", "person")]
        bars = ax.bar(xs + (i - 1) * width, [v["recall_within_tol"] for v in vals], width - 0.02, color=COLOR[b],
                      label=LABEL[b], hatch=["", "//", ".."][i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v['recall_within_tol']:.2f}\n({v['within_tol']}/{v['eligible_gt']})",
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#222222")
    ax.axhspan(0.80, 0.85, color="#999999", alpha=0.18)
    ax.text(1.45, 0.825, "80–85% target", va="center", ha="right", fontweight="bold", color="#444444")
    ax.set_xticks(xs); ax.set_xticklabels(["Vehicle (≤ 1.0 m)", "Person (≤ 1.2 m)"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("Recall within tolerance\n(÷ ALL eligible GT, misses included)")
    ax.set_title(f"Misses-inclusive localisation recall, {len(rows)} paired frames (v11 capture, historical policy)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "fig2_within_tolerance_recall.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.is_file() and p.name != "OUTPUT_SHA256.json"} | {
        "replay/REPLAY.json": hashlib.sha256((replay_dir / "REPLAY.json").read_bytes()).hexdigest()}, indent=2))
    print(open(out / "QUALITY_TABLE.md").read())
    print(json.dumps({k: v for k, v in summary.items() if k != "branches"}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
