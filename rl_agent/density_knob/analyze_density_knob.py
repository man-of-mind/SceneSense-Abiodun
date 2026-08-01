#!/usr/bin/env python3
"""Density-adaptive knob selection -- aggregation, Pareto pick, density->best-knob lookup, plots.

Input : raw/perframe_<model>.csv (one row per profile x frame, from density_knob_eval.py)
        raw/frame_density.csv    (post-hoc density label per frame, from build_frame_density.py)
Output: raw/by_density_profile.csv, raw/best_knob_lookup.csv, plots/*.png, and the tables that
        DENSITY_KNOB_RESULTS.md is written from (printed + dumped as markdown fragments).

Conventions carried from the plan:
  * accuracy is measured on the IN-VIEW objects of that frame only (frustum, <=40 m, min area) --
    the density label and the accuracy denominator are the same object set;
  * bin 0 has no objects, so recall is degenerate there and the reported metric is
    FALSE POSITIVES PER FRAME + payload;
  * payload -> latency is IDEAL LOOPBACK, UPLINK-ONLY (OAI is a separate radio study);
  * ROI is the rank-based drop fraction q (drop the k=round(q*N) lowest-objectness cells), which is
    what the deployed front end does -- NOT a value threshold.
"""
from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

AB = Path("/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun")
HOME = AB / "rl_agent" / "density_knob"
RAW, PLOTS = HOME / "raw", HOME / "plots"
BINS = ["0", "1-2", "3-4", "5+"]
# Pareto acceptance tolerance vs the per-bin reference (the least-lossy measured profile).
TOL_RECALL = 0.02       # absolute recall points
TOL_LOC_M = 0.10        # metres of extra localisation error
TOL_FP_PER_FRAME = 0.05  # bin-0 metric: extra spurious detections per frame
TOL_MIOU = 0.02         # seg mIoU points below the model's OWN no-drop (roi0) seg (matches matrix 2%)


def seg_ref_by_model_bin(cells, frames, profs, bins):
    """The seg reference for a profile is that model's OWN no-ROI-drop (roi 0.0) mIoU IN THE SAME
    density bin -- i.e. how good seg is before any ROI drop, holding density fixed. This isolates the
    ROI-drop cost from density and reproduces the matrix's per-model-clean 2% accept semantics.
    Returns {(model, bin): ref_miou} using the best (max mIoU) roi0.0 profile of that model."""
    ref = {}
    for b in bins:
        for model in {p[0] for p in profs}:
            vals = []
            for prof in profs:
                if prof[0] != model or prof[2] != 0.0:
                    continue
                k = (prof, b)
                if k in cells:
                    mi = seg_metrics(cells[k])["miou"]
                    if mi == mi:
                        vals.append(mi)
            if vals:
                ref[(model, b)] = max(vals)
    return ref


def short(model: str, quant: str, roi: float) -> str:
    return f"{model}/{quant.replace('per_channel_uint', 'u')}/q{roi:g}"


def load():
    dens = {}
    for r in csv.DictReader((RAW / "frame_density.csv").open()):
        dens[r["sample_id"]] = r
    cells = collections.defaultdict(collections.Counter)   # (prof, bin) -> counters
    frames = collections.Counter()                          # (prof, bin) -> n frames
    for p in sorted(RAW.glob("perframe_*.csv")):
        for r in csv.DictReader(p.open()):
            d = dens.get(r["sample_id"])
            if d is None:
                continue
            prof = (r["model"], r["quant"], float(r["roi"]))
            for b in (d["density_bin"], "ALL"):
                k = (prof, b)
                c = cells[k]
                for f in ("payload_bytes", "tp", "fp", "fn", "tp_veh", "fp_veh", "fn_veh",
                          "tp_ped", "fp_ped", "fn_ped", "n_pred", "n_inview"):
                    c[f] += int(r[f])
                c["loc_err_sum"] += float(r["loc_err_sum"])
                c["loc_err_sq_sum"] += float(r["loc_err_sq_sum"])
                c["payload_sq"] += int(r["payload_bytes"]) ** 2
                for ci in range(3):  # seg confusion (row=GT class, col=pred): sum per (prof,bin)
                    for cj in range(3):
                        kk = f"conf_{ci}{cj}"
                        if kk in r and r[kk] != "":
                            c[kk] += int(r[kk])
                frames[k] += 1
    return cells, frames


def seg_metrics(c) -> dict:
    """mIoU + per-class IoU from the summed 3x3 confusion (background/vehicle/person), identical to
    the matrix's class_iou_from_confusion. Returns NaNs if the CSV had no seg columns."""
    if not any(c.get(f"conf_{i}{j}", 0) for i in range(3) for j in range(3)):
        return {"miou": float("nan"), "iou_bg": float("nan"),
                "veh_iou": float("nan"), "person_iou": float("nan")}
    ious = []
    for cls in range(3):
        tp = c[f"conf_{cls}{cls}"]
        fp = sum(c[f"conf_{r}{cls}"] for r in range(3)) - tp
        fn = sum(c[f"conf_{cls}{cj}"] for cj in range(3)) - tp
        d = tp + fp + fn
        ious.append(tp / d if d > 0 else float("nan"))
    valid = [v for v in ious if v == v]
    return {"miou": (sum(valid) / len(valid)) if valid else float("nan"),
            "iou_bg": ious[0], "veh_iou": ious[1], "person_iou": ious[2]}


def metrics(c, n):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    return {
        "frames": n,
        "gt_objs": c["n_inview"],
        "payload_kb": c["payload_bytes"] / max(1, n) / 1024.0,
        "recall": tp / max(1, tp + fn) if (tp + fn) else float("nan"),
        "precision": tp / max(1, tp + fp) if (tp + fp) else float("nan"),
        "recall_veh": c["tp_veh"] / max(1, c["tp_veh"] + c["fn_veh"]) if (c["tp_veh"] + c["fn_veh"]) else float("nan"),
        "recall_ped": c["tp_ped"] / max(1, c["tp_ped"] + c["fn_ped"]) if (c["tp_ped"] + c["fn_ped"]) else float("nan"),
        "loc_m": c["loc_err_sum"] / tp if tp else float("nan"),
        "fp_per_frame": fp / max(1, n),
        "pred_per_frame": c["n_pred"] / max(1, n),
        **seg_metrics(c),
    }


def transport_fit():
    """Uplink transport ms vs payload, calibrated on the MEASURED ideal-loopback points
    (loopback_latency_zstd.json). As of 2026-07-31 this includes the high-ROI q=0.7/0.9/0.98 profiles
    (48 measured total), so the whole ROI range is measured -- no extrapolation."""
    d = json.load((AB / "rl_agent" / "loopback_latency_zstd.json").open())
    x = np.array([v["payload_kb"] for v in d.values()])
    y = np.array([v["transport_ms"] for v in d.values()])
    b, a = np.polyfit(x, y, 1)
    r2 = 1.0 - float(np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2))
    return (lambda kb: a + b * kb), a, b, r2, len(x)


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    cells, frames = load()
    profs = sorted({k[0] for k in cells}, key=lambda p: (p[0], p[1], p[2]))
    tms, fit_a, fit_b, fit_r2, fit_n = transport_fit()
    print(f"transport fit: {fit_a:.3f} + {fit_b:.5f}*payload_kB ms   R2={fit_r2:.3f}  (n={fit_n} measured)")

    # ---------------- full table ----------------
    rows = []
    for prof in profs:
        for b in BINS + ["ALL"]:
            k = (prof, b)
            if k not in cells:
                continue
            m = metrics(cells[k], frames[k])
            rows.append({"model": prof[0], "quant": prof[1], "roi": prof[2], "profile": short(*prof),
                         "density_bin": b, **{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                              for kk, vv in m.items()},
                         "transport_ms_derived": round(tms(m["payload_kb"]), 3)})
    with (RAW / "by_density_profile.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RAW/'by_density_profile.csv'} ({len(rows)} rows)")

    # ---------------- PHYSICS CHECK: payload vs density, ACROSS profiles ----------------
    # Guardrail: never conclude from no-AE roi0 alone (payload there is ~flat by construction).
    print("\n" + "=" * 100)
    print("PAYLOAD vs DENSITY, ACROSS COMPRESSION PROFILES  (KB/frame; the tensor is fixed-size --")
    print("any variation here is content-adaptive compression only, never object count)")
    print("=" * 100)
    phys = []
    print(f"{'profile':<22}" + "".join(f"{('bin '+b):>11}" for b in BINS) + f"{'spread%':>9}")
    for prof in profs:
        vals = []
        for b in BINS:
            k = (prof, b)
            vals.append(metrics(cells[k], frames[k])["payload_kb"] if k in cells else float("nan"))
        if not all(v == v for v in vals):
            continue
        spread = 100.0 * (max(vals) - min(vals)) / max(vals)
        phys.append({"profile": short(*prof), "model": prof[0], "quant": prof[1], "roi": prof[2],
                     **{f"kb_bin{b}": round(v, 1) for b, v in zip(BINS, vals)},
                     "spread_pct": round(spread, 2)})
        if prof[0] in ("noae", "ae128") and prof[1] in ("per_channel_uint8", "per_channel_uint4"):
            print(f"{short(*prof):<22}" + "".join(f"{v:>11.1f}" for v in vals) + f"{spread:>9.1f}")
    with (RAW / "payload_vs_density.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(phys[0].keys()))
        w.writeheader()
        w.writerows(phys)
    by_roi = collections.defaultdict(list)
    for p in phys:
        by_roi[p["roi"]].append(p["spread_pct"])
    print("\n  payload spread across density bins, grouped by ROI drop fraction q "
          "(mean over all model x quant):")
    for q in sorted(by_roi):
        print(f"    q={q:<5g} mean spread = {sum(by_roi[q])/len(by_roi[q]):5.2f}%   "
              f"max = {max(by_roi[q]):5.2f}%")

    # ---------------- per-bin Pareto + best knob (JOINT detection + segmentation) ----------------
    # The accuracy deliverable of the shared map is BOTH object detections AND the dense semantic
    # segmentation (drivable surface, lane, vehicle/person pixel masks). ROI-drop keeps only high-
    # objectness cells, so it is nearly free for detection but destroys the dense seg between objects.
    # The accept rule is therefore JOINT: a profile is only affordable if it holds detection recall/loc
    # AND holds seg mIoU within TOL_MIOU of that model's own no-drop (roi0) seg in the same bin.
    # We report BOTH the detection-only pick (what the first run chose) and the seg-aware pick so the
    # cost of ignoring seg is explicit.
    print("\n" + "=" * 100)
    print("PARETO PICK PER DENSITY BIN  (detection-only vs seg-aware joint)")
    print("=" * 100)
    seg_ref = seg_ref_by_model_bin(cells, frames, profs, BINS)
    have_seg = any(v == v for v in (metrics(cells[k], frames[k])["miou"] for k in cells))
    lookup = []
    pareto_sets = {}

    def lookrow(b, nfr, p, m, kind):
        return {"density_bin": b, "n_frames": nfr, "policy": kind,
                "best_profile": short(*p), "model": p[0], "quant": p[1], "roi_q": p[2],
                "payload_kb": round(m["payload_kb"], 1),
                "transport_ms_derived": round(tms(m["payload_kb"]), 2),
                "recall": round(m["recall"], 4) if m["recall"] == m["recall"] else "",
                "recall_veh": round(m["recall_veh"], 4) if m["recall_veh"] == m["recall_veh"] else "",
                "recall_ped": round(m["recall_ped"], 4) if m["recall_ped"] == m["recall_ped"] else "",
                "loc_m": round(m["loc_m"], 3) if m["loc_m"] == m["loc_m"] else "",
                "miou": round(m["miou"], 4) if m["miou"] == m["miou"] else "",
                "veh_iou": round(m["veh_iou"], 4) if m["veh_iou"] == m["veh_iou"] else "",
                "fp_per_frame": round(m["fp_per_frame"], 3)}

    for b in BINS:
        cand = []
        for prof in profs:
            k = (prof, b)
            if k not in cells:
                continue
            cand.append((prof, metrics(cells[k], frames[k])))
        if not cand:
            continue
        # detection accept
        if b == "0":
            ref_fp = min(m["fp_per_frame"] for _, m in cand)
            det_ok = lambda m: m["fp_per_frame"] <= ref_fp + TOL_FP_PER_FRAME
            det_crit = f"FP/frame <= {ref_fp:.3f}+{TOL_FP_PER_FRAME} (recall degenerate: 0 in-view objects)"
        else:
            ref_recall = max(m["recall"] for _, m in cand)
            ref_loc = min(m["loc_m"] for _, m in cand)
            det_ok = lambda m: m["recall"] >= ref_recall - TOL_RECALL and m["loc_m"] <= ref_loc + TOL_LOC_M
            det_crit = f"recall >= {ref_recall:.3f}-{TOL_RECALL} and loc <= {ref_loc:.2f}+{TOL_LOC_M} m"
        # seg accept: mIoU within TOL_MIOU of that model's OWN roi0 seg in this bin
        def seg_ok(prof, m):
            if not have_seg or m["miou"] != m["miou"]:
                return True  # no seg data -> seg constraint inactive (detection-only fallback)
            ref = seg_ref.get((prof[0], b))
            return ref is None or m["miou"] >= ref - TOL_MIOU

        ok_det = [(p, m) for p, m in cand if det_ok(m)]
        ok_joint = [(p, m) for p, m in ok_det if seg_ok(p, m)]
        best_det = min(ok_det, key=lambda pm: pm[1]["payload_kb"]) if ok_det else None
        best_joint = min(ok_joint, key=lambda pm: pm[1]["payload_kb"]) if ok_joint else None
        # frontier over JOINT-accepted (for the plot)
        front = []
        for p, m in sorted(ok_joint, key=lambda pm: pm[1]["payload_kb"]):
            key = -m["fp_per_frame"] if b == "0" else m["recall"]
            if not front or key > front[-1][2]:
                front.append((p, m, key))
        pareto_sets[b] = {"cand": cand, "front": front, "best_joint": best_joint,
                          "best_det": best_det, "seg_ref": seg_ref.get((None, b)),
                          "ok_det": ok_det, "ok_joint": ok_joint, "det_crit": det_crit}
        n_frames = frames[(profs[0], b)]
        print(f"\nbin {b}  (n={n_frames} frames, {cells[(profs[0], b)]['n_inview']} in-view GT objects)")
        print(f"  detection accept: {det_crit}")
        print(f"  seg accept: mIoU >= (model's own roi0 mIoU) - {TOL_MIOU}")
        print(f"  {len(ok_det)}/{len(cand)} pass detection;  {len(ok_joint)}/{len(cand)} pass BOTH")
        if best_det:
            p, m = best_det
            print(f"  detection-only BEST = {short(*p):<20} {m['payload_kb']:7.1f} KB  "
                  f"recall {m['recall']:.3f}  loc {m['loc_m'] if m['loc_m']==m['loc_m'] else float('nan'):.2f}  "
                  f"mIoU {m['miou']:.3f}  vehIoU {m['veh_iou']:.3f}  <- seg may be wrecked")
            lookup.append(lookrow(b, n_frames, p, m, "detection_only"))
        if best_joint:
            p, m = best_joint
            print(f"  SEG-AWARE   BEST = {short(*p):<20} {m['payload_kb']:7.1f} KB  "
                  f"(uplink {tms(m['payload_kb']):.1f} ms)  recall "
                  f"{m['recall'] if m['recall']==m['recall'] else float('nan'):.3f}  "
                  f"loc {m['loc_m'] if m['loc_m']==m['loc_m'] else float('nan'):.2f}  "
                  f"mIoU {m['miou']:.3f}  vehIoU {m['veh_iou']:.3f}")
            lookup.append(lookrow(b, n_frames, p, m, "seg_aware"))
        # cheapest joint-accepted, for the doc
        for p, m in sorted(ok_joint, key=lambda pm: pm[1]["payload_kb"])[:5]:
            rc = f"{m['recall']:.3f}" if m["recall"] == m["recall"] else " n/a "
            print(f"      {short(*p):<20}{m['payload_kb']:8.1f} KB  recall {rc}  mIoU {m['miou']:.3f}")
    with (RAW / "best_knob_lookup.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lookup[0].keys()))
        w.writeheader()
        w.writerows(lookup)
    print(f"\nwrote {RAW/'best_knob_lookup.csv'}")

    # ---------------- accuracy-cost-of-aggression, per bin ----------------
    print("\n" + "=" * 100)
    print("ACCURACY COST OF THE SAME AGGRESSIVE KNOB, BY DENSITY  (this is where density bites)")
    print("=" * 100)
    for model in ("noae", "ae128"):
        for quant in ("per_channel_uint8",):
            print(f"\n  {model} / {quant.replace('per_channel_uint','u')}")
            hdr = f"    {'q':<7}" + "".join(f"{('bin '+b):>22}" for b in BINS)
            print(hdr)
            print(f"    {'':<7}" + "".join(f"{'recall  loc   FP/f':>22}" for _ in BINS))
            for roi in sorted({p[2] for p in profs}):
                prof = (model, quant, roi)
                line = f"    q={roi:<5g}"
                for b in BINS:
                    k = (prof, b)
                    if k not in cells:
                        line += f"{'-':>22}"
                        continue
                    m = metrics(cells[k], frames[k])
                    r = f"{m['recall']:.3f}" if m["recall"] == m["recall"] else "  -  "
                    lo = f"{m['loc_m']:.2f}" if m["loc_m"] == m["loc_m"] else " -  "
                    line += f"{r:>8}{lo:>7}{m['fp_per_frame']:>7.2f}"
                print(line)

    write_tables(cells, frames, profs, pareto_sets, tms, phys, lookup,
                 (fit_a, fit_b, fit_r2, fit_n))
    make_plots(cells, frames, profs, pareto_sets, tms, phys)
    json.dump({"transport_fit": {"intercept_ms": fit_a, "slope_ms_per_kb": fit_b, "r2": fit_r2,
                                 "n_measured": fit_n},
               "tolerances": {"recall_pts": TOL_RECALL, "loc_m": TOL_LOC_M,
                              "fp_per_frame": TOL_FP_PER_FRAME, "miou": TOL_MIOU},
               "seg_accept": "mIoU >= (model's own roi0 mIoU in same bin) - TOL_MIOU; "
                             "joint accept = detection AND seg"},
              (RAW / "analysis_settings.json").open("w"), indent=2)
    return 0


def write_tables(cells, frames, profs, pareto_sets, tms, phys, lookup, fit):
    """Emit every table as markdown so DENSITY_KNOB_RESULTS.md never transcribes a number by hand."""
    fit_a, fit_b, fit_r2, fit_n = fit
    out = [f"<!-- generated by analyze_density_knob.py -- do not hand-edit numbers -->\n"]

    # T1 bins + confounds
    fd = list(csv.DictReader((RAW / "frame_density.csv").open()))
    g = collections.defaultdict(list)
    for r in fd:
        g[r["density_bin"]].append(r)

    def mean(v, k):
        x = [float(r[k]) for r in v if r[k] != ""]
        return sum(x) / len(x) if x else float("nan")

    out.append("### T1 — density bins (post-hoc GT label on the continuous drive)\n")
    out.append("| bin (in-view objects) | frames | % of drive | in-view GT objects | veh | ped | "
               "mean GT dist m | nearest GT m | mean GT speed m/s | frac moving |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for b in BINS:
        v = g[b]
        nv = sum(int(r["n_inview_veh"]) for r in v)
        npd = sum(int(r["n_inview_ped"]) for r in v)
        d, dm, s, fm = (mean(v, "gt_dist_mean_m"), mean(v, "gt_dist_min_m"),
                        mean(v, "gt_speed_mean_mps"), mean(v, "frac_moving"))
        f = lambda x: "n/a" if x != x else f"{x:.2f}"
        out.append(f"| {b} | {len(v)} | {100*len(v)/len(fd):.1f}% | {nv+npd} | {nv} | {npd} | "
                   f"{f(d)} | {f(dm)} | {f(s)} | {f(fm)} |")

    # T2 payload vs density (all profiles)
    out.append("\n### T2 — payload (KB/frame) vs density, for every profile\n")
    out.append("| profile (model/quant/ROI-q) | bin 0 | bin 1-2 | bin 3-4 | bin 5+ | spread % |")
    out.append("|---|--:|--:|--:|--:|--:|")
    for p in sorted(phys, key=lambda p: -p["kb_bin0"]):
        out.append(f"| {p['profile']} | {p['kb_bin0']} | {p['kb_bin1-2']} | {p['kb_bin3-4']} | "
                   f"{p['kb_bin5+']} | {p['spread_pct']} |")

    # T2b spread grouped by q
    by_roi = collections.defaultdict(list)
    for p in phys:
        by_roi[p["roi"]].append(p["spread_pct"])
    out.append("\n### T2b — payload spread across density bins, by ROI drop fraction q\n")
    out.append("| ROI drop q | mean spread across bins | max | direction |")
    out.append("|--:|--:|--:|---|")
    for q in sorted(by_roi):
        sub = [p for p in phys if p["roi"] == q]
        denser_smaller = sum(1 for p in sub if p["kb_bin5+"] < p["kb_bin0"])
        out.append(f"| {q:g} | {sum(by_roi[q])/len(by_roi[q]):.2f}% | {max(by_roi[q]):.2f}% | "
                   f"denser = SMALLER payload in {denser_smaller}/{len(sub)} profiles |")

    # T3 per-bin Pareto (cheapest joint-accepted) — now with seg columns
    out.append("\n### T3 — cheapest accepted profiles per density bin (joint detection + seg)\n")
    for b in BINS:
        ps = pareto_sets[b]
        nfr = frames[(profs[0], b)]
        empty = b == "0"
        bd = ps["best_det"]
        bj = ps["best_joint"]
        out.append(f"\n**bin {b}** — n={nfr} frames, {cells[(profs[0], b)]['n_inview']} in-view GT objects"
                   + ("  (recall degenerate: no objects → detection metric is FP/frame)" if empty else ""))
        out.append("\n| profile | payload KB | uplink ms | in-view recall | loc MAE m | FP/frame | "
                   "**mIoU** | **veh IoU** | accepts |")
        out.append("|---|--:|--:|--:|--:|--:|--:|--:|:--|")
        f = lambda x: "n/a" if x != x else f"{x:.3f}"
        # show the cheapest few that pass detection, flag whether they ALSO pass seg
        for p, m in sorted(ps["ok_det"], key=lambda pm: pm[1]["payload_kb"])[:8]:
            passes_seg = any(p == pp for pp, _ in ps["ok_joint"])
            tag = []
            if bd and p == bd[0]:
                tag.append("**←det-only pick**")
            if bj and p == bj[0]:
                tag.append("**←seg-aware pick**")
            acc = "det+seg" if passes_seg else "**det only (seg fails)**"
            out.append(f"| {short(*p)} {' '.join(tag)} | {m['payload_kb']:.1f} | {tms(m['payload_kb']):.1f} | "
                       f"{f(m['recall'])} | {f(m['loc_m'])} | {m['fp_per_frame']:.2f} | "
                       f"{f(m['miou'])} | {f(m['veh_iou'])} | {acc} |")

    # T4 lookup — BOTH policies side by side (the cost of ignoring seg)
    out.append("\n### T4 — density → best-knob lookup: detection-only vs seg-aware (the deliverable)\n")
    out.append("| density | n frames | policy | best knob | payload KB | uplink ms | in-view recall | "
               "loc MAE m | mIoU | veh IoU | FP/frame |")
    out.append("|---|--:|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in lookup:
        out.append(f"| {r['density_bin']} | {r['n_frames']} | {r['policy']} | `{r['best_profile']}` | "
                   f"{r['payload_kb']} | {r['transport_ms_derived']} | {r['recall'] or 'n/a'} | "
                   f"{r['loc_m'] or 'n/a'} | {r['miou'] or 'n/a'} | {r['veh_iou'] or 'n/a'} | "
                   f"{r['fp_per_frame']} |")

    # T3b — SEG COLLAPSE: mIoU + veh IoU vs ROI drop q, per density bin (the headline seg evidence)
    out.append("\n### T3b — segmentation vs ROI drop q, per density bin (why ROI is not a free knob)\n")
    out.append("Averaged over all four AE variants at u4 (the detection-cheapest quant). "
               "mIoU / vehicle-IoU; ROI drop keeps only object cells, so dense seg between objects dies.\n")
    rois = sorted({p[2] for p in profs})
    out.append("| ROI drop q | " + " | ".join(f"bin {b} mIoU / vehIoU" for b in BINS) + " |")
    out.append("|--:|" + "--:|" * len(BINS))
    for roi in rois:
        cellsr = []
        for b in BINS:
            mis, vis = [], []
            for model in {p[0] for p in profs}:
                k = ((model, "per_channel_uint4", roi), b)
                if k in cells:
                    s = seg_metrics(cells[k])
                    if s["miou"] == s["miou"]:
                        mis.append(s["miou"]); vis.append(s["veh_iou"])
            cellsr.append(f"{sum(mis)/len(mis):.3f} / {sum(vis)/len(vis):.3f}" if mis else "n/a")
        out.append(f"| {roi:g} | " + " | ".join(cellsr) + " |")

    # T5 accuracy cost of aggression
    rois = sorted({p[2] for p in profs})
    for model in sorted({p[0] for p in profs}):
        out.append(f"\n### T5.{model} — in-view recall vs ROI drop q, per density bin ({model}, u8)\n")
        out.append("| ROI drop q | " + " | ".join(f"bin {b} recall" for b in BINS[1:])
                   + " | bin 0 FP/frame | payload KB (bin 1-2) |")
        out.append("|--:|" + "--:|" * (len(BINS) - 1) + "--:|--:|")
        for roi in rois:
            prof = (model, "per_channel_uint8", roi)
            cellsr = []
            for b in BINS[1:]:
                k = (prof, b)
                cellsr.append(f"{metrics(cells[k], frames[k])['recall']:.3f}" if k in cells else "-")
            k0, k12 = (prof, "0"), (prof, "1-2")
            fp0 = f"{metrics(cells[k0], frames[k0])['fp_per_frame']:.2f}" if k0 in cells else "-"
            kb = f"{metrics(cells[k12], frames[k12])['payload_kb']:.1f}" if k12 in cells else "-"
            out.append(f"| {roi:g} | " + " | ".join(cellsr) + f" | {fp0} | {kb} |")

    out.append(f"\n### Uplink-latency fit (fully measured)\n")
    out.append(f"`transport_ms = {fit_a:.3f} + {fit_b:.5f} x payload_KB`, least-squares fit on the "
               f"{fit_n} MEASURED ideal-loopback profiles in `loopback_latency_zstd.json` "
               f"(R²={fit_r2:.3f}). As of 2026-07-31 this includes the high-ROI q=0.7/0.9/0.98 profiles, "
               f"so the whole ROI range is measured (front ~25 ms flat, transport 1.3–4.1 ms, delivery "
               f"1.00) — no extrapolation. Ideal loopback, uplink-only.\n")
    (RAW / "tables.md").write_text("\n".join(out) + "\n")
    print(f"wrote {RAW/'tables.md'}")


def make_plots(cells, frames, profs, pareto_sets, tms, phys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, GRID = "#22303c", "#d5dde3"
    CB = {"0": "#4a7fb5", "1-2": "#5aa88f", "3-4": "#c98a3e", "5+": "#b05a6d"}

    def style(ax):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=9)

    # ---- 1. Pareto per bin (2x2) ----
    # The accept rule has TWO criteria (recall AND loc), so a single-axis frontier would be
    # misleading: a cheap high-recall profile can still be rejected on localisation error.
    # Accepted profiles are drawn filled, rejected ones grey, and the frontier is over ACCEPTED only.
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), constrained_layout=True)
    for ax, b in zip(axes.ravel(), BINS):
        style(ax)
        ps = pareto_sets[b]
        cand = ps["cand"]
        empty = (b == "0")
        det_ids = {id(m) for _, m in ps["ok_det"]}
        joint_ids = {id(m) for _, m in ps["ok_joint"]}
        crit = ("accept: FP/frame ok AND seg mIoU held" if empty
                else "accept: recall+loc ok AND seg mIoU held")
        yof = lambda m: (m["fp_per_frame"] if empty else m["recall"])
        rej = [m for _, m in cand if id(m) not in det_ids]                      # fails detection
        det_only = [m for _, m in cand if id(m) in det_ids and id(m) not in joint_ids]  # passes det, seg fails
        both = [m for _, m in cand if id(m) in joint_ids]                        # passes both
        ax.scatter([m["payload_kb"] for m in rej], [yof(m) for m in rej],
                   s=18, c="#c3ced6", alpha=0.9, lw=0, label="fails detection")
        ax.scatter([m["payload_kb"] for m in det_only], [yof(m) for m in det_only],
                   s=34, facecolor="none", edgecolor="#d98a2b", lw=1.3, label="detection ok, SEG fails")
        ax.scatter([m["payload_kb"] for m in both], [yof(m) for m in both],
                   s=30, c=CB[b], alpha=0.95, lw=0, label="detection + seg ok")
        fr = []
        for m in sorted(both, key=lambda m: m["payload_kb"]):
            key = -m["fp_per_frame"] if empty else m["recall"]
            if not fr or key > fr[-1][1]:
                fr.append((m, key))
        if fr:
            ax.plot([m["payload_kb"] for m, _ in fr], [yof(m) for m, _ in fr],
                    "-", color=CB[b], lw=1.6, alpha=0.85, label="frontier (seg-aware)")
        if ps["best_det"]:
            p, m = ps["best_det"]
            ax.scatter([m["payload_kb"]], [yof(m)], s=150, marker="X",
                       c="#d98a2b", lw=0, zorder=4, label="detection-only pick")
        if ps["best_joint"]:
            p, m = ps["best_joint"]
            ax.scatter([m["payload_kb"]], [yof(m)], s=170, facecolor="none",
                       edgecolor="#b0343c", lw=2.2, zorder=5, label="seg-aware pick")
            ax.annotate(f"{short(*p)}\n{m['payload_kb']:.1f} KB", (m["payload_kb"], yof(m)),
                        textcoords="offset points", xytext=(11, -14), fontsize=8.5, color="#b0343c")
        ax.set_xscale("log")
        nfr = frames[(profs[0], b)]
        ax.set_title(f"density bin {b}   (n={nfr} frames, "
                     f"{cells[(profs[0], b)]['n_inview']} in-view objects)\n{crit}",
                     fontsize=10, color=INK)
        ax.set_xlabel("uplink payload per frame (KB, log)", fontsize=9.5, color=INK)
        ax.set_ylabel("false positives / frame" if empty else "in-view recall", fontsize=9.5, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right" if not empty else "upper right")
    fig.suptitle("Which knob is affordable at each scene density — 72 profiles (quant × ROI-q × AE)\n"
                 "ideal loopback, uplink-only; accuracy on IN-VIEW GT (frustum, ≤40 m, GT = the column "
                 "the model was trained on)", fontsize=11.5, color=INK)
    fig.savefig(PLOTS / "pareto_per_density_bin.png", dpi=170, facecolor="white")
    plt.close(fig)

    # ---- 2. accuracy cost of aggression, by density ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    style(axes[0]); style(axes[1])
    rois = sorted({p[2] for p in profs})
    combos = sorted({(p[0], p[1]) for p in profs})
    # LEFT: the accuracy COST of raising q, relative to q=0, averaged over all 12 model x quant
    # combinations (the level of recall differs by density simply because dense scenes are harder --
    # the claim under test is about the SLOPE, so plot the delta).
    for b in BINS:
        if b == "0":
            continue
        ys = []
        for roi in rois:
            d = []
            for m, qu in combos:
                k0, kq = ((m, qu, 0.0), b), ((m, qu, roi), b)
                if k0 in cells and kq in cells:
                    d.append(100.0 * (metrics(cells[kq], frames[kq])["recall"]
                                      - metrics(cells[k0], frames[k0])["recall"]))
            ys.append(sum(d) / len(d) if d else float("nan"))
        axes[0].plot(rois, ys, "-o", color=CB[b], lw=1.9, ms=5, label=f"bin {b}")
    axes[0].axhline(0, color="#8b98a3", lw=0.9, ls=":")
    axes[0].set_xlabel("ROI drop fraction q (fraction of feature cells zeroed)", fontsize=9.5, color=INK)
    axes[0].set_ylabel("in-view recall cost vs q=0 (points)", fontsize=9.5, color=INK)
    axes[0].set_title("The SAME aggressive knob costs ~2× more recall in dense scenes\n"
                      "(mean over all 12 model×quant combinations)", fontsize=10.5, color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    # RIGHT: payload is set by q, not by density
    for b in BINS:
        ys = []
        for roi in rois:
            k = (("noae", "per_channel_uint8", roi), b)
            ys.append(metrics(cells[k], frames[k])["payload_kb"] if k in cells else float("nan"))
        axes[1].plot(rois, ys, "-o", color=CB[b], lw=1.9, ms=5, label=f"bin {b}")
    axes[1].set_xlabel("ROI drop fraction q", fontsize=9.5, color=INK)
    axes[1].set_ylabel("payload per frame (KB)", fontsize=9.5, color=INK)
    axes[1].set_title("Payload is set by q, NOT by density\n"
                      "(rank-based drop ⇒ the four curves overlap; no-AE u8)",
                      fontsize=10.5, color=INK)
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(PLOTS / "density_cost_of_roi_drop.png", dpi=170, facecolor="white")
    plt.close(fig)

    # ---- 3. payload spread across density, by q ----
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    style(ax)
    by_roi = collections.defaultdict(list)
    for p in phys:
        by_roi[p["roi"]].append(p["spread_pct"])
    qs = sorted(by_roi)
    ax.bar([str(q) for q in qs], [sum(by_roi[q]) / len(by_roi[q]) for q in qs],
           color="#4a7fb5", width=0.62)
    for i, q in enumerate(qs):
        ax.text(i, sum(by_roi[q]) / len(by_roi[q]) + 0.15,
                f"{sum(by_roi[q])/len(by_roi[q]):.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xlabel("ROI drop fraction q", fontsize=9.5, color=INK)
    ax.set_ylabel("payload spread across density bins (%)", fontsize=9.5, color=INK)
    ax.set_title("How much does scene density move the payload?\n"
                 "mean over all 12 model×quant combinations — the tensor is fixed-size,\n"
                 "so this is purely content-adaptive compression", fontsize=10.5, color=INK)
    fig.savefig(PLOTS / "payload_spread_by_density.png", dpi=170, facecolor="white")
    plt.close(fig)

    # ---- 4. SEG COLLAPSE: mIoU + vehicle IoU vs ROI drop q (why ROI is not a free knob) ----
    # This is the plot that reconciles the density run with the seg-aware knob matrix: detection
    # recall barely moves with q, but seg mIoU / vehicle IoU fall off a cliff as soon as q>0.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    style(axes[0]); style(axes[1])
    rois = sorted({p[2] for p in profs})
    for ax, (metric, title) in zip(axes, [("miou", "Segmentation mIoU"),
                                          ("recall", "Object-detection recall")]):
        for b in BINS:
            ys = []
            for roi in rois:
                vals = []
                for model in {p[0] for p in profs}:
                    k = ((model, "per_channel_uint4", roi), b)
                    if k in cells:
                        v = metrics(cells[k], frames[k])[metric]
                        if v == v:
                            vals.append(v)
                ys.append(sum(vals) / len(vals) if vals else float("nan"))
            ax.plot(rois, ys, "-o", color=CB[b], lw=1.9, ms=5, label=f"bin {b}")
        ax.set_xlabel("ROI drop fraction q", fontsize=9.5, color=INK)
        ax.set_ylabel(title + (" (in-view)" if metric == "recall" else ""), fontsize=9.5, color=INK)
        ax.set_title(title + " vs ROI drop", fontsize=10.5, color=INK)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle("ROI drop is nearly free for detection but destroys segmentation "
                 "(u4, mean over AE variants)\nthis is why the seg-aware policy cannot use the ROI knob",
                 fontsize=11, color=INK)
    fig.savefig(PLOTS / "seg_collapse_vs_roi.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(f"\nwrote plots -> {PLOTS}")


if __name__ == "__main__":
    raise SystemExit(main())
