#!/usr/bin/env python3
"""Aggregate all M' sweep evals into COMPLETE_KNOB_MATRIX.md.

Scans a parent dir for */metrics/test_fusion_evaluation_metrics.json and tabulates the
{quant, entropy, ROI, AE} action profiles vs {accuracy, payload-bytes}. Marks the clean
(no-compression) baseline and flags a simple Pareto front (min payload among configs whose
person-recall and mIoU stay within a tolerance of clean). Latency/reliability-under-channel
are intentionally absent here -- those are the OAI/network phase.

Usage: build_knob_matrix.py <parent_dir> <out.md>
"""
import json, sys
from pathlib import Path

def load(parent):
    rows = []
    for mj in sorted(Path(parent).glob("*/metrics/test_fusion_evaluation_metrics.json")):
        try:
            m = json.load(open(mj))
        except Exception:
            continue
        rows.append({
            "name": mj.parent.parent.name,
            "quant": m.get("quantization_mode", "") or "none",
            "entropy": m.get("entropy_coder", "") or "-",
            "roi": float(m.get("roi_threshold", 0.0) or 0.0),
            "ae": int(m.get("ae_bottleneck", 0) or 0),
            "payload_kb": (m.get("payload_bytes_mean", float("nan")) or float("nan")) / 1024.0,
            "miou": m.get("miou", float("nan")),
            "veh_iou": m.get("vehicle_iou", float("nan")),
            "ped_recall": m.get("learned_person_object_recall", float("nan")),
            "obj_recall": m.get("learned_object_recall", float("nan")),
            "loc_m": m.get("learned_global_xy_mae_m", float("nan")),
            "ped_loc_m": m.get("learned_person_global_xy_mae_m", float("nan")),
        })
    return rows

def fnum(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return "nan"

def main():
    parent, out = sys.argv[1], sys.argv[2]
    lb = {}
    if len(sys.argv) > 3 and Path(sys.argv[3]).exists():
        try:
            lb = json.load(open(sys.argv[3]))  # {"quant|entropy": {front_ms, rtt_ms, delivery_rate, ...}}
        except Exception:
            lb = {}
    rows = load(parent)
    if not rows:
        Path(out).write_text("# COMPLETE KNOB MATRIX\n\n(no eval metrics found)\n"); return
    # clean baseline = the config with no compression at all (or largest payload / no quant)
    clean = None
    for r in rows:
        if r["quant"] in ("none", "") and r["ae"] == 0 and r["roi"] == 0.0:
            clean = r; break
    if clean is None:  # fall back to the largest-payload row
        clean = max(rows, key=lambda r: (r["payload_kb"] if r["payload_kb"] == r["payload_kb"] else -1))
    base_miou, base_ped = clean["miou"], clean["ped_recall"]
    # payload baseline = the largest MEASURED payload (least-compressed config), a robust denominator
    # even if the accuracy-clean row has no payload (non-split path).
    finite_pay = [r["payload_kb"] for r in rows if r["payload_kb"] == r["payload_kb"]]
    base_pay = max(finite_pay) if finite_pay else float("nan")
    TOL = 0.02  # accuracy tolerance for "acceptable" profiles
    # payload -> {front_ms, delivery} curve from the measured loopback profiles, for interpolating the
    # ROI/AE rows (which the loopback client can't run natively). Sorted by payload.
    curve = sorted(({"p": v["payload_kb"], "front": v["front_ms"], "deliv": v["delivery_rate"]}
                    for v in lb.values() if v.get("payload_kb") is not None), key=lambda x: x["p"])
    def interp(p, field):
        if not curve or p is None or p != p:
            return None
        if p <= curve[0]["p"]:
            return curve[0][field]
        if p >= curve[-1]["p"]:
            return curve[-1][field]
        for a, b in zip(curve, curve[1:]):
            if a["p"] <= p <= b["p"]:
                va, vb = a[field], b[field]
                if va is None or vb is None:
                    return va if vb is None else vb
                t = (p - a["p"]) / (b["p"] - a["p"]) if b["p"] > a["p"] else 0.0
                return va + t * (vb - va)
        return None
    for r in rows:
        r["accept"] = (r["miou"] >= base_miou - TOL) and (r["ped_recall"] >= base_ped - TOL)
        r["payload_frac"] = (r["payload_kb"] / base_pay) if base_pay else float("nan")
        # measured loopback latency/reliability (direct match for pure quant x entropy configs);
        # ROI/AE rows get INTERPOLATED estimates from the payload curve (rendered with a leading ~).
        k = f"{r['quant']}|{r['entropy']}"
        m = lb.get(k) if (r["roi"] == 0.0 and r["ae"] == 0) else None
        if m:
            r["front_ms"], r["rtt_ms"], r["delivery"], r["est"] = m["front_ms"], m["rtt_ms"], m["delivery_rate"], False
        else:
            r["front_ms"], r["rtt_ms"], r["delivery"], r["est"] = interp(r["payload_kb"], "front"), None, interp(r["payload_kb"], "deliv"), True
    # Pareto: acceptable configs with a real (finite) payload, minimal payload
    accept = [r for r in rows if r["accept"]]
    accept_pay = [r for r in accept if r["payload_kb"] == r["payload_kb"]]  # drop nan-payload (clean) row
    best = min(accept_pay, key=lambda r: r["payload_kb"]) if accept_pay else None

    rows.sort(key=lambda r: (r["payload_kb"] if r["payload_kb"] == r["payload_kb"] else 1e18))
    L = ["# COMPLETE KNOB MATRIX (M', offline: accuracy + payload)",
         "",
         "Action profiles vs task accuracy and on-wire payload (entropy-coded bytes). "
         "Latency / delivery-rate / reliability under channel = OAI/network phase (not here).",
         "",
         f"Clean baseline: **{clean['name']}** payload={fnum(base_pay,1)}KB mIoU={fnum(base_miou)} "
         f"ped-recall={fnum(base_ped)}  (accept tol = {TOL:.0%})",
         "",
         "| profile | quant | entropy | ROI q | AE | payload KB | payload % | mIoU | veh IoU | ped recall | obj recall | loc m | ped-loc m | front ms | RTT ms | delivery | accept |",
         "|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|"]
    def _o(x, est, d=1):
        if x is None:
            return "-"
        return ("~" if est else "") + fnum(x, d)   # ~ prefix = interpolated from the payload curve
    for r in rows:
        L.append("| {name} | {q} | {e} | {roi} | {ae} | {pay} | {pf} | {mi} | {vi} | {pr} | {orr} | {loc} | {ploc} | {fm} | {rt} | {dl} | {acc} |".format(
            name=r["name"], q=r["quant"], e=r["entropy"], roi=fnum(r["roi"],2), ae=(r["ae"] or "-"),
            pay=fnum(r["payload_kb"],1), pf=fnum(100*r["payload_frac"],0)+"%",
            mi=fnum(r["miou"]), vi=fnum(r["veh_iou"]), pr=fnum(r["ped_recall"]), orr=fnum(r["obj_recall"]),
            loc=fnum(r["loc_m"],2), ploc=fnum(r["ped_loc_m"],2),
            fm=_o(r["front_ms"], r["est"]), rt=_o(r["rtt_ms"], r["est"]), dl=_o(r["delivery"], r["est"], 3),
            acc=("Y" if r["accept"] else "-")))
    L += ["", "## Pareto pick (min payload within accuracy tolerance)"]
    if best:
        L.append(f"**{best['name']}** — payload {fnum(best['payload_kb'],1)}KB "
                 f"({fnum(100*best['payload_frac'],0)}% of clean), mIoU {fnum(best['miou'])}, "
                 f"ped-recall {fnum(best['ped_recall'])}, loc {fnum(best['loc_m'],2)}m.")
    else:
        L.append("No profile stayed within tolerance — the agent must trade accuracy for payload (that's the RL problem).")
    L += ["", "## For the RL controller",
          "- This table is the offline action-cost model: each row is a discrete action, columns are the "
          "reward terms (task utility) and the payload/latency/reliability cost.",
          "- **front ms / RTT ms / delivery** are measured on the loopback (CARLA transport) for the pure "
          "quant x entropy profiles; `~` marks ROI/AE profiles whose latency/reliability follow the same "
          "**payload -> {latency, reliability}** curve (see LOOPBACK_LATENCY.md) via their payload column.",
          "- Loopback delivery reflects payload/fragmentation; TRUE channel loss + variable latency arrive with "
          "the OAI/Sionna network phase, which replaces the loopback transport column.",
          ""]
    Path(out).write_text("\n".join(L))
    print(f"wrote {out}  ({len(rows)} profiles, {len(accept)} within tol)")

if __name__ == "__main__":
    main()
