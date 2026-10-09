"""Revision 01 of presentation_ext_01: reporting corrections only (create-only ``revision_01/``; originals unchanged).

1. Agent compute weighting: separate the 495 policy decisions from the 495 holds and 10 fallbacks (the compute
   sub-report counted 1,000 frame-level records as "decisions").
2. No FLOP fraction is stated for the uncounted operations: output-element counts are not an operation count.
3. AE32/UINT8 completed-only and deadline-aware recall use different object denominators (stated explicitly).
4. The 288-cell campaign is measured historical live evidence, distinct from later modelled evaluations.
No metric value changes; no live service, inference or capture.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_m1_presentation_ext_v1 import finalize as F
from rl_agent.splitfusion_m1_presentation_ext_v1 import metrics as MX
from rl_agent.splitfusion_m1_presentation_ext_v1 import render as R

MODE_FAMILY = {m: ("noAE", "AE128", "AE64", "AE32")[m // 3] for m in range(12)}
MODE_QUANT = {m: ("UINT8", "UINT6", "UINT4")[m % 3] for m in range(12)}


def agent_compute(cost):
    evidence = MX.ROOTS["agent"] / "capture/cells/a71__favorable_stable/run4_phase6/PHASE6_UE_EVIDENCE.json"
    rows = json.loads(evidence.read_bytes())["decisions"]
    family = {f["family"]: f for f in cost["per_family_totals"]}

    def macs(row):
        f = family[MODE_FAMILY[int(row["mode_id"])]]
        ue = f["ue_macs_q_gt0"] if int(row["q_e4"]) > 0 else f["ue_macs_q0"]
        return ue, f["edge_macs"]

    out = dict(source=str(evidence), source_sha256=MX.sha(evidence), frame_level_records=len(rows), kinds={})
    for kind in ("POLICY_DECISION", "POLICY_HOLD", "FALLBACK"):
        sub = [r for r in rows if r["kind"] == kind]
        ue, edge = zip(*(macs(r) for r in sub)) if sub else ((), ())
        out["kinds"][kind] = dict(
            count=len(sub), modes={str(k): v for k, v in sorted(collections.Counter(int(r["mode_id"]) for r in sub).items())},
            q_e4_zero=sum(int(r["q_e4"]) == 0 for r in sub), q_e4_min=min(int(r["q_e4"]) for r in sub),
            q_e4_max=max(int(r["q_e4"]) for r in sub),
            mean_ue_gmacs=float(np.mean(ue)) / 1e9, mean_edge_gmacs=float(np.mean(edge)) / 1e9,
            mean_end_to_end_gmacs=float(np.mean(np.add(ue, edge))) / 1e9)
    allue, alledge = zip(*(macs(r) for r in rows))
    out["per_transmitted_frame_mean_end_to_end_gmacs"] = float(np.mean(np.add(allue, alledge))) / 1e9
    out["definitions"] = dict(
        POLICY_DECISION="an actor call that selects (mode, q) for a two-frame cycle (495)",
        POLICY_HOLD="the second frame of a cycle, executing the held action; not an actor choice (495)",
        FALLBACK="a typed external fallback frame executing mode 11 / q=9800 (a71); not an actor choice (10)",
        weighting="per-decision means use the 495 policy decisions; per-frame means use all 1,000 transmitted frames")
    return out


def compute_figure(out, cost):
    """render.fig_compute with the corrected footer (no fraction stated for uncounted operations)."""
    original = R.canvas

    def canvas(title, subtitle, footer):
        footer = ("Work per inference (batch 1), not GFLOP/s or latency. Counted: Conv/Linear MACs; 1 MAC = 2 FLOPs.\n"
                  "Norms, activations, upsampling, pooling and adds are not counted, and no FLOP figure is claimed for "
                  "them.\nExcluded: preprocessing, quantisation, zstd, postprocessing (decode, NMS), RL actor.")
        return original(title, subtitle, footer)
    R.canvas = canvas
    try:
        R.fig_compute(out, cost)
    finally:
        R.canvas = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args(argv)
    base = args.base
    manifest = json.loads((base / "OUTPUT_SHA256.json").read_text())
    for name, digest in manifest.items():
        if not name.startswith("code/") and MX.sha(base / name) != digest:
            raise ValueError("base presentation_ext_01 output changed: " + name)
    out = base / "revision_01"
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=False)
    doc = json.loads((base / "metrics/DEADLINE_AWARE_QUALITY.json").read_text())
    cost = json.loads((base / "compute_cost/COMPUTE_COST.json").read_text())
    final = json.loads((MX.FINAL_03 / "SUMMARY.json").read_text())

    corrected = agent_compute(cost)
    (out / "AGENT_COMPUTE_WEIGHTING.json").write_text(json.dumps(corrected, indent=1, sort_keys=True) + "\n")

    a54 = doc["sessions"]["arm_a"]["localisation"]
    denominators = {c: (a54[c]["all_completion"]["denominator"], a54[c]["deadline_aware"]["denominator"])
                    for c in MX.CLASSES}
    R.bars_quality(
        plots, doc, final, R.MAIN_KEYS, "02_system_deadline_quality_bars.png",
        "Quality actually available by 170 ms",
        "Light: all completed predictions · solid: deadline-aware (late or absent = no detections, empty mask)",
        f"Light bars divide by objects on completed frames only; solid bars divide by objects on every decision with GT. "
        f"AE32: {denominators['vehicle'][0]} vs ≥{denominators['vehicle'][1]} vehicles,\n"
        f"{denominators['person'][0]} vs ≥{denominators['person'][1]} pedestrians. ≤ = upper bound (30/500 AE32 failures "
        "lack GT segmentation, 4 lack all GT). noAE: 0 completions; seg GT not retained (n/a).")
    R.table_figure(
        plots, doc, R.MAIN_KEYS, "03_system_metric_table.png", "Delivery-aware quality: denominators and boundaries",
        "Prediction boundary: action-open → successful edge prediction materialized · feedback receipt kept separate",
        "Recall: within tolerance / eligible objects. 'All completions' rows: objects on completed frames; '≤170 ms' rows:\n"
        "objects on every decision with GT (different denominators). ≤ = upper bound. * noAE: intersection 0; GT seg not retained.")
    compute_figure(plots, cost)

    tables = F.tables(doc, cost, final)
    old = ("Not counted: FrozenBN, GroupNorm, activations, upsampling, adds, maxpool (≈0.17% of the MAC count in "
           "output elements).")
    if old not in tables:
        raise ValueError("compute paragraph drift")
    tables = tables.replace(old, "Not counted: FrozenBN, GroupNorm, activations, upsampling, pooling and adds; no "
                                 "FLOP figure is claimed for them (their output-element counts are not an operation "
                                 "count).")
    tables = tables.replace("# M1 presentation extension — metric tables",
                            "# M1 presentation extension — metric tables (revision 01)")
    tables = tables.replace(
        "All-completion = within-tolerance matches on every completed prediction / eligible objects on completed frames "
        "(conditional on completion).",
        "All-completion = within-tolerance matches on every completed prediction / eligible objects on completed frames "
        "(conditional on completion). **The two columns use different object denominators wherever some decisions "
        "did not complete** (AE32/UINT8: 801 vs ≥817 vehicles, 622 vs ≥670 pedestrians); they are not a before/after "
        "pair on one denominator.")
    k = corrected["kinds"]
    tables += (f"\nAgent compute weighting (revision 01): the agent capture has {corrected['frame_level_records']} "
               f"frame-level records = {k['POLICY_DECISION']['count']} policy decisions (modes "
               f"{k['POLICY_DECISION']['modes']}) + {k['POLICY_HOLD']['count']} holds (same actions) + "
               f"{k['FALLBACK']['count']} fallbacks (mode 11, q=9800, a71). Mean end-to-end work: "
               f"{k['POLICY_DECISION']['mean_end_to_end_gmacs']:.3f} GMACs per policy decision; "
               f"{corrected['per_transmitted_frame_mean_end_to_end_gmacs']:.3f} GMACs per transmitted frame.\n")
    (out / "METRICS_TABLES_R1.md").write_text(tables)
    print(json.dumps(dict(agent=corrected["kinds"], per_frame=corrected["per_transmitted_frame_mean_end_to_end_gmacs"]),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
