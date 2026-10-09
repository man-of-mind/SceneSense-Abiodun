"""CONTROLLER_TABLE.md from PART3_CONTROLLERS.json + PART3_EXTRAS.json (no hand transcription)."""
import argparse
import json
from pathlib import Path

R = ["loc_vehicle", "loc_person", "seg_vehicle", "seg_person", "precision_vehicle", "precision_person"]
TIER = dict(loc_vehicle=.95, loc_person=.90, seg_vehicle=.85, seg_person=.80, precision_vehicle=.95, precision_person=.90)


def main(root):
    d = json.loads((root / "PART3_CONTROLLERS.json").read_text())
    x = json.loads((root / "PART3_EXTRAS.json").read_text())
    L = ["# Controller comparison (E1-CORRECTED primary; held-out scenes, 3 eval seeds x 6,000 decisions, identical streams)", "",
         "ret = all-scored retention vs the same scene's mode-0/q0 reference (misses count); **bold** = meets the registered tier "
         "absolutely (loc 0.95/0.90, seg 0.85/0.80, prec 0.95/0.90). dP / dTimely vs A with 95% paired block CI (300-decision "
         "blocks). NI = favourable non-inferiority vs A (dP >= -0.005 and CI low >= -0.015; timely >= A - 0.010; every "
         "retention >= A - 0.020).", ""]
    for prof in ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY", "ALL"):
        row = d["settings"]["E1_CORRECTED"][prof]
        L += [f"## {prof}" + (" (primary)" if prof == "FAVORABLE_STABLE" else ""), "",
              "| Ctrl | P | dP vs A [CI] | Timely | R_v@1.0 / R_p@1.2 | ret loc v / p | ret seg v / p | ret prec v / p | Payload P50/P99 KB | Mode-11 share | q P5/P50/P95 (sd) | MCS<=10 n | Backlog>1KB n |"
              + (" NI vs A |" if prof == "FAVORABLE_STABLE" else ""),
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if prof == "FAVORABLE_STABLE" else "")]
        for k, s in row.items():
            pv = s.get("paired_vs_A")
            f = lambda r: (f"**{s['retention'][r]:.3f}**" if s["retention"][r] >= TIER[r] else f"{s['retention'][r]:.3f}")
            q = s["q_p5_p50_p95"]
            line = (f"| {k} | {s['P']:.3f} | " + (f"{pv['dP']:+.4f} [{pv['dP_ci'][0]:+.4f}, {pv['dP_ci'][1]:+.4f}]" if pv else "—")
                    + f" | {s['timely']:.3f}" + (f" ({pv['dtimely']:+.3f})" if pv else "") + f" | {s['R_v']:.3f} / {s['R_p']:.3f} | "
                    f"{f('loc_vehicle')} / {f('loc_person')} | {f('seg_vehicle')} / {f('seg_person')} | {f('precision_vehicle')} / {f('precision_person')} | "
                    f"{s['wire_p50_kb']:.0f} / {s['wire_p99_kb']:.0f} | {s['mode_share'].get('11', 0):.3f} | {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} ({s['q_std']:.0f}) | "
                    f"{s['n_mcs_le_10']} | {s['n_backlog_gt_1kb']} |")
            if prof == "FAVORABLE_STABLE":
                ni = s.get("favorable_noninferior_vs_A")
                line += (" — |" if ni is None else (" PASS |" if ni["all"] else " FAIL: " + ", ".join(c for c, v in ni["checks"].items() if not v) + " |"))
            L.append(line)
        L.append("")
    L += ["## Direct paired differences (P), E1-CORRECTED and E2-CORRECTED", "",
          "| Profile | B - C (learned q vs fixed q6000), E1 | B - D (learned q vs S2 rule), E1 | B - C, E2 | B - D, E2 |", "|---|---|---|---|---|"]
    fm = lambda v: f"{v['dP']:+.4f} [{v['dP_ci'][0]:+.4f}, {v['dP_ci'][1]:+.4f}]"
    for prof in ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY", "ALL"):
        e1, e2 = x["E1_CORRECTED"]["paired"][prof], x["E2_CORRECTED"]["paired"][prof]
        L.append(f"| {prof} | {fm(e1['B_minus_C'])} | {fm(e1['B_minus_D'])} | {fm(e2['B_minus_C'])} | {fm(e2['B_minus_D'])} |")
    L += ["", "## Sensitivity: P and dP vs A [CI] in other settings", "", "| Setting | Profile | A | B | C | D | REF_q7000 |", "|---|---|---|---|---|---|---|"]
    for st in ("E1_OLD", "E2_CORRECTED", "E2_OLD"):
        for prof, row in d["settings"][st].items():
            L.append(f"| {st} | {prof} | {row['A']['P']:.3f} | " + " | ".join(
                f"{row[k]['P']:.3f} ({row[k]['paired_vs_A']['dP']:+.3f} [{row[k]['paired_vs_A']['dP_ci'][0]:+.3f}, {row[k]['paired_vs_A']['dP_ci'][1]:+.3f}])"
                for k in ("B", "C", "D", "REF_q7000")) + " |")
    (root / "CONTROLLER_TABLE.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    main(p.parse_args().out)
