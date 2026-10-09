"""Write METRICS_TABLES.md from metrics/ + compute_cost/, then INPUT_SHA256.json and OUTPUT_SHA256.json (create-only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_agent.splitfusion_m1_presentation_ext_v1 import metrics as MX

KEYS = ("agent", "arm_a", "action0", "arm_b")
NAME = {"agent": "Agent", "arm_a": "Static AE32/UINT8 (a54)", "action0": "Static noAE/UINT8 (a0)",
        "arm_b": "Fixed mode 11, q=0.60 †"}


def pct(v, bound=""):
    return "n/a" if v is None else f"{'≤' if bound.startswith('UPPER') else ''}{100 * v:.1f}%"


def tables(doc, cost, final):
    s = doc["sessions"]
    out = ["# M1 presentation extension — metric tables", "",
           "Primary boundary: **action-open → successful live `edge.compute` end with the CPU prediction materialized**. "
           "Deadline reference: 170 ms (the original feedback deadline). Feedback receipt is a separate column and is never "
           "substituted. Every row is a different live session on FAVORABLE_STABLE; nothing is paired across rows. "
           "† = dedicated adaptation comparison only; excluded from the main system plots.", "",
           "## 1. Prediction completion and GT coverage (unit: policy decisions)", "",
           "| Session | Decisions | ≤170 ms | Later | None | No-completion service terminals | GT objects known / no GT | Feedback ≤170 ms |",
           "|---|---:|---:|---:|---:|---|---:|---:|"]
    for k in KEYS:
        x = s[k]; st = x["prediction_states"]; cov = x["gt_coverage"]
        known = sum(v for c, v in cov.items() if not c.endswith("MISSING"))
        miss = sum(v for c, v in cov.items() if c.endswith("MISSING"))
        term = ", ".join(f"{a} {b}" for a, b in sorted(x["no_completion_service_terminals"].items())) or "—"
        out.append(f"| {NAME[k]} | {x['decisions']} | {st.get('TIMELY', 0)} | {st.get('LATE', 0)} | {st.get('NONE', 0)} | "
                   f"{term} | {known} / {miss} | {x['feedback_by_170']}/{x['decisions']} |")
    out += ["", "Service terminals are the GT service's outcome labels for decisions without a completed prediction; "
            "they are not all packet loss (e.g. `NO_GT_PRODUCER_CAPACITY` is a GT-side refusal, "
            "`UNSCORED_NO_RESULT_BY_HORIZON` means no prediction reached the scorer).", "",
            "## 2. Localisation recall (unit: eligible GT objects)", "",
            "R_c(e,170) = Σ_i [completed_i ∧ latency_i ≤ 170] · within_e_i / Σ_i eligible_i over every GT-known decision. "
            "All-completion = within-tolerance matches on every completed prediction / eligible objects on completed frames "
            "(conditional on completion).", "",
            "| Session | Class (tol.) | All completions | Deadline-aware ≤170 ms | Bound | Lost only to lateness | On failed decisions | Decisions without GT |",
            "|---|---|---:|---:|---|---:|---:|---:|"]
    for k in KEYS:
        for c in MX.CLASSES:
            L = s[k]["localisation"][c]; a, d, acct = L["all_completion"], L["deadline_aware"], L["object_accounting"]
            all_txt = "n/a (0 completed)" if a["recall"] is None else f"{pct(a['recall'])} ({a['numerator']}/{a['denominator']})"
            out.append(f"| {NAME[k]} | {c} ({MX.TOL[c]:g} m) | {all_txt} | {pct(d['recall'], d['bound'])} "
                       f"({d['numerator']}/{d['denominator']}) | {d['bound']} | {acct['within_tol_only_late_prediction']} | "
                       f"{acct['eligible_on_failed_opportunities']} | {acct['opportunities_without_gt']} |")
    out += ["", "## 3. Segmentation IoU (pooled pixels; late/absent prediction = empty predicted foreground)", "",
            "| Session | Class | All completions | Deadline-aware ≤170 ms | Evaluable decisions | Bound |", "|---|---|---:|---:|---:|---|"]
    for k in KEYS:
        for c in MX.CLASSES:
            g = s[k]["segmentation"][c]; a, d = g["all_completion"], g["deadline_aware"]
            out.append(f"| {NAME[k]} | {c} | {pct(a['iou'])} | {pct(d['iou'], d['bound'])} | "
                       f"{d['evaluable_opportunities']}/{d['opportunities']} | {d['bound']} |")
    out += ["", "noAE: intersection is 0 on all 500 decisions (no prediction), but no GT segmentation raster was retained, "
            "so the pooled union is unmeasured; the IoU is 0 whenever any GT foreground exists and is not reported as a number.", "",
            "## 4. Precision (conditional on emitted detections; 3 m match radius)", "",
            "| Session | Vehicle, all completions | Vehicle, ≤170 ms | Pedestrian, all completions | Pedestrian, ≤170 ms |",
            "|---|---:|---:|---:|---:|"]
    for k in KEYS:
        row = []
        for c in MX.CLASSES:
            L = s[k]["localisation"][c]; a, d = L["all_completion"], L["deadline_aware"]
            row += [f"{pct(a['precision'])} ({a['tp']}/{a['tp'] + a['fp']})" if a["precision"] is not None else "n/a",
                    f"{pct(d['precision_timely'])} ({d['tp_timely']}/{d['tp_timely'] + d['fp_timely']})"
                    if d["precision_timely"] is not None else "n/a"]
        out.append(f"| {NAME[k]} | " + " | ".join(row) + " |")
    out += ["", "Precision need not fall when delivery falls; read it with recall and completion coverage.", "",
            "## 5. Agent vs fixed mode 11, q=0.60 (dedicated comparison; separate sessions)", "",
            "| Metric | Agent | Fixed m11 q=0.60 | Fixed ≥ agent? |", "|---|---:|---:|---|"]
    a, b = s["agent"], s["arm_b"]
    def ci(x, key):
        v = x["bootstrap_95"][key]; return f" [{100 * v[0]:.1f}, {100 * v[1]:.1f}]"
    rows = [("Prediction ≤170 ms", f"{a['completed_by_170']}/{a['decisions']}", f"{b['completed_by_170']}/{b['decisions']}",
             b["completed_by_170"] / b["decisions"] >= a["completed_by_170"] / a["decisions"]),
            ("Payload P50 / P99 (kB, n=1,000)", f"{final['payload']['agent']['p50'] / 1e3:.1f} / {final['payload']['agent']['p99'] / 1e3:.1f}",
             f"{final['payload']['arm_b']['p50'] / 1e3:.1f} / {final['payload']['arm_b']['p99'] / 1e3:.1f}", "smaller")]
    for c in MX.CLASSES:
        for kind, lab, f, bk in (("all_completion", "all", "recall", "recall_all"), ("deadline_aware", "≤170", "recall", "recall_deadline")):
            va, vb = a["localisation"][c][kind]["recall"], b["localisation"][c][kind]["recall"]
            rows.append((f"{c} recall ({lab})", pct(va) + ci(a, f"{c}_{bk}"), pct(vb) + ci(b, f"{c}_{bk}"), vb >= va))
        for kind, lab, bk in (("all_completion", "all", "iou_all"), ("deadline_aware", "≤170", "iou_deadline")):
            va, vb = a["segmentation"][c][kind]["iou"], b["segmentation"][c][kind]["iou"]
            rows.append((f"{c} seg IoU ({lab})", pct(va) + ci(a, f"{c}_{bk}"), pct(vb) + ci(b, f"{c}_{bk}"), vb >= va))
    for m, x, y, z in rows:
        out.append(f"| {m} | {x} | {y} | {'yes' if z is True else 'no' if z is False else z} |")
    out += ["", "Brackets: 95% frame-cluster bootstrap within each session (2,000 resamples, seed 20261009); descriptive "
            "only — different sessions and scenes, so overlapping intervals do not establish equivalence and non-overlap "
            "would not establish causation.", "",
            "## 6. Compute per inference (deployed checkpoints; from compute_cost/COMPUTE_COST.json)", "",
            "| Part | GMACs | GFLOPs | Share of full local |", "|---|---:|---:|---:|"]
    p = cost["parts"]; full = p["ue_front"]["flop_counter_macs"] + p["edge_tail"]["flop_counter_macs"]
    for name, macs in (("Full local (no split, no AE)", full), ("UE front: stem + ResNet layer1 → C2", p["ue_front"]["flop_counter_macs"]),
                       ("Edge tail: layer2 → FPN → heads (dense_depth off)", p["edge_tail"]["flop_counter_macs"])):
        out.append(f"| {name} | {macs / 1e9:.3f} | {2 * macs / 1e9:.3f} | {macs / full:.2%} |")
    for f in cost["per_family_totals"]:
        if f["family"] == "noAE":
            continue
        out.append(f"| {f['family']} encoder (UE) / decoder (edge) | {f['ae_encoder_macs'] / 1e9:.3f} / {f['ae_decoder_macs'] / 1e9:.3f} | "
                   f"{2 * f['ae_encoder_macs'] / 1e9:.3f} / {2 * f['ae_decoder_macs'] / 1e9:.3f} | "
                   f"+{f['overhead_vs_full_local_q0']:.2%} |")
    out += [f"| Ranker (UE, only when q>0) | {cost['q_dependence']['ranker_macs_when_q_gt0'] / 1e9:.4f} | "
            f"{2 * cost['q_dependence']['ranker_macs_when_q_gt0'] / 1e9:.4f} | — |", "",
            "Convention: 1 MAC = 2 FLOPs; G = 10⁹; work per inference (batch 1), not GFLOP/s. Counted: Conv2d/ConvTranspose2d/"
            "Linear (FlopCounterMode, cross-checked by an independent hook count: exact agreement). Not counted: FrozenBN, "
            "GroupNorm, activations, upsampling, adds, maxpool (≈0.17% of the MAC count in output elements). Excluded: "
            "preprocessing, quantisation, zstd/packing/zero-scatter, ranker top-k, all postprocessing (decode, NMS, "
            "projection), dense_depth head, RL actor, GT/scoring. Spatial drop q does not reduce counted MACs (AE and tail "
            "run on the dense 112×192 grid).", ""]
    return "\n".join(out)


def sha(path):
    return MX.sha(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    doc = json.loads((args.out / "metrics/DEADLINE_AWARE_QUALITY.json").read_text())
    cost = json.loads((args.out / "compute_cost/COMPUTE_COST.json").read_text())
    final = json.loads((MX.FINAL_03 / "SUMMARY.json").read_text())
    with (args.out / "METRICS_TABLES.md").open("x") as stream:
        stream.write(tables(doc, cost, final))
    inputs = dict(doc["inputs"])
    for key, root in MX.ROOTS.items():
        for name in ("PHASE6_UE_EVIDENCE.json", "per_frame_metrics.csv", "edge_report.json", "SNR_EVIDENCE.json"):
            hits = list((root / "capture").rglob(name))
            if len(hits) == 1:
                inputs[str(hits[0])] = sha(hits[0])
    story = MX.MAIN / "experiments/splitfusion_m1_baselines_v1/presentation_story_01/ACTION_TIMELINE.csv"
    inputs[str(story)] = sha(story)
    (args.out / "INPUT_SHA256.json").write_text(json.dumps(inputs, indent=1, sort_keys=True) + "\n")
    code = Path(__file__).resolve().parent
    outputs = {str(p.relative_to(args.out)): sha(p) for p in sorted(args.out.rglob("*"))
               if p.is_file() and p.name != "OUTPUT_SHA256.json"}
    outputs.update({f"code/{p.name}": sha(p) for p in sorted(code.glob("*.py"))})
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps(outputs, indent=1, sort_keys=True) + "\n")
    print(json.dumps(dict(inputs=len(inputs), outputs=len(outputs))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
