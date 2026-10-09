"""Combined seven-arm table for the fixed-action screen v2 (amendments 01 and 02 disclosed).

Definitions fixed before arms 2-6 were launched (registered in amendment_02):
  admission      raw: admitted opportunities / all opportunities (PHASE6 opportunities);
                 retained P0 metric: coverage.admitted / coverage.scored_opportunities vs min_fraction (report-only)
  timeliness     conditional: timely / admitted decisions; unconditional: timely / all opportunities
  fallbacks      count, executed fallback action(s), their transmitted payload bytes/datagrams, and the share of
                 all transmitted frames and bytes (forced decisions+holds vs fallback frames)
  quality        defined (SCORED, quality defined) / undefined (SCORED, undefined) / missing (decision not SCORED);
                 localization, within-tolerance recall (vehicle 1.0 m, person 1.2 m), recall, misses, precision,
                 error tails and segmentation IoU for all scored decisions and for timely scored decisions
  verdicts       original verdict as recorded, amendment-01 and amendment-02 evaluations
Read-only on attempts; create-only outputs.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

from .analyze import Arm, MAIN, WT, integration, load_jsonl, q, sha
from . import forced_verdict as V1, forced_verdict_v2 as V2
from .report import arm_metrics, fmt, pipeline_identity, rows_for

SCREEN = WT / "rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261005_fixed_action_screen_v2"
E = MAIN / "experiments/splitfusion_fixed_action_screen_v2"
ORIGINAL = {"arm0_a33": ("campaign_02", "launch_arm0_a33.log", "STOPPED_AT_LAUNCH: original I9 'launch exit 0' clause (campaign_02)"),
            "arm1_a65": ("campaign_03", "launch_arm1_a65.log", "STOP under amendment 01: R:cell_passed (P0 policy coverage) (campaign_03)")}


def admission(arm, rows):
    e = json.loads((arm.cell / "run4_phase6/PHASE6_UE_EVIDENCE.json").read_bytes())
    opps = e["opportunities"]
    admitted = sum(bool(o["admitted"]) for o in opps)
    cov = e.get("coverage") or {}
    timely = sum(r["timely"] for r in rows)
    forced_log = load_jsonl(arm.cell / "forced_action/FORCED_ACTIONS.jsonl")
    per_frame = {int(r["frame_id"]): r for r in csv.DictReader(open(arm.cell / "per_frame_metrics.csv"))}
    kinds = {r["frame_id"]: r["kind"] for r in forced_log}

    def payload(frames):
        b = [int(per_frame[f]["payload_bytes"]) for f in frames if per_frame.get(f, {}).get("payload_bytes")]
        c = [int(per_frame[f]["payload_chunks"]) for f in frames if per_frame.get(f, {}).get("payload_chunks")]
        return dict(frames=len(frames), transmitted_with_payload=len(b), bytes_total=sum(b), datagrams_total=sum(c))
    fallback_frames = [f for f, k in kinds.items() if k == "FALLBACK"]
    forced_frames = [f for f, k in kinds.items() if k in ("POLICY_DECISION", "POLICY_HOLD")]
    fb, fo = payload(fallback_frames), payload(forced_frames)
    total_bytes = fb["bytes_total"] + fo["bytes_total"]
    fallback_actions = collections.Counter(f"{r['executed']['action_id']}:{r['executed']['profile_id']}"
                                           for r in forced_log if r["kind"] == "FALLBACK")
    reasons = collections.Counter(reason.split(":")[0] for d in e["decisions"] if d["kind"] == "FALLBACK"
                                  for reason in d.get("fallback_reasons", []))
    return dict(opportunities=len(opps), admitted=admitted, raw_admission=round(admitted / len(opps), 4) if opps else None,
        retained_p0=dict(admitted=cov.get("admitted"), scored_opportunities=cov.get("scored_opportunities"),
                         admitted_fraction=cov.get("admitted_fraction"), min_fraction=cov.get("min_fraction"),
                         verdict=cov.get("verdict"), status="REPORT_ONLY_UNDER_AMENDMENT_02"),
        timely_conditional=f"{timely}/{len(rows)}", timely_conditional_fraction=round(timely / len(rows), 4) if rows else None,
        timely_unconditional=f"{timely}/{len(opps)}", timely_unconditional_fraction=round(timely / len(opps), 4) if opps else None,
        fallbacks=dict(count=len(fallback_frames), executed_actions=dict(fallback_actions), reasons=dict(reasons),
                       frame_ids=sorted(fallback_frames), workload=fb,
                       share_of_transmitted_frames=round(len(fallback_frames) / (len(fallback_frames) + len(forced_frames)), 4),
                       share_of_transmitted_bytes=round(fb["bytes_total"] / total_bytes, 4) if total_bytes else None),
        forced_frames_workload=fo)


def quality_split(rows):
    scored = [r for r in rows if r["service_outcome"] == "SCORED"]
    return dict(decisions=len(rows), defined=sum(1 for r in scored if r["quality_defined"]),
                undefined=sum(1 for r in scored if not r["quality_defined"]),
                missing=len(rows) - len(scored),
                missing_outcomes=dict(collections.Counter(r["service_outcome"] for r in rows if r["service_outcome"] != "SCORED")),
                timely_defined=sum(1 for r in scored if r["quality_defined"] and r["timely"]),
                timely_undefined=sum(1 for r in scored if not r["quality_defined"] and r["timely"]))


def verdicts(arm, check):
    name = arm.dir.name
    if name in ORIGINAL:
        campaign, log, original = ORIGINAL[name]
        launch_log = E / campaign / log
    else:
        launch_log, original = E / "campaign_04" / f"launch_{name}.log", "amendment 02 (campaign_04)"
    integ = integration(arm, None, do_rescore=True)
    evidence = V1.collect(arm.dir, launch_returncode=1, launch_log=launch_log, integration_result=integ)
    return dict(original=original, amendment_01=V1.evaluate(evidence)["verdict"], amendment_02=V2.evaluate(evidence),
                recorded_check=check)


def run(name):
    out = E / name
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads((MAIN / "experiments/splitfusion_fixed_action_screen_v1/selection_01/SELECTION.json").read_bytes())
    arms, rows_all, identities = {}, [], {}
    for index, action in enumerate(selection["arm_order"]):
        arm = Arm(SCREEN / f"arm{index}_a{action}")
        if arm.cell is None:
            arms[arm.dir.name] = dict(action=action, launched=False)
            continue
        rows = rows_for(arm)
        rows_all += rows
        check_path = E / "campaign_04" / f"verdict_{arm.dir.name}.json"
        check = json.loads(check_path.read_bytes())["forced_screen_verdict"] if check_path.exists() else None
        arms[arm.dir.name] = dict(action=action, roles=arm.reg["arm_roles"], forced_action=arm.forced,
            metrics=arm_metrics(rows), admission=admission(arm, rows), quality_split=quality_split(rows),
            verdicts=verdicts(arm, check))
        identities[arm.dir.name] = pipeline_identity(arm)
    with open(out / "DECISIONS.jsonl", "x") as stream:
        for row in rows_all:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    with open(out / "ARMS.json", "x") as stream:
        json.dump(arms, stream, indent=1, sort_keys=True, default=str)
    with open(out / "PIPELINE_IDENTITY.json", "x") as stream:
        json.dump(identities, stream, indent=1, sort_keys=True)
    t1 = ["| Arm | Action / role | Verdict (original; A01; A02) | Admitted / opp. (retained P0) | Timely / admitted | "
          "Timely / opp. | Success / admitted | Receipt P50 / P95 ms | Pred ready P50 | Payload P50 B | Fallbacks (action; bytes share) |",
          "|" + "---|" * 11]
    t2 = ["| Arm | Defined / undefined / missing | Q_perc mean all (n) | Q_perc mean timely (n) | Person R@1.2 / R / P / miss (all) | "
          "Person R@1.2 / R (timely) | Person err P50/P90/max | Vehicle R@1.0 / R / P / miss (all) | Vehicle R@1.0 / R (timely) | "
          "Vehicle err P50/P90/max | Seg IoU P / V (all) | Seg IoU P / V (timely) |", "|" + "---|" * 12]
    for key, a in arms.items():
        if not a.get("metrics"):
            t1.append(f"| {key} | {a['action']} | not launched |" + " |" * 8)
            t2.append(f"| {key} | not launched |" + " |" * 10)
            continue
        m, ad, qs, v, f = a["metrics"], a["admission"], a["quality_split"], a["verdicts"], a["forced_action"]
        qa, qt = m["quality_all_scored_defined"], m["quality_timely_defined"]
        p0 = ad["retained_p0"]
        t1.append(" | ".join(["", key, f"{f['action_id']} {f['family']}/{f['quantizer']} q{f['q_e4']} ({', '.join(a['roles'])})",
            f"{v['original'].split(':')[0]}; {v['amendment_01']}; {v['amendment_02']['verdict']}",
            f"{ad['admitted']}/{ad['opportunities']} ({p0['admitted']}/{p0['scored_opportunities']}={fmt(p0['admitted_fraction'], 3)})",
            ad["timely_conditional"], ad["timely_unconditional"], f"{m['success_defined_quality']}/{m['decisions']}",
            f"{fmt(m['latency_ms']['all'].get('p50'))} / {fmt(m['latency_ms']['all'].get('p95'))}",
            fmt(m["stages_ms"]["prediction_ready"].get("p50")), fmt(m["payload_bytes"].get("p50"), 0),
            f"{ad['fallbacks']['count']} ({', '.join(ad['fallbacks']['executed_actions'])}; {fmt(ad['fallbacks']['share_of_transmitted_bytes'], 3)})", ""]).strip())
        cls = lambda c: f"{fmt(c['recall_within_tolerance'], 3)} / {fmt(c['recall'], 3)} / {fmt(c['precision'], 3)} / {c['fn_misses']}"
        short = lambda c: f"{fmt(c['recall_within_tolerance'], 3)} / {fmt(c['recall'], 3)}"
        err = lambda c: "/".join(fmt(c["matched_error_m"].get(k), 2) for k in ("p50", "p90", "max"))
        t2.append(" | ".join(["", key, f"{qs['defined']} / {qs['undefined']} / {qs['missing']}",
            f"{fmt(qa['q_perc'].get('mean'), 3)} ({qa['quality_defined']})", f"{fmt(qt['q_perc'].get('mean'), 3)} ({qt['quality_defined']})",
            cls(qa["person"]), short(qt["person"]), err(qa["person"]), cls(qa["vehicle"]), short(qt["vehicle"]), err(qa["vehicle"]),
            f"{fmt(qa['person']['seg_iou_pooled'], 3)} / {fmt(qa['vehicle']['seg_iou_pooled'], 3)}",
            f"{fmt(qt['person']['seg_iou_pooled'], 3)} / {fmt(qt['vehicle']['seg_iou_pooled'], 3)}", ""]).strip())
    (out / "TABLE.md").write_text("## Latency and admission\n\n" + "\n".join(t1) + "\n\n## Quality\n\n" + "\n".join(t2) + "\n")
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print("\n".join(t1)); print(); print("\n".join(t2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    run(parser.parse_args().name)
