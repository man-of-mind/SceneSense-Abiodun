"""Analysis of the paired GT GC live qualification (read-only over both arm roots; create-only outputs).

Per arm: the retained full Run-5 rows/summary/figures (``splitfusion_run5_full_live_analysis_v1.plots.render``), the
retained integration verdict (recorded by the campaign) and the GC mechanism metrics from the worker traces and the
service frame events. Then the REGISTERED decision rule (contract.ACCEPTANCE / SAFETY; ACCEPTANCE.json) is applied.
Percentiles: NumPy linear. Headline timeliness/quality differences are between different sessions (different traffic
and actions); they are reported, not attributed to the GC policy.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_run5_full_live_analysis_v1 import plots as PL
from rl_agent.splitfusion_run5_full_live_gc_v1 import contract as C
from rl_agent.splitfusion_run5_full_live_gc_v1 import gc_trace as T

MAIN = Path(__file__).resolve().parents[2]
ROOTS = MAIN.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261006_run5_full_live_gc_v1"
ARMS = ("control", "candidate")
COLORS = {"control": "#4C72B0", "candidate": "#DD8452"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def q(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return dict(n=0)
    p = np.percentile(a, [50, 95, 99])
    return dict(n=int(len(a)), p50=float(p[0]), p95=float(p[1]), p99=float(p[2]), max=float(a.max()), mean=float(a.mean()))


def per_decision_prep(trace):
    """FREEZE and BIND wall per decision index from the prep trace (ms)."""
    dec = T.decision_index(trace["ops"])
    out = collections.defaultdict(dict)
    for r in trace["ops"]:
        if r["op"] in ("FREEZE", "BIND"):
            out[dec[r["seq"]]][r["op"]] = (r["end_raw_ns"] - r["begin_raw_ns"]) / 1e6
            out[dec[r["seq"]]][r["op"] + "_rss1"] = r["rss1"]
    gen2 = collections.defaultdict(list)
    for g in trace["gen2"]:
        if g["op_seq"] is not None and g["op"] in T.FEEDBACK_OPS:
            gen2[dec.get(g["op_seq"], 0)].append(g["wall_ns"] / 1e6)
    return out, gen2


def service_events(cell):
    rows = [json.loads(l) for l in open(cell / "numeric_gt/service/FRAMES.jsonl")]
    out = {}
    for f in rows:
        if not f.get("decision"):
            continue
        ev = {}
        for e in f["events"]:
            ev.setdefault(e["name"], e["raw_ns"])
        a = f["action_open_raw_ns"]
        rel = lambda name: None if name not in ev else (ev[name] - a) / 1e6
        dur = lambda x, y: None if x not in ev or y not in ev else (ev[y] - ev[x]) / 1e6
        out[f["frame_id"]] = dict(gt_ready_ms=rel("published"), parent_freeze_to_bound_ms=dur("freeze_dispatched", "bound"),
                                  outcome=f.get("outcome"))
    return out


def slope_bytes_per_decision(xs, ys):
    if len(xs) < 3:
        return None
    return float(np.polyfit(np.asarray(xs, float), np.asarray(ys, float), 1)[0])


def arm_gc(arm_dir, rows):
    d = arm_dir / C.TRACE_DIR
    prep = json.loads((d / "prep_gc_trace.json").read_bytes())
    score = json.loads((d / "score_gc_trace.json").read_bytes())
    warm = C.GC_POLICY["warmup_decisions"]
    s = T.summarize(prep, warm)
    per, gen2 = per_decision_prep(prep)
    decisions = sorted(k for k in per if k > 0)
    fb = [per[k].get("FREEZE", 0) + per[k].get("BIND", 0) for k in decisions if "FREEZE" in per[k] and "BIND" in per[k]]
    freeze = [per[k]["FREEZE"] for k in decisions if "FREEZE" in per[k]]
    med = float(np.median(freeze)) if freeze else None
    spikes = [k for k in decisions if "FREEZE" in per[k] and per[k]["FREEZE"] > med + 30]
    cell = next((arm_dir / "capture/cells").iterdir())
    svc = service_events(cell)
    rss = [(k, per[k]["BIND_rss1"]) for k in decisions if "BIND_rss1" in per[k]]
    after = [(k, v) for k, v in rss if k > warm]
    half = [(k, v) for k, v in after if k >= (after[0][0] + after[-1][0]) / 2] if after else []
    mem_samples = json.loads((cell / "numeric_gt/service/MEMORY_SAMPLES.json").read_bytes())
    idle = s["idle_wall_ms"]
    return dict(
        decisions_in_trace=s["decisions"],
        gen2_feedback_warmup=s["gen2_feedback_warmup"], gen2_feedback_after_warmup=s["gen2_feedback_after_warmup"],
        gen2_feedback_after_warmup_ms=q(s["gen2_feedback_after_warmup_ms"]),
        gen2_other_requests=s["gen2_other_requests"], gen2_between_requests=s["gen2_between_requests"],
        gen2_total=s["gen2_total"], score_worker_gen2_total=len(score["gen2"]),
        freeze_ms=q(freeze), freeze_bind_ms=q(fb), spikes=len(spikes), spike_decisions=spikes,
        spike_rule=f"FREEZE > arm median ({med:.3f} ms) + 30 ms" if med is not None else None,
        parent_freeze_to_bound_ms=q([v["parent_freeze_to_bound_ms"] for v in svc.values()]),
        gt_ready_ms=q([v["gt_ready_ms"] for v in svc.values()]),
        idle_collections=s["idle_collections"], idle_wall_ms=q(idle), idle_request_arrivals=s["idle_request_arrivals"],
        idle_arrival_fraction=(s["idle_request_arrivals"] / s["idle_collections"]) if s["idle_collections"] else None,
        skipped_pending=s["skipped_pending"], safety_stop=s["safety_stop"],
        rss_first=s["rss_first"], rss_max=s["rss_max"], rss_last=s["rss_last"], rss_final=prep["rss_final"],
        rss_growth_after_warmup=(after[-1][1] - after[0][1]) if len(after) > 1 else None,
        rss_slope_second_half_bytes_per_decision=slope_bytes_per_decision([k for k, _ in half], [v for _, v in half]),
        service_sampled_prep_rss_kib=q([m["prep"] for m in mem_samples]),
        restored_equals_initial=prep["restored_equals_initial"], gc_state_after_restore=prep["gc_state_after_restore"],
        trace_dropped=prep["dropped"], policy_installed=prep["policy_installed"],
        _series=dict(decisions=decisions, freeze=[per[k].get("FREEZE") for k in decisions],
                     gen2=[(k, ms) for k in decisions for ms in gen2.get(k, [])], rss=rss,
                     idle=[(e.get("decision"), e["wall_ms"]) for e in (prep.get("policy_events") or [])
                           if e.get("kind") == "idle_full_collect"]))


def decide(gc, verdicts):
    A, ctl, cand = C.ACCEPTANCE, gc["control"], gc["candidate"]
    integ = {a: verdicts[a].get("verdict") == "PASS_ACTOR_DRIVEN_INTEGRATION" and verdicts[a]["gc_trace"]["passed"]
             for a in ARMS}
    checks = dict(
        both_arms_integration=all(integ.values()),
        no_safety_stop=cand["safety_stop"] is None,
        control_reproduces=ctl["gen2_feedback_after_warmup"] >= A["control_reproduces_min_in_request_gen2"],
        candidate_zero_in_request_after_warmup=cand["gen2_feedback_after_warmup"] == A["candidate_in_request_gen2_after_warmup"],
        idle_p99=cand["idle_wall_ms"].get("p99", 0) <= A["idle_collection_p99_max_ms"],
        idle_max=cand["idle_wall_ms"].get("max", 0) <= A["idle_collection_max_ms"],
        idle_collisions=(cand["idle_arrival_fraction"] or 0) <= A["idle_arrival_collision_max_fraction"],
        rss_growth=(cand["rss_growth_after_warmup"] is not None and ctl["rss_growth_after_warmup"] is not None
                    and cand["rss_growth_after_warmup"] <= ctl["rss_growth_after_warmup"]
                    + A["rss_growth_after_warmup_max_excess_over_control_bytes"]),
        rss_slope=(cand["rss_slope_second_half_bytes_per_decision"] is not None
                   and cand["rss_slope_second_half_bytes_per_decision"] <= A["rss_slope_second_half_max_bytes_per_decision"]))
    if not checks["no_safety_stop"]:
        decision = "REJECT"
    elif not checks["both_arms_integration"]:
        decision = "INCONCLUSIVE"     # unless the failure is attributable to the policy (examined in the report)
    elif not checks["control_reproduces"]:
        decision = "INCONCLUSIVE"
    elif all(checks.values()):
        decision = "RETAIN"
    else:
        decision = "REJECT"
    return dict(decision=decision, checks=checks, integration=integ)


def figures(out, summaries, gc, rows):
    # 1. feedback-latency CDFs
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in ARMS:
        v = np.sort([r["ue_receipt"] for r in rows[arm] if r["ue_receipt"] is not None])
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLORS[arm],
                label=f"{arm} (n={len(v)}, timely {np.mean(v <= 170):.3f})")
    ax.axvline(170, color="k", ls="--", lw=1, label="170 ms deadline (inclusive)")
    ax.set_xlabel("UE feedback receipt after action-open (ms)"); ax.set_ylabel("cumulative fraction of decisions")
    ax.set_title("Feedback latency CDF (raw, unsmoothed; different sessions)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig01_feedback_latency_cdf.png", dpi=130); plt.close(fig)
    # 2. GC / GT-spike timeline
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, arm in zip(axes, ARMS):
        s = gc[arm]["_series"]
        ax.plot(s["decisions"], s["freeze"], ".", ms=3, color=COLORS[arm], label="FREEZE wall (prep worker)")
        if s["gen2"]:
            k, ms = zip(*s["gen2"])
            ax.plot(k, ms, "x", color="crimson", ms=6, label="full (gen2) collection inside FREEZE/BIND")
        if s["idle"]:
            k, ms = zip(*s["idle"])
            ax.plot(k, ms, "|", color="green", ms=6, label="idle full collection (after reply)")
        ax.axvline(C.GC_POLICY["warmup_decisions"] + .5, color="grey", ls=":", lw=1)
        ax.set_ylabel("ms"); ax.set_title(f"{arm}: spikes {gc[arm]['spikes']}, in-request gen2 after warm-up "
                                          f"{gc[arm]['gen2_feedback_after_warmup']}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[-1].set_xlabel("decision index (prep worker FREEZE order)")
    fig.tight_layout(); fig.savefig(out / "fig02_gc_spike_timeline.png", dpi=130); plt.close(fig)
    # 3. memory timeline
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for arm in ARMS:
        r = gc[arm]["_series"]["rss"]
        if r:
            k, v = zip(*r)
            ax.plot(k, np.asarray(v) / 2**20, color=COLORS[arm], lw=1, label=f"{arm} prep RSS after BIND")
    ax.axhline(C.SAFETY["prep_rss_max_bytes"] / 2**20, color="crimson", ls="--", lw=1, label="safety stop (RSS)")
    ax.set_xlabel("decision index"); ax.set_ylabel("MiB"); ax.set_title("GT prep worker RSS over the run")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig03_prep_rss_timeline.png", dpi=130); plt.close(fig)


def run(campaign_dir, out):
    campaign_dir, out = Path(campaign_dir), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    summaries, rows, gc, verdicts = {}, {}, {}, {}
    for arm in ARMS:
        arm_dir = ROOTS / arm
        verdicts[arm] = json.loads((campaign_dir / f"VERDICT_{arm}.json").read_bytes())
        summaries[arm] = PL.render(arm_dir, out / arm, f"{arm}_b1000")
        rows[arm] = [json.loads(l) for l in open(out / arm / f"DECISIONS_{arm}_b1000.jsonl")]
        gc[arm] = arm_gc(arm_dir, rows[arm])
    decision = decide(gc, verdicts)
    figures(out, summaries, gc, rows)
    table = {}
    for arm in ARMS:
        s, g, rs = summaries[arm], gc[arm], rows[arm]
        cls = s["outcome_classes"]
        cell = next((ROOTS / arm / "capture/cells").iterdir())
        actors = []
        for r in rs:
            meta = cell / f"numeric_gt/service/archive/frame_{r['frame_id']}/producer/input_metadata.json"
            if meta.exists():
                actors.append(len(json.loads(meta.read_bytes()).get("actor_boxes", [])))
        table[arm] = dict(
            decisions=s["decisions"], timely=s["timely"], timely_fraction=s["timely"] / s["decisions"],
            timely_wilson=s["timely_wilson"], timely_defined=cls.get("TIMELY_DEFINED", 0),
            timely_undefined=cls.get("TIMELY_UNDEFINED", 0), late=cls.get("LATE", 0), missing=cls.get("MISSING", 0),
            receipt_ms=s["receipt_ms"]["all"], gt_ready_ms=s["boundaries_ms"]["gt_ready_published"],
            prediction_at_service_ms=s["boundaries_ms"]["prediction_at_service"],
            vehicle=s["quality_all_scored"]["vehicle"], person=s["quality_all_scored"]["person"],
            q_seg=s["quality_all_scored"]["q_seg"], payload_bytes=s["payload_bytes"], mode_share=s["mode_share"],
            q_by_mode_11=s["q_by_mode"].get(11), fallbacks=s["fallbacks"], snr_db=s["snr_db"],
            camera_si=s["camera_si"], radar_p40=s["radar_p40"], actor_boxes=q(actors) if actors else None,
            integration=verdicts[arm].get("verdict"), integration_reasons=verdicts[arm].get("reasons"),
            rescored_exact=(verdicts[arm].get("info") or {}).get("rescored_exact"),
            scored=(verdicts[arm].get("info") or {}).get("scored"),
            gc={k: v for k, v in g.items() if k != "_series"})
    result = dict(schema="splitfusion.run5_full_live_gc.pair_analysis.v1", decision=decision, table=table,
                  registered=dict(policy=C.GC_POLICY, safety=C.SAFETY, acceptance=C.ACCEPTANCE),
                  inputs={str(p): sha(p) for arm in ARMS for p in (campaign_dir / f"VERDICT_{arm}.json",
                          ROOTS / arm / "registration.json", ROOTS / arm / C.TRACE_DIR / "prep_gc_trace.json",
                          ROOTS / arm / C.TRACE_DIR / "score_gc_trace.json")},
                  freeze_sha256=sha(campaign_dir / "FREEZE.json"), script_sha256=sha(__file__),
                  percentile_method="numpy linear", sessions_differ=True)
    (out / "PAIR_ANALYSIS.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps(dict(decision=decision), indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.campaign, a.out)
