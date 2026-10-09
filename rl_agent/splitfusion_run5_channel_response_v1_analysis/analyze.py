"""Analysis of the single retention capture (gc_v2 configuration + paired-input retention); read-only, create-only.
The gc_v2 validation is attached as DESCRIPTIVE context (different session); no retention-overhead causal claim.

Original gc_v2 header: Analysis of the single guarded validation (adopted GC policy + numeric producer v2); read-only, create-only.

Reports GT readiness, feedback latency, timeliness, prep-worker memory, exact rescoring and the existing quality
metrics, using the retained full Run-5 rows/figures (``splitfusion_run5_full_live_analysis_v1.plots.render``) and the
GC-trace metrics (``splitfusion_run5_full_live_gc_analysis_v1.analyze.arm_gc``). Earlier sessions (valid_01, the GC
pair) are attached as DESCRIPTIVE context only: different sessions, traffic and actions.

  python3 -m rl_agent.splitfusion_run5_channel_response_v1_analysis.analyze --campaign <campaign dir> --out <dir>
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_agent.splitfusion_run5_full_live_analysis_v1 import plots as PL
from rl_agent.splitfusion_run5_full_live_gc_analysis_v1.analyze import arm_gc

MAIN = Path(__file__).resolve().parents[2]
ROOT = MAIN.parent / ("abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
                      "20261006_run5_channel_response_v1/validation")
CONTEXT = dict(valid_01=MAIN / "experiments/splitfusion_run5_full_live_v1/valid_01/plots/SUMMARY_valid1000_full.json",
               gc_pair=MAIN / "experiments/splitfusion_run5_full_live_gc_v1/pair_01_analysis/PAIR_ANALYSIS.json",
               gc_v2_validation=MAIN / "experiments/splitfusion_run5_full_live_gc_v2/validation_01_analysis/VALIDATION_ANALYSIS.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(campaign, out):
    campaign, out = Path(campaign), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    verdict = json.loads((campaign / "VERDICT_validation.json").read_bytes())
    summary = PL.render(ROOT, out / "validation", "validation_b1000")
    rows = [json.loads(l) for l in open(out / "validation" / "DECISIONS_validation_b1000.jsonl")]
    gc = arm_gc(ROOT, rows)
    cls = summary["outcome_classes"]
    cell = next((ROOT / "capture/cells").iterdir())
    actors = []
    for r in rows:
        meta = cell / f"numeric_gt/service/archive/frame_{r['frame_id']}/producer/input_metadata.json"
        if meta.exists():
            actors.append(len(json.loads(meta.read_bytes()).get("actor_boxes", [])))
    table = dict(
        integration=verdict.get("verdict"), integration_reasons=verdict.get("reasons"),
        gc_and_producer_checks=verdict.get("gc_trace"),
        decisions=summary["decisions"], timely=summary["timely"], timely_fraction=summary["timely"] / summary["decisions"],
        timely_wilson=summary["timely_wilson"], timely_defined=cls.get("TIMELY_DEFINED", 0),
        timely_undefined=cls.get("TIMELY_UNDEFINED", 0), late=cls.get("LATE", 0), missing=cls.get("MISSING", 0),
        feedback_receipt_ms=summary["receipt_ms"], gt_ready_ms=summary["boundaries_ms"]["gt_ready_published"],
        prediction_at_service_ms=summary["boundaries_ms"]["prediction_at_service"],
        rescored_exact=(verdict.get("info") or {}).get("rescored_exact"), scored=(verdict.get("info") or {}).get("scored"),
        vehicle=summary["quality_all_scored"]["vehicle"], person=summary["quality_all_scored"]["person"],
        q_seg=summary["quality_all_scored"]["q_seg"], quality_timely_scored=summary["quality_timely_scored"],
        payload_bytes=summary["payload_bytes"], mode_share=summary["mode_share"], q_by_mode_11=summary["q_by_mode"].get(11),
        fallbacks=summary["fallbacks"], actor_boxes=dict(mean=float(np.mean(actors)), p95=float(np.percentile(actors, 95)))
        if actors else None, gc={k: v for k, v in gc.items() if k != "_series"})
    context = {}
    for name, path in CONTEXT.items():
        if path.exists():
            value = json.loads(path.read_bytes())
            context[name] = dict(path=str(path), sha256=sha(path), note="descriptive only; different session",
                                 summary=value if name == "valid_01" else value.get("table"))
    fig, ax = plt.subplots(figsize=(8, 5))
    v = np.sort([r["ue_receipt"] for r in rows if r["ue_receipt"] is not None])
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", label=f"validation (n={len(v)}, timely {np.mean(v <= 170):.3f})")
    ax.axvline(170, color="k", ls="--", lw=1, label="170 ms (inclusive)")
    ax.set_xlabel("UE feedback receipt after action-open (ms)"); ax.set_ylabel("cumulative fraction"); ax.grid(alpha=.3)
    ax.set_title("Feedback latency CDF (raw)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig01_feedback_latency_cdf.png", dpi=130); plt.close(fig)
    s = gc["_series"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(s["decisions"], s["freeze"], ".", ms=3, label="FREEZE wall (prep worker)")
    if s["idle"]:
        k, ms = zip(*s["idle"]); axes[0].plot(k, ms, "|", color="green", label="idle full collection")
    axes[0].set_ylabel("ms"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=8); axes[0].set_title("GT prep worker")
    if s["rss"]:
        k, r = zip(*s["rss"]); axes[1].plot(k, np.asarray(r) / 2**20, lw=1, label="prep RSS after BIND")
    axes[1].set_xlabel("decision index"); axes[1].set_ylabel("MiB"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig02_prep_worker_timeline.png", dpi=130); plt.close(fig)
    retention = json.loads((ROOT / "RETENTION_RESULT.json").read_bytes()) if (ROOT / "RETENTION_RESULT.json").exists() else None
    table["retention"] = None if retention is None else {k: retention.get(k) for k in (
        "finalized", "error", "status_counts", "counters", "observed_policy_decisions", "host_bytes", "storage_bytes",
        "finalize_wall_s", "decision_path_bookkeeping", "peak_host_bytes", "owner_thread", "faults")}
    result = dict(schema="splitfusion.run5_channel_response_v1.validation_analysis.v1", table=table, context=context,
                  inputs={str(p): sha(p) for p in (campaign / "VERDICT_validation.json", ROOT / "registration.json",
                          ROOT / "gt_gc_trace/prep_gc_trace.json")}, freeze_sha256=sha(campaign / "FREEZE.json"),
                  script_sha256=sha(__file__), percentile_method="numpy linear")
    (out / "VALIDATION_ANALYSIS.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                                        if p.is_file() and p.name != "OUTPUT_SHA256.json"}, indent=1))
    print(json.dumps({k: table[k] for k in ("integration", "decisions", "timely", "timely_fraction", "late", "missing",
                                            "rescored_exact")}, indent=1, default=str))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.campaign, a.out)
