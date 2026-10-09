"""GC-trace and producer checks for the validation (pure over durable evidence)."""
import json
from pathlib import Path

from rl_agent.splitfusion_run5_full_live_gc_analysis_v1.gc_check import check as gc_check
from rl_agent.splitfusion_run5_full_live_gc_v2 import contract as C


def check(arm_dir):
    arm_dir = Path(arm_dir)
    result = gc_check(arm_dir)
    reg = json.loads((arm_dir / "registration.json").read_bytes())
    producer = reg["gt_gc"]["producer"]
    trace_dir = arm_dir / C.TRACE_DIR
    prep = json.loads((trace_dir / "prep_gc_trace.json").read_bytes()) if (trace_dir / "prep_gc_trace.json").exists() else {}
    sub = json.loads((trace_dir / "service_substitution.json").read_bytes()) if (trace_dir / "service_substitution.json").exists() else {}
    from rl_agent.splitfusion_numeric_gt_producer_v2.source_v2 import package_sources
    pc = dict(
        constructed_registered_class=prep.get("producer_constructed") == C.PRODUCER_CLASS[producer],
        switch_restored=(prep.get("producer_switch_restored") is True) if producer == "v2" else prep.get("producer_switch_restored") is None,
        prep_received_registered_producer=any(s.get("role") == "prep" and s.get("producer") == producer for s in sub.get("substitutions", [])),
        score_received_v1=any(s.get("role") == "score" and s.get("producer") == "v1" for s in sub.get("substitutions", [])),
        v2_sources_equal_main=(producer != "v2") or prep.get("producer_v2_sources") == package_sources())
    result["producer"] = dict(producer=producer, checks=pc, passed=all(pc.values()),
                              constructed=prep.get("producer_constructed"))
    result["passed"] = bool(result["passed"] and result["producer"]["passed"])
    return result
