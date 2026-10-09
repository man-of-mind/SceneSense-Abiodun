"""Does the unchanged T_v2b immediate model stay within a supported interpretation at the small positive backlogs the
corrected transition produces? Existing live decisions only (all 8 sessions; FADE labelled diagnostic)."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2
from rl_agent.splitfusion_run5_uplink_lowmcs_v1 import plan as LP

from . import sessions as S


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    model = A2.ProductionTransportModelV2.load(S.ABI / LP.T_V2B_RELPATH)
    fit_support = model.fit_support
    groups = {}
    for name, (rel, _, part) in S.SESSIONS.items():
        for r in S.load_decisions(rel).values():
            if (r.get("executed_action") or {}).get("kind") != "POLICY_DECISION" or r.get("cold"):
                continue
            if None in (r.get("pre_action_rlc_backlog_bytes"), r.get("prior_ul_mcs"), r.get("ue_last_send"), r.get("edge_reassembly_complete")):
                continue
            B = float(r["pre_action_rlc_backlog_bytes"])
            g = ("B=0" if B == 0 else ("0<B<=1KB" if B <= 1000 else "B>1KB")) + ("|DIAG" if part != "FIT" else "")
            pr = model.predict(pre_enqueue_backlog_bytes=B, wire_bytes=V1.udp_application_bytes(int(r["payload_bytes"])), prior_ul_mcs=int(r["prior_ul_mcs"]))
            T = float(r["edge_reassembly_complete"]) - float(r["ue_last_send"])
            groups.setdefault(g, []).append(dict(T=T, mu=pr.conditional_latency_ms, p=pr.on_time_probability, B=B,
                                                 timely=float(r["ue_receipt"]) <= 170.0))
    res = dict(schema="splitfusion.run5_backlog_transition.tv2b_check.v1", fit_support=fit_support, groups={})
    for g, xs in sorted(groups.items()):
        res["groups"][g] = dict(n=len(xs), T_p50=statistics.median(x["T"] for x in xs), mu_p50=statistics.median(x["mu"] for x in xs),
                                mean_error_ms=statistics.fmean(x["mu"] - x["T"] for x in xs),
                                transport_le170_measured=sum(x["T"] <= 170 for x in xs) / len(xs),
                                p_on_time_mean=statistics.fmean(x["p"] for x in xs),
                                feedback_timely=sum(x["timely"] for x in xs) / len(xs), B_max=max(x["B"] for x in xs))
        v = res["groups"][g]
        print(f"{g:16s} n {v['n']:5d} transport T P50 {v['T_p50']:6.1f} ms vs T_v2b mu P50 {v['mu_p50']:6.1f} (mean err {v['mean_error_ms']:+6.1f}); "
              f"transport<=170 measured {v['transport_le170_measured']:.3f} vs p_on_time {v['p_on_time_mean']:.3f}; feedback timely {v['feedback_timely']:.3f}; B max {v['B_max']:.0f}")
    print("T_v2b raw backlog fit support:", fit_support)
    (out / "TV2B_CHECK.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    run(Path(p.parse_args().out))
