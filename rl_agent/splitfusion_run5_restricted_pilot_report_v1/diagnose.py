"""Train-vs-held-out diagnosis: the same fixed mode-11 actions in the TRAIN and EVAL pilot environments, plus the raw
measured chain timeliness of each half. Read-only use of the authorized pilot collectors (evaluation seed 9017,
1,000 decisions per cell). Diagnostic only: nothing here retrains, retunes or reselects anything.
"""
import json
import sys
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CR
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_restricted_pilot_v1 import collector as C
from rl_agent.splitfusion_run5_restricted_pilot_v1.campaign import BINDING, components


def run(authorization: Path, out: Path, n: int = 1000) -> dict:
    shared = CR.load_shared_sources()
    result = {}
    for split in ("TRAIN", "EVAL"):
        lat, cat, trace, fb = components(split)
        cells = {}
        for q in (3000, 4000, 5000, 6000, 7000, 8000):
            col = C.PilotRun5CollectorV1(latency=lat, catalog=cat, trace=trace, fallback=fb, arm="CONTROL", seed=9017,
                                         authorization=authorization, binding_dir=BINDING, shared_sources=shared)
            for i in range(n):
                col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=11, q_e4=q,
                                                                   source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
            d = col.diagnostics
            rec = [x["receipt_ms"] for x in d if x["receipt_ms"] is not None]
            inc = [x for x in d if not x["excluded"]]
            cells[str(q)] = dict(timely=float(np.mean([r <= 170 for r in rec])), receipt_p50=float(np.median(rec)),
                                 receipt_p90=float(np.percentile(rec, 90)),
                                 prediction_at_service_p50=float(np.median([x["prediction_at_service_ms"] for x in d])),
                                 gt_ready_p50=float(np.median([x["gt_ready_ms"] for x in d])),
                                 control_reward=float(np.mean([x["control_reward"] for x in inc])),
                                 candidate_reward=float(np.mean([x["candidate_reward"] for x in inc])))
        chains = {}
        for q in (3000, 7000, 8000):
            st = lat.stratum(11, q)
            chains[str(q)] = dict(n=len(st), timely=float(np.mean([(c.instants["ue_receipt"] or 1e9) <= 170 for c in st])),
                                  by_source={s: float(np.mean([(c.instants["ue_receipt"] or 1e9) <= 170 for c in st if c.source_id == s]))
                                             for s in sorted({c.source_id for c in st})})
        result[split] = dict(environment_fixed_m11=cells, measured_chains=chains, supported_scenes=cat.scene_count)
    Path(out).write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
