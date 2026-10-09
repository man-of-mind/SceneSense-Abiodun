"""Leave-one-arm-out check of the composed latency model on the ten measured fixed-action screen arms (offline).

For each held-out arm, the base pool is the measured chains of the OTHER nine arms (same runtime as the held-out
arm, before input v2); every held-out decision is predicted from its own measured context (scene SI/P40, prior UL
MCS, RLC backlog, payload) and its measured score-channel carry-in, with ``K`` independent draws, and the predicted
receipt distribution is compared with the measured one. A second, in-pool check resamples the deployed base pool
(valid_03) for its own decision contexts. Nothing here changes the training environment.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose

from . import evidence as EV, latency as L

DRAWS = 20


def _summary(values):
    a = np.asarray([v for v in values if v is not None], float)
    return dict(n=len(values), p50=float(np.median(a)), p90=float(np.percentile(a, 90)),
                timely=float(np.mean([v is not None and v <= 170.0 for v in values])))


def predict(model: L.FullSupportLatencyModelV1, held, seed: int):
    rng = random.Random(seed)
    out = []
    for c in held:
        ctx = c.context
        for _ in range(DRAWS):
            d = model.draw(mode_id=ctx.mode_id, q_e4=ctx.q_e4, camera_si=ctx.camera_si, radar_p40=ctx.radar_p40,
                           prior_ul_mcs=ctx.prior_ul_mcs, backlog_bytes=ctx.backlog_bytes, wire_bytes=ctx.wire_bytes,
                           rng=rng)
            out.append(compose(d.vector, carry_in=c.service.carry_in).receipt)
    return out


def run(binding_dir: Path, out: Path, transport) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    document = json.loads((Path(binding_dir) / "LATENCY_EVIDENCE.json").read_text())
    arms = {}
    for source, entry in zip(EV.VALIDATION_SOURCES, document["validation_sources"]):
        chains, _ = EV.load_chains(source, rows_entry=entry["rows"], frames_entry=entry["frames"])
        vectors, _ = EV.decompose_all(chains)
        arms[source["source_id"]] = ([c for c in chains if c.chain_id in vectors], vectors)
    results = {}
    for i, (held_id, (held, held_vectors)) in enumerate(sorted(arms.items())):
        pool_chains = [c for k, (cs, _) in arms.items() if k != held_id for c in cs]
        pool_vectors = {cid: v for k, (_, vs) in arms.items() if k != held_id for cid, v in vs.items()}
        model = L.FullSupportLatencyModelV1(binding_dir, transport, base_override=(pool_chains, pool_vectors))
        measured = [c.instants["ue_receipt"] for c in held]
        predicted = predict(model, held, seed=1000 + i)
        m, p = _summary(measured), _summary(predicted)
        action = (held[0].context.mode_id, held[0].context.q_e4)
        results[held_id] = dict(action=list(action), measured=m, predicted=p,
                                timely_error=p["timely"] - m["timely"], p50_error_ms=p["p50"] - m["p50"])
    base = L.FullSupportLatencyModelV1(binding_dir, transport)
    own = sorted(base.chains.values(), key=lambda c: c.chain_id)
    in_pool = dict(measured=_summary([c.instants["ue_receipt"] for c in own]), predicted=_summary(predict(base, own, 7)))
    summary = dict(
        schema="splitfusion.run5_full.latency_validation.v1", draws_per_decision=DRAWS, leave_one_arm_out=results,
        mean_abs_timely_error=float(np.mean([abs(r["timely_error"]) for r in results.values()])),
        mean_abs_p50_error_ms=float(np.mean([abs(r["p50_error_ms"]) for r in results.values()])),
        in_pool_resampling=in_pool,
        note="offline check on existing measurements; the screen arms predate input v2 (B about 3-8 ms longer)")
    (out / "LATENCY_VALIDATION.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary


if __name__ == "__main__":
    import sys
    from rl_agent.splitfusion_run5_full_v1 import collector as C
    shared = C.load_shared_sources()
    s = run(Path(sys.argv[1]), Path(sys.argv[2]), shared["model"])
    for k, r in sorted(s["leave_one_arm_out"].items()):
        print(k, r["action"], "measured timely %.2f p50 %.1f | predicted timely %.2f p50 %.1f" % (
            r["measured"]["timely"], r["measured"]["p50"], r["predicted"]["timely"], r["predicted"]["p50"]))
    print("MAE timely %.3f  MAE p50 %.2f ms" % (s["mean_abs_timely_error"], s["mean_abs_p50_error_ms"]))
    print("in-pool", s["in_pool_resampling"])
