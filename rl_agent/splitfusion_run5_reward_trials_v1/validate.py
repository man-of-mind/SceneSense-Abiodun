"""Out-of-sample check of the rebound latency model (offline; before any training).

Base = TRAIN_GCV2 only. Targets: HELDOUT_GCV2 (same session, deterministic held blocks) and RETENTION_V1 (another
session, retention-instrumented). Each target decision is predicted from its own measured context (action, payload,
scene SI/P40, prior UL MCS, RLC backlog) and its measured score-channel carry-in, with DRAWS independent draws; the
predicted receipt distribution is compared with the measured one. Also: in-pool resampling of TRAIN_GCV2.
Nothing here is used to fit anything.

  python3 -m rl_agent.splitfusion_run5_reward_trials_v1.validate experiments/splitfusion_run5_reward_trials_v1/binding_02 \
      experiments/splitfusion_run5_reward_trials_v1/validation_01
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_corrected_latency_env_v1 import collector as CR
from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose

from . import latency as L

DRAWS = 20


def _summary(values):
    a = np.asarray([v for v in values if v is not None], float)
    return dict(n=len(values), p50=float(np.median(a)), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)), timely=float(np.mean([v is not None and v <= 170.0 for v in values])))


def predict(model, targets, seed):
    rng = random.Random(seed)
    out, gated = [], 0
    for c in targets:
        ctx = c.context
        for _ in range(DRAWS):
            d = model.draw(mode_id=ctx.mode_id, q_e4=ctx.q_e4, camera_si=ctx.camera_si, radar_p40=ctx.radar_p40,
                           prior_ul_mcs=ctx.prior_ul_mcs, backlog_bytes=ctx.backlog_bytes, wire_bytes=ctx.wire_bytes,
                           rng=rng)
            gated += "CONFIRM_SEND_GATED" in d.floors
            out.append(compose(d.vector, carry_in=c.service.carry_in).receipt)
    return out, gated


def run(binding_dir: Path, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    transport = CR.load_shared_sources()["model"]
    base = L.PoolLatencyModelV1(binding_dir, transport, pool="TRAIN_GCV2")
    results = {}
    for i, pool in enumerate(("HELDOUT_GCV2", "RETENTION_V1", "TRAIN_GCV2")):
        target = L.PoolLatencyModelV1(binding_dir, transport, pool=pool)
        chains = sorted(target.chains.values(), key=lambda c: c.chain_id)
        predicted, gated = predict(base, chains, seed=2000 + i)
        m, p = _summary([c.instants["ue_receipt"] for c in chains]), _summary(predicted)
        results[pool] = dict(measured=m, predicted=p, timely_error=p["timely"] - m["timely"],
                             p50_error_ms=p["p50"] - m["p50"], p95_error_ms=p["p95"] - m["p95"], gated_draws=gated,
                             role="in-pool resampling (not out-of-sample)" if pool == "TRAIN_GCV2" else "out-of-sample")
    summary = dict(schema="splitfusion.run5_reward_trials.latency_validation.v1", base_pool="TRAIN_GCV2",
                   draws_per_decision=DRAWS, binding_sha256=base.binding_sha256, results=results)
    (out / "LATENCY_VALIDATION.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary


if __name__ == "__main__":
    s = run(Path(sys.argv[1]), Path(sys.argv[2]))
    for pool, r in s["results"].items():
        print(pool, "measured timely %.3f p50 %.1f p95 %.1f | predicted timely %.3f p50 %.1f p95 %.1f | gated %d" % (
            r["measured"]["timely"], r["measured"]["p50"], r["measured"]["p95"], r["predicted"]["timely"],
            r["predicted"]["p50"], r["predicted"]["p95"], r["gated_draws"]))
