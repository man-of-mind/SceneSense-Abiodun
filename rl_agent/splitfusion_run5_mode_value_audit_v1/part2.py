"""Part 2: the frozen deployed actor's selection mechanism on RECORDED live states (CPU inference only).

Sessions (recorded states, deployed actor bb9b0d25): gc_v2 validation (FAVORABLE_STABLE, latest pipeline) and the
FADE_RECOVERY channel-response capture (PREVIOUSLY EXAMINED, diagnostic). For every decision: reproduce the recorded
action, then log the 12 mode probabilities and the 12 mode-conditional deterministic q (argmax deployment = argmax mode at
its conditional mean; training exploration = categorical mode sample + Gaussian q). Observed outcomes are the recorded
ones; any alternative-mode payload/quality is a MODELLED lookup in the FIT grid (nearest grid q), never an outcome.

    env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python3 -m rl_agent.splitfusion_run5_mode_value_audit_v1.part2 --out <ABS dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import quantize_q_e4
from rl_agent.splitfusion_run5_full_live_v1 import actor as LA
from rl_agent.splitfusion_run5_full_v1.support import BOUNDS, in_support

ABI = Path(__file__).resolve().parents[2]
PROF = ABI.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
SESSIONS = {
    "FAVORABLE_gc_v2": (PROF / "20261006_run5_full_live_gc_v2/validation/capture/cells/a71__favorable_stable",
                        ABI / "experiments/splitfusion_run5_full_live_gc_v2/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl"),
    "FADE_channel_response_DIAGNOSTIC": (PROF / "20261006_run5_channel_response_v1/validation/capture/cells/a71__fade_recovery",
                                         ABI / "experiments/splitfusion_run5_channel_response_v1/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl"),
}
GRID = ABI / "experiments/splitfusion_run5_env_tradeoff_review_v1/20261007_v1/grid_mode_panel.json"


def grid_lookup():
    g = json.loads(GRID.read_text())["cells"]
    by = {}
    for c in g:
        if c["in_support"]:
            by.setdefault(c["mode"], []).append(c)
    return lambda m, q: min(by[m], key=lambda c: abs(c["q_e4"] - q))


def main(out: Path):
    actor = LA.load_registered_actor()


    module = actor.module
    low, high = module.active_q_e4_bounds()
    support = [(int(a), int(b)) for a, b in zip(low, high)]
    assert tuple(support) == BOUNDS and module.logit_head.out_features == 12
    look = grid_lookup()
    res = dict(actor_weights_sha256=LA.WEIGHTS_SHA256, support_q_e4_bounds={m: support[m] for m in range(12)},
               note="probs = softmax(logits); qm = deterministic conditional-mean q_e4 per mode on that mode's support",
               sessions={})
    for name, (cell, dec_path) in SESSIONS.items():
        spans = {}
        for line in open(cell / "profiler/ue.profile.jsonl"):
            r = json.loads(line)
            if r.get("r") == "span" and r.get("stage") == "agent.actor" and r.get("ok"):
                spans[int(r["frame_id"])] = r
        rows = [json.loads(l) for l in open(dec_path) if l.strip()]
        recs = []
        for x in rows:
            sp = spans.get(int(x["frame_id"]))
            if sp is None:
                continue
            st = torch.tensor((tuple(float(v) for v in sp["features"]),), dtype=torch.float32)
            with torch.inference_mode():
                h = module(st)
                p = torch.softmax(h.logits.double(), -1)[0].numpy()
                qm = quantize_q_e4(module._map_pre_squash_to_q(h.mean[0], mode_index=torch.arange(12))).numpy()
                std = h.log_std[0].exp().numpy()
            m = int(np.argmax(p))
            rec_mode, rec_q = int(sp["decided_mode_id"]), int(sp["decided_q_e4"])
            order = np.argsort(-p)
            ru = int(order[1])
            g11, gru = look(11, int(qm[11])), look(ru, int(qm[ru]))
            recs.append(dict(decision=int(x["decision_index"]), reproduced=(m, int(qm[m])) == (rec_mode, rec_q),
                             mode=rec_mode, q=rec_q, payload_kb=x["payload_bytes"] / 1e3, outcome=x["outcome_class"],
                             snr_db=x.get("snr_db"), prior_ul_mcs=x.get("prior_ul_mcs"),
                             backlog_bytes=x.get("pre_action_rlc_backlog_bytes"), camera_si=x.get("camera_si"),
                             radar_p40=x.get("radar_p40"), p=[float(v) for v in p], qm=[int(v) for v in qm],
                             std_pre_squash=[float(v) for v in std], runner_up=ru, p_runner_up=float(p[ru]),
                             modelled_grid_m11=dict(q=g11["q_e4"], wire_p50_kb=g11["wire_p50_kb"], value=g11["value_mean"]),
                             modelled_grid_runner_up=dict(mode=ru, q=gru["q_e4"], wire_p50_kb=gru["wire_p50_kb"], value=gru["value_mean"])))
        P = np.array([r["p"] for r in recs])
        Q = np.array([r["qm"] for r in recs])
        ru = np.array([r["runner_up"] for r in recs])
        summ = dict(n=len(recs), reproduced=sum(r["reproduced"] for r in recs),
                    argmax_mode_counts={str(k): int(v) for k, v in zip(*np.unique(P.argmax(1), return_counts=True))},
                    p11_quantiles={k: float(np.quantile(P[:, 11], v)) for k, v in (("min", 0), ("p5", .05), ("p50", .5), ("p95", .95))},
                    expected_non11_share_if_sampled=float((1 - P[:, 11]).mean()),
                    mean_prob_by_mode={str(m): float(P[:, m].mean()) for m in range(12)},
                    runner_up_counts={str(k): int(v) for k, v in zip(*np.unique(ru, return_counts=True))},
                    decisions_p11_below_0p6=int((P[:, 11] < 0.6).sum()),
                    qm_p50_by_mode={str(m): float(np.median(Q[:, m])) for m in range(12)},
                    q11_quantiles={k: float(np.quantile(Q[:, 11], v)) for k, v in (("p5", .05), ("p50", .5), ("p95", .95))})
        # Lowest-p11 decisions as the deterministic "closest call" set (small, fixed rule; no cherry-picking)
        closest = sorted(recs, key=lambda r: r["p"][11])[:8]
        res["sessions"][name] = dict(summary=summ, closest_calls=[{k: v for k, v in r.items() if k not in ("p", "qm", "std_pre_squash")}
                                                                  | dict(p11=r["p"][11], qm_runner_up=r["qm"][r["runner_up"]], qm11=r["qm"][11])
                                                                  for r in closest])
        with open(out / f"part2_trace_{name}.jsonl", "x") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    with open(out / "PART2_ACTOR_MECHANISM.json", "x") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v["summary"] for k, v in res["sessions"].items()}, indent=1, default=float))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    main(p.parse_args().out)
