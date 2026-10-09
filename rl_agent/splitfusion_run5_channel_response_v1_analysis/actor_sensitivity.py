"""Bounded CPU-only same-state sensitivity of the frozen deployed Run-5 actor (seed 43 u10000) on recorded states.

Recorded 22-D states and decided (mode, q_e4) come from the capture's ``agent.actor`` spans. Every target decision is
first reproduced exactly (mode, q_e4, state digest, heads digest). Then ONE feature group at a time (or a stated
combination) is replaced by the same group of a recorded DONOR state; everything else stays the target's recorded state.
Groups are copied as whole blocks, so encodings stay valid (one-hot previous mode, prev_present consistency).

This is an action comparison on fixed states. It is not a closed-loop rollout and implies NO payload or latency change
(payload needs re-encoding the same tensor; latency needs a measurement).

  python3 -m rl_agent.splitfusion_run5_channel_response_v1_analysis.actor_sensitivity --out <dir>
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .channel_analysis import ROOT, MAIN, sha

GROUPS = dict(snr=[21], mcs=[2], channel=[2, 21], scene=[0, 1], history=list(range(4, 21)),
              scene_history=[0, 1] + list(range(4, 21)))
TESTS = ("snr", "mcs", "channel", "scene", "history", "scene_history")
WINDOWS = (
    dict(name="W1_first_fade", decisions=(118, 142), fade_step=303,
         donors=dict(PRE=119, REDUCED=138),
         note="PRE = last decision before ADV step 303; REDUCED = first decision of the eventual sustained reduction"),
    dict(name="W2_T_DEG", decisions=(290, 306), fade_step=746,
         donors=dict(PRE=293, REDUCED=302),
         note="PRE = last decision before ADV step 746; REDUCED = first decision of the eventual sustained reduction"))
SNR_TOL, MCS_TOL = 1.0, 2     # empirical joint-support neighbourhood (dB, MCS index) for the swapped channel pair


def load_spans(cell):
    spans = {}
    for line in open(cell / "profiler/ue.profile.jsonl"):
        r = json.loads(line)
        if r.get("r") == "span" and r.get("stage") == "agent.actor" and r.get("ok"):
            spans[int(r["frame_id"])] = r
    return spans


def query(actor, values):
    """Deployed rule (unchanged act_on_vector) + mode probabilities / logit margin from the same heads."""
    decision = actor.act_on_vector(tuple(float(v) for v in values))
    state = torch.tensor((tuple(float(v) for v in values),), dtype=torch.float32)
    with torch.inference_mode():
        heads = actor.module(state)
    logits = heads.logits[0].double().numpy()
    prob = np.exp(logits - logits.max())
    prob /= prob.sum()
    order = np.argsort(-logits)
    return dict(mode=decision.mode_id if hasattr(decision, "mode_id") else int(order[0]), q_e4=int(decision.q_e4),
                p_top=float(prob[order[0]]), p_mode11=float(prob[11]), top=int(order[0]), second=int(order[1]),
                margin=float(logits[order[0]] - logits[order[1]]), decision=decision)


def run(out):
    from rl_agent.splitfusion_run5_full_live_v1.actor import load_registered_actor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cell = next((ROOT / "capture/cells").iterdir())
    spans = load_spans(cell)
    joined = MAIN / "experiments/splitfusion_run5_channel_response_v1/analysis_01/DECISIONS_JOINED.jsonl"
    rows = {}
    for line in open(joined):
        x = json.loads(line)
        rows[int(x["decision_index"])] = x
    actor = load_registered_actor()
    fields = {int(i): f for i, f in enumerate(json.loads(next(
        l for l in open(cell / "profiler/ue.profile.jsonl") if '"policy_runtime_identity"' in l))["binding"]["feature_order"])}
    # MCS normalisation recovered from recorded pairs (raw prior_ul_mcs vs feature 2)
    pairs = [(rows[d]["prior_ul_mcs"], spans[rows[d]["frame_id"]]["features"][2]) for d in rows
             if rows[d].get("prior_ul_mcs") is not None and rows[d]["frame_id"] in spans]
    mcs_scale = float(np.median([m / f for m, f in pairs if f > 0]))
    snr_pairs = [(rows[d]["snr_db"], spans[rows[d]["frame_id"]]["features"][21]) for d in rows if rows[d]["frame_id"] in spans]
    a, b = np.polyfit([f for _, f in snr_pairs], [s for s, _ in snr_pairs], 1)       # snr_db = a * feature + b
    support = [(rows[d]["snr_db"], rows[d]["prior_ul_mcs"]) for d in rows if rows[d].get("prior_ul_mcs") is not None]

    def supported(vec):
        snr_db, mcs = a * vec[21] + b, vec[2] * mcs_scale
        return any(abs(s - snr_db) <= SNR_TOL and abs(m - mcs) <= MCS_TOL for s, m in support)

    reproduction, results = [], []
    for w in WINDOWS:
        donors = {k: list(spans[rows[v]["frame_id"]]["features"]) for k, v in w["donors"].items()}
        for d in range(w["decisions"][0], w["decisions"][1] + 1):
            x = rows[d]
            span = spans[x["frame_id"]]
            vec = list(span["features"])
            base = query(actor, vec)
            ok = dict(mode=base["decision"].mode_id == span["decided_mode_id"] == x["mode"],
                      q=base["q_e4"] == span["decided_q_e4"] == x["q_e4"],
                      state_sha256=base["decision"].state_sha256 == span["state_sha256"],
                      heads_sha256=base["decision"].heads_sha256 == span["heads_sha256"])
            reproduction.append(dict(window=w["name"], decision=d, frame_id=x["frame_id"], **ok))
            for donor_name, donor in donors.items():
                for test in TESTS:
                    cf = list(vec)
                    for i in GROUPS[test]:
                        cf[i] = donor[i]
                    identical = cf == vec
                    r = query(actor, cf)
                    results.append(dict(
                        window=w["name"], decision=d, step=x["step"], state=x["state"], donor=donor_name,
                        donor_decision=w["donors"][donor_name], test=test, identical_to_recorded=identical,
                        channel_pair_supported=supported(cf) if test in ("snr", "mcs", "channel") else None,
                        mode=base["mode"], q=base["q_e4"], cf_mode=r["mode"], cf_q=r["q_e4"], dq=r["q_e4"] - base["q_e4"],
                        p11=base["p_mode11"], cf_p11=r["p_mode11"], margin=base["margin"], cf_margin=r["margin"],
                        second=base["second"], cf_second=r["second"]))
    with open(out / "SENSITIVITY_ROWS.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for w in WINDOWS:
        for donor_name in w["donors"]:
            for test in TESTS:
                rs = [r for r in results if r["window"] == w["name"] and r["donor"] == donor_name and r["test"] == test]
                dq = np.asarray([r["dq"] for r in rs])
                summary.append(dict(window=w["name"], donor=f"{donor_name} (dec {w['donors'][donor_name]})", test=test,
                                    n=len(rs), dq_median=float(np.median(dq)), dq_min=int(dq.min()), dq_max=int(dq.max()),
                                    abs_dq_ge_500=int(np.sum(np.abs(dq) >= 500)),
                                    mode_changes=sum(r["cf_mode"] != r["mode"] for r in rs),
                                    unsupported=sum(r["channel_pair_supported"] is False for r in rs),
                                    p11_min=float(min(r["cf_p11"] for r in rs)),
                                    margin_min=float(min(r["cf_margin"] for r in rs))))
    doc = dict(schema="splitfusion.run5_channel_response_v1.actor_sensitivity.v1",
               actor_boundary_sha256=actor.boundary_sha256, features=fields, groups=GROUPS, windows=list(WINDOWS),
               reproduction=dict(n=len(reproduction), all_exact=all(all(v for k, v in r.items() if k in
                                 ("mode", "q", "state_sha256", "heads_sha256")) for r in reproduction), rows=reproduction),
               base_mode_probability=dict(p11_min=float(min(r["p11"] for r in results)),
                                          margin_min=float(min(r["margin"] for r in results))),
               support_rule=f"swapped (SNR, MCS) pair has a recorded neighbour in this run within {SNR_TOL} dB and "
                            f"{MCS_TOL} MCS indices; SNR feature->dB fit a={a:.4f}, b={b:.4f}; MCS scale {mcs_scale:.4f}",
               summary=summary, joined_sha256=sha(joined),
               claim="same-state action comparison only; no payload/latency inferred; not a closed-loop rollout")
    (out / "SENSITIVITY.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({p.name: sha(p) for p in sorted(out.iterdir())
                                                         if p.name != "OUTPUT_SHA256.json"}, indent=1))
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    d = run(parser.parse_args().out)
    print(json.dumps(dict(reproduced=d["reproduction"]["n"], all_exact=d["reproduction"]["all_exact"]), sort_keys=True))
