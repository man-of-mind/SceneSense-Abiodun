"""Action panel, old versus candidate (same TRAIN_GCV2 pool, same draws, unchanged quality and reward terms).

    python -m rl_agent.splitfusion_run5_uplink_lowmcs_v1.panel --run <dir>        # writes <dir>/panel_01

Method of ``splitfusion_run5_channel_response_v1_analysis.env_panel`` (12 nearest FIT grid scenes by camera SI /
radar P40 = TRAINING examples, NOT the live scene; 5 nearest base chains x 40 lateness uniforms; score-channel
carry-in idle; reward D * Y * (0.7 + 0.3 Q_seg), timeout 0), applied to both latency models.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose
from rl_agent.splitfusion_run5_channel_response_v1_analysis import env_panel as EP
from rl_agent.splitfusion_run5_full_v1 import collector as FC
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import evaluate as EV
from . import model as M
from . import plan as P

FADE_DECISIONS = (120, 130, 138, 294, 299, 302)
CONTROL_DECISIONS = (144, 145)              # first two warm P1 decisions with MCS >= 20 and zero backlog
PANEL_Q = (4000, 5000, 6000, 7000, 8000, 9000, 9400)
SHIFTS = (-10.0, -5.0, 0.0, 5.0, 10.0)
JOINED = P.ROOT / "experiments/splitfusion_run5_channel_response_v1/analysis_01/DECISIONS_JOINED.jsonl"


def run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    out = run_dir / "panel_01"
    out.mkdir(exist_ok=False)
    correction = M.LowMcsCorrectionV1.load(run_dir / "fit_01/CANDIDATE.json")
    _, models = EV.models(correction)
    models = {k: models[k] for k in ("old", "candidate")}
    catalog = FC.load_shared_sources()["catalog"]
    grid = list(SS.Q_E4_GRID)
    rows = {json.loads(l)["decision_index"]: json.loads(l) for l in JOINED.read_text().splitlines() if l.strip()}
    desc = {k: catalog.scene_descriptors(k) for k in catalog.keys}
    si_sd = float(np.std([v[0] for v in desc.values()])) or 1.0
    p40_sd = float(np.std([v[1] for v in desc.values()])) or 1.0
    results = {}
    for d in FADE_DECISIONS + CONTROL_DECISIONS:
        x = rows[d]
        si, p40, mcs = float(x["camera_si"]), float(x["radar_p40"]), int(x["prior_ul_mcs"])
        backlog = float(x.get("pre_action_rlc_backlog_bytes") or 0)
        actions = [("RECORDED", 11, int(x["q_e4"]))] + [(f"q{q}", 11, q) for q in PANEL_Q]
        scenes = [k for _, k in sorted((np.hypot((v[0] - si) / si_sd, (v[1] - p40) / p40_sd), k) for k, v in desc.items())
                  if all(catalog.reward_value(k, mode_id=m, q_e4=q) is not None for _, m, q in actions)][:EP.N_SCENES]
        comp = EP.components_rows(catalog.keys, scenes, set(grid))
        rng = random.Random(20261006 + d)
        draws = [(idx, rng.random()) for idx in range(5) for _ in range(EP.N_U)]
        per_model = {}
        for mname, model in models.items():
            ranked = model.ranked(si, p40)
            per = {}
            for name, m, q in actions:
                receipts, values, wires, flags = [], [], [], set()
                for s in scenes:
                    _, _, wire, *_ = catalog.draw(s, mode_id=m, q_e4=q)
                    v = catalog.reward_value(s, mode_id=m, q_e4=q)
                    for idx, u in draws:
                        dist, cid = ranked[idx]
                        mv = model.move(cid, ranked=ranked, mode_id=m, q_e4=q, prior_ul_mcs=mcs, backlog_bytes=backlog,
                                        wire_bytes=int(wire), u=u, distance=dist)
                        flags |= {f for f in mv.floors if f in (M.UNRESOLVED_BACKLOG, M.EXTRAPOLATED_PAYLOAD, M.EXTRAPOLATED_MCS)}
                        receipts.append(compose(mv.vector, carry_in=None).receipt)
                        values.append(v)
                    wires.append(int(wire))
                receipts, values = np.asarray(receipts), np.asarray(values)
                pooled = {}
                for cls in ("vehicle", "person"):
                    el = hits = tp = fp = inter = union = 0.0
                    for s in scenes:
                        for w, c in EP.comp_at(comp, s, q, grid):
                            c = c[cls]
                            el += w * c["eligible"]; hits += w * c["hits"]; tp += w * c["tp"]; fp += w * c["fp"]
                            inter += w * c["inter"]; union += w * c["union"]
                    pooled[cls] = dict(recall_within_tol=hits / el if el else None,
                                       precision=tp / (tp + fp) if tp + fp else None, seg_iou=inter / union if union else None)
                per[name] = dict(q_e4=q, wire_kb_p50=float(np.median(wires)) / 1e3, receipt_p50=float(np.median(receipts)),
                                 value_if_timely=float(values.mean()), quality_APPROXIMATE=pooled, flags=sorted(flags),
                                 shifts={str(sh): dict(p_timely=float((receipts + sh <= 170.0).mean()),
                                                       e_reward=float(((receipts + sh <= 170.0) * values).mean()))
                                         for sh in SHIFTS},
                                 _r=(receipts <= 170.0) * values)
            brng = np.random.default_rng(d)
            for name in per:
                if name != "RECORDED":
                    diff = (per[name]["_r"] - per["RECORDED"]["_r"]).reshape(len(scenes), len(draws)).mean(1)
                    boots = [diff[brng.integers(0, len(scenes), len(scenes))].mean() for _ in range(4000)]
                    per[name]["d_reward_vs_recorded"] = dict(mean=float(diff.mean()), ci95=[float(np.percentile(boots, 2.5)),
                                                                                           float(np.percentile(boots, 97.5))])
            for name in per:
                per[name].pop("_r")
            best = {str(sh): max(per, key=lambda n: per[n]["shifts"][str(sh)]["e_reward"]) for sh in SHIFTS}
            per_model[mname] = dict(actions=per, best_by_shift=best, base_chains=[c for _, c in ranked[:5]])
        results[d] = dict(context=dict(snr_db=x["snr_db"], prior_ul_mcs=mcs, backlog_bytes=backlog, recorded_q=int(x["q_e4"]),
                                       live_payload_bytes=x["payload_bytes"], live_outcome=x["outcome_class"],
                                       live_receipt_ms=x.get("ue_receipt"),
                                       role="CONTROL_HIGH_MCS" if d in CONTROL_DECISIONS else "FADE"),
                          training_scenes=scenes, models=per_model)
    doc = dict(schema="splitfusion.run5_uplink_lowmcs_v1.panel.v1", correction=correction.to_dict(),
               old_binding=models["old"].binding_sha256, candidate_binding=models["candidate"].binding_sha256,
               reward_binding=catalog.reward_binding_sha256, n_scenes=EP.N_SCENES, n_u=EP.N_U, k_chains=5,
               carry_in="none (idle)", results=results,
               claim="environment-consistency check; quality from nearest TRAINING grid scenes (approximate), not the "
                     "live scene; not a live counterfactual")
    (out / "PANEL.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    doc = run(p.parse_args().run)
    for d, r in doc["results"].items():
        c = r["context"]
        print(d, c["role"], "mcs", c["prior_ul_mcs"], "B", c["backlog_bytes"], c["live_outcome"])
        for m, pm in r["models"].items():
            print("  ", m, "best@0", pm["best_by_shift"]["0.0"], {n: (round(a["wire_kb_p50"]), round(a["shifts"]["0.0"]["p_timely"], 2),
                                                                    round(a["shifts"]["0.0"]["e_reward"], 3)) for n, a in pm["actions"].items()})
