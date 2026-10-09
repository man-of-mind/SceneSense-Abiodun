"""Environment-consistency action panel (CPU only): does the full Run-5 TRAINING environment rank the recorded live
action below supported mode-11 alternatives at the fade decisions?

For each representative live decision the training kernel's own components are used unchanged:
  latency  splitfusion_run5_full_v1.latency.FullSupportLatencyModelV1.move (k=5 nearest measured base chains by the
           decision's recorded camera SI / radar P40; channel = recorded prior UL MCS and RLC backlog) + chains.compose;
  quality  splitfusion_run5_full_v1.quality.FullSceneCatalogV1 (exact FIT quality grid; off-grid q linear) on the
           N nearest FIT grid scenes by the same SI / P40 descriptors (clearly identified TRAINING examples, NOT the
           live scene), reward = D * Y * (0.7 + 0.3 Q_seg), timeout 0, undefined excluded.
Common random numbers: every action sees the same (scene, base chain, uplink uniform) triples. Score-channel carry-in
is set to none (idle) for every action. Latency robustness: the composed receipt is shifted by +-5 / +-10 ms.
This is an environment-consistency check, not a live counterfactual and not proof that any timeout was avoidable.
"""
import argparse
import json
import random
import sqlite3
from pathlib import Path

import numpy as np

from .channel_analysis import MAIN, sha

DECISIONS = (120, 130, 138, 294, 299, 302)
PANEL_Q = (5000, 6000, 7000, 8000, 9000)
N_SCENES, N_U, SHIFTS = 12, 40, (-10.0, -5.0, 0.0, 5.0, 10.0)


def components_rows(catalog_keys, scenes, grid_q):
    from rl_agent.ue_production_queue_capture_v1 import contract as V1
    from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
    from rl_agent.ue_production_transport_model_v2 import scene_source as SS
    from rl_agent.splitfusion_run5_restricted_pilot_v1.quality import _components
    want = set(scenes)
    out = {}
    con = sqlite3.connect(f"file:{C2.ROOT / V1.PAYLOAD_AUTHORITY_RELPATH}?mode=ro&immutable=1", uri=True)
    try:
        for (blob,) in con.execute("SELECT row_json FROM quality_rows WHERE grid_split = ?", (SS.FIT_SPLIT,)):
            row = json.loads(blob)
            key = f"{row['episode_id']}|{row['sample_id']}|{row['frame_id']}"
            if key in want and int(row["mode_id"]) == 11 and int(row["q_e4"]) in grid_q:
                out[(key, int(row["q_e4"]))] = _components(row)
    finally:
        con.close()
    return out


def comp_at(rows, scene, q, grid):
    """Linear (registered off-anchor rule) interpolation of pooled-able counts between bracketing grid rows."""
    import bisect
    if q in grid:
        return [(1.0, rows[(scene, q)])]
    i = bisect.bisect_left(grid, q)
    lo, hi = grid[i - 1], grid[i]
    w = (q - lo) / (hi - lo)
    return [(1 - w, rows[(scene, lo)]), (w, rows[(scene, hi)])]


def run(out):
    from rl_agent.splitfusion_run5_full_v1 import campaign as CP
    from rl_agent.splitfusion_corrected_latency_env_v1.chains import compose
    from rl_agent.ue_production_transport_model_v2 import scene_source as SS
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    shared, latency = CP.components()
    catalog = shared["catalog"]
    grid = list(SS.Q_E4_GRID)
    rows = {json.loads(l)["decision_index"]: json.loads(l) for l in
            open(MAIN / "experiments/splitfusion_run5_channel_response_v1/analysis_01/DECISIONS_JOINED.jsonl")}
    desc = {k: catalog.scene_descriptors(k) for k in catalog.keys}
    si_sd = float(np.std([v[0] for v in desc.values()])) or 1.0
    p40_sd = float(np.std([v[1] for v in desc.values()])) or 1.0
    results = {}
    for d in DECISIONS:
        x = rows[d]
        si, p40, mcs = float(x["camera_si"]), float(x["radar_p40"]), int(x["prior_ul_mcs"])
        backlog = float(x.get("pre_action_rlc_backlog_bytes") or 0)
        rec_q = int(x["q_e4"])
        actions = [("RECORDED", 11, rec_q)] + [(f"q{q}", 11, q) for q in PANEL_Q]
        scenes = [k for _, k in sorted((np.hypot((v[0] - si) / si_sd, (v[1] - p40) / p40_sd), k) for k, v in desc.items())
                  if all(catalog.reward_value(k, mode_id=m, q_e4=q) is not None for _, m, q in actions)][:N_SCENES]
        comp = components_rows(catalog.keys, scenes, set(grid))
        ranked = latency.ranked(si, p40)
        rng = random.Random(20261006 + d)
        draws = [(idx, rng.random()) for idx in range(5) for _ in range(N_U)]
        per = {}
        for name, m, q in actions:
            receipts, values, wires = [], [], []
            for s in scenes:
                _, _, wire, *_ = catalog.draw(s, mode_id=m, q_e4=q)
                v = catalog.reward_value(s, mode_id=m, q_e4=q)
                for idx, u in draws:
                    dist, cid = ranked[idx]
                    mv = latency.move(cid, ranked=ranked, mode_id=m, q_e4=q, prior_ul_mcs=mcs, backlog_bytes=backlog,
                                      wire_bytes=int(wire), u=u, distance=dist)
                    receipts.append(compose(mv.vector, carry_in=None).receipt)
                    values.append(v)
                wires.append(int(wire))
            receipts, values = np.asarray(receipts), np.asarray(values)
            pooled = {}
            for cls in ("vehicle", "person"):
                el = hits = tp = fp = inter = union = 0.0
                for s in scenes:
                    for w, c in comp_at(comp, s, q, grid):
                        c = c[cls]
                        el += w * c["eligible"]; hits += w * c["hits"]; tp += w * c["tp"]; fp += w * c["fp"]
                        inter += w * c["inter"]; union += w * c["union"]
                pooled[cls] = dict(recall_within_tol=hits / el if el else None, precision=tp / (tp + fp) if tp + fp else None,
                                   seg_iou=inter / union if union else None, eligible=el)
            shifts = {}
            for sh in SHIFTS:
                timely = receipts + sh <= 170.0
                shifts[sh] = dict(p_timely=float(timely.mean()), e_reward=float((timely * values).mean()))
            per[name] = dict(mode=m, q_e4=q, wire_bytes_p50=float(np.median(wires)), receipt_p50=float(np.median(receipts)),
                             value_if_timely=float(values.mean()), quality=pooled, shifts=shifts,
                             _r=(receipts <= 170.0) * values)
        # paired bootstrap over scenes for E[r](alt) - E[r](recorded)
        brng = np.random.default_rng(d)
        per_scene = len(draws)
        for name in per:
            if name == "RECORDED":
                continue
            diff = (per[name]["_r"] - per["RECORDED"]["_r"]).reshape(len(scenes), per_scene).mean(1)
            boots = [diff[brng.integers(0, len(scenes), len(scenes))].mean() for _ in range(4000)]
            per[name]["d_reward_vs_recorded"] = dict(mean=float(diff.mean()), ci=[float(np.percentile(boots, 2.5)),
                                                                                  float(np.percentile(boots, 97.5))])
        for name in per:
            per[name].pop("_r")
        best = {str(sh): max(per, key=lambda n: per[n]["shifts"][sh]["e_reward"]) for sh in SHIFTS}
        results[d] = dict(context=dict(snr_db=x["snr_db"], prior_ul_mcs=mcs, backlog_bytes=backlog, camera_si=si,
                                       radar_p40=p40, recorded_q=rec_q, live_outcome=x["outcome_class"],
                                       live_receipt_ms=x.get("ue_receipt"), live_payload_bytes=x["payload_bytes"]),
                          training_scenes=scenes, base_chains=[c for _, c in ranked[:5]], actions=per, best_by_shift=best)
    doc = dict(schema="splitfusion.run5_channel_response_v1.env_panel.v1", latency_binding=latency.binding_sha256,
               reward_binding=catalog.reward_binding_sha256, n_scenes=N_SCENES, n_u=N_U, k_chains=5, shifts_ms=SHIFTS,
               carry_in="none (idle score channel) for every action", results=results,
               claim="environment-consistency check on training components; quality from nearest TRAINING grid scenes, "
                     "not the live scene; no live counterfactual")
    (out / "ENV_PANEL.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"ENV_PANEL.json": sha(out / "ENV_PANEL.json")}, indent=1))
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
