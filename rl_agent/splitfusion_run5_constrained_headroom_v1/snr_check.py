"""SNR/MCS state checks and the joint SNR x MCS summary (offline).

1. Feature scaling: 22-D training-environment states (TRAIN, corrected) vs the live FADE capture's recorded states
   (PREVIOUSLY EXAMINED): feature 21 = causal SNR, feature 2 = prior UL MCS; same linear maps required.
2. Profiles in NEW training decisions; SNR lease semantics are the unchanged collector (source dates checked separately).
3. Joint SNR x MCS cells (observed support only, n >= 200) from the retrain evaluation E1-CORRECTED (PREVIOUSLY EXAMINED).
"""
from __future__ import annotations

import collections
import json

import numpy as np

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_run5_channel_response_v1_analysis import actor_sensitivity as AS
from rl_agent.splitfusion_run5_channel_response_v1_analysis.channel_analysis import ROOT as CAPTURE
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as CAMP
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C

from . import metrics as MT

SNR_BINS = (-1e9, 8.0, 12.0, 16.0, 20.0, 1e9)
SNR_LABELS = ("<8 dB", "8-12", "12-16", "16-20", ">=20")


def feature_maps(n=600):
    shared, latency = CAMP.components()
    col = C.TrialRun5CollectorV1(latency=latency, seed=8101, shared_sources=shared, arm="C0")
    tr = []
    for i in range(n):
        f = col.current_state_features()
        tr.append((f[21], col.context.snr_db, f[2], col.context.prior_ul_mcs, col.context.profile))
        col.collect_transition(orch.ModeledActionRequestV1(decision_ordinal=i, mode_id=11, q_e4=7000,
                                                           source="STOCHASTIC_ACTOR", warmup_q_bin_index=None))
    cell = next((CAPTURE / "capture/cells").iterdir())
    spans = AS.load_spans(cell)
    joined = [json.loads(l) for l in open(MT.REG.ROOT / "experiments/splitfusion_run5_channel_response_v1/analysis_01/DECISIONS_JOINED.jsonl")]
    live = [(spans[r["frame_id"]]["features"][21], r["snr_db"], spans[r["frame_id"]]["features"][2], r["prior_ul_mcs"])
            for r in joined if r["frame_id"] in spans and r.get("prior_ul_mcs") is not None]

    def fit(rows):
        x = np.array([r[1] for r in rows]); y = np.array([r[0] for r in rows])
        a, b = np.polyfit(x, y, 1)
        m = np.array([r[3] for r in rows]); fm = np.array([r[2] for r in rows])
        return dict(snr_feature_slope=float(a), snr_feature_intercept=float(b),
                    snr_fit_max_abs_residual=float(np.abs(y - (a * x + b)).max()),
                    mcs_feature_scale=float(np.median(fm[m > 0] / m[m > 0])),
                    mcs_scale_max_dev=float(np.abs(fm - m * np.median(fm[m > 0] / m[m > 0])).max()),
                    snr_range=[float(x.min()), float(x.max())], mcs_range=[int(m.min()), int(m.max())], n=len(rows))
    return dict(training_env=fit(tr), live_fade_PREVIOUSLY_EXAMINED=fit(live),
                training_profiles=dict(collections.Counter(r[4] for r in tr)))


def joint_table(evald: dict, policies) -> dict:
    out = {}
    for p in policies:
        a = evald[(p, "E1", "CORRECTED")] if isinstance(p, str) else p[1]
        name = p if isinstance(p, str) else p[0]
        snr_bin = np.digitize(a["snr"], SNR_BINS[1:-1])
        mg = np.where(a["mcs"] <= 10, 0, np.where(a["mcs"] <= 15, 1, 2))
        cells = {}
        for sb in range(len(SNR_LABELS)):
            for g, gl in enumerate(("MCS 9-10", "MCS 11-15", "MCS >=16")):
                mask = (snr_bin == sb) & (mg == g)
                if mask.sum() < 200:
                    if mask.sum():
                        cells[f"{SNR_LABELS[sb]}|{gl}"] = dict(n=int(mask.sum()), note="n < 200: not summarised")
                    continue
                s = MT.subset(a, mask)
                m = MT.metrics(s)
                cells[f"{SNR_LABELS[sb]}|{gl}"] = dict(n=int(mask.sum()), q_p50=float(np.median(s["q"])),
                                                       wire_p50_kb=m["wire_p50_kb"], timely=m["timely"], P=m["P"],
                                                       vehicle_seg_retention=m["vehicle"]["retention"]["seg"])
        out[name] = cells
    return out
