"""Frozen fitting plan (written before any fit). Every constant the fit and evaluation use is defined here once.

    python -m rl_agent.splitfusion_run5_uplink_lowmcs_v1.plan --out <dir>    # writes FITTING_PLAN.json (create-only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]                                   # abiodun/
WT = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"

SCHEMA = "splitfusion.run5_uplink_lowmcs_v1.plan.v1"

# Session -> (decision rows, service FRAMES, pipeline label, partition). Rows are the existing per-decision boundary
# tables (stamps in ms from action open, one host clock); FRAMES are the exact-GT service records used by
# chains.decompose. FIT = seven FAVORABLE_STABLE sessions; VALIDATION = the FADE_RECOVERY session (held out of the
# fit, but previously inspected: NOT an untouched confirmatory set).
SESSIONS = {
    "pilot_valid02": ("experiments/splitfusion_run5_pilot_live_v1/valid_02/plots/DECISIONS_valid1000.jsonl",
                      "20261006_run5_pilot_live_v1/valid_b1000_02", "pilot actor; archive capacity repair; input v1", "FIT"),
    "pilot_valid03": ("experiments/splitfusion_run5_pilot_live_v1/valid_03/plots/DECISIONS_valid1000_v3.jsonl",
                      "20261006_run5_pilot_live_v1/valid_b1000_03", "pilot actor; + single-buffer input v2", "FIT"),
    "full_valid01": ("experiments/splitfusion_run5_full_live_v1/valid_01/plots/DECISIONS_valid1000_full.jsonl",
                     "20261006_run5_full_live_v1/valid_b1000", "full Run-5 seed-43 actor (deployed)", "FIT"),
    "gc1_control": ("experiments/splitfusion_run5_full_live_gc_v1/pair_01_analysis/control/DECISIONS_control_b1000.jsonl",
                    "20261006_run5_full_live_gc_v1/control", "deployed actor; GC-fix pair control", "FIT"),
    "gc1_candidate": ("experiments/splitfusion_run5_full_live_gc_v1/pair_01_analysis/candidate/DECISIONS_candidate_b1000.jsonl",
                      "20261006_run5_full_live_gc_v1/candidate", "deployed actor; GC fix v1", "FIT"),
    "gcv2": ("experiments/splitfusion_run5_full_live_gc_v2/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
             "20261006_run5_full_live_gc_v2/validation", "deployed actor; gc_v2 (binding_02 base-chain source)", "FIT"),
    "retention": ("experiments/splitfusion_run5_full_live_retention_v1/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
                  "20261006_run5_full_live_retention_v1/validation", "deployed actor; gc_v2 + retention v2 observer", "FIT"),
    "fade": ("experiments/splitfusion_run5_channel_response_v1/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
             "20261006_run5_channel_response_v1/validation", "deployed actor; retention launcher + FADE_RECOVERY trace",
             "VALIDATION"),
}
CELL = {"fade": "a71__fade_recovery"}                          # every other session: a71__favorable_stable

# Baseline environment: the reward-trial gc_v2 binding (binding_02), TRAIN_GCV2 pool, unchanged T_v2b.
BINDING_02 = ROOT / "experiments/splitfusion_run5_reward_trials_v1/binding_02"
BINDING_02_SHA256 = "36ccc3b30f05ba2e91d50ca23eeb3663a72a3ada69666ee3a6e2854a4f38785d"     # LATENCY_EVIDENCE.json
BINDING_01_SHA256 = "48ee984b43609619264fe41c978f17acf44afa576eed1c3e1a4225d99e9e3e3c"     # deployed actor's binding
BASE_POOL = "TRAIN_GCV2"
T_V2B_RELPATH = "rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json"

# Definitions.
TARGET = "T = edge_reassembly_complete - ue_last_send (ms) = chain stage E = T_v2b boundary"
BYTES = "wire = udp_application_bytes(payload_bytes): L2 SFD frame + 8 B per 12,500-B chunk (the env's base-chain layer)"
ELIGIBLE = "warm POLICY_DECISION rows; low-MCS fit region = prior_ul_mcs in [9, 15] and pre_action_rlc_backlog_bytes == 0"
LOW_MCS_MAX = 15
MCS_REF = 16                       # correction is zero at MCS >= 16 (high-MCS behaviour of T_v2b kept exactly)
W0_BYTES = 6242                    # T_v2b fit_support min_wire_bytes: its measured low-MCS anchor (~6.5 KB at MCS 9)
DEADLINE_MS = 170.0

# Candidate model form (fixed before fitting).
MODEL = {
    "mean": "mu_c(B, w, m) = mu_v2b(B, w, m) + kappa * max(0, w - W0)/1e6 * max(0, MCS_REF - m)   if B == 0, else mu_v2b",
    "spread": "lambda(m, B) = lambda_L if (m <= 15 and B == 0) else 1",
    "env_move": "E_new = E_b + (mu_c(dec) - mu_c(b)) + (lambda(dec)/lambda(b) - 1) * (E_b - mu_c(b) - beta_pool); "
                "floors and the T_v2b deadline-head lateness rule clip(1 - p(dec)/p(b), 0, 1) unchanged",
    "kappa_fit": "least squares through the origin of d_i = (T_i - mu_v2b(i)) - bH_session on x_i = max(0, w_i - W0)/1e6 * (16 - m_i), "
                 "low-MCS fit region of the FIT sessions; bH_session = mean (T - mu_v2b) over that session's MCS >= 16, B == 0 decisions",
    "lambda_fit": "(P90 - P10) of low-MCS residuals d_i - kappa x_i  /  (P90 - P10) of session-centred MCS >= 16, B == 0 residuals, FIT sessions",
    "beta_pool": "mean over the base pool of (E_b - mu_c(b)) (keeps the pool's own offset; only the zero-mean residual is scaled)",
    "unsupported": "B > 0 at MCS <= 15: T_v2b unchanged, flagged UNRESOLVED_BACKLOG; payload above the fitted low-MCS maximum: "
                   "linear term continued, flagged EXTRAPOLATED_PAYLOAD; MCS < 9: flagged EXTRAPOLATED_MCS. Flags counted, never silent.",
    "sensitivities": ["kappa with a free intercept (reported, not used)", "mean-only candidate (lambda_L = 1)",
                      "leave-one-session-out kappa over the 7 FIT sessions", "kappa fitted on the VALIDATION session alone (internal)"],
}

EVALUATION = {
    "populations": "VALIDATION (fade) warm decisions with B == 0 and prior MCS <= 15 (primary) and <= 10 (secondary, nested); "
                   "payload terciles = L2 payload_bytes tercile cut points of the MCS <= 15 population, applied to both",
    "transport": "mean error and median |error| by tercile; coverage of the predictive P50/P80/P90 "
                 "(predictive set = mu(dec) + beta + lambda ratio * residuals of the TRAIN_GCV2 pool, deadline-head lateness as a mass at +inf)",
    "composed_own_chain": "each validation decision's own decomposed chain with stage E replaced by the predictive set "
                          "(expectation over the pool and the lateness uniform), composed unchanged with the measured carry-in; "
                          "P(receipt <= 170) by tercile vs measured on-time with Wilson 95% CI",
    "composed_env": "secondary: the full environment move on the 5 nearest TRAIN_GCV2 chains (camera SI / radar P40), carry-in idle",
    "high_mcs_regression": "MCS >= 16: E identical to the old model whenever the base chain is also high-MCS (test); "
                           "HELDOUT_GCV2 / RETENTION_V1 composed timeliness old vs candidate",
    "panel": "env_panel decisions 120/130/138/294/299/302 + the first two warm P1 decisions with MCS >= 20 and B == 0; "
             "mode 11 recorded q and q5000..q9000; common random numbers; quality = nearest FIT grid scenes (approximate)",
}

ACCEPTANCE = {
    "A1_tercile_timeliness": "candidate own-chain P(timely) inside the measured Wilson 95% CI in all three terciles (MCS <= 15)",
    "A2_contrast": "candidate (low - high tercile) inside the measured contrast's 95% Newcombe interval (MCS <= 15)",
    "A3_transport": "|mean error| <= 10 ms in every tercile and P80 coverage inside the binomial 95% acceptance region for p = 0.8",
    "A4_stability": "every leave-one-FIT-session-out kappa within [0.5, 1.5] x kappa_FIT",
    "A5_high_mcs": "exact identity for high-MCS base and decision; HELDOUT_GCV2 composed timeliness change <= 0.01",
    "A6_support": "training decisions (reward-trial runs, binding_02) flagged UNRESOLVED or EXTRAPOLATED <= 5%",
    "need": "old model fails A1 or A2 (otherwise no correction is needed)",
}
VERDICT_RULE = {
    "READY_FOR_ONE_TARGETED_RETRAINING_CANDIDATE": "A1-A6 pass and the old model fails A1 or A2",
    "REQUIRES_ONE_MISSING_MEASUREMENT": "A3 and A5 pass, but A1/A2/A4/A6 fail for a reason traceable to one identified data gap",
    "REJECTED": "candidate no better than the old model on A1/A2, or A5 fails",
}
STOP = "stop after the assessment; no retraining, no live run, no reward/quality change"


def document() -> dict:
    return dict(schema=SCHEMA, sessions={k: dict(rows=v[0], frames=f"{v[1]}/capture/cells/{CELL.get(k, 'a71__favorable_stable')}"
                                                  "/numeric_gt/service/FRAMES.jsonl", pipeline=v[2], partition=v[3])
                                         for k, v in SESSIONS.items()},
                baseline=dict(binding_02_sha256=BINDING_02_SHA256, pool=BASE_POOL, t_v2b=T_V2B_RELPATH,
                              deployed_actor_binding_01_sha256=BINDING_01_SHA256),
                target=TARGET, bytes=BYTES, eligible=ELIGIBLE, low_mcs_max=LOW_MCS_MAX, mcs_ref=MCS_REF, w0_bytes=W0_BYTES,
                deadline_ms=DEADLINE_MS, model=MODEL, evaluation=EVALUATION, acceptance=ACCEPTANCE,
                verdict_rule=VERDICT_RULE, stop=STOP)


def canonical(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    out = Path(p.parse_args().out)
    doc = document()
    path = out / "FITTING_PLAN.json"
    with open(path, "x") as f:
        json.dump(dict(doc, plan_sha256=canonical(doc)), f, indent=1, sort_keys=True)
    print(path, canonical(doc))
