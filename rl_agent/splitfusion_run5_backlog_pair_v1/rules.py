"""Gate rule and training registration, written BEFORE any gate result or training (create-only documents)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
TRANSITION_RUN = ROOT / "experiments/splitfusion_run5_backlog_transition_v1/20261007_v1"
FIT_JSON = TRANSITION_RUN / "fit_01/FIT.json"
TIMING_JSON = TRANSITION_RUN / "timing_01/CHAIN_TIMING.json"
CONTROL_REFERENCE = ROOT / "experiments/splitfusion_run5_lowmcs_retrain_v1/runs_01/NEW_seed17/RUN_RESULT.json"
LIVE_TIMELY_AT_BPOS = dict(k=13, n=36, value=13 / 36, source="backlog_transition tv2b_check_01: FIT sessions, B > 1 KB")

GATE = dict(
    schema="splitfusion.run5_backlog_pair.gate_rule.v1",
    environment="training environment: TRAIN_GCV2 pool of binding_02 + frozen low-MCS correction, fit scene split, C0 "
                "(standard) reward; actions m11 q3000 / q5000 / q7000 (~176 / 129 / 81 KB), fixed",
    variants=dict(OLD="existing backlog transition, unchanged transport penalty",
                  NEW_A="corrected backlog transition, unchanged transport penalty",
                  NEW_B="corrected backlog transition, DIAGNOSTIC: backlog's additional forced-late probability removed "
                        "(p_late uses p_on_time at zero backlog; conditional latency, payload/MCS effects, queue, reward "
                        "and quality unchanged); never a calibrated model, never trained on"),
    seeds=[9017, 9029, 9043], decisions_per_seed=2400, crn="identical seeds; scene, held-scene, channel and chain RNG "
                                                          "streams are action-independent",
    quantities=dict(S="mean standard reward per included decision (timeout 0; timely-undefined excluded)",
                    delta="Delta_X(large, small, profile) = [S_X(large) - S_OLD(large)] - [S_X(small) - S_OLD(small)], "
                          "paired by (seed, ordinal); 95% CI by block bootstrap (blocks of 50 decisions, 1000 resamples)",
                    pairs=[["q3000", "q5000"], ["q5000", "q7000"]]),
    rule=dict(G1="exposure: NEW_A decisions with backlog > 1 KB, pooled over actions and seeds, >= 100",
              G2="survival: some (pair, profile, incl. ALL) has Delta_NEW_B <= -0.005 with CI upper bound < 0",
              G3="not penalty-only: pooled (ALL) Delta for (q3000, q5000) has Delta_NEW_A < 0 and "
                 "Delta_NEW_B / Delta_NEW_A >= 0.25",
              verdict="PROCEED iff G1 and G2 and G3; STOP_SPARSE if not G1; otherwise STOP_PENALTY_DEPENDENT"),
    reported_not_gating=["action ranking per profile under OLD / NEW_A / NEW_B",
                         "timely at backlog > 1 KB under NEW_A and NEW_B vs live 13/36 (Wilson CI)",
                         "zero-backlog identity: unit test of the move at backlog 0, plus run-level count of "
                         "zero-backlog decisions whose outcome differs (only through carry-in)"],
)

REGISTRATION = dict(
    schema="splitfusion.run5_backlog_pair.registration.v1",
    arms=dict(CONTROL="low-MCS retrain configuration (C0 reward, binding_02 TRAIN_GCV2, frozen low-MCS correction), "
                      "existing backlog transition (BacklogTransitionMixinV1 disabled)",
              CANDIDATE="identical, with the corrected backlog transition (frozen FIT.json + CHAIN_TIMING.json)"),
    seed=17, updates=10_000, unchanged=["Run-5 architecture", "SAC settings and warm-up", "12 modes and q support",
                                        "22 features incl. causal SNR and prior MCS", "C0 reward and quality",
                                        "170-ms inclusive deadline", "transport penalty (production T_v2b)", "runtime",
                                        "deployed actor bb9b0d25"],
    integration_checks=["CONTROL final state digest == existing lowmcs_retrain NEW_seed17 digest (disabled reproduces)",
                        "CANDIDATE binding includes the transition binding; timing substitutions counted (must be "
                        "reported; pool-median substitution allowed only as counted)",
                        "positive-backlog exposure logged per decision"],
    evaluation=dict(policies=["CONTROL_s17", "CANDIDATE_s17", "HIST_seed43 (deployed, frozen comparator)"],
                    environments=dict(E1="held_scene + HELDOUT_GCV2 (primary)", E2="held_scene + RETENTION_V1"),
                    variants=dict(NEW_A="corrected transition, production penalty (primary)",
                                  NEW_B="corrected transition, neutralized backlog forced-late (sensitivity)",
                                  OLD_A="existing transition, production penalty (sensitivity)"),
                    eval_seeds=[9017, 9029, 9043], decisions_per_seed=6000,
                    profiles=["FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY"]),
    success=dict(
        S1="E1/NEW_A: dP(CANDIDATE - CONTROL) >= +0.010 and paired block CI lower bound > 0",
        S2="FAVORABLE_STABLE (E1/NEW_A), vs CONTROL: dP >= -0.005 and CI lower >= -0.015; timeliness >= control - 0.010; "
           "each of the six retentions >= control's - 0.020 (Q4 margins of the constrained registration)",
        S3="guardrails (lowmcs registration R4) on E1/NEW_A: retention >= tier, or control below tier and candidate >= "
           "control - 0.01",
        S4="timeliness E1/NEW_A: candidate >= control - 0.02",
        S5="sensitivity: dP > 0 on E1/NEW_B, E1/OLD_A and E2/NEW_A",
        S6="support: candidate flagged decisions (low-MCS flags) <= 5% on E1/NEW_A",
        verdict="REPLICATION_JUSTIFIED (seeds 29/43 need separate approval) iff S1-S6 all pass; otherwise "
                "NO_EVIDENCE_FOR_REPLICATION",
        note="one seed is a screening result, not robustness evidence"),
)


def write(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name, doc in (("GATE_RULE.json", GATE), ("REGISTRATION.json", REGISTRATION)):
        path = out / name
        if path.exists():
            raise SystemExit(f"{name} is create-only")
        text = json.dumps(doc, indent=1, sort_keys=True)
        path.write_text(text)
        digests[name] = hashlib.sha256(text.encode()).hexdigest()
    print(digests)
    return digests


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    write(Path(p.parse_args().out))
