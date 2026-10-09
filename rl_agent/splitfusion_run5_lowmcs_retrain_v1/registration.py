"""Registration (written before training) for the targeted low-MCS retraining comparison; create-only, verified at launch.

    python3 -m rl_agent.splitfusion_run5_lowmcs_retrain_v1.registration --out <abs dir>      # writes REGISTRATION.json

Only intended training change: the environment's stage-E move uses the frozen candidate
(``splitfusion_run5_uplink_lowmcs_v1`` fit_01/CANDIDATE.json) on binding_02 / TRAIN_GCV2. Everything else is the
reward-trial C0 configuration, imported unchanged (its training source digest is checked).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_models as RM
from rl_agent.splitfusion_run5_full_v1 import support as S
from rl_agent.splitfusion_run5_reward_trials_v1 import authorization as RA
from rl_agent.splitfusion_run5_reward_trials_v1 import rewards as R

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SCHEMA = "splitfusion.run5_lowmcs_retrain_v1.registration.v1"
EVIDENCE_LABEL = "RUN5_LOWMCS_RETRAIN_MODELED"
ARM = "C0"
SEEDS = (17, 29, 43)
UPDATES = 10_000
SMOKE_SEED, SMOKE_AT = 17, 250
SOURCES = ("__init__.py", "registration.py", "campaign.py")

REWARD_TRIALS_SOURCE_DIGEST = "20c93d9314f624efcb3be8e9d9bfa71cf6267aa595886a61c5d9d8422278b303"
LOWMCS_RUN = ROOT / "experiments/splitfusion_run5_uplink_lowmcs_v1/20261006_v1"
CANDIDATE = LOWMCS_RUN / "fit_01/CANDIDATE.json"
LOWMCS_SOURCES = ("model.py", "latency.py", "plan.py")          # the correction's executed code
BINDING_02 = ROOT / "experiments/splitfusion_run5_reward_trials_v1/binding_02"
T_V2B = ROOT / "rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json"

COMPARATORS = {f"C0_seed{s}": dict(path=f"experiments/splitfusion_run5_reward_trials_v1/runs_01/C0_seed{s}/actor_u10000.pt",
                                   sha256=h)
               for s, h in ((17, "7a089014e47c1dc5ecacba45d5dc1943798743e3032025251035b67c9d694d65"),
                            (29, "ff9a684449e7a2aad56951d7c6b5988d069a44ee4ef0c0d687ecabcac8b83d5c"),
                            (43, "7257958da53424a5e836375db7950be34456aa2afc102a0343b7c7dcabd1e9b4"))}
DEPLOYED = dict(label="HIST_seed43", path="experiments/splitfusion_run5_full_v1/runs_01/seed43/actor_u10000.pt",
                sha256="bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce")
FIXED = {"FIXED_m11_q7000": (11, 7000), "FIXED_m11_q8000": (11, 8000)}     # chosen before evaluation

EVALUATION = dict(
    environments={"E1": "held_scene catalogue + HELDOUT_GCV2 pool (primary; deterministic held latency blocks of the "
                        "training session)", "E2": "held_scene catalogue + RETENTION_V1 pool (another session; secondary)"},
    latency_variants={"CORRECTED": "binding_02 pool + frozen candidate (primary)",
                      "OLD": "binding_02 pool without the correction (sensitivity only; not a training arm)"},
    eval_seeds=(9017, 9029, 9043), decisions_per_seed=6000,
    common_random_numbers="identical scene and channel streams for every policy (asserted: scene keys and prior MCS)",
    deterministic_rule="argmax mode at conditional-mean q (live adapter parity 200/200, reward-trial test)",
    fade_session="FADE_RECOVERY capture: PREVIOUSLY EXAMINED (it validated the correction); used only for a labelled "
                 "same-state action query on its recorded 22-D states, never as fresh confirmation",
    strata="prior MCS 9-10 / 11-15 / >=16 x backlog 0 / >0 (backlog >0 reported only where n >= 100)")
PRIMARY = ("P = 0.4 R_v^T + 0.6 R_p^T on E1-CORRECTED, R_c^T = sum(timely * hits_c) / sum(eligible_c) "
           "(late replies and misses fail; undefined-quality decisions stay in latency accounting)")
GUARDRAILS = {"loc": {"vehicle": 0.95, "person": 0.90}, "seg": {"vehicle": 0.85, "person": 0.80},
              "precision": {"vehicle": 0.95, "person": 0.90}}
RULES = {
    "R1_benefit": "mean over seeds of paired dP(NEW_s - C0_s) on E1-CORRECTED >= +0.010 and all three paired dP > 0",
    "R2_low_mcs_target": "in the MCS 9-10 stratum of E1-CORRECTED, all three paired dP > 0",
    "R3_timeliness": "mean NEW timeliness >= mean C0 timeliness - 0.02 (E1-CORRECTED)",
    "R4_guardrails": "seed-mean retention >= tier, or (C0 also below tier and NEW >= C0 - 0.01), E1-CORRECTED",
    "R5_vs_deployed": "mean P(NEW) > P(HIST_seed43) on E1-CORRECTED",
    "R6_model_uncertainty": "(a) E2-CORRECTED mean dP(NEW - C0) > 0 and (b) E1-OLD mean dP(NEW - C0) >= -0.01",
    "R7_support": "NEW flagged decisions (UNRESOLVED_BACKLOG + EXTRAPOLATED_*) <= 5% of E1-CORRECTED decisions and "
                  "the dP contribution from unflagged decisions >= 50% of dP",
    "R8_queue_order": "dP recomputed with the monotone-envelope receipt (no decision may gain from a later prediction "
                      "flipping the score order) >= 50% of dP and > 0",
}
VERDICTS = {
    "MERITS_BOUNDED_LIVE_VALIDATION": "R1-R8 all pass",
    "NO_QUALIFYING_IMPROVEMENT": "any of R1-R5 fails",
    "MODEL_UNCERTAINTY_PREVENTS_DECISION": "R1-R5 pass but any of R6-R8 fails",
}
FUTURE_CHECKPOINT_RULE = ("if live validation is later authorized: the u10000 actor of the NEW seed whose E1-CORRECTED P "
                          "is the median of the three; no other checkpoint or seed may be promoted")
STOPPING = ("each seed trains once to 10,000 updates; any exception or non-finite loss stops that run (no rerun); an "
            "interruption is resumed only from a hash-verified checkpoint bundle by deterministic re-execution, and "
            "the interruption is recorded; stop after the offline comparison")
UNSUPPORTED = ("the candidate's declared behaviour is kept: positive backlog at MCS <= 15 -> T_v2b unchanged + "
               "UNRESOLVED_BACKLOG; wire above 145,556 B at MCS <= 15 -> linear term continued + EXTRAPOLATED_PAYLOAD; "
               "MCS < 9 -> EXTRAPOLATED_MCS. No masking, discarding or altering of actions; flags counted in training "
               "and evaluation (also conditional on MCS <= 15)")
QUEUE_ORDER = ("SERVICE_QUEUE_V2 compose unchanged. Evaluation instruments it read-only: a GT_FIRST decision whose "
               "prediction, moved earlier to the order boundary, would give a LATER receipt is flagged "
               "ORDER_FLIP_GAIN; its envelope receipt is that later value. Primary results keep the real receipt; the "
               "envelope is a labelled sensitivity")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest() -> str:
    return hashlib.sha256(json.dumps({n: sha(PACKAGE / n) for n in SOURCES}, sort_keys=True).encode()).hexdigest()


def lowmcs_digest() -> dict:
    return {n: sha(ROOT / "rl_agent/splitfusion_run5_uplink_lowmcs_v1" / n) for n in LOWMCS_SOURCES}


def document() -> dict:
    manifest = json.loads((BINDING_02 / "OUTPUT_SHA256.json").read_text())
    return dict(
        schema=SCHEMA, evidence_label=EVIDENCE_LABEL, arm=ARM, seeds=list(SEEDS), updates=UPDATES,
        smoke=dict(seed=SMOKE_SEED, at=SMOKE_AT), training_pool="TRAIN_GCV2", scene_split="fit",
        source_digest=source_digest(), reward_trials_source_digest=RA.source_digest(), lowmcs_sources=lowmcs_digest(),
        candidate=dict(path=str(CANDIDATE.relative_to(ROOT)), sha256=sha(CANDIDATE),
                       assessment_sha256=sha(LOWMCS_RUN / "ASSESSMENT.json"),
                       plan_sha256=json.loads((LOWMCS_RUN / "FITTING_PLAN.json").read_text())["plan_sha256"],
                       frozen="parameters read from CANDIDATE.json; never refitted"),
        binding_02=dict(dir=str(BINDING_02.relative_to(ROOT)), manifest=manifest), t_v2b_sha256=sha(T_V2B),
        reward=dict(arm=ARM, equation=R.EQUATIONS[ARM], sha256=R.reward_sha256(ARM)),
        model_binding_sha256=RM.RUN5_TRAINING_MODEL_BINDING_SHA256, support_sha256=S.SUPPORT_SHA256,
        comparators=COMPARATORS, deployed=DEPLOYED, fixed_references={k: list(v) for k, v in FIXED.items()},
        evaluation=EVALUATION, primary_metric=PRIMARY, guardrails=GUARDRAILS, rules=RULES, verdicts=VERDICTS,
        future_checkpoint_rule=FUTURE_CHECKPOINT_RULE, stopping=STOPPING, unsupported_inputs=UNSUPPORTED,
        queue_order=QUEUE_ORDER, scope="no live capture, deployment, reward tuning, perception change or search")


class RegistrationError(RuntimeError):
    pass


def verify(path: Path, *, seed: int) -> dict:
    doc = json.loads(Path(path).read_text())
    now = document()
    if doc.get("schema") != SCHEMA:
        raise RegistrationError("foreign registration")
    for key in ("source_digest", "reward_trials_source_digest", "lowmcs_sources", "candidate", "binding_02",
                "t_v2b_sha256", "reward", "model_binding_sha256", "support_sha256"):
        if doc[key] != now[key]:
            raise RegistrationError(f"{key} changed after registration")
    if doc["reward_trials_source_digest"] != REWARD_TRIALS_SOURCE_DIGEST:
        raise RegistrationError("reward-trial training source differs from the sealed C0 runs")
    if seed not in doc["seeds"]:
        raise RegistrationError("seed outside the registered scope")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    out = p.parse_args().out
    if not out.is_absolute():
        raise SystemExit("use an absolute --out")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "REGISTRATION.json"
    doc = document()
    with open(path, "x") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print(path, sha(path))
