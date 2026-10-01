#!/usr/bin/env python3
"""Bounded successor-MCS development audit: does current SNR sharpen P(MCS')?

Evidence: the retained qualified 273-PRB dynamic capture
``rl_agent/experiments/ue_dynamic_mcs_273prb_v1/20260924_target_radio_capture_v1_retry3``
(two Gauss-Markov profiles, one realization each, contiguous FIT /
INTERNAL_VALIDATION split).  This is **development evidence**, not an
independent generalization claim.

Current SNR at decision ``k`` is reconstructed with the Run-5
:class:`RfsimEffectiveSnrProviderV1` from the cell ``command_log.json``: the
latest ACKed, unclamped target whose ACK precedes the scheduled action-open.
``HOLD`` schedule rows sent no command, so they inherit the previous ACKed
target; the schedule row's own ``target_snr_db`` is never used.  The
reconstruction is cross-checked against the schedule's HOLD structure.

Comparator (pre-registered; fixed before any validation score was computed)
--------------------------------------------------------------------------
Baseline: the exact Run-4 ``FitMcsMarkovModelV1`` weights ``W_m(m')`` fitted by
``mcs_transition_provider.fit_mcs_markov_model`` (binding must equal the
sealed Run-4 acceptance).

Augmentation (exactly one): an exponential tilt of the Run-4 row,

    P(m' | m, s) ∝ W_m(m') * exp(theta * (x - xbar_m) * (m' - m)),
    x = (snr_db - 5.5) / 19.0,
    xbar_m = (sum of FIT x at current m + 1.0 * FIT mean x) / (n_m + 1.0),

with one scalar ``theta >= 0`` fitted by maximum likelihood on FIT
transitions only (bounded scalar search on [0, THETA_MAX]).  For
``theta >= 0`` the likelihood ratio between two SNRs is increasing in ``m'``,
so the successor distribution is stochastically non-decreasing in SNR.  The
shrinkage weight 1.0 is the Run-4 backoff strength.  At ``x = xbar_m`` the
row is exactly Run 4's.

Gate (pre-registered): internal-validation mean NLL **and** mean Brier are
strictly lower pooled **and** in each of MID_VARIABLE and FADE_RECOVERY.
Secondary (reported, not gating): top-1 accuracy and MCS-index MAE of the
predictive median.

Control: temporally shuffled SNR (permuted within each profile x partition),
theta refitted on shuffled FIT, scored on shuffled validation; 200
permutations from seed 17.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from rl_agent.splitfusion_hybrid_sac_run4_v1 import dynamic_mcs_273prb_evidence as EV
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_provider as P
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4

from . import run5_state_contract as C

SCHEMA = "splitfusion.run5.successor_mcs_snr_audit.v1"
EVIDENCE_CLASS = "DEVELOPMENT_EVIDENCE_ONE_REALIZATION_PER_PROFILE__NOT_GENERALIZATION"
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
REGISTERED_RUN4_KERNEL_BINDING = (
    "20988f77b3439b46dece7fe02255a39125b01e83d161ddcc772f2a99c63148eb"
)
PROFILES = (("MID_VARIABLE", "00__mid_variable"), ("FADE_RECOVERY", "01__fade_recovery"))
SNR_LOW_DB = 5.5
SNR_SPAN_DB = 19.0
THETA_MAX = 50.0
SHRINK = P.BACKOFF_STRENGTH
SHUFFLE_SEED = 17
SHUFFLE_COUNT = 200
AUDIT_FILENAME = "SUCCESSOR_MCS_SNR_AUDIT.json"
CLOCK = "HOST_CLOCK_MONOTONIC_RETAINED_273PRB_CAPTURE_20260924"
UE_ID = "oai-nrue-1"
NAMESPACE = uuid.UUID("5f0c7f0e-6a55-4d0a-9d5e-72756e350002")


class SuccessorAuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorAuditError(message)


def scale(snr_db: float) -> float:
    return (float(snr_db) - SNR_LOW_DB) / SNR_SPAN_DB


class Ledger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: dict[str, dict[str, Any]] = {}

    def read(self, relative: Path) -> bytes:
        data = (self.root / relative).read_bytes()
        key = relative.as_posix()
        digest = hashlib.sha256(data).hexdigest()
        require(self.entries.get(key, {}).get("sha256", digest) == digest, f"{key} changed")
        self.entries[key] = {"sha256": digest, "bytes": len(data)}
        return data


# ---------------------------------------------------------------------------
# SNR reconstruction
# ---------------------------------------------------------------------------


def reconstruct_snr(ledger: Ledger) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    run = EV.SOURCE_RUN_RELATIVE_PATH
    observations = list(csv.DictReader(io.StringIO(
        ledger.read(run / "analysis/mcs_observations.csv").decode())))
    snr: dict[tuple[str, int], float] = {}
    report: dict[str, Any] = {}
    for profile, cell in PROFILES:
        log = json.loads(ledger.read(run / "cells" / cell / "command_log.json"))
        schedule = list(csv.DictReader(io.StringIO(
            ledger.read(run / "cells" / cell / "profile_schedule.csv").decode())))
        session = str(uuid.uuid5(NAMESPACE, cell))
        provider = C.RfsimEffectiveSnrProviderV1(
            provider_id="rfsim_effective_snr_provider_v1__retained_273prb",
            session_uuid=session, ue_id=UE_ID, clock_domain=CLOCK,
            records=[C.RfsimSnrCommandRecordV1.from_log_entry(
                e, session_uuid=session, command_seq=i, clock_domain=CLOCK)
                for i, e in enumerate(log)])
        rows = [r for r in observations if r["profile_id"] == profile]
        require(len(rows) == 300, f"{profile}: expected 300 decisions")
        schedule_effective = None
        hold_rows = hold_differs = future = 0
        for row, sched in zip(sorted(rows, key=lambda r: int(r["decision_index"])), schedule):
            index = int(row["decision_index"])
            require(int(sched["step_index"]) == index, "schedule/decision grid drifted")
            open_ns = int(row["scheduled_action_open_monotonic_ns"])
            require(int(sched["scheduled_action_open_monotonic_ns"]) == open_ns,
                    "schedule/decision action-open drifted")
            if sched["command_status"] == "HOLD":
                hold_rows += 1
                hold_differs += int(float(sched["target_snr_db"]) != schedule_effective)
            else:
                require(sched["command_status"] in ("PRIMED", "ACK_ON_TIME"),
                        f"unexpected status {sched['command_status']}")
                schedule_effective = float(sched["target_snr_db"])
            boundary = R4.DecisionBoundaryV1(
                identity=R4.DecisionIdentityV1(session, UE_ID, index),
                state_commit_timestamp_ns=open_ns - 1, action_open_timestamp_ns=open_ns,
                clock_domain=CLOCK)
            observation = provider.observe(boundary)
            require(observation.valid, f"{profile}[{index}]: {observation.missing_reason}")
            future += int(observation.source_timestamp_ns >= open_ns)
            require(observation.value_db == schedule_effective,
                    f"{profile}[{index}]: provider/HOLD reconstruction disagree")
            value = float(observation.value_db)
            require(SNR_LOW_DB <= value <= SNR_LOW_DB + SNR_SPAN_DB,
                    f"{profile}[{index}]: SNR {value} outside registered support")
            snr[(profile, index)] = value
        report[profile] = {"decisions": 300, "hold_rows": hold_rows,
                           "hold_rows_whose_schedule_target_differs_from_effective": hold_differs,
                           "future_joins": future,
                           "min_db": min(v for (p, _), v in snr.items() if p == profile),
                           "max_db": max(v for (p, _), v in snr.items() if p == profile)}
    return snr, report


def transition_rows(ledger: Ledger, evidence, snr) -> list[dict[str, Any]]:
    run = EV.SOURCE_RUN_RELATIVE_PATH
    raw = list(csv.DictReader(io.StringIO(
        ledger.read(run / "analysis/duration2_transitions.csv").decode())))
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in raw:
        grouped.setdefault((row["profile_id"], row["partition"]), []).append(row)
    out = []
    for partition, sequences in (("FIT", evidence.fit_sequences),
                                 ("INTERNAL_VALIDATION", evidence.internal_validation_sequences)):
        for (profile, _), sequence in zip(PROFILES, sequences):
            rows = grouped[(profile, partition)]
            require(len(rows) == len(sequence.transitions), "transition count drifted")
            for row, transition in zip(rows, sequence.transitions):
                current, successor = transition.policy_values()
                require(int(row["current_prior_ul_mcs_index"]) == current
                        and int(row["successor_prior_ul_mcs_index"]) == successor,
                        "identity-erased transition order drifted")
                out.append({"profile": profile, "partition": partition,
                            "current_index": int(row["current_decision_index"]),
                            "current_mcs": current, "successor_mcs": successor,
                            "snr_db": snr[(profile, int(row["current_decision_index"]))]})
    return out


# ---------------------------------------------------------------------------
# Augmented kernel
# ---------------------------------------------------------------------------


class SnrTiltedKernelV1:
    """The single pre-registered SNR-conditioned monotone tilt of Run 4."""

    def __init__(self, base: P.FitMcsMarkovModelV1, theta: float,
                 center_by_mcs: Mapping[int, float], global_center: float) -> None:
        require(theta >= 0.0 and math.isfinite(theta), "theta must be finite and >= 0")
        self.base = base
        self.theta = float(theta)
        self.center_by_mcs = dict(center_by_mcs)
        self.global_center = float(global_center)

    def center(self, mcs: int) -> float:
        return self.center_by_mcs.get(mcs, self.global_center)

    def probabilities(self, current_mcs: int, snr_db: float) -> tuple[float, ...]:
        weights = self.base.weights(current_mcs)
        tilt = self.theta * (scale(snr_db) - self.center(current_mcs))
        exponents = [tilt * (P.MCS_MIN + i - current_mcs) for i in range(P.STATE_COUNT)]
        peak = max(exponents)
        values = [w * math.exp(e - peak) for w, e in zip(weights, exponents)]
        total = sum(values)
        return tuple(v / total for v in values)

    def document(self) -> dict[str, Any]:
        return {"form": "W_m(m') * exp(theta * ((snr_db-5.5)/19 - xbar_m) * (m'-m))",
                "theta": self.theta, "base_binding_sha256": self.base.binding_sha256,
                "center_by_mcs": {str(k): v for k, v in sorted(self.center_by_mcs.items())},
                "global_center": self.global_center, "shrink": SHRINK}


def centers(fit: Sequence[Mapping[str, Any]]) -> tuple[dict[int, float], float]:
    global_center = sum(scale(r["snr_db"]) for r in fit) / len(fit)
    sums: dict[int, list[float]] = {}
    for r in fit:
        sums.setdefault(r["current_mcs"], []).append(scale(r["snr_db"]))
    return ({m: (sum(v) + SHRINK * global_center) / (len(v) + SHRINK)
             for m, v in sums.items()}, global_center)


def fit_tilted(base: P.FitMcsMarkovModelV1, fit: Sequence[Mapping[str, Any]]) -> SnrTiltedKernelV1:
    from scipy.optimize import minimize_scalar

    center_map, global_center = centers(fit)

    def nll(theta: float) -> float:
        kernel = SnrTiltedKernelV1(base, theta, center_map, global_center)
        return -sum(math.log(kernel.probabilities(r["current_mcs"], r["snr_db"])
                             [r["successor_mcs"] - P.MCS_MIN]) for r in fit)

    result = minimize_scalar(nll, bounds=(0.0, THETA_MAX), method="bounded",
                             options={"xatol": 1e-9, "maxiter": 500})
    theta = float(result.x)
    if nll(0.0) <= nll(theta):
        theta = 0.0
    return SnrTiltedKernelV1(base, theta, center_map, global_center)


def score(probability_fn, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nll = brier = top1 = mae = 0.0
    for r in rows:
        probs = probability_fn(r)
        observed = r["successor_mcs"] - P.MCS_MIN
        require(probs[observed] > 0.0, "zero-probability validation transition")
        nll -= math.log(probs[observed])
        brier += sum((p - (1.0 if i == observed else 0.0)) ** 2 for i, p in enumerate(probs))
        top1 += int(max(range(P.STATE_COUNT), key=probs.__getitem__) == observed)
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if cumulative >= 0.5:
                median = i
                break
        mae += abs(median - observed)
    n = len(rows)
    return {"n": n, "mean_nll": nll / n, "mean_brier": brier / n,
            "top1_accuracy": top1 / n, "median_mcs_mae": mae / n}


def evaluate(base, kernel, rows) -> dict[str, Any]:
    out = {}
    for label, subset in (("pooled", rows),
                          *((p, [r for r in rows if r["profile"] == p]) for p, _ in PROFILES)):
        b = score(lambda r: base.probabilities(r["current_mcs"]), subset)
        a = score(lambda r: kernel.probabilities(r["current_mcs"], r["snr_db"]), subset)
        out[label] = {"run4": b, "snr_tilted": a,
                      "nll_improves": a["mean_nll"] < b["mean_nll"],
                      "brier_improves": a["mean_brier"] < b["mean_brier"],
                      "nll_gain": b["mean_nll"] - a["mean_nll"],
                      "brier_gain": b["mean_brier"] - a["mean_brier"]}
    return out


def shuffle_within(rows: Sequence[Mapping[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    out = [dict(r) for r in rows]
    groups: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(out):
        groups.setdefault((r["profile"], r["partition"]), []).append(i)
    for indices in groups.values():
        values = [out[i]["snr_db"] for i in indices]
        rng.shuffle(values)
        for i, v in zip(indices, values):
            out[i]["snr_db"] = v
    return out


def load(evidence_root: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any], Ledger]:
    ledger = Ledger(evidence_root)
    evidence = EV.load_dynamic_mcs_273prb_evidence(repository_root=evidence_root)
    snr, snr_report = reconstruct_snr(ledger)
    rows = transition_rows(ledger, evidence, snr)
    return evidence, rows, snr_report, ledger


def fit_all(evidence_root: Path):
    """FIT-only construction shared with the Run-5 environment."""
    evidence, rows, snr_report, ledger = load(evidence_root)
    base = P.fit_mcs_markov_model(evidence)
    require(base.binding_sha256 == REGISTERED_RUN4_KERNEL_BINDING,
            "Run-4 kernel does not reproduce the sealed acceptance binding")
    fit = [r for r in rows if r["partition"] == "FIT"]
    require(len(fit) == 416, "FIT transition count drifted")
    return evidence, base, fit_tilted(base, fit), rows, snr_report, ledger


def run(evidence_root: Path) -> dict[str, Any]:
    evidence, base, kernel, rows, snr_report, ledger = fit_all(evidence_root)
    for relative in (Path("rl_agent/splitfusion_hybrid_sac_run5_v1/successor_mcs_snr_audit.py"),
                     Path("rl_agent/splitfusion_hybrid_sac_run5_v1/run5_state_contract.py"),
                     Path("rl_agent/splitfusion_hybrid_sac_run4_v1/mcs_transition_provider.py"),
                     Path("rl_agent/splitfusion_hybrid_sac_run4_v1/dynamic_mcs_273prb_evidence.py")):
        data = (WORKTREE_ROOT / relative).read_bytes()
        ledger.entries["worktree:" + relative.as_posix()] = {
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    validation = [r for r in rows if r["partition"] == "INTERNAL_VALIDATION"]
    require(len(validation) == 176, "validation transition count drifted")
    real = evaluate(base, kernel, validation)
    gate = {"pooled_nll": real["pooled"]["nll_improves"],
            "pooled_brier": real["pooled"]["brier_improves"],
            **{f"{p}_nll": real[p]["nll_improves"] for p, _ in PROFILES},
            **{f"{p}_brier": real[p]["brier_improves"] for p, _ in PROFILES}}
    passed = all(gate.values())

    rng = random.Random(SHUFFLE_SEED)
    shuffled_gains, shuffled_thetas = [], []
    for _ in range(SHUFFLE_COUNT):
        permuted = shuffle_within(rows, rng)
        shuffled_kernel = fit_tilted(base, [r for r in permuted if r["partition"] == "FIT"])
        result = evaluate(base, shuffled_kernel,
                          [r for r in permuted if r["partition"] == "INTERNAL_VALIDATION"])
        shuffled_gains.append(result["pooled"]["nll_gain"])
        shuffled_thetas.append(shuffled_kernel.theta)
    shuffled_gains.sort()
    control = {
        "permutations": SHUFFLE_COUNT, "seed": SHUFFLE_SEED,
        "scope": "SNR permuted within profile x partition; theta refitted on shuffled FIT",
        "pooled_nll_gain_mean": sum(shuffled_gains) / SHUFFLE_COUNT,
        "pooled_nll_gain_p95": shuffled_gains[int(0.95 * (SHUFFLE_COUNT - 1))],
        "pooled_nll_gain_max": shuffled_gains[-1],
        "fraction_at_or_above_real_gain": sum(
            g >= real["pooled"]["nll_gain"] for g in shuffled_gains) / SHUFFLE_COUNT,
        "theta_mean": sum(shuffled_thetas) / SHUFFLE_COUNT,
        "theta_zero_fraction": sum(t == 0.0 for t in shuffled_thetas) / SHUFFLE_COUNT,
    }
    return {
        "schema": SCHEMA, "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": EV.EXPECTED_CLAIM_BOUNDARY,
        "evidence_root": "main checkout, read-only (untracked retained capture)",
        "source_evidence_sha256": evidence.canonical_evidence_sha256,
        "run4_kernel_binding_sha256": base.binding_sha256,
        "run4_kernel_reproduces_sealed_acceptance": base.binding_sha256 == REGISTERED_RUN4_KERNEL_BINDING,
        "snr_reconstruction": snr_report,
        "snr_scaling": "(snr_db - 5.5) / 19.0",
        "fit_transitions": 416, "validation_transitions": 176,
        "augmentation": kernel.document(),
        "validation": real,
        "gate": gate, "gate_passed": passed,
        "verdict": ("SNR_CONDITIONED_SUCCESSOR_KERNEL_ACCEPTED_FOR_DEVELOPMENT" if passed
                    else "SNR_CONDITIONED_SUCCESSOR_KERNEL_REJECTED__STOP_BEFORE_ENVIRONMENT"),
        "shuffled_snr_control": control,
        "limitations": [
            "one realization per profile; contiguous internal validation",
            "development evidence, not an independent generalization claim",
            "SNR is the commanded effective target, not a UE measurement",
        ],
        "sources": {k: ledger.entries[k] for k in sorted(ledger.entries)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=WORKTREE_ROOT / "rl_agent/splitfusion_hybrid_sac_run5_v1" / AUDIT_FILENAME)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "audit output is create-only")
    first = run(args.evidence_root)
    second = run(args.evidence_root)
    first["deterministic_rerun_identical"] = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    args.output.write_text(json.dumps(first, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"verdict": first["verdict"], "gate": first["gate"],
                      "theta": first["augmentation"]["theta"],
                      "pooled": {k: first["validation"]["pooled"][k] for k in ("nll_gain", "brier_gain")},
                      "control": first["shuffled_snr_control"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
