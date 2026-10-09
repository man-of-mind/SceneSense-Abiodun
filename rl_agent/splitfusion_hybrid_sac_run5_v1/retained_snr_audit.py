#!/usr/bin/env python3
"""Bounded retained-evidence audit of the Run-5 SNR proxy.  CPU only, offline.

Scope
-----
1. Hash every source artifact that is read.
2. Causally join ``SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB`` to the 2,700
   production transport decisions of
   ``experiments/ue_production_queue_capture_v1/20260929_causal_join_v2b`` with
   the Run-5 :class:`RfsimEffectiveSnrProviderV1` (one provider per cell
   session), then re-derive every join with an independent brute-force scan.
   Gates: full coverage, zero future joins, zero cross-cell/session joins,
   zero non-ACKed/clamped/target-less values, deterministic output.
3. Reproduce the grouped leave-one-whole-FIT-cell-out baseline of
   ``20260929_model_v2b/EVALUATION_V2.json`` by calling the frozen
   ``model_v2.grouped_cross_validation`` unchanged; the result must be equal.
4. Fit exactly one fixed comparator (below) and compare it with the baseline
   under the identical fold protocol.

Pre-registered comparator (fixed before any outcome was examined)
------------------------------------------------------------------
Same family, same predictors, plus ``snr_norm = snr_db / 30.0`` (a fixed
physical constant mirroring ``mcs_norm = mcs / 28``; not data-derived).

* Deadline head: ``sigmoid(w0 - wb*backlog - wB*bytes_mb + wm*mcs + ws*snr)``,
  ``wb, wB, wm, ws >= 0`` by projection, identical optimizer/seed/iterations.
* Latency head: ``170*sigmoid(z0 + a*backlog + b*bytes_mb - c*mcs - d*snr)``,
  ``a, b, c, d >= 0``, exact bounded least squares in logit space.
* Queue head: the frozen baseline service table by MCS bin plus
  ``beta * (snr_db - median uncensored SNR of that bin)``, ``beta >= 0`` by
  one closed-form least-squares slope on the uncensored FIT residuals;
  service is floored at zero.

Pre-registered decision rule
----------------------------
For each question the primary metric is: deadline -> Brier; conditional
latency -> on-time |error| P50; successor backlog -> queue NMAE; reward ->
full-population expected-transport-reward MAE (Q_perc fixed at 1, the same
convention as ``model_v2.action_ranking_sensitivity``).  SNR *improves* a
question iff the comparator's pooled held-out metric is strictly lower **and**
it is strictly lower in at least 5 of the 6 held-out FIT cells.  A head's
direct SNR effect is *identified* iff it improves and its SNR coefficient is
strictly positive in at least 5 of 6 folds.  The reward question is
identified only if it improves and at least one head is identified.
Validation cells are scored once, after both models are frozen, as a
descriptive audit only; they select nothing.

No model family, threshold, scale or bin is searched.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4
from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import model_v2 as M

from . import run5_state_contract as C

AUDIT_SCHEMA = "splitfusion.run5.retained_snr_residual_audit.v1"
EVIDENCE_CLASS = "RETAINED_EVIDENCE_POSTHOC_DIAGNOSTIC__NOT_CONFIRMATORY"

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("rl_agent/experiments/ue_production_queue_capture_v1")
CAPTURE_DIR = EXPERIMENT / "20260928_214832_live"
JOIN_DIR = EXPERIMENT / "20260929_causal_join_v2b"
MODEL_DIR = EXPERIMENT / "20260929_model_v2b"
PACKAGE_DIR = Path("rl_agent/splitfusion_hybrid_sac_run5_v1")
AUDIT_FILENAME = "RETAINED_SNR_RESIDUAL_AUDIT.json"
JOIN_FILENAME = "retained_snr_join.csv"

CODE_SOURCES = (
    Path("rl_agent/ue_production_transport_model_v2/model_v2.py"),
    Path("rl_agent/ue_production_transport_model_v2/contract_v2.py"),
    Path("rl_agent/ue_production_transport_model_v2/causal_join.py"),
    Path("rl_agent/ue_mcs_backlog_run4_analysis_v1/contract.py"),
    Path("rl_agent/ue_production_queue_capture_v1/contract.py"),
    Path("rl_agent/ue_production_queue_capture_v1/runner.py"),
    Path("rl_agent/ue_mcs_backlog_calibration_v1/runner.py"),
    Path("rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py"),
    PACKAGE_DIR / "run5_state_contract.py",
    PACKAGE_DIR / "retained_snr_audit.py",
)

RETAINED_CLOCK_DOMAIN = "HOST_CLOCK_MONOTONIC_RETAINED_CAPTURE_20260928"
RETAINED_UE_ID = "oai-nrue-1"
SESSION_NAMESPACE = uuid.UUID("5f0c7f0e-6a55-4d0a-9d5e-72756e350001")
PROVIDER_ID = "rfsim_effective_snr_provider_v1__retained_command_log"

SNR_NORM_DB = 30.0
FOLD_MAJORITY = 5
NEXT_ACK_LAG_WINDOW_NS = 1_000_000


class AuditError(RuntimeError):
    """An audit gate or invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


class SourceLedger:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: dict[str, dict[str, Any]] = {}

    def read_bytes(self, relative: Path) -> bytes:
        data = (self.root / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        key = relative.as_posix()
        previous = self.entries.get(key)
        require(previous is None or previous["sha256"] == digest,
                f"{key} changed while the audit was reading it")
        self.entries[key] = {"sha256": digest, "bytes": len(data)}
        return data

    def read_json(self, relative: Path) -> Any:
        return json.loads(self.read_bytes(relative))

    def document(self) -> dict[str, Any]:
        return {key: self.entries[key] for key in sorted(self.entries)}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Causal SNR join
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CellSession:
    cell_id: str
    cell_tag: str
    partition: str
    session_uuid: str
    log: tuple[Mapping[str, Any], ...]
    window: tuple[int, int]


def load_decision_table(ledger: SourceLedger) -> list[dict[str, str]]:
    text = ledger.read_bytes(JOIN_DIR / "decisions.csv").decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    require(len(rows) == 2700, f"expected 2700 decisions, found {len(rows)}")
    return rows


def load_sessions(ledger: SourceLedger, rows: Sequence[Mapping[str, str]]) -> dict[str, CellSession]:
    sessions: dict[str, CellSession] = {}
    tags = sorted({(row["cell_tag"], row["cell_id"], row["partition"]) for row in rows})
    require(len(tags) == 12 and len({t[0] for t in tags}) == 12, "expected 12 cells")
    for tag, cell_id, partition in tags:
        cell = CAPTURE_DIR / "cells" / tag
        record = ledger.read_json(cell / "cell_record.json")
        require(record["cell_id"] == cell_id and record["status"] == "CAPTURED",
                f"{tag}: cell record mismatch")
        epoch = ledger.read_json(cell / "shared_epoch.json")
        require(epoch["cell_id"] == cell_id, f"{tag}: epoch cell mismatch")
        log = tuple(ledger.read_json(cell / "command_log.json"))
        session_uuid = str(uuid.uuid5(
            SESSION_NAMESPACE,
            f"{tag}|{epoch['epoch_monotonic_ns']}|{epoch['sender_ready_sha256']}",
        ))
        window = (min(int(e["send_monotonic_ns"]) for e in log),
                  max(int(e["ack_monotonic_ns"]) for e in log))
        sessions[tag] = CellSession(cell_id, tag, partition, session_uuid, log, window)
    ordered = sorted(sessions.values(), key=lambda s: s.window[0])
    for earlier, later in zip(ordered, ordered[1:]):
        require(earlier.window[1] < later.window[0], "cell sessions overlap in time")
    return sessions


def build_provider(session: CellSession) -> C.RfsimEffectiveSnrProviderV1:
    records = [
        C.RfsimSnrCommandRecordV1.from_log_entry(
            entry, session_uuid=session.session_uuid, command_seq=index,
            clock_domain=RETAINED_CLOCK_DOMAIN)
        for index, entry in enumerate(session.log)
    ]
    return C.RfsimEffectiveSnrProviderV1(
        provider_id=PROVIDER_ID, session_uuid=session.session_uuid,
        ue_id=RETAINED_UE_ID, clock_domain=RETAINED_CLOCK_DOMAIN, records=records)


def retained_boundary(session: CellSession, decision_seq: int, frame_open_ns: int) -> R4.DecisionBoundaryV1:
    """Retained capture records no separate state-commit stamp.

    The v2 join already used ``frame_open`` (= action-open) as the causal
    cutoff for backlog and MCS with strict inequality; committing at
    ``frame_open - 1 ns`` reproduces exactly that strict cutoff.
    """
    return R4.DecisionBoundaryV1(
        identity=R4.DecisionIdentityV1(session.session_uuid, RETAINED_UE_ID, decision_seq),
        state_commit_timestamp_ns=frame_open_ns - 1,
        action_open_timestamp_ns=frame_open_ns,
        clock_domain=RETAINED_CLOCK_DOMAIN)


def reference_join(log: Sequence[Mapping[str, Any]], frame_open_ns: int) -> dict[str, Any]:
    """Independent linear scan; shares no code with the provider."""
    best = None
    for entry in log:
        if int(entry["ack_monotonic_ns"]) < frame_open_ns:
            if best is None or int(entry["ack_monotonic_ns"]) > int(best["ack_monotonic_ns"]):
                best = entry
    in_flight = any(
        int(e["send_monotonic_ns"]) < frame_open_ns <= int(e["ack_monotonic_ns"])
        for e in log)
    return {"entry": best, "in_flight": in_flight}


def join_snr(ledger: SourceLedger) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_decision_table(ledger)
    sessions = load_sessions(ledger, rows)
    foreign_acks = {
        tag: {int(e["ack_monotonic_ns"]) for other, s in sessions.items() if other != tag
              for e in s.log}
        for tag in sessions
    }
    providers = {tag: build_provider(s) for tag, s in sessions.items()}
    per_cell_seq: dict[str, int] = {tag: 0 for tag in sessions}
    joined: list[dict[str, Any]] = []
    counts = {
        "decisions": 0, "valid": 0, "missing": 0, "future_joins": 0,
        "cross_cell_or_session_joins": 0, "non_acked_values": 0, "clamped_values": 0,
        "target_less_values": 0, "reference_disagreements": 0,
        "newer_command_in_flight": 0, "next_ack_within_1ms_after_open": 0,
        "profile_or_trace_fields_exposed": 0,
    }
    missing_reasons: dict[str, int] = {}
    ages: list[int] = []
    for row in sorted(rows, key=lambda r: (r["cell_tag"], int(r["frame_index"]))):
        session = sessions[row["cell_tag"]]
        open_ns = int(row["frame_open_monotonic_ns"])
        seq = per_cell_seq[session.cell_tag]
        per_cell_seq[session.cell_tag] += 1
        observation = providers[session.cell_tag].observe(
            retained_boundary(session, seq, open_ns))
        counts["decisions"] += 1
        exposed = set(observation.to_canonical_dict()) & set(C.RFSIM_LOG_FIELDS_NEVER_EXPOSED)
        counts["profile_or_trace_fields_exposed"] += len(exposed)
        reference = reference_join(session.log, open_ns)
        entry = reference["entry"]
        if observation.newer_command_in_flight:
            counts["newer_command_in_flight"] += 1
        acks = sorted(int(e["ack_monotonic_ns"]) for e in session.log)
        after = bisect.bisect_left(acks, open_ns)
        if after < len(acks) and acks[after] - open_ns < NEXT_ACK_LAG_WINDOW_NS:
            counts["next_ack_within_1ms_after_open"] += 1
        if not observation.valid:
            counts["missing"] += 1
            missing_reasons[observation.missing_reason] = (
                missing_reasons.get(observation.missing_reason, 0) + 1)
            snr_db = None
            source_ns = None
        else:
            counts["valid"] += 1
            snr_db = float(observation.value_db)
            source_ns = int(observation.source_timestamp_ns)
            if source_ns >= open_ns:
                counts["future_joins"] += 1
            if (observation.identity.session_uuid != session.session_uuid
                    or source_ns in foreign_acks[session.cell_tag]
                    or not session.window[0] <= source_ns <= session.window[1]):
                counts["cross_cell_or_session_joins"] += 1
            if entry is None or entry.get("status") != "ACK":
                counts["non_acked_values"] += 1
            if entry is None or entry.get("clamped") is not False:
                counts["clamped_values"] += 1
            if entry is None or entry.get("target_snr_db") is None:
                counts["target_less_values"] += 1
            ages.append(open_ns - source_ns)
        reference_value = (None if entry is None or entry.get("status") != "ACK"
                           or entry.get("clamped") is not False
                           else entry.get("target_snr_db"))
        reference_source = None if reference_value is None else int(entry["ack_monotonic_ns"])
        if (reference_value != snr_db or reference_source != source_ns
                or reference["in_flight"] != observation.newer_command_in_flight):
            counts["reference_disagreements"] += 1
        joined.append({
            "cell_id": row["cell_id"], "cell_tag": row["cell_tag"],
            "partition": row["partition"], "frame_index": int(row["frame_index"]),
            "frame_open_monotonic_ns": open_ns,
            "snr_proxy_db": snr_db, "snr_source_ack_monotonic_ns": source_ns,
            "snr_age_ns": None if source_ns is None else open_ns - source_ns,
            "newer_command_in_flight": observation.newer_command_in_flight,
            "valid": observation.valid, "missing_reason": observation.missing_reason,
        })
    counts["coverage"] = counts["valid"] / counts["decisions"]
    ages_sorted = sorted(ages)
    summary = {
        "counts": counts, "missing_reasons": missing_reasons,
        "age_ns": {
            "p50": int(statistics.median(ages_sorted)) if ages_sorted else None,
            "p95": ages_sorted[int(round(0.95 * (len(ages_sorted) - 1)))] if ages_sorted else None,
            "max": ages_sorted[-1] if ages_sorted else None,
            "over_150ms": sum(1 for a in ages_sorted if a > 150_000_000),
        },
        "sessions": {
            tag: {"cell_id": s.cell_id, "partition": s.partition,
                  "session_uuid": s.session_uuid, "commands": len(s.log),
                  "window_monotonic_ns": list(s.window)}
            for tag, s in sorted(sessions.items())
        },
        "cutoff": "RFsim ACK strictly before frame_open (= action-open); state commit := frame_open - 1 ns",
        "clock_domain": RETAINED_CLOCK_DOMAIN,
        "session_identity": "uuid5(namespace, cell_tag|epoch_monotonic_ns|sender_ready_sha256)",
        "selection_rule_id": C.RFSIM_SELECTION_RULE_ID,
        "label": C.SNR_PROXY_LABEL,
    }
    gates = {
        "FULL_COVERAGE": counts["valid"] == counts["decisions"] == 2700,
        "ZERO_FUTURE_JOINS": counts["future_joins"] == 0,
        "ZERO_CROSS_CELL_OR_SESSION_JOINS": counts["cross_cell_or_session_joins"] == 0,
        "ZERO_NON_ACKED_VALUES": counts["non_acked_values"] == 0,
        "ZERO_CLAMPED_VALUES": counts["clamped_values"] == 0,
        "ZERO_TARGET_LESS_VALUES": counts["target_less_values"] == 0,
        "INDEPENDENT_REFERENCE_AGREES": counts["reference_disagreements"] == 0,
        "NO_PROFILE_TRACE_NOISE_FIELDS_EXPOSED": counts["profile_or_trace_fields_exposed"] == 0,
    }
    summary["gates"] = gates
    return joined, summary


def join_csv_bytes(joined: Sequence[Mapping[str, Any]]) -> bytes:
    fields = list(joined[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in joined:
        writer.writerow({k: ("" if row[k] is None else
                             (repr(row[k]) if isinstance(row[k], float) else row[k]))
                         for k in fields})
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Fixed comparator
# ---------------------------------------------------------------------------


def snr_norm(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array([float(row["snr_db"]) for row in rows]) / SNR_NORM_DB


@dataclass(frozen=True, slots=True)
class SnrDeadlineHead:
    w0: float
    w_backlog: float
    w_bytes: float
    w_mcs: float
    w_snr: float
    final_gradient_norm: float

    def probability(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        f = M.build_features(rows)
        z = (self.w0 - self.w_backlog * f.backlog_scaled - self.w_bytes * f.bytes_mb
             + self.w_mcs * f.mcs_norm + self.w_snr * snr_norm(rows))
        return M._sigmoid(z)


@dataclass(frozen=True, slots=True)
class SnrLatencyHead:
    z0: float
    a_backlog: float
    b_bytes: float
    c_mcs: float
    d_snr: float

    def latency_ms(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        f = M.build_features(rows)
        z = (self.z0 + self.a_backlog * f.backlog_scaled + self.b_bytes * f.bytes_mb
             - self.c_mcs * f.mcs_norm - self.d_snr * snr_norm(rows))
        return M.DEADLINE_MS * M._sigmoid(z)


@dataclass(frozen=True, slots=True)
class SnrQueueHead:
    base: M.QueueTransitionHead
    beta_bytes_per_db: float
    center_db_by_bin: Mapping[int, float]
    global_center_db: float

    def next_backlog_bytes(self, row: Mapping[str, Any]) -> float:
        service, _ = self.base.service_bytes(float(row["prior_ul_mcs"]))
        key = M.mcs_bin(float(row["prior_ul_mcs"]))
        center = self.center_db_by_bin.get(key, self.global_center_db)
        adjusted = max(0.0, service + self.beta_bytes_per_db * (float(row["snr_db"]) - center))
        return max(0.0, float(row["pre_enqueue_backlog_bytes"])
                   + float(row["deterministic_action_ingress_bytes"]) - adjusted)


def _y(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.array([1.0 if row["completed_within_deadline"] else 0.0 for row in rows])


def fit_snr_deadline(rows: Sequence[Mapping[str, Any]]) -> SnrDeadlineHead:
    f = M.build_features(rows)
    y = _y(rows)
    x = np.stack([np.ones_like(y), -f.backlog_scaled, -f.bytes_mb, f.mcs_norm,
                  snr_norm(rows)], axis=1)
    rng = np.random.default_rng(C2.MODEL_SEED)
    w = rng.normal(0.0, 0.01, size=5)
    w[1:] = np.abs(w[1:])
    n = len(y)
    for _ in range(M.DEADLINE_ITERATIONS):
        gradient = x.T @ (M._sigmoid(x @ w) - y) / n
        w -= M.DEADLINE_LEARNING_RATE * gradient
        w[1:] = np.maximum(w[1:], 0.0)
    gradient = x.T @ (M._sigmoid(x @ w) - y) / n
    projected = gradient.copy()
    projected[1:][(w[1:] == 0.0) & (gradient[1:] > 0.0)] = 0.0
    return SnrDeadlineHead(*(float(v) for v in w), float(np.linalg.norm(projected)))


def baseline_deadline_gradient_norm(rows: Sequence[Mapping[str, Any]], head: M.DeadlineHead) -> float:
    f = M.build_features(rows)
    y = _y(rows)
    x = np.stack([np.ones_like(y), -f.backlog_scaled, -f.bytes_mb, f.mcs_norm], axis=1)
    w = np.array([head.w0, head.w_backlog, head.w_bytes, head.w_mcs])
    gradient = x.T @ (M._sigmoid(x @ w) - y) / len(y)
    gradient[1:][(w[1:] == 0.0) & (gradient[1:] > 0.0)] = 0.0
    return float(np.linalg.norm(gradient))


def fit_snr_latency(rows: Sequence[Mapping[str, Any]]) -> SnrLatencyHead:
    from scipy.optimize import lsq_linear

    on_time = [row for row in rows if row["completed_within_deadline"]]
    f = M.build_features(on_time)
    y = np.array([float(row["transport_latency_ns"]) / 1e6 for row in on_time])
    y = np.clip(y, M.LOGIT_EPSILON_MS, M.DEADLINE_MS - M.LOGIT_EPSILON_MS)
    target = np.log(y / (M.DEADLINE_MS - y))
    design = np.stack([np.ones_like(y), f.backlog_scaled, f.bytes_mb, -f.mcs_norm,
                       -snr_norm(on_time)], axis=1)
    solution = lsq_linear(design, target,
                          bounds=(np.array([-np.inf, 0, 0, 0, 0]), np.full(5, np.inf)),
                          method="trf", tol=1e-12, max_iter=500)
    return SnrLatencyHead(*(float(v) for v in solution.x))


def fit_snr_queue(rows: Sequence[Mapping[str, Any]]) -> SnrQueueHead:
    base = M.fit_queue_transition_head(rows)
    samples: list[tuple[int, float, float]] = []
    for row in rows:
        if not row.get("has_successor") or row["successor_backlog_bytes"] is None:
            continue
        successor = float(row["successor_backlog_bytes"])
        if successor <= 0:
            continue
        service = (float(row["pre_enqueue_backlog_bytes"])
                   + float(row["deterministic_action_ingress_bytes"]) - successor)
        if service <= 0:
            continue
        key = M.mcs_bin(float(row["prior_ul_mcs"]))
        residual = service - base.service_bytes(float(row["prior_ul_mcs"]))[0]
        samples.append((key, float(row["snr_db"]), residual))
    by_bin: dict[int, list[float]] = {}
    for key, snr, _ in samples:
        by_bin.setdefault(key, []).append(snr)
    centers = {key: statistics.median(values) for key, values in by_bin.items()
               if key in base.service_by_mcs_bin}
    global_center = statistics.median(snr for _, snr, _ in samples)
    dx = np.array([snr - centers.get(key, global_center) for key, snr, _ in samples])
    dy = np.array([residual for _, _, residual in samples])
    denominator = float(dx @ dx)
    beta = max(0.0, float(dx @ dy) / denominator) if denominator > 0 else 0.0
    return SnrQueueHead(base, beta, centers, float(global_center))


@dataclass(frozen=True, slots=True)
class SnrModel:
    deadline: SnrDeadlineHead
    latency: SnrLatencyHead
    queue: SnrQueueHead


def fit_snr_model(rows: Sequence[Mapping[str, Any]]) -> SnrModel:
    return SnrModel(fit_snr_deadline(rows), fit_snr_latency(rows), fit_snr_queue(rows))


# ---------------------------------------------------------------------------
# Symmetric scoring
# ---------------------------------------------------------------------------


def predict_baseline(model: M.TwoPartModel, rows):
    f = M.build_features(rows)
    queue = [model.queue.next_backlog_bytes(
        backlog_bytes=float(r["pre_enqueue_backlog_bytes"]),
        ingress_bytes=float(r["deterministic_action_ingress_bytes"]),
        mcs=float(r["prior_ul_mcs"]))[0] for r in rows]
    return model.deadline.probability(f), model.latency.latency_ms(f), queue


def predict_snr(model: SnrModel, rows):
    queue = [model.queue.next_backlog_bytes(r) for r in rows]
    return model.deadline.probability(rows), model.latency.latency_ms(rows), queue


def _pct(values: Sequence[float], q: float) -> float:
    return float(M._percentile(list(values), q))


def metrics(rows, probability, latency, queue) -> dict[str, Any]:
    y = _y(rows)
    on_time = y == 1.0
    realized_latency = np.array([
        float(r["transport_latency_ns"]) / 1e6 if r["completed_within_deadline"] else np.nan
        for r in rows])
    errors = np.abs(latency[on_time] - realized_latency[on_time])
    reward_errors = [C2.reward_error_for_latency_error_ms(float(e)) for e in errors]
    predicted_positive = probability >= 0.5
    predicted_count = int(predicted_positive.sum())
    false_positive = int((predicted_positive & (y == 0.0)).sum())
    weight = C2.REWARD_LATENCY_WEIGHT / C2.REWARD_DEADLINE_MS
    realized_reward = np.where(on_time, 1.0 - weight * np.nan_to_num(realized_latency), -1.0)
    expected_reward = probability * (1.0 - weight * latency) + (1.0 - probability) * -1.0
    successors = [(i, float(r["successor_backlog_bytes"])) for i, r in enumerate(rows)
                  if r.get("has_successor") and r["successor_backlog_bytes"] is not None]
    observed = np.array([value for _, value in successors])
    predicted_queue = np.array([queue[i] for i, _ in successors])
    return {
        "n": len(rows), "n_on_time": int(on_time.sum()), "n_successor": len(successors),
        "brier": float(np.mean((probability - y) ** 2)),
        "false_success_rate": false_positive / predicted_count if predicted_count else 0.0,
        "latency_abs_error_p50_ms": _pct(errors.tolist(), 0.50),
        "latency_abs_error_p95_ms": _pct(errors.tolist(), 0.95),
        "reward_error_p50": _pct(reward_errors, 0.50),
        "reward_error_p95": _pct(reward_errors, 0.95),
        "queue_nmae": (float(np.abs(predicted_queue - observed).sum() / np.abs(observed).sum())
                       if np.abs(observed).sum() > 0 else math.inf),
        "expected_transport_reward_mae": float(np.mean(np.abs(expected_reward - realized_reward))),
    }


PRIMARY = {
    "deadline_probability": "brier",
    "conditional_latency": "latency_abs_error_p50_ms",
    "successor_backlog": "queue_nmae",
    "reward_error": "expected_transport_reward_mae",
}
SECONDARY = ("false_success_rate", "latency_abs_error_p95_ms", "reward_error_p50",
             "reward_error_p95")
COEFFICIENT = {
    "deadline_probability": "deadline_w_snr",
    "conditional_latency": "latency_d_snr",
    "successor_backlog": "queue_beta_bytes_per_db",
}


def grouped_comparison(fit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(all(row["partition"] == V1.FIT for row in fit_rows),
            "validation rows must never enter the grouped comparison")
    cells = sorted({row["cell_id"] for row in fit_rows})
    folds = []
    pooled: dict[str, dict[str, list]] = {
        name: {"rows": [], "p": [], "l": [], "q": []} for name in ("baseline", "snr")}
    for held in cells:
        train = [r for r in fit_rows if r["cell_id"] != held]
        test = [r for r in fit_rows if r["cell_id"] == held]
        base_model = M.fit_model(train)
        snr_model = fit_snr_model(train)
        predictions = {"baseline": predict_baseline(base_model, test),
                       "snr": predict_snr(snr_model, test)}
        fold = {"held_out_cell": held,
                "baseline": metrics(test, *predictions["baseline"]),
                "snr": metrics(test, *predictions["snr"]),
                "coefficients": {
                    "deadline_w_snr": snr_model.deadline.w_snr,
                    "deadline_w_mcs_baseline": base_model.deadline.w_mcs,
                    "deadline_w_mcs_snr": snr_model.deadline.w_mcs,
                    "latency_d_snr": snr_model.latency.d_snr,
                    "latency_c_mcs_baseline": base_model.latency.c_mcs,
                    "latency_c_mcs_snr": snr_model.latency.c_mcs,
                    "queue_beta_bytes_per_db": snr_model.queue.beta_bytes_per_db,
                },
                "deadline_final_projected_gradient_norm": {
                    "baseline": baseline_deadline_gradient_norm(train, base_model.deadline),
                    "snr": snr_model.deadline.final_gradient_norm,
                }}
        folds.append(fold)
        for name, (p, l, q) in predictions.items():
            pooled[name]["rows"].extend(test)
            pooled[name]["p"].extend(p.tolist())
            pooled[name]["l"].extend(l.tolist())
            pooled[name]["q"].extend(q)
    pooled_metrics = {
        name: metrics(v["rows"], np.array(v["p"]), np.array(v["l"]), v["q"])
        for name, v in pooled.items()}
    questions = {}
    for question, metric in PRIMARY.items():
        wins = sum(1 for f in folds if f["snr"][metric] < f["baseline"][metric])
        pooled_better = pooled_metrics["snr"][metric] < pooled_metrics["baseline"][metric]
        improves = pooled_better and wins >= FOLD_MAJORITY
        entry = {
            "primary_metric": metric,
            "pooled_baseline": pooled_metrics["baseline"][metric],
            "pooled_snr": pooled_metrics["snr"][metric],
            "pooled_relative_change": ((pooled_metrics["snr"][metric]
                                        - pooled_metrics["baseline"][metric])
                                       / pooled_metrics["baseline"][metric]),
            "held_out_cells_improved": wins, "held_out_cells": len(folds),
            "improves": improves,
        }
        if question in COEFFICIENT:
            positive = sum(1 for f in folds if f["coefficients"][COEFFICIENT[question]] > 0.0)
            entry["folds_with_positive_snr_coefficient"] = positive
            entry["direct_effect_identified"] = improves and positive >= FOLD_MAJORITY
        questions[question] = entry
    any_head = any(questions[q]["direct_effect_identified"] for q in COEFFICIENT)
    questions["reward_error"]["direct_effect_identified"] = (
        questions["reward_error"]["improves"] and any_head)
    return {"protocol": C2.SELECTION_PROTOCOL, "folds": folds,
            "pooled": pooled_metrics, "questions": questions}


def identification_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    f = M.build_features(rows)
    snr = np.array([float(r["snr_db"]) for r in rows])
    x = np.stack([np.ones(len(rows)), f.mcs_norm, f.backlog_scaled, f.bytes_mb], axis=1)
    beta, *_ = np.linalg.lstsq(x, snr, rcond=None)
    residual = snr - x @ beta
    r2 = 1.0 - float(residual @ residual) / float(((snr - snr.mean()) ** 2).sum())
    return {
        "rows": len(rows),
        "corr_snr_prior_mcs": float(np.corrcoef(snr, f.mcs_norm)[0, 1]),
        "r2_snr_on_mcs_backlog_bytes": r2,
        "variance_inflation_factor": 1.0 / (1.0 - r2) if r2 < 1 else math.inf,
        "snr_sd_db": float(snr.std()),
        "residual_snr_sd_db": float(residual.std()),
    }


def attach_snr(rows: list[dict[str, Any]], joined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(j["cell_id"], j["frame_index"]): j["snr_proxy_db"] for j in joined}
    out = []
    for row in rows:
        value = lookup[(row["cell_id"], row["frame_index"])]
        require(value is not None, "outcome analysis reached a row without SNR")
        out.append({**row, "snr_db": float(value)})
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def run_once(root: Path = REPO_ROOT) -> tuple[dict[str, Any], bytes]:
    ledger = SourceLedger(root)
    for path in CODE_SOURCES:
        ledger.read_bytes(path)
    joined, join_summary = join_snr(ledger)
    join_bytes = join_csv_bytes(joined)

    evaluation = ledger.read_json(MODEL_DIR / "EVALUATION_V2.json")
    ledger.read_bytes(MODEL_DIR / "transport_model_v2.json")
    ledger.read_bytes(JOIN_DIR / "CAUSAL_JOIN_REPORT.json")
    decisions_path = root / JOIN_DIR / "decisions.csv"
    rows = M.load_decisions(decisions_path)
    ledger.read_bytes(JOIN_DIR / "decisions.csv")
    fit_rows = [r for r in rows if r["partition"] == V1.FIT]
    validation_rows = [r for r in rows if r["partition"] == V1.VALIDATION]
    require(len(fit_rows) == 1350 and len(validation_rows) == 1350, "partition drift")

    reproduced = M.grouped_cross_validation(fit_rows)
    reproduction = {
        "reference": (MODEL_DIR / "EVALUATION_V2.json").as_posix() + "#cross_validation",
        "pooled_equal": _plain(reproduced["pooled"]) == evaluation["cross_validation"]["pooled"],
        "folds_equal": _plain(reproduced["folds"]) == evaluation["cross_validation"]["folds"],
        "protocol": reproduced["protocol"],
        "pooled": _plain(reproduced["pooled"]),
    }
    reproduction["exact"] = reproduction["pooled_equal"] and reproduction["folds_equal"]

    gates_passed = all(join_summary["gates"].values())
    result: dict[str, Any] = {
        "schema": AUDIT_SCHEMA, "evidence_class": EVIDENCE_CLASS,
        "baseline_commit": "42bca44",
        "snr_label": C.SNR_PROXY_LABEL,
        "not_ue_measured": True, "gnb_pusch_snr_role": "VERIFIER_ONLY_NOT_READ",
        "join": join_summary, "join_csv": {"filename": JOIN_FILENAME,
                                           "sha256": hashlib.sha256(join_bytes).hexdigest(),
                                           "rows": len(joined)},
        "baseline_reproduction": reproduction,
        "preregistered_comparator": {
            "snr_normalization": f"snr_db / {SNR_NORM_DB}",
            "deadline": "baseline logistic + ws*snr_norm, ws>=0, same seed/lr/iterations",
            "latency": "baseline bounded-logit lsq + (-d*snr_norm), d>=0",
            "queue": "baseline MCS-bin service + beta*(snr_db - bin median), beta>=0 closed form",
            "primary_metrics": PRIMARY, "secondary_metrics": list(SECONDARY),
            "improvement_rule": ("pooled held-out strictly lower AND strictly lower in "
                                 f">= {FOLD_MAJORITY}/6 held-out FIT cells"),
            "identification_rule": ("improves AND SNR coefficient > 0 in "
                                    f">= {FOLD_MAJORITY}/6 folds; reward requires a head"),
            "validation_cells": "scored once after freezing; descriptive only",
            "searched": "nothing",
        },
    }
    if not (gates_passed and reproduction["exact"]):
        result["outcome_analysis"] = "NOT_RUN__JOIN_OR_REPRODUCTION_GATE_FAILED"
        result["verdict"] = "BLOCKED"
        result["sources"] = ledger.document()
        return _plain(result), join_bytes

    fit_snr = attach_snr(fit_rows, joined)
    validation_snr = attach_snr(validation_rows, joined)
    comparison = grouped_comparison(fit_snr)
    frozen_base = M.fit_model(fit_snr)
    frozen_snr = fit_snr_model(fit_snr)
    validation = {
        "status": "DESCRIPTIVE_ONLY__NOT_USED_FOR_ANY_DECISION",
        "population_status": C2.VALIDATION_POPULATION_STATUS,
        "baseline": metrics(validation_snr, *predict_baseline(frozen_base, validation_snr)),
        "snr": metrics(validation_snr, *predict_snr(frozen_snr, validation_snr)),
        "frozen_snr_coefficients": {
            "deadline_w_snr": frozen_snr.deadline.w_snr,
            "latency_d_snr": frozen_snr.latency.d_snr,
            "queue_beta_bytes_per_db": frozen_snr.queue.beta_bytes_per_db,
        },
    }
    identified = [q for q, v in comparison["questions"].items()
                  if v.get("direct_effect_identified")]
    result.update({
        "identification_diagnostics": {
            "fit": identification_diagnostics(fit_snr),
            "fit_by_cell_prefix": {
                prefix: identification_diagnostics(
                    [r for r in fit_snr if r["cell_id"].startswith(prefix)])
                for prefix in ("favorable_stable", "adverse_stable")
            },
            "trace_replication": ("every cell of one profile replays the same target-SNR "
                                  "trace on the shared epoch; SNR is a function of "
                                  "(trace, elapsed step) and is identified only by two traces"),
        },
        "comparison": comparison,
        "validation_descriptive": validation,
        "identified_questions": identified,
        "verdict": ("SNR_DIRECT_EFFECT_IDENTIFIED" if identified
                    else "SNR_DIRECT_EFFECT_NOT_IDENTIFIED_BY_RETAINED_EVIDENCE"),
        "outcome_model_rule": ("SNR may enter only the outcome heads listed in "
                               "identified_questions; otherwise it stays state-only"),
    })
    result["sources"] = ledger.document()
    return _plain(result), join_bytes


def run(output_dir: Path, root: Path = REPO_ROOT) -> dict[str, Any]:
    first, first_join = run_once(root)
    second, second_join = run_once(root)
    deterministic = (canonical_sha256(first) == canonical_sha256(second)
                     and first_join == second_join)
    first["determinism"] = {"independent_runs": 2, "identical": deterministic,
                            "result_sha256_excluding_this_block": canonical_sha256(second)}
    first["join"]["gates"]["DETERMINISTIC_OUTPUT"] = deterministic
    audit_path = output_dir / AUDIT_FILENAME
    join_path = output_dir / JOIN_FILENAME
    for path in (audit_path, join_path):
        require(not path.exists(), f"{path.name} is create-only")
    join_path.write_bytes(first_join)
    audit_path.write_text(json.dumps(first, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return first


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / PACKAGE_DIR)
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    print(json.dumps({"verdict": result["verdict"],
                      "join_gates": result["join"]["gates"],
                      "baseline_reproduction_exact": result["baseline_reproduction"]["exact"],
                      "identified_questions": result.get("identified_questions")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
