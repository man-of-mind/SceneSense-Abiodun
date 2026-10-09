"""Versioned, observed clocks for a future live numerical-GT archive.

This module only binds observations supplied by their owning process. It does
not read a clock on behalf of an event, translate wall time into RAW time, or
reconstruct the old AVO pre-score-write chronology. The clock descriptor is
written before archive submission; a separate completion record can bind the
actual durable receipt afterward without a hash cycle.
"""
from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
import re

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from .scorer import registration_key


SCHEMA = "splitfusion.numeric_gt.live_clock.v1"
COMPLETION_SCHEMA = "splitfusion.numeric_gt.live_clock_archive_completion.v1"
LATE_SCHEMA = "splitfusion.numeric_gt.live_late_feedback.v1"
ASYNC_HANDOFF_ID = "OWNED_SOURCE_PRE_ACTION__PRODUCER_AFTER_SEND_V2"
ASYNC_SCHEMA = "splitfusion.numeric_gt.live_clock.v2"
ASYNC_COMPLETION_SCHEMA = "splitfusion.numeric_gt.live_clock_archive_completion.v2"
ASYNC_LATE_SCHEMA = "splitfusion.numeric_gt.live_late_feedback.v2"
CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
DEADLINE_NS = 170_000_000
_BOOT = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_EVENTS = {
    "collector": frozenset(("owned_tick_snapshot",)),
    "producer": frozenset(("source_stage_received", "prepare_start", "source_alignment_observed", "numeric_ready",
                           "ready_published")),
    "scorer": frozenset(("target_admitted", "prediction_admitted", "score_start", "score_end",
                         "archive_build_start", "archive_build_end")),
    "coordinator": frozenset(("feedback_emitted",)),
    "ue": frozenset(("source_owned", "action_open", "last_datagram_sent", "feedback_received",
                      "feedback_timeout", "late_feedback_observed")),
    "archive_client": frozenset(("archive_submitted", "durable_completion_observed")),
}
_CAUSAL = (
    (("collector", "owned_tick_snapshot"), ("producer", "source_alignment_observed")),
    (("collector", "owned_tick_snapshot"), ("producer", "source_stage_received")),
    (("producer", "source_stage_received"), ("ue", "action_open")),
    (("producer", "source_stage_received"), ("producer", "prepare_start")),
    (("collector", "owned_tick_snapshot"), ("ue", "action_open")),
    (("producer", "prepare_start"), ("producer", "source_alignment_observed")),
    (("producer", "source_alignment_observed"), ("producer", "numeric_ready")),
    (("producer", "numeric_ready"), ("producer", "ready_published")),
    (("ue", "action_open"), ("ue", "last_datagram_sent")),
    (("ue", "last_datagram_sent"), ("producer", "ready_published")),
    (("producer", "ready_published"), ("scorer", "target_admitted")),
    (("ue", "last_datagram_sent"), ("scorer", "prediction_admitted")),
    (("scorer", "target_admitted"), ("scorer", "score_start")),
    (("scorer", "prediction_admitted"), ("scorer", "score_start")),
    (("scorer", "score_start"), ("scorer", "score_end")),
    (("scorer", "score_end"), ("coordinator", "feedback_emitted")),
    (("ue", "last_datagram_sent"), ("coordinator", "feedback_emitted")),
    (("producer", "ready_published"), ("coordinator", "feedback_emitted")),
    (("coordinator", "feedback_emitted"), ("ue", "feedback_received")),
    (("coordinator", "feedback_emitted"), ("ue", "late_feedback_observed")),
    (("ue", "action_open"), ("ue", "feedback_timeout")),
    (("ue", "feedback_timeout"), ("ue", "late_feedback_observed")),
    (("ue", "feedback_received"), ("scorer", "archive_build_start")),
    (("ue", "feedback_timeout"), ("scorer", "archive_build_start")),
    (("scorer", "score_end"), ("scorer", "archive_build_start")),
    (("scorer", "archive_build_start"), ("scorer", "archive_build_end")),
    (("scorer", "archive_build_end"), ("archive_client", "archive_submitted")),
    (("archive_client", "archive_submitted"), ("archive_client", "durable_completion_observed")),
)
_ASYNC_CAUSAL = tuple(pair for pair in _CAUSAL if pair !=
    (("producer", "source_stage_received"), ("ue", "action_open"))) + (
    (("collector", "owned_tick_snapshot"), ("ue", "source_owned")),
    (("ue", "source_owned"), ("ue", "action_open")),
    (("ue", "last_datagram_sent"), ("producer", "source_stage_received")),
    (("ue", "action_open"), ("producer", "prepare_start")),
)


class LiveClockError(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise LiveClockError(message)


def read_boot_id() -> str:
    """Read the kernel boot identity in the event-owning process."""
    value = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    _require(_BOOT.fullmatch(value) is not None, "valid Linux boot ID required")
    return value


def _identity(raw: bytes) -> dict:
    _require(type(raw) is bytes and len(raw) <= C.MAX_METADATA_BYTES,
             "canonical full ready identity bytes required")
    value = C._parse(raw)
    _require(C.canonical(value) == raw and type(value) is dict
             and set(value) == {"capture_identity", "action_identity", "plan_sha256",
                                "run_binding", "producer_binding", "frame_context",
                                "prediction_contract"}, "canonical full ready identity fields required")
    expected = dict(expected_capture_identity=value["capture_identity"],
                    expected_action_identity=value["action_identity"],
                    plan_sha256=value["plan_sha256"], run_binding=value["run_binding"],
                    producer_binding=value["producer_binding"], frame_context=value["frame_context"],
                    prediction_contract=value["prediction_contract"])
    _require(registration_key(expected) == raw, "ready identity validation differs")
    return value


class LiveClockLedger:
    """One frame and one ticket; every event is tied to its exact ready identity."""

    def __init__(self, ready_identity_bytes: bytes, semantic_sha256: str,
                 *, handoff_id: str | None = None):
        _require(handoff_id in (None, ASYNC_HANDOFF_ID),
                 "registered live source handoff version required")
        self.handoff_id = handoff_id
        self._identity_bytes = bytes(ready_identity_bytes)
        self.identity = _identity(self._identity_bytes)
        _require(C.is_digest(semantic_sha256), "numerical semantic digest required")
        self.semantic_sha256 = semantic_sha256
        self.identity_sha256 = C.sha256(self._identity_bytes)
        self._events = {}
        self._role_owner = {}
        self._role_last = {}
        self._descriptor_bytes = None
        self._completion_bytes = None
        self._late_bytes = None
        self._verifying_receipt = False

    def record(self, *, role: str, name: str, raw_ns: int, pid: int, boot_id: str,
               ready_identity_bytes: bytes, semantic_sha256: str) -> dict:
        _require(self._completion_bytes is None or (role, name) == ("ue", "late_feedback_observed"),
                 "clock ledger already completed")
        _require(type(role) is str and role in _EVENTS and name in _EVENTS[role],
                 "unknown event role/name")
        _require(self.handoff_id == ASYNC_HANDOFF_ID or (role, name) != ("ue", "source_owned"),
                 "source-owned event belongs only to async handoff V2")
        if self._descriptor_bytes is None:
            _require(role != "archive_client", "archive event precedes immutable clock descriptor")
        else:
            _require(role == "archive_client" or (role == "scorer" and name in
                     ("archive_build_end",))
                     or (role, name) == ("ue", "late_feedback_observed"),
                     "prearchive clock ledger is already frozen")
        if name == "durable_completion_observed":
            _require(self._verifying_receipt, "durable event requires verified archive receipt")
        _require(type(raw_ns) is int and 0 < raw_ns < 2**63 and type(pid) is int and pid > 0,
                 "positive observed RAW instant and process PID required")
        _require(type(boot_id) is str and _BOOT.fullmatch(boot_id) is not None,
                 "valid Linux boot ID required")
        _require(type(ready_identity_bytes) is bytes and ready_identity_bytes == self._identity_bytes
                 and semantic_sha256 == self.semantic_sha256,
                 "event belongs to a foreign complete frame or semantic result")
        opened = self._events.get(("ue", "action_open"))
        if role == "ue" and name in ("feedback_received", "feedback_timeout", "late_feedback_observed"):
            _require(opened is not None, "UE terminal clock precedes actual action-open")
            if name == "feedback_received":
                _require(("ue", "feedback_timeout") not in self._events
                         and raw_ns - opened["raw_ns"] <= DEADLINE_NS,
                         "timely receipt conflicts with timeout or arrives after deadline")
            elif name == "feedback_timeout":
                _require(("ue", "feedback_received") not in self._events
                         and raw_ns - opened["raw_ns"] > DEADLINE_NS,
                         "timeout conflicts with timely receipt or precedes deadline")
            else:
                timed_out = self._events.get(("ue", "feedback_timeout"))
                _require(timed_out is not None and raw_ns >= timed_out["raw_ns"],
                         "late feedback requires a prior terminal timeout")
        key = (role, name)
        _require(key not in self._events, "duplicate live clock event")
        if self.handoff_id == ASYNC_HANDOFF_ID:
            prerequisite = {
                ("ue", "source_owned"): ("collector", "owned_tick_snapshot"),
                ("ue", "action_open"): ("ue", "source_owned"),
                ("producer", "source_stage_received"): ("ue", "last_datagram_sent"),
                ("producer", "prepare_start"): ("producer", "source_stage_received"),
            }.get(key)
            _require(prerequisite is None or prerequisite in self._events,
                     "async handoff event lacks actual predecessor: " + name)
        owner = (boot_id, pid)
        _require(role not in self._role_owner or self._role_owner[role] == owner,
                 "process owner changed within a role")
        _require(role not in self._role_last or raw_ns >= self._role_last[role],
                 "per-role RAW clock regressed")
        event = dict(role=role, name=name, raw_ns=raw_ns, pid=pid, boot_id=boot_id,
                     clock_domain=CLOCK_DOMAIN, ready_identity=self.identity,
                     ready_identity_sha256=self.identity_sha256,
                     semantic_sha256=self.semantic_sha256)
        # Validate a prospective copy so a rejected event never enters the ledger.
        candidate = dict(self._events)
        candidate[key] = event
        for before, after in (_ASYNC_CAUSAL if self.handoff_id == ASYNC_HANDOFF_ID else _CAUSAL):
            left, right = candidate.get(before), candidate.get(after)
            if left is not None and right is not None and left["boot_id"] == right["boot_id"]:
                _require(left["raw_ns"] <= right["raw_ns"],
                         "same-boot causal event order regressed: " + before[1] + " -> " + after[1])
        self._events[key] = event
        self._role_owner[role], self._role_last[role] = owner, raw_ns
        return C._json_owned(event)

    def deadline(self) -> dict:
        """Inclusive UE timer; a missing reply needs its own observed timeout."""
        opened = self._events.get(("ue", "action_open"))
        received = self._events.get(("ue", "feedback_received"))
        timed_out = self._events.get(("ue", "feedback_timeout"))
        late = self._events.get(("ue", "late_feedback_observed"))
        _require(opened is not None and (received is not None or timed_out is not None),
                 "actual UE action-open and receipt or timeout required")
        _require((received is None) != (timed_out is None), "exactly one UE terminal outcome required")
        elapsed = None if received is None else received["raw_ns"] - opened["raw_ns"]
        _require(elapsed is None or elapsed >= 0, "UE feedback precedes action-open")
        within = elapsed is not None and elapsed <= DEADLINE_NS
        return dict(schema="splitfusion.numeric_gt.live_feedback_deadline.v1", clock_domain=CLOCK_DOMAIN,
                    ready_identity_sha256=self.identity_sha256, semantic_sha256=self.semantic_sha256,
                    action_open_raw_ns=opened["raw_ns"],
                    feedback_received_raw_ns=None if received is None else received["raw_ns"],
                    timeout_observed_raw_ns=None if timed_out is None else timed_out["raw_ns"],
                    late_feedback_observed_raw_ns=None if late is None else late["raw_ns"],
                    late_feedback_elapsed_ns=None if late is None else late["raw_ns"] - opened["raw_ns"],
                    deadline_ns=DEADLINE_NS, elapsed_ns=elapsed, within_deadline=within,
                    terminal="FEEDBACK_ON_TIME" if within else "FEEDBACK_TIMEOUT")

    def archive_descriptor(self, *, avo_sha256: str, source_synchronization_sha256: str,
                           fault_feedback_sha256: str | None = None,
                           fault_reason: str | None = None) -> bytes:
        """Canonical prearchive clock sidecar; old AVO pre-score fields are absent."""
        _require(self._descriptor_bytes is None, "archive clock descriptor is create-only")
        _require(C.is_digest(avo_sha256) and C.is_digest(source_synchronization_sha256),
                 "numerical AVO and source synchronization digests required")
        required = {("collector", "owned_tick_snapshot"),
                    ("ue", "action_open"), ("ue", "last_datagram_sent"),
                    ("producer", "source_stage_received"),
                    ("producer", "prepare_start"), ("producer", "source_alignment_observed"),
                    ("producer", "numeric_ready"), ("producer", "ready_published"),
                    }
        if self.handoff_id == ASYNC_HANDOFF_ID:
            required.add(("ue", "source_owned"))
        _require(required <= self._events.keys(),
                 "complete observed source and UE terminal events required")
        if self.handoff_id == ASYNC_HANDOFF_ID:
            tick = self._events[("collector", "owned_tick_snapshot")]
            owned = self._events[("ue", "source_owned")]
            sent = self._events[("ue", "last_datagram_sent")]
            staged = self._events[("producer", "source_stage_received")]
            _require(tick["boot_id"] == owned["boot_id"] == sent["boot_id"] == staged["boot_id"],
                     "async after-send source claim requires one observed Linux boot")
        terminal = self.deadline()
        scorer = {name for role, name in self._events if role == "scorer"}
        emitted = ("coordinator", "feedback_emitted") in self._events
        if "score_end" in scorer:
            _require(fault_feedback_sha256 is None and fault_reason is None,
                     "scored feedback cannot carry an unscored fault claim")
            _require({"target_admitted", "prediction_admitted", "score_start"} <= scorer,
                     "scored frame lacks actual scorer admissions/start")
            _require("archive_build_start" in scorer,
                     "scored frame lacks scorer-owned archive build start")
            scoring_state = "SCORED_FEEDBACK_EMITTED" if emitted else "SCORED_NOT_EMITTED"
        else:
            _require(not {"archive_build_start", "archive_build_end"} & scorer,
                     "archive build lacks completed scoring")
            if emitted:
                _require(C.is_digest(fault_feedback_sha256)
                         and type(fault_reason) is str
                         and re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", fault_reason) is not None,
                         "unscored feedback needs bounded actual fault packet digest/reason")
                scoring_state = "UNSCORED_FAULT_FEEDBACK_EMITTED"
            else:
                _require(fault_feedback_sha256 is None and fault_reason is None,
                         "fault packet claim lacks actual coordinator emission")
                scoring_state = "SCORING_AT_TIMEOUT" if "score_start" in scorer else "NOT_SCORED_AT_TIMEOUT"
        if terminal["terminal"] == "FEEDBACK_ON_TIME":
            _require(scoring_state in ("SCORED_FEEDBACK_EMITTED", "UNSCORED_FAULT_FEEDBACK_EMITTED"),
                     "on-time feedback lacks actual score or fault emission")
        if ("ue", "late_feedback_observed") in self._events:
            _require(emitted, "observed late feedback lacks coordinator emission")
        document = dict(schema=ASYNC_SCHEMA if self.handoff_id == ASYNC_HANDOFF_ID else SCHEMA,
                        clock_domain=CLOCK_DOMAIN,
                        clock_scope="OBSERVED_LIVE_EVENTS_NO_LEGACY_PRESCORE_CLAIM",
                        ready_identity=self.identity, ready_identity_sha256=self.identity_sha256,
                        semantic_sha256=self.semantic_sha256,
                        avo_sha256=avo_sha256,
                        source_synchronization_sha256=source_synchronization_sha256,
                        deadline=terminal, scoring_state=scoring_state,
                        unscored_fault=None if not emitted or "score_end" in scorer else dict(
                            feedback_packet_sha256=fault_feedback_sha256, reason=fault_reason,
                            score_exists=False, quality_value=None),
                        events=list(self._events.values()),
                        incomparable_boot_pairs=sorted({"/".join(sorted((a["boot_id"], b["boot_id"])))
                            for a in self._events.values() for b in self._events.values()
                            if a["boot_id"] != b["boot_id"]}))
        if self.handoff_id == ASYNC_HANDOFF_ID:
            document["handoff_id"] = ASYNC_HANDOFF_ID
        raw = C.canonical(document)
        _require(len(raw) <= C.MAX_METADATA_BYTES, "bounded live clock sidecar required")
        self._descriptor_bytes = raw
        return raw

    def durability_record(self, *, descriptor_bytes: bytes, receipt: dict,
                          raw_ns: int, pid: int, boot_id: str,
                          ready_identity_bytes: bytes, semantic_sha256: str) -> bytes:
        """Bind an actual ArchiveClient receipt after its durable completion."""
        _require(self._descriptor_bytes is not None and type(descriptor_bytes) is bytes
                 and descriptor_bytes == self._descriptor_bytes,
                 "original prearchive clock descriptor required")
        _require(self._completion_bytes is None, "durability record is create-only")
        scored = C._parse(self._descriptor_bytes)["scoring_state"].startswith("SCORED_")
        if scored:
            _require(("scorer", "archive_build_start") in self._events
                     and ("scorer", "archive_build_end") in self._events,
                     "actual scorer archive build completion required after descriptor")
        else:
            _require(("scorer", "archive_build_start") not in self._events
                     and ("scorer", "archive_build_end") not in self._events,
                     "unscored terminal must not invent archive build")
        _require(("archive_client", "archive_submitted") in self._events,
                 "archive completion precedes actual submission")
        value = C._json_owned(receipt)
        _require(type(value) is dict and value.get("status") == "COMPLETE"
                 and value.get("durable") is True and value.get("error") is None,
                 "actual durable COMPLETE receipt required")
        _require(value.get("identity") == self.identity["capture_identity"],
                 "archive receipt belongs to another capture")
        written = value.get("written")
        _require(type(written) is list and 0 < len(written) <= C.MAX_ROWS
                 and all(type(row) is dict and type(row.get("root")) is int and row["root"] >= 0
                         and type(row.get("relative")) is str and bool(row["relative"])
                         and not PurePosixPath(row["relative"]).is_absolute()
                         and ".." not in PurePosixPath(row["relative"]).parts
                         and C.is_digest(row.get("sha256")) and type(row.get("bytes")) is int
                         and row["bytes"] >= 0 and row.get("ready") is False for row in written),
                 "bounded non-READY durable file inventory required")
        _require(len({row["relative"] for row in written}) == len(written),
                 "duplicate durable archive file")
        _require(sum(PurePosixPath(row["relative"]).name == "live_clock.json"
                     and row["sha256"] == C.sha256(descriptor_bytes)
                     and row["bytes"] == len(descriptor_bytes) for row in written) == 1,
                 "durable receipt lacks exact live clock sidecar")
        self._verifying_receipt = True
        try:
            self.record(role="archive_client", name="durable_completion_observed", raw_ns=raw_ns,
                        pid=pid, boot_id=boot_id, ready_identity_bytes=ready_identity_bytes,
                        semantic_sha256=semantic_sha256)
        finally:
            self._verifying_receipt = False
        document = dict(schema=ASYNC_COMPLETION_SCHEMA if self.handoff_id == ASYNC_HANDOFF_ID
                        else COMPLETION_SCHEMA, ready_identity=self.identity,
                        ready_identity_sha256=self.identity_sha256,
                        semantic_sha256=self.semantic_sha256,
                        clock_descriptor_sha256=C.sha256(descriptor_bytes),
                        archive_receipt_sha256=C.sha256(C.canonical(value)),
                        archive_scope="SCORER_NUMERIC_ARCHIVE" if scored else "UNSCORED_TERMINAL_SIDECAR_ONLY",
                        archive_build_start_event=self._events.get(("scorer", "archive_build_start")),
                        archive_build_end_event=self._events.get(("scorer", "archive_build_end")),
                        archive_submit_event=self._events[("archive_client", "archive_submitted")],
                        archive_completion_event=self._events[("archive_client", "durable_completion_observed")],
                        receipt_status="COMPLETE", durable=True, written_files=len(written))
        if self.handoff_id == ASYNC_HANDOFF_ID:
            document["handoff_id"] = ASYNC_HANDOFF_ID
        self._completion_bytes = C.canonical(document)
        return self._completion_bytes

    def late_feedback_sidecar(self) -> bytes:
        """Separate evidence for an orphan received after the frozen timeout."""
        _require(self._descriptor_bytes is not None and self._late_bytes is None,
                 "late feedback sidecar requires a frozen descriptor and is create-only")
        late = self._events.get(("ue", "late_feedback_observed"))
        timeout = self._events.get(("ue", "feedback_timeout"))
        _require(late is not None and timeout is not None,
                 "actual late reply and prior timeout required")
        document = dict(schema=ASYNC_LATE_SCHEMA if self.handoff_id == ASYNC_HANDOFF_ID
                        else LATE_SCHEMA, ready_identity=self.identity,
                        ready_identity_sha256=self.identity_sha256,
                        semantic_sha256=self.semantic_sha256,
                        clock_descriptor_sha256=C.sha256(self._descriptor_bytes),
                        timeout_event=timeout, late_feedback_event=late,
                        terminal="FEEDBACK_TIMEOUT", late_elapsed_ns=self.deadline()["late_feedback_elapsed_ns"])
        if self.handoff_id == ASYNC_HANDOFF_ID:
            document["handoff_id"] = ASYNC_HANDOFF_ID
        self._late_bytes = C.canonical(document)
        return self._late_bytes


__all__ = ["LiveClockError", "LiveClockLedger", "read_boot_id", "SCHEMA", "COMPLETION_SCHEMA",
           "LATE_SCHEMA", "ASYNC_HANDOFF_ID", "ASYNC_SCHEMA", "ASYNC_COMPLETION_SCHEMA",
           "ASYNC_LATE_SCHEMA", "CLOCK_DOMAIN", "DEADLINE_NS"]
