"""Bounded, causal, actor-facing UE telemetry provider for the Run-4 live path.

Transport is **reused, not added**: the qualified launcher opens UE T port
2023, OAI ``multi`` relays it to 2123 while the durable ``record`` client
keeps writing ``ue.raw``, and this module attaches three extra ``csv -f``
clients to the same relay (``NRUE_MAC_DCI_GRANT``,
``NRUE_MAC_RLC_BUFFER_STATUS``, ``NR_PDCP_TX_SDU``).  Nothing here opens an
OAI socket of its own, edits or rebuilds OAI, or tails ``ue.raw``.

Hot-path rule
-------------
Reader threads own all parsing, clock conversion and aggregation.  After each
accepted update they publish one immutable :class:`TelemetrySnapshotV2` by a
single reference assignment.  :meth:`UeTelemetryProviderV2.snapshot` is one
attribute read: no lock, socket read, disk access, wait or unbounded copy.
Every cache is a fixed-capacity deque.  Optional audit lines go through a
bounded queue to a separate writer thread.

Semantics
---------
* DCI: uplink (``direction == 1``), table 0, HARQ round 0, the bound RNTI.
  NDI 0 **and** 1 are accepted: NR signals new data by an NDI *toggle*, so
  ``ndi == 1`` is not "new data".  MCS outside ``[0, 28]`` is passed through
  so the registered selector fails closed instead of the value being hidden.
* RLC: rows are grouped by the complete ``(rnti, ue_id, frame, slot)``
  scheduler tick; the latest value per distinct LCID is summed.  A tick is
  published only when a row of the *next* tick proves it complete; its
  availability is that proof instant.  The open group is never exposed.  A
  measured zero is a valid ``0``; missing telemetry is typed missing.
* Clock: the Run-4 contract clock is ``CLOCK_MONOTONIC_RAW``.  The T header
  is ``CLOCK_REALTIME`` printed as local time-of-day in microseconds.  It is
  reconstructed against the host ``CLOCK_REALTIME`` receipt instant (safe
  across midnight), mapped REALTIME -> MONOTONIC by the median of the most
  recent *already received* ``NR_PDCP_TX_SDU`` dual-stamped anchors, and
  MONOTONIC -> RAW by the latest in-process host clock pair.  Warm-up,
  discontinuities, negative ingest latency and future times fail closed.

Decisions use the unchanged Run-4 selectors
(:func:`state_adapter.select_prior_new_data_ul_mcs`,
:func:`state_adapter.select_pre_action_rlc_backlog`) and the exact training
freshness policy.  Any failure yields typed missing evidence plus the
registered :class:`run4_contract.ExternalFallbackRequired`; the actor is never
called after a failed guard.

Importing this module starts nothing.
"""

from __future__ import annotations

import collections
import enum
import queue
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as contract
from rl_agent.splitfusion_hybrid_sac_run4_v1 import state_adapter as SA

__all__ = [
    "CLOCK_DOMAIN",
    "DCI_FIELDS",
    "RLC_FIELDS",
    "PDCP_FIELDS",
    "TRAINING_FRESHNESS",
    "TRAINING_FRESHNESS_SHA256",
    "HostClockPairV1",
    "read_host_clock_pair",
    "raw_now_ns",
    "reconstruct_event_realtime_ns",
    "CausalClockBridgeV2",
    "TelemetrySnapshotV2",
    "UeTelemetryProviderV2",
    "LiveEventReaderV2",
    "RadioDecisionEvidenceV2",
    "open_decision",
    "assemble_radio_evidence",
    "require_admitted",
    "act_or_fallback",
]

CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
UL_DIRECTION = 1
SOURCE_ID = "splitfusion_run4_live_v2:ue_ttracer_multi_csv"

DCI_FIELDS: Tuple[str, ...] = (
    "time", "direction", "dci_format", "rnti_type", "rnti", "dci_frame",
    "dci_slot", "sched_frame", "sched_slot", "mcs", "mcs_table", "rb_start",
    "rb_size", "start_symbol", "nr_symbols", "tbs", "harq_pid", "ndi", "rv",
    "round", "qam_mod_order", "target_code_rate", "tpc", "n_cce", "N_cce",
)
RLC_FIELDS: Tuple[str, ...] = (
    "time", "rnti", "ue_id", "frame", "slot", "lcid", "lcgid",
    "bytes_in_buffer", "bj", "pbr", "priority",
)
PDCP_FIELDS: Tuple[str, ...] = (
    "time", "mono_sec", "mono_nsec", "ue_id", "rb_id", "sdu_bytes",
)
EVENT_FIELDS: Mapping[str, Tuple[str, ...]] = {
    "NRUE_MAC_DCI_GRANT": DCI_FIELDS,
    "NRUE_MAC_RLC_BUFFER_STATUS": RLC_FIELDS,
    "NR_PDCP_TX_SDU": PDCP_FIELDS,
}

# The exact freshness object the seed-43 run trained with.  Its digest is the
# checkpoint's replay_binding_document.freshness_policy_sha256.
TRAINING_FRESHNESS = contract.FreshnessPolicyV2(
    policy_id="run4-production-transport-v2-freshness",
    policy_version=1,
    evidence_sha256=(
        "23a73949d54278dc6e04dd043944806c98241f08e83d0416488cd7d39be4b79b"
    ),
    camera_si_max_age_ns=100_000_000,
    radar_p40_max_age_ns=100_000_000,
    prior_ul_mcs_max_age_ns=100_000_000,
    pre_action_rlc_backlog_max_age_ns=100_000_000,
)
TRAINING_FRESHNESS_SHA256 = (
    "6c694ebed2edd12a6e935cfff0262f78b2c8395f180d76bfb7060db77e3a5d65"
)
if TRAINING_FRESHNESS.canonical_sha256() != TRAINING_FRESHNESS_SHA256:
    raise RuntimeError("training freshness policy reconstruction drifted")

DAY_US = 86_400_000_000
MAX_INGEST_US = 2_000_000          # reconstruction window, not a freshness rule
MAX_HOST_PAIR_SPREAD_NS = 20_000
DISCONTINUITY_NS = 1_000_000
_TOD = re.compile(r"^([0-2][0-9]):([0-5][0-9]):([0-5][0-9])\.([0-9]{6})$")


# ---------------------------------------------------------------------------
# Host clocks
# ---------------------------------------------------------------------------


def raw_now_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


@dataclass(frozen=True, slots=True)
class HostClockPairV1:
    """One bracketed read of REALTIME, MONOTONIC and MONOTONIC_RAW."""

    real_ns: int
    mono_ns: int
    raw_ns: int
    spread_ns: int
    utc_offset_s: int

    @property
    def usable(self) -> bool:
        return 0 <= self.spread_ns <= MAX_HOST_PAIR_SPREAD_NS


def read_host_clock_pair() -> HostClockPairV1:
    first = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    real = time.clock_gettime_ns(time.CLOCK_REALTIME)
    last = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return HostClockPairV1(
        real_ns=real, mono_ns=mono, raw_ns=(first + last) // 2,
        spread_ns=last - first,
        utc_offset_s=int(time.localtime(real // 1_000_000_000).tm_gmtoff),
    )


def parse_time_of_day_us(text: str) -> int:
    match = _TOD.match(text)
    if match is None:
        raise ValueError(f"malformed T time-of-day {text!r}")
    hours, minutes, seconds, micros = (int(item) for item in match.groups())
    if hours > 23:
        raise ValueError(f"malformed T hour {text!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000_000 + micros


def reconstruct_event_realtime_ns(tod_text: str, receipt: HostClockPairV1) -> int:
    """Reconstruct the T header CLOCK_REALTIME instant, midnight-safe.

    The event precedes its receipt.  A time-of-day *after* the receipt wraps
    to nearly a day and is rejected as future; a gap beyond the window is
    rejected as unreconstructable.
    """
    event_tod = parse_time_of_day_us(tod_text)
    receipt_us = receipt.real_ns // 1000
    receipt_tod = (receipt_us + receipt.utc_offset_s * 1_000_000) % DAY_US
    delta = (receipt_tod - event_tod) % DAY_US
    if delta > MAX_INGEST_US:
        raise ValueError(
            "T time-of-day is after receipt (future) or outside the "
            f"reconstruction window: delta={delta} us")
    return (receipt_us - delta) * 1000


# ---------------------------------------------------------------------------
# Causal clock bridge
# ---------------------------------------------------------------------------


class CausalClockBridgeV2:
    """Past-only REALTIME -> MONOTONIC_RAW bridge with fail-closed resets."""

    def __init__(self, *, min_anchors: int = 8, window: int = 32,
                 residual_capacity: int = 65_536) -> None:
        if not 1 <= min_anchors <= window:
            raise ValueError("require 1 <= min_anchors <= window")
        self._lock = threading.Lock()
        self._min = min_anchors
        self._offsets: collections.deque[int] = collections.deque(maxlen=window)
        self._estimate: Optional[int] = None
        self._mono_to_raw: Optional[int] = None
        self._host_real_to_mono: Optional[int] = None
        self.generation = 0
        self.residuals_ns: collections.deque[int] = collections.deque(
            maxlen=residual_capacity)
        self.counters: collections.Counter[str] = collections.Counter()

    def _reset(self, reason: str) -> None:
        self._offsets.clear()
        self._estimate = None
        self.generation += 1
        self.counters[f"reset_{reason}"] += 1

    def observe_host(self, pair: HostClockPairV1) -> None:
        if not pair.usable:
            self.counters["host_pair_unusable"] += 1
            return
        with self._lock:
            mono_to_raw = pair.raw_ns - pair.mono_ns
            real_to_mono = pair.mono_ns - pair.real_ns
            if (self._mono_to_raw is not None
                    and abs(mono_to_raw - self._mono_to_raw) > DISCONTINUITY_NS):
                self._reset("monotonic_raw_jump")
            if (self._host_real_to_mono is not None
                    and abs(real_to_mono - self._host_real_to_mono)
                    > DISCONTINUITY_NS):
                self._reset("realtime_step")
            self._mono_to_raw = mono_to_raw
            self._host_real_to_mono = real_to_mono

    def observe_anchor(self, event_real_ns: int, event_mono_ns: int) -> Optional[int]:
        """Add one dual-stamped anchor; return its out-of-sample residual."""
        with self._lock:
            offset = event_mono_ns - event_real_ns
            residual = None
            if self._estimate is not None:
                residual = offset - self._estimate
                if abs(residual) > DISCONTINUITY_NS:
                    self._reset("anchor_discontinuity")
                    residual = None
                else:
                    self.residuals_ns.append(residual)
            self._offsets.append(offset)
            if len(self._offsets) >= self._min:
                ordered = sorted(self._offsets)
                self._estimate = ordered[len(ordered) // 2]
            self.counters["anchors"] += 1
            return residual

    @property
    def warm(self) -> bool:
        return self._estimate is not None and self._mono_to_raw is not None

    def to_raw(self, event_real_ns: int) -> Optional[int]:
        with self._lock:
            if self._estimate is None or self._mono_to_raw is None:
                return None
            return event_real_ns + self._estimate + self._mono_to_raw

    def mono_to_raw(self, mono_ns: int) -> Optional[int]:
        with self._lock:
            return None if self._mono_to_raw is None else mono_ns + self._mono_to_raw


# ---------------------------------------------------------------------------
# Snapshot and provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TelemetrySnapshotV2:
    seq: int
    session_uuid: str
    ue_label: Optional[str]
    bridge_warm: bool
    bridge_generation: int
    readers_alive: Tuple[Tuple[str, bool], ...]
    dci: Tuple[SA.RawUeUlDciGrantCandidateV1, ...]
    rlc: Tuple[SA.RawUeRlcBacklogSampleV1, ...]

    @property
    def all_readers_alive(self) -> bool:
        return bool(self.readers_alive) and all(alive for _, alive in self.readers_alive)


@dataclass
class _OpenTick:
    key: Tuple[int, int, int, int]
    by_lcid: dict[int, int] = field(default_factory=dict)
    source_real_ns: int = 0
    last_tod: str = ""


class UeTelemetryProviderV2:
    """Owns parsing state; publishes immutable snapshots for the actor path."""

    REQUIRED_READERS = tuple(EVENT_FIELDS)

    def __init__(self, *, bridge: CausalClockBridgeV2,
                 session_uuid: Optional[str] = None,
                 dci_capacity: int = 8, rlc_capacity: int = 8) -> None:
        self.session_uuid = session_uuid or str(uuid.uuid4())
        contract.SampleIdentityV1(self.session_uuid, "probe", 0)  # validates
        self.bridge = bridge
        self._lock = threading.Lock()
        self._dci: collections.deque = collections.deque(maxlen=dci_capacity)
        self._rlc: collections.deque = collections.deque(maxlen=rlc_capacity)
        self._open: Optional[_OpenTick] = None
        self._rnti: Optional[int] = None
        self._oai_ue_id: Optional[int] = None
        self._alive = {name: False for name in self.REQUIRED_READERS}
        self._dci_seq = 0
        self._rlc_seq = 0
        self._seen_generation = bridge.generation
        self.unbound_ue_candidates: set[Tuple[int, int]] = set()
        self.counters: collections.Counter[str] = collections.Counter()
        self._snapshot_seq = 0
        self._snapshot = self._build_snapshot()

    # -- identity -------------------------------------------------------
    def bind_ue(self, *, rnti: int, oai_ue_id: int) -> None:
        with self._lock:
            if self._rnti is not None:
                raise RuntimeError("provider UE binding is immutable per session")
            self._rnti, self._oai_ue_id = int(rnti), int(oai_ue_id)
            self._publish()

    @property
    def ue_label(self) -> Optional[str]:
        if self._rnti is None:
            return None
        return f"oai-ue{self._oai_ue_id}-rnti{self._rnti}"

    # -- snapshot (actor path) -----------------------------------------------
    def snapshot(self) -> TelemetrySnapshotV2:
        """O(1): one reference read of an immutable published snapshot."""
        return self._snapshot

    def _build_snapshot(self) -> TelemetrySnapshotV2:
        self._snapshot_seq += 1
        return TelemetrySnapshotV2(
            seq=self._snapshot_seq, session_uuid=self.session_uuid,
            ue_label=self.ue_label, bridge_warm=self.bridge.warm,
            bridge_generation=self.bridge.generation,
            readers_alive=tuple(sorted(self._alive.items())),
            dci=tuple(self._dci), rlc=tuple(self._rlc))

    def _publish(self) -> None:
        self._snapshot = self._build_snapshot()

    def _sync_bridge_generation(self) -> None:
        if self.bridge.generation != self._seen_generation:
            # Samples timed under a pre-discontinuity estimate are discarded.
            self._dci.clear()
            self._rlc.clear()
            self._open = None
            self._seen_generation = self.bridge.generation
            self.counters["caches_cleared_on_bridge_reset"] += 1

    def cache_sizes(self) -> dict[str, int]:
        return {"dci": len(self._dci), "dci_capacity": self._dci.maxlen,
                "rlc": len(self._rlc), "rlc_capacity": self._rlc.maxlen,
                "bridge_residuals": len(self.bridge.residuals_ns),
                "bridge_residual_capacity": self.bridge.residuals_ns.maxlen}

    # -- reader lifecycle -------------------------------------------------------
    def reader_alive(self, name: str, alive: bool) -> None:
        with self._lock:
            if name not in self._alive:
                raise KeyError(name)
            self._alive[name] = bool(alive)
            if not alive:
                self.counters[f"reader_dead_{name}"] += 1
                if name == "NRUE_MAC_RLC_BUFFER_STATUS":
                    self._open = None   # an unproven tick is never published
            self._publish()

    # -- event handlers (reader threads) ---------------------------------------
    def _timed(self, row: Mapping[str, Any], pair: HostClockPairV1,
               kind: str) -> Optional[int]:
        try:
            return reconstruct_event_realtime_ns(row["time"], pair)
        except ValueError:
            self.counters[f"{kind}_rejected_future_or_unreconstructable_time"] += 1
            return None

    def on_pdcp(self, row: Mapping[str, Any], pair: HostClockPairV1) -> None:
        self.bridge.observe_host(pair)
        real = self._timed(row, pair, "pdcp")
        if real is None:
            return
        mono = int(row["mono_sec"]) * 1_000_000_000 + int(row["mono_nsec"])
        if not 0 <= int(row["mono_nsec"]) < 1_000_000_000:
            self.counters["pdcp_malformed_mono"] += 1
            return
        if mono > pair.mono_ns:
            self.counters["pdcp_rejected_future_mono"] += 1
            return
        self.bridge.observe_anchor(real, mono)
        with self._lock:
            self._sync_bridge_generation()
            self._publish()

    def on_dci(self, row: Mapping[str, Any], pair: HostClockPairV1) -> None:
        self.bridge.observe_host(pair)
        with self._lock:
            self._sync_bridge_generation()
            if self._rnti is None:
                self.counters["dci_unbound"] += 1
                return
            if int(row["rnti"]) != self._rnti:
                self.counters["dci_rejected_cross_ue"] += 1
                return
            if int(row["direction"]) != UL_DIRECTION:
                self.counters["dci_rejected_direction"] += 1
                return
            if int(row["mcs_table"]) != contract.UL_MCS_TABLE_ID:
                self.counters["dci_rejected_table"] += 1
                return
            if int(row["round"]) != 0:
                self.counters["dci_rejected_retransmission_round"] += 1
                return
            ndi = int(row["ndi"])
            if ndi not in (0, 1):
                self.counters["dci_malformed_ndi"] += 1
                return
            mcs = int(row["mcs"])
            if mcs < 0:
                self.counters["dci_malformed_mcs"] += 1
                return
            if mcs > contract.UL_MCS_INDEX_MAX:
                self.counters["dci_out_of_range_mcs_passed_to_selector"] += 1
            real = self._timed(row, pair, "dci")
            if real is None:
                return
            source = self.bridge.to_raw(real)
            if source is None:
                self.counters["dci_rejected_bridge_not_warm"] += 1
                return
            available = pair.raw_ns
            if source > available:
                self.counters["dci_rejected_negative_ingest"] += 1
                return
            self._dci_seq += 1
            candidate = SA.RawUeUlDciGrantCandidateV1(
                identity=contract.SampleIdentityV1(
                    self.session_uuid, self.ue_label, self._dci_seq),
                grant_identity=(
                    f"{self.session_uuid[:8]}:{self._dci_seq}:{row['time']}:"
                    f"{row['dci_frame']}.{row['dci_slot']}:h{row['harq_pid']}"),
                link_direction=contract.LinkDirection.UPLINK,
                mcs_table=int(row["mcs_table"]), mcs_index=mcs,
                harq_round=int(row["round"]), new_data_indicator=ndi,
                scheduler_policy_id=contract.UL_MCS_POLICY_ID,
                source=SOURCE_ID, source_timestamp_ns=source,
                available_timestamp_ns=available, clock_domain=CLOCK_DOMAIN)
            self._dci.append(candidate)
            self.counters["dci_accepted"] += 1
            self._publish()

    def on_rlc(self, row: Mapping[str, Any], pair: HostClockPairV1) -> None:
        self.bridge.observe_host(pair)
        with self._lock:
            self._sync_bridge_generation()
            rnti, ue_id = int(row["rnti"]), int(row["ue_id"])
            if self._rnti is None:
                self.counters["rlc_unbound"] += 1
                if len(self.unbound_ue_candidates) < 8:
                    self.unbound_ue_candidates.add((rnti, ue_id))
                return
            if (rnti, ue_id) != (self._rnti, self._oai_ue_id):
                self.counters["rlc_rejected_cross_ue"] += 1
                return
            backlog = int(row["bytes_in_buffer"])
            if backlog < 0:
                self.counters["rlc_malformed_negative_bytes"] += 1
                return
            real = self._timed(row, pair, "rlc")
            if real is None:
                return
            key = (rnti, ue_id, int(row["frame"]), int(row["slot"]))
            lcid = int(row["lcid"])
            if self._open is not None and self._open.key != key:
                self._finalize_locked(proof=pair)
            if self._open is None:
                self._open = _OpenTick(key=key)
            tick = self._open
            if lcid in tick.by_lcid:
                if tick.by_lcid[lcid] == backlog:
                    self.counters["rlc_duplicate_lcid_row"] += 1
                else:
                    self.counters["rlc_contradictory_lcid_row_latest_wins"] += 1
            tick.by_lcid[lcid] = backlog
            tick.source_real_ns = max(tick.source_real_ns, real)
            tick.last_tod = row["time"]

    def _finalize_locked(self, *, proof: HostClockPairV1) -> None:
        tick, self._open = self._open, None
        assert tick is not None
        source = self.bridge.to_raw(tick.source_real_ns)
        if source is None:
            self.counters["rlc_rejected_bridge_not_warm"] += 1
            return
        available = proof.raw_ns
        if source > available:
            self.counters["rlc_rejected_negative_ingest"] += 1
            return
        self._rlc_seq += 1
        self._rlc.append(SA.RawUeRlcBacklogSampleV1(
            identity=contract.SampleIdentityV1(
                self.session_uuid, self.ue_label, self._rlc_seq),
            backlog_bytes=sum(tick.by_lcid.values()),
            link_direction=contract.LinkDirection.UPLINK,
            source=f"{SOURCE_ID}:tick={tick.key[2]}.{tick.key[3]}"
                   f":tod={tick.last_tod}:lcids={len(tick.by_lcid)}",
            source_timestamp_ns=source, available_timestamp_ns=available,
            clock_domain=CLOCK_DOMAIN))
        self.counters["rlc_ticks_published"] += 1
        self._publish()


# ---------------------------------------------------------------------------
# Live csv reader (bounded drain thread)
# ---------------------------------------------------------------------------


class _ReaderState(enum.Enum):
    WAITING_HEADER = "WAITING_HEADER"
    STREAMING = "STREAMING"
    DEAD = "DEAD"


def csv_reader_argv(tracer_dir: Path, messages: Path, relay_port: int,
                    event: str) -> list[str]:
    return [str(Path(tracer_dir) / "csv"), "-d", str(messages), "-ip",
            "127.0.0.1", "-p", str(int(relay_port)), "-f", "-s", ",",
            "-t", "time", event, *EVENT_FIELDS[event]]


class LiveEventReaderV2:
    """One ``csv -f`` client; parses in its own thread, never in the actor."""

    def __init__(self, event: str, handler: Callable[[Mapping[str, Any],
                 HostClockPairV1], None], provider: UeTelemetryProviderV2, *,
                 audit_capacity: int = 262_144,
                 clock: Callable[[], HostClockPairV1] = read_host_clock_pair
                 ) -> None:
        if event not in EVENT_FIELDS:
            raise KeyError(event)
        self.event = event
        self.fields = EVENT_FIELDS[event]
        self._header = ",".join(self.fields)
        self._handler = handler
        self._provider = provider
        self._clock = clock
        self.state = _ReaderState.WAITING_HEADER
        self.counters: collections.Counter[str] = collections.Counter()
        self.audit: "queue.Queue[str]" = queue.Queue(maxsize=audit_capacity)
        self.process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = False

    # Feeding one line is the unit-testable core.
    def feed(self, line: str) -> None:
        pair = self._clock()
        text = line.rstrip("\n")
        try:
            self.audit.put_nowait(f"{pair.raw_ns},{pair.real_ns},{text}")
        except queue.Full:
            self.counters["audit_dropped"] += 1
        if self.state is _ReaderState.DEAD:
            return
        if self.state is _ReaderState.WAITING_HEADER:
            if text.startswith("connecting to ") or text == f"turning ON {self.event}":
                return
            if text == self._header:
                self.state = _ReaderState.STREAMING
                self._provider.reader_alive(self.event, True)
                return
            self.counters["malformed_header"] += 1
            self._die()
            return
        parts = text.split(",")
        if len(parts) != len(self.fields):
            self.counters["malformed_row"] += 1
            return
        row: dict[str, Any] = {"time": parts[0]}
        try:
            parse_time_of_day_us(parts[0])
            for name, value in zip(self.fields[1:], parts[1:]):
                row[name] = int(value)
        except ValueError:
            self.counters["malformed_row"] += 1
            return
        self.counters["rows"] += 1
        self._handler(row, pair)

    def _die(self) -> None:
        self.state = _ReaderState.DEAD
        self._provider.reader_alive(self.event, False)

    def eof(self) -> None:
        if not self._stopping:
            self.counters["unexpected_eof"] += 1
        if self.state is not _ReaderState.DEAD:
            self._die()

    def drain(self, stream: Iterable[str]) -> None:
        try:
            for line in stream:
                self.feed(line)
        finally:
            self.eof()

    def start(self, argv: Sequence[str], *, cwd: Path) -> None:
        self.process = subprocess.Popen(
            list(argv), cwd=str(cwd), text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1, start_new_session=True)
        assert self.process.stdout is not None
        self._thread = threading.Thread(
            target=self.drain, args=(self.process.stdout,), daemon=True,
            name=f"telemetry-{self.event}")
        self._thread.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        self._stopping = True
        if self.process is not None and self.process.poll() is None:
            self.process.send_signal(2)
            try:
                self.process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=timeout_s)
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        if self.process is not None and self.process.stdout is not None:
            self.process.stdout.close()


class AuditWriterV2:
    """Drains reader audit queues to disk off the decision path."""

    def __init__(self, readers: Sequence[LiveEventReaderV2], directory: Path) -> None:
        self._readers = tuple(readers)
        self._directory = Path(directory)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="telemetry-audit")

    def start(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=False)
        self._handles = {reader.event: (self._directory / f"{reader.event}_live.csv")
                         .open("x", encoding="utf-8") for reader in self._readers}
        self._thread.start()

    def _flush_once(self) -> None:
        for reader in self._readers:
            handle = self._handles[reader.event]
            while True:
                try:
                    handle.write(reader.audit.get_nowait() + "\n")
                except queue.Empty:
                    break
            handle.flush()

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            self._flush_once()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._flush_once()
        for handle in self._handles.values():
            handle.close()


# ---------------------------------------------------------------------------
# Decision-time assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RadioDecisionEvidenceV2:
    identity: contract.DecisionIdentityV1
    boundary: contract.DecisionBoundaryV1
    payload_enqueue_timestamp_ns: int
    snapshot_seq: int
    prior_ul_mcs: contract.PriorUlGrantObservationV1
    pre_action_rlc_backlog: contract.ScalarObservationV1
    mcs_age_ns: Optional[int]
    backlog_age_ns: Optional[int]
    fallback_reasons: Tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return not self.fallback_reasons


def open_decision(
    provider: UeTelemetryProviderV2, identity: contract.DecisionIdentityV1,
    *, clock: Callable[[], int] = raw_now_ns,
) -> Tuple[TelemetrySnapshotV2, contract.DecisionBoundaryV1, int]:
    """Snapshot first, then commit, then open; returns snapshot latency."""
    before = clock()
    snapshot = provider.snapshot()
    after = clock()
    commit = clock()
    while commit <= after:
        commit = clock()
    action_open = clock()
    while action_open <= commit:
        action_open = clock()
    boundary = contract.DecisionBoundaryV1(
        identity=identity, state_commit_timestamp_ns=commit,
        action_open_timestamp_ns=action_open, clock_domain=CLOCK_DOMAIN)
    return snapshot, boundary, after - before


def _slot_age(observation: contract.ScalarObservationV1,
              boundary: contract.DecisionBoundaryV1,
              kind: contract.MeasurementKind,
              freshness: contract.FreshnessPolicyV2) -> Tuple[Optional[int], Optional[str]]:
    """The registered guard's radio-slot rule (source<=available<=commit)."""
    meta = observation.metadata
    if not meta.valid or observation.value is None:
        return None, f"{kind.value}:MISSING:{observation.missing_reason}"
    if meta.available_timestamp_ns > boundary.state_commit_timestamp_ns:
        return None, f"{kind.value}:NOT_AVAILABLE_AT_COMMIT"
    age = boundary.action_open_timestamp_ns - meta.source_timestamp_ns
    if age < 0:
        return age, f"{kind.value}:FUTURE_SOURCE"
    if age > freshness.max_age_ns(kind):
        return age, f"{kind.value}:STALE:{age}"
    return age, None


def assemble_radio_evidence(
    snapshot: TelemetrySnapshotV2, *,
    identity: contract.DecisionIdentityV1,
    boundary: contract.DecisionBoundaryV1,
    payload_enqueue_timestamp_ns: int,
    freshness: contract.FreshnessPolicyV2 = TRAINING_FRESHNESS,
) -> RadioDecisionEvidenceV2:
    reasons: list[str] = []
    if boundary.identity != identity or boundary.clock_domain != CLOCK_DOMAIN:
        raise contract.MetadataError("decision boundary identity/clock mismatch")
    if not payload_enqueue_timestamp_ns > boundary.action_open_timestamp_ns:
        raise contract.MetadataError("payload enqueue must follow action open")
    if not snapshot.all_readers_alive:
        reasons.append("TELEMETRY_READER_DEAD")
    if not snapshot.bridge_warm:
        reasons.append("CLOCK_BRIDGE_INVALID_OR_WARMING")
    if snapshot.session_uuid != identity.session_uuid:
        reasons.append("CROSS_SESSION_TELEMETRY")
    if snapshot.ue_label is None or snapshot.ue_label != identity.ue_id:
        reasons.append("CROSS_UE_OR_UNBOUND_TELEMETRY")
    usable = not reasons
    dci = snapshot.dci if usable else ()
    rlc = snapshot.rlc if usable else ()
    try:
        prior = SA.select_prior_new_data_ul_mcs(dci, boundary)
    except SA.StateAdapterError as exc:
        reasons.append(f"UL_MCS_SELECTOR_REFUSED:{type(exc).__name__}")
        prior = SA.select_prior_new_data_ul_mcs((), boundary)
    try:
        backlog = SA.select_pre_action_rlc_backlog(
            rlc, boundary, payload_enqueue_timestamp_ns=payload_enqueue_timestamp_ns)
    except SA.StateAdapterError as exc:
        reasons.append(f"RLC_SELECTOR_REFUSED:{type(exc).__name__}")
        backlog = SA.select_pre_action_rlc_backlog(
            (), boundary, payload_enqueue_timestamp_ns=payload_enqueue_timestamp_ns)
    mcs_age, reason = _slot_age(
        prior.observation, boundary,
        contract.MeasurementKind.UE_PRIOR_NEW_DATA_UL_MCS_INDEX, freshness)
    if reason:
        reasons.append(reason)
    backlog_age, reason = _slot_age(
        backlog, boundary,
        contract.MeasurementKind.UE_PRE_ACTION_RLC_BACKLOG_BYTES, freshness)
    if reason:
        reasons.append(reason)
    return RadioDecisionEvidenceV2(
        identity=identity, boundary=boundary,
        payload_enqueue_timestamp_ns=payload_enqueue_timestamp_ns,
        snapshot_seq=snapshot.seq, prior_ul_mcs=prior,
        pre_action_rlc_backlog=backlog, mcs_age_ns=mcs_age,
        backlog_age_ns=backlog_age, fallback_reasons=tuple(reasons))


def require_admitted(evidence: RadioDecisionEvidenceV2) -> None:
    if not evidence.admitted:
        raise contract.ExternalFallbackRequired(
            "radio telemetry refused: " + "; ".join(evidence.fallback_reasons))


def act_or_fallback(evidence: RadioDecisionEvidenceV2,
                    actor_call: Callable[[RadioDecisionEvidenceV2], Any]) -> Any:
    """Call the actor only after an admitted radio guard; else raise."""
    require_admitted(evidence)
    return actor_call(evidence)
