#!/usr/bin/env python3
"""Read-only UE-state evidence audit for the SplitFusion Hybrid-SAC BSR feature.

This module answers one question from retained evidence only: can the OAI
T-tracer traces already on disk support a *causal, action-conditioned*
pre-action backlog feature in the Hybrid-SAC state?

It is deliberately inert on import.  Nothing is read, scanned, spawned or
initialised until :func:`audit_evidence_root` or :func:`main` is called.  The
module never writes into the evidence tree; the only optional write is the JSON
report the caller explicitly asks for via ``--json-out``.

Two UE sources are candidates, and the audit keeps them strictly apart:

``NRUE_MAC_RLC_BUFFER_STATUS``
    UE MAC's read of RLC buffer occupancy, emitted once per LCID per UL MAC
    tick *before* any grant is multiplexed.

``NRUE_MAC_BSR_STATUS``
    The BSR MAC-CE state emitted *after* logical-channel multiplexing, so its
    per-LCG byte counts are the residual left over once the current grant has
    already been filled.

The distinction matters because only a pre-enqueue observation can enter a
policy state without leaking the action into its own input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

AUDIT_ID = "splitfusion_hybrid_sac_ue_state_evidence_audit_v1"
AUDIT_VERSION = 1

# --------------------------------------------------------------------------
# Source-code provenance.  Line numbers are from the worktree copy of OAI that
# built the softmodems these traces came from; the audit reports them so the
# report can be re-checked against the tree rather than trusted.
# --------------------------------------------------------------------------

OAI_UE_SCHEDULER = "OAI/openairinterface5g/openair2/LAYER2/NR_MAC_UE/nr_ue_scheduler.c"
OAI_T_MESSAGES = "OAI/openairinterface5g/common/utils/T/T_messages.txt"
OAI_T_HEADER = "OAI/openairinterface5g/common/utils/T/T.h"
OAI_TRACER_CSV = "OAI/openairinterface5g/common/utils/T/tracer/csv.c"
OAI_UE_PHY_PROC = "OAI/openairinterface5g/openair1/SCHED_NR_UE/phy_procedures_nr_ue.c"
OAI_UE_MEASUREMENTS = (
    "OAI/openairinterface5g/openair1/PHY/NR_UE_ESTIMATION/nr_ue_measurements.c"
)
OAI_UE_CSI_RX = "OAI/openairinterface5g/openair1/PHY/NR_UE_TRANSPORT/csi_rx.c"

CODE_CITATIONS: Mapping[str, str] = {
    "rlc_buffer_emit": f"{OAI_UE_SCHEDULER}:1462 (in nr_update_rlc_buffers_status)",
    "rlc_buffer_caller": f"{OAI_UE_SCHEDULER}:2611 (nr_ue_ul_scheduler, before grant handling)",
    "bsr_status_emit": f"{OAI_UE_SCHEDULER}:2182 (in nr_ue_get_sdu_mac_ce_post)",
    "bsr_status_caller": f"{OAI_UE_SCHEDULER}:2561 (end of nr_ue_get_sdu)",
    "lcg_bytes_accumulate": f"{OAI_UE_SCHEDULER}:1511 (nr_update_bsr, += LCID_buffer_remain)",
    "lcg_bytes_decrement": f"{OAI_UE_SCHEDULER}:2414 (fill_mac_sdu, -= sdu_length)",
    "get_sdu_call": f"{OAI_UE_SCHEDULER}:2669 (nr_ue_get_sdu invoked only when a grant exists)",
    "bsr_index_encode": f"{OAI_UE_SCHEDULER}:2093,2108 (nr_locate_BsrIndexByBufferSize)",
    "t_timestamp_clock": f"{OAI_T_HEADER}:176,195,204 (T_HEADER uses CLOCK_REALTIME)",
    "t_csv_time_render": f"{OAI_TRACER_CSV}:32 (localtime; HH:MM:SS.microseconds, date dropped)",
    "t_message_rlc_desc": f"{OAI_T_MESSAGES}:248-251",
    "t_message_bsr_desc": f"{OAI_T_MESSAGES}:252-255",
    "t_message_pdcp_tx_sdu": f"{OAI_T_MESSAGES}:256-259",
    "t_message_rlc_tx_sdu": f"{OAI_T_MESSAGES}:260-263",
    "t_message_rlc_tx_dequeue": f"{OAI_T_MESSAGES}:264-267",
    # --- UE-side downlink channel measurement -------------------------
    "ue_phy_meas_message": f"{OAI_T_MESSAGES}:1510-1513 (UE_PHY_MEAS)",
    "ue_phy_meas_emit": (
        f"{OAI_UE_PHY_PROC}:411 (T_UE_PHY_MEAS, inside "
        f"nr_ue_measurement_procedures, guarded by #if T_TRACER)"
    ),
    "ue_phy_meas_gate": (
        f"{OAI_UE_PHY_PROC}:403,409 (emitted only when l == 2 and "
        f"nr_slot_rx == 0, i.e. once per 10 ms SFN frame at most)"
    ),
    "ue_phy_meas_caller": (
        f"{OAI_UE_PHY_PROC}:529 (the only call site, reached from the PDSCH "
        f"channel-estimation path, so a slot-0 PDSCH allocation is required)"
    ),
    "ue_phy_meas_snr_expression": (
        f"{OAI_UE_PHY_PROC}:415 (snr = rx_power_avg_dB[0] - n0_power_avg_dB, "
        f"emitted as plain integer dB, not x10)"
    ),
    "ue_measurements_compute": (
        f"{OAI_UE_MEASUREMENTS}:99-100 (rx_power_avg_dB and n0_power_avg_dB "
        f"via dB_fixed over the downlink channel estimates)"
    ),
    "ue_measurements_wideband_cqi": (
        f"{OAI_UE_MEASUREMENTS}:102 (wideband_cqi_avg = rx_power_avg_dB - "
        f"n0_power_avg_dB: the same expression as the emitted snr field)"
    ),
    "ue_measurements_rssi": f"{OAI_UE_MEASUREMENTS}:103-106 (rx_rssi_dBm)",
    "csi_rs_cqi_table": (
        f"{OAI_UE_CSI_RX}:666-695 (nr_csi_rs_cqi_estimation maps a precoded "
        f"SINR to a standardized 0-15 CQI index)"
    ),
    "csi_rs_cqi_call": f"{OAI_UE_CSI_RX}:926,958 (populated only on CSI-RS reception)",
}


class AuditError(Exception):
    """Base class for every refusal this audit can raise."""


class SchemaError(AuditError):
    """A CSV header or field value does not match the declared schema."""


class RunIsolationError(AuditError):
    """An operation tried to combine evidence from two different logical runs."""


class ClockDomainError(AuditError):
    """A join was attempted across clock domains without a measured bridge."""


class JoinAmbiguityError(AuditError):
    """A join key matched more than one counterpart inside the allowed skew."""


class ActionLeakageError(AuditError):
    """A source contaminated by the current action was offered as policy state."""


class ClockDomain(Enum):
    """Where a timestamp column's numbers come from."""

    #: HH:MM:SS.ffffff rendered by the T-tracer from CLOCK_REALTIME at the T()
    #: call site.  Same domain for every softmodem on one host; the date and the
    #: timezone offset are dropped by the renderer.
    T_TRACER_REALTIME_LOCAL = "T_TRACER_REALTIME_LOCAL"
    #: Python ``time.time()`` epoch seconds written by the traffic generator.
    #: Same underlying clock as above, but the printed forms cannot be compared
    #: without reconstructing the trace date and UTC offset.
    EPOCH_WALL_SECONDS = "EPOCH_WALL_SECONDS"
    #: Collector ingest epoch/monotonic nanoseconds prepended by the live
    #: capture wrapper.  Explicitly *not* the RF application instant.
    COLLECTOR_INGEST_NS = "COLLECTOR_INGEST_NS"
    UNRESOLVED = "UNRESOLVED"


class SourceVisibility(Enum):
    """Which node could actually observe the quantity at runtime."""

    UE = "UE"
    GNB = "GNB"
    TRAFFIC_GENERATOR = "TRAFFIC_GENERATOR"
    UNRESOLVED = "UNRESOLVED"


class EnqueueOrdering(Enum):
    """Where a sample sits relative to the current grant's multiplexing."""

    #: Read before the MAC fills the current grant, so it still shows the full
    #: RLC occupancy for this tick.
    PRE_MULTIPLEX_WITHIN_MAC_SLOT = "PRE_MULTIPLEX_WITHIN_MAC_SLOT"
    #: What is left after the current grant has been filled.
    POST_MULTIPLEX_RESIDUAL = "POST_MULTIPLEX_RESIDUAL"
    UNRESOLVED = "UNRESOLVED"


class PreActionQualification(Enum):
    """Whether a source proves the pre-action read ordering the policy needs."""

    QUALIFIED = "PRE_ACTION_BSR_SOURCE_QUALIFIED"
    UNRESOLVED = "PRE_ACTION_BSR_SOURCE_UNRESOLVED"
    DISQUALIFIED_ACTION_CONTAMINATED = "PRE_ACTION_BSR_SOURCE_DISQUALIFIED_ACTION_CONTAMINATED"


class CoverageVerdict(Enum):
    """The single final status the audit must assign."""

    ACTION_CONDITIONED = "ADEQUATE_FOR_ACTION_CONDITIONED_QUEUE_MODEL"
    CARRIER_AND_NORMALIZATION_ONLY = "ADEQUATE_FOR_CARRIER_AND_NORMALIZATION_AUDIT_ONLY"
    INSUFFICIENT = "INSUFFICIENT_OR_CAUSALLY_UNRESOLVED"


# --------------------------------------------------------------------------
# Declared schemas.  Headers are compared exactly: a silent column rename or
# reorder must fail the audit rather than be absorbed by name-based lookup.
# --------------------------------------------------------------------------

RLC_BUFFER_HEADER: Tuple[str, ...] = (
    "time", "rnti", "ue_id", "frame", "slot", "lcid", "lcgid",
    "bytes_in_buffer", "bj", "pbr", "priority",
)

BSR_STATUS_HEADER: Tuple[str, ...] = (
    "time", "rnti", "ue_id", "frame", "slot", "bsr_type", "trigger_mask",
    "bsr_sent", "padding_len", "num_sdus", "sdu_bytes",
    "lcg0_bytes", "lcg1_bytes", "lcg2_bytes", "lcg3_bytes",
    "lcg4_bytes", "lcg5_bytes", "lcg6_bytes", "lcg7_bytes",
    "bsr_lcg_id", "bsr_index",
    "bsr_long0_index", "bsr_long1_index", "bsr_long2_index", "bsr_long3_index",
    "bsr_long4_index", "bsr_long5_index", "bsr_long6_index", "bsr_long7_index",
)

DCI_GRANT_HEADER: Tuple[str, ...] = (
    "time", "direction", "dci_format", "rnti_type", "rnti", "dci_frame",
    "dci_slot", "sched_frame", "sched_slot", "mcs", "mcs_table", "rb_start",
    "rb_size", "start_symbol", "nr_symbols", "tbs", "harq_pid", "ndi", "rv",
    "round", "qam_mod_order", "target_code_rate", "tpc", "n_cce", "N_cce",
)

TX_BITS_HEADER: Tuple[str, ...] = (
    "time", "frame", "slot", "rnti", "rb_size", "rb_start", "qam_mod_order",
    "mcs_index", "number_of_bits",
)

PUSCH_POWER_HEADER: Tuple[str, ...] = (
    "time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
    "txpower_calc", "rbSize", "mcs", "rssi",
)

#: ``UE_PHY_MEAS``. Header is the tracer's ``time`` column followed by the
#: FORMAT field names at ``T_messages.txt:1513``.
UE_PHY_MEAS_HEADER: Tuple[str, ...] = (
    "time", "eNB_ID", "frame", "subframe", "rsrp", "rssi", "snr",
    "rx_power", "noise_power", "w_cqi", "freq_offset",
)

#: Application-enqueue candidates. Both are PDCP/RLC *ingress* events carrying
#: a radio-bearer id and a monotonic source timestamp.
PDCP_TX_SDU_HEADER: Tuple[str, ...] = (
    "time", "mono_sec", "mono_nsec", "ue_id", "rb_id", "sdu_bytes",
)
RLC_TX_SDU_HEADER: Tuple[str, ...] = (
    "time", "mono_sec", "mono_nsec", "ue_id", "rb_id", "sdu_bytes",
)

#: ``NR_RLC_TX_DEQUEUE`` is *service*, not ingress: it fires when a PDU is
#: handed down to MAC for a grant, which ends the RLC queue wait. It carries an
#: ``lcid`` and ``pdu_bytes`` rather than a bearer id and SDU bytes.
RLC_TX_DEQUEUE_HEADER: Tuple[str, ...] = (
    "time", "mono_sec", "mono_nsec", "ue_id", "lcid", "pdu_bytes",
)

#: The per-decision record a future runtime carrier must write for the causal
#: ordering to be checkable at all. No retained run has one.
DECISION_RECORD_HEADER: Tuple[str, ...] = (
    "decision_id", "frame_index", "ue_id", "rnti", "lcid", "rb_id",
    "t_measure_mono_ns", "t_available_to_agent_mono_ns",
    "t_state_commit_mono_ns", "t_action_mono_ns",
    "t_current_payload_enqueue_mono_ns",
)

SENDER_HEADER: Tuple[str, ...] = (
    "wall_time_s", "elapsed_s", "frame_index", "chunk_index", "chunk_bytes",
    "frame_bytes", "period_s", "scheduled_frame_time_s", "send_lag_ms",
)

LCG_COUNT = 8

#: 1024 SFN frames of 10 ms each.  ``(frame, slot)`` repeats on this period, so
#: any key-only join over a run longer than this is many-to-many by
#: construction and must be disambiguated by time.
SFN_WRAP_MICROSECONDS = 1024 * 10_000

MICROSECONDS_PER_DAY = 86_400_000_000

#: A backwards jump smaller than this is treated as event reordering inside one
#: day, not as a midnight rollover.  Twelve hours cannot arise from tracer
#: jitter on runs of this length.
MIDNIGHT_WRAP_GUARD_MICROSECONDS = 12 * 3600 * 1_000_000

#: The scale currently frozen into the Hybrid-SAC normalization spec
#: (``empirical_contextual_environment.py:309``, ``bsr_log1p_scale=1.0``),
#: applied by ``state_reward_transition_contract.py:5827`` as
#: ``clip(log1p(bsr_bytes) / scale, 0, 1)``.
DEPLOYED_BSR_LOG1P_SCALE = 1.0

# --------------------------------------------------------------------------
# Payload authority.
#
# The action range is the *frozen FCOS 72-action catalogue*, not the legacy
# PERMODEL knob matrix.  The legacy pair (49_400, 2_835_000) is retired: it
# named a range that no longer exists, and it placed the retained 12.5 kB and
# 25 kB traffic points *below* the action range when in fact they sit inside
# its low end.  The values below are checked against the authoritative files by
# :func:`verify_payload_authority`; they are not trusted on sight.
# --------------------------------------------------------------------------

#: Path, relative to the repository root, of the frozen 72-action catalogue.
SPLITFUSION_ACTION_CATALOG_RELPATH = (
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json"
)

#: Path, relative to the repository root, of the Run-3 modeled support contract.
RUN3_MODELED_SUPPORT_RELPATH = (
    "rl_agent/splitfusion_hybrid_sac_v1/modeled_smoke_support.py"
)

#: Name of the Run-3 support constant re-read by :func:`verify_payload_authority`.
RUN3_MODELED_SUPPORT_SYMBOL = "ALL_PROFILE_TOTAL_TRANSMITTED_BYTES_SUPPORT"

SPLITFUSION_ACTION_COUNT = 72

#: Per-frame zstd median payload of the smallest and largest of the 72 frozen
#: actions: action 71 ``split_ae32_uint4_q9800`` and action 0
#: ``split_noae_uint8_q0000``.
SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES = 6_229
SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES = 3_568_326

#: Inclusive ``total_transmitted_bytes`` support the Run-3 contextual surrogate
#: was actually fitted on.  Narrower than the catalogue at both ends, so a
#: statement about "the action range" and one about "modeled support" are not
#: interchangeable and the audit keeps them apart.
RUN3_MODELED_SUPPORT_MIN_BYTES = 6_423
RUN3_MODELED_SUPPORT_MAX_BYTES = 427_605

#: Values that must never come back. A test asserts none of them is reachable
#: as a payload-authority constant, so a silent revert to the PERMODEL range
#: fails loudly instead of quietly re-widening the bound.
RETIRED_PERMODEL_PAYLOAD_BYTES: Tuple[int, ...] = (49_400, 2_835_000)


class PayloadAuthorityError(AuditError):
    """A payload constant disagrees with the file that is authoritative for it."""


def _catalog_payload_bounds(catalog_path: Path) -> Tuple[int, int, int]:
    """Re-derive ``(count, min, max)`` zstd median payload from the catalogue."""
    data = json.loads(catalog_path.read_text())
    profiles = data["profiles"]
    medians = [int(entry["payload"]["zstd_median_bytes"]) for entry in profiles]
    return len(profiles), min(medians), max(medians)


def _run3_support_bounds(support_path: Path) -> Tuple[int, int]:
    """Read the Run-3 support tuple by parsing, not importing.

    ``modeled_smoke_support`` imports the heavy Hybrid-SAC surfaces at module
    scope, and this audit must stay importable with nothing but the standard
    library.  The literal is recovered from the AST instead.
    """
    import ast

    tree = ast.parse(support_path.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.AnnAssign, ast.Assign)):
            continue
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else list(node.targets)
        )
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == RUN3_MODELED_SUPPORT_SYMBOL
                and node.value is not None
            ):
                low, high = ast.literal_eval(node.value)
                return int(low), int(high)
    raise PayloadAuthorityError(
        f"{support_path}: {RUN3_MODELED_SUPPORT_SYMBOL} not found"
    )


def verify_payload_authority(repo_root: Path) -> Dict[str, object]:
    """Check every payload constant against its authoritative file.

    Raises :class:`PayloadAuthorityError` on any disagreement, so the audit
    cannot keep quoting a range the catalogue no longer has.  Performs file I/O
    and is therefore never called on import.
    """
    catalog_path = repo_root / SPLITFUSION_ACTION_CATALOG_RELPATH
    support_path = repo_root / RUN3_MODELED_SUPPORT_RELPATH
    if not catalog_path.is_file():
        raise PayloadAuthorityError(f"action catalogue missing at {catalog_path}")
    if not support_path.is_file():
        raise PayloadAuthorityError(f"Run-3 support missing at {support_path}")

    count, low, high = _catalog_payload_bounds(catalog_path)
    run3_low, run3_high = _run3_support_bounds(support_path)

    mismatches: List[str] = []
    if count != SPLITFUSION_ACTION_COUNT:
        mismatches.append(f"action count {count} != {SPLITFUSION_ACTION_COUNT}")
    if low != SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES:
        mismatches.append(
            f"catalogue min {low} != {SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES}"
        )
    if high != SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES:
        mismatches.append(
            f"catalogue max {high} != {SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES}"
        )
    if run3_low != RUN3_MODELED_SUPPORT_MIN_BYTES:
        mismatches.append(f"Run-3 min {run3_low} != {RUN3_MODELED_SUPPORT_MIN_BYTES}")
    if run3_high != RUN3_MODELED_SUPPORT_MAX_BYTES:
        mismatches.append(f"Run-3 max {run3_high} != {RUN3_MODELED_SUPPORT_MAX_BYTES}")
    if mismatches:
        raise PayloadAuthorityError(
            "payload authority drifted from its source files: "
            + "; ".join(mismatches)
        )
    return {
        "action_catalog_relpath": SPLITFUSION_ACTION_CATALOG_RELPATH,
        "action_catalog_sha256": sha256_file(catalog_path),
        "action_count": count,
        "action_payload_min_bytes": low,
        "action_payload_max_bytes": high,
        "run3_support_relpath": RUN3_MODELED_SUPPORT_RELPATH,
        "run3_support_min_bytes": run3_low,
        "run3_support_max_bytes": run3_high,
        "retired_permodel_payload_bytes": list(RETIRED_PERMODEL_PAYLOAD_BYTES),
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FileProvenance:
    """Identity of one evidence file, bound by content hash."""

    relative_path: str
    size_bytes: int
    sha256: str
    row_count: int
    header: Tuple[str, ...]

    def to_json(self) -> Dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "header": list(self.header),
        }


def sha256_file(path: Path) -> str:
    """Content hash of ``path``, streamed so large traces do not load fully."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path, root: Path) -> FileProvenance:
    """Hash, size, header and data-row count for one CSV, without mutating it."""
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration:
            header = ()
        rows = sum(1 for _ in reader)
    return FileProvenance(
        relative_path=str(path.relative_to(root)),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        row_count=rows,
        header=header,
    )


# --------------------------------------------------------------------------
# Logical runs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalRun:
    """One measurement run, identified by directory structure rather than name.

    The run root is the directory that *contains* ``ttracer/``.  Two files only
    belong to the same logical run when they resolve to the same run root; the
    audit refuses to join on filename similarity alone.
    """

    run_id: str
    root: Path
    ue_csv_dir: Path
    gnb_csv_dir: Optional[Path]
    traffic_sender: Optional[Path]

    def ue_csv(self, name: str) -> Optional[Path]:
        candidate = self.ue_csv_dir / name
        return candidate if candidate.is_file() else None

    def gnb_csv(self, name: str) -> Optional[Path]:
        if self.gnb_csv_dir is None:
            return None
        candidate = self.gnb_csv_dir / name
        return candidate if candidate.is_file() else None


def logical_run_root(csv_path: Path) -> Path:
    """Directory owning ``csv_path``'s run: the parent of its ``ttracer`` dir."""
    parts = csv_path.parts
    try:
        index = len(parts) - 1 - parts[::-1].index("ttracer")
    except ValueError as exc:
        raise RunIsolationError(
            f"cannot establish a logical run identity for {csv_path}: no "
            f"'ttracer' component in its path"
        ) from exc
    if index == 0:
        raise RunIsolationError(
            f"cannot establish a logical run identity for {csv_path}: "
            f"'ttracer' has no parent directory"
        )
    return Path(*parts[:index])


def require_same_run(left: Path, right: Path) -> Path:
    """Return the shared run root, or refuse the pairing."""
    left_root = logical_run_root(left)
    right_root = logical_run_root(right)
    if left_root != right_root:
        raise RunIsolationError(
            f"refusing a cross-run join: {left} belongs to {left_root} but "
            f"{right} belongs to {right_root}"
        )
    return left_root


def discover_runs(evidence_root: Path) -> List[LogicalRun]:
    """Find every logical run that retains a UE BSR trace, in sorted order."""
    runs: Dict[str, LogicalRun] = {}
    pattern = "**/ttracer/ue/csv/NRUE_MAC_BSR_STATUS.csv"
    for bsr_path in sorted(evidence_root.glob(pattern)):
        root = logical_run_root(bsr_path)
        run_id = str(root.relative_to(evidence_root))
        gnb_dir = root / "ttracer" / "gnb" / "csv"
        sender = root / "traffic" / "sender.csv"
        runs[run_id] = LogicalRun(
            run_id=run_id,
            root=root,
            ue_csv_dir=bsr_path.parent,
            gnb_csv_dir=gnb_dir if gnb_dir.is_dir() else None,
            traffic_sender=sender if sender.is_file() else None,
        )
    return [runs[key] for key in sorted(runs)]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_tracer_time(text: str) -> int:
    """``HH:MM:SS.ffffff`` from the T-tracer CSV sink to microseconds-of-day.

    The renderer at ``tracer/csv.c:32`` drops the date and the timezone, so this
    is a within-day offset only.  Callers must not treat it as an epoch.
    """
    field_text = text.strip()
    parts = field_text.split(":")
    if len(parts) != 3:
        raise SchemaError(f"malformed tracer timestamp {text!r}")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_text, _, fraction_text = parts[2].partition(".")
        seconds = int(seconds_text)
        micros = int(fraction_text.ljust(6, "0")) if fraction_text else 0
    except ValueError as exc:
        raise SchemaError(f"malformed tracer timestamp {text!r}") from exc
    if not (0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60):
        raise SchemaError(f"out-of-range tracer timestamp {text!r}")
    if not 0 <= micros < 1_000_000:
        raise SchemaError(f"out-of-range tracer timestamp {text!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1_000_000 + micros


def unwrap_tracer_times(times: Sequence[int]) -> List[int]:
    """Add a day to every sample after a midnight rollover.

    Only drops larger than :data:`MIDNIGHT_WRAP_GUARD_MICROSECONDS` count as a
    rollover, so ordinary event reordering is left alone rather than being
    silently promoted to a 24-hour jump.
    """
    unwrapped: List[int] = []
    offset = 0
    previous: Optional[int] = None
    for value in times:
        if previous is not None and value < previous - MIDNIGHT_WRAP_GUARD_MICROSECONDS:
            offset += MICROSECONDS_PER_DAY
        unwrapped.append(value + offset)
        previous = value
    return unwrapped


def _require_header(path: Path, actual: Sequence[str], expected: Sequence[str]) -> None:
    if tuple(actual) != tuple(expected):
        raise SchemaError(
            f"{path}: unexpected header\n  expected {list(expected)}\n"
            f"  found    {list(actual)}"
        )


def _nonneg_int(path: Path, row_index: int, name: str, text: str) -> int:
    try:
        value = int(text)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"{path} row {row_index}: {name}={text!r} is not an integer"
        ) from exc
    if value < 0:
        raise SchemaError(
            f"{path} row {row_index}: {name}={value} is negative; byte counts "
            f"must be non-negative and are never clamped by this audit"
        )
    return value


def _signed_int(path: Path, row_index: int, name: str, text: str) -> int:
    try:
        return int(text)
    except (TypeError, ValueError) as exc:
        raise SchemaError(
            f"{path} row {row_index}: {name}={text!r} is not an integer"
        ) from exc


@dataclass(frozen=True)
class RlcBufferRow:
    time_us: int
    frame: int
    slot: int
    lcid: int
    lcgid: int
    bytes_in_buffer: int


@dataclass(frozen=True)
class BsrStatusRow:
    time_us: int
    frame: int
    slot: int
    bsr_type: int
    bsr_sent: int
    num_sdus: int
    sdu_bytes: int
    lcg_bytes: Tuple[int, ...]
    bsr_lcg_id: int
    bsr_index: int

    @property
    def lcg_total_bytes(self) -> int:
        """Total post-multiplex residual across every logical channel group."""
        return sum(self.lcg_bytes)


def read_rlc_buffer_rows(path: Path) -> List[RlcBufferRow]:
    """Parse ``NRUE_MAC_RLC_BUFFER_STATUS.csv`` with an exact schema check."""
    rows: List[RlcBufferRow] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        _require_header(path, header, RLC_BUFFER_HEADER)
        for index, record in enumerate(reader, start=1):
            if len(record) != len(RLC_BUFFER_HEADER):
                raise SchemaError(
                    f"{path} row {index}: expected {len(RLC_BUFFER_HEADER)} "
                    f"fields, found {len(record)}"
                )
            rows.append(
                RlcBufferRow(
                    time_us=parse_tracer_time(record[0]),
                    frame=_nonneg_int(path, index, "frame", record[3]),
                    slot=_nonneg_int(path, index, "slot", record[4]),
                    lcid=_nonneg_int(path, index, "lcid", record[5]),
                    lcgid=_signed_int(path, index, "lcgid", record[6]),
                    bytes_in_buffer=_nonneg_int(
                        path, index, "bytes_in_buffer", record[7]
                    ),
                )
            )
    times = unwrap_tracer_times([row.time_us for row in rows])
    return [
        RlcBufferRow(
            time_us=when,
            frame=row.frame,
            slot=row.slot,
            lcid=row.lcid,
            lcgid=row.lcgid,
            bytes_in_buffer=row.bytes_in_buffer,
        )
        for row, when in zip(rows, times)
    ]


def read_bsr_status_rows(path: Path) -> List[BsrStatusRow]:
    """Parse ``NRUE_MAC_BSR_STATUS.csv`` with an exact schema check."""
    rows: List[BsrStatusRow] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        _require_header(path, header, BSR_STATUS_HEADER)
        lcg_start = BSR_STATUS_HEADER.index("lcg0_bytes")
        for index, record in enumerate(reader, start=1):
            if len(record) != len(BSR_STATUS_HEADER):
                raise SchemaError(
                    f"{path} row {index}: expected {len(BSR_STATUS_HEADER)} "
                    f"fields, found {len(record)}"
                )
            lcg_bytes = tuple(
                _nonneg_int(path, index, f"lcg{group}_bytes", record[lcg_start + group])
                for group in range(LCG_COUNT)
            )
            rows.append(
                BsrStatusRow(
                    time_us=parse_tracer_time(record[0]),
                    frame=_nonneg_int(path, index, "frame", record[3]),
                    slot=_nonneg_int(path, index, "slot", record[4]),
                    bsr_type=_signed_int(path, index, "bsr_type", record[5]),
                    bsr_sent=_signed_int(path, index, "bsr_sent", record[7]),
                    num_sdus=_nonneg_int(path, index, "num_sdus", record[9]),
                    sdu_bytes=_nonneg_int(path, index, "sdu_bytes", record[10]),
                    lcg_bytes=lcg_bytes,
                    bsr_lcg_id=_signed_int(path, index, "bsr_lcg_id", record[19]),
                    bsr_index=_nonneg_int(path, index, "bsr_index", record[20]),
                )
            )
    times = unwrap_tracer_times([row.time_us for row in rows])
    return [
        BsrStatusRow(
            time_us=when,
            frame=row.frame,
            slot=row.slot,
            bsr_type=row.bsr_type,
            bsr_sent=row.bsr_sent,
            num_sdus=row.num_sdus,
            sdu_bytes=row.sdu_bytes,
            lcg_bytes=row.lcg_bytes,
            bsr_lcg_id=row.bsr_lcg_id,
            bsr_index=row.bsr_index,
        )
        for row, when in zip(rows, times)
    ]


def read_simple_rows(
    path: Path, expected_header: Sequence[str]
) -> List[Dict[str, str]]:
    """Read a tracer CSV as raw strings after an exact header check."""
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        _require_header(path, header, expected_header)
        out: List[Dict[str, str]] = []
        for index, record in enumerate(reader, start=1):
            if len(record) != len(expected_header):
                raise SchemaError(
                    f"{path} row {index}: expected {len(expected_header)} "
                    f"fields, found {len(record)}"
                )
            out.append(dict(zip(expected_header, record)))
        return out


# --------------------------------------------------------------------------
# Backlog aggregation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RlcTick:
    """One UL MAC tick's complete RLC view, summed across logical channels."""

    time_us: int
    frame: int
    slot: int
    total_bytes: int
    per_lcg_bytes: Tuple[int, ...]
    per_lcid_bytes: Tuple[Tuple[int, int], ...]


def rlc_ticks(rows: Sequence[RlcBufferRow]) -> List[RlcTick]:
    """Group per-LCID rows into per-tick totals.

    ``nr_update_rlc_buffers_status`` emits one row per active LCID inside a
    single call, in a fixed LCID order and all sharing ``(frame, slot)``.  A
    tick therefore ends when ``(frame, slot)`` changes or when an LCID repeats,
    which starts the next sweep of the list.  Grouping on the timestamp instead
    would be wrong: the rows of one call can straddle a microsecond boundary,
    which would split a single call into two apparent ticks.

    Total backlog is the plain sum over the tick's LCIDs.  Rows reading zero are
    kept, so an empty buffer stays a measured zero rather than becoming a gap.
    """
    ticks: List[RlcTick] = []
    current: List[RlcBufferRow] = []

    def flush() -> None:
        if not current:
            return
        per_lcg = [0] * LCG_COUNT
        for member in current:
            if 0 <= member.lcgid < LCG_COUNT:
                per_lcg[member.lcgid] += member.bytes_in_buffer
        ticks.append(
            RlcTick(
                time_us=current[0].time_us,
                frame=current[0].frame,
                slot=current[0].slot,
                total_bytes=sum(member.bytes_in_buffer for member in current),
                per_lcg_bytes=tuple(per_lcg),
                per_lcid_bytes=tuple(
                    sorted((member.lcid, member.bytes_in_buffer) for member in current)
                ),
            )
        )
        current.clear()

    for row in rows:
        if current and (
            (row.frame, row.slot) != (current[0].frame, current[0].slot)
            or any(member.lcid == row.lcid for member in current)
        ):
            flush()
        current.append(row)
    flush()
    return ticks


def min_same_key_recurrence_us(
    records: Sequence[object],
    key_of: Callable[[object], Tuple[int, ...]],
    time_of: Callable[[object], int],
) -> Optional[int]:
    """Smallest observed gap between two records sharing a join key.

    A join window wider than half this gap could reach two counterparts, so
    this is what a skew bound has to be checked against.  It is measured from
    the evidence rather than assumed from the SFN period, because the emitter's
    own cadence, not just the frame counter, decides when a key comes back.
    """
    buckets: Dict[Tuple[int, ...], List[int]] = {}
    for record in records:
        buckets.setdefault(key_of(record), []).append(time_of(record))
    smallest: Optional[int] = None
    for times in buckets.values():
        if len(times) < 2:
            continue
        times.sort()
        for earlier, later in zip(times, times[1:]):
            gap = later - earlier
            if smallest is None or gap < smallest:
                smallest = gap
    return smallest


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    """Nearest-rank percentile.

    Nearest-rank is used rather than an interpolating definition so that a
    percentile of integer byte counts stays an exactly-observed byte count and
    the result is reproducible across Python versions.
    """
    if not values:
        return None
    if not 0.0 < quantile <= 1.0:
        raise AuditError(f"quantile must be in (0, 1], got {quantile}")
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered))
    return float(ordered[min(rank, len(ordered)) - 1])


@dataclass(frozen=True)
class DistributionStats:
    count: int
    zero_count: int
    zero_fraction: Optional[float]
    p50: Optional[float]
    p90: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    maximum: Optional[float]

    def to_json(self) -> Dict[str, object]:
        return {
            "count": self.count,
            "zero_count": self.zero_count,
            "zero_fraction": self.zero_fraction,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.maximum,
        }


def describe_distribution(values: Sequence[float]) -> DistributionStats:
    """Zero mass and tail quantiles, with ``None`` rather than a filled-in 0."""
    count = len(values)
    zeros = sum(1 for value in values if value == 0)
    return DistributionStats(
        count=count,
        zero_count=zeros,
        zero_fraction=(zeros / count) if count else None,
        p50=percentile(values, 0.50),
        p90=percentile(values, 0.90),
        p95=percentile(values, 0.95),
        p99=percentile(values, 0.99),
        maximum=float(max(values)) if count else None,
    )


def sampling_interval_stats(times_us: Sequence[int]) -> DistributionStats:
    """Inter-sample interval distribution, used to justify join skew bounds."""
    deltas = [
        float(later - earlier)
        for earlier, later in zip(times_us, times_us[1:])
        if later >= earlier
    ]
    return describe_distribution(deltas)


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentResult:
    """Outcome of one within-run join between two evidence sources."""

    left_source: str
    right_source: str
    join_keys: Tuple[str, ...]
    left_domain: ClockDomain
    right_domain: ClockDomain
    domain_relation: str
    max_skew_us: int
    max_skew_justification: str
    min_same_key_recurrence_us: Optional[int]
    left_rows: int
    right_rows: int
    matched_rows: int
    unmatched_left_rows: int
    reused_right_rows: int
    skew_p50_us: Optional[float]
    skew_p95_us: Optional[float]
    skew_p99_us: Optional[float]
    skew_max_us: Optional[float]

    @property
    def coverage(self) -> Optional[float]:
        if not self.left_rows:
            return None
        return self.matched_rows / self.left_rows

    def to_json(self) -> Dict[str, object]:
        return {
            "left_source": self.left_source,
            "right_source": self.right_source,
            "join_keys": list(self.join_keys),
            "left_clock_domain": self.left_domain.value,
            "right_clock_domain": self.right_domain.value,
            "domain_relation": self.domain_relation,
            "max_skew_us": self.max_skew_us,
            "max_skew_justification": self.max_skew_justification,
            "min_same_key_recurrence_us": self.min_same_key_recurrence_us,
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "matched_rows": self.matched_rows,
            "unmatched_left_rows": self.unmatched_left_rows,
            "reused_right_rows": self.reused_right_rows,
            "coverage": self.coverage,
            "signed_skew_us": {
                "p50": self.skew_p50_us,
                "p95": self.skew_p95_us,
                "p99": self.skew_p99_us,
                "max_abs": self.skew_max_us,
            },
        }


def align_one_to_one(
    *,
    left_source: str,
    right_source: str,
    left: Sequence[object],
    right: Sequence[object],
    key_of: Callable[[object], Tuple[int, ...]],
    time_of: Callable[[object], int],
    right_key_of: Optional[Callable[[object], Tuple[int, ...]]] = None,
    right_time_of: Optional[Callable[[object], int]] = None,
    join_keys: Sequence[str],
    left_domain: ClockDomain,
    right_domain: ClockDomain,
    max_skew_us: int,
    max_skew_justification: str,
    bridge_us: Optional[int] = None,
) -> AlignmentResult:
    """Join ``left`` to ``right`` on an exact key plus a bounded time window.

    The key alone is never trusted: ``(frame, slot)`` repeats once per SFN wrap
    at most, and in practice on whatever slower beat the emitter runs on, so a
    key-only join over a run longer than that is many-to-many.  Many-to-many is
    rejected up front by checking the requested window against the *measured*
    smallest same-key recurrence in the right-hand source: a window wider than
    half that gap is refused outright, before a single row is matched, so no
    left row is ever quietly snapped to the nearer of two counterparts.  The
    per-key check further down is a backstop that this guard should already
    make unreachable.

    ``key_of``/``time_of`` read the left-hand records; ``right_key_of`` and
    ``right_time_of`` default to them and exist only because the two sides may
    be different record types.

    ``bridge_us`` is a *measured* offset added to right-hand timestamps.  When
    the two domains differ and no bridge is supplied the join is refused.
    """
    right_key_of = right_key_of or key_of
    right_time_of = right_time_of or time_of
    if left_domain != right_domain and bridge_us is None:
        raise ClockDomainError(
            f"refusing to join {left_source} ({left_domain.value}) to "
            f"{right_source} ({right_domain.value}) without a measured clock "
            f"bridge"
        )
    if max_skew_us <= 0:
        raise AuditError(f"max_skew_us must be positive, got {max_skew_us}")
    recurrence = min_same_key_recurrence_us(right, right_key_of, right_time_of)
    if recurrence is not None and max_skew_us * 2 >= recurrence:
        raise AuditError(
            f"{left_source}->{right_source}: max_skew_us={max_skew_us} is at "
            f"least half the measured smallest same-key recurrence of "
            f"{recurrence} us in {right_source}; the join key would stay "
            f"ambiguous"
        )

    offset = bridge_us or 0
    buckets: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    for index, record in enumerate(right):
        buckets.setdefault(right_key_of(record), []).append(
            (right_time_of(record) + offset, index)
        )
    for entries in buckets.values():
        entries.sort()

    matched = 0
    skews: List[int] = []
    right_use: Dict[int, int] = {}
    for record in left:
        key = key_of(record)
        when = time_of(record)
        candidates = [
            (right_time, index)
            for right_time, index in buckets.get(key, ())
            if abs(right_time - when) <= max_skew_us
        ]
        if not candidates:
            continue
        if len(candidates) > 1:
            raise JoinAmbiguityError(
                f"{left_source}->{right_source}: key {key} matched "
                f"{len(candidates)} rows within {max_skew_us} us; the join is "
                f"ambiguous and is refused rather than resolved by proximity"
            )
        right_time, index = candidates[0]
        matched += 1
        skews.append(right_time - when)
        right_use[index] = right_use.get(index, 0) + 1

    absolute = [float(abs(value)) for value in skews]
    signed = [float(value) for value in skews]
    return AlignmentResult(
        left_source=left_source,
        right_source=right_source,
        join_keys=tuple(join_keys),
        left_domain=left_domain,
        right_domain=right_domain,
        domain_relation=(
            "IDENTICAL" if left_domain == right_domain else "BRIDGED_BY_MEASURED_OFFSET"
        ),
        max_skew_us=max_skew_us,
        max_skew_justification=max_skew_justification,
        min_same_key_recurrence_us=recurrence,
        left_rows=len(left),
        right_rows=len(right),
        matched_rows=matched,
        unmatched_left_rows=len(left) - matched,
        reused_right_rows=sum(1 for count in right_use.values() if count > 1),
        skew_p50_us=percentile(signed, 0.50),
        skew_p95_us=percentile(signed, 0.95),
        skew_p99_us=percentile(signed, 0.99),
        skew_max_us=percentile(absolute, 1.0),
    )


# --------------------------------------------------------------------------
# Source classification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDescriptor:
    """What a source measures, who can see it, and when it is emitted."""

    name: str
    measured_object: str
    byte_semantics: str
    group_coverage: str
    emission_event: str
    ordering: EnqueueOrdering
    ordering_evidence: str
    visibility: SourceVisibility
    clock_domain: ClockDomain
    zero_meaning: str
    ue_runtime_available: bool
    can_contain_current_payload: bool

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "measured_object": self.measured_object,
            "byte_semantics": self.byte_semantics,
            "group_coverage": self.group_coverage,
            "emission_event": self.emission_event,
            "enqueue_ordering": self.ordering.value,
            "ordering_evidence": self.ordering_evidence,
            "visibility": self.visibility.value,
            "clock_domain": self.clock_domain.value,
            "zero_meaning": self.zero_meaning,
            "ue_runtime_available": self.ue_runtime_available,
            "can_contain_current_payload": self.can_contain_current_payload,
        }


RLC_BUFFER_SOURCE = SourceDescriptor(
    name="NRUE_MAC_RLC_BUFFER_STATUS",
    measured_object=(
        "UE MAC's read of RLC transmit-buffer occupancy for one logical "
        "channel, returned by nr_mac_rlc_status_ind"
    ),
    byte_semantics=(
        "true queue occupancy in bytes, unquantized; not a BSR index and not a "
        "post-multiplex remainder"
    ),
    group_coverage=(
        "one row per active LCID per tick, each carrying its LCGID; total "
        "backlog is the plain sum of bytes_in_buffer over the tick's rows"
    ),
    emission_event=(
        "once per active LCID inside nr_update_rlc_buffers_status, called once "
        "per UL MAC tick while the UE is CONNECTED, independently of whether a "
        "grant exists"
    ),
    ordering=EnqueueOrdering.PRE_MULTIPLEX_WITHIN_MAC_SLOT,
    ordering_evidence=(
        f"emitted at {CODE_CITATIONS['rlc_buffer_emit']}; its caller runs at "
        f"{CODE_CITATIONS['rlc_buffer_caller']}, strictly before "
        f"{CODE_CITATIONS['get_sdu_call']}"
    ),
    visibility=SourceVisibility.UE,
    clock_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
    zero_meaning=(
        "no bytes were waiting in this RLC transmit buffer at this MAC tick. It "
        "does not mean zero network delay: PDUs already handed to lower layers, "
        "HARQ retransmissions, scheduler wait, gNB/core/edge queueing and the "
        "payload not yet enqueued are all outside this measurement"
    ),
    ue_runtime_available=True,
    can_contain_current_payload=True,
)

BSR_STATUS_SOURCE = SourceDescriptor(
    name="NRUE_MAC_BSR_STATUS",
    measured_object=(
        "the BSR MAC control element the UE is about to write into the current "
        "grant, plus the LCG byte counts it was encoded from"
    ),
    byte_semantics=(
        "lcg*_bytes are post-multiplex residual bytes: accumulated from "
        f"LCID_buffer_remain at {CODE_CITATIONS['lcg_bytes_accumulate']} then "
        f"decremented per multiplexed SDU at {CODE_CITATIONS['lcg_bytes_decrement']}. "
        f"bsr_index / bsr_long*_index are the coarse BSR table indices those "
        f"residuals encode to ({CODE_CITATIONS['bsr_index_encode']}), not bytes"
    ),
    group_coverage=(
        "all eight LCGs in one row, but only the groups the encoder populated "
        "are meaningful; a short BSR reports one LCG"
    ),
    emission_event=(
        "once per filled UL grant, inside nr_ue_get_sdu_mac_ce_post after the "
        "multiplexing loop has already drained the buffer into the PDU"
    ),
    ordering=EnqueueOrdering.POST_MULTIPLEX_RESIDUAL,
    ordering_evidence=(
        f"emitted at {CODE_CITATIONS['bsr_status_emit']}, reached from "
        f"{CODE_CITATIONS['bsr_status_caller']}, after "
        f"{CODE_CITATIONS['lcg_bytes_decrement']} has run for every SDU placed "
        f"in this grant"
    ),
    visibility=SourceVisibility.UE,
    clock_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
    zero_meaning=(
        "nothing was left to report after the current grant was filled. This is "
        "the weakest of the three zeros: it is consistent with a large backlog "
        "that the current grant happened to drain. It is produced by the same "
        "grant the policy is reasoning about, which rules it out as that "
        "action's own pre-action state; it does not rule out a correctly "
        "lagged reading describing a previous action"
    ),
    ue_runtime_available=True,
    can_contain_current_payload=True,
)

GNB_ESTIMATED_BUFFER_SOURCE = SourceDescriptor(
    name="GNB_MAC_UL_MCS_DECISION.estimated_ul_buffer",
    measured_object="the gNB scheduler's estimate of the UE's uplink backlog",
    byte_semantics=(
        "decoded BSR table value, so it inherits both the post-multiplex "
        "semantics of the UE report and the BSR table's coarse quantization"
    ),
    group_coverage="scheduler-side aggregate; no per-LCID decomposition",
    emission_event="per UL scheduling decision at the gNB",
    ordering=EnqueueOrdering.POST_MULTIPLEX_RESIDUAL,
    ordering_evidence=(
        "derived from the UE's reported BSR index, which is itself "
        f"post-multiplex ({CODE_CITATIONS['bsr_status_emit']})"
    ),
    visibility=SourceVisibility.GNB,
    clock_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
    zero_meaning="the gNB believes the UE has nothing pending",
    ue_runtime_available=False,
    can_contain_current_payload=True,
)

# --------------------------------------------------------------------------
# A4. Physical-channel candidates.
#
# Four quantities are routinely conflated. They are kept apart here by link
# direction, by who can observe them, and by whether the number is a
# standardized index or a raw power ratio.
# --------------------------------------------------------------------------


class LinkDirection(Enum):
    """Which link a channel quantity actually measures."""

    #: Measured by the UE on its receive chain: a downlink quantity.
    UE_DOWNLINK_RECEIVE = "UE_DOWNLINK_RECEIVE"
    #: Measured by the gNB on its receive chain: an uplink quantity.
    GNB_UPLINK_RECEIVE = "GNB_UPLINK_RECEIVE"


@dataclass(frozen=True)
class ChannelSignalDescriptor:
    """One candidate physical-channel signal, pinned to its link direction."""

    name: str
    direction: LinkDirection
    visibility: SourceVisibility
    quantity: str
    units: str
    standardized_index: bool
    cadence: str
    availability_precondition: str
    code_citation: str
    ue_runtime_available: bool
    caveat: str

    @property
    def display_label(self) -> str:
        """Label that may be used in a figure or table.

        Never the bare word "SNR": the whole point of this taxonomy is that two
        different links both produce something called SNR.
        """
        if self.direction is LinkDirection.UE_DOWNLINK_RECEIVE:
            return f"UE downlink {self.quantity}"
        return f"gNB received uplink {self.quantity}"

    def to_json(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "link_direction": self.direction.value,
            "visibility": self.visibility.value,
            "quantity": self.quantity,
            "units": self.units,
            "standardized_index": self.standardized_index,
            "cadence": self.cadence,
            "availability_precondition": self.availability_precondition,
            "code_citation": self.code_citation,
            "ue_runtime_available": self.ue_runtime_available,
            "display_label": self.display_label,
            "caveat": self.caveat,
        }


UE_PHY_MEAS_SNR = ChannelSignalDescriptor(
    name="UE_PHY_MEAS.snr",
    direction=LinkDirection.UE_DOWNLINK_RECEIVE,
    visibility=SourceVisibility.UE,
    quantity="SNR",
    units="integer dB (NOT scaled by 10)",
    standardized_index=False,
    cadence=(
        "at most once per 10 ms SFN frame: the emit is gated on "
        "nr_slot_rx == 0 inside the l == 2 measurement step"
    ),
    availability_precondition=(
        "requires a PDSCH allocation to this UE in slot 0; the only call site "
        "sits in the PDSCH channel-estimation path, so with no downlink "
        "traffic the event simply does not fire"
    ),
    code_citation=(
        f"{CODE_CITATIONS['ue_phy_meas_emit']}; {CODE_CITATIONS['ue_phy_meas_gate']}; "
        f"{CODE_CITATIONS['ue_phy_meas_snr_expression']}; "
        f"{CODE_CITATIONS['ue_measurements_compute']}"
    ),
    ue_runtime_available=True,
    caveat=(
        "a receive-side downlink power ratio computed from the downlink "
        "channel estimates. It is NOT the gNB's uplink PUSCH SNR and must "
        "never be labelled an uplink measurement"
    ),
)

UE_PHY_MEAS_W_CQI = ChannelSignalDescriptor(
    name="UE_PHY_MEAS.w_cqi",
    direction=LinkDirection.UE_DOWNLINK_RECEIVE,
    visibility=SourceVisibility.UE,
    quantity="wideband power/noise ratio reported as w_cqi",
    units="integer dB",
    standardized_index=False,
    cadence="same event as UE_PHY_MEAS.snr",
    availability_precondition="same PDSCH precondition as UE_PHY_MEAS.snr",
    code_citation=CODE_CITATIONS["ue_measurements_wideband_cqi"],
    ue_runtime_available=True,
    caveat=(
        "despite the name this is not a standardized CQI index. It is "
        "wideband_cqi_avg = rx_power_avg_dB - n0_power_avg_dB, the identical "
        "expression to the emitted snr field, so in this build w_cqi and snr "
        "carry the same number. Diagnostic only"
    ),
)

CSI_RS_CQI = ChannelSignalDescriptor(
    name="CSI-RS CQI (nr_csi_rs_cqi_estimation)",
    direction=LinkDirection.UE_DOWNLINK_RECEIVE,
    visibility=SourceVisibility.UE,
    quantity="CQI",
    units="standardized index 0-15",
    standardized_index=True,
    cadence="per configured CSI-RS measurement/report occasion",
    availability_precondition=(
        "only produced when CSI-RS measurement and reporting are actually "
        "configured; absent otherwise, and it has no T-tracer event of its own"
    ),
    code_citation=(
        f"{CODE_CITATIONS['csi_rs_cqi_table']}; {CODE_CITATIONS['csi_rs_cqi_call']}"
    ),
    ue_runtime_available=False,
    caveat=(
        "the only genuinely standardized downlink CQI candidate, but it is "
        "conditional on CSI-RS configuration and is not emitted by UE_PHY_MEAS"
    ),
)

GNB_PUSCH_SNR = ChannelSignalDescriptor(
    name="GNB_MAC_PUSCH_POWER_CONTROL.snrx10",
    direction=LinkDirection.GNB_UPLINK_RECEIVE,
    visibility=SourceVisibility.GNB,
    quantity="PUSCH SNR",
    units="dB x10 (divide by 10 for dB)",
    standardized_index=False,
    cadence="per PUSCH reception/power-control decision at the gNB",
    availability_precondition="requires uplink PUSCH transmissions to decode",
    code_citation="emitted by the gNB softmodem; decoded from the gNB T-tracer CSV",
    ue_runtime_available=False,
    caveat=(
        "measured at the gNB on the uplink. It is not available to the UE at "
        "runtime and therefore cannot itself be a UE policy feature"
    ),
)

CHANNEL_SIGNAL_CANDIDATES: Tuple[ChannelSignalDescriptor, ...] = (
    UE_PHY_MEAS_SNR,
    UE_PHY_MEAS_W_CQI,
    CSI_RS_CQI,
    GNB_PUSCH_SNR,
)


def assert_direction_not_mislabelled(
    descriptor: ChannelSignalDescriptor, claimed: LinkDirection
) -> None:
    """Refuse to describe a channel quantity as the wrong link direction.

    Guards the specific error this audit was corrected for: calling the UE's
    receive-side measurement an uplink SNR, or presenting the gNB's uplink
    PUSCH SNR as something the UE can observe.
    """
    if descriptor.direction is not claimed:
        raise AuditError(
            f"{descriptor.name} is {descriptor.direction.value}; refusing to "
            f"label it {claimed.value}. {descriptor.caveat}"
        )


def assert_ue_observable(descriptor: ChannelSignalDescriptor) -> None:
    """Refuse a signal the UE cannot read at runtime as a UE policy feature."""
    if not descriptor.ue_runtime_available:
        raise AuditError(
            f"{descriptor.name} is not available to the UE at runtime "
            f"(visibility={descriptor.visibility.value}); it cannot be a UE "
            f"policy feature. {descriptor.caveat}"
        )


CANDIDATE_SOURCES: Tuple[SourceDescriptor, ...] = (
    RLC_BUFFER_SOURCE,
    BSR_STATUS_SOURCE,
    GNB_ESTIMATED_BUFFER_SOURCE,
)


class BsrTemporalUse(Enum):
    """How a post-multiplex sample is being aligned to an action.

    The distinction the audit previously collapsed. A post-multiplex BSR is not
    universally invalid; it is invalid *as the pre-action state of the very
    action that produced it*.
    """

    #: Aligned to the action whose grant produced the residual. Forbidden: the
    #: value is a function of that action, so the action enters its own state.
    SAME_ACTION_PRE_STATE = "SAME_ACTION_PRE_STATE_FORBIDDEN"
    #: Aligned to a strictly later decision, so it describes a *previous*
    #: action. Admissible in principle, subject to proof of the lag.
    LAGGED_PREVIOUS_ACTION = "LAGGED_PREVIOUS_ACTION_ADMISSIBLE_IF_LAG_PROVEN"


#: Why the pre-multiplex RLC read stays the preferred candidate even where a
#: lagged BSR would be admissible.
BSR_VS_RLC_PREFERENCE_NOTE = (
    "NRUE_MAC_RLC_BUFFER_STATUS remains the preferred candidate: it is a more "
    "direct observation (true RLC occupancy in bytes, read before the grant is "
    "filled) and it is less quantized (unquantized bytes, where the BSR path "
    "reports coarse BSR-table indices). A lagged NRUE_MAC_BSR_STATUS is a "
    "fallback, not an equal."
)


def classify_bsr_temporal_use(
    source: SourceDescriptor, *, aligned_to_same_action: bool
) -> BsrTemporalUse:
    """Say which of the two alignments is being attempted for a source."""
    if source.ordering is not EnqueueOrdering.POST_MULTIPLEX_RESIDUAL:
        raise AuditError(
            f"{source.name} is {source.ordering.value}, not a post-multiplex "
            f"residual; this classification does not apply to it"
        )
    return (
        BsrTemporalUse.SAME_ACTION_PRE_STATE
        if aligned_to_same_action
        else BsrTemporalUse.LAGGED_PREVIOUS_ACTION
    )


def assert_no_action_leakage(
    source: SourceDescriptor, *, aligned_to_same_action: bool = True
) -> None:
    """Refuse a post-multiplex sample as the *current* action's pre-action state.

    The precise claim, which the earlier wording overstated:

    * A post-multiplex BSR generated **after** the current action must not be
      aligned as that action's pre-action state. Its bytes are what remained
      once the current payload had already been multiplexed into the current
      grant, so the action would enter its own observation.
    * A **correctly lagged** BSR may legitimately describe a *previous* action.
      That use is not blocked here; it requires the lag to be proven, not
      assumed, which is what the ordering chain exists to do.
    * Either way :data:`BSR_VS_RLC_PREFERENCE_NOTE` applies: the pre-multiplex
      RLC read is more direct and less quantized, so it stays preferred.

    ``aligned_to_same_action=False`` declares the caller is taking the lagged
    reading and accepts the burden of proving the lag.
    """
    if source.ordering is not EnqueueOrdering.POST_MULTIPLEX_RESIDUAL:
        return
    if not aligned_to_same_action:
        return
    raise ActionLeakageError(
        f"{source.name} is {source.ordering.value}: its bytes are what remains "
        f"after the current payload was multiplexed into the current grant, so "
        f"aligning it as that same action's pre-action state leaks the action "
        f"into its own observation. A correctly lagged alignment describing a "
        f"previous action is a different question and is not refused here. "
        f"{BSR_VS_RLC_PREFERENCE_NOTE}"
    )


def qualify_pre_action_source(
    source: SourceDescriptor, *, enqueue_instant_evidence: bool
) -> PreActionQualification:
    """Decide whether a source proves the ordering a pre-action feature needs.

    The required chain is: read older UE backlog -> build state -> select
    ``(mode, q)`` -> build and enqueue the payload.  Proving it needs two
    separate facts: the sample must precede the MAC's use of the grant, *and*
    the run must show where the application enqueued the payload, because the
    UE MAC tick is asynchronous to the SplitFusion frame.  Being pre-multiplex
    settles only the first.
    """
    if source.ordering is EnqueueOrdering.POST_MULTIPLEX_RESIDUAL:
        return PreActionQualification.DISQUALIFIED_ACTION_CONTAMINATED
    if source.ordering is EnqueueOrdering.UNRESOLVED:
        return PreActionQualification.UNRESOLVED
    if not enqueue_instant_evidence:
        return PreActionQualification.UNRESOLVED
    return PreActionQualification.QUALIFIED


# --------------------------------------------------------------------------
# A2. Enqueue-instant qualification.
#
# The previous implementation qualified a causal queue source from *filename
# existence*: a zero-row, wrong-schema, wrong-run or dequeue-only file counted.
# Qualification now requires parsed records, an exact schema, ingress
# semantics, same-run identity, UE/RNTI and bearer/LCID identity, a decision
# identity, both timestamps, and a demonstrated ordering for every admitted
# decision. Anything missing fails closed.
# --------------------------------------------------------------------------

#: T-tracer events that genuinely timestamp the application enqueue instant.
#: Both are PDCP/RLC *ingress* events carrying a radio-bearer id.
ENQUEUE_INSTANT_EVENTS: Tuple[str, ...] = (
    "NR_PDCP_TX_SDU",
    "NR_RLC_TX_SDU",
)

#: Explicitly NOT enqueue evidence. ``NR_RLC_TX_DEQUEUE`` fires when a PDU is
#: handed down to MAC for a grant, which *ends* the RLC queue wait. Treating it
#: as an enqueue instant would date the payload to when it was served rather
#: than when it arrived, inverting the very ordering being tested.
SERVICE_DEQUEUE_EVENTS: Tuple[str, ...] = ("NR_RLC_TX_DEQUEUE",)

ENQUEUE_EVENT_HEADERS: Mapping[str, Tuple[str, ...]] = {
    "NR_PDCP_TX_SDU": PDCP_TX_SDU_HEADER,
    "NR_RLC_TX_SDU": RLC_TX_SDU_HEADER,
}

SERVICE_DEQUEUE_EVENT_HEADERS: Mapping[str, Tuple[str, ...]] = {
    "NR_RLC_TX_DEQUEUE": RLC_TX_DEQUEUE_HEADER,
}

#: Where a future runtime carrier must write its per-decision record.
DECISION_LOG_RELPATH = ("decisions", "decision_log.csv")


class EnqueueRequirement(Enum):
    """Every condition that must hold before enqueue evidence qualifies."""

    FILE_PRESENT = "FILE_PRESENT"
    SCHEMA_EXACT = "SCHEMA_EXACT"
    NONEMPTY_RECORDS = "NONEMPTY_PARSED_RECORDS"
    INGRESS_SEMANTICS = "INGRESS_NOT_DEQUEUE_SEMANTICS"
    SAME_RUN_IDENTITY = "SAME_RUN_IDENTITY"
    UE_IDENTITY = "UE_AND_RNTI_IDENTITY"
    BEARER_IDENTITY = "LCID_OR_DRB_IDENTITY"
    DECISION_IDENTITY = "DECISION_AND_FRAME_IDENTITY"
    SOURCE_TIMESTAMP = "SOURCE_TIMESTAMP_PRESENT"
    AVAILABILITY_TIMESTAMP = "AVAILABILITY_TIMESTAMP_PRESENT"
    CAUSAL_ORDERING = "CAUSAL_ORDERING_DEMONSTRATED"


#: The chain every admitted decision must satisfy, in order. ``<=`` between the
#: first three because a measurement can be published and committed within the
#: same nanosecond tick; strict ``<`` from the state commit onwards because the
#: action must follow a frozen state and precede the payload it causes.
ORDERING_CHAIN_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("t_measure_mono_ns", "t_available_to_agent_mono_ns"),
    ("t_available_to_agent_mono_ns", "t_state_commit_mono_ns"),
    ("t_state_commit_mono_ns", "t_action_mono_ns"),
    ("t_action_mono_ns", "t_current_payload_enqueue_mono_ns"),
)
ORDERING_CHAIN_STRICT: Tuple[bool, ...] = (False, False, True, True)

ORDERING_CHAIN_TEXT = (
    "t_measure <= t_available_to_agent <= t_state_commit < t_action "
    "< t_current_payload_enqueue"
)


@dataclass(frozen=True)
class RequirementOutcome:
    """Whether one named requirement held, and why."""

    requirement: EnqueueRequirement
    satisfied: bool
    detail: str

    def to_json(self) -> Dict[str, object]:
        return {
            "requirement": self.requirement.value,
            "satisfied": self.satisfied,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EnqueueEvidenceCheck:
    """Per-run outcome of the enqueue-instant qualification."""

    run_id: str
    event_name: Optional[str]
    outcomes: Tuple[RequirementOutcome, ...]
    admitted_decisions: int
    ordered_decisions: int

    @property
    def qualified(self) -> bool:
        """True only when every requirement held for every admitted decision."""
        if not self.outcomes:
            return False
        if not all(item.satisfied for item in self.outcomes):
            return False
        return self.admitted_decisions > 0 and (
            self.ordered_decisions == self.admitted_decisions
        )

    @property
    def first_failure(self) -> Optional[str]:
        for item in self.outcomes:
            if not item.satisfied:
                return f"{item.requirement.value}: {item.detail}"
        if self.admitted_decisions == 0:
            return "CAUSAL_ORDERING_DEMONSTRATED: no decision was admitted"
        return None

    def to_json(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "event_name": self.event_name,
            "qualified": self.qualified,
            "admitted_decisions": self.admitted_decisions,
            "ordered_decisions": self.ordered_decisions,
            "required_ordering": ORDERING_CHAIN_TEXT,
            "first_failure": self.first_failure,
            "requirements": [item.to_json() for item in self.outcomes],
        }


def check_decision_ordering(record: Mapping[str, str]) -> Tuple[bool, str]:
    """Verify the causal chain for one decision record.

    A missing, blank or non-integer instant fails closed; it is never treated
    as a zero and never forward-filled from a neighbour.
    """
    values: Dict[str, int] = {}
    for earlier, later in ORDERING_CHAIN_FIELDS:
        for name in (earlier, later):
            if name in values:
                continue
            raw = record.get(name)
            text = "" if raw is None else str(raw).strip()
            if not text:
                return False, f"{name} is missing; fails closed rather than assumed"
            try:
                values[name] = int(text)
            except ValueError:
                return False, f"{name}={text!r} is not an integer nanosecond instant"
    for (earlier, later), strict in zip(ORDERING_CHAIN_FIELDS, ORDERING_CHAIN_STRICT):
        lhs, rhs = values[earlier], values[later]
        if strict and not lhs < rhs:
            return False, f"requires {earlier} < {later}, found {lhs} >= {rhs}"
        if not strict and not lhs <= rhs:
            return False, f"requires {earlier} <= {later}, found {lhs} > {rhs}"
    return True, "ordering holds"


def read_decision_records(path: Path) -> List[Dict[str, str]]:
    """Read the per-decision carrier log with an exact schema check."""
    return read_simple_rows(path, DECISION_RECORD_HEADER)


def _ue_identity_from_rlc(path: Path) -> Tuple[set, set]:
    """``(rnti values, ue_id values)`` observed in the RLC backlog trace."""
    rnti: set = set()
    ue_ids: set = set()
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        _require_header(path, header, RLC_BUFFER_HEADER)
        for record in reader:
            if len(record) != len(RLC_BUFFER_HEADER):
                continue
            rnti.add(record[1].strip())
            ue_ids.add(record[2].strip())
    return rnti, ue_ids


def qualify_enqueue_evidence(run: LogicalRun) -> EnqueueEvidenceCheck:
    """Decide, from parsed content, whether a run proves the enqueue instant.

    Never qualifies from a filename. Each requirement is evaluated in turn and
    the first unmet one stops the chain, so the report names the actual reason
    rather than a generic refusal.
    """
    outcomes: List[RequirementOutcome] = []

    def record(req: EnqueueRequirement, ok: bool, detail: str) -> bool:
        outcomes.append(RequirementOutcome(req, ok, detail))
        return ok

    def finish(event: Optional[str] = None, admitted: int = 0, ordered: int = 0):
        return EnqueueEvidenceCheck(
            run_id=run.run_id,
            event_name=event,
            outcomes=tuple(outcomes),
            admitted_decisions=admitted,
            ordered_decisions=ordered,
        )

    # 1. A genuine ingress event file must exist. A dequeue-only run is called
    #    out by name so the refusal is not mistaken for "nothing retained".
    present = [name for name in ENQUEUE_INSTANT_EVENTS if run.ue_csv(f"{name}.csv")]
    dequeue_present = [
        name for name in SERVICE_DEQUEUE_EVENTS if run.ue_csv(f"{name}.csv")
    ]
    if not present:
        detail = (
            f"none of {list(ENQUEUE_INSTANT_EVENTS)} is retained"
        )
        if dequeue_present:
            detail += (
                f"; {dequeue_present} is retained but is service/dequeue "
                f"evidence ({CODE_CITATIONS['t_message_rlc_tx_dequeue']}) and "
                f"does not date the enqueue instant"
            )
        record(EnqueueRequirement.FILE_PRESENT, False, detail)
        return finish()
    event_name = present[0]
    event_path = run.ue_csv(f"{event_name}.csv")
    assert event_path is not None
    record(EnqueueRequirement.FILE_PRESENT, True, f"{event_name}.csv retained")

    record(
        EnqueueRequirement.INGRESS_SEMANTICS,
        True,
        f"{event_name} is PDCP/RLC ingress, not {list(SERVICE_DEQUEUE_EVENTS)}",
    )

    # 2. Exact schema, then genuinely parsed rows.
    try:
        rows = read_simple_rows(event_path, ENQUEUE_EVENT_HEADERS[event_name])
    except SchemaError as exc:
        record(EnqueueRequirement.SCHEMA_EXACT, False, str(exc))
        return finish(event_name)
    record(EnqueueRequirement.SCHEMA_EXACT, True, "header matches exactly")

    if not rows:
        record(
            EnqueueRequirement.NONEMPTY_RECORDS,
            False,
            "file parses but holds no data rows; a header alone proves nothing",
        )
        return finish(event_name)
    record(EnqueueRequirement.NONEMPTY_RECORDS, True, f"{len(rows)} parsed records")

    # 3. Same-run identity, established structurally rather than by name.
    rlc_path = run.ue_csv("NRUE_MAC_RLC_BUFFER_STATUS.csv")
    if rlc_path is None:
        record(
            EnqueueRequirement.SAME_RUN_IDENTITY,
            False,
            "run retains no NRUE_MAC_RLC_BUFFER_STATUS to bind the enqueue "
            "evidence to",
        )
        return finish(event_name)
    try:
        require_same_run(rlc_path, event_path)
    except RunIsolationError as exc:
        record(EnqueueRequirement.SAME_RUN_IDENTITY, False, str(exc))
        return finish(event_name)
    record(
        EnqueueRequirement.SAME_RUN_IDENTITY,
        True,
        f"both resolve to run root {logical_run_root(event_path)}",
    )

    # 4. UE/RNTI identity must actually agree with the backlog trace.
    _, rlc_ue_ids = _ue_identity_from_rlc(rlc_path)
    event_ue_ids = {(row.get("ue_id") or "").strip() for row in rows}
    if "" in event_ue_ids:
        record(EnqueueRequirement.UE_IDENTITY, False, "a record has a blank ue_id")
        return finish(event_name)
    shared = event_ue_ids & rlc_ue_ids
    if not shared:
        record(
            EnqueueRequirement.UE_IDENTITY,
            False,
            f"enqueue ue_id {sorted(event_ue_ids)} does not intersect backlog "
            f"ue_id {sorted(rlc_ue_ids)}; the two traces describe different UEs",
        )
        return finish(event_name)
    record(
        EnqueueRequirement.UE_IDENTITY, True, f"shared ue_id {sorted(shared)}"
    )

    # 5. Bearer identity. The ingress events carry rb_id while the backlog
    #    trace carries lcid, so a declared mapping is required; it is not
    #    invented here.
    bearer_ids = {(row.get("rb_id") or "").strip() for row in rows}
    if "" in bearer_ids:
        record(EnqueueRequirement.BEARER_IDENTITY, False, "a record has a blank rb_id")
        return finish(event_name)
    record(
        EnqueueRequirement.BEARER_IDENTITY,
        True,
        f"rb_id values {sorted(bearer_ids)} present on every record",
    )

    # 6. Both timestamps. The monotonic pair is the source instant; the
    #    tracer's realtime column is when the sample became observable.
    for field_name, requirement in (
        ("mono_sec", EnqueueRequirement.SOURCE_TIMESTAMP),
        ("time", EnqueueRequirement.AVAILABILITY_TIMESTAMP),
    ):
        missing = sum(1 for row in rows if not (row.get(field_name) or "").strip())
        if missing:
            record(
                requirement,
                False,
                f"{missing} record(s) have no {field_name}; missing timestamps "
                f"fail closed and are never filled in",
            )
            return finish(event_name)
    record(
        EnqueueRequirement.SOURCE_TIMESTAMP, True, "mono_sec/mono_nsec present"
    )
    record(
        EnqueueRequirement.AVAILABILITY_TIMESTAMP,
        True,
        "tracer realtime column present",
    )

    # 7. Decision/frame identity. Without a per-decision record there is no
    #    object to check the ordering *against*, so this fails closed.
    decision_path = run.root.joinpath(*DECISION_LOG_RELPATH)
    if not decision_path.is_file():
        record(
            EnqueueRequirement.DECISION_IDENTITY,
            False,
            f"no per-decision carrier log at "
            f"{Path(*DECISION_LOG_RELPATH)}; the enqueue events carry no "
            f"decision or SplitFusion frame id of their own, so no action can "
            f"be bound to an enqueue instant",
        )
        return finish(event_name)
    try:
        decisions = read_decision_records(decision_path)
    except SchemaError as exc:
        record(EnqueueRequirement.DECISION_IDENTITY, False, str(exc))
        return finish(event_name)
    if not decisions:
        record(
            EnqueueRequirement.DECISION_IDENTITY,
            False,
            "decision log holds no records",
        )
        return finish(event_name)
    blank = [
        row
        for row in decisions
        if not (row.get("decision_id") or "").strip()
        or not (row.get("frame_index") or "").strip()
    ]
    if blank:
        record(
            EnqueueRequirement.DECISION_IDENTITY,
            False,
            f"{len(blank)} decision record(s) lack decision_id or frame_index",
        )
        return finish(event_name)
    record(
        EnqueueRequirement.DECISION_IDENTITY,
        True,
        f"{len(decisions)} decision record(s) with decision_id and frame_index",
    )

    # 8. The ordering must hold for EVERY admitted decision, not on average.
    failures: List[str] = []
    ordered = 0
    for row in decisions:
        ok, why = check_decision_ordering(row)
        if ok:
            ordered += 1
        elif len(failures) < 5:
            failures.append(f"decision {row.get('decision_id')!r}: {why}")
    all_ordered = ordered == len(decisions)
    record(
        EnqueueRequirement.CAUSAL_ORDERING,
        all_ordered,
        (
            f"{ordered}/{len(decisions)} decisions satisfy {ORDERING_CHAIN_TEXT}"
            + ("" if all_ordered else f"; first failures: {failures}")
        ),
    )
    return finish(event_name, admitted=len(decisions), ordered=ordered)


def has_enqueue_instant_evidence(run: LogicalRun) -> bool:
    """True only when :func:`qualify_enqueue_evidence` fully qualifies the run.

    Kept as a name for call sites, but it is no longer a filename test.
    """
    return qualify_enqueue_evidence(run).qualified


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SaturationReport:
    """What one ``log1p``/scale choice does to a run's measured backlog."""

    scale_label: str
    scale: float
    status: str
    sample_count: int
    zero_preserved_count: int
    zero_output_count: int
    saturated_count: int
    saturated_fraction: Optional[float]
    saturated_fraction_of_nonzero: Optional[float]
    distinct_outputs: int
    smallest_saturating_byte_count: Optional[int]

    def to_json(self) -> Dict[str, object]:
        return {
            "scale_label": self.scale_label,
            "scale": self.scale,
            "status": self.status,
            "sample_count": self.sample_count,
            "zero_preserved_count": self.zero_preserved_count,
            "zero_output_count": self.zero_output_count,
            "saturated_count": self.saturated_count,
            "saturated_fraction": self.saturated_fraction,
            "saturated_fraction_of_nonzero": self.saturated_fraction_of_nonzero,
            "distinct_outputs": self.distinct_outputs,
            "smallest_saturating_byte_count": self.smallest_saturating_byte_count,
        }


def scale_bsr_bytes(byte_count: int, scale: float) -> float:
    """Reproduce ``NormalizationSpecV1.scale_bsr_bytes`` exactly.

    Mirrors ``state_reward_transition_contract.py:5827``:
    ``clip(log1p(bytes) / scale, 0, 1)``.
    """
    if byte_count < 0:
        raise AuditError(f"byte_count must be non-negative, got {byte_count}")
    if scale <= 0.0:
        raise AuditError(f"scale must be positive, got {scale}")
    return min(max(math.log1p(float(byte_count)) / scale, 0.0), 1.0)


def smallest_saturating_bytes(scale: float) -> Optional[int]:
    """Smallest integer byte count that the given scale maps to exactly 1.0."""
    if scale <= 0.0:
        raise AuditError(f"scale must be positive, got {scale}")
    threshold = math.expm1(scale)
    if not math.isfinite(threshold):
        return None
    candidate = max(0, int(math.floor(threshold)))
    for value in range(candidate, candidate + 4):
        if scale_bsr_bytes(value, scale) >= 1.0:
            return value
    return None


def saturation_report(
    values: Sequence[int], *, scale: float, label: str, status: str
) -> SaturationReport:
    """Zero preservation and clipping behaviour of one candidate scale."""
    outputs = [scale_bsr_bytes(value, scale) for value in values]
    zeros_in = sum(1 for value in values if value == 0)
    nonzero = len(values) - zeros_in
    saturated = sum(
        1 for value, output in zip(values, outputs) if value > 0 and output >= 1.0
    )
    return SaturationReport(
        scale_label=label,
        scale=scale,
        status=status,
        sample_count=len(values),
        zero_preserved_count=sum(
            1 for value, output in zip(values, outputs) if value == 0 and output == 0.0
        ),
        zero_output_count=sum(1 for output in outputs if output == 0.0),
        saturated_count=saturated,
        saturated_fraction=(saturated / len(values)) if values else None,
        saturated_fraction_of_nonzero=(saturated / nonzero) if nonzero else None,
        distinct_outputs=len({round(output, 12) for output in outputs}),
        smallest_saturating_byte_count=smallest_saturating_bytes(scale),
    )


def candidate_scales(values: Sequence[int]) -> List[Tuple[str, float, str]]:
    """Diagnostic scale candidates. Every one is PROVISIONAL_NOT_FROZEN.

    Freezing a scaler needs a resolved causal source, a declared train-only
    split, and payload coverage that reaches the SplitFusion action range. None
    of those hold yet, so these exist to quantify saturation, not to be adopted.
    """
    out: List[Tuple[str, float, str]] = [
        (
            "DEPLOYED_bsr_log1p_scale_1.0",
            DEPLOYED_BSR_LOG1P_SCALE,
            "CURRENTLY_FROZEN_IN_CONTRACT",
        )
    ]
    positive = [value for value in values if value > 0]
    for quantile, name in ((0.95, "P95"), (0.99, "P99")):
        anchor = percentile(positive, quantile) if positive else None
        if anchor and anchor > 0:
            scale = math.log1p(anchor)
            if scale > 0:
                out.append(
                    (
                        f"CANDIDATE_log1p_nonzero_{name}_{int(anchor)}B",
                        scale,
                        "PROVISIONAL_NOT_FROZEN",
                    )
                )
    out.append(
        (
            "CANDIDATE_log1p_engineering_bound_8192B",
            math.log1p(8192.0),
            "PROVISIONAL_NOT_FROZEN",
        )
    )
    out.append(
        (
            f"CANDIDATE_log1p_action_max_payload_"
            f"{SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES}B",
            math.log1p(float(SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES)),
            "PROVISIONAL_NOT_FROZEN",
        )
    )
    out.append(
        (
            f"CANDIDATE_log1p_run3_support_max_"
            f"{RUN3_MODELED_SUPPORT_MAX_BYTES}B",
            math.log1p(float(RUN3_MODELED_SUPPORT_MAX_BYTES)),
            "PROVISIONAL_NOT_FROZEN",
        )
    )
    return out


# --------------------------------------------------------------------------
# Offered load
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OfferedLoadProfile:
    """The traffic actually offered in a run, from its own sender log."""

    present: bool
    row_count: int
    distinct_frame_bytes: Tuple[int, ...]
    distinct_period_s: Tuple[float, ...]
    within_action_range: bool
    within_run3_support: bool
    varies_within_run: bool
    note: str

    def to_json(self) -> Dict[str, object]:
        return {
            "present": self.present,
            "row_count": self.row_count,
            "distinct_frame_bytes": list(self.distinct_frame_bytes),
            "distinct_period_s": list(self.distinct_period_s),
            "within_action_range": self.within_action_range,
            "within_run3_support": self.within_run3_support,
            "varies_within_run": self.varies_within_run,
            "note": self.note,
        }


def read_offered_load(run: LogicalRun) -> OfferedLoadProfile:
    """Summarize the run's offered traffic without joining it to any trace."""
    if run.traffic_sender is None:
        return OfferedLoadProfile(
            present=False,
            row_count=0,
            distinct_frame_bytes=(),
            distinct_period_s=(),
            within_action_range=False,
            within_run3_support=False,
            varies_within_run=False,
            note="no traffic/sender.csv retained for this run",
        )
    rows = read_simple_rows(run.traffic_sender, SENDER_HEADER)
    frame_bytes = sorted({int(row["frame_bytes"]) for row in rows})
    periods = sorted({float(row["period_s"]) for row in rows})
    within_action = any(
        SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES
        <= value
        <= SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES
        for value in frame_bytes
    )
    within_run3 = any(
        RUN3_MODELED_SUPPORT_MIN_BYTES <= value <= RUN3_MODELED_SUPPORT_MAX_BYTES
        for value in frame_bytes
    )
    varies = len(frame_bytes) > 1
    if not varies:
        note = (
            "single fixed offered payload size; no within-run payload "
            "variation, so no action-conditioned queue transition is "
            "identifiable from this run"
        )
    else:
        note = f"{len(frame_bytes)} distinct offered payload sizes"
    return OfferedLoadProfile(
        present=True,
        row_count=len(rows),
        distinct_frame_bytes=tuple(frame_bytes),
        distinct_period_s=tuple(periods),
        within_action_range=within_action,
        within_run3_support=within_run3,
        varies_within_run=varies,
        note=note,
    )


# --------------------------------------------------------------------------
# Per-run audit
# --------------------------------------------------------------------------


@dataclass
class RunAudit:
    """Everything the audit establishes about one logical run."""

    run_id: str
    provenance: List[FileProvenance] = field(default_factory=list)
    rlc_total_backlog: Optional[DistributionStats] = None
    rlc_per_lcid_backlog: Dict[str, DistributionStats] = field(default_factory=dict)
    rlc_tick_interval: Optional[DistributionStats] = None
    bsr_residual_backlog: Optional[DistributionStats] = None
    bsr_interval: Optional[DistributionStats] = None
    grant_mcs_counts: Dict[str, int] = field(default_factory=dict)
    grant_tbs: Optional[DistributionStats] = None
    gnb_snr_db: Optional[DistributionStats] = None
    alignments: List[AlignmentResult] = field(default_factory=list)
    refused_alignments: List[Dict[str, str]] = field(default_factory=list)
    offered_load: Optional[OfferedLoadProfile] = None
    saturation: List[SaturationReport] = field(default_factory=list)
    lagged_association: Dict[str, object] = field(default_factory=dict)
    enqueue_instant_evidence: bool = False
    enqueue_evidence: Optional["EnqueueEvidenceCheck"] = None
    ue_phy_meas_present: bool = False
    ue_phy_meas_rows: int = 0
    pre_action_qualification: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "provenance": [item.to_json() for item in self.provenance],
            "rlc_total_backlog_bytes": (
                self.rlc_total_backlog.to_json() if self.rlc_total_backlog else None
            ),
            "rlc_per_lcid_backlog_bytes": {
                key: value.to_json() for key, value in self.rlc_per_lcid_backlog.items()
            },
            "rlc_tick_interval_us": (
                self.rlc_tick_interval.to_json() if self.rlc_tick_interval else None
            ),
            "bsr_residual_backlog_bytes": (
                self.bsr_residual_backlog.to_json()
                if self.bsr_residual_backlog
                else None
            ),
            "bsr_interval_us": (
                self.bsr_interval.to_json() if self.bsr_interval else None
            ),
            "grant_mcs_counts": dict(self.grant_mcs_counts),
            "grant_tbs_bytes": self.grant_tbs.to_json() if self.grant_tbs else None,
            "gnb_pusch_snr_db": self.gnb_snr_db.to_json() if self.gnb_snr_db else None,
            "alignments": [item.to_json() for item in self.alignments],
            "refused_alignments": list(self.refused_alignments),
            "offered_load": (
                self.offered_load.to_json() if self.offered_load else None
            ),
            "normalization_saturation": [item.to_json() for item in self.saturation],
            "lagged_association": dict(self.lagged_association),
            "enqueue_instant_evidence": self.enqueue_instant_evidence,
            "enqueue_evidence": (
                self.enqueue_evidence.to_json() if self.enqueue_evidence else None
            ),
            "ue_phy_meas_present": self.ue_phy_meas_present,
            "ue_phy_meas_rows": self.ue_phy_meas_rows,
            "pre_action_qualification": dict(self.pre_action_qualification),
        }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Plain Pearson r. Descriptive only; this audit claims no causality."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    n = float(len(xs))
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def recurrence_derived_skew(
    records: Sequence[object],
    key_of: Callable[[object], Tuple[int, ...]],
    time_of: Callable[[object], int],
    *,
    source_label: str,
    purpose: str,
) -> Tuple[int, str]:
    """Widest skew window that still leaves one counterpart per key.

    For joins where the two sides are genuinely different instants that share a
    slot label - a grant received K2 slots before the slot it schedules, or a
    PUSCH the gNB processes after the UE sent it - the window must not be
    narrowed to a same-tick bound, or every row goes unmatched for a reason
    that has nothing to do with the evidence.  Half the measured same-key
    recurrence is the widest bound that is still unambiguous.
    """
    recurrence = min_same_key_recurrence_us(records, key_of, time_of)
    if recurrence is None:
        bound = SFN_WRAP_MICROSECONDS // 2 - 1
        reason = (
            f"no key repeats in {source_label}, so the join is unambiguous on "
            f"the key alone; the window falls back to half the SFN wrap. "
            f"{purpose}"
        )
    else:
        bound = max(1, recurrence // 2 - 1)
        reason = (
            f"just under half the measured smallest same-key recurrence of "
            f"{recurrence} us in {source_label}. The bound exists only to "
            f"select which occurrence is meant; {purpose}"
        )
    return bound, reason


def descriptive_associations(
    *,
    ticks: Sequence[RlcTick],
    bsr_rows: Sequence[BsrStatusRow],
    granted_tb_bytes: Sequence[Tuple[int, int, int, float]],
    same_tick_skew_us: int,
) -> Dict[str, object]:
    """Lagged descriptive associations around backlog. Never causal evidence.

    Three quantities are kept apart because they mean different things:

    ``served_sdu_bytes``
        bytes the MAC actually multiplexed into the grant, from the BSR row's
        ``sdu_bytes``.  This is service delivered.
    ``granted_tb_bytes``
        the transport block size from ``UE_PHY_UL_PAYLOAD_TX_BITS``.  This is
        the grant, padding included, so it is *not* a measure of data sent and
        must not be read as one.
    ``next_backlog``
        the following tick's pre-multiplex total, i.e. where the queue went.

    Every number here is a correlation on a fixed-rate traffic generator. None
    of it identifies a response to an action that was never varied.
    """
    served: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    for row in bsr_rows:
        served.setdefault((row.frame, row.slot), []).append(
            (row.time_us, float(row.sdu_bytes))
        )
    granted: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    for frame, slot, when, value in granted_tb_bytes:
        granted.setdefault((frame, slot), []).append((when, value))
    for table in (served, granted):
        for entries in table.values():
            entries.sort()

    def same_tick(table, tick: RlcTick) -> Optional[float]:
        for when, value in table.get((tick.frame, tick.slot), ()):
            if abs(when - tick.time_us) <= same_tick_skew_us:
                return value
        return None

    backlog: List[float] = []
    served_values: List[float] = []
    backlog_for_grant: List[float] = []
    granted_values: List[float] = []
    backlog_now: List[float] = []
    backlog_next: List[float] = []
    served_then: List[float] = []
    backlog_after_service: List[float] = []

    for index, tick in enumerate(ticks):
        served_value = same_tick(served, tick)
        granted_value = same_tick(granted, tick)
        if served_value is not None:
            backlog.append(float(tick.total_bytes))
            served_values.append(served_value)
        if granted_value is not None:
            backlog_for_grant.append(float(tick.total_bytes))
            granted_values.append(granted_value)
        if index + 1 < len(ticks):
            backlog_now.append(float(tick.total_bytes))
            backlog_next.append(float(ticks[index + 1].total_bytes))
            if served_value is not None:
                served_then.append(served_value)
                backlog_after_service.append(float(ticks[index + 1].total_bytes))

    return {
        "claim": "DESCRIPTIVE_ONLY_NOT_CAUSAL",
        "traffic_regime": "FIXED_RATE_GENERATOR_NO_ACTION_VARIATION",
        "pairs_backlog_vs_served_sdu_bytes": len(backlog),
        "r_backlog_vs_served_sdu_bytes": _pearson(backlog, served_values),
        "pairs_backlog_vs_granted_tb_bytes": len(backlog_for_grant),
        "r_backlog_vs_granted_tb_bytes": _pearson(backlog_for_grant, granted_values),
        "granted_tb_bytes_note": (
            "transport block size includes padding; it is the grant, not the "
            "data actually sent"
        ),
        "pairs_backlog_t_vs_backlog_t_plus_1": len(backlog_now),
        "r_backlog_t_vs_backlog_t_plus_1": _pearson(backlog_now, backlog_next),
        "pairs_served_t_vs_backlog_t_plus_1": len(served_then),
        "r_served_t_vs_backlog_t_plus_1": _pearson(served_then, backlog_after_service),
    }


def audit_run(run: LogicalRun, evidence_root: Path) -> RunAudit:
    """Audit one logical run end to end, reading nothing outside its root."""
    result = RunAudit(run_id=run.run_id)

    rlc_path = run.ue_csv("NRUE_MAC_RLC_BUFFER_STATUS.csv")
    bsr_path = run.ue_csv("NRUE_MAC_BSR_STATUS.csv")
    dci_path = run.ue_csv("NRUE_MAC_DCI_GRANT.csv")
    tx_path = run.ue_csv("UE_PHY_UL_PAYLOAD_TX_BITS.csv")
    pusch_path = run.gnb_csv("GNB_MAC_PUSCH_POWER_CONTROL.csv")

    for path in (rlc_path, bsr_path, dci_path, tx_path, pusch_path):
        if path is not None:
            result.provenance.append(describe_file(path, evidence_root))
    if run.traffic_sender is not None:
        result.provenance.append(describe_file(run.traffic_sender, evidence_root))

    if bsr_path is None or rlc_path is None:
        raise AuditError(f"{run.run_id}: both UE backlog traces are required")
    require_same_run(rlc_path, bsr_path)

    rlc_rows = read_rlc_buffer_rows(rlc_path)
    bsr_rows = read_bsr_status_rows(bsr_path)
    ticks = rlc_ticks(rlc_rows)

    result.rlc_total_backlog = describe_distribution(
        [float(tick.total_bytes) for tick in ticks]
    )
    per_lcid: Dict[int, List[float]] = {}
    for row in rlc_rows:
        per_lcid.setdefault(row.lcid, []).append(float(row.bytes_in_buffer))
    result.rlc_per_lcid_backlog = {
        f"lcid_{lcid}": describe_distribution(values)
        for lcid, values in sorted(per_lcid.items())
    }
    result.rlc_tick_interval = sampling_interval_stats([tick.time_us for tick in ticks])
    result.bsr_residual_backlog = describe_distribution(
        [float(row.lcg_total_bytes) for row in bsr_rows]
    )
    result.bsr_interval = sampling_interval_stats([row.time_us for row in bsr_rows])

    # Skew bound for same-tick UE joins: half the measured tick cadence. It is
    # derived from this run's own sampling interval rather than assumed, and it
    # is far inside half an SFN wrap, so the (frame, slot) key is unambiguous.
    tick_p50 = result.rlc_tick_interval.p50 or 0.0
    same_tick_skew = max(1, int(tick_p50 // 2))
    same_tick_reason = (
        f"half the measured RLC tick interval P50 of {tick_p50:.0f} us in this "
        f"run; {same_tick_skew} us is far inside half the "
        f"{SFN_WRAP_MICROSECONDS} us SFN wrap, so (frame, slot) resolves to one "
        f"counterpart"
    )

    key_frame_slot = lambda row: (row.frame, row.slot)  # noqa: E731
    time_us = lambda row: row.time_us  # noqa: E731

    tick_index = {(tick.frame, tick.slot, tick.time_us): tick for tick in ticks}
    result.alignments.append(
        align_one_to_one(
            left_source="NRUE_MAC_BSR_STATUS",
            right_source="NRUE_MAC_RLC_BUFFER_STATUS(tick totals)",
            left=bsr_rows,
            right=ticks,
            key_of=key_frame_slot,
            time_of=time_us,
            join_keys=("frame", "slot", "tracer_time"),
            left_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
            right_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
            max_skew_us=same_tick_skew,
            max_skew_justification=same_tick_reason,
        )
    )

    if dci_path is not None:
        require_same_run(bsr_path, dci_path)
        dci_rows = [
            row
            for row in read_simple_rows(dci_path, DCI_GRANT_HEADER)
            if row["direction"] == "1"
        ]
        parsed_dci = [
            {
                "time_us": parse_tracer_time(row["time"]),
                "frame": int(row["sched_frame"]),
                "slot": int(row["sched_slot"]),
                "mcs": int(row["mcs"]),
                "tbs": int(row["tbs"]),
            }
            for row in dci_rows
        ]
        unwrapped = unwrap_tracer_times([row["time_us"] for row in parsed_dci])
        for row, when in zip(parsed_dci, unwrapped):
            row["time_us"] = when

        counts: Dict[str, int] = {}
        for row in parsed_dci:
            counts[str(row["mcs"])] = counts.get(str(row["mcs"]), 0) + 1
        result.grant_mcs_counts = dict(sorted(counts.items(), key=lambda kv: int(kv[0])))
        result.grant_tbs = describe_distribution([float(r["tbs"]) for r in parsed_dci])

        # The DCI row is logged when the grant is *received*, but its
        # sched_frame/sched_slot name a later slot, so this is deliberately not
        # a same-tick join: the K2 offset between the two instants is the thing
        # being measured. The window is widened to just under half the measured
        # same-key recurrence so it only disambiguates which grant is meant.
        dci_skew, dci_reason = recurrence_derived_skew(
            parsed_dci,
            lambda row: (row["frame"], row["slot"]),
            lambda row: row["time_us"],
            source_label="NRUE_MAC_DCI_GRANT(UL)",
            purpose=(
                "the residual skew is the K2 offset from grant reception to "
                "the scheduled slot and is reported, not constrained"
            ),
        )
        result.alignments.append(
            align_one_to_one(
                left_source="NRUE_MAC_BSR_STATUS",
                right_source="NRUE_MAC_DCI_GRANT(UL)",
                left=bsr_rows,
                right=parsed_dci,
                key_of=key_frame_slot,
                time_of=time_us,
                right_key_of=lambda row: (row["frame"], row["slot"]),
                right_time_of=lambda row: row["time_us"],
                join_keys=("sched_frame", "sched_slot", "tracer_time"),
                left_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
                right_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
                max_skew_us=dci_skew,
                max_skew_justification=dci_reason,
            )
        )

    if tx_path is not None:
        require_same_run(bsr_path, tx_path)
        tx_rows = read_simple_rows(tx_path, TX_BITS_HEADER)
        parsed_tx = [
            {
                "time_us": parse_tracer_time(row["time"]),
                "frame": int(row["frame"]),
                "slot": int(row["slot"]),
                "bits": int(row["number_of_bits"]),
            }
            for row in tx_rows
        ]
        unwrapped = unwrap_tracer_times([row["time_us"] for row in parsed_tx])
        for row, when in zip(parsed_tx, unwrapped):
            row["time_us"] = when
        result.alignments.append(
            align_one_to_one(
                left_source="NRUE_MAC_BSR_STATUS",
                right_source="UE_PHY_UL_PAYLOAD_TX_BITS",
                left=bsr_rows,
                right=parsed_tx,
                key_of=key_frame_slot,
                time_of=time_us,
                right_key_of=lambda row: (row["frame"], row["slot"]),
                right_time_of=lambda row: row["time_us"],
                join_keys=("frame", "slot", "tracer_time"),
                left_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
                right_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
                max_skew_us=same_tick_skew,
                max_skew_justification=same_tick_reason,
            )
        )

        result.lagged_association = descriptive_associations(
            ticks=ticks,
            bsr_rows=bsr_rows,
            granted_tb_bytes=[
                (row["frame"], row["slot"], row["time_us"], float(row["bits"]) / 8.0)
                for row in parsed_tx
            ],
            same_tick_skew_us=same_tick_skew,
        )

    if pusch_path is not None:
        # gNB and UE softmodems run on one host under RFsim, and every T event
        # is stamped with CLOCK_REALTIME at its call site, so the two CSVs share
        # a clock domain. The join key still needs wrap disambiguation, and the
        # residual skew is the physical UE-transmit to gNB-process offset, which
        # is reported rather than assumed away.
        pusch_rows = read_simple_rows(pusch_path, PUSCH_POWER_HEADER)
        parsed_pusch = [
            {
                "time_us": parse_tracer_time(row["time"]),
                "frame": int(row["frame"]),
                "slot": int(row["slot"]),
                "snr_db": int(row["snrx10"]) / 10.0,
            }
            for row in pusch_rows
        ]
        unwrapped = unwrap_tracer_times([row["time_us"] for row in parsed_pusch])
        for row, when in zip(parsed_pusch, unwrapped):
            row["time_us"] = when
        result.gnb_snr_db = describe_distribution(
            [float(row["snr_db"]) for row in parsed_pusch]
        )
        cross_node_skew, cross_node_reason = recurrence_derived_skew(
            parsed_pusch,
            lambda row: (row["frame"], row["slot"]),
            lambda row: row["time_us"],
            source_label="GNB_MAC_PUSCH_POWER_CONTROL",
            purpose=(
                "the residual skew is the physical UE-transmit to "
                "gNB-process offset and is reported, not constrained"
            ),
        )
        result.alignments.append(
            align_one_to_one(
                left_source="NRUE_MAC_BSR_STATUS(UE)",
                right_source="GNB_MAC_PUSCH_POWER_CONTROL(gNB)",
                left=bsr_rows,
                right=parsed_pusch,
                key_of=key_frame_slot,
                time_of=time_us,
                right_key_of=lambda row: (row["frame"], row["slot"]),
                right_time_of=lambda row: row["time_us"],
                join_keys=("frame", "slot", "tracer_time"),
                left_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
                right_domain=ClockDomain.T_TRACER_REALTIME_LOCAL,
                max_skew_us=cross_node_skew,
                max_skew_justification=cross_node_reason,
            )
        )

    # Offered traffic is summarized from its own log but never joined: the
    # sender writes epoch seconds while the tracer writes a date-less local
    # time, and no measured bridge between the two printed forms exists.
    result.offered_load = read_offered_load(run)
    if run.traffic_sender is not None:
        result.refused_alignments.append(
            {
                "left_source": "NRUE_MAC_RLC_BUFFER_STATUS",
                "right_source": "traffic/sender.csv",
                "refusal": ClockDomainError.__name__,
                "reason": (
                    "sender wall_time_s is EPOCH_WALL_SECONDS; the tracer CSV "
                    "renders CLOCK_REALTIME as date-less local HH:MM:SS "
                    f"({CODE_CITATIONS['t_csv_time_render']}). The underlying "
                    "clock is the same, but recovering the offset needs the "
                    "trace date and UTC offset reconstructed, which is an "
                    "assumption and not a measured bridge"
                ),
            }
        )

    backlog_values = [tick.total_bytes for tick in ticks]
    result.saturation = [
        saturation_report(backlog_values, scale=scale, label=label, status=status)
        for label, scale, status in candidate_scales(backlog_values)
    ]

    result.enqueue_evidence = qualify_enqueue_evidence(run)
    result.enqueue_instant_evidence = result.enqueue_evidence.qualified

    # A4: does this run retain the UE-side downlink measurement at all?
    meas_path = run.ue_csv("UE_PHY_MEAS.csv")
    if meas_path is not None:
        meas = describe_file(meas_path, evidence_root)
        result.provenance.append(meas)
        result.ue_phy_meas_rows = meas.row_count
        result.ue_phy_meas_present = meas.row_count > 0

    result.pre_action_qualification = {
        source.name: qualify_pre_action_source(
            source, enqueue_instant_evidence=result.enqueue_instant_evidence
        ).value
        for source in CANDIDATE_SOURCES
    }
    return result


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


@dataclass
class AuditReport:
    """The audit's complete, serializable finding."""

    audit_id: str
    audit_version: int
    evidence_root: str
    runs: List[RunAudit]
    coverage_verdict: CoverageVerdict
    coverage_rationale: str
    recommended_pre_action_source: str
    pre_action_status: PreActionQualification
    qualified_run_ids: Tuple[str, ...]
    total_run_count: int
    calibration_required: bool

    def to_json(self) -> Dict[str, object]:
        return {
            "audit_id": self.audit_id,
            "audit_version": self.audit_version,
            "evidence_root": self.evidence_root,
            "sources": [source.to_json() for source in CANDIDATE_SOURCES],
            "channel_signal_candidates": [
                item.to_json() for item in CHANNEL_SIGNAL_CANDIDATES
            ],
            "payload_authority": {
                "action_catalog_relpath": SPLITFUSION_ACTION_CATALOG_RELPATH,
                "action_count": SPLITFUSION_ACTION_COUNT,
                "action_payload_min_bytes": SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES,
                "action_payload_max_bytes": SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES,
                "run3_support_min_bytes": RUN3_MODELED_SUPPORT_MIN_BYTES,
                "run3_support_max_bytes": RUN3_MODELED_SUPPORT_MAX_BYTES,
                "retired_permodel_payload_bytes": list(
                    RETIRED_PERMODEL_PAYLOAD_BYTES
                ),
            },
            "required_ordering": ORDERING_CHAIN_TEXT,
            "bsr_vs_rlc_preference": BSR_VS_RLC_PREFERENCE_NOTE,
            "ue_phy_meas_retained_anywhere": any(
                run.ue_phy_meas_present for run in self.runs
            ),
            "code_citations": dict(CODE_CITATIONS),
            "runs": [run.to_json() for run in self.runs],
            "coverage_verdict": self.coverage_verdict.value,
            "coverage_rationale": self.coverage_rationale,
            "recommended_pre_action_source": self.recommended_pre_action_source,
            "pre_action_status": self.pre_action_status.value,
            "qualified_run_ids": list(self.qualified_run_ids),
            "qualified_run_count": len(self.qualified_run_ids),
            "total_run_count": self.total_run_count,
            "calibration_required": self.calibration_required,
        }


def decide_coverage(runs: Sequence[RunAudit]) -> Tuple[CoverageVerdict, str]:
    """Assign exactly one coverage status from what the runs actually vary.

    Identifying how 12 modes and a continuous ``q`` move future backlog needs
    two things at once: the offered payload must *move* across the action range
    while backlog is observed, and the causal ordering must be demonstrated for
    every admitted decision in every run.

    The second condition is the one the earlier implementation omitted.
    ACTION_CONDITIONED is now unreachable while any run's ordering is
    unresolved, so payload variation alone can no longer promote the verdict.
    """
    offered: List[int] = []
    any_within_run_variation = False
    for run in runs:
        if run.offered_load is None or not run.offered_load.present:
            continue
        sizes = run.offered_load.distinct_frame_bytes
        offered.extend(sizes)
        if run.offered_load.varies_within_run:
            any_within_run_variation = True

    distinct = sorted(set(offered))
    if not distinct:
        return (
            CoverageVerdict.INSUFFICIENT,
            "no retained run records its offered traffic, so offered load "
            "cannot be related to observed backlog at all",
        )

    # Causal ordering must hold for every run, not for one lucky run.
    unresolved = [
        run.run_id
        for run in runs
        if not (run.enqueue_evidence and run.enqueue_evidence.qualified)
    ]
    ordering_resolved = not unresolved

    inside = [
        value
        for value in distinct
        if SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES
        <= value
        <= SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES
    ]
    spans_range = len(inside) > 1

    if any_within_run_variation and spans_range and ordering_resolved:
        return (
            CoverageVerdict.ACTION_CONDITIONED,
            "at least one run varies offered payload across the frozen "
            f"{SPLITFUSION_ACTION_COUNT}-action range while backlog is "
            "observed, and the causal ordering is demonstrated for every "
            "admitted decision in every run",
        )

    rationale = (
        f"offered payload takes {len(distinct)} distinct value(s) across all "
        f"runs ({', '.join(f'{value} B' for value in distinct)}), each fixed "
        f"within its own run at a fixed period. Against the frozen "
        f"{SPLITFUSION_ACTION_COUNT}-action catalogue "
        f"[{SPLITFUSION_ACTION_PAYLOAD_MIN_BYTES}, "
        f"{SPLITFUSION_ACTION_PAYLOAD_MAX_BYTES}] B these values sit *inside* "
        f"the low end of the action range rather than below it, and "
        f"{len(inside)} of them also fall inside the Run-3 modeled support "
        f"[{RUN3_MODELED_SUPPORT_MIN_BYTES}, {RUN3_MODELED_SUPPORT_MAX_BYTES}] B. "
        f"They remain insufficient for a different reason than coverage of the "
        f"low end: there are only {len(distinct)} sizes, payload is constant "
        f"within each run, and no run pairs a changed payload with an observed "
        f"queue transition, so no action-conditioned queue response is "
        f"identifiable"
    )
    if not ordering_resolved:
        shown = ", ".join(unresolved[:3]) + ("..." if len(unresolved) > 3 else "")
        rationale += (
            f". The causal ordering is also unresolved in {len(unresolved)} of "
            f"{len(runs)} run(s) ({shown}): no run demonstrates "
            f"{ORDERING_CHAIN_TEXT}, so the pre-action read ordering is "
            f"unproven independently of payload coverage"
        )
        return CoverageVerdict.INSUFFICIENT, rationale
    return CoverageVerdict.CARRIER_AND_NORMALIZATION_ONLY, rationale


def audit_evidence_root(evidence_root: Path) -> AuditReport:
    """Run the full audit over every logical run under ``evidence_root``."""
    evidence_root = evidence_root.resolve()
    if not evidence_root.is_dir():
        raise AuditError(f"evidence root {evidence_root} is not a directory")
    runs = [audit_run(run, evidence_root) for run in discover_runs(evidence_root)]
    verdict, rationale = decide_coverage(runs)

    # Qualification is per run and is only promoted to a global statement when
    # EVERY run qualifies. One qualified file, or one qualified run, must never
    # stand in for the rest.
    qualified_runs = [
        run.run_id
        for run in runs
        if run.enqueue_evidence and run.enqueue_evidence.qualified
    ]
    every_run_qualified = bool(runs) and len(qualified_runs) == len(runs)
    status = qualify_pre_action_source(
        RLC_BUFFER_SOURCE, enqueue_instant_evidence=every_run_qualified
    )
    if status is PreActionQualification.QUALIFIED:
        recommended = RLC_BUFFER_SOURCE.name
    else:
        recommended = "UNRESOLVED"
    return AuditReport(
        audit_id=AUDIT_ID,
        audit_version=AUDIT_VERSION,
        evidence_root=str(evidence_root),
        runs=runs,
        coverage_verdict=verdict,
        coverage_rationale=rationale,
        recommended_pre_action_source=recommended,
        pre_action_status=status,
        qualified_run_ids=tuple(qualified_runs),
        total_run_count=len(runs),
        calibration_required=(
            status is not PreActionQualification.QUALIFIED
            or verdict is not CoverageVerdict.ACTION_CONDITIONED
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of retained OAI UE traces for a pre-action "
            "backlog feature. Reads evidence; never modifies it."
        )
    )
    parser.add_argument(
        "--evidence-root",
        required=True,
        type=Path,
        help="directory to search for '**/ttracer/ue/csv/NRUE_MAC_BSR_STATUS.csv'",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path for the JSON report (must not be inside the evidence tree)",
    )
    args = parser.parse_args(argv)

    report = audit_evidence_root(args.evidence_root)
    payload = json.dumps(report.to_json(), indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n")
    else:
        print(payload)

    print(
        f"\nruns={len(report.runs)} "
        f"coverage={report.coverage_verdict.value} "
        f"pre_action={report.pre_action_status.value} "
        f"recommended_source={report.recommended_pre_action_source} "
        f"calibration_required={report.calibration_required}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
