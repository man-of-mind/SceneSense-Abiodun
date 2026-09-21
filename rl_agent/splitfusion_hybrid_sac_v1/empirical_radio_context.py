"""Hash-pinned OAI-calibrated radio context for the D1 simulator pilot.

The CSV is simulator/testbed calibration evidence, not a deployable UE policy
carrier.  Sampling first chooses the hidden profile uniformly and then one
joint-valid row uniformly within that profile.  Achieved PUSCH SNR and MCS
always come from that same row; target/profile/trace identities never enter
the policy observation.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .anchor_store import NETWORK_PROFILE_ORDER
from .state_reward_transition_contract import (
    BsrReportType,
    BsrReportV1,
    BsrScope,
    BsrSource,
    RadioEventProvenanceV1,
    RadioObservationV1,
    RadioSourceWall,
)
from .transaction_identity import canonical_sha256

__all__ = [
    "CALIBRATION_RELATIVE_PATH",
    "CALIBRATION_SHA256",
    "GENESIS_BSR_JUSTIFICATION",
    "MCS_TABLE_ID",
    "MCS_TABLE_MAX_INDEX",
    "OaiRadioCalibrationStoreV1",
    "RadioCalibrationError",
    "RadioCalibrationRowV1",
    "RadioContextDrawV1",
    "RadioContextSamplerV1",
    "RadioSamplerStateV1",
    "round_mcs_median_unbiased",
]


CALIBRATION_RELATIVE_PATH = (
    "rl_agent/experiments/oai_target_snr_replay_pilot_v1/"
    "20260822_live_01/replay_intervals.csv"
)
CALIBRATION_SHA256 = (
    "00144e6970cf7a605584a8d77450f631f1a1ca83acedd3998b4137252c96fa3a"
)
MCS_TABLE_ID = "NR_UL_MCS_TABLE_0"
MCS_TABLE_MAX_INDEX = 28
GENESIS_BSR_JUSTIFICATION = (
    "TRUE_ONE_STEP_EPISODE_GENESIS_EMPTY_QUEUE_OBSERVED_BEFORE_THE_SELECTED_"
    "ACTION_IS_ENQUEUED; NO_PREDECESSOR_TRAFFIC; NOT_MISSING_DATA_ZERO_FILL"
)


class RadioCalibrationError(ValueError):
    """The calibration bytes, inventory, or sampled row violate D1."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_optional(text: object) -> Optional[float]:
    if text is None or str(text).strip() == "":
        return None
    try:
        value = float(str(text))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class RadioCalibrationRowV1:
    """Environment-hidden row; no identity field is policy-visible."""

    csv_row_number: int
    network_profile: str
    trace_id: str
    trace_step_index: int
    target_snr_db: float
    achieved_pusch_snr_median_db: float
    mcs_median: float
    row_sha256: str


@dataclass(frozen=True, slots=True)
class RadioContextDrawV1:
    """Hidden audit result plus the only policy-admissible radio record."""

    observation: RadioObservationV1
    hidden_profile: str
    hidden_csv_row_number: int
    hidden_trace_id: str
    hidden_trace_step_index: int
    hidden_target_snr_db: float
    hidden_row_sha256: str
    mcs_median: float
    rounded_mcs_index: int
    rounding_status: str
    genesis_bsr_justification: str = GENESIS_BSR_JUSTIFICATION


@dataclass(frozen=True, slots=True)
class RadioSamplerStateV1:
    master_seed: int
    draw_count: int
    profile_rng_state: tuple
    row_rng_state: tuple
    rounding_rng_state: tuple


def _domain_seed(master_seed: int, domain: str) -> int:
    payload = f"{master_seed}:{domain}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def round_mcs_median_unbiased(value: float, rng: random.Random) -> Tuple[int, str]:
    """Seeded stochastic floor/ceil only for exact half-integer medians."""
    if not isinstance(rng, random.Random):
        raise RadioCalibrationError("rng must be an explicit local random.Random")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RadioCalibrationError("MCS median must be a finite scalar")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= MCS_TABLE_MAX_INDEX:
        raise RadioCalibrationError(f"MCS median is outside table 0: {value!r}")
    floor = math.floor(numeric)
    fraction = numeric - floor
    if fraction == 0.0:
        return floor, "EXACT_INTEGER_NO_ROUNDING"
    if fraction != 0.5:
        raise RadioCalibrationError(
            f"calibration MCS must be integral or half-integral, got {numeric}"
        )
    rounded = floor + rng.getrandbits(1)
    return rounded, "SEEDED_UNBIASED_FLOOR_CEIL_FOR_HALF_INTEGER"


@dataclass(frozen=True, slots=True)
class OaiRadioCalibrationStoreV1:
    """Immutable joint-valid calibration population grouped by profile."""

    rows: Tuple[RadioCalibrationRowV1, ...]
    rows_by_profile: Mapping[str, Tuple[RadioCalibrationRowV1, ...]]
    source_sha256: str
    source_relative_path: str = CALIBRATION_RELATIVE_PATH
    source_row_count: int = 400
    joint_valid_row_count: int = 399

    @classmethod
    def load_registered(
        cls, *, project_root: Optional[Path] = None
    ) -> "OaiRadioCalibrationStoreV1":
        root = (
            Path(__file__).resolve().parents[2]
            if project_root is None
            else Path(project_root).resolve(strict=True)
        )
        path = (root / CALIBRATION_RELATIVE_PATH).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RadioCalibrationError("calibration path escapes project root") from exc
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != CALIBRATION_SHA256:
            raise RadioCalibrationError(
                "radio calibration SHA-256 drift: "
                f"expected {CALIBRATION_SHA256}, observed {observed_sha256}"
            )

        valid: list[RadioCalibrationRowV1] = []
        source_count = 0
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "profile_id",
                "trace_id",
                "trace_step_index",
                "target_snr_db",
                "achieved_pusch_snr_median_db",
                "mcs_median",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise RadioCalibrationError("calibration CSV header drift")
            for source_count, raw in enumerate(reader, start=1):
                profile = str(raw["profile_id"])
                if profile not in NETWORK_PROFILE_ORDER:
                    raise RadioCalibrationError(f"unknown profile {profile!r}")
                achieved = _finite_optional(raw["achieved_pusch_snr_median_db"])
                mcs = _finite_optional(raw["mcs_median"])
                if achieved is None or mcs is None:
                    continue
                target = _finite_optional(raw["target_snr_db"])
                if target is None:
                    raise RadioCalibrationError("joint-valid row has invalid target SNR")
                if not 0.0 <= mcs <= MCS_TABLE_MAX_INDEX:
                    raise RadioCalibrationError("joint-valid row has out-of-table MCS")
                if (mcs - math.floor(mcs)) not in (0.0, 0.5):
                    raise RadioCalibrationError("MCS median is not integer/half-integer")
                row_document = {key: raw[key] for key in sorted(raw)}
                valid.append(
                    RadioCalibrationRowV1(
                        csv_row_number=source_count + 1,
                        network_profile=profile,
                        trace_id=str(raw["trace_id"]),
                        trace_step_index=int(raw["trace_step_index"]),
                        target_snr_db=target,
                        achieved_pusch_snr_median_db=achieved,
                        mcs_median=mcs,
                        row_sha256=canonical_sha256(row_document),
                    )
                )
        if source_count != 400 or len(valid) != 399:
            raise RadioCalibrationError(
                f"calibration inventory drift: rows={source_count}, valid={len(valid)}"
            )
        grouped: Dict[str, Tuple[RadioCalibrationRowV1, ...]] = {
            profile: tuple(row for row in valid if row.network_profile == profile)
            for profile in NETWORK_PROFILE_ORDER
        }
        expected_counts = {
            "FAVORABLE_STABLE": 100,
            "MID_VARIABLE": 100,
            "ADVERSE_STABLE": 99,
            "FADE_RECOVERY": 100,
        }
        if {key: len(value) for key, value in grouped.items()} != expected_counts:
            raise RadioCalibrationError("joint-valid per-profile inventory drift")
        snr_values = [row.achieved_pusch_snr_median_db for row in valid]
        mcs_values = [row.mcs_median for row in valid]
        if (min(snr_values), max(snr_values)) != (6.0, 23.5):
            raise RadioCalibrationError("achieved-SNR calibration range drift")
        if (min(mcs_values), max(mcs_values)) != (9.0, 28.0):
            raise RadioCalibrationError("MCS calibration range drift")
        pair_profiles: Dict[Tuple[float, float], set[str]] = {}
        for row in valid:
            pair_profiles.setdefault(
                (row.achieved_pusch_snr_median_db, row.mcs_median), set()
            ).add(row.network_profile)
        aliased_rows = sum(
            len(pair_profiles[(row.achieved_pusch_snr_median_db, row.mcs_median)]) > 1
            for row in valid
        )
        if aliased_rows != 140:
            raise RadioCalibrationError("cross-profile SNR/MCS alias inventory drift")
        return cls(
            rows=tuple(valid),
            rows_by_profile=MappingProxyType(grouped),
            source_sha256=observed_sha256,
        )

    @property
    def achieved_snr_range_db(self) -> Tuple[float, float]:
        values = tuple(row.achieved_pusch_snr_median_db for row in self.rows)
        return min(values), max(values)

    @property
    def profile_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {profile: len(self.rows_by_profile[profile]) for profile in NETWORK_PROFILE_ORDER}
        )

    @property
    def cross_profile_aliased_row_count(self) -> int:
        pair_profiles: Dict[Tuple[float, float], set[str]] = {}
        for row in self.rows:
            pair_profiles.setdefault(
                (row.achieved_pusch_snr_median_db, row.mcs_median), set()
            ).add(row.network_profile)
        return sum(
            len(pair_profiles[(row.achieved_pusch_snr_median_db, row.mcs_median)]) > 1
            for row in self.rows
        )


class RadioContextSamplerV1:
    """Stateful local-RNG sampler; global RNG state is never touched."""

    def __init__(self, store: OaiRadioCalibrationStoreV1, *, seed: int) -> None:
        if not isinstance(store, OaiRadioCalibrationStoreV1):
            raise RadioCalibrationError("store must be OaiRadioCalibrationStoreV1")
        if type(seed) is not int:
            raise RadioCalibrationError("seed must be an exact integer")
        self._store = store
        self._master_seed = seed
        self._profile_rng = random.Random(
            _domain_seed(seed, "D1_RADIO_PROFILE_V1")
        )
        self._row_rng = random.Random(_domain_seed(seed, "D1_RADIO_ROW_V1"))
        self._rounding_rng = random.Random(
            _domain_seed(seed, "D1_MCS_ROUNDING_V1")
        )
        self._draw_count = 0

    def state_dict(self) -> RadioSamplerStateV1:
        return RadioSamplerStateV1(
            master_seed=self._master_seed,
            draw_count=self._draw_count,
            profile_rng_state=self._profile_rng.getstate(),
            row_rng_state=self._row_rng.getstate(),
            rounding_rng_state=self._rounding_rng.getstate(),
        )

    def load_state_dict(self, state: RadioSamplerStateV1) -> None:
        self.validate_state_dict(state)
        # Validation above used disposable RNGs, so these three commits cannot
        # leave a partially restored sampler on malformed input.
        self._profile_rng.setstate(state.profile_rng_state)
        self._row_rng.setstate(state.row_rng_state)
        self._rounding_rng.setstate(state.rounding_rng_state)
        self._draw_count = state.draw_count

    def validate_state_dict(self, state: RadioSamplerStateV1) -> None:
        """Validate a checkpoint without mutating any live RNG stream."""
        if type(state) is not RadioSamplerStateV1:
            raise RadioCalibrationError("radio state must be RadioSamplerStateV1")
        if state.master_seed != self._master_seed:
            raise RadioCalibrationError("radio sampler master-seed mismatch")
        if type(state.draw_count) is not int or state.draw_count < 0:
            raise RadioCalibrationError("invalid radio draw count")
        try:
            probes = (random.Random(), random.Random(), random.Random())
            for probe, candidate in zip(
                probes,
                (
                    state.profile_rng_state,
                    state.row_rng_state,
                    state.rounding_rng_state,
                ),
            ):
                probe.setstate(candidate)
        except (TypeError, ValueError) as exc:
            raise RadioCalibrationError("invalid radio RNG state") from exc

    def sample(self, *, observed_ns: int, control_session_id: str) -> RadioContextDrawV1:
        if type(observed_ns) is not int or observed_ns < 0:
            raise RadioCalibrationError("observed_ns must be a non-negative int")
        profile = NETWORK_PROFILE_ORDER[
            self._profile_rng.randrange(len(NETWORK_PROFILE_ORDER))
        ]
        population = self._store.rows_by_profile[profile]
        row = population[self._row_rng.randrange(len(population))]
        mcs_index, rounding_status = round_mcs_median_unbiased(
            row.mcs_median, self._rounding_rng
        )
        epoch = "d1-oai-calibrated-simulator-epoch"
        event = RadioEventProvenanceV1(
            source_wall=RadioSourceWall.SIMULATOR_TESTBED,
            source_event_id=f"calibration-csv-row-{row.csv_row_number}",
            source_event_index=row.csv_row_number,
            source_event_timestamp_ns=observed_ns,
            collector_ingest_wall_time_ns=observed_ns,
            collector_ingest_monotonic_ns=observed_ns,
            ran_epoch_id=epoch,
            control_session_id=control_session_id,
            raw_event_sha256=row.row_sha256,
        )
        bsr_sha = canonical_sha256(
            {
                "justification": GENESIS_BSR_JUSTIFICATION,
                "lcg_bytes": [0] * 8,
                "observed_ns": observed_ns,
                "record": "d1_genesis_bsr_v1",
            }
        )
        bsr_event = RadioEventProvenanceV1(
            source_wall=RadioSourceWall.SIMULATOR_TESTBED,
            source_event_id="d1-genesis-empty-bsr",
            source_event_index=0,
            source_event_timestamp_ns=observed_ns,
            collector_ingest_wall_time_ns=observed_ns,
            collector_ingest_monotonic_ns=observed_ns,
            ran_epoch_id=epoch,
            control_session_id=control_session_id,
            raw_event_sha256=bsr_sha,
        )
        bsr = BsrReportV1(
            lcg_bytes=(0,) * 8,
            valid_mask=(True,) * 8,
            missing_reasons=(None,) * 8,
            scope=BsrScope.ALL_GROUPS_LATEST,
            logical_channel_group=0,
            report_type=BsrReportType.SIMULATOR_VECTOR,
            source=BsrSource.SIMULATOR_TESTBED_PRIVILEGED,
            measured_ns=observed_ns,
            event=bsr_event,
        )
        observation = RadioObservationV1.for_simulator_testbed(
            achieved_snr_db=row.achieved_pusch_snr_median_db,
            snr_measured_ns=observed_ns,
            mcs_index=mcs_index,
            mcs_table_id=MCS_TABLE_ID,
            mcs_measured_ns=observed_ns,
            bsr_bytes=0,
            bsr_scope=BsrScope.ALL_GROUPS_LATEST,
            bsr_logical_channel_group=0,
            bsr_measured_ns=observed_ns,
            snr_event=event,
            mcs_event=event,
            bsr_report=bsr,
            source_id="OAI_REPLAY_CALIBRATED_SIMULATOR_TESTBED_CONTEXT",
            source_sha256=self._store.source_sha256,
        )
        result = RadioContextDrawV1(
            observation=observation,
            hidden_profile=profile,
            hidden_csv_row_number=row.csv_row_number,
            hidden_trace_id=row.trace_id,
            hidden_trace_step_index=row.trace_step_index,
            hidden_target_snr_db=row.target_snr_db,
            hidden_row_sha256=row.row_sha256,
            mcs_median=row.mcs_median,
            rounded_mcs_index=mcs_index,
            rounding_status=rounding_status,
        )
        self._draw_count += 1
        return result
