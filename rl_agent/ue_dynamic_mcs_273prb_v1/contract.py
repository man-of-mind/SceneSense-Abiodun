"""Frozen, network-only design for the Run-4 dynamic prior-MCS capture.

This experiment fills one narrow evidence gap.  The retained dynamic MCS
sequence was collected on the legacy 106-PRB/7D2U radio, whereas Run 4 and the
288-cell response surface are bound to 273 PRB/100 MHz/4D5U.  We therefore
capture only the two genuinely time-varying registered channel profiles on the
target radio.  This is not a second response-surface campaign and it does not
measure perception quality.

Importing this module performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

CONTRACT_ID = "ue_dynamic_mcs_273prb_v1"
CONTRACT_VERSION = 1
SCHEMA = "scenesense.ue_dynamic_mcs_273prb.v1"
EXECUTION_TOKEN = "AUTHORIZE_RUN4_DYNAMIC_MCS_273PRB_CAPTURE_V1"
SUCCESS_TERMINAL = "RUN4_DYNAMIC_MCS_273PRB_CAPTURED"
CLAIM_BOUNDARY = (
    "TWO_REGISTERED_DYNAMIC_PROFILE_UE_DECODED_PRIOR_UL_MCS_SEQUENCE_UNDER_"
    "OAI_N78_100MHZ_273PRB_4D5U_V1_FOR_RUN4_OFFLINE_FIT_AND_INTERNAL_"
    "VALIDATION_ONLY_NOT_A_GENERALIZATION_OR_DEPLOYMENT_CLAIM"
)

RADIO_PROFILE_ID = "OAI_N78_100MHZ_273PRB_4D5U_V1"
RADIO = {
    "band": 78,
    "bandwidth_mhz": 100,
    "prb": 273,
    "numerology": 1,
    "downlink_slots": 4,
    "uplink_slots": 5,
    "ue_count": 1,
    "ue_ip": "10.0.0.2",
    "mcs_policy": "sinr",
}

PROFILE_IDS = ("MID_VARIABLE", "FADE_RECOVERY")
PROFILE_IDENTITIES = {
    "MID_VARIABLE": {
        "trace_id": "GM_V2_MID_VARIABLE_SEED_2026082102",
        "seed": 2026082102,
        "trace_sha256": (
            "0d2ef7ff9e7f342c772d0d88a4cec4e9ce9ac1c9946c1402d3087c55c8f0f811"
        ),
    },
    "FADE_RECOVERY": {
        "trace_id": "GM_V2_FADE_RECOVERY_SEED_2026082104",
        "seed": 2026082104,
        "trace_sha256": (
            "ef2ae2d6448d9c7f3208bc81123d301a1a9f5d9b16be3bac4412c8cecf94e7c0"
        ),
    },
}

PERIOD_NS = 100_000_000
FRAMES_PER_PROFILE = 300
FIT_FIRST_INDEX = 0
FIT_LAST_INDEX = 209
VALIDATION_FIRST_INDEX = 210
VALIDATION_LAST_INDEX = 299
ACTION_HOLD_DURATION_TENSORS = 2
SUCCESSOR_DELTA_NS = ACTION_HOLD_DURATION_TENSORS * PERIOD_NS
MCS_MAX_AGE_NS = 200_000_000

# The fixed load is deliberately sub-capacity and exists only to make UE UL
# grants observable.  It is a real registered SplitFusion payload, not iperf.
PROBE_ACTION_ID = 68
PROBE_PROFILE_ID = "split_ae32_uint4_q5000"
PROBE_Q_E4 = 5000
PROBE_PAYLOAD_BYTES = 129_707
PROBE_CHUNK_BYTES = 60_000
PROBE_CHUNKS_PER_FRAME = 3
PROBE_OFFERED_MBPS = PROBE_PAYLOAD_BYTES * 8 * (1e9 / PERIOD_NS) / 1e6

COMMAND_GUARD_NS = 10_000_000
SENDER_START_LEAD_NS = 2_000_000_000
PROFILE_WARMUP_S = 4.0
RECEIVER_TAIL_S = 5.0
PROBE_PORTS = {"MID_VARIABLE": 5461, "FADE_RECOVERY": 5462}

FIT = "FIT"
INTERNAL_VALIDATION = "INTERNAL_VALIDATION"

# Gates fixed before collection.  Missing and stale samples remain explicit in
# the evidence; these gates only decide whether the sequence is informative
# enough to fit the initial environment kernel.
MAX_SCHEDULE_LAG_P99_MS = 5.0
MAX_CLOCK_BRIDGE_RESIDUAL_P95_US = 1.0
MIN_VALID_MCS_FRACTION_PER_PARTITION = 0.90
MIN_UNIQUE_MCS_PER_PROFILE = 3
MIN_GNB_PROVENANCE_COVERAGE = 0.99


class ContractError(RuntimeError):
    """The preregistered design or one of its frozen sources drifted."""


# These inputs are outside the new package and are frozen before any live
# attempt.  The target-radio runner is reused only at these exact bytes.
SOURCE_PINS: Mapping[str, tuple[str, str]] = {
    "target_radio_binding": (
        "rl_agent/ue_mcs_backlog_near_capacity_v1/radio_binding.py",
        "7585f5925cb03ae5d6c0aca68fb311b84ef0af7154a850bbdfc0f46460694356",
    ),
    "target_radio_runner": (
        "rl_agent/ue_mcs_backlog_near_capacity_v1/runner.py",
        "89ef69d4f0a1d9d708b93bb9d74c0f99bb94a925f95132a1efe4a30017f4dd02",
    ),
    "target_radio_config": (
        "rl_agent/ue_mcs_backlog_near_capacity_v1/config_v1.json",
        "96e241aa69bae97799ae65ea4e269d96b364b7d15e53c56686a381c2164dcf48",
    ),
    "base_runner": (
        "rl_agent/ue_mcs_backlog_calibration_v1/runner.py",
        "6c2a74d79e1c2415cd11eb4593314d3782ebcd0536a9f5ab7d9d4a393538741d",
    ),
    "causal_join": (
        "rl_agent/ue_mcs_backlog_calibration_v1/decision_join.py",
        "fd2d0fd7e8ad239dbea231449fc88f41aaf9ae63f5c703ae8fb455123896f22b",
    ),
    "base_contract": (
        "rl_agent/ue_mcs_backlog_calibration_v1/contract.py",
        "d87b4bbddfa116ecd0a98f57bd92ef187537efc92a37b5e4e31dab0e4a991f11",
    ),
    "profile_binding": (
        "rl_agent/configs/splitfusion_phase14b_corrected_four_profile_replay_v1.json",
        "0dafbe9150859111956618f9573b94c48ad83b9edd43d7b06bad9da04f577f69",
    ),
    "profile_traces": (
        "rl_agent/experiments/network_profile_design_v2/20260822_route_b_v2/traces.csv",
        "32f1be66e976cba322803c128eefdeb81a8d31ed64bd253ff85ea1ab0583303d",
    ),
    "profile_design": (
        "rl_agent/experiments/network_profile_design_v2/20260822_route_b_v2/resolved_config.json",
        "ecc14fd000180db1257a96f2501a1e73b95c3ddbab032290ec5600ce6a9d34c7",
    ),
    "campaign": (
        "rl_agent/configs/ue_288_campaign_v1.yaml",
        "f9c8382af5a4b16e95f71fa0e273437ee54eb114b5c39d3d428335a71c9e9057",
    ),
    "action_catalog": (
        "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json",
        "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3",
    ),
    "production_receiver": (
        "rl_agent/ue_n3_structured_udp_receiver.py",
        "3e92ba8f756d2c028dcfe51c2a35f18d4f268a041516ea1e054f349f1b916446",
    ),
    "extractor": (
        "scripts/ttracer_extract_csv_smoke.sh",
        "82f23ae18136379c4ea2511744eb1ed932d74850851885584b8075f9f3cdb488",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources(root: Path = ROOT) -> dict[str, Any]:
    files: dict[str, Any] = {}
    problems: list[str] = []
    for label, (relative, expected) in SOURCE_PINS.items():
        path = root / relative
        observed = sha256_file(path) if path.is_file() else None
        matches = observed == expected
        files[label] = {
            "path": relative,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": matches,
        }
        if not matches:
            problems.append(f"{label}: {observed!r} != {expected}")

    # Independent semantic checks prevent a same-file bookkeeping mistake
    # from silently turning this back into the legacy radio or wrong action.
    if not problems:
        radio = json.loads(
            (root / "rl_agent/configs/oai_radio_baseline_100mhz_4d5u_v1.json")
            .read_text(encoding="utf-8")
        )
        selected = radio["radio"]
        tdd = selected["tdd"]
        observed = {
            "profile_id": radio.get("profile_id"),
            "bandwidth_mhz": int(selected.get("bandwidth_mhz", -1)),
            "prb": int(selected.get("prb", selected.get("n_rb_dl", -1))),
            "downlink_slots": int(tdd.get("downlink_slots", -1)),
            "uplink_slots": int(tdd.get("uplink_slots", -1)),
        }
        expected = {
            "profile_id": RADIO_PROFILE_ID,
            "bandwidth_mhz": 100,
            "prb": 273,
            "downlink_slots": 4,
            "uplink_slots": 5,
        }
        if observed != expected:
            problems.append(f"radio lock semantics drifted: {observed} != {expected}")

        catalog = json.loads(
            (root / SOURCE_PINS["action_catalog"][0]).read_text(encoding="utf-8")
        )
        action = next(
            (row for row in catalog["profiles"]
             if int(row["action_id"]) == PROBE_ACTION_ID),
            None,
        )
        expected_action = (PROBE_PROFILE_ID, PROBE_Q_E4, PROBE_PAYLOAD_BYTES)
        observed_action = None if action is None else (
            str(action["profile_id"]), int(action["q_e4"]),
            int(action["payload"]["zstd_median_bytes"]),
        )
        if observed_action != expected_action:
            problems.append(
                f"probe action drifted: {observed_action} != {expected_action}"
            )
        elif not (
            action["capabilities"]["transport_valid"]
            and action["capabilities"]["agent_action_enabled"]
        ):
            problems.append("probe action is no longer transport-valid and enabled")

    report = {
        "contract_id": CONTRACT_ID,
        "radio_profile_id": RADIO_PROFILE_ID,
        "files": files,
        "problems": problems,
        "verified": not problems,
    }
    if problems:
        raise ContractError("source verification failed: " + "; ".join(problems))
    return report


@dataclass(frozen=True)
class ProfilePlan:
    profile_id: str
    trace_id: str
    seed: int
    trace_sha256: str
    run_index: int
    port: int

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "trace_id": self.trace_id,
            "seed": self.seed,
            "trace_sha256": self.trace_sha256,
            "run_index": self.run_index,
            "port": self.port,
            "frames": FRAMES_PER_PROFILE,
        }


def build_plan() -> tuple[ProfilePlan, ...]:
    return tuple(
        ProfilePlan(
            profile_id=profile_id,
            trace_id=str(PROFILE_IDENTITIES[profile_id]["trace_id"]),
            seed=int(PROFILE_IDENTITIES[profile_id]["seed"]),
            trace_sha256=str(PROFILE_IDENTITIES[profile_id]["trace_sha256"]),
            run_index=index,
            port=PROBE_PORTS[profile_id],
        )
        for index, profile_id in enumerate(PROFILE_IDS)
    )


def partition_for(index: int) -> str:
    if type(index) is not int or not (0 <= index < FRAMES_PER_PROFILE):
        raise ContractError(f"decision index outside [0,{FRAMES_PER_PROFILE - 1}]")
    return FIT if index <= FIT_LAST_INDEX else INTERNAL_VALIDATION


def successor_index(index: int, duration: int = ACTION_HOLD_DURATION_TENSORS) -> int:
    if type(duration) is not int or duration <= 0:
        raise ContractError("duration must be a positive exact int")
    target = index + duration
    if target >= FRAMES_PER_PROFILE:
        raise ContractError("successor leaves the measured profile")
    if partition_for(index) != partition_for(target):
        raise ContractError("successor crosses the preregistered fit/validation reset")
    return target


def registered_transition_indices(
    duration: int = ACTION_HOLD_DURATION_TENSORS,
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for index in range(FRAMES_PER_PROFILE):
        try:
            pairs.append((index, successor_index(index, duration)))
        except ContractError:
            continue
    return tuple(pairs)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def design_record() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "radio_profile_id": RADIO_PROFILE_ID,
        "radio": dict(RADIO),
        "profiles": [item.to_json() for item in build_plan()],
        "decision_grid": {
            "period_ns": PERIOD_NS,
            "frames_per_profile": FRAMES_PER_PROFILE,
            "fit": [FIT_FIRST_INDEX, FIT_LAST_INDEX],
            "internal_validation": [
                VALIDATION_FIRST_INDEX,
                VALIDATION_LAST_INDEX,
            ],
            "hold_duration_tensors": ACTION_HOLD_DURATION_TENSORS,
            "successor_delta_ns": SUCCESSOR_DELTA_NS,
        },
        "probe": {
            "action_id": PROBE_ACTION_ID,
            "profile_id": PROBE_PROFILE_ID,
            "q_e4": PROBE_Q_E4,
            "payload_bytes": PROBE_PAYLOAD_BYTES,
            "chunks_per_frame": PROBE_CHUNKS_PER_FRAME,
            "offered_mbps": PROBE_OFFERED_MBPS,
            "role": "MCS_OBSERVABILITY_PROBE_NOT_A_POLICY_ACTION_LABEL",
        },
        "selection": {
            "source": "UE_DECODED_UL_DCI_ONLY",
            "rule": "LATEST_STRICTLY_PRIOR_ROUND0_TABLE0_UL_GRANT",
            "max_age_ns": MCS_MAX_AGE_NS,
            "missing_and_stale": "EXPLICIT_NULL_NEVER_ZERO_NEVER_IMPUTED",
            "gnb": "VERIFIER_ONLY_NEVER_POLICY_INPUT",
            "target_snr_and_profile_id": "HIDDEN_VERIFIER_METADATA_NEVER_POLICY_INPUT",
        },
        "gates": {
            "max_schedule_lag_p99_ms": MAX_SCHEDULE_LAG_P99_MS,
            "max_clock_bridge_residual_p95_us": (
                MAX_CLOCK_BRIDGE_RESIDUAL_P95_US
            ),
            "min_valid_mcs_fraction_per_partition": (
                MIN_VALID_MCS_FRACTION_PER_PARTITION
            ),
            "min_unique_mcs_per_profile": MIN_UNIQUE_MCS_PER_PROFILE,
            "min_gnb_provenance_coverage": MIN_GNB_PROVENANCE_COVERAGE,
        },
        "limitations": [
            "one measured realization per registered dynamic profile",
            "fit and internal validation are contiguous disjoint portions of that realization",
            "external Route-B live validation remains required",
            "the probe payload is fixed and no action effect is inferred from this capture",
        ],
    }
