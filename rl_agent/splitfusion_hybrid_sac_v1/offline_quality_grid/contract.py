"""Frozen inputs and canonical identities for Phase A1a.

Only episode 05 and episode 06 are named here.  Callers are not allowed to
discover sibling episode directories: this makes the locked 07/08 test wall an
enforced allowlist rather than a convention.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_ID = "splitfusion_exact_offline_quality_grid_v1"
SCHEMA_VERSION = 1
SELECTION_SCHEMA_ID = "splitfusion_exact_offline_frame_selection_v1"
RUN_MANIFEST_SCHEMA_ID = "splitfusion_exact_offline_quality_grid_manifest_v1"
EVIDENCE_LABEL = "EXACT_OFFLINE_CARLA_GT_FOR_EXECUTED_FRAME_MODE_Q"
DEPLOYMENT_CLAIM = "NONE_OFFLINE_ONLY_NO_LIVE_NETWORK_OR_DEPLOYMENT_CLAIM"
PROVISIONAL_QUALITY_CALIBRATION = "PROVISIONAL_RECOMPUTABLE_QUALITY_CALIBRATION"
FROZEN_QUALITY_CALIBRATION = "REGISTERED_FROZEN_QUALITY_CALIBRATION"
# There is deliberately no approved production calibration at this phase.
# Adding a digest here is a reviewed source change; caller-controlled
# RewardSpec provenance can never promote itself into this allowlist.
REGISTERED_FROZEN_REWARD_SPEC_SHA256: frozenset[str] = frozenset()

FAMILIES = ("noAE", "AE128", "AE64", "AE32")
QUANTIZERS = ("UINT8", "UINT6", "UINT4")
Q_E4_GRID = (0, 1500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9400, 9800)
ACTION_CATALOG_Q_E4 = (0, 3000, 5000, 7000, 9000, 9800)
MODE_COUNT = len(FAMILIES) * len(QUANTIZERS)
FIT_SELECTION_COUNT = 512
HELD_SCENE_SELECTION_COUNT = 256
TOTAL_SELECTED_FRAMES = FIT_SELECTION_COUNT + HELD_SCENE_SELECTION_COUNT
EXPECTED_GRID_ROWS = TOTAL_SELECTED_FRAMES * MODE_COUNT * len(Q_E4_GRID)
SPATIAL_CELLS = 21_504
UDP_CHUNK_BYTES_INCLUDING_HEADER = 12_500
UDP_CHUNK_HEADER_BYTES = 8
UDP_PAYLOAD_BYTES_PER_DATAGRAM = (
    UDP_CHUNK_BYTES_INCLUDING_HEADER - UDP_CHUNK_HEADER_BYTES
)
SFD1_CONTEXT_PROTOCOL_VERSION = 2
SFD1_COMMON_HEADER_BYTES = 36
SFD1_CONTEXT_FIXED_BYTES = 144
SFD1_DYNAMIC_ACTION_SENTINEL = (1 << 32) - 1
SFD1_REGISTERED_ACTION_STATUS = "REGISTERED_ANCHOR_ACTION_ID"
SFD1_OFF_ANCHOR_SIZE_STATUS = (
    "EXACT_V2_SIZE_ACCOUNTING_WITH_NONDISPATCHABLE_DYNAMIC_ACTION_SENTINEL"
)

DATASET_ROOT_RELPATH = "data_collection/experiments/route_b_perception_v3"
MODEL_DATASET_RELPATH = (
    "experiments/route_b_v3_1_expanded_train_camera_plane_v1/20260828_094151"
)
MODEL_DATASET_MANIFEST_RELPATH = f"{MODEL_DATASET_RELPATH}/dataset/manifest.csv"
MODEL_EVALUATION_ROOT_RELPATH = f"{MODEL_DATASET_RELPATH}/contracts/v010/val"
VALIDATION_AVO_TABLE_RELPATH = (
    "experiments/actor_volume_observability_model_comparison_v1/"
    "20260901_repaired_tolerance_cpu_once/actor_volume_observability_table.csv"
)
VALIDATION_AVO_TABLE_SHA256 = (
    "abb976f388ad33e8806d080750e9e7fbe1b1eb60e7e18ea55bedc60dce011386"
)
MODEL_DATASET_MANIFEST_SHA256 = (
    "5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2"
)
MODEL_VALIDATION_FRAMES = 3_345


@dataclass(frozen=True, slots=True)
class EpisodeBinding:
    episode_id: str
    grid_split: str
    expected_rows: int
    selected_rows: int
    manifest_sha256: str

    @property
    def manifest_relpath(self) -> str:
        return f"{DATASET_ROOT_RELPATH}/{self.episode_id}/manifest.csv"

    @property
    def root_relpath(self) -> str:
        return f"{DATASET_ROOT_RELPATH}/{self.episode_id}"


EPISODES = (
    EpisodeBinding(
        "canonical_v3_05_val_30_30_s601_tm1601",
        "fit",
        1_539,
        FIT_SELECTION_COUNT,
        "7e35769b0310d14033f7806b46c4ceef5aa62480b17cfeb98dffe6182fae8d99",
    ),
    EpisodeBinding(
        "canonical_v3_06_val_50_50_s602_tm1602",
        "held_scene",
        1_814,
        HELD_SCENE_SELECTION_COUNT,
        "db7c331d6c8b763bbce1df81a1bc1a2178f10e345a871e890aafd84c65a5e2c9",
    ),
)
ALLOWED_EPISODE_IDS = frozenset(item.episode_id for item in EPISODES)

ACTION_CATALOG_RELPATH = (
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json"
)
ACTION_CATALOG_SHA256 = (
    "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3"
)

CHECKPOINTS: Mapping[str, tuple[str, str]] = {
    "FCOS": (
        "experiments/supervisor_handoff/splitfusion_models_and_288_results_20260909/"
        "checkpoints/splitfusion_fcos_epoch_026.pt",
        "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f",
    ),
    "ranker": (
        "experiments/supervisor_handoff/splitfusion_models_and_288_results_20260909/"
        "checkpoints/hybrid_q_ranker_epoch_04.pt",
        "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5",
    ),
    "AE128": (
        "experiments/supervisor_handoff/splitfusion_models_and_288_results_20260909/"
        "checkpoints/ae128_epoch_08.pt",
        "0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72",
    ),
    "AE64": (
        "experiments/supervisor_handoff/splitfusion_models_and_288_results_20260909/"
        "checkpoints/ae64_epoch_12.pt",
        "dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0",
    ),
    "AE32": (
        "experiments/supervisor_handoff/splitfusion_models_and_288_results_20260909/"
        "checkpoints/ae32_epoch_08.pt",
        "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271",
    ),
}

# Non-Python files which directly select or parameterize the executed runtime.
# These are pinned independently rather than allowing a mutable JSON file to
# redirect a loader and self-authorize a new package/checkpoint.  The paths are
# the files actually opened by the frozen runtime, not supervisor-pack copies.
RUNTIME_BEHAVIORAL_ARTIFACTS: Mapping[str, tuple[str, str]] = {
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/recovery_config.json": (
        "0fef30f44cbbb3694627c8add8b30e569b38b5306ab3f065766118fa864de677",
        "numerical-recovery package/checkpoint/config selector",
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json": (
        "91889e4af2a5088853d192c4c16c39249c58e44b41b566e0e6d5586e5f717631",
        "base model architecture, dataset and inference configuration",
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_v1/transport_schema.json": (
        "6353862377e4e74583085919b866f7f8dce8d15dafcd9c0f1508419ec4853f8a",
        "base split transport schema selected by recovery configuration",
    ),
    "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1/"
    "20260829_214123/TRAIN_ONLY_PRIORS.json": (
        "90ce336604ff83ccd7ea813ae4c12434b6f122b90a906e83d36742ff7f6700f0",
        "runtime model-construction priors",
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/"
    "PERCEPTION_FORWARD_LOCK_P025_V1.json": (
        "86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1",
        "frozen perception forward/checkpoint lock",
    ),
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/locked_config.json": (
        "b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6",
        "frozen continuous-q/ranker configuration",
    ),
    "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/"
    "20260830_recovered_epoch10_gate_v1/checkpoints/epoch_026.pt": (
        "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f",
        "actual runtime perception checkpoint",
    ),
    "experiments/splitfusion_fcos_hybrid_q_v1/20260901_185725_phase5_ranker_training/"
    "checkpoints/ranker_epoch_04.pt": (
        "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5",
        "actual runtime saliency-ranker checkpoint",
    ),
    "experiments/splitfusion_fcos_ae_v1/20260902_220623_phase9c_ae128_training/"
    "checkpoints/ae128_epoch_08.pt": (
        "0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72",
        "actual runtime AE128 checkpoint",
    ),
    "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae64_training/"
    "checkpoints/ae64_epoch_12.pt": (
        "dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0",
        "actual runtime AE64 checkpoint",
    ),
    "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae32_training/"
    "checkpoints/ae32_epoch_08.pt": (
        "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271",
        "actual runtime AE32 checkpoint",
    ),
    "experiments/splitfusion_fcos_ae_v1/20260902_220623_phase9c_ae128_training/"
    "holdout_selection/holdout_selection.json": (
        "69e49deac302fc46c1eec56036e3ab3d769b3aac10b76541cfb4abb80f878194",
        "AE128 selected-checkpoint decision",
    ),
    "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae64_training/"
    "holdout_selection_ae64/ae64_holdout_selection.json": (
        "0d2fe444574d3fdc9aee287448084bf2cfc1efa2d0ec6944ac07355d9ff7c87e",
        "AE64 selected-checkpoint decision",
    ),
    "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae32_training/"
    "holdout_selection_ae32/ae32_holdout_selection.json": (
        "e3dfbfb736bb8847ad11d92b1573f88058e1c4319ac4a0180284db2171afac34",
        "AE32 selected-checkpoint decision",
    ),
}

# Behavioral closure is deliberately package-wide for frozen compositions.  A
# phase runner imports helpers transitively and sometimes by file path; hashing
# only the directly imported module would therefore be an incomplete binding.
BEHAVIORAL_SOURCE_PACKAGE_ROLES: Mapping[str, str] = {
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid":
        "A1a producer, selection, schema, quality adapter, store and CLI",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1":
        "continuous-q, UINT8/zstd, ranker, runtime loading and Phase-6 scoring composition",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1":
        "AE family models, UINT8/low-bit wires and Phase-10/11 runtime composition",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1":
        "p025 service policy and provenance",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1":
        "combined service records and provenance",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_person_instance_consolidation_v1":
        "person-instance consolidation used by service runtime",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1":
        "frozen perception model loader and numerical-recovery forward",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_v1":
        "immutable base model/data/inference implementation",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "route_b_v3_1_clean_base_v1": "frozen segmentation scorer package",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "route_b_v3_1_targeted_refinement_v1": "frozen detection matcher/scorer package",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_candidate_quality_v1":
        "candidate labeling transitively used by person-instance consolidation",
    "data_collection/route_b_publication_actor_volume_observability_model_comparison_v1":
        "frozen person AVO matcher/scorer package",
    "data_collection/route_b_publication_actor_volume_visibility_v1":
        "AVO visibility/eligibility implementation used by person scorer",
    "rl_agent/splitfusion_live_dispatch_v1":
        "exact SFD1 envelope/frame-context and live codec composition",
}

BEHAVIORAL_SOURCE_FILE_ROLES: Mapping[str, str] = {
    "rl_agent/splitfusion_hybrid_sac_v1/scene_descriptors.py":
        "frozen SI/P40 descriptor implementation",
    "rl_agent/splitfusion_hybrid_sac_v1/protocol_v2_contract.py":
        "exact Q_seg/Q_loc/Q_perc derivation",
    "rl_agent/splitfusion_hybrid_sac_v1/state_reward_transition_contract.py":
        "RewardSpecV1 type and quality semantics",
    "rl_agent/splitfusion_hybrid_sac_v1/action_contract.py":
        "continuous mode/q execution identity",
    "rl_agent/splitfusion_hybrid_sac_v1/reward_ticket_controller.py":
        "reward outcome and terminal semantics",
    "rl_agent/splitfusion_hybrid_sac_v1/transaction_identity.py":
        "tensor/action/reward identity semantics",
    "phase2_map_sharing/transport.py":
        "deployed !IHH UDP chunking and reassembly semantics",
    **{
        relative: role
        for relative, (_digest, role) in RUNTIME_BEHAVIORAL_ARTIFACTS.items()
    },
}

# Non-vacuous minimum audited by tests.  Package closure adds many more files.
REQUIRED_BEHAVIORAL_SOURCE_ROLES: Mapping[str, str] = {
    "rl_agent/splitfusion_hybrid_sac_v1/offline_quality_grid/executor.py":
        "A1a producer, selection, schema, quality adapter, store and CLI",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/continuous_q.py":
        "continuous-q, UINT8/zstd, ranker, runtime loading and Phase-6 scoring composition",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/uint8_codec.py":
        "continuous-q, UINT8/zstd, ranker, runtime loading and Phase-6 scoring composition",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/phase6_validation.py":
        "continuous-q, UINT8/zstd, ranker, runtime loading and Phase-6 scoring composition",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_uint8_transport.py":
        "AE family models, UINT8/low-bit wires and Phase-10/11 runtime composition",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_transport.py":
        "AE family models, UINT8/low-bit wires and Phase-10/11 runtime composition",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "route_b_v3_1_clean_base_v1/score_contract_v1.py":
        "frozen segmentation scorer package",
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "route_b_v3_1_targeted_refinement_v1/audit_v1.py":
        "frozen detection matcher/scorer package",
    "data_collection/route_b_publication_actor_volume_observability_model_comparison_v1/"
    "run_comparison.py": "frozen person AVO matcher/scorer package",
    "data_collection/route_b_publication_actor_volume_visibility_v1/core.py":
        "AVO visibility/eligibility implementation used by person scorer",
    "rl_agent/splitfusion_live_dispatch_v1/envelope.py":
        "exact SFD1 envelope/frame-context and live codec composition",
    "rl_agent/splitfusion_live_dispatch_v1/frame_context.py":
        "exact SFD1 envelope/frame-context and live codec composition",
    "rl_agent/splitfusion_live_dispatch_v1/ue_runtime.py":
        "exact SFD1 envelope/frame-context and live codec composition",
    "phase2_map_sharing/transport.py":
        "deployed !IHH UDP chunking and reassembly semantics",
    **BEHAVIORAL_SOURCE_FILE_ROLES,
}

SAMPLING_SEED = "splitfusion_phase_a1a_route_b_05_06_selection_v1"
STRATIFICATION = {
    "camera_si": "episode-local empirical quartile; invalid is its own stratum",
    "radar_p40": "episode-local empirical quartile; invalid is its own stratum",
    "vehicle_density": "zero/episode-local positive tertile of manifest vehicle_pixels",
    "person_density": "zero/episode-local positive tertile of manifest person_pixels",
    "allocation": "proportional Hamilton largest remainder, then SHA-256 rank",
    "replacement": False,
}


class OfflineGridContractError(ValueError):
    """A frozen Phase-A1a identity or invariant was violated."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def keep_count(q_e4: int) -> int:
    if isinstance(q_e4, bool) or not isinstance(q_e4, int) or q_e4 not in Q_E4_GRID:
        raise OfflineGridContractError(f"q_e4 is not on the frozen grid: {q_e4!r}")
    drop = (q_e4 * SPATIAL_CELLS + 5_000) // 10_000
    return SPATIAL_CELLS - drop


def datagram_count(total_transmitted_bytes: int) -> int:
    if isinstance(total_transmitted_bytes, bool) or not isinstance(total_transmitted_bytes, int):
        raise OfflineGridContractError("total_transmitted_bytes must be an exact integer")
    if total_transmitted_bytes <= 0:
        raise OfflineGridContractError("total_transmitted_bytes must be positive")
    return math.ceil(total_transmitted_bytes / UDP_PAYLOAD_BYTES_PER_DATAGRAM)


def sfd1_accounting_identity(mode_id: int, q_e4: int) -> tuple[int, str]:
    """Return the v2 action field used only for exact envelope byte accounting.

    Registered anchors carry their real catalog action ID.  SFD1 v2 has no
    representation for an off-anchor continuous-q action, so such rows use an
    explicit uint32 sentinel and a non-deployable status.  The sentinel does
    not change envelope length and must never be interpreted as dispatchable.
    """

    if isinstance(mode_id, bool) or not isinstance(mode_id, int) or not 0 <= mode_id < MODE_COUNT:
        raise OfflineGridContractError(f"invalid mode_id {mode_id!r}")
    if isinstance(q_e4, bool) or not isinstance(q_e4, int) or q_e4 not in Q_E4_GRID:
        raise OfflineGridContractError(f"q_e4 is not on the frozen grid: {q_e4!r}")
    if q_e4 in ACTION_CATALOG_Q_E4:
        return (
            mode_id * len(ACTION_CATALOG_Q_E4) + ACTION_CATALOG_Q_E4.index(q_e4),
            SFD1_REGISTERED_ACTION_STATUS,
        )
    return SFD1_DYNAMIC_ACTION_SENTINEL, SFD1_OFF_ANCHOR_SIZE_STATUS


def sfd1_accounting_stream_id(episode_id: str, mode_id: int) -> str:
    if not isinstance(episode_id, str) or not episode_id:
        raise OfflineGridContractError("episode_id must be non-empty")
    if isinstance(mode_id, bool) or not isinstance(mode_id, int) or not 0 <= mode_id < MODE_COUNT:
        raise OfflineGridContractError(f"invalid mode_id {mode_id!r}")
    return f"offline-a1a/{mode_id:02d}/{episode_id}"


def mode_inventory() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (index, family, quantizer)
        for index, (family, quantizer) in enumerate(
            (family, quantizer)
            for family in FAMILIES
            for quantizer in QUANTIZERS
        )
    )


def contract_descriptor() -> dict[str, Any]:
    return {
        "schema": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "evidence_label": EVIDENCE_LABEL,
        "deployment_claim": DEPLOYMENT_CLAIM,
        "quality_calibration_default": PROVISIONAL_QUALITY_CALIBRATION,
        "registered_frozen_reward_spec_sha256": sorted(
            REGISTERED_FROZEN_REWARD_SPEC_SHA256
        ),
        "runtime_behavioral_artifact_sha256": {
            path: digest
            for path, (digest, _role) in sorted(RUNTIME_BEHAVIORAL_ARTIFACTS.items())
        },
        "episodes": [
            {
                "episode_id": item.episode_id,
                "grid_split": item.grid_split,
                "expected_rows": item.expected_rows,
                "selected_rows": item.selected_rows,
                "manifest_sha256": item.manifest_sha256,
            }
            for item in EPISODES
        ],
        "families": list(FAMILIES),
        "quantizers": list(QUANTIZERS),
        "q_e4": list(Q_E4_GRID),
        "catalog_q_e4": list(ACTION_CATALOG_Q_E4),
        "selected_frames": TOTAL_SELECTED_FRAMES,
        "expected_grid_rows": EXPECTED_GRID_ROWS,
        "spatial_cells": SPATIAL_CELLS,
        "sampling_seed": SAMPLING_SEED,
        "stratification": dict(STRATIFICATION),
        "action_catalog_sha256": ACTION_CATALOG_SHA256,
        "model_dataset_manifest_sha256": MODEL_DATASET_MANIFEST_SHA256,
        "behavioral_source_package_roles": dict(BEHAVIORAL_SOURCE_PACKAGE_ROLES),
        "behavioral_source_file_roles": dict(BEHAVIORAL_SOURCE_FILE_ROLES),
        "checkpoints": {
            name: {"path": path, "sha256": digest}
            for name, (path, digest) in CHECKPOINTS.items()
        },
    }


CONTRACT_SHA256 = canonical_sha256(contract_descriptor())
