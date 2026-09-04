#!/usr/bin/env python3
"""Build the locked 72-profile SplitFusion action catalog from frozen JSON.

This builder is intentionally standard-library-only.  It hashes and reads
completed evidence; it never imports torch, initializes CUDA, loads a model or
checkpoint, or opens dataset/prediction artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "splitfusion_72_action_catalog_v1"
BASE_COMMIT = "ba681dbd33514d80d96784ac4cd0d67f3151e800"
FAMILIES = ("noAE", "AE128", "AE64", "AE32")
QUANTIZERS = ("UINT8", "UINT6", "UINT4")
Q_ANCHORS = ((0.0, 0), (0.30, 3000), (0.50, 5000), (0.70, 7000), (0.90, 9000), (0.98, 9800))
CELLS = 21_504
DENSE_FP32_Q0_BYTES = 22_020_140
TERMINAL = "SPLITFUSION_72_PROFILE_ACTION_CATALOG_LOCKED"

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]


def _binding(path: str, sha256: str) -> dict[str, str]:
    return {"path": path, "sha256": sha256}


INPUTS = {
    "uint8_noae_report": _binding(
        "experiments/splitfusion_fcos_hybrid_q_v1/20260902_223610_phase8b_uint8_validation/phase8b_uint8_validation.json",
        "a2779f5fb0a585b1c317dc755b5ab577fa7c34963ab7945cb704e0d4146bb029",
    ),
    "uint8_noae_terminal": _binding(
        "experiments/splitfusion_fcos_hybrid_q_v1/20260902_223610_phase8b_uint8_validation/HYBRID_Q_UINT8_VALIDATION_COMPLETE",
        "e4f534996dd5ae3e43ef037c34133cbfd5601e4a6407f1df6177b0d051c8b254",
    ),
    "uint8_ae128_report": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase9d_ae128_uint8_validation/phase9d_ae128_uint8_validation.json",
        "89cc7c706fc3383106a5680d3d54d5fb514dcd5d808c13d9eaf1a2c380785963",
    ),
    "uint8_ae128_terminal": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase9d_ae128_uint8_validation/SPLITFUSION_AE128_UINT8_VALIDATION_COMPLETE",
        "c375e0d87a2ac5aa9a00b66b6684a29a03b2188f826cc23781318f22db25c588",
    ),
    "uint8_ae64_report": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10b_ae64_uint8_validation/phase10b_ae64_uint8_validation.json",
        "eead1786a5d12294b9d61d9271431049aac28540a20f2c4608db33ab66de3aad",
    ),
    "uint8_ae64_terminal": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10b_ae64_uint8_validation/SPLITFUSION_AE64_PHASE10B_UINT8_VALIDATION_COMPLETE",
        "cd84b9894b29e7533b9fe665790d55e5905118e4e77121a06ce5e7e521945a27",
    ),
    "uint8_ae32_report": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10b_ae32_uint8_validation/phase10b_ae32_uint8_validation.json",
        "db4b944eb5992a82e9fe6b0befd2e3bcf583629a6f27ad2ad6cb4075f03a90ec",
    ),
    "uint8_ae32_terminal": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10b_ae32_uint8_validation/SPLITFUSION_AE32_PHASE10B_UINT8_VALIDATION_COMPLETE",
        "3bd2b240a368019dce7ff0607ed61eec7097e614a2e627a9369d2ce9aea5c553",
    ),
    "phase11b_report": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase11b_lowbit_gpu_qualification/phase11b_lowbit_gpu_qualification.json",
        "379aa07148e3e47384cfbebbe0ede5990c07f11b8a4bdef056d6a533cee5fc01",
    ),
    "phase11b_terminal": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase11b_lowbit_gpu_qualification/SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED",
        "83f41560a3327c4207834f5725e5e313ceb6b3e0f9e22ea1f8c37b6dcf0b56e2",
    ),
    "phase11c_report": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase11c_zstd_level_sweep/phase11c_zstd_level_sweep.json",
        "4bcd2eddff502cc55d799bfdf5af920ccc1378ae87bd2c0d6017d3d2586c7b2d",
    ),
    "phase11c_terminal": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase11c_zstd_level_sweep/HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP_COMPLETE",
        "198585a7384382b6c858a4f17595b132b008e7ef8f9a58d2c595032acdb1b5c2",
    ),
    "layout_report": _binding(
        "experiments/splitfusion_fcos_transport_layout_feasibility_v1/20260903_layout_feasibility_cuda_retry1/transport_layout_feasibility.json",
        "5535bc080a12d3aa3eb17ef99e20a39361e4e908b971c7f67a9ced4ad10c6071",
    ),
    "layout_terminal": _binding(
        "experiments/splitfusion_fcos_transport_layout_feasibility_v1/20260903_layout_feasibility_cuda_retry1/TRANSPORT_LAYOUT_FEASIBILITY_COMPLETE",
        "aed941acd79994c52c1e94516f312a84c1202a5c720eac62093ef685441d67b5",
    ),
    "phase11d_report": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase11d_lowbit_validation/phase11d_lowbit_validation.json",
        "2680e6dc21469e6fdcabe1ce79e9d2d333d9a137c639cd033338b9b1c01c2862",
    ),
    "phase11d_manifest": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase11d_lowbit_validation/run_manifest.json",
        "117844888f8eac2e0133ebc37ba61a29dff5abf593dfc3a9d54b2bed96125e23",
    ),
    "phase11d_terminal": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase11d_lowbit_validation/SPLITFUSION_LOWBIT_PHASE11D_VALIDATION_COMPLETE",
        "c79d41ca1fef1618a6f7e556a3c01594c11ef539d050efde1716ed174cf53983",
    ),
    "perception_checkpoint": _binding(
        "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_numerical_recovery_v1/20260830_recovered_epoch10_gate_v1/checkpoints/epoch_026.pt",
        "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f",
    ),
    "ranker_checkpoint": _binding(
        "experiments/splitfusion_fcos_hybrid_q_v1/20260901_185725_phase5_ranker_training/checkpoints/ranker_epoch_04.pt",
        "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5",
    ),
    "ae128_checkpoint": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260902_220623_phase9c_ae128_training/checkpoints/ae128_epoch_08.pt",
        "0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72",
    ),
    "ae128_selection": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260902_220623_phase9c_ae128_training/holdout_selection/holdout_selection.json",
        "69e49deac302fc46c1eec56036e3ab3d769b3aac10b76541cfb4abb80f878194",
    ),
    "ae64_checkpoint": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae64_training/checkpoints/ae64_epoch_12.pt",
        "dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0",
    ),
    "ae64_selection": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae64_training/holdout_selection_ae64/ae64_holdout_selection.json",
        "0d2fe444574d3fdc9aee287448084bf2cfc1efa2d0ec6944ac07355d9ff7c87e",
    ),
    "ae32_checkpoint": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae32_training/checkpoints/ae32_epoch_08.pt",
        "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271",
    ),
    "ae32_selection": _binding(
        "experiments/splitfusion_fcos_ae_v1/20260903_phase10_ae32_training/holdout_selection_ae32/ae32_holdout_selection.json",
        "e3dfbfb736bb8847ad11d92b1573f88058e1c4319ac4a0180284db2171afac34",
    ),
    "p025_forward_lock": _binding(
        "pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1/PERCEPTION_FORWARD_LOCK_P025_V1.json",
        "86d6f13ae9168b33b697df5b785c5f7c320afc52cfdcded5b632d94a6d943fe1",
    ),
    "hybrid_q_locked_config": _binding(
        "pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/locked_config.json",
        "b2b0d8427bd867f46058ebba49ac6a183eb89413b4d69326fef93b150ebfcde6",
    ),
    "noae_uint8_codec": _binding(
        "pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/uint8_codec.py",
        "b6f07723860821e93c0dc2aeec456073617942b5ed8446f4555fe87b2c7ca803",
    ),
    "ae_uint8_codec": _binding(
        "pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_uint8_transport.py",
        "4162b162d554f764332d469b6ca7b5038298e77a563a9de73c1622cd99531423",
    ),
    "lowbit_codec": _binding(
        "pole_lraspp_multimodal_fusion/object_head_pilot_v1/splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_transport.py",
        "c708389982b10968978002d2d8423984d857229021149d51ec0d135619d69f12",
    ),
    "network_profile_v1": _binding(
        "rl_agent/configs/network_profile_design_v1.json",
        "97d3bf835fcb611f00704f9d69c0d0958d94bd703059ed814099f064ceffb2f5",
    ),
    "network_profile_v2": _binding(
        "rl_agent/configs/network_profile_design_v2.json",
        "056247e5731c1ae9ac281432034e1b79d1e6da24ab4a2d579a7fe2d85917e483",
    ),
}

UINT8_SOURCES = {
    "noAE": ("uint8_noae_report", "uint8_noae_terminal", ("ratio_vs_compressed_uint8_q0", "ratio_vs_framed_fp32_q0")),
    "AE128": ("uint8_ae128_report", "uint8_ae128_terminal", ("payload_ratios",)),
    "AE64": ("uint8_ae64_report", "uint8_ae64_terminal", ("payload_ratios", "secondary_classification")),
    "AE32": ("uint8_ae32_report", "uint8_ae32_terminal", ("payload_ratios", "secondary_classification")),
}

FAMILY_INFO = {
    "noAE": {"family_id": 0, "latent_width": None, "transported_channels": 256, "checkpoint_input": None, "selection_input": None},
    "AE128": {"family_id": 1, "latent_width": 128, "transported_channels": 128, "checkpoint_input": "ae128_checkpoint", "selection_input": "ae128_selection"},
    "AE64": {"family_id": 2, "latent_width": 64, "transported_channels": 64, "checkpoint_input": "ae64_checkpoint", "selection_input": "ae64_selection"},
    "AE32": {"family_id": 3, "latent_width": 32, "transported_channels": 32, "checkpoint_input": "ae32_checkpoint", "selection_input": "ae32_selection"},
}

LOCALIZATION_REQUIREMENTS = {
    "vehicle_precision": ("higher", 0.80),
    "vehicle_recall": ("higher", 0.85),
    "vehicle_xy_mae_m": ("lower", 1.00),
    "person_avo_precision": ("higher", 0.70),
    "person_avo_recall": ("higher", 0.70),
    "person_avo_f1": ("higher", 0.70),
    "person_avo_xy_mae_m": ("lower", 1.20),
    "person_avo_recall_0_30m": ("higher", 0.70),
}
SEGMENTATION_REQUIREMENTS = {
    "vehicle_iou": 0.85,
    "person_box_mask_iou": 0.50,
    "foreground_miou": 0.675,
}


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_binding(binding: dict[str, str]) -> Path:
    path = REPO_ROOT / binding["path"]
    _expect(path.is_file(), f"required input is missing: {binding['path']}")
    actual = _sha256(path)
    _expect(actual == binding["sha256"], f"SHA-256 mismatch for {binding['path']}: {actual}")
    return path


def _read_json(binding: dict[str, str]) -> dict[str, Any]:
    return json.loads(_verify_binding(binding).read_text(encoding="utf-8"))


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dynamic_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    _expect(path.is_file(), f"durable record is missing: {path.relative_to(REPO_ROOT)}")
    actual = _sha256(path)
    _expect(actual == expected_sha256, f"durable-record SHA-256 mismatch: {path.relative_to(REPO_ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _keep_drop(q_e4: int) -> tuple[int, int]:
    dropped = (q_e4 * CELLS + 5000) // 10_000
    return CELLS - dropped, dropped


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _passed(value: float, direction: str, target: float) -> bool:
    return value >= target if direction == "higher" else value <= target


def _reconcile_uint8() -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    detail: dict[str, Any] = {}
    expected_schemas = {
        "noAE": "splitfusion_fcos_hybrid_q_phase8b_uint8_validation_v1",
        "AE128": "splitfusion_fcos_ae128_phase9d_uint8_validation_v1",
        "AE64": "splitfusion_fcos_ae64_phase10b_uint8_validation_v1",
        "AE32": "splitfusion_fcos_ae32_phase10b_uint8_validation_v1",
    }
    for family in FAMILIES:
        report_key, terminal_key, enrichments = UINT8_SOURCES[family]
        report_path = _verify_binding(INPUTS[report_key])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _verify_binding(INPUTS[terminal_key])
        _expect(report.get("schema") == expected_schemas[family], f"unexpected {family} UINT8 schema")
        _expect(report.get("terminal") in Path(INPUTS[terminal_key]["path"]).name, f"{family} terminal binding disagrees")
        curve = report.get("curve")
        _expect(isinstance(curve, list) and len(curve) == 6, f"{family} UINT8 curve must contain six rows")
        by_q = {row.get("q_e4"): row for row in curve}
        _expect(set(by_q) == {item[1] for item in Q_ANCHORS}, f"{family} UINT8 q coverage mismatch")
        setting_refs = report.get("settings")
        _expect(isinstance(setting_refs, dict) and len(setting_refs) == 6, f"{family} settings inventory mismatch")
        durable_hashes: dict[str, str] = {}
        cleanup_count = 0
        for q, q_e4 in Q_ANCHORS:
            row = by_q[q_e4]
            _expect(row["q"] == q, f"{family} q/q_e4 disagreement at {q_e4}")
            ref = setting_refs.get(f"q{q_e4:04d}")
            _expect(isinstance(ref, dict), f"{family} missing setting q{q_e4:04d}")
            setting_path = report_path.parent / ref["path"]
            durable = _dynamic_json(setting_path, ref["sha256"])
            reconciled = dict(row)
            for key in enrichments:
                reconciled.pop(key, None)
            _expect(durable == reconciled, f"{family} q{q_e4:04d} summary/durable record mismatch")
            durable_hashes[f"q{q_e4:04d}"] = ref["sha256"]
            if family == "noAE":
                _expect(row.get("prediction_artifacts_removed_after_scoring") is True, f"{family} q{q_e4:04d} cleanup not confirmed")
            else:
                cleanup_path = report_path.parent / "cleanup" / f"q{q_e4:04d}.json"
                _expect(cleanup_path.is_file(), f"{family} q{q_e4:04d} cleanup record missing")
                cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
                _expect(cleanup.get("setting_sha256") == ref["sha256"], f"{family} cleanup setting hash mismatch")
                _expect(cleanup.get("prediction_artifacts_removed_after_scoring") is True, f"{family} cleanup removal mismatch")
                _expect(cleanup.get("durability_order", [])[:2] == [
                    "atomically write settings/<q>.json",
                    "remove working_predictions/<q>",
                ], f"{family} cleanup order mismatch")
                cleanup_count += 1
            keep, drop = _keep_drop(q_e4)
            _expect((row["retained_cells"], row["dropped_cells"]) == (keep, drop), f"{family} keep/drop mismatch at q{q_e4:04d}")
        rows[family] = by_q
        detail[family] = {
            "report": INPUTS[report_key],
            "terminal": INPUTS[terminal_key],
            "summary_rows": 6,
            "durable_records_reconciled": 6,
            "cleanup_records_reconciled": cleanup_count,
            "durable_setting_sha256": durable_hashes,
        }
    return rows, detail


def _reconcile_phase11d(
    uint8_rows: dict[str, dict[int, dict[str, Any]]]
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    report_path = _verify_binding(INPUTS["phase11d_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = _read_json(INPUTS["phase11d_manifest"])
    _verify_binding(INPUTS["phase11d_terminal"])
    _expect(report.get("schema") == "splitfusion_fcos_phase11d_lowbit_validation_v1", "unexpected Phase-11D schema")
    _expect(report.get("terminal") == "SPLITFUSION_LOWBIT_PHASE11D_VALIDATION_COMPLETE", "unexpected Phase-11D terminal")
    _expect(report.get("manifest", {}).get("sha256") == INPUTS["phase11d_manifest"]["sha256"], "Phase-11D manifest binding mismatch")
    _expect(manifest.get("expected_setting_count") == 48, "Phase-11D manifest setting count mismatch")
    curve = report.get("curve")
    _expect(isinstance(curve, list) and len(curve) == 48, "Phase-11D curve must contain 48 rows")
    by_key = {row.get("setting_key"): row for row in curve}
    expected_keys = []
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    setting_hashes: dict[str, str] = {}
    for family in FAMILIES:
        for quantizer in ("UINT6", "UINT4"):
            for q, q_e4 in Q_ANCHORS:
                key = f"{family.lower()}_{quantizer.lower()}_q{q_e4:04d}"
                expected_keys.append(key)
                row = by_key.get(key)
                _expect(isinstance(row, dict), f"Phase-11D setting missing: {key}")
                setting_path = report_path.parent / "settings" / f"{key}.json"
                _expect(setting_path.is_file(), f"Phase-11D durable setting missing: {key}")
                setting_sha = _sha256(setting_path)
                durable = json.loads(setting_path.read_text(encoding="utf-8"))
                _expect(durable == row, f"Phase-11D summary/durable setting mismatch: {key}")
                cleanup_path = report_path.parent / "cleanup" / f"{key}.json"
                _expect(cleanup_path.is_file(), f"Phase-11D cleanup missing: {key}")
                cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
                _expect(cleanup.get("setting_sha256") == setting_sha, f"Phase-11D cleanup hash mismatch: {key}")
                _expect(cleanup.get("prediction_artifacts_removed_after_durable_record") is True, f"Phase-11D cleanup order mismatch: {key}")
                _expect(row["family"] == family and row["quantizer"] == quantizer, f"Phase-11D identity mismatch: {key}")
                _expect(row["q"] == q and row["q_e4"] == q_e4, f"Phase-11D q mismatch: {key}")
                keep, drop = _keep_drop(q_e4)
                _expect((row["retained_cells"], row["dropped_cells"]) == (keep, drop), f"Phase-11D keep/drop mismatch: {key}")

                reference = row["same_family_same_q_uint8_reference"]
                uint8 = uint8_rows[family][q_e4]
                report_key = UINT8_SOURCES[family][0]
                _expect(reference["source_path"] == INPUTS[report_key]["path"], f"Phase-11D UINT8 source path mismatch: {key}")
                _expect(reference["source_sha256"] == INPUTS[report_key]["sha256"], f"Phase-11D UINT8 source hash mismatch: {key}")
                _expect(reference["family"] == family and reference["q_e4"] == q_e4, f"Phase-11D UINT8 identity mismatch: {key}")
                _expect(reference["metrics"] == uint8["metrics"], f"Phase-11D UINT8 metrics mismatch: {key}")
                _expect(reference["canonical_person_metrics"] == uint8["canonical_person_metrics"], f"Phase-11D UINT8 canonical metrics mismatch: {key}")
                _expect(reference["absolute_service_gates"] == uint8["absolute_service_gates"], f"Phase-11D UINT8 absolute gates mismatch: {key}")
                _expect(reference["retained_cells"] == uint8["retained_cells"], f"Phase-11D UINT8 keep count mismatch: {key}")
                for metric, gate in row["same_family_same_q_uint8_preservation"]["gates"].items():
                    _expect(gate["noae_same_q"] == uint8["metrics"][metric], f"Phase-11D preservation baseline mismatch: {key}/{metric}")
                rows[(family, quantizer, q_e4)] = row
                setting_hashes[key] = setting_sha
    _expect(manifest.get("expected_settings") == expected_keys, "Phase-11D manifest order/content mismatch")
    _expect(set(by_key) == set(expected_keys), "Phase-11D curve contains unexpected settings")
    integrity = report.get("integrity", {})
    _expect(integrity.get("settings_completed") == 48 and integrity.get("required_settings") == 48, "Phase-11D completion count mismatch")
    _expect(integrity.get("zstd_decompressions") == 160_560, "Phase-11D decompression count mismatch")
    _expect(integrity.get("all_frozen_state_equal") is True, "Phase-11D frozen-state check failed")
    return rows, {
        "report": INPUTS["phase11d_report"],
        "terminal": INPUTS["phase11d_terminal"],
        "manifest": INPUTS["phase11d_manifest"],
        "summary_rows": 48,
        "durable_records_reconciled": 48,
        "cleanup_records_reconciled": 48,
        "frame_settings": 160_560,
        "durable_setting_sha256": setting_hashes,
    }


def _verify_campaign_bindings() -> dict[str, Any]:
    p025 = _read_json(INPUTS["p025_forward_lock"])
    _expect(p025.get("schema") == "splitfusion_fcos_perception_forward_lock_p025_v1", "unexpected p025 lock schema")
    phase11b = _read_json(INPUTS["phase11b_report"])
    _verify_binding(INPUTS["phase11b_terminal"])
    _expect(phase11b.get("terminal") == "SPLITFUSION_LOWBIT_PHASE11B_GPU_QUALIFIED", "Phase-11B terminal mismatch")
    _expect(phase11b.get("integrity", {}).get("settings_completed") == 16, "Phase-11B incomplete")
    _expect(phase11b.get("integrity", {}).get("all_reconstructed_c2_finite_fp32_cuda0") is True, "Phase-11B device qualification failed")
    phase11c = _read_json(INPUTS["phase11c_report"])
    _verify_binding(INPUTS["phase11c_terminal"])
    _expect(phase11c.get("terminal") == "HYBRID_Q_PHASE11C_ZSTD_LEVEL_SWEEP_COMPLETE", "Phase-11C terminal mismatch")
    _expect(phase11c.get("integrity", {}).get("exact_round_trips") == 27_648, "Phase-11C integrity mismatch")
    layout = _read_json(INPUTS["layout_report"])
    _verify_binding(INPUTS["layout_terminal"])
    _expect(layout.get("terminal") == "TRANSPORT_LAYOUT_FEASIBILITY_COMPLETE", "layout terminal mismatch")
    _expect(layout.get("classification", {}).get("classification") == "NOT_USEFUL", "layout decision changed")
    _expect(layout.get("classification", {}).get("production_layout_change_recommended") is False, "layout study recommends a production change")
    _expect(layout.get("integrity", {}).get("production_codec_modified") is False, "layout study modified a production codec")
    return {
        "p025_forward_lock": INPUTS["p025_forward_lock"],
        "phase11b": {"report": INPUTS["phase11b_report"], "terminal": INPUTS["phase11b_terminal"]},
        "phase11c": {
            "report": INPUTS["phase11c_report"],
            "terminal": INPUTS["phase11c_terminal"],
            "interpretation": "The bounded 1/3/5 study selected no new level; Phase-11D retained mandatory level 1.",
        },
        "transport_layout": {
            "report": INPUTS["layout_report"],
            "terminal": INPUTS["layout_terminal"],
            "classification": "NOT_USEFUL",
            "production_layout_retained": "CURRENT_CELL_MAJOR",
        },
    }


def _network_metadata() -> dict[str, Any]:
    candidates = []
    for key in ("network_profile_v1", "network_profile_v2"):
        data = _read_json(INPUTS[key])
        profiles = data.get("profiles")
        _expect(isinstance(profiles, list) and len(profiles) == 4, f"{key} must contain four profiles")
        candidates.append({
            **INPUTS[key],
            "schema": data.get("schema"),
            "profile_ids": [profile.get("profile_id") for profile in profiles],
            "route": data.get("route"),
            "target_snr": data.get("target_snr"),
        })
    _expect(candidates[0]["profile_ids"] == candidates[1]["profile_ids"], "network candidate profile IDs differ")
    _expect(candidates[0]["schema"] != candidates[1]["schema"], "network candidates unexpectedly unambiguous")
    _expect(candidates[0]["route"] != candidates[1]["route"] or candidates[0]["target_snr"] != candidates[1]["target_snr"], "network candidates do not materially conflict")
    return {
        "binding_status": "UNRESOLVED_MULTIPLE_CONFLICTING_DEFINITIONS",
        "bound_definition": None,
        "candidates": candidates,
        "conflict": "v1 and v2 share four profile IDs but differ in schema and route/target-SNR operating definitions; neither is selected silently.",
        "perception_catalog_impact": "none; all 72 SplitFusion profiles remain registered",
    }


def _wire(family: str, quantizer: str) -> dict[str, Any]:
    if quantizer == "UINT8" and family == "noAE":
        return {"layout": "CURRENT_CELL_MAJOR", "magic_ascii": "HQ8\\0", "version": 1, "codec_id": 1, "codec_source": INPUTS["noae_uint8_codec"]}
    if quantizer == "UINT8":
        return {"layout": "CURRENT_CELL_MAJOR", "magic_ascii": "AE8\\0", "version": 1, "codec_id": 2, "codec_source": INPUTS["ae_uint8_codec"]}
    return {"layout": "CURRENT_CELL_MAJOR", "magic_ascii": "HQLB", "version": 1, "codec_id": 3, "codec_source": INPUTS["lowbit_codec"]}


def _payload(row: dict[str, Any], family: str, quantizer: str) -> dict[str, float | int]:
    if quantizer == "UINT8" and family == "noAE":
        return {
            "pre_zstd_analytical_bytes": row["analytical_uint8_sparse_bytes"],
            "zstd_median_bytes": row["compressed_zstd_bytes"]["median"],
            "zstd_p95_bytes": row["compressed_zstd_bytes"]["p95"],
            "ratio_vs_dense_fp32_q0": row["ratio_vs_framed_fp32_q0"],
            "ratio_vs_same_family_same_q_uint8": 1.0,
        }
    if quantizer == "UINT8":
        return {
            "pre_zstd_analytical_bytes": row["payload"]["analytical_pre_zstd_bytes"],
            "zstd_median_bytes": row["payload"]["zstd_bytes"]["median"],
            "zstd_p95_bytes": row["payload"]["zstd_bytes"]["p95"],
            "ratio_vs_dense_fp32_q0": row["payload_ratios"]["vs_framed_fp32_noae_q0"]["zstd_median"],
            "ratio_vs_same_family_same_q_uint8": 1.0,
        }
    return {
        "pre_zstd_analytical_bytes": row["payload"]["analytical_pre_zstd_bytes"],
        "zstd_median_bytes": row["payload"]["zstd_bytes"]["median"],
        "zstd_p95_bytes": row["payload"]["zstd_bytes"]["p95"],
        "ratio_vs_dense_fp32_q0": row["payload_ratios"]["vs_framed_fp32_noae_q0"],
        "ratio_vs_same_family_same_q_uint8": row["payload_ratios"]["vs_same_family_same_q_uint8_zstd"],
    }


def _perception(row: dict[str, Any], family: str, quantizer: str) -> dict[str, float | None]:
    metrics = row["metrics"]
    canonical = row["canonical_person_metrics"]
    if quantizer in ("UINT6", "UINT4"):
        range_detail = row["classification"]["person_range_stratified"]
    elif family in ("AE64", "AE32"):
        range_detail = row["person_range_stratified"]
    else:
        range_detail = None
    return {
        "vehicle_precision": metrics["vehicle_precision"],
        "vehicle_recall": metrics["vehicle_recall"],
        "vehicle_f1": metrics["vehicle_f1"],
        "vehicle_xy_mae_m": metrics["vehicle_xy_mae_m"],
        "canonical_person_precision": canonical["person_precision"],
        "canonical_person_recall": canonical["person_recall"],
        "canonical_person_f1": canonical["person_f1"],
        "canonical_person_xy_mae_m": canonical["person_xy_mae_m"],
        "person_avo_precision": metrics["person_avo_precision"],
        "person_avo_recall": metrics["person_avo_recall"],
        "person_avo_f1": metrics["person_avo_f1"],
        "person_avo_xy_mae_m": metrics["person_avo_xy_mae_m"],
        "person_avo_recall_0_30m": None if range_detail is None else range_detail["person_avo_recall_0_30m"],
        "person_avo_recall_30_40m_diagnostic": None if range_detail is None else range_detail["bins"]["30_40m"]["recall"],
        "vehicle_iou": metrics["vehicle_iou"],
        "person_box_mask_iou": metrics["person_box_mask_iou"],
        "foreground_miou": metrics["foreground_miou"],
    }


def _legacy_capabilities(row: dict[str, Any], perception: dict[str, float | None]) -> dict[str, Any]:
    passed = 0
    evaluated = 0
    failed = []
    not_evaluable = []
    for metric, (direction, target) in LOCALIZATION_REQUIREMENTS.items():
        value = perception[metric]
        if value is None:
            not_evaluable.append(metric)
            continue
        evaluated += 1
        if _passed(float(value), direction, target):
            passed += 1
        else:
            failed.append(metric)
    segmentation_count = sum(perception[name] >= target for name, target in SEGMENTATION_REQUIREMENTS.items())
    return {
        "localization_requirements_passed": evaluated == 8 and passed == 8,
        "localization_requirement_count": passed,
        "localization_requirements_total": 8,
        "localization_requirements_evaluated": evaluated,
        "localization_not_evaluable": not_evaluable,
        "localization_failed": failed,
        "segmentation_installable": segmentation_count == 3,
        "segmentation_requirement_count": segmentation_count,
    }


def _capabilities(row: dict[str, Any], family: str, quantizer: str, perception: dict[str, float | None]) -> dict[str, Any]:
    if quantizer == "UINT8" and family == "noAE":
        preservation = row["quantization_preservation_vs_same_q_fp32"]
        source_tier = row["emergency_mode_status"]["designation"]
        stress = row["emergency_mode_status"]["is_emergency_anchor"]
        legacy = _legacy_capabilities(row, perception)
    elif quantizer == "UINT8" and family == "AE128":
        preservation = row["same_q_preservation_vs_noae_uint8_zstd"]
        source_tier = row["profile_status"]["designation"]
        stress = row["profile_status"]["is_stress_anchor"]
        legacy = _legacy_capabilities(row, perception)
    else:
        preservation = row["same_q_preservation_vs_noae_uint8_zstd"] if quantizer == "UINT8" else row["same_family_same_q_uint8_preservation"]
        classification = row["secondary_classification"] if quantizer == "UINT8" else row["classification"]
        localization = classification["localization_priority"]
        segmentation = classification["segmentation"]
        source_tier = classification["tier"]
        stress = classification["is_registered_stress_profile"] if quantizer == "UINT8" else classification["stress_anchor_forced_emergency_only"]
        legacy = {
            "localization_requirements_passed": localization["all_passed"],
            "localization_requirement_count": localization["passed_count"],
            "localization_requirements_total": localization["registered_total"],
            "localization_requirements_evaluated": localization["registered_total"] - len(localization.get("not_evaluable", [])),
            "localization_not_evaluable": localization.get("not_evaluable", []),
            "localization_failed": localization.get("failed", []),
            "segmentation_installable": segmentation["segmentation_installable"],
            "segmentation_requirement_count": segmentation["passed_count"],
        }
        for name, requirement in localization["requirements"].items():
            _expect(requirement["value"] == perception[name], f"localization source metric mismatch: {family}/{quantizer}/{row['q_e4']}/{name}")
        _expect(segmentation["segmentation_installable"] == all(perception[name] >= target for name, target in SEGMENTATION_REQUIREMENTS.items()), f"segmentation gate mismatch: {family}/{quantizer}/{row['q_e4']}")
    relative_count = sum(gate["passed"] for gate in preservation["gates"].values())
    _expect(len(preservation["gates"]) == 12, f"preservation gate count mismatch: {family}/{quantizer}/{row['q_e4']}")
    absolute = row["absolute_service_gates"]
    capabilities = {
        "relative_preservation_passed": preservation["all_passed"],
        "relative_preservation_count": relative_count,
        "relative_preservation_total": 12,
        "absolute_service_ready": absolute["all_pass"],
        "absolute_service_count": absolute["pass_count"],
        "absolute_service_total": 9,
        **legacy,
        "segmentation_behavior": "install_current_segmentation" if legacy["segmentation_installable"] else "retain_previous_segmentation_layer_with_original_timestamp",
        "source_contract_tier": source_tier,
        "stress_or_emergency_anchor": stress,
        "transport_valid": True,
        "agent_action_enabled": True,
        "state_infeasible_assigned_offline": False,
    }
    _expect(relative_count == sum(gate["passed"] for gate in preservation["gates"].values()), "relative capability mismatch")
    _expect(absolute["pass_count"] == sum(target["passed"] for target in absolute["targets"].values()), "absolute capability mismatch")
    return capabilities


def _source_and_durable(
    family: str, quantizer: str, q_e4: int, row: dict[str, Any], phase11d_path: Path
) -> tuple[dict[str, str], dict[str, str]]:
    if quantizer == "UINT8":
        report_key = UINT8_SOURCES[family][0]
        report_path = REPO_ROOT / INPUTS[report_key]["path"]
        ref = json.loads(report_path.read_text(encoding="utf-8"))["settings"][f"q{q_e4:04d}"]
        return INPUTS[report_key], {
            "path": str((report_path.parent / ref["path"]).relative_to(REPO_ROOT)),
            "sha256": ref["sha256"],
        }
    key = row["setting_key"]
    setting_path = phase11d_path.parent / "settings" / f"{key}.json"
    return INPUTS["phase11d_report"], {
        "path": str(setting_path.relative_to(REPO_ROOT)),
        "sha256": _sha256(setting_path),
    }


def _build_profiles(
    uint8_rows: dict[str, dict[int, dict[str, Any]]],
    lowbit_rows: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    profiles = []
    phase11d_path = REPO_ROOT / INPUTS["phase11d_report"]["path"]
    action_id = 0
    for family in FAMILIES:
        info = FAMILY_INFO[family]
        checkpoint = INPUTS[info["checkpoint_input"]] if info["checkpoint_input"] else None
        selection = INPUTS[info["selection_input"]] if info["selection_input"] else None
        routing_tag = 0 if checkpoint is None else int(checkpoint["sha256"][:8], 16)
        decoder_identity = "NO_AE_ZERO_SCATTER_TO_C2" if checkpoint is None else f"{family}_DECODER_{checkpoint['sha256']}"
        for quantizer in QUANTIZERS:
            bit_width = int(quantizer.removeprefix("UINT"))
            for q, q_e4 in Q_ANCHORS:
                row = uint8_rows[family][q_e4] if quantizer == "UINT8" else lowbit_rows[(family, quantizer, q_e4)]
                keep, drop = _keep_drop(q_e4)
                payload = _payload(row, family, quantizer)
                perception = _perception(row, family, quantizer)
                capabilities = _capabilities(row, family, quantizer, perception)
                source, durable = _source_and_durable(family, quantizer, q_e4, row, phase11d_path)
                if quantizer != "UINT8":
                    _expect(row["payload"]["routing_tag"] == routing_tag, f"routing tag mismatch: {row['setting_key']}")
                    _expect(row["payload"]["transported_channels"] == info["transported_channels"], f"transported channel mismatch: {row['setting_key']}")
                    _expect(row["payload"]["zstd_level"] == 1, f"zstd level mismatch: {row['setting_key']}")
                    _expect(row["integrity"]["retained_lowbit_values_within_registered_error_bound"] is True, f"low-bit error bound failed: {row['setting_key']}")
                    _expect(row["integrity"]["dropped_cells_reconstruct_to_exact_zero"] is True, f"zero reconstruction failed: {row['setting_key']}")
                for name, value in perception.items():
                    _expect(value is None or _finite(value), f"non-finite metric: {family}/{quantizer}/q{q_e4:04d}/{name}")
                for name, value in payload.items():
                    _expect(_finite(value) and float(value) > 0.0, f"invalid payload field: {family}/{quantizer}/q{q_e4:04d}/{name}")
                expected_stress = q_e4 in (9000, 9800)
                _expect(capabilities["stress_or_emergency_anchor"] is expected_stress, f"stress flag mismatch: {family}/{quantizer}/q{q_e4:04d}")
                _expect(capabilities["source_contract_tier"] != "INVALID", f"unexpected INVALID profile: {family}/{quantizer}/q{q_e4:04d}")
                _expect(row["retained_cells"] == keep and row["dropped_cells"] == drop, f"row keep/drop mismatch: {family}/{quantizer}/q{q_e4:04d}")
                profile = {
                    "action_id": action_id,
                    "profile_id": f"split_{family.lower()}_{quantizer.lower()}_q{q_e4:04d}",
                    "execution_mode": "SPLIT",
                    "family": family,
                    "family_id": info["family_id"],
                    "latent_width": info["latent_width"],
                    "transported_channels": info["transported_channels"],
                    "quantizer": quantizer,
                    "bit_width": bit_width,
                    "q": q,
                    "q_e4": q_e4,
                    "keep_count": keep,
                    "drop_count": drop,
                    "zstd_level": 1,
                    "wire": _wire(family, quantizer),
                    "checkpoint_sha256": INPUTS["perception_checkpoint"]["sha256"] if checkpoint is None else checkpoint["sha256"],
                    "perception_checkpoint": INPUTS["perception_checkpoint"],
                    "ae_checkpoint": checkpoint,
                    "ae_selection": selection,
                    "routing_tag": routing_tag,
                    "decoder_identity": decoder_identity,
                    "ranker": {
                        "bypassed": q_e4 == 0,
                        "checkpoint": None if q_e4 == 0 else INPUTS["ranker_checkpoint"],
                        "identity": "BYPASS_Q0" if q_e4 == 0 else "STABLE_EPOCH4_RANKER",
                    },
                    "payload": payload,
                    "perception": perception,
                    "capabilities": capabilities,
                    "source_evidence": source,
                    "durable_setting_evidence": durable,
                }
                profiles.append(profile)
                action_id += 1
    return profiles


def _validate_profiles(profiles: list[dict[str, Any]]) -> None:
    _expect(len(profiles) == 72, "catalog must contain exactly 72 profiles")
    _expect([profile["action_id"] for profile in profiles] == list(range(72)), "action IDs are not contiguous 0-71")
    profile_ids = [profile["profile_id"] for profile in profiles]
    _expect(len(set(profile_ids)) == 72, "profile IDs are not unique")
    actual = {(p["family"], p["quantizer"], p["q_e4"]) for p in profiles}
    expected = {(f, z, q_e4) for f in FAMILIES for z in QUANTIZERS for _, q_e4 in Q_ANCHORS}
    _expect(actual == expected, "Cartesian-product coverage mismatch")
    for profile in profiles:
        keep, drop = _keep_drop(profile["q_e4"])
        _expect(profile["keep_count"] == keep and profile["drop_count"] == drop, f"catalog keep/drop mismatch: {profile['profile_id']}")
        _expect(profile["wire"]["layout"] == "CURRENT_CELL_MAJOR", f"wire layout mismatch: {profile['profile_id']}")
        _expect(profile["zstd_level"] == 1, f"zstd mismatch: {profile['profile_id']}")
        capabilities = profile["capabilities"]
        for field in (
            "relative_preservation_passed",
            "absolute_service_ready",
            "localization_requirements_passed",
            "segmentation_installable",
            "stress_or_emergency_anchor",
            "transport_valid",
            "agent_action_enabled",
        ):
            _expect(type(capabilities[field]) is bool, f"capability is not boolean: {profile['profile_id']}/{field}")
        _expect(capabilities["relative_preservation_total"] == 12, "relative total mismatch")
        _expect(capabilities["absolute_service_total"] == 9, "absolute total mismatch")
        _expect(capabilities["localization_requirements_total"] == 8, "localization total mismatch")
        _expect(capabilities["transport_valid"] and capabilities["agent_action_enabled"], f"valid profile disabled: {profile['profile_id']}")
        _expect(profile["ranker"]["bypassed"] == (profile["q_e4"] == 0), f"ranker bypass mismatch: {profile['profile_id']}")
        _expect((profile["ranker"]["checkpoint"] is None) == (profile["q_e4"] == 0), f"ranker binding mismatch: {profile['profile_id']}")


def _group_summary(profiles: list[dict[str, Any]], field: str, order: Iterable[Any]) -> list[dict[str, Any]]:
    result = []
    for value in order:
        group = [profile for profile in profiles if profile[field] == value]
        medians = [profile["payload"]["zstd_median_bytes"] for profile in group]
        result.append({
            field: value,
            "profiles": len(group),
            "relative_preservation_passed": sum(p["capabilities"]["relative_preservation_passed"] for p in group),
            "absolute_service_ready": sum(p["capabilities"]["absolute_service_ready"] for p in group),
            "localization_requirements_passed": sum(p["capabilities"]["localization_requirements_passed"] for p in group),
            "segmentation_installable": sum(p["capabilities"]["segmentation_installable"] for p in group),
            "stress_or_emergency": sum(p["capabilities"]["stress_or_emergency_anchor"] for p in group),
            "zstd_median_bytes_min": min(medians),
            "zstd_median_bytes_max": max(medians),
        })
    return result


def _dominance(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    def vector(profile: dict[str, Any]) -> tuple[float, int, int, int, int]:
        cap = profile["capabilities"]
        return (
            -float(profile["payload"]["zstd_median_bytes"]),
            cap["relative_preservation_count"],
            cap["absolute_service_count"],
            cap["localization_requirement_count"],
            int(cap["segmentation_installable"]),
        )

    dominated_by: dict[str, list[str]] = {}
    for candidate in profiles:
        cv = vector(candidate)
        dominators = []
        for other in profiles:
            if other is candidate:
                continue
            ov = vector(other)
            if all(a >= b for a, b in zip(ov, cv)) and any(a > b for a, b in zip(ov, cv)):
                dominators.append(other["profile_id"])
        if dominators:
            dominated_by[candidate["profile_id"]] = sorted(dominators)
    nondominated = [p["profile_id"] for p in profiles if p["profile_id"] not in dominated_by]
    return {
        "definition": "Pareto dominance over lower zstd median bytes and higher relative-preservation count, absolute-service count, localization count, and segmentation-installability; at least one objective must be strict.",
        "profiles_retained": 72,
        "dominated_profile_count": len(dominated_by),
        "nondominated_profile_count": len(nondominated),
        "nondominated_profile_ids": nondominated,
        "dominated_by": dominated_by,
        "policy": "diagnostic only; no dominated profile is removed or disabled",
    }


def _summary(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": {
            "profiles": len(profiles),
            "families": 4,
            "quantizers": 3,
            "q_anchors": 6,
            "transport_valid": sum(p["capabilities"]["transport_valid"] for p in profiles),
            "agent_action_enabled": sum(p["capabilities"]["agent_action_enabled"] for p in profiles),
            "stress_or_emergency": sum(p["capabilities"]["stress_or_emergency_anchor"] for p in profiles),
        },
        "by_family": _group_summary(profiles, "family", FAMILIES),
        "by_quantizer": _group_summary(profiles, "quantizer", QUANTIZERS),
        "by_q": _group_summary(profiles, "q_e4", (q_e4 for _, q_e4 in Q_ANCHORS)),
    }


CSV_COLUMNS = (
    "action_id", "profile_id", "execution_mode", "family", "latent_width", "quantizer", "bit_width", "q", "q_e4",
    "keep_count", "drop_count", "zstd_level", "wire_layout", "wire_version", "checkpoint_sha256", "routing_tag", "decoder_identity",
    "pre_zstd_analytical_bytes", "zstd_median_bytes", "zstd_p95_bytes", "ratio_vs_dense_fp32_q0", "ratio_vs_same_family_same_q_uint8",
    "vehicle_precision", "vehicle_recall", "vehicle_f1", "vehicle_xy_mae_m", "canonical_person_precision", "canonical_person_recall",
    "canonical_person_f1", "canonical_person_xy_mae_m", "person_avo_precision", "person_avo_recall", "person_avo_f1", "person_avo_xy_mae_m",
    "person_avo_recall_0_30m", "person_avo_recall_30_40m_diagnostic", "vehicle_iou", "person_box_mask_iou", "foreground_miou",
    "relative_preservation_passed", "relative_preservation_count", "absolute_service_ready", "absolute_service_count",
    "localization_requirements_passed", "localization_requirement_count", "localization_requirements_evaluated", "segmentation_installable",
    "segmentation_behavior", "source_contract_tier", "stress_or_emergency_anchor", "transport_valid", "agent_action_enabled",
    "source_evidence_path", "source_evidence_sha256", "durable_setting_path", "durable_setting_sha256",
)


def _csv_bytes(profiles: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for p in profiles:
        row = {key: p.get(key) for key in CSV_COLUMNS}
        row.update({f"wire_{key}": p["wire"][key] for key in ("layout", "version")})
        row.update(p["payload"])
        row.update(p["perception"])
        row.update({key: value for key, value in p["capabilities"].items() if key in CSV_COLUMNS})
        row.update({
            "source_evidence_path": p["source_evidence"]["path"],
            "source_evidence_sha256": p["source_evidence"]["sha256"],
            "durable_setting_path": p["durable_setting_evidence"]["path"],
            "durable_setting_sha256": p["durable_setting_evidence"]["sha256"],
        })
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def _markdown_table(rows: list[dict[str, Any]], key: str) -> list[str]:
    lines = [
        f"| {key} | profiles | relative | service | localization | segmentation | stress | zstd median range (bytes) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row[key]} | {row['profiles']} | {row['relative_preservation_passed']} | "
            f"{row['absolute_service_ready']} | {row['localization_requirements_passed']} | "
            f"{row['segmentation_installable']} | {row['stress_or_emergency']} | "
            f"{row['zstd_median_bytes_min']:.1f}–{row['zstd_median_bytes_max']:.1f} |"
        )
    return lines


def _report(catalog: dict[str, Any]) -> bytes:
    summary = catalog["summary"]
    dominance = catalog["diagnostics"]["pareto_dominance"]
    lines = [
        "# SplitFusion 72-profile action catalog",
        "",
        f"Frozen evidence base: `{BASE_COMMIT}`.",
        "",
        "The catalog locks 72 enabled SPLIT actions in family → quantizer → q order. All entries use CURRENT_CELL_MAJOR, mandatory zstd level 1, and the frozen p025 perception path. The builder reconciled 24 UINT8 and 48 low-bit durable setting records to their summaries without loading models or accessing frames.",
        "",
        "FULL_PRESERVATION is only the relative 12-gate result; it does not imply 9/9 absolute service readiness. Localization and segmentation are independent capabilities. q=0.90 and q=0.98 remain stress/emergency actions. Perception degradation does not disable a technically valid action, and STATE_INFEASIBLE remains a dynamic runtime state.",
        "",
        "When `segmentation_installable` is true, install current segmentation. Otherwise retain the prior segmentation layer with its original timestamp. The 0–30 m boundary is evaluation-only and never filters, suppresses, relabels, or rescores runtime detections.",
        "",
        "## Inventory",
        "",
        f"- Profiles: {summary['overall']['profiles']} (all transport-valid and agent-enabled)",
        f"- Stress/emergency profiles: {summary['overall']['stress_or_emergency']}",
        "- Durable settings reconciled: 72 (24 UINT8 + 48 UINT6/UINT4)",
        "- Phase-11D frame-settings bound: 160,560",
        "",
        "## By family",
        "",
        *_markdown_table(summary["by_family"], "family"),
        "",
        "## By quantizer",
        "",
        *_markdown_table(summary["by_quantizer"], "quantizer"),
        "",
        "## By q",
        "",
        *_markdown_table(summary["by_q"], "q_e4"),
        "",
        "## Evidence qualifications",
        "",
        "The frozen noAE and AE128 UINT8 records predate per-bin 0–30/30–40 reporting. Their range fields are null, their all-eight localization boolean is false, and the catalog records seven evaluated gates plus explicit not-evaluable provenance. No range metric is inferred from the historical 20–40 m aggregate.",
        "",
        "Two four-profile network definitions exist (`network_profile_design_v1.json` and `network_profile_design_v2.json`) with conflicting schema/route/target-SNR definitions. Their exact paths and hashes are recorded, but the binding is `UNRESOLVED_MULTIPLE_CONFLICTING_DEFINITIONS`; this does not alter the 72 actions.",
        "",
        "The transport supports continuous q mechanically at 1e-4 wire resolution over the registered mechanical range, but this initial RL action catalog includes only the six measured anchors. No unmeasured continuous-q action was generated.",
        "",
        "LOCAL_CPU, LOCAL_GPU, and SKIP are not action IDs here. A future top-level policy head may choose SPLIT (then one of these 72), LOCAL_CPU, LOCAL_GPU, or SKIP; SKIP sends no perception output and advances map AoI.",
        "",
        "## Pareto/dominance diagnostic",
        "",
        dominance["definition"],
        "",
        f"Nondominated: {dominance['nondominated_profile_count']}; dominated: {dominance['dominated_profile_count']}. All 72 profiles are retained and enabled.",
        "",
        "## Validation",
        "",
        "The builder failed closed on fixed input hashes, source terminals/schemas, Cartesian coverage, IDs, q/keep/drop arithmetic, summary/durable equality, Phase-11D same-family UINT8 references, finite defined metrics, positive payloads, codec/layout/zstd bindings, stress flags, source gate counts, transport integrity, and cleanup ordering.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> None:
    for binding in INPUTS.values():
        _verify_binding(binding)
    campaign = _verify_campaign_bindings()
    network = _network_metadata()
    uint8_rows, uint8_reconciliation = _reconcile_uint8()
    lowbit_rows, lowbit_reconciliation = _reconcile_phase11d(uint8_rows)
    profiles = _build_profiles(uint8_rows, lowbit_rows)
    _validate_profiles(profiles)
    summary = _summary(profiles)
    dominance = _dominance(profiles)

    reconciliation = {
        "uint8": uint8_reconciliation,
        "phase11d_lowbit": lowbit_reconciliation,
        "totals": {
            "summary_rows": 72,
            "durable_records_reconciled": 72,
            "cleanup_records_reconciled": 66,
            "noae_uint8_records_with_in_record_cleanup_confirmation": 6,
        },
        "same_family_same_q_uint8_references_reconciled": 48,
    }
    catalog = {
        "schema": SCHEMA,
        "frozen_evidence_base_commit": BASE_COMMIT,
        "action_order": {"family": list(FAMILIES), "quantizer": list(QUANTIZERS), "q_e4": [q_e4 for _, q_e4 in Q_ANCHORS]},
        "fixed_transport_contract": {
            "wire_layout": "CURRENT_CELL_MAJOR",
            "zstd": {"mandatory": True, "level": 1},
            "spatial_cells": CELLS,
            "q_wire_resolution": "1e-4",
            "keep_drop_rule": "drop=floor(q*N+0.5); keep=N-drop",
        },
        "continuous_q": {
            "mechanically_supported": True,
            "initial_rl_catalog_is_anchor_only": True,
            "validated_anchor_q_e4": [q_e4 for _, q_e4 in Q_ANCHORS],
            "unmeasured_continuous_q_actions_generated": False,
        },
        "mode_policy_boundary": {
            "catalog_execution_mode": "SPLIT",
            "SPLIT": "selects one of these 72 actions",
            "LOCAL_CPU": "separate future top-level execution mode; not an action ID here",
            "LOCAL_GPU": "separate future top-level execution mode; not an action ID here",
            "SKIP": "separate future top-level mode; no perception transmission and map AoI advances",
        },
        "segmentation_policy": {
            "install_when": "segmentation_installable=true",
            "otherwise": "retain the previous segmentation layer with its original timestamp",
        },
        "range_policy": {
            "primary_evaluation_range": "0 <= ground-truth distance < 30 m",
            "extended_diagnostic_range": "30 <= ground-truth distance <= 40 m",
            "runtime_detection_filtering": False,
        },
        "network_profiles": network,
        "evidence_bindings": {"inputs": INPUTS, "campaign": campaign},
        "reconciliation": reconciliation,
        "summary": summary,
        "diagnostics": {"pareto_dominance": dominance},
        "profiles": profiles,
    }

    catalog_path = PACKAGE_DIR / "splitfusion_72_action_catalog.json"
    csv_path = PACKAGE_DIR / "splitfusion_72_action_catalog.csv"
    report_path = PACKAGE_DIR / "CATALOG_REPORT.md"
    terminal_path = PACKAGE_DIR / TERMINAL
    provenance_path = PACKAGE_DIR / "provenance_hashes.json"
    _write(catalog_path, _canonical_json_bytes(catalog))
    _write(csv_path, _csv_bytes(profiles))
    _write(report_path, _report(catalog))
    catalog_sha = _sha256(catalog_path)
    _write(terminal_path, f"{TERMINAL} {catalog_sha}\n".encode("utf-8"))
    provenance = {
        "schema": "splitfusion_72_action_catalog_provenance_v1",
        "frozen_evidence_base_commit": BASE_COMMIT,
        "builder": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": _sha256(Path(__file__).resolve())},
        "input_bindings": INPUTS,
        "generated_artifacts": {
            catalog_path.name: _sha256(catalog_path),
            csv_path.name: _sha256(csv_path),
            report_path.name: _sha256(report_path),
            terminal_path.name: _sha256(terminal_path),
        },
        "validation": {
            "profiles": 72,
            "transport_valid": 72,
            "agent_action_enabled": 72,
            "durable_records_reconciled": 72,
            "same_family_same_q_references_reconciled": 48,
            "input_hashes_verified": len(INPUTS),
            "network_binding_status": network["binding_status"],
        },
    }
    _write(provenance_path, _canonical_json_bytes(provenance))
    print(TERMINAL)


if __name__ == "__main__":
    main()
