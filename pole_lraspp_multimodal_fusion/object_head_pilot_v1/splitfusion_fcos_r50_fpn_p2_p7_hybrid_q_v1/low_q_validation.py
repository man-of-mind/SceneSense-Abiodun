"""Fixed low-q validation sweep over the completed continuous-q interface.

Measures the validation accuracy-payload curve at five unregistered low q values
-- 0.05, 0.10, 0.15, 0.20, 0.25 -- to answer one question: does a near-lossless
segmentation operating point exist below q=0.30? Phase 6 measured q=0.30 as
`localization-preserving/segmentation-reduced` with 7 of the 12 registered
near-lossless preservation gates passing, so the whole 0 < q < 0.30 interval was
unmeasured and the near-lossless boundary, if it exists at all, lies inside it.

This is measurement only. Nothing here trains, fine-tunes, recalibrates, selects
a checkpoint, moves a threshold, edits a model parameter, changes the transport
contract or invents a gate. The stable Phase-5 `ranker_epoch_04.pt` is loaded
read-only; the diverged epoch-8 and epoch-12 q-aware checkpoints are neither
loaded nor referenced; the locked test split, CARLA and the Phase-4 teacher maps
are not opened (the Phase-4 shards are hash-verified as bound inputs, exactly as
Phase 6 did, and no teacher tensor is read).

The transport path is `continuous_q.transport()`, called directly. `contract.
snap_continuous_q` is deliberately *not* called: snapping would serve q=0 for
every setting in this sweep and measure nothing. Every reused row is reused, not
recomputed:

  * q=0    -- the frozen p025 validation prediction set, re-scored by the same
              Phase-6 scoring functions and required to reproduce the published
              row exactly, so the new q rows are comparable.
  * q=0.30 -- read verbatim out of the completed Phase-6 `validation_curve.json`
              under an exact sha256 binding.

Reused-row caveat, unchanged: the accuracy of a q measured here is measured, not
interpolated. The five new rows say nothing about q values between them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
    apply_p025_service_policy,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
    combined_records,
)

from . import continuous_q, contract, guards
from .codec import HEADER_BYTES, _unpack_bitmask
from .gpu_qualification import load_frozen_perception, sha256_file
from .phase5_common import load_frozen_scorers, source_delta
from .phase6_validation import (
    DATALOADER_WORKERS,
    INFERENCE_BATCH,
    ROUNDTRIP_VERIFY_FRAMES,
    _collate,
    _person_only,
    bind_phase6_inputs,
    evaluate_preservation_gates,
    frozen_q0_pass,
    load_validation_person_truth,
    require_frozen_q0_reproduced,
    score_validation_pass,
)
from .ranker import build_ranker

EXECUTE_TOKEN = "HYBRID_Q_LOW_Q_VALIDATION_CURVE"
TERMINAL = "HYBRID_Q_LOW_Q_VALIDATION_CURVE_COMPLETE"
SCHEMA = "splitfusion_fcos_hybrid_q_low_q_validation_curve_v1"

# The five q settings this sweep measures. Fixed before the run; not tuned, not
# extended by the result, and every one of them is unregistered.
LOW_Q_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25)

# The already-measured anchors this sweep is bracketed by, reused verbatim.
REUSED_LOWER_Q = 0.00
REUSED_UPPER_Q = 0.30

# The full ordered ladder used for the neighbour-nesting check: each new q must
# retain a strict superset of its more-aggressive neighbour's cells and a strict
# subset of its less-aggressive neighbour's cells.
LADDER = (REUSED_LOWER_Q,) + LOW_Q_VALUES + (REUSED_UPPER_Q,)

# Frames per pass on which the neighbour-nesting property is measured directly
# from the same q-independent score map rather than asserted.
NESTING_VERIFY_FRAMES = 16

# The completed Phase-6 artifact whose q=0.30 row is reused. Bound by exact hash
# so a rebased or regenerated curve fails closed instead of silently changing the
# upper bracket of this sweep.
PHASE6_CURVE_RELPATH = (
    "experiments/splitfusion_fcos_hybrid_q_v1/"
    "20260902_182401_phase6_validation_curve/validation_curve.json"
)
PHASE6_CURVE_SHA256 = (
    "54987920a7430564425664e82511d1121e77935beabfbd4cf2f34bee5cadfc74"
)


def _q_slug(q: float) -> str:
    """Stable filesystem slug at wire resolution: q=0.05 -> `q0500` (e4 units)."""
    return f"q{contract._q_to_e4(float(q)):04d}"


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> str:
    """Write one completed setting atomically, then return its sha256.

    A completed setting must survive a later setting failing, so the payload is
    staged beside its destination and moved into place with a single
    `os.replace`: a reader either sees the previous state or the whole document.
    """
    text = json.dumps(document, indent=2, sort_keys=True, default=str) + "\n"
    staging = path.with_name(path.name + ".partial")
    with staging.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_file(path)


def neighbours_of(q: float) -> tuple[float, float]:
    """The (less-aggressive, more-aggressive) ladder neighbours of one new q."""
    index = [contract._q_to_e4(value) for value in LADDER].index(contract._q_to_e4(q))
    if not 0 < index < len(LADDER) - 1:
        raise guards.HybridQConfigError(f"q={q!r} is not an interior ladder setting")
    return LADDER[index - 1], LADDER[index + 1]


def bind_reused_upper_row() -> dict[str, Any]:
    """Load the reused Phase-6 q=0.30 row under an exact artifact hash binding."""
    root = contract.repository_root()
    path = (root / PHASE6_CURVE_RELPATH).resolve(strict=True)
    observed = sha256_file(path)
    if observed != PHASE6_CURVE_SHA256:
        raise guards.HybridQConfigError("reused Phase-6 validation curve sha256 drift")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document["terminal"] != contract.PHASE6_TERMINAL:
        raise guards.HybridQConfigError("reused Phase-6 artifact is not complete")
    if document["schema"] != contract.PHASE6_SCHEMA:
        raise guards.HybridQConfigError("reused Phase-6 artifact schema drift")
    if int(document["scope"]["validation_frames"]) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("reused Phase-6 validation frame count drift")
    if bool(document["scope"]["test_accessed"]):
        raise guards.HybridQConfigError("reused Phase-6 artifact reports test access")

    matches = [
        entry
        for entry in document["measured_q_passes"]
        if contract._q_to_e4(float(entry["q"])) == contract._q_to_e4(REUSED_UPPER_Q)
    ]
    if len(matches) != 1:
        raise guards.HybridQConfigError("reused Phase-6 q=0.30 row is not unique")
    row = matches[0]
    if int(row["retained_cells"]) != contract.keep_count(REUSED_UPPER_Q):
        raise guards.HybridQConfigError("reused Phase-6 q=0.30 keep-count drift")
    return {
        "path": PHASE6_CURVE_RELPATH,
        "sha256": observed,
        "terminal": document["terminal"],
        "row": row,
    }


# ---------------------------------------------------------------------------
# One continuous-q configuration over the fixed validation split
# ---------------------------------------------------------------------------


def run_continuous_validation_pass(
    *, model: torch.nn.Module, base: Any, ranker: torch.nn.Module, q: float,
    dataset: Any, positions: Sequence[int], frame_ids: Sequence[str],
    device: torch.device, output: Path, workers: int, limit: int | None = None,
) -> dict[str, Any]:
    """Transport, decode and serve one unregistered q over the validation split.

    Structurally identical to `phase6_validation.run_validation_pass` -- same
    frozen front end, same stable ranker, same exact-cardinality selection, same
    framed v1 codec, same frozen p025 service policy, same split and order. The
    only difference is that selection, masking and framing go through
    `continuous_q.transport()`, which serves the requested q exactly instead of
    admitting registered anchors only.
    """
    plan = continuous_q.quantize_q(q)
    if plan.is_bypass:
        raise guards.HybridQConfigError(
            "q=0 is reused from the frozen p025 validation result and must not be rerun"
        )
    if plan.is_registered:
        raise guards.HybridQConfigError(
            f"q={plan.wire_q!r} is a registered anchor with a published row; "
            "this sweep measures unregistered low q only"
        )
    if ranker is None:
        raise guards.HybridQConfigError("a nonzero q requires the stable ranker")

    lower_q, upper_q = neighbours_of(plan.wire_q)
    lower_plan = continuous_q.quantize_q(lower_q)
    upper_plan = continuous_q.quantize_q(upper_q)

    output.mkdir(parents=True, exist_ok=False)
    (output / "segmentation").mkdir()
    detections_path = output / "detections.csv"
    manifest_path = output / "segmentation_manifest.csv"

    loader = DataLoader(
        Subset(dataset, list(positions)),
        batch_size=INFERENCE_BATCH, shuffle=False, num_workers=workers,
        collate_fn=_collate, drop_last=False, pin_memory=False,
    )
    payload_bytes: set[int] = set()
    keep_counts: set[int] = set()
    wire_q_values: set[int] = set()
    snapped_values: set[bool] = set()
    requested_equals_wire = True
    roundtrip_exact = True
    mask_exact = True
    roundtrip_verified = 0
    nested_below = True
    nested_above = True
    nesting_verified = 0
    observed_ids: list[str] = []
    segmentation_rows: list[dict[str, Any]] = []
    detection_count = 0
    person_count = 0
    vehicle_count = 0
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    with detections_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=base.infer.FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for batch_index, (fused, rows, calibrations) in enumerate(loader):
                if limit is not None and batch_index >= limit:
                    break
                inputs = fused.to(device, non_blocking=True)
                c2 = model.encode_front(inputs).float()
                guards.require_frozen_batched_c2(c2, what="frozen validation C2")

                transported = []
                for index in range(c2.shape[0]):
                    frame = c2[index]
                    result = continuous_q.transport(frame, ranker, plan.wire_q)
                    selection = result.selection
                    if selection is None:
                        raise guards.HybridQPayloadError("continuous transport bypassed")

                    # (6) requested q == wire q, and the request was never snapped.
                    requested_equals_wire &= (
                        contract._q_to_e4(result.plan.requested_q)
                        == contract._q_to_e4(result.plan.wire_q)
                        == plan.q_e4
                    )
                    snapped_values.add(bool(result.plan.snapped))
                    wire_q_values.add(contract._q_to_e4(result.plan.wire_q))
                    keep_counts.add(int(selection.keep_count))
                    payload_bytes.add(int(result.payload.total_bytes))

                    decoded, decoded_q = continuous_q.decode(result.payload)
                    if contract._q_to_e4(decoded_q) != plan.q_e4:
                        raise guards.HybridQPayloadError("decoded q drift")

                    if roundtrip_verified < ROUNDTRIP_VERIFY_FRAMES:
                        # Bit-exactness is defined on the retained set, the dropped
                        # set and the mask -- not on the whole dense tensor, which
                        # a lossy-by-design drop can never reproduce.
                        cpu_frame = frame.detach().cpu()
                        mask = selection.keep_mask.unsqueeze(0).expand_as(cpu_frame).cpu()
                        roundtrip_exact &= bool(torch.equal(decoded[mask], cpu_frame[mask]))
                        roundtrip_exact &= bool((decoded[~mask] == 0).all())
                        wire_indices = torch.from_numpy(
                            _unpack_bitmask(
                                result.payload.data[
                                    HEADER_BYTES:HEADER_BYTES + result.payload.mask_bytes
                                ],
                                contract.SPLIT_CELLS,
                            )
                        )
                        mask_exact &= bool(
                            torch.equal(wire_indices, selection.keep_indices.cpu())
                        )
                        roundtrip_verified += 1
                        del cpu_frame, wire_indices

                    if nesting_verified < NESTING_VERIFY_FRAMES:
                        # The ranker never sees q, so one score map fixes one
                        # ordering and each setting is a prefix of it. Measure the
                        # nesting rather than assume it.
                        scores = ranker.score_cells(frame)
                        kept = set(selection.keep_indices.cpu().tolist())
                        if upper_plan.is_bypass:
                            raise guards.HybridQConfigError(
                                "the more-aggressive neighbour cannot be the dense bypass"
                            )
                        above = set(
                            continuous_q.select_cells(scores, upper_plan.wire_q)
                            .keep_indices.cpu().tolist()
                        )
                        nested_above &= above.issubset(kept)
                        if lower_plan.is_bypass:
                            # q=0 retains every cell by construction; there is no
                            # selection object to build and nothing to rank.
                            nested_below &= len(kept) < lower_plan.keep_count
                        else:
                            below = set(
                                continuous_q.select_cells(scores, lower_plan.wire_q)
                                .keep_indices.cpu().tolist()
                            )
                            nested_below &= kept.issubset(below)
                        nesting_verified += 1
                        del scores

                    transported.append(decoded.to(device))
                    del result

                hybrid = torch.stack(transported)
                outputs = model.decode_tail(hybrid, dense=False)
                calibration_gpu = [
                    {name: tensor.to(device) for name, tensor in calibration.items()}
                    for calibration in calibrations
                ]
                detections = model.postprocess(outputs, calibration_gpu)
                for index, row in enumerate(rows):
                    frame_view = {"semantic_logits": outputs["semantic_logits"][index:index + 1]}
                    served, original_indices = apply_p025_service_policy(
                        frame_view, detections[index]
                    )
                    records = combined_records(base, row, served, original_indices)
                    for record in records:
                        writer.writerow(record)
                        if record["class_name"] == "person":
                            person_count += 1
                        else:
                            vehicle_count += 1
                    detection_count += len(records)
                    observed_ids.append(str(row["sample_id"]))

                    source_hw = (int(row["camera_height"]), int(row["camera_width"]))
                    labels = F.interpolate(
                        outputs["semantic_logits"][index:index + 1].float(),
                        size=source_hw, mode="bilinear", align_corners=False,
                    ).argmax(1)[0]
                    array = labels.cpu().numpy().astype(np.uint8)
                    relative = Path("segmentation") / f"{row['sample_id']}.png"
                    if not cv2.imwrite(str(output / relative), array):
                        raise RuntimeError(f"failed segmentation write {relative}")
                    segmentation_rows.append({
                        "sample_id": row["sample_id"], "prediction_path": str(relative),
                        "width": array.shape[1], "height": array.shape[0],
                    })
                del c2, hybrid, outputs, detections, transported

    with manifest_path.open("x", encoding="utf-8", newline="") as stream:
        manifest_writer = csv.DictWriter(
            stream, fieldnames=("sample_id", "prediction_path", "width", "height")
        )
        manifest_writer.writeheader()
        manifest_writer.writerows(segmentation_rows)

    del loader
    if limit is None:
        if observed_ids != list(frame_ids):
            raise guards.HybridQConfigError("validation inference order drift")
        if len(set(observed_ids)) != contract.VALIDATION_FRAMES:
            raise guards.HybridQConfigError("validation frame coverage drift")
    if keep_counts != {plan.keep_count}:
        raise guards.HybridQPayloadError(f"observed keep counts {sorted(keep_counts)}")
    if wire_q_values != {plan.q_e4}:
        raise guards.HybridQPayloadError(f"observed wire q_e4 {sorted(wire_q_values)}")
    if snapped_values != {False}:
        raise guards.HybridQPayloadError("a continuous request reported itself snapped")
    if not requested_equals_wire:
        raise guards.HybridQPayloadError("requested q did not equal wire q")
    if not roundtrip_exact:
        raise guards.HybridQPayloadError("framed encode/decode was not exact")
    if not mask_exact:
        raise guards.HybridQPayloadError("decoded mask does not match the selected cells")
    if not (nested_above and nested_below):
        raise guards.HybridQPayloadError("q mask is not nested between its neighbours")
    observed_payload = sorted(payload_bytes)
    if len(observed_payload) != 1:
        raise guards.HybridQPayloadError(f"non-constant framed payload {observed_payload}")

    return {
        "q": plan.wire_q,
        "frames": len(observed_ids),
        "prediction_root": str(output),
        "detections_csv_sha256": sha256_file(detections_path),
        "detections": detection_count,
        "person_service_outputs": person_count,
        "vehicle_service_outputs": vehicle_count,
        "retained_cells": plan.keep_count,
        "dropped_cells": plan.drop_count,
        "framed_payload_bytes": observed_payload[0],
        "framed_payload_ratio": contract.framed_payload_ratio(observed_payload[0]),
        "raw_fp32_ratio": contract.raw_fp32_ratio(observed_payload[0]),
        "framed_encode_decode_exact": True,
        "framed_encode_decode_frames_verified": roundtrip_verified,
        "inference_run_here": True,
        "transport_path": "continuous_q.transport",
        "snap_continuous_q_called": False,
        "continuous_q_verification": {
            "requested_q": plan.requested_q,
            "wire_q": plan.wire_q,
            "wire_q_e4": plan.q_e4,
            "wire_resolution": continuous_q.WIRE_Q_RESOLUTION,
            "requested_q_equals_wire_q_every_frame": True,
            "snapped_every_frame": sorted(snapped_values),
            "is_registered_anchor": plan.is_registered,
            "expected_keep_count": plan.keep_count,
            "observed_keep_counts": sorted(keep_counts),
            "exact_keep_count_every_frame": True,
            "observed_framed_payload_bytes": observed_payload,
            "bit_exact_encode_decode": True,
            "bit_exact_frames_verified": roundtrip_verified,
            "bit_exact_definition": {
                "retained_values_bit_identical_to_source": True,
                "dropped_cells_decode_to_exact_zero": True,
                "decoded_mask_matches_selected_indices": True,
                "note": (
                    "bit-exactness is defined on the retained set, the dropped set "
                    "and the mask; the dense decoded C2 is not, and cannot be, equal "
                    "to the original dense C2 at q>0"
                ),
            },
            "nesting": {
                "frames_checked": nesting_verified,
                "less_aggressive_neighbour_q": lower_plan.wire_q,
                "less_aggressive_neighbour_keep_count": lower_plan.keep_count,
                "more_aggressive_neighbour_q": upper_plan.wire_q,
                "more_aggressive_neighbour_keep_count": upper_plan.keep_count,
                "subset_of_less_aggressive_neighbour": bool(nested_below),
                "superset_of_more_aggressive_neighbour": bool(nested_above),
                "nested_between_neighbours": bool(nested_above and nested_below),
                "less_aggressive_neighbour_is_dense_bypass": lower_plan.is_bypass,
                "dense_bypass_note": (
                    "q=0 retains all 21504 cells by construction, so containment is "
                    "checked as a strict cardinality bound and no ranking is run"
                    if lower_plan.is_bypass else None
                ),
            },
        },
        "wall_seconds": time.time() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def episode_diagnostic(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Compact per-episode view, read off the existing frozen AVO person scorer.

    `run_comparison.score_person_view` already returns per-episode counts, so this
    is a projection of numbers the sweep computed anyway. It adds no scoring pass,
    no uncertainty estimate and no evaluation machinery, and it is a diagnostic:
    the reported curve and the verdict are the split-level numbers.
    """
    episodes = entry["person_avo_detail"]["episodes"]
    return {
        name: {
            "person_avo_precision": float(row["precision"]),
            "person_avo_recall": float(row["recall"]),
            "person_avo_f1": float(row["f1"]),
            "person_avo_xy_mae_m": float(row["xy_mae_m"]),
            "observable_gt": int(row["observable_gt"]),
            "tp": int(row["tp"]),
            "fp": int(row["fp"]),
            "fn": int(row["fn"]),
        }
        for name, row in episodes.items()
    }


def curve_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One compact reporting row: payload, all four metric views and both gate sets."""
    preservation = entry["preservation_gates"]
    return {
        "q": float(entry["q"]),
        "source": entry["source"],
        "retained_cells": int(entry["retained_cells"]),
        "dropped_cells": int(entry["dropped_cells"]),
        "framed_payload_bytes": int(entry["framed_payload_bytes"]),
        "framed_payload_ratio": float(entry["framed_payload_ratio"]),
        "raw_fp32_ratio": float(entry["raw_fp32_ratio"]),
        "metrics": dict(entry["metrics"]),
        "canonical_person_metrics": dict(entry["canonical_person_metrics"]),
        "absolute_service_pass_count": int(entry["absolute_service_gates"]["pass_count"]),
        "failed_absolute_service_gates": list(entry["absolute_service_gates"]["failed"]),
        "person_avo_by_episode": episode_diagnostic(entry),
        "preservation_gates_passed": int(preservation["pass_count"]),
        "preservation_gate_count": int(preservation["gate_count"]),
        "preservation_all_passed": bool(preservation["all_passed"]),
        "failed_preservation_gates": list(preservation["failed"]),
        "worst_preservation_degradation": float(preservation["worst_degradation"]),
    }


def near_lossless_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The largest measured q that passes all 12 registered preservation gates.

    Reads the registered gates only. No gate is invented, retuned or relaxed, and
    q=0 is excluded from the answer because it is the dense identity baseline the
    gates are defined against -- it passes by construction and is not an operating
    point that saves any payload.
    """
    candidates = [
        row for row in rows
        if row["preservation_all_passed"] and float(row["q"]) > 0.0
    ]
    best = max(candidates, key=lambda row: float(row["q"])) if candidates else None
    return {
        "gate_set": "contract.HOLDOUT_PRESERVATION_GATES",
        "gate_count": len(contract.HOLDOUT_PRESERVATION_GATES),
        "gates_invented_or_tuned": False,
        "q_values_considered": [float(row["q"]) for row in rows],
        "passing_q_values": [float(row["q"]) for row in candidates],
        "largest_q_passing_all_gates": None if best is None else float(best["q"]),
        "near_lossless_operating_point_exists_below_0_30": bool(candidates),
        "baseline_excluded_from_verdict": REUSED_LOWER_Q,
        "baseline_exclusion_reason": (
            "q=0 is the dense identity the gates are measured against; it passes with "
            "zero degradation by construction and transmits the full payload"
        ),
        "selected_row": None if best is None else {
            "q": float(best["q"]),
            "retained_cells": best["retained_cells"],
            "framed_payload_bytes": best["framed_payload_bytes"],
            "framed_payload_ratio": best["framed_payload_ratio"],
            "absolute_service_pass_count": best["absolute_service_pass_count"],
        },
        "scope": (
            "measured on the registered 3,345-frame validation split at the stable "
            "epoch-4 ranker; accuracy at any q not listed here is neither measured "
            "nor interpolated"
        ),
        "status": (
            "any q reported here as passing 12/12 is a validation-selected "
            "engineering operating point, not independent unseen-test confirmation; "
            "the locked test split was not opened"
        ),
        "independent_test_confirmation": False,
    }


CSV_COLUMNS = (
    "q", "source", "retained_cells", "framed_payload_bytes", "framed_payload_ratio",
    "vehicle_precision", "vehicle_recall", "vehicle_f1", "vehicle_xy_mae_m",
    "person_canonical_precision", "person_canonical_recall", "person_canonical_f1",
    "person_canonical_xy_mae_m",
    "person_avo_precision", "person_avo_recall", "person_avo_f1",
    "person_avo_xy_mae_m", "person_avo_recall_20_40m",
    "vehicle_iou", "person_box_mask_iou", "foreground_miou",
    "absolute_service_pass_count", "preservation_gates_passed",
    "preservation_all_passed", "failed_preservation_gates",
)


def write_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            canonical = row["canonical_person_metrics"]
            writer.writerow({
                "q": f"{row['q']:.4f}",
                "source": row["source"],
                "retained_cells": row["retained_cells"],
                "framed_payload_bytes": row["framed_payload_bytes"],
                "framed_payload_ratio": f"{row['framed_payload_ratio']:.6f}",
                **{
                    name: f"{float(row['metrics'][name]):.6f}"
                    for name in contract.PROTECTED_METRICS
                },
                "person_canonical_precision": f"{canonical['person_precision']:.6f}",
                "person_canonical_recall": f"{canonical['person_recall']:.6f}",
                "person_canonical_f1": f"{canonical['person_f1']:.6f}",
                "person_canonical_xy_mae_m": f"{canonical['person_xy_mae_m']:.6f}",
                "absolute_service_pass_count": row["absolute_service_pass_count"],
                "preservation_gates_passed": row["preservation_gates_passed"],
                "preservation_all_passed": row["preservation_all_passed"],
                "failed_preservation_gates": "|".join(row["failed_preservation_gates"]),
            })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid-q fixed low-q validation sweep over the continuous-q interface"
    )
    parser.add_argument("--execute", required=True, choices=(EXECUTE_TOKEN,))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DATALOADER_WORKERS)
    parser.add_argument("--keep-segmentation", action="store_true")
    args = parser.parse_args()

    output = args.output
    settings_dir = output / "settings"
    if not torch.cuda.is_available():
        raise RuntimeError("the low-q validation sweep requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(contract.RANKER_INIT_SEED)
    started = time.time()

    output.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(parents=True, exist_ok=True)
    for stale in settings_dir.glob("*.partial"):
        stale.unlink()

    binding = bind_phase6_inputs()
    delta = source_delta(binding)
    reused_upper = bind_reused_upper_row()
    model, base, perception = load_frozen_perception(device)
    frozen_snapshot = guards.snapshot_module_state(model)

    payload = torch.load(
        contract.repository_root() / contract.VALIDATION_RANKER_RELPATH,
        map_location="cpu", weights_only=False,
    )
    if int(payload["epoch"]) != contract.VALIDATION_RANKER_EPOCH:
        raise guards.HybridQConfigError("stable ranker epoch drift")
    if int(payload["parameter_count"]) != contract.RANKER_PARAMETER_COUNT:
        raise guards.HybridQConfigError("stable ranker parameter count drift")
    ranker = build_ranker()
    ranker.load_state_dict(payload["ranker"])
    ranker = ranker.to(device).eval()
    for parameter in ranker.parameters():
        parameter.requires_grad_(False)
    ranker_snapshot = guards.snapshot_module_state(ranker)

    root = contract.repository_root()
    config = json.loads(
        (root / "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
         "splitfusion_fcos_r50_fpn_p2_p7_v1/config.json").read_text(encoding="utf-8")
    )
    dataset_root = (root / config["dataset_root"]).resolve(strict=True)
    truth = load_validation_person_truth()
    frame_ids = list(truth["frame_ids"])

    inference = base.data.InferenceDataset(dataset_root, "val")
    position_by_id = {row["sample_id"]: index for index, row in enumerate(inference.rows)}
    if len(position_by_id) != contract.VALIDATION_FRAMES:
        raise guards.HybridQConfigError("inference dataset validation frame count drift")
    positions = [position_by_id[sample_id] for sample_id in frame_ids]

    scorers = load_frozen_scorers()
    gt, gt_states = scorers.load_gt(dataset_root, contract.PRIMARY_CONTRACT)
    validation_gt = {sample_id: gt.get(sample_id, []) for sample_id in frame_ids}
    person_gt = _person_only(validation_gt)
    ignore_cache: dict[str, Any] = {}

    print(f"[low-q] {len(frame_ids)} validation frames; sweep {list(LOW_Q_VALUES)}; "
          f"reusing q={REUSED_LOWER_Q} and q={REUSED_UPPER_Q}", flush=True)

    # --- q = 0: reuse the frozen p025 validation result, scored by this path ---
    baseline = score_validation_pass(
        result=frozen_q0_pass(), scorers=scorers, truth=truth, experiment=dataset_root,
        frame_ids=frame_ids, gt=validation_gt, person_gt=person_gt,
        ignore_cache=ignore_cache,
    )
    reproduction = require_frozen_q0_reproduced(baseline)
    baseline["source"] = "frozen p025 q=0 validation result, reused verbatim"
    baseline["frozen_q0_reproduction"] = reproduction
    # The gates are defined as a degradation from this exact row, so evaluating
    # them against itself is the identity: 12/12 with zero degradation.
    baseline["preservation_gates"] = evaluate_preservation_gates(
        baseline["metrics"], baseline["metrics"]
    )
    if not baseline["preservation_gates"]["all_passed"]:
        raise guards.HybridQConfigError("q=0 failed its own identity gate evaluation")
    print(json.dumps({
        "row": "q=0.00",
        "frozen_q0_reproduced_exactly": True,
        "preservation_gates_passed": baseline["preservation_gates"]["pass_count"],
    }), flush=True)

    # --- q = 0.30: reuse the completed Phase-6 row verbatim ---
    upper = dict(reused_upper["row"])
    upper["source"] = "completed Phase-6 validation curve, reused verbatim"
    upper["inference_run_here"] = False
    upper["reused_artifact"] = {
        "path": reused_upper["path"], "sha256": reused_upper["sha256"],
    }
    if upper["preservation_gates"]["pass_count"] != 7:
        raise guards.HybridQConfigError("reused q=0.30 preservation pass count drift")

    # --- the five new q settings, sequential, each saved atomically ---
    measured: list[dict[str, Any]] = []
    for value in LOW_Q_VALUES:
        slug = _q_slug(value)
        completed = settings_dir / f"{slug}.json"
        if completed.exists():
            entry = json.loads(completed.read_text(encoding="utf-8"))
            if contract._q_to_e4(float(entry["q"])) != contract._q_to_e4(value):
                raise guards.HybridQConfigError(f"{completed} holds a different q")
            if entry.get("schema") != SCHEMA:
                raise guards.HybridQConfigError(f"{completed} schema drift")
            measured.append(entry)
            print(json.dumps({"reused_completed_setting": slug, "q": value}), flush=True)
            continue

        prediction_root = output / "predictions" / slug
        if prediction_root.exists():
            # No completed result exists for this setting, so whatever is on disk
            # is a partial prediction set from an interrupted attempt.
            print(f"[low-q] clearing incomplete prediction set {prediction_root}",
                  flush=True)
            shutil.rmtree(prediction_root)

        raw = run_continuous_validation_pass(
            model=model, base=base, ranker=ranker, q=value, dataset=inference,
            positions=positions, frame_ids=frame_ids, device=device,
            output=prediction_root, workers=int(args.workers),
        )
        guards.require_module_state_unchanged(model, frozen_snapshot)
        guards.require_module_state_unchanged(ranker, ranker_snapshot)
        scored = score_validation_pass(
            result=raw, scorers=scorers, truth=truth, experiment=dataset_root,
            frame_ids=frame_ids, gt=validation_gt, person_gt=person_gt,
            ignore_cache=ignore_cache,
        )
        scored["schema"] = SCHEMA
        scored["source"] = "measured here through continuous_q.transport"
        scored["preservation_gates"] = evaluate_preservation_gates(
            baseline["metrics"], scored["metrics"]
        )
        scored["absolute_change_from_q0"] = {
            name: float(scored["metrics"][name]) - float(baseline["metrics"][name])
            for name in contract.PROTECTED_METRICS
        }
        if not args.keep_segmentation:
            directory = Path(scored["prediction_root"]) / "segmentation"
            if directory.is_dir():
                shutil.rmtree(directory)
            scored["segmentation_masks_removed_after_scoring"] = True
        scored["completed_utc"] = datetime.now(timezone.utc).isoformat()
        digest = _atomic_write_json(completed, scored)
        measured.append(scored)
        print(json.dumps({
            "row": f"q={value:.2f}",
            "retained_cells": scored["retained_cells"],
            "framed_payload_bytes": scored["framed_payload_bytes"],
            "framed_payload_ratio": scored["framed_payload_ratio"],
            "absolute_service_pass_count": scored["absolute_service_gates"]["pass_count"],
            "preservation_gates_passed": scored["preservation_gates"]["pass_count"],
            "preservation_all_passed": scored["preservation_gates"]["all_passed"],
            "failed_preservation_gates": scored["preservation_gates"]["failed"],
            "saved": str(completed), "setting_sha256": digest,
        }, indent=2), flush=True)

    guards.require_module_state_unchanged(model, frozen_snapshot)
    guards.require_module_state_unchanged(ranker, ranker_snapshot)

    entries = [baseline] + measured + [upper]
    rows = [curve_row(entry) for entry in entries]
    if [contract._q_to_e4(row["q"]) for row in rows] != [
        contract._q_to_e4(value) for value in LADDER
    ]:
        raise guards.HybridQConfigError(
            "reported ladder does not match the registered sweep ladder"
        )
    verdict = near_lossless_verdict(rows)

    report = {
        "schema": SCHEMA,
        "terminal": TERMINAL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "measure whether a near-lossless segmentation operating point exists "
            "below q=0.30 on the registered validation split"
        ),
        "phase_role": (
            "measurement only: no training, tuning, recalibration, checkpoint "
            "selection, threshold change or architecture change"
        ),
        "scope": {
            "evaluated_split": "registered fixed validation split",
            "validation_episodes": list(contract.VALIDATION_EPISODES),
            "validation_frames": contract.VALIDATION_FRAMES,
            "frames_used_per_setting": contract.VALIDATION_FRAMES,
            "test_accessed": False,
            "carla_launched": False,
            "teacher_maps_read": False,
            "teacher_cache_shards_hash_verified_only": True,
            "measured_q_values": list(LOW_Q_VALUES),
            "reused_q_values": [REUSED_LOWER_Q, REUSED_UPPER_Q],
            "inference_passes_run": sum(
                1 for entry in measured if entry.get("inference_run_here")
            ),
            "training_run": False,
            "tuning_or_recalibration": False,
            "thresholds_changed": False,
            "model_parameters_changed": False,
            "ranker_modified": False,
            "architecture_changed": False,
            "new_gate_invented_or_tuned": False,
            "excluded_ranker_epochs": list(contract.VALIDATION_EXCLUDED_RANKER_EPOCHS),
            "excluded_ranker_epochs_loaded": False,
            "excluded_ranker_reason": contract.VALIDATION_EXCLUDED_RANKER_REASON,
        },
        "transport": {
            "interface": "continuous_q.transport",
            "snap_continuous_q_called": False,
            "snap_note": (
                "snapping every requested q down to the nearest registered anchor "
                "would serve q=0 for all five settings and measure nothing"
            ),
            "wire_q_resolution": continuous_q.WIRE_Q_RESOLUTION,
            "codec": "frozen v1 sparse codec, 44-byte header, unmodified",
            "wire_format_changed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "inference_precision": "fp32 inference_mode, no autocast",
        },
        "binding": {k: v for k, v in binding.items() if k != "teacher_cache_shards"},
        "source_delta": delta,
        "perception_binding": perception,
        "reused_upper_anchor": {
            "q": REUSED_UPPER_Q,
            "path": reused_upper["path"],
            "sha256": reused_upper["sha256"],
            "terminal": reused_upper["terminal"],
            "rerun": False,
        },
        "reused_lower_anchor": {
            "q": REUSED_LOWER_Q,
            "prediction_root": contract.FROZEN_Q0_PREDICTION_ROOT,
            "detections_sha256": contract.FROZEN_Q0_DETECTIONS_SHA256,
            "inference_rerun": False,
            "rescored_by_this_path": True,
            "published_row_reproduced_exactly": True,
            "reproduction": reproduction,
        },
        "service_pipeline": {
            "policy": (
                "splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1."
                "apply_p025_service_policy"
            ),
            "person_output_threshold": contract.PERSON_SERVICE_SCORE_THRESHOLD,
            "person_avo_threshold": contract.PERSON_AVO_THRESHOLD,
            "vehicle_score_point": contract.VEHICLE_SCORE_THRESHOLD,
            "thresholds_or_postprocessing_changed": False,
        },
        "scoring": {
            "vehicle_and_canonical_person": "frozen audit_v1.score_arm",
            "segmentation": "frozen score_contract_v1.score_segmentation",
            "person_avo": "frozen validation run_comparison.score_person_view",
            "frozen_scorer_sha256": scorers.sha256,
            "contract": contract.PRIMARY_CONTRACT,
            "gt_contract_states": gt_states,
            "geometry_and_segmentation_semantics_changed": False,
            "validation_person_truth": {
                "avo_table": contract.VALIDATION_AVO_TABLE_RELPATH,
                "avo_table_sha256": contract.VALIDATION_AVO_TABLE_SHA256,
                "avo_table_rows": truth["avo_table_rows"],
                "raw_input_hashes": truth["input_hashes"],
            },
        },
        "absolute_service_targets": [
            {"metric": name, "target": target, "direction": direction}
            for name, target, direction in contract.ABSOLUTE_SERVICE_TARGETS
        ],
        "preservation_gates": [
            {"metric": name, "direction": direction, "bound": bound}
            for name, direction, bound in contract.HOLDOUT_PRESERVATION_GATES
        ],
        "curve": rows,
        "near_lossless_verdict": verdict,
        "continuous_q_verification": {
            f"{entry['q']:.4f}": entry["continuous_q_verification"]
            for entry in measured
        },
        "settings": {
            _q_slug(entry["q"]): {
                "path": f"settings/{_q_slug(entry['q'])}.json",
                "sha256": sha256_file(settings_dir / f"{_q_slug(entry['q'])}.json"),
            }
            for entry in measured
        },
        "q0_baseline": baseline,
        "measured_q_passes": measured,
        "reused_q30_pass": upper,
        "wall_seconds": time.time() - started,
        "frozen_state_unchanged_at_end": True,
    }

    (output / "low_q_validation_curve.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_curve_csv(output / "low_q_validation_curve.csv", rows)
    (output / TERMINAL).write_text(
        f"{TERMINAL} {report['generated_utc']}\n", encoding="utf-8"
    )
    print(json.dumps({
        "terminal": TERMINAL,
        "output": str(output),
        "near_lossless_verdict": verdict,
        "curve": [
            {
                "q": row["q"],
                "retained_cells": row["retained_cells"],
                "framed_payload_ratio": row["framed_payload_ratio"],
                "preservation_gates_passed": row["preservation_gates_passed"],
                "failed_preservation_gates": row["failed_preservation_gates"],
            }
            for row in rows
        ],
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - runner entry point
    raise SystemExit(main())
