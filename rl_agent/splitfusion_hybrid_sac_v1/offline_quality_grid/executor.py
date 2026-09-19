"""GPU execution engine for the exact offline grid.

Nothing in this module imports torch or a frozen runtime at import time.  The
CLI performs metadata/spec/selection/store checks first, then calls
``execute_grid`` only behind the exact execution token.
"""

from __future__ import annotations

import csv
import hashlib
import math
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .contract import (
    EPISODES,
    FAMILIES,
    Q_E4_GRID,
    UDP_CHUNK_BYTES_INCLUDING_HEADER,
    canonical_sha256,
    datagram_count,
    keep_count,
    mode_inventory,
    repository_root,
    sfd1_accounting_identity,
    sfd1_accounting_stream_id,
    sha256_file,
)
from .schema import finalize_row, row_key

EXECUTE_TOKEN = "SPLITFUSION_EXACT_OFFLINE_GRID_A1A_101376"
EQUIVALENCE_Q_E4 = 5_000


def _natural_key(frame: Mapping[str, Any], mode_id: int, family: str, quantizer: str, q_e4: int) -> str:
    return canonical_sha256(
        {
            "episode_id": frame["episode_id"],
            "sample_id": frame["sample_id"],
            "frame_id": frame["frame_id"],
            "grid_split": frame["grid_split"],
            "mode_id": mode_id,
            "family": family,
            "quantizer": quantizer,
            "q_e4": q_e4,
        }
    )


def _runtime_modules() -> dict[str, Any]:
    """Late imports: this boundary is after every metadata fail-closed gate."""

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as torch_f

    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_person_p025_calibration_v1.runtime import (
        apply_p025_service_policy,
    )
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_service_candidate_v1.runtime import (
        combined_records,
    )
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
        continuous_q,
        uint8_codec,
        uint8_zstd_transport,
    )
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.phase5_common import (
        load_frozen_scorers,
    )
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import phase6_validation as phase6
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.zstd_transport import (
        ZstdWireCodec,
    )
    from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
        ae_phase11b_gpu_qualification as phase11b,
        ae_phase11d_lowbit_validation as phase11d,
        ae_uint8_transport,
        lowbit_transport,
    )
    from phase2_map_sharing.transport import CHUNK_HEADER, chunk_payload
    from rl_agent.splitfusion_live_dispatch_v1.envelope import (
        CONTEXT_FIXED_BYTES,
        CONTEXT_PROTOCOL_VERSION,
        HEADER_BYTES,
        pack_envelope,
        unpack_envelope,
    )
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

    return locals()


def _runtime_preflight_evidence(value: Any, torch: Any) -> Any:
    """Return a JSON-safe, audit-useful copy of the frozen runtime preflight.

    Phase-11D's preflight intentionally returns the CPU checkpoint payloads
    which its loader consumes. Those payloads contain tensors and therefore
    cannot be written as JSON. Execution retains that original object in
    memory, while this function replaces each tensor with an exact structural
    and byte-digest descriptor for durable evidence. It refuses any other
    non-JSON runtime type rather than silently stringifying it.
    """

    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.device.type != "cpu":
            raise RuntimeError(
                "runtime preflight checkpoint tensors must remain CPU-resident"
            )
        contiguous = detached.contiguous()
        payload = contiguous.view(torch.uint8).numpy().tobytes(order="C")
        expected_bytes = contiguous.numel() * contiguous.element_size()
        if len(payload) != expected_bytes:
            raise RuntimeError("runtime preflight tensor byte-count drift")
        return {
            "record": "checkpoint_tensor_evidence_v1",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "device": str(contiguous.device),
            "numel": int(contiguous.numel()),
            "storage_bytes": expected_bytes,
            "value_bytes_sha256": hashlib.sha256(payload).hexdigest(),
            "requires_grad": bool(detached.requires_grad),
        }
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                evidence_key = key
            elif isinstance(key, int) and not isinstance(key, bool):
                # Phase-11D uses exact q_e4 integers as same-family reference
                # keys. Make that conversion explicit instead of relying on
                # json.dumps' implicit and potentially colliding key coercion.
                evidence_key = str(key)
            else:
                raise RuntimeError(
                    "runtime preflight mappings require string or integer keys"
                )
            if evidence_key in output:
                raise RuntimeError(
                    f"runtime preflight key normalization collision: {evidence_key!r}"
                )
            output[evidence_key] = _runtime_preflight_evidence(item, torch)
        return output
    if isinstance(value, (list, tuple)):
        return [_runtime_preflight_evidence(item, torch) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("runtime preflight contains a non-finite float")
        return value
    raise RuntimeError(
        f"runtime preflight contains unsupported evidence type {type(value).__name__}"
    )


def _rehash_bound_runtime_sources(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Close metadata/execution TOCTOU for sources, GT tables and checkpoints."""

    root = repository_root()
    binding = manifest.get("run_binding", {})
    categories = {
        "repository_sources": "source_module_sha256",
        "evaluation_sources": "evaluation_source_sha256",
        "checkpoints": "checkpoint_sha256",
    }
    observed_by_category: dict[str, dict[str, str]] = {}
    for category, field_name in categories.items():
        expected = binding.get(field_name)
        if not isinstance(expected, Mapping) or not expected:
            raise RuntimeError(f"run manifest has no {category} closure")
        observed: dict[str, str] = {}
        for relative, digest in sorted(expected.items()):
            path = (root / str(relative)).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"bound runtime artifact escapes repository: {relative}"
                ) from exc
            current = sha256_file(path)
            if current != digest:
                raise RuntimeError(
                    f"bound {category} artifact changed after preflight: {relative}"
                )
            observed[str(relative)] = current
        observed_by_category[category] = observed
    total = sum(len(values) for values in observed_by_category.values())
    return {
        "status": "PASS_ALL_BOUND_RUNTIME_INPUTS_REHASHED_BEFORE_MODEL_LOAD",
        "files": total,
        "category_counts": {
            category: len(values)
            for category, values in observed_by_category.items()
        },
        "runtime_input_binding_sha256": canonical_sha256(observed_by_category),
    }


def _rehash_selection_global_sources(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash raw truth/episode tables immediately before they are loaded."""

    root = repository_root()
    expected = selection.get("source_global_sha256")
    if not isinstance(expected, Mapping):
        raise RuntimeError("selection has no global raw-source closure")
    names = (
        "manifest.csv", "metadata.json", "resolved_config.json", "object_boxes.csv",
        "object_visibility.csv", "depth_frames.csv",
    )
    observed: dict[str, dict[str, str]] = {}
    for episode in EPISODES:
        declared = expected.get(episode.episode_id)
        if not isinstance(declared, Mapping) or set(declared) != set(names):
            raise RuntimeError(
                f"selection global raw-source inventory drift: {episode.episode_id}"
            )
        episode_root = root / episode.root_relpath
        values = {name: sha256_file(episode_root / name) for name in names}
        if values != dict(declared):
            raise RuntimeError(
                f"selection global raw source changed before truth load: {episode.episode_id}"
            )
        observed[episode.episode_id] = values
    return {
        "status": "PASS_SELECTION_GLOBAL_TRUTH_SOURCES_REHASHED_BEFORE_LOAD",
        "files": sum(len(values) for values in observed.values()),
        "source_binding_sha256": canonical_sha256(observed),
    }


def _verify_runtime_input_binding(
    *, frame: Mapping[str, Any], inference: Any, inference_row: Mapping[str, Any]
) -> dict[str, str]:
    """Resolve and hash the exact model inputs immediately before inference."""

    root = repository_root()
    episode_root = root / "data_collection/experiments/route_b_perception_v3" / str(
        frame["episode_id"]
    )
    observed: dict[str, str] = {}
    for field_name in ("rgb_path", "radar_tensor_path"):
        source_relative = str(frame["source_paths"][field_name])
        expected_model_relative = f"{frame['episode_id']}/{source_relative}"
        if str(inference_row[field_name]) != expected_model_relative:
            raise RuntimeError(
                f"runtime/source {field_name} identity drift at {frame['sample_id']}"
            )
        raw_path = (episode_root / source_relative).resolve(strict=True)
        runtime_path = (Path(inference.dataset) / expected_model_relative).resolve(strict=True)
        if runtime_path != raw_path:
            raise RuntimeError(
                f"runtime {field_name} does not resolve to bound raw source at {frame['sample_id']}"
            )
        expected_digest = str(frame["source_sha256"][field_name])
        raw_digest = sha256_file(raw_path)
        runtime_digest = sha256_file(runtime_path)
        if raw_digest != expected_digest or runtime_digest != expected_digest:
            raise RuntimeError(
                f"runtime {field_name} bytes drifted at {frame['sample_id']}"
            )
        observed[field_name] = runtime_digest
    return observed


def _verify_runtime_scoring_binding(
    *, frame: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, str]:
    """Rehash the exact per-frame GT/masks immediately before frozen scoring."""

    root = repository_root()
    sample_id = str(frame["sample_id"])
    runtime_root = Path(runtime["dataset_root"]).resolve(strict=True)
    runtime_relatives = {
        "segmentation_gt_path":
            f"contracts/v010/val/segmentation_masks/{sample_id}.png",
        "object_ignore_mask_path":
            f"contracts/v010/val/object_ignore_masks/{sample_id}.png",
    }
    observed: dict[str, str] = {}
    for field_name, runtime_relative in runtime_relatives.items():
        source_path = (root / str(frame["evaluation_source_paths"][field_name])).resolve(
            strict=True
        )
        runtime_path = (runtime_root / runtime_relative).resolve(strict=True)
        if runtime_path != source_path:
            raise RuntimeError(
                f"runtime scoring {field_name} does not resolve to bound GT at {sample_id}"
            )
        digest = sha256_file(runtime_path)
        if digest != frame["evaluation_source_sha256"][field_name]:
            raise RuntimeError(f"runtime scoring {field_name} bytes drifted at {sample_id}")
        observed[field_name] = digest
    return observed


def _wire_accounting(
    *, modules: Mapping[str, Any], frame: Mapping[str, Any], mode_id: int,
    q_e4: int, inner_payload: bytes,
) -> dict[str, Any]:
    """Use the deployed encoders for exact SFD1-v2 and UDP byte accounting.

    Off-anchor q has no dispatchable SFD1-v2 catalog action.  Such rows use the
    explicit uint32 sentinel registered by :func:`sfd1_accounting_identity`;
    the resulting envelope is a byte-accounting artifact, not a live dynamic
    dispatch claim.
    """

    action_field, identity_status = sfd1_accounting_identity(mode_id, q_e4)
    stream_id = sfd1_accounting_stream_id(str(frame["episode_id"]), mode_id)
    pose = frame["ego_world_pose"]
    sequence_id = int(frame["frame_id"])
    capture_timestamp_ns = int(round(float(frame["timestamp"]) * 1_000_000_000))
    context = modules["build_frame_context_v1"](
        stream_id=stream_id,
        frame_id=int(frame["frame_id"]),
        sequence_id=sequence_id,
        capture_timestamp_ns=capture_timestamp_ns,
        ego_world_x=float(pose["x"]),
        ego_world_y=float(pose["y"]),
        ego_world_z=float(pose["z"]),
        ego_world_pitch=float(pose["pitch"]),
        ego_world_yaw=float(pose["yaw"]),
        ego_world_roll=float(pose["roll"]),
    )
    wire = modules["pack_envelope"](
        inner_payload,
        action_id=action_field,
        sequence_id=sequence_id,
        capture_timestamp_ns=capture_timestamp_ns,
        frame_context=context,
    )
    outer = modules["unpack_envelope"](wire)
    if (
        outer.protocol_version != modules["CONTEXT_PROTOCOL_VERSION"]
        or outer.inner_payload != inner_payload
        or outer.action_id != action_field
        or outer.frame_context != context
    ):
        raise RuntimeError("exact SFD1 accounting envelope failed round-trip")
    datagrams = modules["chunk_payload"](
        wire,
        message_id=int(frame["frame_id"]),
        chunk_bytes=UDP_CHUNK_BYTES_INCLUDING_HEADER,
    )
    header_bytes = len(datagrams) * int(modules["CHUNK_HEADER"].size)
    udp_application_bytes = sum(len(item) for item in datagrams)
    if udp_application_bytes != len(wire) + header_bytes:
        raise RuntimeError("exact UDP chunk accounting failed reconciliation")
    frame_context_bytes = outer.header_bytes - int(modules["HEADER_BYTES"])
    if frame_context_bytes != int(modules["CONTEXT_FIXED_BYTES"]) + len(
        stream_id.encode("utf-8")
    ):
        raise RuntimeError("exact SFD1 frame-context byte count drift")
    return {
        "scientific_inner_payload_bytes": len(inner_payload),
        "scientific_inner_payload_sha256": hashlib.sha256(inner_payload).hexdigest(),
        "sfd1_protocol_version": int(outer.protocol_version),
        "sfd1_action_id_field": action_field,
        "sfd1_action_identity_status": identity_status,
        "sfd1_stream_id": stream_id,
        "sfd1_common_header_bytes": int(modules["HEADER_BYTES"]),
        "sfd1_frame_context_bytes": frame_context_bytes,
        "sfd1_outer_envelope_bytes": int(outer.header_bytes),
        "total_transmitted_bytes": len(wire),
        "total_transmitted_sha256": hashlib.sha256(wire).hexdigest(),
        "datagram_count": len(datagrams),
        "udp_chunk_bytes_including_header": UDP_CHUNK_BYTES_INCLUDING_HEADER,
        "udp_chunk_header_bytes_per_datagram": int(modules["CHUNK_HEADER"].size),
        "udp_chunk_header_bytes_total": header_bytes,
        "udp_application_bytes": udp_application_bytes,
    }


class SharedFrameTransport:
    """Reuse front/ranker/AE/range intermediates without changing frozen codecs."""

    def __init__(self, modules: Mapping[str, Any], c2: Any, ranker: Any, autoencoders: Mapping[str, Any], device: Any) -> None:
        self.m = modules
        self.c2 = c2
        self.ranker = ranker
        self.autoencoders = autoencoders
        self.device = device
        continuous_q = modules["continuous_q"]
        self.selections: dict[int, Any] = {0: None}
        scores = ranker.score_cells(c2)
        for q_e4 in Q_E4_GRID:
            if q_e4:
                self.selections[q_e4] = continuous_q.select_cells(scores, q_e4 / 10_000.0)
        self.features = {"noAE": c2}
        for family, autoencoder in autoencoders.items():
            self.features[family] = autoencoder.encode(c2).detach()
        self.noae_uint8 = modules["uint8_codec"].prepare(c2)
        self.ae_uint8 = {
            name: modules["ae_uint8_transport"].prepare(feature)
            for name, feature in self.features.items()
            if name != "noAE"
        }
        self.lowbit = {}
        for name, feature in self.features.items():
            if name == "noAE":
                family_id, routing_tag = 0, 0
            else:
                family_id = int(autoencoders[name].family_id)
                routing_tag = int(autoencoders[name].routing_tag)
            self.lowbit[name] = modules["lowbit_transport"].prepare_feature(
                feature, family_id=family_id, routing_tag=routing_tag
            )
        self.wire = modules["ZstdWireCodec"]()

    def encode_decode(self, family: str, quantizer: str, q_e4: int) -> tuple[Any, bytes]:
        q = q_e4 / 10_000.0
        selection = self.selections[q_e4]
        wire = self.wire
        if quantizer == "UINT8" and family == "noAE":
            sparse = self.m["uint8_codec"].encode(self.noae_uint8, q, selection)
            compressed = wire.compress(sparse.data)
            restored = wire.decompress(compressed.data, expected_bytes=sparse.total_bytes)
            reconstructed, decoded_q = self.m["uint8_codec"].decode(restored)
            if self.m["continuous_q"].quantize_q(decoded_q).q_e4 != q_e4:
                raise RuntimeError("decoded noAE UINT8 q drift")
            return reconstructed.to(self.device), compressed.data
        if quantizer == "UINT8":
            autoencoder = self.autoencoders[family]
            sparse = self.m["ae_uint8_transport"].encode_sparse(
                self.ae_uint8[family], q, selection, routing_tag=autoencoder.routing_tag
            )
            compressed = wire.compress(sparse.data)
            restored = wire.decompress(compressed.data, expected_bytes=sparse.total_bytes)
            latent, mask, decoded_q, parsed = self.m["ae_uint8_transport"].decode_sparse(restored)
            self.m["ae_uint8_transport"].require_family_agreement(parsed, autoencoder)
            if self.m["continuous_q"].quantize_q(decoded_q).q_e4 != q_e4:
                raise RuntimeError("decoded AE UINT8 q drift")
            device = self.m["ae_uint8_transport"].decoder_device(autoencoder)
            return autoencoder.decode(latent.to(device), mask.to(device)), compressed.data
        bits = int(quantizer.removeprefix("UINT"))
        sparse = self.m["lowbit_transport"].encode_sparse(
            self.lowbit[family], q, bits, selection
        )
        compressed = wire.compress(sparse.data)
        restored = wire.decompress(compressed.data, expected_bytes=sparse.total_bytes)
        parsed = self.m["lowbit_transport"].inspect(restored)
        feature, mask, decoded_q = self.m["lowbit_transport"].decode_inspected(parsed)
        if self.m["continuous_q"].quantize_q(decoded_q).q_e4 != q_e4:
            raise RuntimeError("decoded low-bit q drift")
        if family == "noAE":
            return feature.to(self.device), compressed.data
        autoencoder = self.autoencoders[family]
        if (
            parsed.family_id != autoencoder.family_id
            or parsed.routing_tag != autoencoder.routing_tag
            or parsed.channels != autoencoder.bottleneck
        ):
            raise RuntimeError("decoded low-bit AE family/routing drift")
        device = next(autoencoder.parameters()).device
        return autoencoder.decode(feature.to(device), mask.to(device)), compressed.data

    def verify_single_action_equivalence(self) -> dict[str, str]:
        """Bit-compare every optimized family/quantizer path with public encoders."""

        q = EQUIVALENCE_Q_E4 / 10_000.0
        output: dict[str, str] = {}
        for _mode_id, family, quantizer in mode_inventory():
            _reconstructed, shared = self.encode_decode(family, quantizer, EQUIVALENCE_Q_E4)
            single_wire = self.m["ZstdWireCodec"]()
            if family == "noAE" and quantizer == "UINT8":
                result = self.m["uint8_zstd_transport"].encode(
                    self.m["uint8_codec"].prepare(self.c2), self.ranker, q, wire_codec=single_wire
                )
            elif quantizer == "UINT8":
                result = self.m["ae_uint8_transport"].encode_frame(
                    self.c2, self.autoencoders[family], self.ranker, q, wire_codec=single_wire
                )
            else:
                bits = int(quantizer.removeprefix("UINT"))
                if family == "noAE":
                    result = self.m["lowbit_transport"].encode_noae_frame(
                        self.c2, self.ranker, q, bits, wire_codec=single_wire
                    )
                else:
                    result = self.m["lowbit_transport"].encode_ae_frame(
                        self.c2, self.autoencoders[family], self.ranker, q, bits,
                        wire_codec=single_wire,
                    )
            single = result.packet.data
            if shared != single:
                raise RuntimeError(f"shared/single-action wire mismatch: {family}/{quantizer}")
            output[f"{family}/{quantizer}"] = hashlib.sha256(shared).hexdigest()
        return output


def _xy_errors(predictions: list[Mapping[str, Any]], targets: list[Mapping[str, Any]], matched: Mapping[int, int]) -> list[float]:
    return [
        math.hypot(
            float(predictions[pred]["world_x"]) - float(targets[gt]["world_x"]),
            float(predictions[pred]["world_y"]) - float(targets[gt]["world_y"]),
        )
        for pred, gt in sorted(matched.items())
    ]


def _score_one(
    *, modules: Mapping[str, Any], runtime: Mapping[str, Any], row: Mapping[str, Any],
    records: list[dict[str, Any]], labels: Any, scorers: Any, validation_gt: Mapping[str, Any],
    truth: Mapping[str, Any], ignore_cache: dict[str, Any], scratch: Path,
) -> dict[str, Any]:
    """Call the frozen Phase-6 scorer, then retain its per-frame sufficient rows."""

    sample_id = str(row["sample_id"])
    prediction_root = scratch / "prediction"
    segmentation_root = prediction_root / "segmentation"
    segmentation_root.mkdir(parents=True, exist_ok=True)
    detection_path = prediction_root / "detections.csv"
    segmentation_path = segmentation_root / f"{sample_id}.png"
    segmentation_manifest = prediction_root / "segmentation_manifest.csv"
    with detection_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runtime["base"].infer.FIELDS)
        writer.writeheader()
        writer.writerows(records)
    if not modules["cv2"].imwrite(str(segmentation_path), labels):
        raise RuntimeError("failed to stage exact segmentation prediction")
    with segmentation_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "prediction_path", "width", "height"))
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": sample_id,
                "prediction_path": str(Path("segmentation") / segmentation_path.name),
                "width": int(labels.shape[1]),
                "height": int(labels.shape[0]),
            }
        )

    frame_gt = list(validation_gt.get(sample_id, []))
    person_gt = modules["phase6"]._person_only({sample_id: frame_gt})
    subtruth = {
        **truth,
        "frame_ids": [sample_id],
        "qualified_gt": {sample_id: list(truth["qualified_gt"].get(sample_id, []))},
        "structural_gt": {sample_id: list(truth["structural_gt"].get(sample_id, []))},
        "episode_by_sample": {sample_id: truth["episode_by_sample"][sample_id]},
    }
    scored = modules["phase6"].score_validation_pass(
        result={"prediction_root": str(prediction_root)},
        scorers=scorers,
        truth=subtruth,
        experiment=runtime["dataset_root"],
        frame_ids=[sample_id],
        gt={sample_id: frame_gt},
        person_gt=person_gt,
        ignore_cache=ignore_cache,
    )
    confusion = modules["np"].asarray(
        scored["segmentation_detail"]["confusion_matrix"], dtype=modules["np"].int64
    )
    output: dict[str, Any] = {}
    for name, class_id in (("vehicle", 1), ("person", 2)):
        intersection = int(confusion[class_id, class_id])
        gt_pixels = int(confusion[class_id, :].sum())
        pred_pixels = int(confusion[:, class_id].sum())
        output.update(
            {
                f"seg_{name}_gt_pixels": gt_pixels,
                f"seg_{name}_pred_pixels": pred_pixels,
                f"seg_{name}_intersection_pixels": intersection,
                f"seg_{name}_union_pixels": gt_pixels + pred_pixels - intersection,
            }
        )

    threshold = modules["phase6"].contract.VEHICLE_SCORE_THRESHOLD
    vehicle_predictions = [item for item in records if float(item["score"]) >= threshold]
    _used_pred, _used_gt, vehicle_match = scorers.match_frame(vehicle_predictions, frame_gt)
    vehicle_errors = [
        value for (pred, gt), value in zip(
            sorted(vehicle_match.items()), _xy_errors(vehicle_predictions, frame_gt, vehicle_match)
        )
        if str(frame_gt[gt]["class_name"]) == "vehicle"
    ]
    vehicle = scored["vehicle_detail"]
    output.update(
        {
            "loc_vehicle_eligible_gt": int(vehicle["eligible_gt"]),
            "loc_vehicle_tp": int(vehicle["tp"]),
            "loc_vehicle_fp": int(vehicle["fp"]),
            "loc_vehicle_fn": int(vehicle["fn"]),
            "loc_vehicle_ignored_predictions": int(vehicle["ignored_predictions"]),
            "loc_vehicle_matched_xy_errors_m": vehicle_errors,
        }
    )

    people, _count = modules["phase6"].avo.load_person_predictions(detection_path)
    person_predictions = [
        item for item in people.get(sample_id, [])
        if float(item["score"]) >= modules["phase6"].contract.PERSON_SERVICE_SCORE_THRESHOLD
    ]
    qualified = list(subtruth["qualified_gt"][sample_id])
    eligible = [
        item for item in qualified
        if float(item["avo"]) >= modules["phase6"].contract.PERSON_AVO_THRESHOLD
    ]
    person_match, _used = modules["phase6"].avo.greedy_match(person_predictions, eligible)
    person_errors = _xy_errors(person_predictions, eligible, person_match)
    person = scored["person_avo_detail"]["overall"]
    output.update(
        {
            "loc_person_eligible_gt": int(person["observable_gt"]),
            "loc_person_tp": int(person["tp"]),
            "loc_person_fp": int(person["fp"]),
            "loc_person_fn": int(person["fn"]),
            "loc_person_ignored_predictions": int(person["ignored_predictions"]),
            "loc_person_matched_xy_errors_m": person_errors,
        }
    )
    if len(vehicle_errors) != output["loc_vehicle_tp"] or len(person_errors) != output["loc_person_tp"]:
        raise RuntimeError("frozen scorer/matched-error reconciliation failure")
    detection_path.unlink()
    segmentation_path.unlink()
    segmentation_manifest.unlink()
    return output


def execute_grid(
    *, selection: Mapping[str, Any], manifest: Mapping[str, Any], reward_spec: Any,
    store: Any, max_new_rows: int | None = None,
) -> dict[str, Any]:
    """Execute/resume the exact grid.  The caller has already verified all hashes."""

    m = _runtime_modules()
    torch = m["torch"]
    pre_phase11d_source_audit = _rehash_bound_runtime_sources(manifest)
    # The frozen Phase-11 preflight itself precedes the first CUDA query/load.
    frozen_preflight = m["phase11d"].phase11d_preflight()
    runtime_preflight_evidence = _runtime_preflight_evidence(frozen_preflight, torch)
    if not torch.cuda.is_available():
        raise RuntimeError("exact offline grid execution requires the qualified CUDA runtime")
    device = torch.device("cuda:0")
    runtime_environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda_runtime": str(torch.version.cuda),
        "numpy": str(m["np"].__version__),
        "opencv": str(m["cv2"].__version__),
        "cuda_device": str(torch.cuda.get_device_name(device)),
    }
    torch.manual_seed(m["phase11d"].contract.RANKER_INIT_SEED)
    pre_model_load_source_audit = _rehash_bound_runtime_sources(manifest)
    if (
        pre_model_load_source_audit["runtime_input_binding_sha256"]
        != pre_phase11d_source_audit["runtime_input_binding_sha256"]
    ):
        raise RuntimeError("runtime inputs changed across Phase-11D preflight")
    runtime_source_audit = {
        "pre_phase11d_preflight": pre_phase11d_source_audit,
        "immediately_before_model_load": pre_model_load_source_audit,
        "selection_global_truth_sources_immediately_before_load":
            _rehash_selection_global_sources(selection),
    }
    runtime = m["phase11d"]._load_runtime(device)
    autoencoders: dict[str, Any] = {}
    for family in m["phase11d"].FAMILIES:
        if family.bottleneck is None:
            continue
        autoencoders[family.name] = m["phase11b"]._load_selected_autoencoder(
            family.name,
            family.bottleneck,
            m["phase11b"].FROZEN_INPUTS[family.name],
            frozen_preflight["historical_checkpoint_and_source_provenance"]["checkpoint_payloads"][family.name],
            device,
        )
    guards = m["phase11d"].guards
    frozen_snapshots = {
        "model": guards.snapshot_module_state(runtime["model"]),
        "ranker": guards.snapshot_module_state(runtime["ranker"]),
        **{
            f"autoencoder:{name}": guards.snapshot_module_state(autoencoder)
            for name, autoencoder in autoencoders.items()
        },
    }

    def require_frozen_state() -> None:
        guards.require_module_state_unchanged(runtime["model"], frozen_snapshots["model"])
        guards.require_module_state_unchanged(runtime["ranker"], frozen_snapshots["ranker"])
        for name, autoencoder in autoencoders.items():
            guards.require_module_state_unchanged(
                autoencoder, frozen_snapshots[f"autoencoder:{name}"]
            )
    scorers = m["load_frozen_scorers"]()
    gt, _ = scorers.load_gt(runtime["dataset_root"], m["phase11d"].contract.PRIMARY_CONTRACT)
    validation_gt = {sample_id: gt.get(sample_id, []) for sample_id in runtime["frame_ids"]}
    position_by_id = {
        str(row["sample_id"]): index for index, row in enumerate(runtime["inference"].rows)
    }
    completed = store.completed_keys()
    ignore_cache: dict[str, Any] = {}
    episode_hash = {item.episode_id: item.manifest_sha256 for item in EPISODES}
    new_rows = 0
    equivalence: dict[str, str] | None = None
    with tempfile.TemporaryDirectory(prefix="splitfusion_a1a_score_") as temporary:
        scratch = Path(temporary)
        with torch.inference_mode():
            for frame in selection["selected_frames"]:
                pending = [
                    (mode_id, family, quantizer, q_e4)
                    for mode_id, family, quantizer in mode_inventory()
                    for q_e4 in Q_E4_GRID
                    if _natural_key(frame, mode_id, family, quantizer, q_e4) not in completed
                ]
                if not pending and equivalence is not None:
                    continue
                position = position_by_id[str(frame["sample_id"])]
                runtime_row = runtime["inference"].rows[position]
                _verify_runtime_input_binding(
                    frame=frame, inference=runtime["inference"], inference_row=runtime_row
                )
                item = runtime["inference"][position]
                fused, inference_row, calibration = item
                if str(inference_row["sample_id"]) != frame["sample_id"]:
                    raise RuntimeError("selection/runtime sample identity mismatch")
                c2 = runtime["model"].encode_front(fused.unsqueeze(0).to(device)).float()[0]
                shared = SharedFrameTransport(m, c2, runtime["ranker"], autoencoders, device)
                if equivalence is None:
                    equivalence = shared.verify_single_action_equivalence()
                if not pending:
                    continue
                for mode_id, family, quantizer, q_e4 in pending:
                    reconstructed, payload = shared.encode_decode(family, quantizer, q_e4)
                    transport_accounting = _wire_accounting(
                        modules=m, frame=frame, mode_id=mode_id, q_e4=q_e4,
                        inner_payload=payload,
                    )
                    outputs = runtime["model"].decode_tail(reconstructed.unsqueeze(0), dense=False)
                    calibration_gpu = {name: tensor.to(device) for name, tensor in calibration.items()}
                    postprocessed = runtime["model"].postprocess(outputs, [calibration_gpu])
                    served, original_indices = m["apply_p025_service_policy"](
                        {"semantic_logits": outputs["semantic_logits"]}, postprocessed[0]
                    )
                    records = m["combined_records"](
                        runtime["base"], dict(inference_row), served, original_indices
                    )
                    source_hw = (
                        int(inference_row["camera_height"]), int(inference_row["camera_width"])
                    )
                    labels = (
                        m["torch_f"].interpolate(
                            outputs["semantic_logits"].float(), size=source_hw,
                            mode="bilinear", align_corners=False,
                        )
                        .argmax(1)[0]
                        .cpu().numpy().astype(m["np"].uint8)
                    )
                    _verify_runtime_scoring_binding(frame=frame, runtime=runtime)
                    sufficient = _score_one(
                        modules=m, runtime=runtime, row=inference_row, records=records,
                        labels=labels, scorers=scorers, validation_gt=validation_gt,
                        truth=runtime["truth"], ignore_cache=ignore_cache, scratch=scratch,
                    )
                    partial = {
                        "run_binding_sha256": manifest["run_binding_sha256"],
                        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
                        "episode_manifest_sha256": episode_hash[frame["episode_id"]],
                        "source_binding_sha256": frame["source_binding_sha256"],
                        "episode_id": frame["episode_id"],
                        "sample_id": frame["sample_id"],
                        "frame_id": int(frame["frame_id"]),
                        "timestamp": float(frame["timestamp"]),
                        "grid_split": frame["grid_split"],
                        "selection_rank_within_split": int(frame["selection_rank_within_split"]),
                        "inclusion_probability": float(frame["inclusion_probability"]),
                        "sampling_weight": float(frame["sampling_weight"]),
                        "mode_id": mode_id,
                        "family": family,
                        "quantizer": quantizer,
                        "q_e4": q_e4,
                        "keep_count": keep_count(q_e4),
                        "camera_si": frame["camera_si"],
                        "camera_si_valid": frame["camera_si_valid"],
                        "camera_si_status": frame["camera_si_status"],
                        "radar_p40": frame["radar_p40"],
                        "radar_p40_valid": frame["radar_p40_valid"],
                        "radar_p40_status": frame["radar_p40_status"],
                        **transport_accounting,
                        **sufficient,
                    }
                    finalized = finalize_row(partial, reward_spec=reward_spec)
                    if finalized["row_key_sha256"] != _natural_key(
                        frame, mode_id, family, quantizer, q_e4
                    ):
                        raise RuntimeError("row natural-key construction drift")
                    store.insert(finalized)
                    new_rows += 1
                    if max_new_rows is not None and new_rows >= max_new_rows:
                        require_frozen_state()
                        store_audit = store.audit_rows()
                        return {
                            "status": "INTENTIONALLY_INCOMPLETE_BOUNDED_EXECUTION",
                            "new_rows": new_rows,
                            "store_rows": store.row_count,
                            "store_audit": store_audit,
                            "runtime_preflight": runtime_preflight_evidence,
                            "runtime_environment": runtime_environment,
                            "runtime_source_audit": runtime_source_audit,
                            "shared_single_action_equivalence": equivalence,
                        }
    require_frozen_state()
    store_audit = store.audit_rows()
    return {
        "status": "COMPLETE" if store_audit["complete"] else "INCOMPLETE",
        "new_rows": new_rows,
        "store_rows": store.row_count,
        "store_audit": store_audit,
        "runtime_preflight": runtime_preflight_evidence,
        "runtime_environment": runtime_environment,
        "runtime_source_audit": runtime_source_audit,
        "shared_single_action_equivalence": equivalence,
    }
