#!/usr/bin/env python3
"""Thin Route B adapter over the repository's established split runtime.

No codec, quantizer, entropy coder, AE, UDP, chunking, or worker implementation
is duplicated here. The adapter only namespaces the complete RGB/radar feature
bundle so the existing generic per-level framework can carry it later.
"""

from __future__ import annotations

import inspect
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import carla_split_inference_udp_data_collect as established_runtime  # noqa: E402


GROUP_SEPARATOR = "::"
BOUNDARY_GROUPS = ("rgb_fpn", "radar_fpn")
UDPMessageSocket = established_runtime.UDPMessageSocket
TransportConfig = established_runtime.TransportConfig
FeatureAutoencoder = established_runtime.FeatureAutoencoder


def reconstruct_image_list(
    batch_shape: Tuple[int, ...],
    image_sizes: List[Tuple[int, int]],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
):
    return established_runtime.reconstruct_image_list(
        batch_shape, image_sizes, device, dtype=dtype
    )


def flatten_complete_bundle(bundle: Dict[str, object]) -> "OrderedDict[str, torch.Tensor]":
    """Namespace all RGB and radar pyramid levels for the generic codec."""
    flattened: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for group in BOUNDARY_GROUPS:
        features = bundle[group]
        for level, tensor in features.items():
            flattened[f"{group}{GROUP_SEPARATOR}{level}"] = tensor
    return flattened


def restore_complete_bundle(
    flattened: "OrderedDict[str, torch.Tensor]",
    metadata: Dict[str, object],
) -> Dict[str, object]:
    restored = {group: OrderedDict() for group in BOUNDARY_GROUPS}
    for name, tensor in flattened.items():
        group, level = str(name).split(GROUP_SEPARATOR, 1)
        if group not in restored:
            raise ValueError(f"unexpected boundary group {group!r}")
        restored[group][level] = tensor
    return {
        **restored,
        "image_batch_shape": list(metadata["image_batch_shape"]),
        "image_sizes": [tuple(value) for value in metadata["image_sizes"]],
        "original_image_sizes": [tuple(value) for value in metadata["original_image_sizes"]],
    }


def serialize_complete_bundle(
    bundle: Dict[str, object],
    feature_codecs: Dict[str, object],
    *,
    quantization_mode: str,
    per_level_compress_probe: bool = False,
    entropy_coder=None,
):
    """Delegate serialization to the established heterogeneous-level codec."""
    return established_runtime.serialize_feature_maps(
        flatten_complete_bundle(bundle),
        feature_codecs,
        quantization_mode=quantization_mode,
        per_level_compress_probe=per_level_compress_probe,
        entropy_coder=entropy_coder,
    )


def deserialize_complete_bundle(
    serialized: Dict[str, Dict[str, bytes]],
    metadata: Dict[str, object],
    device: torch.device,
    feature_codecs: Dict[str, object],
    *,
    quantization_mode: str,
) -> Dict[str, object]:
    batch_size = int(metadata["image_batch_shape"][0])
    flattened = established_runtime.deserialize_feature_maps(
        serialized,
        device,
        batch_size=batch_size,
        feature_codecs=feature_codecs,
        quantization_mode=quantization_mode,
    )
    return restore_complete_bundle(flattened, metadata)


def bundle_metadata(bundle: Dict[str, object]) -> Dict[str, object]:
    return {
        "image_batch_shape": list(bundle["image_batch_shape"]),
        "image_sizes": [list(value) for value in bundle["image_sizes"]],
        "original_image_sizes": [list(value) for value in bundle["original_image_sizes"]],
    }


def apply_existing_rank_drop(
    features: "OrderedDict[str, torch.Tensor]", q: float
):
    """Future q hook; delegates to the existing rank-based implementation.

    Clean qualification always passes q=0 and does not call this function.
    """
    from carla_split_inference_udp_segmentation_demo import saliency_drop_masks

    masks, total, per_level = saliency_drop_masks(features, q)
    if not masks:
        return features, total, per_level
    dropped = OrderedDict(
        (name, tensor * masks[name].to(tensor.device, tensor.dtype).unsqueeze(1))
        for name, tensor in features.items()
    )
    return dropped, total, per_level


def runtime_reuse_manifest() -> Dict[str, object]:
    """Machine-readable proof that the package delegates to repository code."""
    return {
        "established_module": str(Path(established_runtime.__file__).resolve()),
        "reused_symbols": {
            name: str(Path(inspect.getsourcefile(symbol)).resolve())
            for name, symbol in {
                "UDPMessageSocket": UDPMessageSocket,
                "TransportConfig": TransportConfig,
                "FeatureAutoencoder": FeatureAutoencoder,
                "serialize_feature_maps": established_runtime.serialize_feature_maps,
                "deserialize_feature_maps": established_runtime.deserialize_feature_maps,
                "reconstruct_image_list": established_runtime.reconstruct_image_list,
                "run_faster_rcnn_back_half": established_runtime.run_faster_rcnn_back_half,
            }.items()
        },
        "future_q_source": str((REPO_ROOT / "carla_split_inference_udp_segmentation_demo.py").resolve()),
        "ae_policy": (
            "reuse the established per-level AE implementation framework only; "
            "do not load old AE weights because the complete RGB/radar boundary level shapes differ"
        ),
        "clean_qualification": "q=0, no quantization, no AE, no UDP execution",
    }

