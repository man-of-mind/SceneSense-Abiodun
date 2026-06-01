from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import carla_split_inference_udp_data_collect as od_collect


class MultimodalLRASPPSplitModel:
    """Backbone/classifier wrapper used by the UDP split path."""

    def __init__(self, model: torch.nn.Module, device: torch.device, input_size: Tuple[int, int]) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.input_width, self.input_height = int(input_size[0]), int(input_size[1])

    def encode(self, tensor: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        features = self.model.backbone(tensor.to(self.device))
        if isinstance(features, torch.Tensor):
            return OrderedDict([("0", features)])
        return OrderedDict((str(name), value) for name, value in features.items())

    def decode_logits(self, features: "OrderedDict[str, torch.Tensor]", output_size: Tuple[int, int]) -> torch.Tensor:
        logits = self.model.classifier(features)
        if isinstance(logits, dict):
            logits = logits["out"]
        if tuple(logits.shape[-2:]) != (self.input_height, self.input_width):
            logits = F.interpolate(logits, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
        if tuple(output_size) != (self.input_height, self.input_width):
            logits = F.interpolate(logits, size=tuple(output_size), mode="bilinear", align_corners=False)
        return logits

    def decode_object_maps(self, features: "OrderedDict[str, torch.Tensor]", output_size: Tuple[int, int]) -> torch.Tensor:
        if not hasattr(self.model, "object_head"):
            raise RuntimeError("The loaded model does not expose learned object-localization heads.")
        high = features.get("high") if isinstance(features, dict) else None
        if high is None:
            high = features.get("out") if isinstance(features, dict) and "out" in features else list(features.values())[-1]
        if bool(getattr(self.model, "fuse_low_into_object_head", False)):
            low = features.get("low") if isinstance(features, dict) else None
            if low is None:
                raise RuntimeError(
                    "Model expects a 'low' backbone feature for the fused object head, "
                    "but it was not present in the deserialized feature dict."
                )
            if tuple(high.shape[-2:]) != tuple(low.shape[-2:]):
                high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
            object_input = torch.cat([low, high], dim=1)
        else:
            object_input = high
        object_maps = self.model.object_head(object_input)
        if tuple(object_maps.shape[-2:]) != (self.input_height, self.input_width):
            object_maps = F.interpolate(object_maps, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
        if tuple(output_size) != (self.input_height, self.input_width):
            object_maps = F.interpolate(object_maps, size=tuple(output_size), mode="bilinear", align_corners=False)
        return object_maps

    def decode_outputs(self, features: "OrderedDict[str, torch.Tensor]", output_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        with torch.inference_mode():
            outputs = {"out": self.decode_logits(features, output_size)}
            if hasattr(self.model, "object_head"):
                outputs["object"] = self.decode_object_maps(features, output_size)
        return outputs

    def decode_mask(self, features: "OrderedDict[str, torch.Tensor]", output_size: Tuple[int, int]) -> np.ndarray:
        with torch.inference_mode():
            logits = self.decode_logits(features, output_size)
        return logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)


def serialize_backbone_features(
    features: "OrderedDict[str, torch.Tensor]",
    transport: "od_collect.TransportConfig",
    feature_codecs: Dict[str, object],
) -> Tuple[Dict[str, object], int]:
    serialized, uncompressed_bytes, _, _ = od_collect.serialize_feature_maps(
        features,
        feature_codecs,
        quantization_mode=transport.quantization_mode,
        per_level_compress_probe=False,
        entropy_coder=transport.make_entropy_coder(),
    )
    return serialized, int(uncompressed_bytes)


def deserialize_backbone_features(
    payload: Dict[str, object],
    *,
    device: torch.device,
    transport: "od_collect.TransportConfig",
    feature_codecs: Dict[str, object],
) -> "OrderedDict[str, torch.Tensor]":
    return od_collect.deserialize_feature_maps(
        payload,
        device,
        batch_size=1,
        feature_codecs=feature_codecs,
        quantization_mode=transport.quantization_mode,
    )
