"""Frozen FCOS input-content geometry, independent of the agent's SI grid.

The original model consumes 768x432 RGB/radar content, followed by 16 zero
bottom rows.  Padding occurs *after* RGB normalization.  A 448-row radar
raster must be regenerated from its source points, never resized or cropped.

No import starts a service or initializes CUDA.  ``prepare_live_input`` may
transfer the completed tensor to a device supplied explicitly by its caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class PerceptionParityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PerceptionInputContract:
    schema: str = "splitfusion.frozen_fcos_content_geometry.v1"
    native_width: int = 1280
    native_height: int = 720
    content_width: int = 768
    content_height: int = 432
    bottom_padding: int = 16
    radar_channels: int = 4
    native_fx: float = 369.5041722813606
    native_fy: float = 369.5041722813606
    native_cx: float = 640.0
    native_cy: float = 360.0
    horizontal_fov_degrees: float = 120.0
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    rgb_resampling: str = "PIL.Image.Resampling.BILINEAR"
    padding_semantics: str = "ZERO_ALL_7_CHANNELS_AFTER_NORMALIZATION"
    radar_semantics: str = "RERASTERIZE_SOURCE_POINTS_ON_CONTENT_GRID_NO_RESIZE"
    scene_si_semantics: str = "UNCHANGED_SEPARATE_768x448_LUMA_CONTRACT"

    @property
    def network_height(self) -> int:
        return self.content_height + self.bottom_padding

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


CONTRACT = PerceptionInputContract()
REFERENCE_DATA_PATH = (
    "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
    "splitfusion_fcos_r50_fpn_p2_p7_v1/data.py"
)
REFERENCE_DATA_SHA256 = "b2530718dfc04e6436baa4a8ea0135cedc901210918e829b8050579948dc9047"
REFERENCE_CONTEXT_PATH = "rl_agent/splitfusion_live_dispatch_v1/frame_context.py"
REFERENCE_CONTEXT_SHA256 = "845c3886f8862ce3f1cd618323b5b7c8baba05b043e5709ff0cd83d36fc33676"


def verify_reference_sources(repository_root: Path) -> dict[str, str]:
    """Startup-only source pin check; never hash files on the frame hot path."""
    verified = {}
    for relative, expected in ((REFERENCE_DATA_PATH, REFERENCE_DATA_SHA256),
                               (REFERENCE_CONTEXT_PATH, REFERENCE_CONTEXT_SHA256)):
        path = Path(repository_root) / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise PerceptionParityError(f"frozen reference source drift: {relative}")
        verified[relative] = actual
    return verified


def content_intrinsics(*, dtype: Any = np.float64) -> np.ndarray:
    """The same independently scaled native camera matrix as model_intrinsic."""
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise PerceptionParityError("content intrinsics require float32 or float64")
    c = CONTRACT
    sx, sy = c.content_width / c.native_width, c.content_height / c.native_height
    matrix = np.asarray(((c.native_fx * sx, 0.0, c.native_cx * sx),
                         (0.0, c.native_fy * sy, c.native_cy * sy),
                         (0.0, 0.0, 1.0)), dtype=dtype)
    matrix.flags.writeable = False
    return matrix


def require_edge_intrinsic_parity(intrinsic: Any) -> None:
    """Validate an already-host-resident edge matrix; never initiate GPU work."""
    if isinstance(intrinsic, torch.Tensor):
        if intrinsic.device.type != "cpu":
            raise PerceptionParityError("inspect edge calibration on CPU before device transfer")
        value = intrinsic.detach().numpy()
    else:
        value = np.asarray(intrinsic)
    if (value.shape != (3, 3) or value.dtype not in (np.float32, np.float64)
            or not np.isfinite(value).all()
            or not np.array_equal(value.astype(np.float32), content_intrinsics(dtype=np.float32))):
        raise PerceptionParityError("edge intrinsic differs from frozen 432-content geometry")


def radar_projection_intrinsics() -> np.ndarray:
    """Literal collection arithmetic, including its float64 rounding order.

    The original radar collector computes focal length directly from content
    width/FOV.  The model separately scales saved native intrinsics and casts
    to float32.  They agree in model float32, but their float64 intermediates
    can differ by one ULP, so preserve both original operation orders.
    """
    c = CONTRACT
    focal = (float(c.content_width) / 2.0) / math.tan(math.radians(c.horizontal_fov_degrees) / 2.0)
    value = np.asarray(((focal, 0.0, c.content_width / 2.0),
                        (0.0, focal, c.content_height / 2.0),
                        (0.0, 0.0, 1.0)), dtype=np.float64)
    require_edge_intrinsic_parity(value)
    value.flags.writeable = False
    return value


def bind_collector_geometry(collector: Any) -> dict[str, Any]:
    """Bind before the collector's original _spawn_sensors call, not after init.

    The retained collector initializes model_size/intrinsics before spawning
    sensors and creates worker threads afterwards.  Its unchanged radar builder
    reads these fields when projecting original detections.  SI/P40 still use
    their independent source functions and are not touched here.
    """
    if getattr(collector, "_perception_input_contract_sha256", None) is not None:
        raise PerceptionParityError("collector geometry already bound")
    if any(hasattr(collector, field) for field in (
            "worker", "segmentation_worker", "evaluation_worker", "feedback_worker")):
        raise PerceptionParityError("geometry must bind before worker construction")
    if getattr(collector, "sensors", None) != []:
        raise PerceptionParityError("geometry must bind before any sensor is spawned")
    old_size = tuple(getattr(collector, "model_size", ()))
    if old_size not in ((768, 448), (768, 432)):
        raise PerceptionParityError("unrecognized collector geometry")
    previous = np.asarray(getattr(collector, "intrinsics", None))
    if previous.shape != (3, 3) or not np.isfinite(previous).all():
        raise PerceptionParityError("missing finite collector calibration")
    expected_previous = radar_projection_intrinsics().copy()
    expected_previous[1, 2] = old_size[1] / 2.0
    if not np.allclose(previous, expected_previous, rtol=0, atol=1e-10):
        raise PerceptionParityError("collector native camera differs from frozen camera")
    collector.model_size = (CONTRACT.content_width, CONTRACT.content_height)
    collector.intrinsics = radar_projection_intrinsics()
    collector._perception_input_contract_sha256 = CONTRACT.sha256
    return {"schema": CONTRACT.schema, "contract_sha256": CONTRACT.sha256,
            "previous_model_size": list(old_size), "model_size": list(collector.model_size),
            "intrinsics": collector.intrinsics.tolist(),
            "scene_descriptors_modified": False, "radar_tracker_modified": False}


def prepare_live_input(frame_bgr: np.ndarray, radar_tensor: np.ndarray,
                       device: torch.device | str) -> torch.Tensor:
    """Prepare exactly the original inference input from equivalent raw pixels.

    This intentionally performs normalization on CPU in the original operation
    order, then one completed-input transfer.  It does not simulate JPEG
    encoding of the old dataset; same-input parity is about decoded pixels.
    """
    c = CONTRACT
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
            or frame_bgr.shape != (c.native_height, c.native_width, 3)):
        raise PerceptionParityError("expected native 720x1280 uint8 BGR frame")
    if (not isinstance(radar_tensor, np.ndarray) or radar_tensor.dtype != np.float32
            or radar_tensor.shape != (c.radar_channels, c.content_height, c.content_width)
            or not np.isfinite(radar_tensor).all()):
        raise PerceptionParityError("expected finite float32 4x432x768 content radar; rerasterize source points")
    image = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1])).convert("RGB")
    image = image.resize((c.content_width, c.content_height), Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor(c.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(c.std, dtype=torch.float32).view(3, 1, 1)
    rgb = (rgb - mean) / std
    radar = torch.from_numpy(np.ascontiguousarray(radar_tensor))
    content = torch.cat([rgb, radar], dim=0)
    fused = F.pad(content, (0, 0, 0, c.bottom_padding), value=0.0)
    return fused.unsqueeze(0).to(device=device, dtype=torch.float32)
