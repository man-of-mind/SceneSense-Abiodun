"""Single-buffer UE input preparation candidate v2 (opt-in seam, bit-exact to the installed v1 candidate).

``prepare_live_input_fused`` performs exactly the arithmetic of ``splitfusion_ue_input_prep_v1.candidate.
prepare_live_input_bgr2rgb`` (identical validation and messages, cv2 BGR->RGB, PIL bilinear resize to 768x432,
float32 conversion, x/255 then (x-mean)/std in float32, radar channels 3..6, 16 zero rows, one final
``.to(device, float32)``) but writes them into ONE freshly allocated (1,7,448,768) float32 host tensor:
  * drops ``.convert("RGB")`` on an image that ``Image.fromarray`` already created in mode RGB (a pure copy);
  * drops the ``.copy()`` of the freshly converted float32 array;
  * normalizes in place in the output slice (``torch.div(..., out=)``, ``sub_``, ``div_``: the same float32
    element operations in the same order) instead of three full temporaries;
  * copies radar into its slice and zeroes the padding rows instead of ``torch.cat`` + ``F.pad`` (two full copies).
No buffer is reused across calls (each call owns a fresh output), so no cross-thread/process ownership is needed.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from rl_agent.splitfusion_live_parity_repair_v1 import perception as P
from rl_agent.splitfusion_ue_input_prep_v1 import candidate as V1

ROOT = Path(__file__).resolve().parents[2]
V1_SOURCE = "rl_agent/splitfusion_ue_input_prep_v1/candidate.py"
V1_SOURCE_SHA256 = "ecd312ee6a21b5a6d129a99674a118973919dfab2eb0a8b1b536de0cec81bef9"
CANDIDATE_ID = "UE_INPUT_SINGLE_BUFFER_V2"


class CandidateSeamError(RuntimeError):
    pass


def prepare_live_input_fused(frame_bgr: np.ndarray, radar_tensor: np.ndarray,
                             device: torch.device | str) -> torch.Tensor:
    c = P.CONTRACT
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
            or frame_bgr.shape != (c.native_height, c.native_width, 3)):
        raise P.PerceptionParityError("expected native 720x1280 uint8 BGR frame")
    if (not isinstance(radar_tensor, np.ndarray) or radar_tensor.dtype != np.float32
            or radar_tensor.shape != (c.radar_channels, c.content_height, c.content_width)
            or not np.isfinite(radar_tensor).all()):
        raise P.PerceptionParityError("expected finite float32 4x432x768 content radar; rerasterize source points")
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if image.mode != "RGB":
        raise P.PerceptionParityError("expected an RGB image from the 3-channel uint8 frame")
    image = image.resize((c.content_width, c.content_height), Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1)
    out = torch.empty((1, 3 + c.radar_channels, c.content_height + c.bottom_padding, c.content_width), dtype=torch.float32)
    head = out[0, :3, :c.content_height]
    torch.div(rgb, 255.0, out=head)
    head.sub_(torch.tensor(c.mean, dtype=torch.float32).view(3, 1, 1))
    head.div_(torch.tensor(c.std, dtype=torch.float32).view(3, 1, 1))
    out[0, 3:, :c.content_height].copy_(torch.from_numpy(np.ascontiguousarray(radar_tensor)))
    out[0, :, c.content_height:].zero_()
    return out.to(device=device, dtype=torch.float32)


def verify_v1_source() -> str:
    actual = hashlib.sha256((ROOT / V1_SOURCE).read_bytes()).hexdigest()
    if actual != V1_SOURCE_SHA256:
        raise CandidateSeamError("v1 input candidate drifted; re-prove parity before enabling")
    V1.verify_original_source()
    return actual


@contextmanager
def install_fused_candidate(enabled: bool = True):
    """Enter inside ``install_bgr2rgb_candidate()``; swaps the installed v1 candidate for v2 and restores it."""
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
    if not enabled:
        yield {"candidate": CANDIDATE_ID, "enabled": False}
        return
    verify_v1_source()
    if B._prepare_live_input is not V1.prepare_live_input_bgr2rgb:
        raise CandidateSeamError("v1 BGR->RGB candidate is not installed; enter install_bgr2rgb_candidate() first")
    previous = B._prepare_live_input
    B._prepare_live_input = prepare_live_input_fused
    try:
        yield {"candidate": CANDIDATE_ID, "enabled": True, "v1_sha256": V1_SOURCE_SHA256}
    finally:
        if B._prepare_live_input is prepare_live_input_fused:
            B._prepare_live_input = previous
