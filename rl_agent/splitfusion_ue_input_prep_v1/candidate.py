"""BGR->RGB conversion candidate for the corrected UE input preparation (opt-in seam).

`prepare_live_input_bgr2rgb` is `splitfusion_live_parity_repair_v1.perception.prepare_live_input`
with exactly one line changed:

    original : Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1])).convert("RGB")
    candidate: Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

Both produce an identical C-contiguous uint8 RGB array (a pure channel permutation), so the PIL
bilinear resize, float32 normalisation order, radar validation (including the finite-value
check), 768x432 content geometry, 16 zero rows across all seven channels, and the single final
`.to(device, float32)` placement are unchanged.

`install_bgr2rgb_candidate()` swaps `live_pilot_runtime._prepare_live_input` from the corrected
function to this candidate and restores it on exit. It refuses to install unless the corrected
function is the one currently installed (i.e. it must be entered inside
`splitfusion_live_parity_repair_v1.runtime.install_ue_perception()`), and refuses if the pinned
original source has drifted. Disabled by default: nothing changes unless the seam is entered.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from rl_agent.splitfusion_live_parity_repair_v1 import perception as P

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PERCEPTION = "rl_agent/splitfusion_live_parity_repair_v1/perception.py"
ORIGINAL_PERCEPTION_SHA256 = "e8eb43a89f90809731a3af233239f61492650cdfc23288ea0357768a7158c792"
CANDIDATE_ID = "UE_INPUT_BGR2RGB_CV2_V1"


class CandidateSeamError(RuntimeError):
    pass


def prepare_live_input_bgr2rgb(frame_bgr: np.ndarray, radar_tensor: np.ndarray,
                               device: torch.device | str) -> torch.Tensor:
    """Bit-identical to perception.prepare_live_input; only the BGR->RGB copy differs."""
    c = P.CONTRACT
    if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
            or frame_bgr.shape != (c.native_height, c.native_width, 3)):
        raise P.PerceptionParityError("expected native 720x1280 uint8 BGR frame")
    if (not isinstance(radar_tensor, np.ndarray) or radar_tensor.dtype != np.float32
            or radar_tensor.shape != (c.radar_channels, c.content_height, c.content_width)
            or not np.isfinite(radar_tensor).all()):
        raise P.PerceptionParityError("expected finite float32 4x432x768 content radar; rerasterize source points")
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
    image = image.resize((c.content_width, c.content_height), Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor(c.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(c.std, dtype=torch.float32).view(3, 1, 1)
    rgb = (rgb - mean) / std
    radar = torch.from_numpy(np.ascontiguousarray(radar_tensor))
    content = torch.cat([rgb, radar], dim=0)
    fused = F.pad(content, (0, 0, 0, c.bottom_padding), value=0.0)
    return fused.unsqueeze(0).to(device=device, dtype=torch.float32)


def verify_original_source() -> str:
    actual = hashlib.sha256((ROOT / ORIGINAL_PERCEPTION).read_bytes()).hexdigest()
    if actual != ORIGINAL_PERCEPTION_SHA256:
        raise CandidateSeamError("corrected perception source drifted; re-prove parity before enabling")
    return actual


@contextmanager
def install_bgr2rgb_candidate(enabled: bool = True):
    """Opt-in seam. Enter inside install_ue_perception(); with enabled=False it is a no-op."""
    from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as B
    if not enabled:
        yield {"candidate": CANDIDATE_ID, "enabled": False}
        return
    verify_original_source()
    if B._prepare_live_input is not P.prepare_live_input:
        raise CandidateSeamError("corrected perception input is not installed; enter install_ue_perception() first")
    previous = B._prepare_live_input
    B._prepare_live_input = prepare_live_input_bgr2rgb
    try:
        yield {"candidate": CANDIDATE_ID, "enabled": True, "original_sha256": ORIGINAL_PERCEPTION_SHA256}
    finally:
        if B._prepare_live_input is prepare_live_input_bgr2rgb:
            B._prepare_live_input = previous
