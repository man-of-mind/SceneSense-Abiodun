#!/usr/bin/env python3
"""Provable teacher <-> student ROI coordinate alignment.

Both sides pool with ``torchvision.ops.roi_align(..., aligned=True)`` from boxes given
in ONE frame: model-input pixels on the 768x432 canvas. The teacher's transform is
pinned so that its own pixel frame is that same canvas (see ``teacher_v1``), and each
feature level declares the spatial scale implied by its own shape.

The round trip is proved, not assumed. For a feature map whose channel 0 holds the
pixel x-coordinate of each cell centre and channel 1 the pixel y-coordinate,
``roi_align`` with ``aligned=True`` and sampling_ratio ``r`` returns, in output bin
(ph, pw), exactly

    x = x0 + (pw + 0.5) * (x1 - x0) / k
    y = y0 + (ph + 0.5) * (y1 - y0) / k

because bilinear interpolation of a linear coordinate field is exact and the ``r``
sample points inside a bin average to the bin centre. The expression contains no
``spatial_scale``: every level, teacher or student, therefore addresses the SAME pixel
locations for the same box. Any level whose measured deviation exceeds the tolerance
aborts the run with CONTRACT_INVALID; nothing is approximated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
from torchvision.ops import roi_align

ROUND_TRIP_TOLERANCE_PX = 1e-3


def coordinate_feature(
    height: int, width: int, spatial_scale: float, device: torch.device
) -> torch.Tensor:
    """[1,2,H,W] map whose channels hold the pixel (x, y) of each feature-cell centre."""
    stride = 1.0 / float(spatial_scale)
    xs = (torch.arange(int(width), dtype=torch.float64, device=device) + 0.5) * stride
    ys = (torch.arange(int(height), dtype=torch.float64, device=device) + 0.5) * stride
    grid_x = xs.view(1, 1, 1, int(width)).expand(1, 1, int(height), int(width))
    grid_y = ys.view(1, 1, int(height), 1).expand(1, 1, int(height), int(width))
    return torch.cat([grid_x, grid_y], dim=1).to(torch.float32)


def expected_bin_centres(boxes: torch.Tensor, output_size: int, device: torch.device) -> torch.Tensor:
    """[N,2,k,k] analytic bin centres in model-input pixels, independent of any level."""
    x0 = boxes[:, 0].view(-1, 1, 1)
    y0 = boxes[:, 1].view(-1, 1, 1)
    width = (boxes[:, 2] - boxes[:, 0]).view(-1, 1, 1)
    height = (boxes[:, 3] - boxes[:, 1]).view(-1, 1, 1)
    index = (torch.arange(int(output_size), dtype=torch.float32, device=device) + 0.5) / float(output_size)
    centre_x = x0 + width * index.view(1, 1, int(output_size))
    centre_y = y0 + height * index.view(1, int(output_size), 1)
    centre_x = centre_x.expand(-1, int(output_size), -1)
    centre_y = centre_y.expand(-1, -1, int(output_size))
    return torch.stack([centre_x, centre_y], dim=1)


def verify_level(
    *,
    name: str,
    height: int,
    width: int,
    spatial_scale: float,
    boxes: torch.Tensor,
    output_size: int,
    sampling_ratio: int,
    device: torch.device,
    canvas: Sequence[int],
) -> Dict[str, Any]:
    feature = coordinate_feature(height, width, spatial_scale, device)
    pooled = roi_align(feature, [boxes], output_size=output_size, spatial_scale=float(spatial_scale),
                       sampling_ratio=int(sampling_ratio), aligned=True)
    expected = expected_bin_centres(boxes, output_size, device)
    deviation = (pooled - expected).abs()
    canvas_height, canvas_width = int(canvas[0]), int(canvas[1])
    return {
        "level": name,
        "feature_shape": [int(height), int(width)],
        "declared_spatial_scale": float(spatial_scale),
        "declared_stride": 1.0 / float(spatial_scale),
        "width_stride_matches_canvas": abs(int(width) * (1.0 / float(spatial_scale)) - canvas_width) < 1e-9,
        "height_covers_canvas": int(height) * (1.0 / float(spatial_scale)) >= canvas_height,
        "boxes": int(boxes.shape[0]),
        "max_abs_deviation_px": float(deviation.max().item()),
        "mean_abs_deviation_px": float(deviation.mean().item()),
        "round_trip_exact": float(deviation.max().item()) <= ROUND_TRIP_TOLERANCE_PX,
    }


def synthetic_boxes(device: torch.device) -> torch.Tensor:
    """Deliberately awkward boxes: tiny, thin, off-grid, corner-touching, full canvas."""
    return torch.tensor(
        [
            [0.0, 0.0, 768.0, 432.0],
            [1.0, 1.0, 9.0, 25.0],
            [383.5, 215.5, 384.5, 216.5],
            [700.25, 380.75, 767.5, 431.5],
            [17.3, 4.9, 41.7, 92.1],
            [0.5, 0.5, 3.5, 431.5],
            [12.0, 200.0, 756.0, 203.0],
        ],
        dtype=torch.float32, device=device,
    )


def verify_round_trip(
    *,
    levels: Sequence[Dict[str, Any]],
    real_boxes: torch.Tensor,
    output_size: int,
    sampling_ratio: int,
    device: torch.device,
    canvas: Sequence[int] = (432, 768),
) -> Dict[str, Any]:
    """Run the coordinate round trip on synthetic and on real v3.1 person GT boxes."""
    synthetic = synthetic_boxes(device)
    results: Dict[str, Any] = {
        "output_size": int(output_size),
        "sampling_ratio": int(sampling_ratio),
        "aligned": True,
        "canvas_height_width": [int(canvas[0]), int(canvas[1])],
        "tolerance_px": ROUND_TRIP_TOLERANCE_PX,
        "synthetic": [],
        "real": [],
        "synthetic_box_count": int(synthetic.shape[0]),
        "real_box_count": int(real_boxes.shape[0]),
    }
    for level in levels:
        results["synthetic"].append(verify_level(
            name=str(level["name"]), height=int(level["height"]), width=int(level["width"]),
            spatial_scale=float(level["spatial_scale"]), boxes=synthetic,
            output_size=output_size, sampling_ratio=sampling_ratio, device=device, canvas=canvas,
        ))
        if real_boxes.numel():
            results["real"].append(verify_level(
                name=str(level["name"]), height=int(level["height"]), width=int(level["width"]),
                spatial_scale=float(level["spatial_scale"]), boxes=real_boxes,
                output_size=output_size, sampling_ratio=sampling_ratio, device=device, canvas=canvas,
            ))
    checks: List[bool] = [entry["round_trip_exact"] for entry in results["synthetic"]]
    checks += [entry["round_trip_exact"] for entry in results["real"]]
    checks += [entry["width_stride_matches_canvas"] for entry in results["synthetic"]]
    checks += [entry["height_covers_canvas"] for entry in results["synthetic"]]
    results["all_levels_exact"] = bool(checks) and all(checks)
    results["real_boxes_evaluated"] = bool(real_boxes.numel())
    results["pass"] = bool(results["all_levels_exact"] and results["real_boxes_evaluated"])
    return results


def student_roi_embedding(
    low: torch.Tensor, high: torch.Tensor, boxes: List[torch.Tensor],
    *, output_size: int = 7, sampling_ratio: int = 2,
) -> torch.Tensor:
    """ROI-pool the student's native transported features. [N, 1000, k, k].

    ``low`` and ``high`` are exactly the two tensors that cross the split boundary, so
    the distilled representation is a function of the transported bundle and nothing
    else. No teacher tensor, raw RGB, raw radar or metadata is added to that bundle.
    """
    low_scale = float(low.shape[-1]) / 768.0
    high_scale = float(high.shape[-1]) / 768.0
    pooled_low = roi_align(low.float(), boxes, output_size=output_size, spatial_scale=low_scale,
                           sampling_ratio=sampling_ratio, aligned=True)
    pooled_high = roi_align(high.float(), boxes, output_size=output_size, spatial_scale=high_scale,
                            sampling_ratio=sampling_ratio, aligned=True)
    return torch.cat([pooled_low, pooled_high], dim=1)
