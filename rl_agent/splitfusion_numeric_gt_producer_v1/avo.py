"""Numerical AVO from the frozen pedestrian geometry; documents are deferred.

``write_avo_bundle`` interleaves numerics with artifact work: two ``np.save``
copies (4.6 MB) hashed and written, a canonical pre-score JSON document hashed
and written, and the final document hashed and written. Here the numerical part
(``frozen_people_rows`` + ``score_live_avo_people`` on the shared decoded depth)
is separated; ``artifacts.avo_documents`` rebuilds the two original documents
byte-exactly from the same values and the recorded clocks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError, score_live_avo_people


@dataclass(frozen=True)
class NumericAvo:
    frame_id: int
    timestamp: float
    intrinsics: np.ndarray
    rows: tuple          # raw pedestrian rows, original order
    avo: Mapping[str, Any]


def numeric_avo(*, identity: Mapping[str, Any], rgb: Mapping[str, Any], semantic_image: Any, depth_image: Any,
                frozen_world: Any, camera_matrix: np.ndarray, camera_inverse: np.ndarray, depth: np.ndarray,
                sync_validated: bool) -> NumericAvo:
    """Original AVO validators and numerics; the depth array is the shared decode."""
    if not sync_validated:
        raise LiveGtContractError("source synchronization must be validated before AVO")
    frame_id = int(identity["frame_id"])
    for image in (semantic_image, depth_image):
        if int(image.frame) != frame_id or float(image.timestamp) != float(rgb["timestamp"]):
            raise LiveGtContractError("RGB/semantic/depth frame or timestamp mismatch")
        if (int(image.width), int(image.height)) != (int(rgb["width"]), int(rgb["height"])):
            raise LiveGtContractError("RGB/semantic/depth image geometry mismatch")
        if not np.allclose(np.asarray(image.transform.get_matrix()), camera_matrix, rtol=0, atol=1e-5):
            raise LiveGtContractError("RGB/semantic/depth pose mismatch")
    if int(rgb["frame_id"]) != frame_id or not np.allclose(rgb["camera_matrix"], camera_matrix, rtol=0, atol=1e-5):
        raise LiveGtContractError("RGB/ticket identity or pose mismatch")
    width, height = int(depth_image.width), int(depth_image.height)
    if depth.shape != (height, width) or depth.dtype != np.float32:
        raise LiveGtContractError("shared decoded depth does not match the depth image")
    k = LG.native_intrinsics(width, height)
    rows = LG.frozen_people_rows(frozen_world, frame_id=frame_id, timestamp=float(rgb["timestamp"]),
                                 camera_matrix=camera_matrix, camera_inverse=camera_inverse,
                                 intrinsics=k, width=width, height=height)
    avo = score_live_avo_people(frame_id=frame_id, timestamp=float(rgb["timestamp"]),
                                depth_frame_id=int(depth_image.frame), depth_timestamp=float(depth_image.timestamp),
                                depth_m=depth, camera_matrix=camera_matrix, camera_inverse=camera_inverse,
                                intrinsics=k, raw_actor_rows=rows, width=width, height=height)
    return NumericAvo(frame_id=frame_id, timestamp=float(rgb["timestamp"]), intrinsics=k, rows=tuple(rows), avo=avo)
