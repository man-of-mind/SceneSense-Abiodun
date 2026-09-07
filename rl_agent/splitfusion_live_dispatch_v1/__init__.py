"""Preloaded, SPLIT-only dispatch for the locked 72-action catalog."""

from .edge_runtime import EdgeDispatchResult, PreloadedSplitEdgeRuntime
from .frame_context import (
    FrameContextV1,
    Pose6D,
    StaticCameraRegistry,
    build_frame_context_v1,
)
from .registry import ActionProfile, DispatchContractError, SplitActionRegistry
from .ue_runtime import EncodedSplitFrame, PreloadedSplitUERuntime

__all__ = [
    "ActionProfile",
    "DispatchContractError",
    "EdgeDispatchResult",
    "EncodedSplitFrame",
    "FrameContextV1",
    "Pose6D",
    "PreloadedSplitEdgeRuntime",
    "PreloadedSplitUERuntime",
    "SplitActionRegistry",
    "StaticCameraRegistry",
    "build_frame_context_v1",
]
