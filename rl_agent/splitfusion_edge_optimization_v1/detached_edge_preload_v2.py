"""Resident edge preload for the detached v2 optimization candidate."""

from __future__ import annotations

from typing import Any

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_phase11b_gpu_qualification as phase11b,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    guards,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import (
    load_frozen_perception,
)
from rl_agent.splitfusion_live_dispatch_v1.context_tail import (
    ContextualFrozenP025TailAdapter,
)
from rl_agent.splitfusion_live_dispatch_v1.frame_context import StaticCameraRegistry
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _AE, _Ledger
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.transport import ProductionSplitCodec

from .detached_runtime import DetachedPreloadedSplitEdgeRuntime
from .detached_tail_v2 import DetachedOptimizedTailAdapterV2


class DetachedOptimizedEdgeV2:
    """One resident v2 detached edge plus the frozen parity reference."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.registry = SplitActionRegistry.from_runtime_binding()
        model, base, binding = load_frozen_perception(device)
        phase11b.common.freeze(model)
        autoencoders: dict[str, Any] = {}
        for family, _family_id, bottleneck in phase11b.FAMILIES:
            if bottleneck is None:
                continue
            item = phase11b.FROZEN_INPUTS[family]
            payload = torch.load(
                phase11b._repository_path(item["path"]),
                map_location="cpu",
                weights_only=False,
            )
            autoencoders[family] = phase11b._load_selected_autoencoder(
                family, bottleneck, item, payload, device
            )
            del payload
        guards.require_frozen_perception([model, *autoencoders.values()])
        guards.require_eval_mode([model, *autoencoders.values()])
        self.model = model
        self.base = base
        self.perception_binding = binding
        self.autoencoders = autoencoders
        self.ledger = _Ledger()
        self.camera_registry = StaticCameraRegistry.audited()
        wrapped = {
            family: _AE(family, autoencoder, self.ledger)
            for family, autoencoder in autoencoders.items()
        }
        self.tail = DetachedOptimizedTailAdapterV2(
            model=model,
            base=base,
            camera_registry=self.camera_registry,
            device=device,
            ledger=self.ledger,
        )
        self.reference_tail = ContextualFrozenP025TailAdapter(
            model=model,
            base=base,
            camera_registry=self.camera_registry,
            device=device,
            ledger=self.ledger,
        )
        self.runtime = DetachedPreloadedSplitEdgeRuntime(
            self.registry,
            detached_tail=self.tail,
            ae_decoders=wrapped,
            tail_device=device,
            codec=ProductionSplitCodec(),
            prepare_modules=False,
            startup_model_load_operations=4,
            startup_model_construction_operations=4,
            camera_registry=self.camera_registry,
            require_frame_context=True,
        )


def preload_detached_optimized_edge_v2(
    device: torch.device,
) -> DetachedOptimizedEdgeV2:
    return DetachedOptimizedEdgeV2(device)
