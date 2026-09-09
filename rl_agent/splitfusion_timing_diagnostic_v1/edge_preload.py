"""Resident edge preload that mirrors the qualified Phase-15 preload exactly.

``rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime._preload_edge``
builds the deployed edge: one frozen perception model, three resident AE
decoders, the production zstd/UINT codec and the frozen contextual p025 tail.
This module reproduces that construction verbatim and changes exactly one
thing -- the tail adapter class is the instrumented subclass -- so the
diagnostic measures the deployed path rather than a re-implementation.

It also returns one *unmodified* production adapter over the same resident
model, base and camera registry, so the instrumented tail can be proved
bit-equivalent before any measurement is recorded. No checkpoint is loaded
twice and no module is constructed per frame.
"""

from __future__ import annotations

from typing import Any, Callable

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
from rl_agent.splitfusion_live_dispatch_v1.edge_runtime import PreloadedSplitEdgeRuntime
from rl_agent.splitfusion_live_dispatch_v1.frame_context import StaticCameraRegistry
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _AE, _Ledger
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
from rl_agent.splitfusion_live_dispatch_v1.transport import ProductionSplitCodec
from rl_agent.splitfusion_live_dispatch_v1.ue_runtime import (
    PreloadedSplitUERuntime,
)
from rl_agent.splitfusion_live_dispatch_v1.live_pilot_runtime import _Front, _Ranker

from .instrumented_tail import InstrumentedFrozenP025TailAdapter


class InstrumentedEdge:
    """One resident, instrumented edge with its equivalence reference."""

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
        self.tail = InstrumentedFrozenP025TailAdapter(
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
        self.runtime = PreloadedSplitEdgeRuntime(
            self.registry,
            frozen_p025_tail=self.tail,
            ae_decoders=wrapped,
            tail_device=device,
            codec=ProductionSplitCodec(),
            output_serializer=self.tail.serialize,
            prepare_modules=False,
            startup_model_load_operations=4,
            startup_model_construction_operations=4,
            camera_registry=self.camera_registry,
            require_frame_context=True,
        )


def preload_instrumented_edge(device: torch.device) -> InstrumentedEdge:
    return InstrumentedEdge(device)


def preload_ue(
    device: torch.device,
) -> tuple[PreloadedSplitUERuntime, _Ledger, list[Any], Any, Any]:
    """The deployed UE preload: one front, the epoch-4 ranker, three encoders."""

    registry = SplitActionRegistry.from_runtime_binding()
    model, base, _binding = load_frozen_perception(device)
    phase11b.common.freeze(model)
    ranker = phase11b._load_ranker(device)
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
    guards.require_frozen_perception([model, ranker, *autoencoders.values()])
    guards.require_eval_mode([model, ranker, *autoencoders.values()])
    ledger = _Ledger()
    wrapped = {
        family: _AE(family, autoencoder, ledger)
        for family, autoencoder in autoencoders.items()
    }
    runtime = PreloadedSplitUERuntime(
        registry,
        front=_Front(model, ledger),
        ranker=_Ranker(ranker, ledger),
        ae_encoders=wrapped,
        device=device,
        codec=ProductionSplitCodec(),
        prepare_modules=False,
        startup_model_load_operations=5,
        startup_model_construction_operations=5,
    )
    return runtime, ledger, [model, ranker, *autoencoders.values()], base, registry


ProgressCallback = Callable[[int, int], None]
