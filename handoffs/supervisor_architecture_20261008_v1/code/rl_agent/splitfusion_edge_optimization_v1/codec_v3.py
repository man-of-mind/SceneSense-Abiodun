"""Output-identical edge codec with one authoritative reconstructed-C2 check."""

from __future__ import annotations

import torch

from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_ae_v1 import (
    ae_uint8_transport,
    lowbit_transport,
)
from pole_lraspp_multimodal_fusion.object_head_pilot_v1.splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import (
    continuous_q,
    guards,
    uint8_codec,
)
from rl_agent.splitfusion_live_dispatch_v1.registry import DispatchContractError
from rl_agent.splitfusion_live_dispatch_v1.timing import StageRecorder
from rl_agent.splitfusion_live_dispatch_v1.transport import (
    DecodedC2,
    InspectedInnerPayload,
    ProductionSplitCodec,
)


class SingleFiniteCheckProductionSplitCodec(ProductionSplitCodec):
    """Retain the frozen C2 guard while removing its identical second scan.

    The production adapter first calls ``require_frozen_c2`` (which checks the
    entire reconstructed tensor for finiteness) and immediately evaluates the
    same ``torch.isfinite(...).all()`` expression again.  One authoritative
    fail-closed scan is sufficient and returns the same tensor unchanged.
    """

    def decode(
        self,
        inspected: InspectedInnerPayload,
        *,
        decoder: object | None,
        tail_device: torch.device,
        timing: StageRecorder,
    ) -> DecodedC2:
        identity = inspected.identity
        with timing.stage("unpack_dequantize"):
            if inspected.kind == "noae_uint8":
                decoded, q = uint8_codec.decode(inspected.sparse_bytes)
                keep_mask = None
            elif inspected.kind == "ae_uint8":
                decoded, keep_mask, q, _ = ae_uint8_transport.decode_sparse(
                    inspected.sparse_bytes
                )
            elif inspected.kind == "lowbit":
                decoded, keep_mask, q = lowbit_transport.decode_inspected(
                    inspected.parsed
                )
            else:
                raise DispatchContractError(
                    "inspected inner payload kind is unsupported"
                )
            if continuous_q.quantize_q(q).q_e4 != identity.q_e4:
                raise DispatchContractError(
                    "decoded q disagrees with inspected inner header"
                )
        with timing.stage("ae_decode"):
            if identity.family == "noAE":
                if decoder is not None:
                    raise DispatchContractError(
                        "noAE action must not select an AE decoder"
                    )
                reconstructed = decoded.to(tail_device)
            else:
                if decoder is None:
                    raise DispatchContractError(
                        f"preloaded {identity.family} decoder unavailable"
                    )
                if (
                    int(getattr(decoder, "family_id", -1)) != identity.family_id
                    or int(getattr(decoder, "bottleneck", -1))
                    != identity.transported_channels
                    or int(getattr(decoder, "routing_tag", -1))
                    != identity.routing_tag
                ):
                    raise DispatchContractError(
                        "selected decoder identity disagrees with inner header"
                    )
                reconstructed = decoder.decode(  # type: ignore[attr-defined]
                    decoded.to(tail_device), keep_mask.to(tail_device)
                )
        guards.require_frozen_c2(
            reconstructed, what="preloaded edge reconstructed C2"
        )
        if reconstructed.device != tail_device:
            raise DispatchContractError(
                f"reconstructed C2 device {reconstructed.device} != "
                f"tail device {tail_device}"
            )
        return DecodedC2(
            c2=reconstructed, finite=True, device=reconstructed.device
        )
