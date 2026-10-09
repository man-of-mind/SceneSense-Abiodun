"""Restricted action support for the Run-5 pilot: modes 5, 8 and 11 within measured q brackets; others masked.

The Run-5 network is unchanged (22 inputs, 2x128 trunk, 12 logits / 12 conditional means / 12 log-stds, twin
12-mode critics). Two mechanical changes implement the restricted support:

* discrete mask: a finite bias ``MASK_LOGIT`` is added to the logits of masked modes inside ``forward``. Their
  probabilities are exactly 0.0 in float32, so every 12-mode enumeration (soft value, actor objective, entropy)
  receives zero weight and zero gradient from them while every tensor stays finite for the trainer's checks;
* continuous support: the actor's per-mode affine map z -> q uses the pilot brackets instead of the original
  smoke-support bounds. The continuous log density is the original normalized-z density, which does not depend on
  the bracket width, so log-probabilities and the continuous entropy term are unchanged in form.

The brackets are exact measured endpoints (and 7000 inside mode 11's bracket); every value lies inside the frozen
Run-5 actor's original support.
"""
from __future__ import annotations

import hashlib
import json
from types import MappingProxyType
from typing import Optional, Tuple

import torch
from torch import Tensor

from rl_agent.splitfusion_hybrid_sac_run4_v1 import exploration
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import ActorHeads, ConditionalHybridActor, _check_state
from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import MODELED_SMOKE_SUPPORT

SCHEMA = "splitfusion.run5_restricted_pilot.support.v1"
MODE_COUNT = 12
ALLOWED_MODES = (5, 8, 11)
BRACKETS = MappingProxyType({5: (7000, 9800), 8: (5000, 9800), 11: (3000, 8000)})
MEASURED_POINTS = MappingProxyType({5: (7000, 9800), 8: (5000, 9800), 11: (3000, 7000, 8000)})
MASK_LOGIT = -1.0e9
PROFILE = "FAVORABLE_STABLE"
ORIGINAL_BOUNDS = tuple(tuple(pair) for pair in MODELED_SMOKE_SUPPORT.mode_q_e4_bounds)


class SupportError(ValueError):
    pass


def _bounds() -> Tuple[Tuple[int, int], ...]:
    out = []
    for mode in range(MODE_COUNT):
        out.append(tuple(BRACKETS[mode]) if mode in BRACKETS else ORIGINAL_BOUNDS[mode])
    return tuple(out)


PILOT_BOUNDS = _bounds()
SUPPORT_DOCUMENT = MappingProxyType(dict(
    schema=SCHEMA, allowed_modes=list(ALLOWED_MODES), brackets={str(k): list(v) for k, v in BRACKETS.items()},
    measured_points={str(k): list(v) for k, v in MEASURED_POINTS.items()}, mask_logit=MASK_LOGIT, profile=PROFILE,
    masked_mode_bounds_unused="original smoke-support bounds kept only so the 12-mode tensors stay well defined",
    pilot_bounds=[list(p) for p in PILOT_BOUNDS]))
SUPPORT_SHA256 = hashlib.sha256(json.dumps(dict(SUPPORT_DOCUMENT), sort_keys=True,
                                           separators=(",", ":")).encode("ascii")).hexdigest()


def _check_brackets() -> None:
    for mode, (lo, hi) in BRACKETS.items():
        olo, ohi = ORIGINAL_BOUNDS[mode]
        if not (olo <= lo < hi <= ohi):
            raise SupportError(f"mode {mode} bracket {lo}-{hi} escapes the frozen actor support {olo}-{ohi}")
        points = MEASURED_POINTS[mode]
        if points[0] != lo or points[-1] != hi:
            raise SupportError(f"mode {mode} bracket endpoints are not measured points")


_check_brackets()


def in_support(mode_id: int, q_e4: int) -> bool:
    return mode_id in BRACKETS and BRACKETS[mode_id][0] <= q_e4 <= BRACKETS[mode_id][1]


def mask_bias(dtype=torch.float32) -> Tensor:
    bias = torch.full((MODE_COUNT,), MASK_LOGIT, dtype=dtype)
    bias[list(ALLOWED_MODES)] = 0.0
    return bias


class PilotRestrictedActor(ConditionalHybridActor):
    """The Run-5 actor with the pilot's registered support and discrete mask (architecture unchanged)."""

    def __init__(self, config=None) -> None:
        super().__init__(config)
        lower, upper, digest = self._expected_support_buffer_values()
        with torch.no_grad():
            self._support_q_e4_lower.copy_(lower)
            self._support_q_e4_upper.copy_(upper)
            self._support_sha256_bytes.copy_(digest)
        self.register_buffer("_pilot_mask_bias", mask_bias(self.config.dtype), persistent=True)

    @staticmethod
    def _expected_support_buffer_values() -> Tuple[Tensor, Tensor, Tensor]:
        bounds = torch.tensor(PILOT_BOUNDS, dtype=torch.int64)
        digest = torch.tensor(list(bytes.fromhex(SUPPORT_SHA256)), dtype=torch.uint8)
        return bounds[:, 0], bounds[:, 1], digest

    def forward(self, state: Tensor) -> ActorHeads:
        state = _check_state(state, self.config.state_dim)
        if not torch.equal(self._pilot_mask_bias, mask_bias(self._pilot_mask_bias.dtype)):
            raise SupportError("pilot mask buffer differs from the registered mask")
        latent = self.encoder(state)
        log_std = torch.clamp(self.log_std_head(latent), min=self.config.log_std_min, max=self.config.log_std_max)
        return ActorHeads(logits=self.logit_head(latent) + self._pilot_mask_bias, mean=self.mean_head(latent),
                          log_std=log_std)


def warmup_actions(master_seed: int, q_bin_count: int, samples_per_q_bin: int):
    """The original stratified warm-up built on the pilot bounds, restricted to the allowed modes (order kept)."""
    schedule = exploration.StratifiedWarmupSchedule(exploration.WarmupScheduleConfig(
        mode_q_e4_bounds=PILOT_BOUNDS, q_bin_count=q_bin_count, samples_per_q_bin=samples_per_q_bin,
        master_seed=master_seed, support_contract_id=SUPPORT_SHA256))
    actions = [schedule.action_at(i) for i in range(len(schedule))]
    kept = [a for a in actions if a.mode_id in ALLOWED_MODES]
    if len(kept) != len(ALLOWED_MODES) * q_bin_count * samples_per_q_bin:
        raise SupportError("restricted warm-up does not cover every allowed mode/q-bin cell")
    if not all(in_support(a.mode_id, a.q_e4) for a in kept):
        raise SupportError("warm-up action outside the pilot support")
    return tuple(kept), schedule.config.schedule_id
