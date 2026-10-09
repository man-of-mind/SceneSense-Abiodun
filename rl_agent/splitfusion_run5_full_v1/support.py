"""Full Run-5 action support: all 12 modes, the original registered continuous-q support, no mask.

The actor is the exact original ``ConditionalHybridActor`` built by ``run5_models.build_run5_models``; its
continuous support buffers are the original ``MODELED_SMOKE_SUPPORT`` bounds (the frozen Run-5 actor's support),
and the warm-up is the original 12-mode x 6-bin x 4-sample stratified schedule (288 decisions).
"""
from __future__ import annotations

from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import MODELED_SMOKE_SUPPORT, MODELED_SMOKE_SUPPORT_SHA256

MODE_COUNT = 12
BOUNDS = tuple(tuple(int(v) for v in pair) for pair in MODELED_SMOKE_SUPPORT.mode_q_e4_bounds)
SUPPORT_SHA256 = MODELED_SMOKE_SUPPORT_SHA256
assert len(BOUNDS) == MODE_COUNT


def in_support(mode_id: int, q_e4: int) -> bool:
    return 0 <= mode_id < MODE_COUNT and BOUNDS[mode_id][0] <= q_e4 <= BOUNDS[mode_id][1]
