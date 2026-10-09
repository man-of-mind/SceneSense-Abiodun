"""Matched environments for the backlog pair: the low-MCS retrain (gc_v2) configuration, with
* the backlog transition selectable (``BacklogTransitionMixinV1``; disabled = the existing transition), and
* a DIAGNOSTIC-ONLY latency variant that removes backlog's additional forced-late probability.

Diagnostic NEW_B (``NeutralizedBacklogLateModelV1``): identical to the production candidate move except that the
uplink lateness probability uses T_v2b's on-time probability evaluated at zero backlog (same payload, same MCS):
    p_late_B = clip(1 - p_on_time(backlog=0, wire, MCS) / p_base, 0, 1)
Conditional latency (with the real backlog), the low-MCS term, spread, floors, the CONFIRM_SEND gate, queue evolution,
reward and quality are untouched. At zero backlog it is the production move exactly (test). Never used for training.
"""
from __future__ import annotations

import math
from dataclasses import replace

from rl_agent.splitfusion_run5_full_v1 import latency as LF
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import campaign as LCAMP
from rl_agent.splitfusion_run5_reward_trials_v1 import collector as C
from rl_agent.splitfusion_run5_reward_trials_v1 import latency as LR
from rl_agent.splitfusion_run5_reward_trials_v1 import trainer as TR
from rl_agent.splitfusion_run5_uplink_lowmcs_v1 import latency as LC
from rl_agent.splitfusion_run5_backlog_transition_v1.collector import BacklogTransitionMixinV1, BacklogTransitionV1
from rl_agent.splitfusion_run5_lowmcs_retrain_v1 import registration as LREG

from . import rules as RU

ARM = LREG.ARM


class _NeutralMove(LC._CandidateFullMove):
    def move(self, chain_id, *, ranked, mode_id, q_e4, prior_ul_mcs, backlog_bytes, wire_bytes, u, distance=0.0):
        draw = super().move(chain_id, ranked=ranked, mode_id=mode_id, q_e4=q_e4, prior_ul_mcs=prior_ul_mcs,
                            backlog_bytes=backlog_bytes, wire_bytes=wire_bytes, u=u, distance=distance)
        if backlog_bytes == 0:
            return draw                                               # production move, untouched
        _, p0 = self.mu(backlog_bytes=0.0, wire_bytes=wire_bytes, prior_ul_mcs=prior_ul_mcs)
        p_late = min(1.0, max(0.0, 1.0 - p0 / self._base_p[chain_id]))
        late = u < p_late
        if late == draw.uplink_late:
            return replace(draw, uplink_late_probability=p_late)
        # recompute the E stage without (or with) the forced-late floor; everything else as produced by the move
        vector, _ = self._with_donor(chain_id, ranked)
        v = draw.vector
        e_unforced = vector.E + draw.delta_ms["E"]
        access = max(0.0, vector.subsplits["E_access"])
        e_unforced = max(e_unforced, access)
        e = max(e_unforced, LF.DEADLINE_MS - (vector.A + vector.B + v.C + v.D)) if late else e_unforced
        return replace(draw, vector=replace(v, E=e), uplink_late_probability=p_late, uplink_late=late)


class NeutralizedBacklogLateModelV1(LC.CandidatePoolLatencyModelV1, _NeutralMove):
    """DIAGNOSTIC ONLY. MRO: CONFIRM_SEND gate (PoolLatencyModelV1) -> _NeutralMove -> production low-MCS move."""

    def __init__(self, binding_dir, transport, *, pool, correction) -> None:
        super().__init__(binding_dir, transport, pool=pool, correction=correction)
        self.binding_sha256 = LF.EV.canonical({"diagnostic": "NEUTRALIZED_BACKLOG_FORCED_LATE", "base": self.binding_sha256})


class TrialBacklogCollectorV1(BacklogTransitionMixinV1, C.TrialRun5CollectorV1):
    """The low-MCS retrain collector with the opt-in backlog transition (None -> existing transition, unchanged)."""

    def __init__(self, *, latency, seed, shared_sources, backlog_transition) -> None:
        self.backlog_transition = backlog_transition
        super().__init__(latency=latency, seed=seed, shared_sources=shared_sources, arm=ARM)
        if backlog_transition is not None:
            import hashlib, json
            self.collector_binding_sha256 = hashlib.sha256(json.dumps(
                {"base": self.collector_binding_sha256, "backlog_transition": backlog_transition.binding_sha256},
                sort_keys=True).encode()).hexdigest()


def components(*, split="fit", pool=TR.TRAIN_POOL, neutralized=False, lowmcs=True):
    shared = C.load_shared_sources(ARM, split)
    correction = LCAMP.load_correction()
    if not lowmcs:
        latency = LR.PoolLatencyModelV1(LREG.BINDING_02, shared["model"], pool=pool)
    elif neutralized:
        latency = NeutralizedBacklogLateModelV1(LREG.BINDING_02, shared["model"], pool=pool, correction=correction)
    else:
        latency = LC.CandidatePoolLatencyModelV1(LREG.BINDING_02, shared["model"], pool=pool, correction=correction)
    return shared, latency


def transition():
    return BacklogTransitionV1(RU.FIT_JSON, RU.TIMING_JSON)
