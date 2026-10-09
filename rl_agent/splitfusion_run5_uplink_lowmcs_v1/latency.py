"""Candidate environment latency: binding_02 pool model with only the stage-E move replaced.

MRO: CandidatePoolLatencyModelV1 -> PoolLatencyModelV1 (unchanged constructor and CONFIRM_SEND gate) ->
_CandidateFullMove (this file: E move) -> FullSupportLatencyModelV1 (ranking, donor roles, C/D/F deltas, draw).

Stage E (last send -> complete reassembly) of base chain b moved to decision d:

    E_new = E_b + (mu_c(d) - mu_c(b)) + (lambda(d)/lambda(b) - 1) * (E_b - mu_c(b) - beta_pool)

then the unchanged floors (E >= the base chain's measured access interval) and the unchanged T_v2b deadline-head
lateness rule (late with probability clip(1 - p(d)/p(b), 0, 1), represented as reassembly no earlier than the
170-ms deadline). Everything else (A, B, C, D, F, H, the concurrent GT path, the serial score channel and its
carry-over) is untouched, so each stage is still counted once and compose() is called unchanged.
With the IDENTITY correction this is bit-identical to PoolLatencyModelV1 (test).
"""
from __future__ import annotations

import math
from dataclasses import replace

from rl_agent.splitfusion_run5_full_v1 import latency as LF
from rl_agent.splitfusion_run5_reward_trials_v1 import latency as LR

from . import model as M

_require = LF._require


class _CandidateFullMove(LF.FullSupportLatencyModelV1):
    correction: M.LowMcsCorrectionV1

    def mu(self, *, backlog_bytes: float, wire_bytes: int, prior_ul_mcs: int):
        p = self.transport.predict(pre_enqueue_backlog_bytes=float(backlog_bytes), wire_bytes=int(wire_bytes),
                                   prior_ul_mcs=int(prior_ul_mcs))
        term = self.correction.term_ms(backlog_bytes=backlog_bytes, wire_bytes=wire_bytes, prior_ul_mcs=prior_ul_mcs)
        return p.conditional_latency_ms + term, p.on_time_probability

    def move(self, chain_id: str, *, ranked: list, mode_id: int, q_e4: int, prior_ul_mcs: int, backlog_bytes: float,
             wire_bytes: int, u: float, distance: float = 0.0) -> LF.FullDrawV1:
        vector, donor = self._with_donor(chain_id, ranked)
        ctx = self.chains[chain_id].context
        deltas = {s: self.delta(s, mode_id, q_e4, ctx.mode_id, ctx.q_e4) for s in ("C", "D", "F")}
        mu_d, p_d = self.mu(backlog_bytes=backlog_bytes, wire_bytes=wire_bytes, prior_ul_mcs=prior_ul_mcs)
        mu_b = self._base_mu[chain_id]
        ratio = (self.correction.spread(backlog_bytes=backlog_bytes, prior_ul_mcs=prior_ul_mcs)
                 / self._base_lambda[chain_id])
        deltas["E"] = (mu_d - mu_b) + (ratio - 1.0) * (vector.E - mu_b - self.beta_pool)
        p_base = self._base_p[chain_id]
        p_late = min(1.0, max(0.0, 1.0 - p_d / p_base))
        late = u < p_late
        stages = {s: getattr(vector, s) + deltas[s] for s in ("C", "D", "E", "F")}
        floors = list(self.correction.flags(backlog_bytes=backlog_bytes, wire_bytes=wire_bytes, prior_ul_mcs=prior_ul_mcs))
        if stages["D"] < 0.0:
            stages["D"], floors = 0.0, floors + ["D_ZERO"]
        access = max(0.0, vector.subsplits["E_access"])
        if stages["E"] < access:
            stages["E"], floors = access, floors + ["E_ACCESS"]
        for s, value in stages.items():
            _require(math.isfinite(value) and value >= 0.0, f"stage {s} became {value!r} for chain {chain_id}")
        if late:
            stages["E"] = max(stages["E"], LF.DEADLINE_MS - (vector.A + vector.B + stages["C"] + stages["D"]))
        moved = replace(vector, chain_id=f"FULL[{chain_id}|m{mode_id}|q{q_e4}]", **stages)
        return LF.FullDrawV1(vector=moved, base_chain_id=chain_id, donor_chain_id=donor, base_action=(ctx.mode_id, ctx.q_e4),
                             delta_ms=deltas, uplink_on_time_decision=p_d, uplink_on_time_base=p_base,
                             uplink_late_probability=p_late, uplink_late=late, scene_distance=distance,
                             floors=tuple(floors))


class CandidatePoolLatencyModelV1(LR.PoolLatencyModelV1, _CandidateFullMove):
    def __init__(self, binding_dir, transport, *, pool: str, correction: M.LowMcsCorrectionV1) -> None:
        super().__init__(binding_dir, transport, pool=pool)
        self.correction = correction
        self._base_mu, self._base_lambda = {}, {}
        for cid in self._ids:
            ctx = self.chains[cid].context
            self._base_mu[cid], _ = self.mu(backlog_bytes=ctx.backlog_bytes, wire_bytes=ctx.wire_bytes,
                                            prior_ul_mcs=ctx.prior_ul_mcs)
            self._base_lambda[cid] = correction.spread(backlog_bytes=ctx.backlog_bytes, prior_ul_mcs=ctx.prior_ul_mcs)
        self.beta_pool = sum(self.vectors[c].E - self._base_mu[c] for c in self._ids) / len(self._ids)
        self.binding_sha256 = LF.EV.canonical({"pool_model": self.binding_sha256, "correction": correction.to_dict()})
