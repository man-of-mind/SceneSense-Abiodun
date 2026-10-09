"""Versioned backlog-transition correction for the Run-5 modelled collectors (opt-in; old transition selectable).

``BacklogTransitionMixinV1`` overrides only ``advance_once``. Disabled (``backlog_transition=None``) it calls the
inherited ``advance_once`` unchanged, so every record, state, RNG stream and binding digest is the old one. Enabled:
1. the channel is advanced first, exactly as before (no RNG is used by the backlog), giving MCS_{k+1};
2. B_{k+1} is the content of one byte-conserving FIFO at the next decision's live sample instant
   (``transition.next_backlog``): carry-in B_k, the decision frame (RHO x reward-frame UDP bytes) at its moved first
   send A+B+C, the held frame (RHO x held-frame UDP bytes) at that send plus the base chain's measured live hold offset,
   service r(MCS_k) before the held enqueue and r(MCS_{k+1}) after, access delay d0 after an enqueue on an empty queue;
3. the immediate T_v2b heads are unchanged and keep conditioning on the resulting backlog (no extra penalty).
Nothing here is an actor input except the resulting B_{k+1}, observed at k+1 exactly like the live feature.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from rl_agent.splitfusion_run5_full_v1 import collector as FC

from .transition import RHO, RateTableV1, next_backlog

SCHEMA = "splitfusion.run5_backlog_transition.collector_mixin.v1"


class BacklogTransitionV1:
    """Hash-bound rate table + chain timing."""

    def __init__(self, fit_json: Path, timing_json: Path) -> None:
        fit_doc = json.loads(Path(fit_json).read_text())
        self.table = RateTableV1.from_dict(fit_doc["all_fit"])
        timing_text = Path(timing_json).read_text()
        timing = json.loads(timing_text)
        self.chains = timing["chains"]
        self.medians = timing["pool_medians"]
        self.binding_sha256 = hashlib.sha256(json.dumps(
            dict(schema=SCHEMA, rho=RHO, table=self.table.to_dict(),
                 timing=hashlib.sha256(timing_text.encode()).hexdigest()), sort_keys=True).encode()).hexdigest()
        self.flags = {"timing_from_pool_median": 0, "sample_before_held_enqueue": 0}

    def events(self, resolved: dict) -> tuple[list, float, float]:
        chain = resolved["base_chain_id"]
        stage = resolved["stage_ms"]
        first = float(stage["A"]) + float(stage["B"]) + float(stage["C"])
        t = self.chains.get(chain)
        source = chain.split("#", 1)[0]
        if t is None or t["held_first_send"] is None or t["sample"] is None:
            self.flags["timing_from_pool_median"] += 1
            held = first + self.medians[source]["held_offset_after_first_send"]
            sample = self.medians[source]["sample"]
        else:
            held = first + (t["held_first_send"] - t["first_send"])
            sample = t["sample"]
        if sample <= held:
            self.flags["sample_before_held_enqueue"] += 1
        ev = [(first, RHO * int(resolved["wire_reward"])), (held, RHO * int(resolved["wire_held"]))]
        return ev, held, sample

    def next_backlog(self, *, carry: float, resolved: dict, mcs_k: int, mcs_k1: int) -> int:
        ev, held, sample = self.events(resolved)
        return int(round(next_backlog(carry_bytes=float(carry), events=ev, t_sample=sample, mcs_k=mcs_k, mcs_k1=mcs_k1,
                                      t_switch=held, table=self.table)))


class BacklogTransitionMixinV1:
    backlog_transition: Optional[BacklogTransitionV1] = None

    def advance_once(self) -> None:
        bt = self.backlog_transition
        if bt is None:
            return super().advance_once()                 # the old transition, unchanged
        context = self.context
        FC._require(context is not None and context.resolved is not None, "cannot advance an unresolved cycle")
        if context.advanced:
            return
        mcs_k = int(context.prior_ul_mcs)
        self._channel_advance(context)                    # identical channel step and RNG use as the old order
        mcs_k1 = int(self._mcs_current)
        self._backlog_bytes = bt.next_backlog(carry=float(context.backlog_bytes), resolved=context.resolved,
                                              mcs_k=mcs_k, mcs_k1=mcs_k1)
        context.resolved["backlog_next_bytes"] = self._backlog_bytes
        context.advanced = True
        self.previous_context = context
        self.context = self._new_context()


class FullRun5BacklogCollectorV1(BacklogTransitionMixinV1, FC.FullRun5CollectorV1):
    """The deployed actor's environment (binding_01) with the opt-in backlog transition."""

    def __init__(self, *, latency, seed: int, shared_sources, backlog_transition: Optional[BacklogTransitionV1]) -> None:
        self.backlog_transition = backlog_transition
        super().__init__(latency=latency, seed=seed, shared_sources=shared_sources)
        if backlog_transition is not None:
            self.collector_binding_sha256 = hashlib.sha256(json.dumps(
                {"base": self.collector_binding_sha256, "backlog_transition": backlog_transition.binding_sha256},
                sort_keys=True).encode()).hexdigest()
