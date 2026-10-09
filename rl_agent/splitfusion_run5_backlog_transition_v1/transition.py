"""Byte-conserving single-queue RLC backlog transition (pure, deterministic, no RNG).

Timeline in ms after decision k's action open. One FIFO: carry-in at t=0 already in service; each frame enqueues its
RLC bytes at its first send; service drains at r(MCS) bytes/ms whenever the queue is non-empty; an enqueue onto an empty
queue is served only after the access delay d0; the rate uses MCS_k before ``t_switch`` and MCS_{k+1} after it.
Service is counted once, on the one queue: overlapping frames are never drained independently.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Sequence

RHO = 1.0158          # RLC-entering bytes per UDP application byte (contract_01)


class TransitionError(ValueError):
    pass


@dataclass(frozen=True)
class RateTableV1:
    bins: tuple            # ((lo, hi), ...) inclusive MCS ranges, ascending
    rate_bytes_per_ms: tuple
    d0_ms: float
    sparse: tuple = ()

    def rate(self, mcs: int) -> float:
        if mcs is None:
            raise TransitionError("MCS missing")
        m = min(max(int(mcs), self.bins[0][0]), self.bins[-1][1])
        for (lo, hi), r in zip(self.bins, self.rate_bytes_per_ms):
            if lo <= m <= hi:
                return r
        raise TransitionError(f"MCS {mcs} outside the rate bins")

    def to_dict(self) -> dict:
        return dict(bins=[list(b) for b in self.bins], rate_bytes_per_ms=list(self.rate_bytes_per_ms), d0_ms=self.d0_ms,
                    sparse=list(self.sparse))

    @classmethod
    def from_dict(cls, d: dict) -> "RateTableV1":
        return cls(tuple(tuple(b) for b in d["bins"]), tuple(float(r) for r in d["rate_bytes_per_ms"]), float(d["d0_ms"]),
                   tuple(d.get("sparse", ())))


def next_backlog(*, carry_bytes: float, events: Sequence[tuple[float, float]], t_sample: float, mcs_k: int,
                 mcs_k1: int, t_switch: float, table: RateTableV1) -> float:
    """Queue content (bytes) at t_sample. ``events`` = (enqueue time ms, bytes), any order; times may be < 0 only if
    they belong to the carry-in (rejected here)."""
    if carry_bytes < 0 or not math.isfinite(carry_bytes):
        raise TransitionError("carry-in must be finite and >= 0")
    ev = sorted((float(t), float(b)) for t, b in events if t < t_sample)
    for t, b in ev:
        if t < 0 or b < 0:
            raise TransitionError("enqueue before action open or negative bytes")
    q = float(carry_bytes)
    t = 0.0
    serve_from = 0.0                          # carry-in is already in service
    i = 0
    while True:
        nxt_ev = ev[i][0] if i < len(ev) else math.inf
        horizon = min(nxt_ev, t_sample)
        # drain over [t, horizon) in pieces split at serve_from and t_switch
        while t < horizon and q > 0:
            start = max(t, serve_from)
            if start >= horizon:
                t = horizon
                break
            cut = t_switch if start < t_switch < horizon else horizon
            rate = table.rate(mcs_k if start < t_switch else mcs_k1)
            served = rate * (cut - start)
            if served >= q:
                q = 0.0
            else:
                q -= served
            t = cut
        t = horizon
        if horizon >= t_sample or i >= len(ev):
            if horizon >= t_sample:
                return q
        if i < len(ev) and ev[i][0] < t_sample:
            et, eb = ev[i]
            if q <= 0.0:
                serve_from = et + table.d0_ms
            q += eb
            t = et
            i += 1
        else:
            # no more events before the sample: drain to the sample
            while t < t_sample and q > 0:
                start = max(t, serve_from)
                if start >= t_sample:
                    break
                cut = t_switch if start < t_switch < t_sample else t_sample
                rate = table.rate(mcs_k if start < t_switch else mcs_k1)
                q = max(0.0, q - rate * (cut - start))
                t = cut
            return q
