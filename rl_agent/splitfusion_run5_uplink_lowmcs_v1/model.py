"""Candidate transport mean/spread: T_v2b plus one low-MCS payload-rate term and one low-MCS residual-spread factor.

    mu_c(B, w, m) = mu_v2b(B, w, m) + kappa * max(0, w - W0)/1e6 * max(0, 16 - m)      (B == 0 only)
    lambda(m, B)  = lambda_L if m <= 15 and B == 0 else 1

kappa is milliseconds per MB per MCS index below 16: at zero backlog each extra MB beyond T_v2b's own low-MCS
anchor (6.2 KB) costs kappa*(16 - m) ms more than T_v2b says. At MCS >= 16, or with backlog, mu_c == mu_v2b
exactly. T_v2b itself (both heads, artifact, support checks) is used unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import plan as P

SCHEMA = "splitfusion.run5_uplink_lowmcs_v1.candidate.v1"
UNRESOLVED_BACKLOG = "UNRESOLVED_BACKLOG"
EXTRAPOLATED_PAYLOAD = "EXTRAPOLATED_PAYLOAD"
EXTRAPOLATED_MCS = "EXTRAPOLATED_MCS"


@dataclass(frozen=True)
class LowMcsCorrectionV1:
    kappa_ms_per_mb_mcs: float
    lambda_low: float
    fit_wire_max: int                     # largest wire in the low-MCS fit region
    fit_mcs_min: int
    w0: int = P.W0_BYTES
    mcs_ref: int = P.MCS_REF
    low_max: int = P.LOW_MCS_MAX
    provenance: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if not (self.kappa_ms_per_mb_mcs >= 0.0 and self.lambda_low > 0.0):
            raise ValueError("kappa must be >= 0 and lambda_low > 0")

    def term_ms(self, *, backlog_bytes: float, wire_bytes: int, prior_ul_mcs: int) -> float:
        if backlog_bytes != 0 or prior_ul_mcs >= self.mcs_ref:
            return 0.0
        return self.kappa_ms_per_mb_mcs * max(0, wire_bytes - self.w0) / 1e6 * (self.mcs_ref - prior_ul_mcs)

    def spread(self, *, backlog_bytes: float, prior_ul_mcs: int) -> float:
        return self.lambda_low if (prior_ul_mcs <= self.low_max and backlog_bytes == 0) else 1.0

    def flags(self, *, backlog_bytes: float, wire_bytes: int, prior_ul_mcs: int) -> tuple:
        if prior_ul_mcs > self.low_max:
            return ()
        if backlog_bytes != 0:
            return (UNRESOLVED_BACKLOG,)
        out = ()
        if wire_bytes > self.fit_wire_max:
            out += (EXTRAPOLATED_PAYLOAD,)
        if prior_ul_mcs < self.fit_mcs_min:
            out += (EXTRAPOLATED_MCS,)
        return out

    def to_dict(self) -> dict:
        return dict(schema=SCHEMA, kappa_ms_per_mb_mcs=self.kappa_ms_per_mb_mcs, lambda_low=self.lambda_low,
                    fit_wire_max=self.fit_wire_max, fit_mcs_min=self.fit_mcs_min, w0=self.w0, mcs_ref=self.mcs_ref,
                    low_max=self.low_max, provenance=self.provenance)

    @classmethod
    def load(cls, path: Path) -> "LowMcsCorrectionV1":
        d = json.loads(Path(path).read_text())
        if d.get("schema") != SCHEMA:
            raise ValueError("foreign candidate artifact")
        if (d["w0"], d["mcs_ref"], d["low_max"]) != (P.W0_BYTES, P.MCS_REF, P.LOW_MCS_MAX):
            raise ValueError("candidate constants differ from the frozen plan")
        return cls(d["kappa_ms_per_mb_mcs"], d["lambda_low"], d["fit_wire_max"], d["fit_mcs_min"],
                   provenance=d.get("provenance", {}))


IDENTITY = LowMcsCorrectionV1(0.0, 1.0, fit_wire_max=10**9, fit_mcs_min=0)     # reproduces the old model exactly
