"""Pilot latency model: measured joint stage vectors, the unchanged SERVICE_QUEUE_V2 composition, and registered
action-dependent interpolation between measured q endpoints.

Chain loading, decomposition and composition are the corrected candidate's (``splitfusion_corrected_latency_env_v1``
``calibration._load_source``, ``chains.decompose`` / ``chains.compose``), unchanged. What this module adds:

* source identity: every chain keeps its source id (run); the two m11/q3000 sessions both populate that stratum;
* registered rejections: a chain whose decomposition violates the queue contract is rejected, never repaired; the
  rejected set must equal the binding's registered list exactly;
* the held-out split from ``binding.SPLIT.json`` (TRAIN chains for training, EVAL chains for evaluation);
* channel-consistent selection (``CHAIN_ONLY``, registered pilot assumption): the k nearest chains of the stratum in
  standardized (camera SI, radar P40, pre-action UL MCS) jointly; the measured uplink of the chosen chain is used as
  recorded (no transport-model transfer). A hard MCS window is not used because each stratum holds only ~15-31
  chains concentrated at MCS 25-27; every draw reports its |MCS gap| and whether the decision MCS lies outside the
  stratum's measured MCS range (both summarised in the assumptions record);
* scene support as bound: SI and P40 must lie inside the stratum's measured range (margin 0), else refusal;
* interpolation for q strictly between two measured points of the same mode: ONE index i into the k-nearest lists is
  drawn; the chain at i in the lower and the upper endpoint stratum are paired; the base vector is the chain of the
  nearer endpoint (lower on a tie); only the action-dependent prediction-path stages C (codec -> first send),
  D (send span), E (uplink to reassembly) and F (reassembly -> prediction materialised, incl. edge compute) are
  replaced by (1 - w) * lower + w * upper with w = (q - q_lo) / (q_hi - q_lo); every other stage (state/actor, input,
  prediction hand-off to the service, the whole GT path, score-channel request costs, return, carry-in, BIND
  instant) stays the base chain's. The queue is then recomposed by ``compose``; receipts are never interpolated.
Exactly one ``rng.randrange`` call is made per draw regardless of the action (common random numbers).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from rl_agent.splitfusion_corrected_latency_env_v1 import CANDIDATE_VERSION
from rl_agent.splitfusion_corrected_latency_env_v1 import calibration as CAL
from rl_agent.splitfusion_corrected_latency_env_v1.chains import (COMPOSITION_MODEL, ChainError, FrameChainV1, Ordering,
                                                                    RoleCostsV1, StageVectorV1, UncalibratedTermError,
                                                                    decompose)

from . import support as S

INTERPOLATED_STAGES = ("C", "D", "E", "F")
REGISTERED_REJECTIONS = {
    "v3_a67_m11_q3000#1695": "TARGET request: worker begin stamped 0.88 ms before the service dispatch stamp "
                             "(negative send interval; the only one of 2,880 measured requests)",
}


class PilotLatencyError(ValueError):
    pass


@dataclass(frozen=True)
class PilotDrawV1:
    vector: StageVectorV1
    chain_id: str
    partner_chain_id: Optional[str]
    donor_chain_id: Optional[str]
    weight: Optional[float]
    endpoints: tuple
    scene_distance: float
    mcs_gap: int
    mcs_outside_stratum_range: bool
    stratum_size: int


class PilotLatencyModelV1:
    def __init__(self, binding_dir: Path, split: str) -> None:
        if split not in ("TRAIN", "EVAL"):
            raise PilotLatencyError("split must be TRAIN or EVAL")
        binding_dir = Path(binding_dir)
        document = json.loads((binding_dir / "PILOT_BINDING.json").read_text())
        split_doc = json.loads((binding_dir / "SPLIT.json").read_text())
        if (document.get("schema") != CAL.BINDING_SCHEMA or document.get("composition_model") != COMPOSITION_MODEL
                or document.get("candidate_version") != CANDIDATE_VERSION or document.get("deadline_ms") != CAL.DEADLINE_MS):
            raise PilotLatencyError("binding schema/composition/version/deadline differs")
        identity = document["pipeline_identity"]
        policy = document["policy"]
        if policy["uplink"]["mode"] != "CHAIN_ONLY" or policy["action_match"]["q_tolerance_e4"] != 0:
            raise PilotLatencyError("pilot binding must be CHAIN_ONLY with exact action strata")
        if policy["uplink"].get("chain_selection") != "JOINT_KNN_STANDARDIZED_SI_P40_MCS":
            raise PilotLatencyError("pilot binding must declare the joint SI/P40/MCS chain selection")
        self.k = int(policy["scene_conditioning"]["k"])
        self.margin = float(policy["scene_conditioning"]["support_margin_fraction"])
        chains, rejected = [], {}
        for source in document["sources"]:
            if source["pipeline_identity"] != identity:
                raise PilotLatencyError(f"{source['source_id']}: pipeline identity differs")
            for chain in CAL._load_source(source):
                if chain.context.cold:
                    continue
                chains.append(chain)
        vectors = {}
        for chain in chains:
            try:
                vectors[chain.chain_id] = decompose(chain)
            except ChainError as exc:
                rejected[chain.chain_id] = str(exc)
        if set(rejected) != set(REGISTERED_REJECTIONS):
            raise PilotLatencyError(f"rejected chains {sorted(rejected)} differ from the registered list "
                                    f"{sorted(REGISTERED_REJECTIONS)}")
        self.rejected = rejected
        keep = {cid for cid, part in split_doc["chains"].items() if part == split}
        self.chains = {c.chain_id: c for c in chains if c.chain_id in vectors and c.chain_id in keep}
        self.vectors = {cid: vectors[cid] for cid in self.chains}
        self.split = split
        self.binding_sha256 = CAL.canonical_sha256(document)
        self.split_sha256 = CAL.canonical_sha256(split_doc)
        self.identity = identity
        for mode, points in S.MEASURED_POINTS.items():
            for q in points:
                if len(self.stratum(mode, q)) < self.k:
                    raise PilotLatencyError(f"{split}: stratum m{mode}/q{q} has fewer than k={self.k} chains")

    def stratum(self, mode_id: int, q_e4: int) -> list[FrameChainV1]:
        return sorted((c for c in self.chains.values() if c.context.mode_id == mode_id and c.context.q_e4 == q_e4),
                      key=lambda c: c.chain_id)

    def strata_summary(self) -> dict[str, Any]:
        out = {}
        for mode, points in S.MEASURED_POINTS.items():
            for q in points:
                st = self.stratum(mode, q)
                out[f"m{mode}_q{q}"] = dict(n=len(st), sources=sorted({c.source_id for c in st}),
                                            mcs=[min(c.context.prior_ul_mcs for c in st), max(c.context.prior_ul_mcs for c in st)])
        return out

    # ------------------------------------------------------------------ selection
    def _ranked(self, mode_id: int, q_e4: int, si: float, p40: float, mcs: int) -> tuple[list, int]:
        stratum = self.stratum(mode_id, q_e4)
        sis = [c.context.camera_si for c in stratum]
        p40s = [c.context.radar_p40 for c in stratum]
        for name, values, x in (("camera_si", sis, si), ("radar_p40", p40s, p40)):
            lo, hi = min(values), max(values)
            pad = self.margin * (hi - lo)
            if not (lo - pad <= x <= hi + pad):
                raise UncalibratedTermError("scene_conditioning", f"m{mode_id}/q{q_e4}: {name}={x:.4f} outside "
                                                                  f"measured support [{lo:.4f}, {hi:.4f}]")
        def spread(values):
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            return sd if sd > 0 else 1.0
        mcs_values = [c.context.prior_ul_mcs for c in stratum]
        s_si, s_p, s_m = spread(sis), spread(p40s), spread(mcs_values)
        ranked = sorted((math.sqrt(((c.context.camera_si - si) / s_si) ** 2 + ((c.context.radar_p40 - p40) / s_p) ** 2
                                   + ((c.context.prior_ul_mcs - mcs) / s_m) ** 2), c.chain_id) for c in stratum)
        return ranked, len(stratum)

    def _with_donor(self, mode_id: int, q_e4: int, si: float, p40: float, chain_id: str) -> tuple[StageVectorV1, Optional[str]]:
        """Fill the unmeasured worker-order roles from the nearest same-stratum chain of the opposite order (as bound)."""
        vector = self.vectors[chain_id]
        stratum = self.stratum(mode_id, q_e4)
        sis = [c.context.camera_si for c in stratum]
        p40s = [c.context.radar_p40 for c in stratum]
        sd = lambda v: (math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) or 1.0)
        s_si, s_p = sd(sis), sd(p40s)
        opposite = sorted((math.hypot((c.context.camera_si - si) / s_si, (c.context.radar_p40 - p40) / s_p), c.chain_id)
                          for c in stratum if self.vectors[c.chain_id].realized_ordering is not vector.realized_ordering)
        if not opposite:
            return vector, None
        donor_id = opposite[0][1]
        donor, own = self.vectors[donor_id].costs, vector.costs
        return vector.with_costs(RoleCostsV1(
            target_admit=own.target_admit or donor.target_admit, prediction_score=own.prediction_score or donor.prediction_score,
            prediction_store=own.prediction_store or donor.prediction_store, target_score=own.target_score or donor.target_score,
            donor_chain_id=donor_id)), donor_id

    def draw(self, *, mode_id: int, q_e4: int, profile: Optional[str], camera_si: float, radar_p40: float,
             prior_ul_mcs: int, rng: random.Random) -> PilotDrawV1:
        if profile != S.PROFILE:
            raise UncalibratedTermError("channel_profile", f"profile {profile!r} is not the measured {S.PROFILE}")
        if not S.in_support(mode_id, q_e4):
            raise UncalibratedTermError("pilot_support", f"(mode {mode_id}, q_e4 {q_e4}) outside the pilot support")
        points = S.MEASURED_POINTS[mode_id]
        index = rng.randrange(self.k)                       # the only random draw (CRN across actions)
        if q_e4 in points:
            ranked, size = self._ranked(mode_id, q_e4, camera_si, radar_p40, prior_ul_mcs)
            distance, chain_id = ranked[index]
            vector, donor = self._with_donor(mode_id, q_e4, camera_si, radar_p40, chain_id)
            return PilotDrawV1(vector=vector, chain_id=chain_id, partner_chain_id=None, donor_chain_id=donor, weight=None,
                               endpoints=(q_e4,), scene_distance=distance,
                               mcs_gap=abs(self.chains[chain_id].context.prior_ul_mcs - prior_ul_mcs),
                               mcs_outside_stratum_range=self._outside(mode_id, q_e4, prior_ul_mcs), stratum_size=size)
        lo = max(p for p in points if p < q_e4)
        hi = min(p for p in points if p > q_e4)
        w = (q_e4 - lo) / (hi - lo)
        ranked_lo, size_lo = self._ranked(mode_id, lo, camera_si, radar_p40, prior_ul_mcs)
        ranked_hi, size_hi = self._ranked(mode_id, hi, camera_si, radar_p40, prior_ul_mcs)
        d_lo, id_lo = ranked_lo[index]
        d_hi, id_hi = ranked_hi[index]
        base_q, base_id, base_d, partner_id = (lo, id_lo, d_lo, id_hi) if w <= 0.5 else (hi, id_hi, d_hi, id_lo)
        base, donor = self._with_donor(mode_id, base_q, camera_si, radar_p40, base_id)
        v_lo, v_hi = self.vectors[id_lo], self.vectors[id_hi]
        mixed = {s: (1.0 - w) * getattr(v_lo, s) + w * getattr(v_hi, s) for s in INTERPOLATED_STAGES}
        vector = replace(base, chain_id=f"INTERP[{id_lo}|{id_hi}|w={w:.6f}]", **mixed)
        return PilotDrawV1(vector=vector, chain_id=base_id, partner_chain_id=partner_id, donor_chain_id=donor, weight=w,
                           endpoints=(lo, hi), scene_distance=base_d,
                           mcs_gap=max(abs(self.chains[i].context.prior_ul_mcs - prior_ul_mcs) for i in (id_lo, id_hi)),
                           mcs_outside_stratum_range=self._outside(mode_id, lo, prior_ul_mcs)
                           or self._outside(mode_id, hi, prior_ul_mcs), stratum_size=min(size_lo, size_hi))

    def _outside(self, mode_id: int, q_e4: int, mcs: int) -> bool:
        values = [c.context.prior_ul_mcs for c in self.stratum(mode_id, q_e4)]
        return not (min(values) <= mcs <= max(values))

    def scene_supported(self, si: float, p40: float) -> bool:
        """Inside the SI/P40 support of every measured stratum of this split (the pilot's supported scenes)."""
        for mode, points in S.MEASURED_POINTS.items():
            for q in points:
                st = self.stratum(mode, q)
                if not (min(c.context.camera_si for c in st) <= si <= max(c.context.camera_si for c in st)
                        and min(c.context.radar_p40 for c in st) <= p40 <= max(c.context.radar_p40 for c in st)):
                    return False
        return True
