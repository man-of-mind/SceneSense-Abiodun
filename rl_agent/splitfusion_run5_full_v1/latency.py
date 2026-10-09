"""Full-support latency model: action-open -> UE receipt for any (mode, q) in the original support, any channel state.

One decision = one measured base chain (deployed pipeline, scene k-nearest neighbour) whose action- and channel-
dependent stages are moved to the decision's action and channel; every stage is counted exactly once and the
explicit exact-GT service queue (``chains.compose``, SERVICE_QUEUE_V2, unchanged) recombines them:

  A  state build, actor, plan, GT hook ............ base chain (measured)
  B  7-channel input preparation ................... base chain (measured; deployed single-buffer input v2)
  C  codec -> first send ........................... base + scale_C x (288 UE prep(decision) - 288 UE prep(base action))
  D  send span ..................................... base + scale_D x (288 send loop(decision) - 288 send loop(base)),
                                                     floored at 0 (sub-millisecond spans; a shift can cross zero)
  E  uplink, last send -> complete reassembly ...... base + T_v2b(wire, MCS, backlog) - T_v2b(base wire, MCS, backlog),
                                                     never earlier than the base chain's own first-datagram arrival
                                                     (its measured access interval, last send -> first datagram)
  F  reassembly -> prediction (incl. edge compute) . base + scale_F x (288 edge(decision) - 288 edge(base action))
  H  prediction -> service ......................... base chain
  GT path (freeze hand-off, FREEZE, BIND, CONFIRM_SEND, PUBLISH), concurrent with A..H, never added to it
  score channel (serial, PREDICTION priority), carry-in from earlier decisions, return R -> base chain

Uplink lateness: the T_v2b deadline head gives p_on_time for the decision and for the base chain's own transfer
(which completed, being measured). With one uniform u per decision the uplink is LATE when
u < clip(1 - p(decision) / p(base), 0, 1); a late uplink is represented by reassembly completing no earlier than the
170-ms deadline (a lower bound on its true time), so the composed receipt is late and the outcome is TIMEOUT.

Random draws per decision, independent of the action (common random numbers): exactly one ``randrange(k)`` (which
of the k nearest base chains) and one ``random()`` (uplink lateness).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from rl_agent.splitfusion_corrected_latency_env_v1.chains import (COMPOSITION_MODEL, CompositionV1, FrameChainV1,
                                                                    Ordering, RoleCostsV1, StageVectorV1, compose)
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2

from . import evidence as EV
from . import support as S

DEADLINE_MS = 170.0
K_NEAREST = 5


class LatencyModelError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LatencyModelError(message)


@dataclass(frozen=True)
class FullDrawV1:
    vector: StageVectorV1
    base_chain_id: str
    donor_chain_id: Optional[str]
    base_action: tuple
    delta_ms: Mapping[str, float]
    uplink_on_time_decision: float
    uplink_on_time_base: float
    uplink_late_probability: float
    uplink_late: bool
    scene_distance: float
    floors: tuple


class FullSupportLatencyModelV1:
    def __init__(self, binding_dir: Path, transport: A2.ProductionTransportModelV2, *,
                 base_override: Optional[tuple] = None) -> None:
        binding_dir = Path(binding_dir)
        manifest = json.loads((binding_dir / "OUTPUT_SHA256.json").read_text())
        for name, digest in manifest.items():
            _require(EV.sha(binding_dir / name) == digest, f"binding file drift: {name}")
        document = json.loads((binding_dir / "LATENCY_EVIDENCE.json").read_text())
        _require(document["schema"] == "splitfusion.run5_full.latency_evidence.v1", "foreign evidence binding")
        _require(document["composition_model"] == COMPOSITION_MODEL and document["deadline_ms_inclusive"] == 170,
                 "composition model or deadline differs")
        self.document = document
        self.binding_sha256 = EV.canonical(document)
        self.structure = EV.Structure288(document["structure_288"]["per_action"])
        self.scale = {stage: float(v["scale"]) for stage, v in document["transfer"]["stages"].items()}
        _require(set(self.scale) == {"C", "D", "F"}, "transfer scales must cover C, D and F")
        base = document["base"]
        if base_override is None:
            chains, refused = EV.load_chains(EV.BASE_SOURCE, rows_entry=base["rows"], frames_entry=base["frames"])
            _require(not refused, "base rows refused at load")
            vectors, rejected = EV.decompose_all(chains)
            _require(sorted(rejected) == sorted(base["registered_rejections"]), "base rejections differ from the binding")
            self.chains = {c.chain_id: c for c in chains if c.chain_id in vectors}
            _require(len(self.chains) == base["warm_chains"], "base chain count differs from the binding")
        else:                                            # validation only: an explicit pool of measured chains
            chains, vectors = base_override
            self.chains = {c.chain_id: c for c in chains if c.chain_id in vectors}
            self.binding_sha256 = EV.canonical({"evidence": self.binding_sha256, "override": sorted(self.chains)})
        self.vectors = vectors
        self.transport = transport
        self._ids = sorted(self.chains)
        sis = [self.chains[i].context.camera_si for i in self._ids]
        p40s = [self.chains[i].context.radar_p40 for i in self._ids]
        self._si_sd = _sd(sis)
        self._p40_sd = _sd(p40s)
        self.si_range = (min(sis), max(sis))
        self.p40_range = (min(p40s), max(p40s))
        self._base_p = {}
        self._base_t = {}
        for cid in self._ids:
            ctx = self.chains[cid].context
            prediction = transport.predict(pre_enqueue_backlog_bytes=float(ctx.backlog_bytes),
                                           wire_bytes=int(ctx.wire_bytes), prior_ul_mcs=int(ctx.prior_ul_mcs))
            self._base_p[cid] = prediction.on_time_probability
            self._base_t[cid] = prediction.conditional_latency_ms

    # ------------------------------------------------------------------ selection
    def ranked(self, camera_si: float, radar_p40: float) -> list[tuple[float, str]]:
        return sorted((math.hypot((self.chains[i].context.camera_si - camera_si) / self._si_sd,
                                  (self.chains[i].context.radar_p40 - radar_p40) / self._p40_sd), i) for i in self._ids)

    def _with_donor(self, chain_id: str, ranked: list[tuple[float, str]]) -> tuple[StageVectorV1, Optional[str]]:
        """Fill the worker-order roles this chain did not measure from the nearest chain of the opposite order."""
        vector = self.vectors[chain_id]
        for _, other in ranked:
            if self.vectors[other].realized_ordering is not vector.realized_ordering:
                donor, own = self.vectors[other].costs, vector.costs
                return vector.with_costs(RoleCostsV1(
                    target_admit=own.target_admit or donor.target_admit,
                    prediction_score=own.prediction_score or donor.prediction_score,
                    prediction_store=own.prediction_store or donor.prediction_store,
                    target_score=own.target_score or donor.target_score, donor_chain_id=other)), other
        raise LatencyModelError("no opposite-order donor chain in the base pool")

    # ------------------------------------------------------------------ one decision
    def delta(self, stage: str, mode_id: int, q_e4: int, base_mode: int, base_q: int) -> float:
        return self.scale[stage] * (self.structure.value(stage, mode_id, q_e4) - self.structure.value(stage, base_mode, base_q))

    def draw(self, *, mode_id: int, q_e4: int, camera_si: float, radar_p40: float, prior_ul_mcs: int,
             backlog_bytes: float, wire_bytes: int, rng: random.Random) -> FullDrawV1:
        _require(S.in_support(mode_id, q_e4), f"(mode {mode_id}, q_e4 {q_e4}) outside the original Run-5 support")
        index = rng.randrange(K_NEAREST)                        # common random numbers: action independent
        u = rng.random()
        ranked = self.ranked(camera_si, radar_p40)
        distance, chain_id = ranked[index]
        return self.move(chain_id, ranked=ranked, mode_id=mode_id, q_e4=q_e4, prior_ul_mcs=prior_ul_mcs,
                         backlog_bytes=backlog_bytes, wire_bytes=wire_bytes, u=u, distance=distance)

    def move(self, chain_id: str, *, ranked: list, mode_id: int, q_e4: int, prior_ul_mcs: int, backlog_bytes: float,
             wire_bytes: int, u: float, distance: float = 0.0) -> FullDrawV1:
        """Deterministic: move one base chain to the decision's action and channel state."""
        vector, donor = self._with_donor(chain_id, ranked)
        ctx = self.chains[chain_id].context
        deltas = {s: self.delta(s, mode_id, q_e4, ctx.mode_id, ctx.q_e4) for s in ("C", "D", "F")}
        decision = self.transport.predict(pre_enqueue_backlog_bytes=float(backlog_bytes), wire_bytes=int(wire_bytes),
                                          prior_ul_mcs=int(prior_ul_mcs))
        deltas["E"] = decision.conditional_latency_ms - self._base_t[chain_id]
        p_base = self._base_p[chain_id]
        p_late = min(1.0, max(0.0, 1.0 - decision.on_time_probability / p_base))
        late = u < p_late
        stages = {s: getattr(vector, s) + deltas[s] for s in ("C", "D", "E", "F")}
        floors = []
        if stages["D"] < 0.0:
            stages["D"], floors = 0.0, floors + ["D_ZERO"]
        access = max(0.0, vector.subsplits["E_access"])
        if stages["E"] < access:
            stages["E"], floors = access, floors + ["E_ACCESS"]
        for s, value in stages.items():
            _require(math.isfinite(value) and value >= 0.0, f"stage {s} became {value!r} for chain {chain_id}")
        if late:
            stages["E"] = max(stages["E"], DEADLINE_MS - (vector.A + vector.B + stages["C"] + stages["D"]))
        moved = replace(vector, chain_id=f"FULL[{chain_id}|m{mode_id}|q{q_e4}]", **stages)
        return FullDrawV1(vector=moved, base_chain_id=chain_id, donor_chain_id=donor, base_action=(ctx.mode_id, ctx.q_e4),
                          delta_ms=deltas, uplink_on_time_decision=decision.on_time_probability,
                          uplink_on_time_base=p_base, uplink_late_probability=p_late, uplink_late=late,
                          scene_distance=distance, floors=tuple(floors))


def _sd(values) -> float:
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return sd if sd > 0 else 1.0
