"""Full-support latency model on one named pool of binding_02 (rebound current runtime chains).

Identical to ``splitfusion_run5_full_v1.latency.FullSupportLatencyModelV1`` (subclassed; ``ranked``,
``_with_donor``, ``delta``, ``draw`` and ``move`` are inherited unchanged: each stage once, the concurrent GT path,
the explicit serial score channel with carry-over, T_v2b uplink, common random numbers). Only the constructor
differs: it verifies binding_02 and serves exactly one pool (TRAIN_GCV2 for training; HELDOUT_GCV2 or RETENTION_V1
for evaluation), so measured chains are recomposed exactly and pools are never mixed.

``move`` adds one measured dependency the faster gc_v2 GT path exposes (binding_02 ``confirm_send_gate``): the
service dispatches CONFIRM_SEND no earlier than the UE's last send + delta. If the moved last send binds, the
vector's ``gt_confirm_send_dispatch`` segment grows by the excess (so GT is ready later) and ``bind_end`` -- read only
by compose's guard against this then-unmodelled dependency -- is set to the effective gate instant. Every measured
chain is reproduced exactly at zero delta (the gated TRAIN chain defines delta; all others clear it by >= 50 ms).
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from rl_agent.splitfusion_corrected_latency_env_v1.chains import COMPOSITION_MODEL
from rl_agent.splitfusion_run5_full_v1 import evidence as EVF
from rl_agent.splitfusion_run5_full_v1 import latency as LF
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2

from . import evidence as EV

_require = LF._require
LatencyModelError = LF.LatencyModelError


class PoolLatencyModelV1(LF.FullSupportLatencyModelV1):
    def __init__(self, binding_dir: Path, transport: A2.ProductionTransportModelV2, *, pool: str) -> None:
        binding_dir = Path(binding_dir)
        _require(pool in EV.POOLS, f"unknown pool {pool!r}")
        manifest = json.loads((binding_dir / "OUTPUT_SHA256.json").read_text())
        for name, digest in manifest.items():
            _require(EVF.sha(binding_dir / name) == digest, f"binding file drift: {name}")
        document = json.loads((binding_dir / "LATENCY_EVIDENCE.json").read_text())
        _require(document["schema"] == EV.SCHEMA, "foreign evidence binding")
        _require(document["composition_model"] == COMPOSITION_MODEL and document["deadline_ms_inclusive"] == 170,
                 "composition model or deadline differs")
        self.document = document
        self.pool = pool
        self.confirm_gate_ms = float(document["confirm_send_gate"]["delta_ms"])
        self.structure = EVF.Structure288(document["structure_288"]["per_action"])
        self.scale = {stage: float(v["scale"]) for stage, v in document["transfer"]["stages"].items()}
        _require(set(self.scale) == {"C", "D", "F"}, "transfer scales must cover C, D and F")
        spec = document["pools"][pool]
        section = document["sources"][spec["source"]]
        chains, refused = EVF.load_chains(dict(source_id=section["source_id"], row_filter=None),
                                          rows_entry=section["rows"], frames_entry=section["frames"])
        _require(not refused, "rows refused at load")
        vectors, rejected = EVF.decompose_all(chains)
        _require(rejected == section["registered_rejections"], "rejections differ from the binding")
        wanted = set(spec["chain_ids"])
        _require(wanted <= set(vectors), "pool names a chain that does not decompose")
        self.chains = {c.chain_id: c for c in chains if c.chain_id in wanted}
        self.vectors = {k: v for k, v in vectors.items() if k in wanted}
        _require(len(self.chains) == spec["chains"], "pool size differs from the binding")
        _require(len({v.realized_ordering for v in self.vectors.values()}) == 2,
                 "pool lacks one of the two realized orderings (donor roles)")
        self.binding_sha256 = EVF.canonical({"evidence": EVF.canonical(document), "pool": pool})
        self.transport = transport
        self._ids = sorted(self.chains)
        sis = [self.chains[i].context.camera_si for i in self._ids]
        p40s = [self.chains[i].context.radar_p40 for i in self._ids]
        self._si_sd = LF._sd(sis)
        self._p40_sd = LF._sd(p40s)
        self.si_range = (min(sis), max(sis))
        self.p40_range = (min(p40s), max(p40s))
        self._base_p, self._base_t = {}, {}
        for cid in self._ids:
            ctx = self.chains[cid].context
            prediction = transport.predict(pre_enqueue_backlog_bytes=float(ctx.backlog_bytes),
                                           wire_bytes=int(ctx.wire_bytes), prior_ul_mcs=int(ctx.prior_ul_mcs))
            self._base_p[cid] = prediction.on_time_probability
            self._base_t[cid] = prediction.conditional_latency_ms

    def move(self, chain_id: str, **kwargs) -> LF.FullDrawV1:
        draw = super().move(chain_id, **kwargs)
        v = draw.vector
        last_send = v.A + v.B + v.C + v.D
        if last_send <= v.bind_end:
            return draw                                   # the send cannot gate CONFIRM_SEND (unchanged path)
        segments = dict(v.gt_segments)
        excess = last_send + self.confirm_gate_ms - (v.bind_end + segments["gt_confirm_send_dispatch"])
        floors = draw.floors + ("CONFIRM_SEND_AFTER_BIND",)
        adjusted = v.gt_segments
        if excess > 0.0:
            adjusted = tuple((name, value + excess if name == "gt_confirm_send_dispatch" else value)
                             for name, value in v.gt_segments)
            floors += ("CONFIRM_SEND_GATED",)
        # bind_end is read only by compose's guard against the (previously unmodelled) send dependency; it now
        # carries the effective gate instant, which the CONFIRM_SEND dispatch above already respects.
        vector = replace(v, gt_segments=adjusted, bind_end=last_send)
        return replace(draw, vector=vector, floors=floors)
