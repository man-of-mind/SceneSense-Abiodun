"""Latency rebinding (binding_02): current runtime chains from the gc_v2 validation; nothing new is measured.

Unchanged from ``splitfusion_run5_full_v1/binding_01`` (copied verbatim from its pinned LATENCY_EVIDENCE.json):
the 288-cell action structure (C/D/F per action, profile-pooled medians, linear between the six anchors), the live
transfer scales (C 1.026, D 1.200, F 0.587; fitted on 42 profiler captures that end with valid_03 -- neither gc_v2
nor the retention capture took part, so no held-out data fits them), the T_v2b uplink rule and SERVICE_QUEUE_V2.

Rebound: the action-independent base chains (A, B, H, the concurrent GT path, score channel, return). They now come
from the gc_v2 validation (``splitfusion_run5_full_live_gc_v2/validation_01``: adopted prep-worker GC, numeric
producer v2, single-buffer input v2, CPU partition/thread caps; uninstrumented), decomposed by the unchanged
``chains.decompose``; rejected chains are recorded, never repaired.

CONFIRM_SEND send gate (measured, new in this binding): the early-GT service dispatches CONFIRM_SEND only after the
UE's last send. valid_03 never showed it (GT slower than the send in all 491 chains); with the faster gc_v2 GT path
one TRAIN chain (a 149-ms codec stall) has its last send after the BIND reply, and its CONFIRM_SEND dispatch begins
delta ms after the last send. delta = median over the gated TRAIN_GCV2 chains (send -> CONFIRM_SEND dispatch begin).
``latency.PoolLatencyModelV1`` applies it to the moved vector so the unchanged SERVICE_QUEUE_V2 recomposes it.

Deterministic latency blocks (pure function of the decision index, fixed before any training/evaluation):
    block = decision_index // 25;  HELDOUT if block % 4 == 3, else TRAIN.
Pools:
  TRAIN_GCV2          gc_v2 warm chains in TRAIN blocks          -> the only pool any training run uses
  HELDOUT_GCV2        gc_v2 warm chains in HELDOUT blocks        -> primary evaluation pool (same session)
  RETENTION_V1        all warm chains of the retention capture   -> secondary evaluation pool; labelled
                      (``splitfusion_run5_full_live_retention_v1/validation_01``: the gc_v2 launcher plus the
                      retention-v2 observer), kept separate, never pooled with gc_v2 and never used in training.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from rl_agent.splitfusion_run5_full_v1 import evidence as EV

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SCHEMA = "splitfusion.run5_reward_trials.latency_evidence.v1"
BINDING_01 = ROOT / "experiments/splitfusion_run5_full_v1/binding_01"
BLOCK_SIZE = 25
HELDOUT_BLOCK_MOD = (4, 3)
POOLS = ("TRAIN_GCV2", "HELDOUT_GCV2", "RETENTION_V1")
SOURCES = {
    "gcv2": dict(
        source_id="gcv2",
        rows=ROOT / "experiments/splitfusion_run5_full_live_gc_v2/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
        frames=EV.PROFILER / "20261006_run5_full_live_gc_v2/validation" / EV.CELL / "numeric_gt/service/FRAMES.jsonl",
        row_filter=None, instrumentation="NONE",
        pipeline="gc_v2 validation: adopted prep-worker GC policy + numeric producer v2 + single-buffer input v2 + "
                 "mask-lookup admission + archive capacity repair + CPU partition/thread caps (seed-43 full actor)"),
    "retention": dict(
        source_id="retention",
        rows=ROOT / "experiments/splitfusion_run5_full_live_retention_v1/validation_01_analysis/validation/DECISIONS_validation_b1000.jsonl",
        frames=EV.PROFILER / "20261006_run5_full_live_retention_v1/validation" / EV.CELL / "numeric_gt/service/FRAMES.jsonl",
        row_filter=None, instrumentation="RETENTION_V2_OBSERVER",
        pipeline="gc_v2 launcher + paired-retention v2 observer (stride-5 sample retention; quality-diagnostic run)"),
}


def block_of(decision_index: int) -> str:
    mod, held = HELDOUT_BLOCK_MOD
    return "HELDOUT" if (decision_index // BLOCK_SIZE) % mod == held else "TRAIN"


def decision_index_by_frame(rows_path: Path) -> dict[int, int]:
    rows = [json.loads(l) for l in Path(rows_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {int(r["frame_id"]): int(r["decision_index"]) for r in rows}
    EV._require(len(out) == len(rows), f"{rows_path}: frame ids are not unique")
    return out


def _source_section(key: str) -> tuple[dict[str, Any], dict[str, str]]:
    source = SOURCES[key]
    chains, refused = EV.load_chains(source)
    EV._require(not refused, f"{key}: rows refused at load: {refused[:3]}")
    vectors, rejected = EV.decompose_all(chains)
    index = decision_index_by_frame(source["rows"])
    blocks = {c.chain_id: block_of(index[c.frame_id]) for c in chains if c.chain_id in vectors}
    gated = {}
    for c in chains:
        if c.chain_id in vectors:
            v = vectors[c.chain_id]
            last_send = v.A + v.B + v.C + v.D
            if last_send > v.bind_end:
                gated[c.chain_id] = dict(last_send_ms=last_send, bind_end_ms=v.bind_end,
                                         confirm_dispatch_begin_ms=c.service.requests["CONFIRM_SEND"]["svc_begin"],
                                         send_to_confirm_ms=c.service.requests["CONFIRM_SEND"]["svc_begin"] - last_send)
    section = dict(source_id=source["source_id"], rows=EV.pin(source["rows"]), frames=EV.pin(source["frames"]),
                   instrumentation=source["instrumentation"], pipeline=source["pipeline"],
                   decisions=len(chains), cold=sum(1 for c in chains if c.context.cold), warm_chains=len(vectors),
                   registered_rejections=rejected, send_gated_chains=gated)
    return section, blocks, gated


def build(out: Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    old_path = BINDING_01 / "LATENCY_EVIDENCE.json"
    old_manifest = json.loads((BINDING_01 / "OUTPUT_SHA256.json").read_text())
    EV._require(EV.sha(old_path) == old_manifest["LATENCY_EVIDENCE.json"], "binding_01 drifted")
    old = json.loads(old_path.read_text())
    EV._require(old["schema"] == "splitfusion.run5_full.latency_evidence.v1", "binding_01 schema differs")
    fitted_on = sorted(old["transfer"]["capture_profiles"])
    EV._require(not any("gc_v2" in c or "retention" in c for c in fitted_on),
                "a rebound evaluation capture was used to fit the transfer scales")
    gc_section, gc_blocks, gc_gated = _source_section("gcv2")
    ret_section, ret_blocks, ret_gated = _source_section("retention")
    pools = {
        "TRAIN_GCV2": sorted(c for c, b in gc_blocks.items() if b == "TRAIN"),
        "HELDOUT_GCV2": sorted(c for c, b in gc_blocks.items() if b == "HELDOUT"),
        "RETENTION_V1": sorted(ret_blocks),
    }
    train_gated = [g["send_to_confirm_ms"] for cid, g in gc_gated.items() if cid in set(pools["TRAIN_GCV2"])]
    EV._require(train_gated, "no gated TRAIN chain measures the CONFIRM_SEND send gate")
    document = dict(
        confirm_send_gate=dict(delta_ms=statistics.median(train_gated), fitted_on="TRAIN_GCV2 gated chains",
                               chains=len(train_gated),
                               rule="CONFIRM_SEND dispatch begin >= composed UE last send + delta_ms; when binding, "
                                    "the moved vector's gt_confirm_send_dispatch grows by the excess (GT later)"),
        schema=SCHEMA, deadline_ms_inclusive=170, composition_model=old["composition_model"],
        inherited=dict(binding_01=EV.pin(old_path), sections=["structure_288", "transfer", "uplink"],
                       transfer_fit_captures=len(fitted_on)),
        structure_288=old["structure_288"], transfer=old["transfer"], uplink=old["uplink"],
        sources={"gcv2": gc_section, "retention": ret_section},
        blocks=dict(rule=f"block = decision_index // {BLOCK_SIZE}; HELDOUT if block % {HELDOUT_BLOCK_MOD[0]} == "
                         f"{HELDOUT_BLOCK_MOD[1]} else TRAIN", applies_to="gcv2"),
        pools={name: dict(source="gcv2" if name.endswith("GCV2") else "retention", chain_ids=ids, chains=len(ids))
               for name, ids in pools.items()},
        pool_use=dict(TRAIN_GCV2="training (all arms, all seeds)", HELDOUT_GCV2="primary evaluation only",
                      RETENTION_V1="secondary evaluation only; retention-instrumented, never pooled"))
    (out / "LATENCY_EVIDENCE.json").write_text(json.dumps(document, indent=1, sort_keys=True))
    manifest = {p.name: EV.sha(p) for p in sorted(out.iterdir()) if p.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return dict(evidence_sha256=EV.canonical(document), pools={k: len(v) for k, v in pools.items()},
                rejections={"gcv2": gc_section["registered_rejections"], "retention": ret_section["registered_rejections"]})


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1])), indent=1))
