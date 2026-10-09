"""Latency evidence binding for the full-support Run-5 environment (existing evidence only; nothing new is measured).

Three bound evidence groups, each pinned by sha256 and written create-only into one binding directory:

1. BASE CHAINS (deployed pipeline). Every warm policy decision of the pilot actor's 1,000-frame validation
   ``valid_03`` (2026-10-06: single-buffer UE input v2, mask-lookup admission, archive capacity repair; exact-GT
   service queue). Each decision is a measured joint action-open -> UE-receipt chain (B's boundary rows joined to
   the service's own request records) that ``chains.decompose`` inverts into a stage vector A..H, GT path, score
   channel and return. Chains violating the queue contract are rejected, never repaired (registered list).
   These supply every action-independent stage once: state/actor (A), 7-channel input (B), prediction hand-off (H),
   the whole concurrent GT path, score-channel requests, return (R) and post-scoring occupancy.

2. 288-CELL ACTION STRUCTURE. The reconciled 288-cell table (``action_72x4_summary.csv``, 72 actions x 4 profiles).
   Per action, the median over the four profile cells of the per-cell medians of: UE preparation
   (``s3_total_ue_preparation_ms``), datagram send loop (``s4_datagram_send_loop_ms``) and edge processing
   (``s8_total_edge_processing_ms``). These compute stages do not depend on the channel profile (checked in the
   table), so pooling over profiles is a variance choice, not a transfer. Between the six measured q anchors of a
   mode the value is linear in q.

3. LIVE TRANSFER SCALES. The 288 runtime (September) is older than the deployed one, so its action differences are
   rescaled per stage by a factor fitted on every already-collected live profiler capture (2026-10-02 onwards, both
   UE and edge profiles present, live-phase spans only): per (capture, mode, q/500 bin) the median live span is
   regressed on the 288 value with one fixed effect per capture (weights sqrt(frames)), so only within-session
   action differences inform the scale. Spans: ``ue.codec.total_ue_preparation`` (stage C), ``ue.udp_send``
   (stage D), ``edge.process`` (stage F). The span/stage correspondence is checked on valid_03 itself.

Validation chains (not used by the environment): the ten fixed-action screen arms (v2: seven, v3: three), current
pipeline before input v2, for the leave-one-point-out check of the composed model.
"""
from __future__ import annotations

import collections
import csv
import glob
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from rl_agent.splitfusion_corrected_latency_env_v1.chains import (ChainContextV1, ChainError, FrameChainV1,
                                                                    chain_from_boundary_row, decompose, service_records)
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.ue_production_queue_capture_v1 import contract as V1

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]                                     # abiodun/
NEU = ROOT.parent
PROFILER = NEU / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
CELL = "capture/cells/a71__favorable_stable"
TABLE_288 = ROOT / "experiments/splitfusion_288_offline_rl_dataset_v1/20260909_offline_consolidation_v1/action_72x4_summary.csv"
PROFILES = ("FAVORABLE_STABLE", "MID_VARIABLE", "ADVERSE_STABLE", "FADE_RECOVERY")
ANCHORS = (0, 3000, 5000, 7000, 9000, 9800)
STAGE_METRICS = {"C": ("ue.codec.total_ue_preparation", "s3_total_ue_preparation_ms"),
                 "D": ("ue.udp_send", "s4_datagram_send_loop_ms"),
                 "F": ("edge.process", "s8_total_edge_processing_ms")}
SPAN_ROLE = {"ue.codec.total_ue_preparation": "ue", "ue.udp_send": "ue", "edge.process": "edge"}
CAPTURE_MIN_DATE = "20261002"
Q_BIN = 500

BASE_SOURCE = dict(
    source_id="valid03",
    rows=ROOT / "experiments/splitfusion_run5_pilot_live_v1/valid_03/plots/DECISIONS_valid1000_v3.jsonl",
    frames=PROFILER / "20261006_run5_pilot_live_v1/valid_b1000_03" / CELL / "numeric_gt/service/FRAMES.jsonl",
    row_filter=None)
_V2 = ROOT / "experiments/splitfusion_fixed_action_screen_v2/results_combined_01/DECISIONS.jsonl"
_V3 = ROOT / "experiments/splitfusion_fixed_action_screen_v3/results_01/DECISIONS.jsonl"
VALIDATION_SOURCES = tuple(
    dict(source_id=f"{tag}_{arm}", rows=rows, row_filter={"arm": arm},
         frames=PROFILER / campaign / arm / CELL / "numeric_gt/service/FRAMES.jsonl")
    for tag, campaign, rows, arms in (
        ("v2", "20261005_fixed_action_screen_v2", _V2,
         ("arm0_a33", "arm1_a65", "arm2_a17", "arm3_a50", "arm4_a67", "arm5_a53", "arm6_a35")),
        ("v3", "20261005_fixed_action_m11_frontier_v3", _V3,
         ("arm0_m11_q3000_a67", "arm1_m11_q7000_a69", "arm2_m11_q8000_offanchor")))
    for arm in arms)
# Chains whose decomposition violates the queue contract; rejected, never repaired. Must match exactly.
REGISTERED_REJECTIONS = {
    "valid03": ("negative BIND interval", "negative score-channel interval"),
}


class EvidenceError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def rel(path: Path) -> str:
    return os.path.relpath(Path(path).resolve(), NEU)


def pin(path: Path) -> dict[str, str]:
    return {"path": rel(path), "sha256": sha(path)}


def verified(entry: Mapping[str, str]) -> Path:
    path = NEU / entry["path"]
    _require(path.is_file(), f"bound file missing: {path}")
    _require(sha(path) == entry["sha256"], f"bound file drifted: {entry['path']}")
    return path


# --------------------------------------------------------------------------- 288 structure
def table_288(path: Path = TABLE_288) -> dict[int, dict[str, Any]]:
    """Per action: profile-pooled medians of the three compute stages, live payload per profile (cross-check)."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    _require(len(rows) == 288, "288 table is not 288 cells")
    out: dict[int, dict[str, Any]] = {}
    for a in range(72):
        cells = [r for r in rows if int(r["action_id"]) == a]
        _require(sorted(r["network_profile"] for r in cells) == sorted(PROFILES), f"action {a}: profiles incomplete")
        entry = dict(family=cells[0]["family"], quantizer=cells[0]["quantizer"], q_e4=int(cells[0]["q_e4"]),
                     payload_by_profile={r["network_profile"]: float(r["live_scientific_inner_bytes_median"])
                                         for r in cells})
        for stage, (_, metric) in STAGE_METRICS.items():
            values = [float(r[f"{metric}__median"]) for r in cells if r[f"{metric}__median"] not in ("", None)]
            entry[stage] = statistics.median(values) if values else None
            entry[f"{stage}_profiles_available"] = len(values)
        out[a] = entry
    return out


class Structure288:
    def __init__(self, table: Mapping[int, Mapping[str, Any]]) -> None:
        self.table = {int(k): v for k, v in table.items()}
        self.catalog = action_contract.load_contract()
        for mode in self.catalog.modes:
            for anchor in self.catalog.anchors_for_mode(mode.mode_id):
                row = self.table[anchor.action_id]
                _require((row["family"], row["quantizer"], row["q_e4"]) == (mode.family, mode.quantizer, anchor.q_e4),
                         f"288 action {anchor.action_id} differs from the locked catalogue")
        self._anchor = {(m.mode_id, a.q_e4): a.action_id for m in self.catalog.modes
                        for a in self.catalog.anchors_for_mode(m.mode_id)}

    def value(self, stage: str, mode_id: int, q_e4: int) -> float:
        _require(0 <= q_e4 <= 9800, f"q_e4 {q_e4} outside the registered range")
        if q_e4 in ANCHORS:
            v = self.table[self._anchor[(mode_id, q_e4)]][stage]
            _require(v is not None, f"288 {stage} unmeasured at mode {mode_id} q {q_e4}")
            return float(v)
        lo = max(a for a in ANCHORS if a < q_e4)
        hi = min(a for a in ANCHORS if a > q_e4)
        w = (q_e4 - lo) / (hi - lo)
        return (1.0 - w) * self.value(stage, mode_id, lo) + w * self.value(stage, mode_id, hi)


# --------------------------------------------------------------------------- live spans and transfer scales
def capture_profiles(root: Path = PROFILER) -> list[tuple[str, Path, Path]]:
    out = []
    for ue in sorted(glob.glob(str(root / "*/**/profiler/ue.profile.jsonl"), recursive=True)):
        name = os.path.relpath(ue, root)
        if name[:8] < CAPTURE_MIN_DATE:
            continue
        edge = Path(ue).parent / "edge/edge.profile.jsonl"
        if edge.is_file():
            out.append((name.split("/profiler/")[0], Path(ue), edge))
    return out


def live_spans(ue: Path, edge: Path) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = collections.defaultdict(dict)
    for role, path in (("ue", ue), ("edge", edge)):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if '"span"' not in line:
                    continue
                x = json.loads(line)
                if (x.get("r") != "span" or x.get("ph") != "live" or SPAN_ROLE.get(x.get("stage")) != role
                        or x.get("frame_id") is None or x.get("mode_id") is None):
                    continue
                f = frames[int(x["frame_id"])]
                f["mode_id"], f["q_e4"] = int(x["mode_id"]), int(x["q_e4"])
                f[x["stage"]] = (x["end_raw_ns"] - x["start_raw_ns"]) / 1e6
    return frames


def fit_transfer(groups: list[dict[str, Any]], captures: list[str]) -> dict[str, Any]:
    index = {c: i for i, c in enumerate(captures)}
    m = np.zeros((len(groups), len(captures) + 1))
    y = np.zeros(len(groups))
    w = np.zeros(len(groups))
    for i, g in enumerate(groups):
        m[i, index[g["capture"]]] = 1.0
        m[i, -1] = g["x288"]
        y[i] = g["live_median"]
        w[i] = math.sqrt(g["frames"])
    coef, *_ = np.linalg.lstsq(m * w[:, None], y * w, rcond=None)
    residual = y - m @ coef
    by_mode = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for g, e in zip(groups, residual):
        s = by_mode[g["mode_id"]]
        s[0] += e * g["frames"]
        s[1] += g["frames"]
        s[2] += 1
    informative = sorted({g["capture"] for g in groups
                          if len({(h["mode_id"], h["q_bin"]) for h in groups if h["capture"] == g["capture"]}) > 1})
    return dict(scale=float(coef[-1]), groups=len(groups), frames=int(sum(g["frames"] for g in groups)),
                captures=len(captures), captures_with_action_variation=informative,
                residual_by_mode={str(k): dict(weighted_mean_ms=v[0] / v[1], frames=int(v[1]), groups=v[2])
                                  for k, v in sorted(by_mode.items())},
                residual_abs_p50_ms=float(np.median(np.abs(residual))),
                residual_abs_p90_ms=float(np.percentile(np.abs(residual), 90)))


def build_transfer(structure: Structure288) -> tuple[dict[str, Any], dict[str, Any]]:
    captures = capture_profiles()
    _require(captures, "no live profiler captures found")
    pins, rows = {}, []
    for name, ue, edge in captures:
        pins[name] = {"ue": pin(ue), "edge": pin(edge)}
        for fid, f in live_spans(ue, edge).items():
            rows.append(dict(capture=name, frame_id=fid, **f))
    result = {}
    for stage, (span, _) in STAGE_METRICS.items():
        cells = collections.defaultdict(list)
        for r in rows:
            if span in r:
                cells[(r["capture"], r["mode_id"], round(r["q_e4"] / Q_BIN))].append(r)
        groups = []
        for (cap, mode, qbin), members in sorted(cells.items()):
            groups.append(dict(capture=cap, mode_id=mode, q_bin=qbin, frames=len(members),
                               live_median=statistics.median(x[span] for x in members),
                               x288=statistics.median(structure.value(stage, mode, x["q_e4"]) for x in members)))
        names = sorted({g["capture"] for g in groups})
        result[stage] = dict(span=span, metric=STAGE_METRICS[stage][1], **fit_transfer(groups, names))
        result[stage]["mode_coverage"] = sorted({g["mode_id"] for g in groups})
    return result, pins


def span_stage_correspondence(chains: Iterable[FrameChainV1], spans: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Check on the base capture that each span measures the chain stage it rescales."""
    pairs = collections.defaultdict(list)
    for c in chains:
        s = spans.get(c.frame_id)
        if not s:
            continue
        t = c.instants
        stage = {"C": t["ue_first_send"] - t["ue_input_end"], "D": t["ue_last_send"] - t["ue_first_send"],
                 "F": t["prediction_materialized"] - t["edge_reassembly_complete"]}
        for key, (span, _) in STAGE_METRICS.items():
            if span in s:
                pairs[key].append((stage[key], s[span]))
    out = {}
    for key, values in pairs.items():
        a = np.asarray(values, float)
        out[key] = dict(n=len(values), stage_p50=float(np.median(a[:, 0])), span_p50=float(np.median(a[:, 1])),
                        median_stage_minus_span=float(np.median(a[:, 0] - a[:, 1])),
                        pearson=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]))
    return out


# --------------------------------------------------------------------------- chains
def load_chains(source: Mapping[str, Any], *, rows_entry=None, frames_entry=None) -> tuple[list[FrameChainV1], list]:
    """Boundary rows + service records -> chains (contexts from the row itself; family/quantizer from the catalogue)."""
    rows_path = verified(rows_entry) if rows_entry else Path(source["rows"])
    frames_path = verified(frames_entry) if frames_entry else Path(source["frames"])
    catalog = action_contract.load_contract()
    rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if source.get("row_filter"):
        rows = [r for r in rows if all(r.get(k) == v for k, v in source["row_filter"].items())]
    _require(rows, f"{source['source_id']}: no rows")
    records = service_records([json.loads(l) for l in frames_path.read_text(encoding="utf-8").splitlines() if l.strip()])
    chains, refused = [], []
    for r in sorted(rows, key=lambda r: r["decision_index"]):
        action = r["executed_action"]
        mode = catalog.modes[int(action["mode_id"])]
        context = ChainContextV1(
            mode_id=int(action["mode_id"]), q_e4=int(action["q_e4"]), family=mode.family, quantizer=mode.quantizer,
            wire_bytes=V1.udp_application_bytes(int(r["payload_bytes"])), prior_ul_mcs=int(r["prior_ul_mcs"]),
            backlog_bytes=float(r["pre_action_rlc_backlog_bytes"]), camera_si=float(r["camera_si"]),
            radar_p40=float(r["radar_p40"]), network_profile=str(r["network_profile_id"]), cold=bool(r["cold"]))
        try:
            chains.append(chain_from_boundary_row(r, source_id=source["source_id"], context=context,
                                                  service=records.get(int(r["frame_id"]))))
        except ChainError as exc:
            refused.append((int(r["frame_id"]), str(exc)))
    return chains, refused


def decompose_all(chains: list[FrameChainV1]) -> tuple[dict, dict]:
    vectors, rejected = {}, {}
    for c in chains:
        if c.context.cold:
            continue
        try:
            vectors[c.chain_id] = decompose(c)
        except ChainError as exc:
            rejected[c.chain_id] = str(exc)
    return vectors, rejected


def _rejections_match(source_id: str, rejected: Mapping[str, str]) -> bool:
    expected = sorted(REGISTERED_REJECTIONS.get(source_id, ()))
    got = sorted(msg.split(": ", 1)[1] for msg in rejected.values())
    return expected == got


# --------------------------------------------------------------------------- build
def build(out: Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    table = table_288()
    structure = Structure288(table)
    transfer, capture_pins = build_transfer(structure)

    base_chains, base_refused = load_chains(BASE_SOURCE)
    base_vectors, base_rejected = decompose_all(base_chains)
    _require(not base_refused, f"base rows refused at load: {base_refused[:3]}")
    _require(_rejections_match("valid03", base_rejected), f"base rejections {base_rejected} differ from the registered list")
    base_spans = live_spans(PROFILER / "20261006_run5_pilot_live_v1/valid_b1000_03" / CELL / "profiler/ue.profile.jsonl",
                            PROFILER / "20261006_run5_pilot_live_v1/valid_b1000_03" / CELL / "profiler/edge/edge.profile.jsonl")
    correspondence = span_stage_correspondence([c for c in base_chains if c.chain_id in base_vectors], base_spans)

    validation = []
    for source in VALIDATION_SOURCES:
        chains, refused = load_chains(source)
        vectors, rejected = decompose_all(chains)
        validation.append(dict(source_id=source["source_id"], row_filter=source["row_filter"], rows=pin(source["rows"]),
                               frames=pin(source["frames"]), warm_chains=len(vectors), rejected=rejected,
                               refused_at_load=refused))

    document = dict(
        schema="splitfusion.run5_full.latency_evidence.v1", deadline_ms_inclusive=170,
        composition_model="SERVICE_QUEUE_V2",
        base=dict(source_id="valid03", rows=pin(BASE_SOURCE["rows"]), frames=pin(BASE_SOURCE["frames"]),
                  warm_chains=len(base_vectors), registered_rejections=base_rejected,
                  pipeline="pilot live validation valid_03: single-buffer UE input v2 + mask-lookup admission + "
                           "archive capacity repair (the deployed pipeline)",
                  span_stage_correspondence=correspondence),
        structure_288=dict(table=pin(TABLE_288), anchors=list(ANCHORS), stages={k: v[1] for k, v in STAGE_METRICS.items()},
                           pooling="median over the four profile cells of the per-cell medians",
                           q_rule="linear in q between the six measured anchors of the mode",
                           per_action={str(a): v for a, v in table.items()}),
        transfer=dict(rule="delta_stage = scale x (288 value at the decision action - 288 value at the base chain action)",
                      fit="per (capture, mode, q/500) median live span ~ capture fixed effect + scale x 288 value, "
                          "weights sqrt(frames), live-phase spans, captures >= 2026-10-02 with UE and edge profiles",
                      stages=transfer, capture_profiles=capture_pins),
        uplink=dict(model="T_v2b (ue_production_transport_model_v2, pinned artifact) boundary LAST_UDP_SOCKET_HANDOFF -> "
                          "COMPLETE_RECEIVER_REASSEMBLY = chain stage E",
                    rule="E = E_base + T(wire, MCS, backlog) - T(base wire, base MCS, base backlog) (conditional head); "
                         "late with probability clip(1 - p_on_time(decision) / p_on_time(base), 0, 1) (deadline head)"),
        validation_sources=validation)
    (out / "LATENCY_EVIDENCE.json").write_text(json.dumps(document, indent=1, sort_keys=True))
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "OUTPUT_SHA256.json"}
    (out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return dict(base_chains=len(base_vectors), rejected=base_rejected,
                transfer={k: dict(scale=v["scale"], groups=v["groups"], frames=v["frames"], modes=v["mode_coverage"])
                          for k, v in transfer.items()},
                correspondence=correspondence, validation={v["source_id"]: v["warm_chains"] for v in validation},
                evidence_sha256=canonical(document))


if __name__ == "__main__":
    import sys
    print(json.dumps(build(Path(sys.argv[1])), indent=1))
