#!/usr/bin/env python3
"""Operational action-open -> UE-receipt latency, ``L_op = A + S + T + E + D``.

A  action open -> first socket send.  Pooled valid POLICY_DECISION rows of the
   split-host 300-frame characterization.  The live engine stamps action open
   inside ``build_state`` and calls the actor afterwards, so A already contains
   actor inference and front preparation; no actor reserve is added.
S  first -> last socket send, registered 0.513047 ns per wire byte.
T  last socket handoff -> complete edge reassembly, accepted transport v2b.
E  complete edge reassembly -> publish start.  Pooled complete-case
   reward-requested POLICY_DECISION rows of the same split-host run.
D  evaluation completed -> UE receipt in the retained action-50 quality-probe
   rows: the measured compact-datagram preparation/downlink proxy.

Excluded by construction: GT wait, GT scoring, Q_perc computation, map
installation, prediction-ready -> evaluation-enqueue, and the old actor
reserve.  A and E are pooled across representation families; this is the
explicit ``EXPLORATORY_POOLED_FAMILY_TRANSFER_ASSUMPTION``.

Every source file and every derived pool is SHA-256 pinned.  A, E and D are
drawn from three independent deterministic streams, once per decision and
before the action is known, so the draws never depend on the policy.

Importing this module performs no file I/O and no RNG operation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]

SCHEMA_ID = "scenesense.operational_latency_provider.v1"
LABEL = "EXPLORATORY_POOLED_FAMILY_TRANSFER_ASSUMPTION"
COMPOSITION = "L_op = A + S + T + E + D"
DEADLINE_NS = 170_000_000  # inclusive
SEND_SPAN_NS_PER_BYTE = 0.513047
EXCLUDED_COMPONENTS = (
    "GT_WAIT", "GT_SCORING", "QPERC_COMPUTATION", "MAP_INSTALLATION",
    "PREDICTION_READY_TO_EVALUATION_ENQUEUE", "OLD_ACTOR_RESERVE",
)
STREAM_DOMAIN = "SCENESENSE_OPERATIONAL_LATENCY_STREAM_V1"
STREAM_LABELS = ("A", "E", "D")

SPLIT_HOST = ("rl_agent/experiments/splitfusion_run4_split_host_l10319_v1/"
              "20261001T020000Z_full_300_characterization/attempt/")
PROBE = ("experiments/splitfusion_quality_feedback_probe_v1/"
         "20260916_action50_favorable_adverse_retry4/cells/")

# Exact pinned sources.  Name -> (repo-relative path, file SHA-256).
SOURCES: Mapping[str, tuple[str, str]] = {
    "phase6_ue_evidence": (
        SPLIT_HOST + "run4_phase6/PHASE6_UE_EVIDENCE.json",
        "52b73103a95bc6ee110df07c578e24703c0a59c668315cbffa60e95489418c41"),
    "direct_map_ingest": (
        SPLIT_HOST + "direct_edge_map/direct_map_ingest.csv",
        "536f57be1883240c80dd27bdd6960f3232bf29050e682ff7e0f66266a4a52a5f"),
    "probe_favorable": (
        PROBE + "a50__favorable_stable/quality_feedback_timing_join.csv",
        "a2740342de391e8888d6a517c9571e9c0548020070d6d8080806c9863995f1d6"),
    "probe_adverse": (
        PROBE + "a50__adverse_stable/quality_feedback_timing_join.csv",
        "f9113fda612083dfb8d3f905935651798474e2e9fd7b2ae9cc01978688a17367"),
    "transport_v2b": (
        "rl_agent/experiments/ue_production_queue_capture_v1/"
        "20260929_model_v2b/transport_model_v2.json",
        "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c"),
}
SOURCE_STATUS = {
    "phase6_ue_evidence": (
        "split-host run status FAILED on an evaluator/ground-truth fault; "
        "characterization only, not a formal qualification. The fault lies "
        "outside the A and E boundaries."),
}

# Pinned derived pools: name -> (row count, canonical SHA-256 of the ns list).
EXPECTED_POOLS: Mapping[str, tuple[int, str]] = {
    "A": (138, "149deeadbabc9b6dc2e75dbc4ebf69f144a5d1a1acc05d994d6015ff28c813ea"),
    "E": (127, "40f06df7df6b657991f6cb4cb054cb923806669b384562fa6bad102a513f9123"),
    "D": (555, "fd984be80b03d9b611d49283660df445215313a37cdf115d27562eba8288365d"),
    "E_HOLD_SENSITIVITY": (148, "aa33efb7bbff76bbc0335d6e7d9483090b0e46afba63f42851b5af8cda4359b1"),
}

D_COLUMN = "evaluation_completed_to_ue_receive_ms"


class ProviderError(RuntimeError):
    """A source, pool or composition invariant failed (fail closed)."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _decimal(text: Any) -> Optional[Decimal]:
    try:
        value = Decimal(str(text).strip())
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _to_ns(value: Decimal, scale: int) -> int:
    return int((value * scale).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


# ---------------------------------------------------------------------------
# Pure pool extraction (no pin check; the loader adds pins)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PoolV1:
    name: str
    values_ns: tuple[int, ...]
    exclusions: Mapping[str, int]
    definition: str

    @property
    def sha256(self) -> str:
        return canonical_sha256(list(self.values_ns))

    def summary(self) -> dict[str, Any]:
        ordered = sorted(self.values_ns)

        def pct(p: float) -> float:
            return ordered[min(len(ordered) - 1,
                               int(p * (len(ordered) - 1) + 0.5))] / 1e6
        return {"n": len(ordered), "sha256": self.sha256,
                "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
                "min_ms": ordered[0] / 1e6, "max_ms": ordered[-1] / 1e6,
                "exclusions": dict(sorted(self.exclusions.items())),
                "definition": self.definition}


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def extract_pools(paths: Mapping[str, Path]) -> dict[str, PoolV1]:
    """Derive A, E, D and the hold-sensitivity pool from explicit files."""
    evidence = json.loads(Path(paths["phase6_ue_evidence"]).read_text(
        encoding="utf-8"))

    # A: valid POLICY_DECISION action open -> first socket send (one clock).
    a_values: list[int] = []
    a_excl: dict[str, int] = {}
    kind_by_frame: dict[int, str] = {}
    for row in evidence["decisions"]:
        kind_by_frame[int(row["frame_id"])] = str(row["kind"])
        if row["kind"] != "POLICY_DECISION":
            _bump(a_excl, f"kind_{row['kind']}")
            continue
        opened = row.get("action_open") or {}
        first = (row.get("stages") or {}).get("first_packet_send_raw_ns")
        if (opened.get("domain") != "CLOCK_MONOTONIC_RAW"
                or type(opened.get("ns")) is not int or type(first) is not int):
            _bump(a_excl, "missing_or_foreign_clock")
            continue
        value = first - opened["ns"]
        if value < 0:
            _bump(a_excl, "negative")
            continue
        a_values.append(value)

    # E: complete-case reward-requested POLICY_DECISION rows, exact frame join.
    frames: dict[int, Mapping[str, Any]] = {}
    for frame in evidence["frames"]:
        frame_id = int(frame["frame_id"])
        _require(frame_id not in frames, f"duplicate frame record {frame_id}")
        frames[frame_id] = frame
    e_values: list[int] = []
    hold_values: list[int] = []
    e_excl: dict[str, int] = {}
    seen: set[int] = set()
    with Path(paths["direct_map_ingest"]).open(newline="",
                                               encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_id = int(row["frame_id"])
            _require(frame_id not in seen, f"duplicate ingest frame {frame_id}")
            seen.add(frame_id)
            frame = frames.get(frame_id)
            if frame is None:
                _bump(e_excl, "no_frame_record")
                continue
            kind = kind_by_frame.get(frame_id)
            start = _decimal(row["edge_reassembly_complete_wall_s"])
            stop = _decimal(row["edge_publish_start_wall_s"])
            if start is None or stop is None:
                _bump(e_excl, "nonfinite_timestamp")
                continue
            if stop < start:
                _bump(e_excl, "publish_before_reassembly")
                continue
            value = _to_ns(stop - start, 1_000_000_000)
            if kind == "POLICY_DECISION" and frame["reward_requested"] is True:
                e_values.append(value)
            elif kind == "POLICY_HOLD" and frame["reward_requested"] is False:
                _bump(e_excl, "policy_hold_sensitivity_only")
                hold_values.append(value)
            else:
                _bump(e_excl, f"kind_{kind}_reward_{frame['reward_requested']}")
    for frame_id, frame in frames.items():
        if (frame_id not in seen and kind_by_frame.get(frame_id) == "POLICY_DECISION"
                and frame["reward_requested"] is True):
            _bump(e_excl, "policy_decision_without_ingest_row")

    # D: evaluation completed -> UE receipt; excludes GT wait and scoring.
    d_values: list[int] = []
    d_excl: dict[str, int] = {}
    for name in ("probe_favorable", "probe_adverse"):
        with Path(paths[name]).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = _decimal(row.get(D_COLUMN, ""))
                if value is None:
                    _bump(d_excl, "nonfinite_or_missing")
                    continue
                if value < 0:
                    _bump(d_excl, "negative")
                    continue
                d_values.append(_to_ns(value, 1_000_000))

    return {
        "A": PoolV1("A", tuple(a_values), a_excl,
                    "POLICY_DECISION first_packet_send_raw_ns - action_open.ns "
                    "(CLOCK_MONOTONIC_RAW)"),
        "E": PoolV1("E", tuple(e_values), e_excl,
                    "edge_publish_start_wall_s - edge_reassembly_complete_wall_s"
                    " for kind=POLICY_DECISION and reward_requested=true"),
        "D": PoolV1("D", tuple(d_values), d_excl,
                    f"{D_COLUMN} from both retained action-50 probe cells"),
        "E_HOLD_SENSITIVITY": PoolV1(
            "E_HOLD_SENSITIVITY", tuple(hold_values), {},
            "same E boundary for POLICY_HOLD rows; sensitivity only, never "
            "used by compose()"),
    }


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ComponentDrawV1:
    """One decision's action-independent A/E/D draw."""

    a_index: int
    e_index: int
    d_index: int
    a_ns: int
    e_ns: int
    d_ns: int

    def to_dict(self) -> dict[str, int]:
        return {"a_index": self.a_index, "e_index": self.e_index,
                "d_index": self.d_index, "a_ns": self.a_ns,
                "e_ns": self.e_ns, "d_ns": self.d_ns}


@dataclass(frozen=True, slots=True)
class OperationalOutcomeV1:
    a_ns: int
    s_ns: int
    t_ns: int
    e_ns: int
    d_ns: int
    total_ns: int
    on_time_probability: float
    transport_success: bool
    timely: bool

    def to_dict(self) -> dict[str, Any]:
        return {"a_ns": self.a_ns, "s_ns": self.s_ns, "t_ns": self.t_ns,
                "e_ns": self.e_ns, "d_ns": self.d_ns,
                "total_ns": self.total_ns,
                "on_time_probability": self.on_time_probability,
                "transport_success": self.transport_success,
                "timely": self.timely}


def send_span_ns(wire_bytes: int) -> int:
    _require(type(wire_bytes) is int and wire_bytes > 0, "wire_bytes invalid")
    return int(round(SEND_SPAN_NS_PER_BYTE * wire_bytes))


def compose_total_ns(*, a_ns: int, s_ns: int, t_ns: int, e_ns: int,
                     d_ns: int) -> int:
    """Each component appears exactly once; nothing else is added."""
    for value, name in ((a_ns, "A"), (s_ns, "S"), (t_ns, "T"), (e_ns, "E"),
                        (d_ns, "D")):
        _require(type(value) is int and value >= 0,
                 f"{name} must be a non-negative exact int")
    return a_ns + s_ns + t_ns + e_ns + d_ns


def is_timely(*, transport_success: bool, total_ns: int) -> bool:
    """Inclusive deadline: exactly 170 ms succeeds, 170 ms + 1 ns does not."""
    return bool(transport_success) and total_ns <= DEADLINE_NS


class LatencyStreamsV1:
    """Three independent deterministic A/E/D streams with exact state I/O."""

    def __init__(self, master_seed: int, pool_sizes: Mapping[str, int]) -> None:
        _require(type(master_seed) is int and master_seed >= 0,
                 "master_seed must be a non-negative int")
        self._sizes = {k: int(pool_sizes[k]) for k in STREAM_LABELS}
        self._rngs = {label: random.Random(stream_seed(master_seed, label))
                      for label in STREAM_LABELS}

    def draw_indices(self) -> tuple[int, int, int]:
        return tuple(self._rngs[label].randrange(self._sizes[label])
                     for label in STREAM_LABELS)

    def get_state(self) -> dict[str, list]:
        return {label: _state_to_list(self._rngs[label].getstate())
                for label in STREAM_LABELS}

    def set_state(self, state: Mapping[str, Sequence]) -> None:
        _require(set(state) == set(STREAM_LABELS), "stream state labels differ")
        for label in STREAM_LABELS:
            self._rngs[label].setstate(_state_from_list(state[label]))


def stream_seed(master_seed: int, label: str) -> int:
    material = {"domain": STREAM_DOMAIN, "label": label,
                "master_seed": master_seed}
    return int.from_bytes(hashlib.sha256(
        canonical_sha256(material).encode("ascii")).digest()[:8], "big") & (
            (1 << 63) - 1)


def _state_to_list(state: tuple) -> list:
    version, internal, gauss = state
    return [int(version), [int(x) for x in internal], gauss]


def _state_from_list(value: Sequence) -> tuple:
    version, internal, gauss = value
    return (int(version), tuple(int(x) for x in internal),
            None if gauss is None else float(gauss))


class OperationalLatencyProviderV1:
    """Pinned pools + transport v2b + exact composition."""

    def __init__(self, *, pools: Mapping[str, PoolV1], transport_model: Any,
                 source_sha256: Mapping[str, str], pinned: bool) -> None:
        for name in STREAM_LABELS:
            _require(len(pools[name].values_ns) > 0, f"pool {name} is empty")
        self._pools = dict(pools)
        self._transport = transport_model
        self._sources = dict(source_sha256)
        self.pinned = bool(pinned)
        self._binding = self._binding_document()
        self.binding_sha256 = canonical_sha256(self._binding)

    @classmethod
    def load(cls, root: Path = ROOT) -> "OperationalLatencyProviderV1":
        """Verify every source and derived-pool pin, then construct."""
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for name, (relpath, expected) in SOURCES.items():
            path = Path(root) / relpath
            _require(path.is_file(), f"pinned source missing: {relpath}")
            observed = sha256_file(path)
            _require(observed == expected, f"source hash drift: {name}")
            paths[name] = path
            hashes[name] = observed
        pools = extract_pools(paths)
        for name, (count, digest) in EXPECTED_POOLS.items():
            _require(len(pools[name].values_ns) == count,
                     f"pool {name} row count {len(pools[name].values_ns)} "
                     f"!= pinned {count}")
            _require(pools[name].sha256 == digest, f"pool {name} hash drift")
        return cls(pools=pools, transport_model=_load_transport(paths),
                   source_sha256=hashes, pinned=True)

    @classmethod
    def unpinned_for_tests(cls, paths: Mapping[str, Path]
                           ) -> "OperationalLatencyProviderV1":
        """Test seam: identical extraction/composition without pin checks."""
        hashes = {name: sha256_file(Path(paths[name])) for name in SOURCES}
        return cls(pools=extract_pools(paths),
                   transport_model=_load_transport(paths),
                   source_sha256=hashes, pinned=False)

    # -- binding ---------------------------------------------------------
    def _binding_document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "label": LABEL,
            "composition": COMPOSITION,
            "deadline_ns_inclusive": DEADLINE_NS,
            "send_span_ns_per_byte": SEND_SPAN_NS_PER_BYTE,
            "excluded_components": list(EXCLUDED_COMPONENTS),
            "family_conditioning": "NONE_POOLED_A_AND_E",
            "stream_domain": STREAM_DOMAIN,
            "stream_labels": list(STREAM_LABELS),
            "draw_rule": ("one randrange index per stream per decision, drawn "
                          "before the action is known"),
            "transport_document_sha256": self._transport.document_sha256,
            "sources": {name: {"relpath": SOURCES[name][0],
                               "sha256": self._sources[name]}
                        for name in sorted(SOURCES)},
            "source_status": dict(SOURCE_STATUS),
            "pools": {name: {"n": len(pool.values_ns), "sha256": pool.sha256,
                             "definition": pool.definition,
                             "used_by_compose": name in STREAM_LABELS}
                      for name, pool in sorted(self._pools.items())},
        }

    def binding_document(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._binding))

    def pool(self, name: str) -> PoolV1:
        return self._pools[name]

    @property
    def transport_model(self) -> Any:
        return self._transport

    def streams(self, master_seed: int) -> LatencyStreamsV1:
        return LatencyStreamsV1(
            master_seed, {k: len(self._pools[k].values_ns)
                          for k in STREAM_LABELS})

    def draw(self, streams: LatencyStreamsV1) -> ComponentDrawV1:
        a, e, d = streams.draw_indices()
        return ComponentDrawV1(
            a_index=a, e_index=e, d_index=d,
            a_ns=self._pools["A"].values_ns[a],
            e_ns=self._pools["E"].values_ns[e],
            d_ns=self._pools["D"].values_ns[d])

    def resolve(self, *, draw: ComponentDrawV1, wire_bytes: int,
                pre_enqueue_backlog_bytes: int, prior_ul_mcs: int,
                success_uniform: float) -> OperationalOutcomeV1:
        """One transport draw, one composed operational total."""
        _require(type(draw) is ComponentDrawV1, "draw has a foreign type")
        _require(0.0 <= success_uniform < 1.0, "success_uniform out of [0,1)")
        prediction = self._transport.predict(
            pre_enqueue_backlog_bytes=float(pre_enqueue_backlog_bytes),
            wire_bytes=int(wire_bytes), prior_ul_mcs=int(prior_ul_mcs))
        success = success_uniform < prediction.on_time_probability
        s_ns = send_span_ns(int(wire_bytes))
        t_ns = int(round(prediction.conditional_latency_ms * 1e6))
        total = compose_total_ns(a_ns=draw.a_ns, s_ns=s_ns, t_ns=t_ns,
                                 e_ns=draw.e_ns, d_ns=draw.d_ns)
        return OperationalOutcomeV1(
            a_ns=draw.a_ns, s_ns=s_ns, t_ns=t_ns, e_ns=draw.e_ns,
            d_ns=draw.d_ns, total_ns=total,
            on_time_probability=float(prediction.on_time_probability),
            transport_success=bool(success),
            timely=is_timely(transport_success=success, total_ns=total))


def _load_transport(paths: Mapping[str, Path]) -> Any:
    from rl_agent.ue_production_transport_model_v2 import artifact_v2
    return artifact_v2.ProductionTransportModelV2.load(
        Path(paths["transport_v2b"]))


def default_paths(root: Path = ROOT) -> dict[str, Path]:
    return {name: Path(root) / relpath for name, (relpath, _) in SOURCES.items()}


def main() -> int:
    """Write the provider binding evidence next to this module."""
    provider = OperationalLatencyProviderV1.load()
    document = {
        "binding": provider.binding_document(),
        "binding_sha256": provider.binding_sha256,
        "pools": {name: provider.pool(name).summary()
                  for name in sorted(EXPECTED_POOLS)},
    }
    out = Path(__file__).resolve().parent / "PROVIDER_BINDING.json"
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(provider.binding_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
