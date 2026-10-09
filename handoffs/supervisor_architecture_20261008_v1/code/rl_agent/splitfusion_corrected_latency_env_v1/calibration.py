"""Hash-bound calibration binding, chain loading, stratum sampling and the launch-approval gate.

Nothing here invents a distribution. A decision is calibrated only if:
  * its exact action stratum (mode_id, q_e4 within the declared tolerance, default 0) has enough measured chains;
  * its channel profile is measured (or, for the unlabeled Run-4 channel, its MCS lies inside the chains' MCS);
  * its scene descriptors lie inside the stratum's measured descriptor support;
  * any uplink transfer to a different payload/channel state uses the pinned transport model only inside a
    hash-bound validation scope.
Otherwise ``UncalibratedTermError`` is raised and the decision (hence training) cannot proceed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2
from rl_agent.ue_production_queue_capture_v1 import contract as V1
from rl_agent.ue_production_transport_model_v2 import artifact_v2 as A2

from . import CANDIDATE_VERSION
from .chains import (COMPOSITION_MODEL, ChainContextV1, ChainError, FrameChainV1, Ordering, RoleCostsV1,
                     StageVectorV1, UncalibratedTermError, chain_from_boundary_row, decompose, service_records)

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]          # abiodun/
NEU = ROOT.parent                      # neu_collab/ (holds the profiler worktree)
BINDING_SCHEMA = "splitfusion.corrected_latency_env.calibration_binding.v2"
SUPERSEDED_SCHEMAS = ("splitfusion.corrected_latency_env.calibration_binding.v1",)
# Per-class quality fields carried for joint latency/quality assessment when a source delivers them.
QUALITY_FIELDS = ("quality_defined", "vehicle_eligible_gt", "vehicle_tp", "vehicle_fp", "vehicle_fn",
                  "vehicle_matched_xy_errors_m", "person_eligible_gt", "person_tp", "person_fp", "person_fn",
                  "person_matched_xy_errors_m", "seg_vehicle_intersection_pixels", "seg_vehicle_union_pixels",
                  "seg_person_intersection_pixels", "seg_person_union_pixels")
APPROVAL_PATH = PACKAGE_DIR / "APPROVAL.json"          # deliberately absent: launch stays blocked
TRANSPORT_ARTIFACT_RELPATH = "rl_agent/experiments/ue_production_queue_capture_v1/20260929_model_v2b/transport_model_v2.json"
TRANSPORT_ARTIFACT_SHA256 = "9919e5285d454ec742d877ca33af0df30277df82fe3cf665288c1102b6be286c"
DEADLINE_MS = 170
SOURCE_KINDS = ("BOUNDARIES_WITH_PHASE6_CONTEXT_V1", "BOUNDARIES_WITH_INLINE_CONTEXT_V1")
UPLINK_MODES = ("TRANSPORT_MODEL_DELTA", "CHAIN_ONLY")
# Live feature scaling used by the deployed actors (unchanged state definition).
# Inverting these recovers raw SI/backlog from recorded live features; they are the training scaling (hash-checked
# by live_state_v2 at import) and are not changed by this candidate.
TRAINING_SCALING = live_state_v2.TRAINING_SCALING
CAMERA_SI_CENTER = TRAINING_SCALING.camera_si_center
CAMERA_SI_SCALE = TRAINING_SCALING.camera_si_scale
BACKLOG_LOG1P_SCALE = TRAINING_SCALING.backlog_log1p_scale
UL_MCS_MAX = 28
INLINE_REQUIRED_FIELDS = ("mode_id", "q_e4", "family", "quantizer", "payload_bytes", "prior_ul_mcs",
                          "backlog_bytes", "camera_si", "radar_p40", "network_profile", "cold")


class CalibrationBindingError(ValueError):
    """The binding is missing, malformed, drifted or incompatible. It is rejected, never repaired."""


class LaunchBlockedError(RuntimeError):
    """Training launch is blocked until the corrected model and calibration are explicitly approved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationBindingError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode("ascii")).hexdigest()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else NEU / p


def _bound_file(entry: Mapping[str, Any], label: str) -> Path:
    _require(isinstance(entry, Mapping) and set(entry) == {"path", "sha256"}, f"{label}: needs exactly path+sha256")
    path = resolve_path(entry["path"])
    _require(path.is_file(), f"{label}: missing file {path}")
    actual = sha256_file(path)
    _require(actual == entry["sha256"], f"{label}: sha256 {actual} differs from bound {entry['sha256']}")
    return path


# --------------------------------------------------------------------------- chain sources
def _phase6_contexts(evidence_path: Path, per_frame_path: Path, profile: str,
                     cold_frames: set[int]) -> dict[int, ChainContextV1]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    with per_frame_path.open(newline="", encoding="utf-8") as stream:
        per_frame = {int(r["frame_id"]): r for r in csv.DictReader(stream)}
    out = {}
    for d in evidence["decisions"]:
        if d.get("kind") != "POLICY_DECISION" or d.get("features") is None:
            continue
        fid, f = int(d["frame_id"]), d["features"]
        row = per_frame.get(fid)
        _require(row is not None and row.get("payload_bytes"), f"frame {fid}: payload bytes unavailable")
        payload = int(row["payload_bytes"])
        _require(int(row["datagrams"]) == V1.datagram_count(payload),
                 f"frame {fid}: payload_bytes is not the chunking coordinate")
        out[fid] = ChainContextV1(
            mode_id=int(d["mode_id"]), q_e4=int(d["q_e4"]), family=row["model_family"], quantizer=row["quantizer"],
            wire_bytes=V1.udp_application_bytes(payload),
            prior_ul_mcs=int(round(f[2] * UL_MCS_MAX)),
            backlog_bytes=float(round(math.expm1(f[3] * BACKLOG_LOG1P_SCALE))),
            camera_si=f[0] * CAMERA_SI_SCALE + CAMERA_SI_CENTER, radar_p40=float(f[1]),
            network_profile=profile, cold=fid in cold_frames)
    return out


def _load_source(source: Mapping[str, Any]) -> list[FrameChainV1]:
    kind = source.get("kind")
    _require(kind in SOURCE_KINDS, f"source {source.get('source_id')}: unknown kind {kind!r}")
    files = source.get("files") or {}
    boundaries = _bound_file(files.get("boundaries"), f"{source['source_id']}.boundaries")
    rows = [json.loads(line) for line in boundaries.read_text(encoding="utf-8").splitlines() if line.strip()]
    row_filter = source.get("row_filter") or {}
    rows = [r for r in rows if all(r.get(k) == v for k, v in row_filter.items())]
    _require(rows, f"{source['source_id']}: no rows")
    service_path = _bound_file(files.get("service_frames"), f"{source['source_id']}.service_frames")
    records = service_records([json.loads(line) for line in service_path.read_text(encoding="utf-8").splitlines()
                               if line.strip()])

    def service_for(row):
        record = records.get(int(row["frame_id"]))
        _require(record is not None, f"{source['source_id']}: frame {row['frame_id']} has no complete service record")
        return record

    def quality_for(row):
        return {k: row[k] for k in QUALITY_FIELDS if k in row}
    chains = []
    if kind == "BOUNDARIES_WITH_PHASE6_CONTEXT_V1":
        evidence = _bound_file(files.get("evidence"), f"{source['source_id']}.evidence")
        per_frame = _bound_file(files.get("per_frame"), f"{source['source_id']}.per_frame")
        contexts = _phase6_contexts(evidence, per_frame, source["network_profile"],
                                    {int(r["frame_id"]) for r in rows if r.get("cold")})
        for row in rows:
            context = contexts.get(int(row["frame_id"]))
            _require(context is not None, f"{source['source_id']}: frame {row['frame_id']} has no decision context")
            action = row.get("action") or {}
            _require((action.get("mode_id"), action.get("q_e4")) == (context.mode_id, context.q_e4),
                     f"{source['source_id']}: frame {row['frame_id']} action differs between boundary and evidence")
            chains.append(chain_from_boundary_row(row, source_id=source["source_id"], context=context,
                                                  service=service_for(row), quality=quality_for(row)))
    else:
        field_map = source.get("field_map") or {}
        _require(set(INLINE_REQUIRED_FIELDS) <= set(field_map),
                 f"{source['source_id']}: field_map lacks {sorted(set(INLINE_REQUIRED_FIELDS) - set(field_map))}")

        def get(row, name):
            value = row
            for part in field_map[name].split("."):
                _require(isinstance(value, Mapping) and part in value,
                         f"{source['source_id']}: frame {row.get('frame_id')} lacks {field_map[name]}")
                value = value[part]
            _require(value is not None, f"{source['source_id']}: frame {row.get('frame_id')} {name} unavailable")
            return value
        for row in rows:
            payload = int(get(row, "payload_bytes"))
            context = ChainContextV1(
                mode_id=int(get(row, "mode_id")), q_e4=int(get(row, "q_e4")), family=str(get(row, "family")),
                quantizer=str(get(row, "quantizer")), wire_bytes=V1.udp_application_bytes(payload),
                prior_ul_mcs=int(get(row, "prior_ul_mcs")), backlog_bytes=float(get(row, "backlog_bytes")),
                camera_si=float(get(row, "camera_si")), radar_p40=float(get(row, "radar_p40")),
                network_profile=str(get(row, "network_profile")), cold=bool(get(row, "cold")))
            _require(context.network_profile == source["network_profile"],
                     f"{source['source_id']}: row profile differs from the declared source profile")
            chains.append(chain_from_boundary_row(row, source_id=source["source_id"], context=context,
                                                  service=service_for(row), quality=quality_for(row)))
    return chains


# --------------------------------------------------------------------------- uplink validation scope
@dataclass(frozen=True)
class UplinkValidationScopeV1:
    profiles: tuple[str, ...]
    mcs_range: tuple[int, int]
    wire_range: tuple[int, int]
    median_abs_error_ms: float


def _load_uplink_validation(policy: Mapping[str, Any]) -> Optional[UplinkValidationScopeV1]:
    validation = policy.get("validation")
    if validation is None:
        return None
    path = _bound_file(validation.get("file"), "uplink.validation")
    document = json.loads(path.read_text(encoding="utf-8"))
    declared = validation.get("scope") or {}
    for key in ("profiles", "mcs_range", "wire_range", "max_median_abs_error_ms", "evidence_keys"):
        _require(key in declared, f"uplink validation scope lacks {key}")
    mcs_lo, mcs_hi, wire_lo, wire_hi, worst = math.inf, -math.inf, math.inf, -math.inf, 0.0
    for key in declared["evidence_keys"]:
        node = document
        for part in key.split("."):
            _require(isinstance(node, Mapping) and part in node, f"uplink validation evidence lacks {key}")
            node = node[part]
        mcs_lo, mcs_hi = min(mcs_lo, node["mcs"]["min"]), max(mcs_hi, node["mcs"]["max"])
        wire_lo, wire_hi = min(wire_lo, node["wire_bytes"]["min"]), max(wire_hi, node["wire_bytes"]["max"])
        worst = max(worst, abs(node["measured_minus_modeled_ms"]["p50"]))
    # The declared scope may not exceed what the bound evidence measured.
    _require(declared["mcs_range"][0] >= mcs_lo and declared["mcs_range"][1] <= mcs_hi,
             "declared uplink MCS scope exceeds the validation evidence")
    _require(declared["wire_range"][0] >= wire_lo and declared["wire_range"][1] <= wire_hi,
             "declared uplink wire scope exceeds the validation evidence")
    _require(worst <= float(declared["max_median_abs_error_ms"]),
             f"validation median error {worst:.3f} ms exceeds the declared tolerance")
    return UplinkValidationScopeV1(profiles=tuple(declared["profiles"]),
                                   mcs_range=tuple(int(v) for v in declared["mcs_range"]),
                                   wire_range=tuple(int(v) for v in declared["wire_range"]),
                                   median_abs_error_ms=worst)


# --------------------------------------------------------------------------- loaded binding
@dataclass(frozen=True)
class ChainDrawV1:
    vector: StageVectorV1
    chain_id: str
    donor_chain_id: Optional[str]
    uplink_delta_ms: float
    scene_distance: float
    stratum_size: int


class CalibrationSetV1:
    """A validated binding: measured chains, their vectors, and the declared conditioning policy."""

    def __init__(self, document: Mapping[str, Any], *, transport_model: Optional[A2.ProductionTransportModelV2] = None):
        _require(isinstance(document, Mapping), "binding must be a JSON object")
        _require(document.get("schema") not in SUPERSEDED_SCHEMAS,
                 "binding schema v1 is superseded: its composition lacks the explicit exact-GT service queue")
        _require(document.get("schema") == BINDING_SCHEMA, f"binding schema {document.get('schema')!r} is not {BINDING_SCHEMA}")
        _require(document.get("composition_model") == COMPOSITION_MODEL,
                 f"binding composition model must be {COMPOSITION_MODEL}")
        _require(document.get("candidate_version") == CANDIDATE_VERSION, "binding targets a different candidate version")
        _require(document.get("deadline_ms") == DEADLINE_MS, "binding deadline differs from the frozen inclusive 170 ms")
        identity = document.get("pipeline_identity")
        _require(isinstance(identity, Mapping) and identity, "binding lacks a pipeline identity")
        sources = document.get("sources")
        _require(isinstance(sources, list) and sources, "binding has no chain sources")
        policy = document.get("policy") or {}
        for key in ("exclude_cold", "action_match", "min_chains_per_stratum", "scene_conditioning", "uplink",
                    "fallback"):
            _require(key in policy, f"binding policy lacks {key}")
        match = policy["action_match"]
        _require(match.get("rule") == "EXACT_MODE_Q" and int(match.get("q_tolerance_e4", -1)) >= 0,
                 "action_match must be EXACT_MODE_Q with a non-negative q tolerance")
        self.q_tolerance_e4 = int(match["q_tolerance_e4"])
        self.min_chains = int(policy["min_chains_per_stratum"])
        _require(self.min_chains >= 1, "min_chains_per_stratum must be >= 1")
        scene = policy["scene_conditioning"]
        _require(scene.get("descriptors") == ["camera_si", "radar_p40"], "scene descriptors must be camera_si, radar_p40")
        self.k = int(scene["k"])
        self.support_margin = float(scene["support_margin_fraction"])
        _require(self.k >= 1 and self.support_margin >= 0, "invalid scene conditioning")
        uplink = policy["uplink"]
        _require(uplink.get("mode") in UPLINK_MODES, f"unknown uplink mode {uplink.get('mode')!r}")
        self.uplink_mode = uplink["mode"]
        _require(uplink.get("transport_artifact_sha256") == TRANSPORT_ARTIFACT_SHA256,
                 "uplink transport artifact is not the pinned v2b model")
        self.uplink_scope = _load_uplink_validation(uplink)
        self.chain_only_mcs_tolerance = int(uplink.get("chain_only_mcs_tolerance", 0))
        self.fallback_status = policy["fallback"].get("status")
        _require(self.fallback_status in ("UNCALIBRATED", "CALIBRATED"), "fallback status must be declared")
        _require(self.fallback_status == "UNCALIBRATED" or "rate_evidence" in policy["fallback"],
                 "a calibrated fallback needs bound rate evidence")

        chains: list[FrameChainV1] = []
        seen = set()
        for source in sources:
            _require(source.get("pipeline_identity") == identity,
                     f"source {source.get('source_id')}: pipeline identity differs from the binding's")
            _require(source.get("source_id") not in seen, "duplicate source id")
            seen.add(source["source_id"])
            try:
                loaded = _load_source(source)
            except ChainError as exc:
                raise CalibrationBindingError(f"source {source['source_id']}: {exc}") from exc
            chains.extend(c for c in loaded if not (policy["exclude_cold"] and c.context.cold))
        _require(chains, "no usable chains after the cold policy")
        try:
            self.vectors = {c.chain_id: decompose(c) for c in chains}
        except ChainError as exc:
            raise CalibrationBindingError(str(exc)) from exc
        self.chains = {c.chain_id: c for c in chains}
        self.document = document
        self.binding_sha256 = canonical_sha256(document)
        if self.uplink_mode == "TRANSPORT_MODEL_DELTA":
            if transport_model is None:
                path = ROOT / TRANSPORT_ARTIFACT_RELPATH
                _require(sha256_file(path) == TRANSPORT_ARTIFACT_SHA256, "transport artifact drifted")
                transport_model = A2.ProductionTransportModelV2.load(path)
            self.transport = transport_model
        else:
            self.transport = None

    # ---------------------------------------------------------------- coverage
    def stratum(self, mode_id: int, q_e4: int, profile: Optional[str]) -> list[FrameChainV1]:
        return sorted((c for c in self.chains.values()
                       if c.context.mode_id == mode_id and abs(c.context.q_e4 - q_e4) <= self.q_tolerance_e4
                       and (profile is None or c.context.network_profile == profile)), key=lambda c: c.chain_id)

    def coverage(self) -> dict[str, Any]:
        strata: dict[str, dict[str, Any]] = {}
        for c in self.chains.values():
            key = f"{c.context.network_profile}|mode{c.context.mode_id}|q{c.context.q_e4}"
            entry = strata.setdefault(key, dict(n=0, gt_first=0, prediction_first=0))
            entry["n"] += 1
            entry["gt_first" if self.vectors[c.chain_id].realized_ordering is Ordering.GT_FIRST
                  else "prediction_first"] += 1
        usable = {k: v for k, v in strata.items() if v["n"] >= self.min_chains}
        return dict(profiles=sorted({c.context.network_profile for c in self.chains.values()}),
                    strata=strata, usable_strata=sorted(usable), chains=len(self.chains),
                    continuous_q_coverage="EXACT_POINTS_ONLY" if self.q_tolerance_e4 == 0
                    else f"+-{self.q_tolerance_e4}_E4_AROUND_MEASURED_POINTS",
                    uplink_mode=self.uplink_mode,
                    uplink_validated_profiles=None if self.uplink_scope is None else list(self.uplink_scope.profiles),
                    fallback=self.fallback_status)

    def launch_blockers(self) -> list[str]:
        """Calibration gaps that block training unless an approval explicitly waives each one."""
        from rl_agent.splitfusion_hybrid_sac_run5_v1.run5_channel import TRAINING_PROFILES
        blockers = []
        covered = {c.context.network_profile for c in self.chains.values()}
        missing = [p for p in TRAINING_PROFILES if p not in covered]
        if missing:
            blockers.append(f"PROFILES_UNMEASURED:{','.join(missing)}")
        if self.uplink_mode == "TRANSPORT_MODEL_DELTA":
            validated = set() if self.uplink_scope is None else set(self.uplink_scope.profiles)
            unvalidated = [p for p in TRAINING_PROFILES if p not in validated]
            if unvalidated:
                blockers.append(f"UPLINK_TRANSFER_UNVALIDATED:{','.join(unvalidated)}")
        if self.q_tolerance_e4 == 0:
            blockers.append("CONTINUOUS_Q_EXACT_POINTS_ONLY")
        modes = {c.context.mode_id for c in self.chains.values()}
        if len(modes) < 12:
            blockers.append(f"MODES_UNMEASURED:{','.join(str(m) for m in range(12) if m not in modes)}")
        if self.fallback_status != "CALIBRATED":
            blockers.append("FALLBACK_RATE_AND_MECHANISM_UNCALIBRATED")
        purpose = str(self.document.get("purpose", ""))
        if purpose.startswith("REGRESSION_FIXTURE"):
            blockers.append("BINDING_IS_A_REGRESSION_FIXTURE")
        elif not purpose.startswith("CALIBRATION"):
            blockers.append(f"BINDING_PURPOSE_NOT_CALIBRATION:{purpose.split(':')[0]}")
        return blockers

    # ---------------------------------------------------------------- sampling
    def _scene_distance(self, stratum: Sequence[FrameChainV1], si: float, p40: float) -> list[tuple[float, str]]:
        sis = [c.context.camera_si for c in stratum]
        p40s = [c.context.radar_p40 for c in stratum]
        for name, values, x in (("camera_si", sis, si), ("radar_p40", p40s, p40)):
            lo, hi = min(values), max(values)
            pad = self.support_margin * (hi - lo)
            if not (lo - pad <= x <= hi + pad):
                raise UncalibratedTermError("scene_conditioning",
                                            f"{name}={x:.4f} outside measured stratum support [{lo:.4f}, {hi:.4f}]")
        def spread(values):
            mean = sum(values) / len(values)
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            return sd if sd > 0 else 1.0
        s_si, s_p = spread(sis), spread(p40s)
        return sorted((math.hypot((c.context.camera_si - si) / s_si, (c.context.radar_p40 - p40) / s_p), c.chain_id)
                      for c in stratum)

    def _uplink_delta(self, chain: FrameChainV1, *, profile: Optional[str], mcs: int, backlog: float,
                      wire: int) -> float:
        ctx = chain.context
        if self.uplink_mode == "CHAIN_ONLY":
            if abs(mcs - ctx.prior_ul_mcs) > self.chain_only_mcs_tolerance:
                raise UncalibratedTermError("E_uplink", f"MCS {mcs} vs chain {ctx.prior_ul_mcs} beyond CHAIN_ONLY tolerance")
            return 0.0
        scope = self.uplink_scope
        if scope is None:
            raise UncalibratedTermError("E_uplink", "TRANSPORT_MODEL_DELTA has no bound validation scope")
        if profile is not None and profile not in scope.profiles:
            raise UncalibratedTermError("E_uplink", f"transport model not validated for profile {profile}")
        for label, m, w in (("training", mcs, wire), ("chain", ctx.prior_ul_mcs, ctx.wire_bytes)):
            if not (scope.mcs_range[0] <= m <= scope.mcs_range[1]):
                raise UncalibratedTermError("E_uplink", f"{label} MCS {m} outside validated {scope.mcs_range}")
            if not (scope.wire_range[0] <= w <= scope.wire_range[1]):
                raise UncalibratedTermError("E_uplink", f"{label} wire {w} outside validated {scope.wire_range}")
        try:
            train = self.transport.predict(pre_enqueue_backlog_bytes=float(backlog), wire_bytes=int(wire),
                                           prior_ul_mcs=int(mcs))
            source = self.transport.predict(pre_enqueue_backlog_bytes=float(ctx.backlog_bytes),
                                            wire_bytes=int(ctx.wire_bytes), prior_ul_mcs=int(ctx.prior_ul_mcs))
        except A2.OutOfSupport as exc:
            raise UncalibratedTermError("E_uplink", f"transport model out of fit support: {exc}") from exc
        return train.conditional_latency_ms - source.conditional_latency_ms

    def draw(self, *, mode_id: int, q_e4: int, profile: Optional[str], camera_si: float, radar_p40: float,
             prior_ul_mcs: int, backlog_bytes: float, wire_bytes: int, rng: random.Random) -> ChainDrawV1:
        """One joint stage vector for this decision, or ``UncalibratedTermError``."""
        stratum = self.stratum(mode_id, q_e4, profile)
        if len(stratum) < self.min_chains:
            raise UncalibratedTermError(
                "action_stratum", f"mode {mode_id} q_e4 {q_e4} profile {profile or 'UNLABELED'}: "
                                  f"{len(stratum)} measured chains < {self.min_chains}")
        if profile is None:   # unlabeled Run-4 channel: stay inside the chains' measured MCS
            mcs_values = [c.context.prior_ul_mcs for c in stratum]
            if not (min(mcs_values) <= prior_ul_mcs <= max(mcs_values)):
                raise UncalibratedTermError("channel_profile", f"unlabeled channel MCS {prior_ul_mcs} outside "
                                                               f"measured [{min(mcs_values)}, {max(mcs_values)}]")
        ranked = self._scene_distance(stratum, camera_si, radar_p40)
        nearest = ranked[:min(self.k, len(ranked))]
        distance, chain_id = nearest[rng.randrange(len(nearest))]
        chain = self.chains[chain_id]
        vector = self.vectors[chain_id]
        donor_id = None
        opposite = [cid for _, cid in ranked if self.vectors[cid].realized_ordering is not vector.realized_ordering]
        if opposite:
            donor_id = opposite[0]
            donor = self.vectors[donor_id].costs
            own = vector.costs
            vector = vector.with_costs(RoleCostsV1(
                target_admit=own.target_admit if own.target_admit is not None else donor.target_admit,
                prediction_score=own.prediction_score if own.prediction_score is not None else donor.prediction_score,
                prediction_store=own.prediction_store if own.prediction_store is not None else donor.prediction_store,
                target_score=own.target_score if own.target_score is not None else donor.target_score,
                donor_chain_id=donor_id))
        delta = self._uplink_delta(chain, profile=profile, mcs=prior_ul_mcs, backlog=backlog_bytes, wire=wire_bytes)
        if delta:
            vector = vector.with_uplink(vector.E + delta)
        return ChainDrawV1(vector=vector, chain_id=chain_id, donor_chain_id=donor_id, uplink_delta_ms=delta,
                           scene_distance=distance, stratum_size=len(stratum))


def load_binding(path: Path, **kwargs) -> CalibrationSetV1:
    path = Path(path)
    _require(path.is_file(), f"calibration binding missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationBindingError(f"binding is not JSON: {exc}") from exc
    return CalibrationSetV1(document, **kwargs)


def candidate_source_digest() -> str:
    files = sorted(p for p in PACKAGE_DIR.glob("*.py") if not p.name.startswith("test_"))
    return canonical_sha256({p.name: sha256_file(p) for p in files})


def require_launch_approval(calibration: CalibrationSetV1) -> None:
    """Training launch requires an explicit approval naming this binding and this candidate source."""
    if not APPROVAL_PATH.is_file():
        raise LaunchBlockedError("corrected model and calibration are not approved (no APPROVAL.json); launch blocked")
    approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    if approval.get("calibration_binding_sha256") != calibration.binding_sha256:
        raise LaunchBlockedError("approval names a different calibration binding")
    if approval.get("candidate_source_digest") != candidate_source_digest():
        raise LaunchBlockedError("approval names a different candidate source")
    if not approval.get("approved_by") or not approval.get("approved_at"):
        raise LaunchBlockedError("approval lacks approver or date")
    blockers = calibration.launch_blockers()
    if sorted(approval.get("waived_blockers", [])) != sorted(blockers):
        raise LaunchBlockedError(f"approval does not explicitly waive the current blockers {blockers}")
