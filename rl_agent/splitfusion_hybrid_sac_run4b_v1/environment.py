"""Run-4B modeled offline environment over registered evidence.

One persistent causal session per seed.  Scene, MCS and queue dynamics are
the Run-4 registered sources, reused unchanged (FIT scene catalog, sealed MCS
Markov provider, transport-v2b queue head).  Latency is the shared
operational provider ``L_op = A + S + T + E + D``.

Per decision, in causal order:

1. the context (scene, held scene, transport success uniform, A/E/D draw) is
   drawn from independent streams before the action is known;
2. the 20-feature state is built from current measurements and the
   operational prior only;
3. the action resolves the outcome; Q_perc enters the reward only;
4. the next prior records action, mode, q and operational ACK latency;
5. backlog and MCS advance exactly once, then the next context is drawn.

Every transition continues the session with duration 2 and discount 0.99^2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from rl_agent.splitfusion_hybrid_sac_run4_v1 import dynamic_mcs_273prb_evidence as MCSEV
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_acceptance as MCSA
from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_provider as MCSP
from rl_agent.splitfusion_hybrid_sac_v1 import action_contract
from rl_agent.splitfusion_operational_latency_v1 import provider as OPL
from rl_agent.ue_production_transport_model_v2 import contract_v2 as C2
from rl_agent.ue_production_transport_model_v2 import scene_source as SS

from . import contract as C

ENV_SCHEMA = "splitfusion.run4b.modeled_environment.v1"
BACKLOG_LOG1P_SCALE = math.log1p(C2.RLC_AM_TX_ADMISSION_CEILING_BYTES)
MCS_ACCEPTANCE_RELPATH = ("rl_agent/splitfusion_hybrid_sac_run4_v1/"
                          "sealed_mcs_transition_v1/MCS_TRANSITION_ACCEPTANCE.json")
STREAM_OFFSETS = {"scene": 11, "held_scene": 67, "transport": 23, "mcs": 53}


class EnvironmentError_(RuntimeError):
    """A Run-4B environment invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EnvironmentError_(message)


def family_of_mode(mode_id: int) -> str:
    return ("noAE", "AE128", "AE64", "AE32")[mode_id // 3]


@dataclass
class SharedSourcesV1:
    """Immutable registered sources shared by every seed/restore."""

    provider: OPL.OperationalLatencyProviderV1
    catalog: SS.FitSceneCatalog
    mcs_model: Any
    mcs_evidence_sha256: str
    mcs_acceptance_sha256: str
    action_catalog: Any

    @classmethod
    def load(cls, provider: Optional[OPL.OperationalLatencyProviderV1] = None
             ) -> "SharedSourcesV1":
        C2.verify_preserved()
        C2.verify_oai_ceiling()
        MCSA.load_registered_acceptance()
        evidence = MCSEV.load_dynamic_mcs_273prb_evidence()
        return cls(
            provider=provider or OPL.OperationalLatencyProviderV1.load(),
            catalog=SS.FitSceneCatalog(),
            mcs_model=MCSP.fit_mcs_markov_model(evidence),
            mcs_evidence_sha256=evidence.canonical_evidence_sha256,
            mcs_acceptance_sha256=C2.sha256_file(
                C2.ROOT / MCS_ACCEPTANCE_RELPATH),
            action_catalog=action_contract.load_contract())

    def scaling(self) -> C.ScalingV1:
        values = [self.catalog.scene_descriptors(k)[0] for k in self.catalog.keys]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return C.ScalingV1(float(mean), float(max(1e-6, math.sqrt(variance))),
                           float(BACKLOG_LOG1P_SCALE))

    def binding_document(self) -> dict[str, Any]:
        return {
            "schema": ENV_SCHEMA,
            "operational_latency_provider_sha256": self.provider.binding_sha256,
            "operational_latency_label": OPL.LABEL,
            "transport_document_sha256":
                self.provider.transport_model.document_sha256,
            "scene_catalog_sha256": self.catalog.binding_sha256,
            "mcs_model_binding_sha256": self.mcs_model.binding_sha256,
            "mcs_evidence_sha256": self.mcs_evidence_sha256,
            "mcs_acceptance_sha256": self.mcs_acceptance_sha256,
            "action_catalog_sha256": action_contract.CATALOG_SHA256,
            "scaling_sha256": self.scaling().sha256,
            "stream_offsets": dict(STREAM_OFFSETS),
            "gamma": C.GAMMA, "duration": C.DURATION,
            "feature_schema_sha256": C.FEATURE_SCHEMA_SHA256,
            "reward_schema_sha256": C.REWARD_SCHEMA_SHA256,
        }

    @property
    def binding_sha256(self) -> str:
        return C.canonical_sha256(self.binding_document())


@dataclass(frozen=True, slots=True)
class TransitionV1:
    decision_seq: int
    state: tuple[float, ...]
    mode_id: int
    q_e4: int
    reward: float
    discount: float
    next_state: tuple[float, ...]
    diagnostics: Mapping[str, Any]

    def digest(self) -> str:
        return C.canonical_sha256({
            "decision_seq": self.decision_seq, "state": list(self.state),
            "mode_id": self.mode_id, "q_e4": self.q_e4,
            "reward": self.reward, "discount": self.discount,
            "next_state": list(self.next_state)})


def _rng_state_to_json(state: tuple) -> list:
    version, internal, gauss = state
    return [int(version), [int(x) for x in internal], gauss]


def _rng_state_from_json(value: list) -> tuple:
    return (int(value[0]), tuple(int(x) for x in value[1]),
            None if value[2] is None else float(value[2]))


class Run4BEnvironmentV1:
    """Deterministic causal session; complete JSON-serializable state."""

    def __init__(self, sources: SharedSourcesV1, *, seed: int) -> None:
        _require(type(seed) is int and seed >= 0, "seed must be an int >= 0")
        self.sources = sources
        self.seed = seed
        self.scaling = sources.scaling()
        base = seed * 1_000_003
        self._rng = {name: random.Random(base + offset)
                     for name, offset in STREAM_OFFSETS.items() if name != "mcs"}
        self._mcs = MCSP.FitMcsMarkovProviderV1(
            sources.mcs_model, seed=base + STREAM_OFFSETS["mcs"])
        self._latency_streams = sources.provider.streams(seed)
        self._backlog = 0
        self._mcs_current = int(self._mcs.reset())
        self.prior = C.OperationalPriorV1.genesis()
        self.decision_seq = 0
        self.context = self._new_context()

    # -- causal context --------------------------------------------------
    def _new_context(self) -> dict[str, Any]:
        catalog = self.sources.catalog
        reward_key = catalog.keys[self._rng["scene"].randrange(catalog.scene_count)]
        held_key = catalog.keys[
            self._rng["held_scene"].randrange(catalog.scene_count)]
        camera_si, radar_p40 = catalog.scene_descriptors(reward_key)
        draw = self.sources.provider.draw(self._latency_streams)
        return {"reward_scene_key": reward_key, "held_scene_key": held_key,
                "camera_si": float(camera_si), "radar_p40": float(radar_p40),
                "prior_ul_mcs": int(self._mcs_current),
                "backlog_bytes": int(self._backlog),
                "success_uniform": self._rng["transport"].random(),
                "latency_draw": draw.to_dict()}

    def observation(self) -> C.ObservationV1:
        ctx = self.context
        return C.ObservationV1(ctx["camera_si"], ctx["radar_p40"],
                               ctx["prior_ul_mcs"], ctx["backlog_bytes"])

    def current_features(self) -> tuple[float, ...]:
        return C.build_features(self.observation(), self.prior, self.scaling)

    # -- one decision ----------------------------------------------------
    def step(self, mode_id: int, q_e4: int) -> TransitionV1:
        catalog = self.sources.catalog
        executable = self.sources.action_catalog.resolve(
            mode_id, q_e4 / float(action_contract.Q_E4_SCALE))
        _require(executable.q_e4 == q_e4 and executable.mode.mode_id == mode_id,
                 "catalog changed the requested action")
        ctx = self.context
        state = self.current_features()
        reward_draw = catalog.draw(ctx["reward_scene_key"], mode_id=mode_id,
                                   q_e4=q_e4)
        held_draw = catalog.draw(ctx["held_scene_key"], mode_id=mode_id,
                                 q_e4=q_e4)
        outcome = self.sources.provider.resolve(
            draw=OPL.ComponentDrawV1(**ctx["latency_draw"]),
            wire_bytes=int(reward_draw.wire_bytes),
            pre_enqueue_backlog_bytes=ctx["backlog_bytes"],
            prior_ul_mcs=ctx["prior_ul_mcs"],
            success_uniform=ctx["success_uniform"])
        if outcome.timely:
            resolution = C.resolve_reward(
                kind=C.RewardKind.TIMELY_SUCCESS,
                q_perc=float(reward_draw.q_perc),
                operational_latency_ns=outcome.total_ns)
        else:
            resolution = C.resolve_reward(kind=C.RewardKind.TIMEOUT_OR_FAILURE)
        # Operational prior: never receives Q_perc.
        self.prior = C.OperationalPriorV1.from_outcome(
            mode_id=mode_id, q_e4=q_e4, timely=outcome.timely,
            operational_latency_ns=outcome.total_ns if outcome.timely else None)
        ingress = int(reward_draw.wire_bytes) + int(held_draw.wire_bytes)
        self._backlog = int(round(
            self.sources.provider.transport_model.predict_next_backlog_bytes(
                pre_enqueue_backlog_bytes=float(ctx["backlog_bytes"]),
                deterministic_action_ingress_bytes=ingress,
                prior_ul_mcs=ctx["prior_ul_mcs"])))
        self._mcs_current = int(self._mcs.step().successor_mcs)
        diagnostics = {
            "decision_seq": self.decision_seq, "mode_id": mode_id,
            "q_e4": q_e4, "family": family_of_mode(mode_id),
            "terminal": resolution.kind.value,
            "reward": resolution.reward,
            "q_perc_training_only": float(reward_draw.q_perc),
            "operational_latency_ns": (outcome.total_ns if outcome.timely
                                       else None),
            "composed_total_ns": outcome.total_ns,
            "components_ns": {"A": outcome.a_ns, "S": outcome.s_ns,
                              "T": outcome.t_ns, "E": outcome.e_ns,
                              "D": outcome.d_ns},
            "transport_success": outcome.transport_success,
            "on_time_probability": outcome.on_time_probability,
            "camera_si": ctx["camera_si"], "radar_p40": ctx["radar_p40"],
            "prior_ul_mcs": ctx["prior_ul_mcs"],
            "pre_action_backlog_bytes": ctx["backlog_bytes"],
            "reward_wire_bytes": int(reward_draw.wire_bytes),
            "held_wire_bytes": int(held_draw.wire_bytes),
        }
        self.decision_seq += 1
        self.context = self._new_context()
        return TransitionV1(
            decision_seq=self.decision_seq - 1, state=state, mode_id=mode_id,
            q_e4=q_e4, reward=float(resolution.reward), discount=C.DISCOUNT,
            next_state=self.current_features(), diagnostics=diagnostics)

    # -- durable state ---------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        mcs = self._mcs.checkpoint()
        return {
            "schema": ENV_SCHEMA, "seed": self.seed,
            "decision_seq": self.decision_seq,
            "backlog_bytes": self._backlog, "mcs_current": self._mcs_current,
            "prior": self.prior.to_dict(), "context": dict(self.context),
            "rng": {name: _rng_state_to_json(rng.getstate())
                    for name, rng in sorted(self._rng.items())},
            "mcs_provider": {"current_mcs": mcs.current_mcs,
                             "transitions_emitted": mcs.transitions_emitted,
                             "rng_state": _rng_state_to_json(mcs.rng_state)},
            "latency_streams": self._latency_streams.get_state(),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        _require(value["schema"] == ENV_SCHEMA, "environment schema differs")
        _require(int(value["seed"]) == self.seed, "environment seed differs")
        for name, state in value["rng"].items():
            self._rng[name].setstate(_rng_state_from_json(state))
        _require(set(value["rng"]) == set(self._rng), "RNG stream set differs")
        mcs = value["mcs_provider"]
        self._mcs.restore(MCSP.McsProviderCheckpointV1(
            model_binding_sha256=self.sources.mcs_model.binding_sha256,
            current_mcs=mcs["current_mcs"],
            transitions_emitted=mcs["transitions_emitted"],
            rng_state=_rng_state_from_json(mcs["rng_state"])))
        self._latency_streams.set_state(value["latency_streams"])
        self._backlog = int(value["backlog_bytes"])
        self._mcs_current = int(value["mcs_current"])
        self.prior = C.OperationalPriorV1.from_dict(dict(value["prior"]))
        self.decision_seq = int(value["decision_seq"])
        self.context = dict(value["context"])
