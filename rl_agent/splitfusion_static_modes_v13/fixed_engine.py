"""Actor-free external fixed-action engine seam for the v13 mode arms.

Versioned copy of ``splitfusion_m1_fixed_live_v2.fixed`` (qualified live in the two M1 fixed arms, 2026-10-07) with
only these differences:
  * the registered identity also binds ``ranker_bypassed`` (q = 0 arms) besides the M1 9-tuple;
  * the log directory is ``<attempt>/static_fixed_action/``;
  * the summary records the arm key.
Behaviour is unchanged: the retained hold path and controller run; at a decision opportunity the observational state
is still assembled (typed refusals are logged and the fixed action continues); the profile is resolved by the live
``resolve_q_e4``; every decision AND hold is checked against the registered identity and any mismatch raises; the
actor is never called; there is no fallback substitution and no clamp to the actor's support.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time

LOG_DIR = "static_fixed_action"
LOG_FILE = "FIXED_ACTIONS.jsonl"
SUMMARY_FILE = "FIXED_ACTION_SUMMARY.json"
SCHEMA = "splitfusion.static_modes_v13.fixed_scope.v1"


def raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class FixedBaselineError(RuntimeError):
    pass


def _builder_record(builder):
    cells = dict(zip(builder.__code__.co_freevars, (cell.cell_contents for cell in (builder.__closure__ or ()))))
    record = cells.get("record")
    return record if isinstance(record, dict) else None


class FixedLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.path, "x", buffering=1)
        self.lock = threading.Lock()
        self.counters = dict(decisions=0, holds=0, fallbacks=0, actor_calls=0, observational_state_ok=0,
                             observational_admission_refused=0, observational_reasons=0, executed_mismatch=0)

    def write(self, value):
        with self.lock:
            self.stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")

    def close(self):
        with self.lock:
            if not self.stream.closed:
                self.stream.flush()
                os.fsync(self.stream.fileno())
                self.stream.close()


def identity_tuple(profile):
    return (int(profile.mode_id), int(profile.q_e4), profile.action_id, profile.profile_id,
            profile.execution_bundle_sha256, int(profile.keep_count), int(profile.drop_count), profile.family,
            profile.quantizer)


def expected_tuple(fixed_action):
    return (fixed_action["mode_id"], fixed_action["q_e4"], fixed_action["action_id"], fixed_action["profile_id"],
            fixed_action["execution_bundle_sha256"], fixed_action["keep_count"], fixed_action["drop_count"],
            fixed_action["family"], fixed_action["quantizer"])


def make_engine_class(base, log, fixed_action):
    from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as state_contract
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import reward_hold_controller_v2 as R
    fixed_action = dict(fixed_action)

    class StaticFixedEngine(base):
        _static_fixed_v13_installed = True

        def plan_frame(self, *, frame_id, capture_wall_ns, now_raw_ns, build_state):
            if self.faulted is not None:
                raise E.EngineError(f"engine faulted: {self.faulted}")
            controller = self.controller
            if controller.current is not None and not controller.can_decide(now_raw_ns):
                plan = super().plan_frame(frame_id=frame_id, capture_wall_ns=capture_wall_ns,
                                          now_raw_ns=now_raw_ns, build_state=build_state)
                if plan.kind is not E.FrameKind.POLICY_HOLD:
                    raise FixedBaselineError("retained hold path produced a non-hold")
                log.counters["holds"] += 1
                _check(plan, log, fixed_action)
                log.write(_row(plan, observation="HOLD_NO_NEW_ADMISSION", reasons=[]))
                return plan
            if not self.new_opportunity_allowed:
                raise E.DecisionCapacityExhausted("no complete k_min decision cycle fits; stopping at a closed cycle")
            index = self._opportunity
            self._opportunity += 1
            try:
                previous = controller.previous_for_next_decision()
            except R.SessionBreakRequired:
                self._new_session()
                self.counters.session_rollovers += 1
                controller, previous = self.controller, None
            identity = controller.next_identity()
            reasons, observation, opened = [], "STATE_ASSEMBLED_OBSERVATION_ONLY", None
            try:
                guarded, _features = build_state(identity, previous)
                opened = int(guarded.boundary.action_open_timestamp_ns)
                log.counters["observational_state_ok"] += 1
            except state_contract.ExternalFallbackRequired as exc:
                reasons = tuple(str(exc).split("; ")) or ("UNSPECIFIED",)
                record = _builder_record(build_state)
                stamp = None if record is None else (record.get("action_open") or {}).get("ns")
                if stamp is None:
                    raise FixedBaselineError("typed observational refusal omitted the registered action-open record") \
                        from exc
                opened = int(stamp)
                observation = "ADMISSION_REFUSED_RECORDED_FIXED_ACTION_CONTINUES"
                log.counters["observational_admission_refused"] += 1
                log.counters["observational_reasons"] += len(reasons)
            profile = self._contract.resolve_q_e4(fixed_action["mode_id"], fixed_action["q_e4"])
            if bool(profile.execution_bundle.ranker_bypassed) is not bool(fixed_action["ranker_bypassed"]):
                raise FixedBaselineError("ranker-bypass identity differs from the registered fixed action")
            action = X.executed_identity_from_profile(profile, self._contract)
            controller.open_decision(identity=identity, action=action,
                                     execution_bundle_sha256=profile.execution_bundle_sha256, action_open_ns=opened)
            self._sync_tensor_seq(controller)
            assignment = controller.next_frame(frame_id=frame_id, capture_timestamp_ns=capture_wall_ns,
                                               now_ns=max(now_raw_ns, opened))
            self.counters.policy_decisions += 1
            self.opportunities.append(dict(
                opportunity_index=index, admitted=True, admission_semantics="FIXED_EXTERNAL",
                observational_admission_refused=bool(reasons), observational_reasons=list(reasons),
                frame_id=frame_id, decision_seq=identity.decision_seq, session_uuid=identity.session_uuid))
            plan = self._plan(E.FrameKind.POLICY_DECISION, assignment, profile, frame_id, capture_wall_ns, (), index,
                              None)
            log.counters["decisions"] += 1
            _check(plan, log, fixed_action)
            log.write(_row(plan, observation=observation, reasons=reasons, action_open_raw_ns=opened))
            return plan

        def coverage(self):
            total = len(self.opportunities)
            return dict(scored_opportunities=total, admitted=total, admitted_fraction=1.0 if total else 0.0,
                        fallback_fraction=0.0, min_fraction=None, verdict="FIXED_EXTERNAL_NOT_ACTOR_COVERAGE")

    return StaticFixedEngine


def _row(plan, *, observation, reasons, action_open_raw_ns=None):
    return dict(frame_id=int(plan.frame_id), tensor_seq=int(plan.tensor_seq), kind=plan.kind.value,
                mode_id=int(plan.profile.mode_id), q_e4=int(plan.profile.q_e4), action_id=plan.profile.action_id,
                profile_id=plan.profile.profile_id, execution_bundle_sha256=plan.profile.execution_bundle_sha256,
                actor_called=False, state_is_admission_gate=False, observation=observation,
                observational_reasons=list(reasons), action_open_raw_ns=action_open_raw_ns, raw_ns=raw())


def _check(plan, log, fixed_action):
    got, expected = identity_tuple(plan.profile), expected_tuple(fixed_action)
    if got != expected:
        log.counters["executed_mismatch"] += 1
        raise FixedBaselineError(f"executed fixed action mismatch: {got!r} expected {expected!r}")


@contextmanager
def install_fixed_engine(*, arm, fixed_action, attempt):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_decision_engine_v2 as E
    original = E.Run4DecisionEngineV2
    if getattr(original, "_static_fixed_v13_installed", False) or getattr(original, "_m1_fixed_v2_installed", False):
        raise FixedBaselineError("nested fixed-engine installation")
    log_path = Path(attempt) / LOG_DIR / LOG_FILE
    log = FixedLog(log_path)
    E.Run4DecisionEngineV2 = make_engine_class(original, log, fixed_action)
    state = dict(schema=SCHEMA, arm=arm, fixed_action=dict(fixed_action), actor_proposal_required=False,
                 actor_state_admission_required=False, fallback_substitution=False, clamp_to_actor_support=False)
    try:
        yield state
    finally:
        E.Run4DecisionEngineV2 = original
        state["counters"] = dict(log.counters)
        state["restored"] = E.Run4DecisionEngineV2 is original
        log.close()
        with open(log_path.with_name(SUMMARY_FILE), "x") as stream:
            json.dump(state, stream, indent=1, sort_keys=True)
