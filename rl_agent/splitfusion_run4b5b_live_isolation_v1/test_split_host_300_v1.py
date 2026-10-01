"""Composed offline tests for the split-host 300-frame B entry point.

The real factory start/execute/stop, the real hold gate inside the real
processor ``__call__``, the real ``BRouteBridgeV4`` at budget 300, the real
``execute_300`` loop and operational ledger, the real dependency builder and
the real teardown order run here.  Only external services (OAI, CARLA, the
remote edge, sockets, CUDA models) and the front/codec numerics are faked.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import json
import math
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import rl_agent
from rl_agent.splitfusion_hybrid_sac_run4_v1 import run4_contract as R4C
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as R4B

from . import b_edge_engineering_request_v2 as ER
from . import b_edge_runtime_v2 as EDGE
from . import b_one_frame_production_factory_v1 as P
from . import b_production_dependencies_v1 as D
from . import b_route_bridge_v4 as V4
from . import b_ue_process_v1 as UE
from . import final_actor_gate_v2 as F
from . import one_frame_engineering_v1 as O
from . import operational_ack_v1 as A
from . import postrun_artifact_v1 as ART
from . import split_host_300_v1 as S
from . import test_b_one_frame_composed_lifecycle_v1 as CL
from . import test_b_one_frame_production_factory_v1 as FX


def _config(base: Path, variant: str) -> S.Split300ConfigV1:
    one = FX.config(base, variant)
    registered = P.OLD.REMOTE_REPOSITORY
    values = {name: getattr(one, name)
              for name in S.Split300ConfigV1.__dataclass_fields__
              if name in one.__dataclass_fields__}
    values.update(
        run_id="split300_01", remote_repository=registered,
        edge_campaign_config=registered / "rl_agent/configs/campaign.json",
        remote_attempt_root=Path("/srv/attempts/split300_01"),
        transmitted_budget=300, purpose=S.PURPOSE,
        policy_performance_claim=False, maximum_loop_sim_s=600.0,
        factory_module=S.FACTORY_MODULE)
    return S.Split300ConfigV1(**values)


def _profile(mode_id: int, q_e4: int) -> SimpleNamespace:
    fallback = (mode_id, q_e4) == (S.FALLBACK["mode_id"], S.FALLBACK["q_e4"])
    return SimpleNamespace(
        mode_id=mode_id, q_e4=q_e4, keep_count=1000 + q_e4,
        action_id=S.FALLBACK["anchor_action_id"] if fallback else None,
        profile_id=S.FALLBACK["profile_id"] if fallback else None,
        family=S.FALLBACK["family"] if fallback else "AE64",
        quantizer=S.FALLBACK["quantizer"] if fallback else "UINT4",
        execution_bundle_sha256=hashlib.sha256(
            f"{mode_id}:{q_e4}".encode()).hexdigest())


def _features() -> tuple[float, ...]:
    return (0.1, 0.4, (20 - R4B.UL_MCS_MIN) / (R4B.UL_MCS_MAX - R4B.UL_MCS_MIN),
            math.log1p(12_345) / math.log1p(50_000_000)) + (0.0,) * 16


def make_processor(request: UE.BUEProcessRequestV1, timing: Path, *,
                   fallback_at=frozenset(), seen=None):
    """The real hold-gated ``__call__`` over faked state/front numerics."""
    proc = S.HoldGatedOpportunityProcessorV1.__new__(
        S.HoldGatedOpportunityProcessorV1)
    proc.request = request
    proc.gate = S.OperationalHoldGateV1()
    proc.timing_path = timing
    proc.counters = {kind: 0 for kind in S.FRAME_KINDS}
    proc._rows = 0
    proc.scaling = R4B.ScalingV1(camera_si_center=117.0, camera_si_scale=13.0,
                                 backlog_log1p_scale=math.log1p(50_000_000))
    proc.contract = SimpleNamespace(resolve_q_e4=_profile)
    proc.fallback_profile = _profile(S.FALLBACK["mode_id"], S.FALLBACK["q_e4"])
    proc.actor = SimpleNamespace(loaded=SimpleNamespace(module=object()))
    proc.transmitted = []
    proc.previous_seen = [] if seen is None else seen

    def features(opportunity, previous):
        proc.previous_seen.append(previous)
        if opportunity.sequence in fallback_at:
            raise R4C.ExternalFallbackRequired(
                "radio telemetry refused: PRIOR_UL_MCS_STALE")
        return _features()

    def transmit(opportunity, profile, mark, *, decision_seq, reward_requested):
        for name in ("input_7ch", "front_ae_codec", "pack_chunk",
                     "first_send", "last_send"):
            mark(name)
        proc.stage_timing["datagrams"] = 3
        identity = A.FrameActionIdentityV1(
            run_id=request.run_id, cell_id="a71__favorable_stable",
            stream_id="ue288_a71__favorable_stable",
            session_uuid=CL._Telemetry.session_uuid,
            controller_lineage_sha256=CL.LINEAGE,
            decision_seq=decision_seq, ticket_seq=decision_seq,
            frame_id=opportunity.frame_id, tensor_seq=opportunity.sequence,
            capture_timestamp_ns=opportunity.capture_timestamp_ns,
            mode_id=profile.mode_id, q_e4=profile.q_e4,
            keep_count=profile.keep_count,
            anchor_action_id=profile.action_id, profile_id=profile.profile_id,
            execution_bundle_sha256=profile.execution_bundle_sha256)
        sent = UE.BTransmissionV1(
            identity=identity,
            action_open_monotonic_raw_ns=opportunity.action_open_monotonic_raw_ns,
            payload_bytes=200_000 + profile.q_e4,
            decision_frame=reward_requested)
        proc.transmitted.append(sent)
        return sent

    proc._features = features
    proc._transmit = transmit
    return proc


_ACTOR_CALLS = [0]


def _fake_actor(module, features):
    _ACTOR_CALLS[0] += 1
    return 8, 4900 + (_ACTOR_CALLS[0] % 7)


def route_driver(activation: Path, *, fail_after=None):
    def driver(bridge):
        activation.touch(exist_ok=False)
        frame = 0
        while not bridge.stop_requested.is_set():
            if fail_after is not None and bridge.transmitted >= fail_after:
                raise RuntimeError("synthetic route failure")
            frame += 1
            opened = CL._now()
            try:
                bridge.offer_prepared(V4.RouteOpportunityV4(
                    sequence=bridge.transmitted, frame_id=frame,
                    capture_timestamp_ns=opened - 1_000_000,
                    action_open_monotonic_raw_ns=opened, submit_kwargs={}))
            except V4.OpportunitySuperseded:
                continue
            except (V4.RouteBudgetReached, V4.RouteStopped):
                return
    return driver


class _Receiver:
    """Every third decision times out; its ACK then arrives late, first,
    inside the next decision's window (the execute_300 regression)."""

    instances: list = []

    def __init__(self, edge):
        self.edge, self.closed = edge, False
        self.pending_late = None
        self.windows: dict[str, int] = {}
        self.decisions = 0
        _Receiver.instances.append(self)

    def receive_until(self, identity, deadline):
        digest = identity.exact_sha256()
        if digest not in self.windows:
            self.windows[digest] = self.decisions
            self.decisions += 1
            if self.pending_late is not None:
                packet, self.pending_late = self.pending_late, None
                return packet
        if self.windows[digest] % 3 == 2:
            # Arrives 5 ms after its own deadline, i.e. a genuinely late ACK.
            self.pending_late = (self.edge.ack(identity),
                                 deadline + 5_000_000)
            return None
        return self.edge.ack(identity), CL._now()

    def close(self):
        self.closed = True


class SplitHost300ComposedTest(unittest.TestCase):
    def compose(self, base: Path, variant: str, *, fallback_at=frozenset(),
                fail_after=None, evaluator_error=None):
        cfg = _config(base, variant)
        selected = FX.actor(variant)
        paths = P.attempt_paths(cfg)
        tracer = base / "tracer"; tracer.mkdir()
        (tracer / "T_messages.txt").write_text("T")
        edge = CL._Edge(base / "remote_host")
        system = CL._System(cfg.local_attempt_root, tracer)
        ops = S.RealSplitHost300OpsV1.__new__(S.RealSplitHost300OpsV1)
        ops.settings, ops.system = None, system
        ops.live_result = ops.postrun = ops.result_document = None
        timeline: list[str] = []
        built: dict = {}
        materialized: list = []
        evaluated: dict = {}

        def start_remote_edge(state):
            timeline.append("edge-start")
            built["edge_request"] = ER.decode_and_validate_300(
                state.edge_plan.request_b64)
            built["edge_command"] = list(
                state.edge_plan.compose["services"]["oai-perception-rx"]["command"])
            edge.bind(state.edge_plan.runtime_root)

        def ssh(argv, timeout_s=60.0, data=None):
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        def checked_ssh(argv, label, timeout_s=60.0):
            timeline.append(label)
            return b"edge log\n"

        def build_pipeline(**kwargs):
            built.update(kwargs)
            request = kwargs["request"]
            processor = make_processor(request, kwargs["timing_path"],
                                       fallback_at=fallback_at)
            built["processor"] = processor

            def materializer(spool_root, evidence_root):
                timeline.append("gt-materialize")
                materialized.append(Path(spool_root))
                store = ART.GroundTruthEvidenceStoreV1.create(evidence_root)
                for sent in processor.transmitted:
                    store.write(identity=sent.identity, eligible_objects=(),
                                semantic_mask=np.zeros((2, 2), dtype=np.uint8),
                                recorded_monotonic_raw_ns=CL._now())
                return len(processor.transmitted)

            return V4.BRouteBridgeV4(
                variant=request.variant,
                feature_schema_sha256=request.feature_schema_sha256,
                actor_boundary_sha256=request.actor_boundary_sha256,
                processor=processor,
                route_driver=route_driver(
                    Path(kwargs["route_kwargs"]["campaign"]["_target_start_file"]),
                    fail_after=fail_after),
                raw_spool_root=kwargs["raw_spool_root"],
                postrun_materializer=materializer, transmitted_budget=300)

        def download(state, local):
            timeline.append("prediction-download")
            local.mkdir(exist_ok=False)
            shutil.copytree(edge.local_path(state.edge_plan.runtime_root
                                            / "prediction"),
                            local / "prediction")
            return local / "prediction"

        class Evaluator:
            def __init__(self, *, reward_spec):
                self.reward_spec = reward_spec

        def evaluate(evaluator, *, operational_trace_root, prediction_root,
                     ground_truth_root, output_root):
            timeline.append("evaluate")
            if evaluator_error is not None:
                raise evaluator_error
            evaluated.update(
                trace=operational_trace_root, prediction=prediction_root,
                gt_ids={r.identity_sha256 for r in
                        ART.GroundTruthEvidenceStoreV1.open_existing(
                            ground_truth_root).verify_all()})
            output_root.mkdir()
            return SimpleNamespace(frame_count=len(evaluated["gt_ids"]),
                                   quality_defined_count=0,
                                   summary_path=output_root / "summary.json")

        import torch
        from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as BASE
        from . import postrun_evaluator_v1 as EV
        from . import postrun_operational_population_v1 as POP
        from . import registered_quality_v1 as RQ
        ue_stub = SimpleNamespace(_front=object(), _ranker=object(),
                                  _ae_encoders={}, _codec=object())
        pinned = CL._Pinned()
        patches = [
            mock.patch.dict(sys.modules, {
                "rl_agent.ue_route_b_split_cell_adapter_v1": pinned}),
            mock.patch.object(rl_agent, "ue_route_b_split_cell_adapter_v1",
                              pinned, create=True),
            mock.patch.object(P.CO, "validate_campaign_binding"),
            mock.patch.object(ops, "_start_remote_edge", start_remote_edge),
            mock.patch.object(ops, "_ssh", ssh),
            mock.patch.object(ops, "_checked_ssh", checked_ssh),
            mock.patch.object(ops, "_start_map", lambda campaign, root, port, **_: (
                (Path(root) / "map").mkdir(exist_ok=False)
                or SimpleNamespace(poll=lambda: None))),
            mock.patch.object(ops, "_download_predictions", download),
            mock.patch.object(S, "build_split_host_300_pipeline", build_pipeline),
            mock.patch.object(S.BASE, "_actor_action", _fake_actor),
            mock.patch.object(S, "ThreadedOperationalAckReceiverV1",
                              lambda host, port: _Receiver(edge)),
            mock.patch.object(EV, "PostRunEvaluatorV1", Evaluator),
            mock.patch.object(POP, "evaluate_from_operational_trace", evaluate),
            mock.patch.object(RQ, "load_registered_quality_spec",
                              lambda root: "spec"),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(BASE, "_preload_ue", return_value=(ue_stub, None, [])),
            mock.patch.object(D.DEC, "load_dynamic_execution_contract",
                              return_value=object()),
            mock.patch.object(D.X, "ContinuousUERuntimeV2", lambda *a, **k: object()),
            mock.patch.object(D.P6, "_SqueezedFront", lambda *a: object()),
            mock.patch.object(D.T, "UeTelemetryProviderV2", CL._Telemetry),
            mock.patch.object(D.T, "CausalClockBridgeV2", lambda: None),
            mock.patch.object(D.T, "LiveEventReaderV2", CL._Reader),
            mock.patch.object(D.T, "csv_reader_argv", lambda *a: ["fake"]),
            mock.patch.object(D.PW, "warm_ue", lambda *a, **k: {
                "completed": True, "modes_warmed": list(range(12)), "paths": []}),
            mock.patch.object(D.socket, "socket", CL._Socket),
        ]
        return SimpleNamespace(
            cfg=cfg, selected=selected, paths=paths, edge=edge, system=system,
            ops=ops, timeline=timeline, built=built, materialized=materialized,
            evaluated=evaluated, patches=patches)

    def run_full(self, variant: str, check, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            c = self.compose(Path(directory), variant, **kwargs)
            with contextlib.ExitStack() as stack:
                for patcher in c.patches:
                    stack.enter_context(patcher)
                lifecycle = S.ProductionSplitHost300LifecycleV1(None, ops=c.ops)
                lifecycle.start(c.cfg, c.selected)
                state = lifecycle.state
                self.check_after_start(c, state)
                keepalive = state.dependencies.uplink_keepalive
                live = lifecycle.execute(c.cfg, c.selected)
                self.assertTrue(keepalive._thread.is_alive(),
                                "keepalive stopped during the run")
                self.assertFalse(keepalive._stop.is_set())
                self.assertNotIn("edge stop", c.timeline)
                self.assertNotIn("gt-materialize", c.timeline)
                operational = self.digest_tree(c.paths.get("ue_evidence")
                                               / "operational_evidence")
                gate_after_live = (state.pipeline.processor.gate.active,
                                   state.pipeline.processor.gate.tensors)
                lifecycle.stop(c.cfg)
                # post-run is after service teardown and before the CARLA stop
                order = [item for item in c.timeline if item in {
                    "edge stop", "gt-materialize", "prediction-download",
                    "evaluate"}]
                self.assertEqual(order[0], "edge stop")
                self.assertEqual(c.system.events[-1],
                                 f"carla-stop:{P.CARLA_STOP_GRACE_S}")
                self.assertFalse(keepalive._thread.is_alive())
                # post-run analysis cannot change operational evidence/state
                self.assertEqual(operational, self.digest_tree(
                    c.paths.get("ue_evidence") / "operational_evidence"))
                processor = c.built["processor"]
                self.assertEqual(gate_after_live,
                                 (processor.gate.active, processor.gate.tensors))
                self.assertLessEqual(CL._present_top(c.paths), CL._declared(c.paths))
                check(c, live, c.built["processor"])

    @staticmethod
    def digest_tree(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def check_after_start(self, c, state) -> None:
        for phase in ("execute", "postrun"):
            for owned in c.paths.phase(phase):
                self.assertFalse(owned.exists(), owned)
        self.assertEqual(state.ue_request.transmitted_budget, 300)
        state.ue_request.validate()
        self.assertEqual(c.built["edge_request"]["transmitted_budget"], 300)
        self.assertEqual(c.built["edge_request"]["schema"], ER.SCHEMA_300)
        self.assertEqual(c.built["edge_request"]["split_host"][
            "ack_receiver_host"], O.UE_TUNNEL_IP)
        self.assertEqual(c.built["route_kwargs"]["maximum_loop_sim_s"], 600.0)
        self.assertEqual(c.built["timing_path"],
                         c.paths.get("frame_stage_timing"))
        self.assertIs(state.pipeline.processor, c.built["processor"],
                      "the 300 path must not wrap the processor")

    def check_run(self, c, live, processor, *, fallbacks: int) -> None:
        live.validate(c.cfg)
        self.assertEqual(live.transmitted_frames, 300)
        self.assertEqual(live.fallback_frames, fallbacks)
        self.assertEqual(live.policy_decisions + live.held_frames
                         + live.fallback_frames, 300)
        rows = [json.loads(line) for line in
                c.paths.get("frame_stage_timing").read_text().splitlines()]
        self.assertEqual(live.held_frames, live.policy_decisions - (
            1 if rows[-1]["kind"] == "POLICY_DECISION" else 0))
        self.assertEqual([row["frame_index"] for row in rows], list(range(300)))
        decisions = [row for row in rows if row["kind"] == "POLICY_DECISION"]
        self.assertEqual(len(decisions), live.policy_decisions)
        self.assertEqual([row["decision_seq"] for row in decisions],
                         list(range(len(decisions))))
        active = None
        for row in rows:
            if row["kind"] == "POLICY_DECISION":
                active = row
                self.assertEqual(row["observation"]["prior_ul_mcs"], 20)
                self.assertEqual(row["observation"]["pre_action_rlc_backlog_bytes"],
                                 12_345)
            elif row["kind"] == "POLICY_HOLD":
                for key in ("decision_seq", "mode_id", "q_e4", "keep_count",
                            "profile_id"):
                    self.assertEqual(row[key], active[key], key)
                self.assertEqual(row["previous_identity_sha256"],
                                 active["identity_sha256"])
            else:
                self.assertEqual((row["mode_id"], row["q_e4"]), (11, 9800))
                self.assertEqual(row["decision_seq"], S.FALLBACK_SEQ)
                self.assertTrue(row["fallback_reasons"])
        # a hold always follows a decision; a decision never follows a
        # decision without K_MIN = 2 tensors of the earlier action
        kinds = [row["kind"] for row in rows]
        for index, kind in enumerate(kinds[:-1]):
            if kind == "POLICY_DECISION":
                self.assertEqual(kinds[index + 1], "POLICY_HOLD")
        # each decision's state prior is the previous decision's outcome
        previous_decision = None
        for row in decisions:
            self.assertEqual(row["previous_identity_sha256"], previous_decision)
            previous_decision = row["identity_sha256"]
        # exact terminal reconciliation (every third decision timed out and
        # its late ACK arrived first inside the next decision's window)
        expected_timeouts = sum(1 for k in range(live.policy_decisions)
                                if k % 3 == 2)
        self.assertEqual(live.operational_timeouts, expected_timeouts)
        self.assertEqual(live.operational_successes,
                         live.policy_decisions - expected_timeouts)
        late_after = expected_timeouts - (1 if (live.policy_decisions - 1) % 3 == 2
                                          else 0)
        self.assertEqual(live.late_orphan_acks, late_after)
        snapshot = A.OperationalEvidenceStoreV1.open_existing(
            c.paths.get("ue_evidence") / "operational_evidence"
        ).verify_all(require_all_resolved=True)
        decision_ids = {row["identity_sha256"] for row in decisions}
        self.assertEqual({o.identity.exact_sha256() for o in snapshot.outcomes},
                         decision_ids)
        for outcome in snapshot.outcomes:
            if outcome.success:
                self.assertLessEqual(outcome.observed_latency_ns, A.ACK_DEADLINE_NS)
                self.assertIsNotNone(outcome.tail_output_sha256)
        report = json.loads((c.paths.get("ue_output") / UE.REPORT_NAME).read_text())
        self.assertFalse(report["live_qperc_computed"])
        self.assertFalse(report["gt_used_for_ack_or_state"])
        self.assertEqual(report["ground_truth_records"], 0)
        # raw spool: identities for all 300 frames, nothing processed live
        spool = c.paths.get("raw_gt_spool")
        self.assertEqual(len(list((spool / "identity").glob("*.json"))), 300)
        self.assertEqual(c.materialized, [spool])
        # durable result, decision-only scoring population
        result = json.loads(c.paths.get("split_host_300_result").read_text())
        self.assertEqual(result["live"]["transmitted_frames"], 300)
        self.assertEqual(result["operational_status"], "LIVE_300_COMPLETE")
        return decision_ids, result

    def test_run4b_300_frames_hold_gated_and_postrun(self) -> None:
        def check(c, live, processor):
            decision_ids, result = self.check_run(c, live, processor,
                                                  fallbacks=0)
            self.assertEqual(live.policy_decisions, 150)
            self.assertEqual(result["postrun"]["status"],
                             "QUALITY_ANALYSIS_COMPLETE")
            self.assertEqual(c.evaluated["gt_ids"], decision_ids)
            self.assertEqual(result["postrun"]["ground_truth_decisions"][
                "all_frame_gt_records"], 300)
            self.assertEqual(result["postrun"]["predictions"]["records"],
                             live.policy_decisions)
            self.assertEqual(c.edge.acks, live.policy_decisions)
        self.run_full(F.RUN4B_VARIANT, check)

    def test_run5b_300_frames_with_registered_fallbacks(self) -> None:
        # 10 and 11 are consecutive refused opportunities; 58 is the next
        # even decision opportunity after them.
        def check(c, live, processor):
            self.check_run(c, live, processor, fallbacks=3)
        self.run_full(F.RUN5B_VARIANT, check,
                      fallback_at=frozenset({10, 11, 58}))

    def test_postrun_failure_never_rewrites_the_live_run(self) -> None:
        def check(c, live, processor):
            _ids, result = self.check_run(c, live, processor, fallbacks=0)
            self.assertEqual(result["postrun"]["status"],
                             "QUALITY_ANALYSIS_INCOMPLETE")
            self.assertIn("scorer exploded",
                          result["postrun"]["evaluation"]["error"])
            self.assertEqual(result["live"], dataclasses.asdict(live))
        self.run_full(F.RUN4B_VARIANT, check,
                      evaluator_error=RuntimeError("scorer exploded"))

    def test_failure_mid_run_still_tears_down_and_writes_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            c = self.compose(Path(directory), F.RUN4B_VARIANT, fail_after=40)
            with contextlib.ExitStack() as stack:
                for patcher in c.patches:
                    stack.enter_context(patcher)
                lifecycle = S.ProductionSplitHost300LifecycleV1(None, ops=c.ops)
                with mock.patch.object(S, "preflight", return_value=c.selected):
                    with self.assertRaises(BaseException):
                        S.run(c.cfg, lifecycle)
                self.assertEqual(c.system.events[-1],
                                 f"carla-stop:{P.CARLA_STOP_GRACE_S}")
                self.assertIn("edge stop", c.timeline)
                self.assertNotIn("gt-materialize", c.timeline)
                self.assertFalse(c.paths.get("split_host_300_result").exists())
                self.assertFalse(c.paths.get("postrun_quality").exists())


class HoldGateTest(unittest.TestCase):
    def identity(self, seq: int) -> A.FrameActionIdentityV1:
        return A.FrameActionIdentityV1(
            run_id="r", cell_id="c", stream_id="s",
            session_uuid=CL._Telemetry.session_uuid,
            controller_lineage_sha256=CL.LINEAGE, decision_seq=seq,
            ticket_seq=seq, frame_id=seq, tensor_seq=seq,
            capture_timestamp_ns=1, mode_id=8, q_e4=5000, keep_count=9000,
            anchor_action_id=None, profile_id=None,
            execution_bundle_sha256="b" * 64)

    def outcome(self, identity) -> A.OperationalOutcomeV1:
        return A.OperationalOutcomeV1(
            identity=identity, terminal=A.OperationalTerminal.TIMEOUT,
            action_open_monotonic_raw_ns=10,
            resolution_monotonic_raw_ns=10 + A.ACK_DEADLINE_NS + 1,
            observed_latency_ns=None, state_latency_ns=0,
            accepted_ack_sha256=None, tail_output_sha256=None)

    def test_registered_k_min_hold(self) -> None:
        gate = S.OperationalHoldGateV1()
        self.assertFalse(gate.must_hold(None))
        first = self.identity(0)
        gate.opened(S.ActiveDecisionV1(first, object(), 0))
        self.assertTrue(gate.must_hold(self.outcome(first)))
        gate.held()
        self.assertFalse(gate.must_hold(self.outcome(first)))

    def test_unresolved_or_foreign_resolution_fails_closed(self) -> None:
        gate = S.OperationalHoldGateV1()
        first = self.identity(0)
        gate.opened(S.ActiveDecisionV1(first, object(), 0))
        with self.assertRaisesRegex(S.SplitHost300Error, "unresolved"):
            gate.must_hold(None)
        with self.assertRaisesRegex(S.SplitHost300Error, "unresolved"):
            gate.must_hold(self.outcome(self.identity(1)))
        with self.assertRaisesRegex(S.SplitHost300Error, "contiguous"):
            gate.opened(S.ActiveDecisionV1(self.identity(5), object(), 5))

    def test_k_min_is_the_registered_constant(self) -> None:
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
            reward_hold_controller_v2 as RHC)
        self.assertEqual(S.K_MIN, RHC.K_MIN)
        self.assertEqual(S.K_MIN, 2)


class ContractTest(unittest.TestCase):
    def test_config_seal_round_trip_and_exact_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(Path(directory), F.RUN4B_VARIANT)
            path = Path(directory) / "seal.json"
            path.write_text(json.dumps(S.seal(cfg)))
            self.assertEqual(S.load_config(path), cfg)
            with self.assertRaisesRegex(S.SplitHost300Error, "exactly 300"):
                dataclasses.replace(cfg, transmitted_budget=1)
            with self.assertRaisesRegex(S.SplitHost300Error, "safety bound"):
                dataclasses.replace(cfg, maximum_loop_sim_s=60.0)
            with self.assertRaises(O.OneFrameEngineeringError):
                O.load_config(path)

    def test_edge_requests_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(Path(directory), F.RUN4B_VARIANT)
            selected = FX.actor(F.RUN4B_VARIANT)
            encoded, raw = S.build_edge_request_300(cfg, selected)
            self.assertEqual(raw["transmitted_budget"], 300)
            with self.assertRaises(ER.EngineeringRequestError):
                ER.decode_and_validate(encoded)
            runtime = EDGE._validated_request(encoded)
            self.assertEqual((runtime["purpose"], runtime["claim_scope"],
                              runtime["transmitted_budget"]),
                             (ER.PURPOSE_300, ER.CLAIM_SCOPE_300, 300))

            def encode(value):
                payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")
                return base64.urlsafe_b64encode(payload).decode().rstrip("=")
            for change in ({"transmitted_budget": 1},
                           {"transmitted_budget": 299},
                           {"transmitted_budget": True},
                           {"purpose": ER.PURPOSE},
                           {"claim_scope": ER.CLAIM_SCOPE}):
                with self.subTest(change=change):
                    with self.assertRaises(ER.EngineeringRequestError):
                        ER.decode_and_validate_300(encode({**raw, **change}))
            one = encode({**raw, "schema": ER.SCHEMA, "purpose": ER.PURPOSE,
                          "claim_scope": ER.CLAIM_SCOPE,
                          "transmitted_budget": 300})
            with self.assertRaisesRegex(ER.EngineeringRequestError,
                                        "exactly one"):
                ER.decode_and_validate(one)

    def test_ue_request_is_the_frozen_300_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(Path(directory), F.RUN4B_VARIANT)
            request = S.build_ue_request_300(cfg, FX.actor(F.RUN4B_VARIANT))
            request.validate()
            self.assertEqual(request.deadline_ns, 170_000_000)

    def test_ops_hooks_bind_300_and_leave_one_frame_defaults(self) -> None:
        ops = S.RealSplitHost300OpsV1.__new__(S.RealSplitHost300OpsV1)
        self.assertEqual(ops._edge_ready_scope(),
                         (ER.PURPOSE_300, ER.CLAIM_SCOPE_300, 300))
        base = P.RealProductionOpsV1.__new__(P.RealProductionOpsV1)
        self.assertEqual(base._edge_ready_scope(),
                         (ER.PURPOSE, ER.CLAIM_SCOPE, 1))
        self.assertEqual(base._maximum_loop_sim_s(None), P.SAFETY_TIMEOUT_S)

    def test_live_module_imports_no_postrun_scorer_at_import_time(self) -> None:
        source = Path(S.__file__).read_text()
        header = "\n".join(
            line for line in source[:source.index("CONFIG_SCHEMA =")].splitlines()
            if line.startswith(("import ", "from ", "    ")))
        for name in ("postrun_", "registered_quality", "scoring",
                     "offline_quality_grid"):
            self.assertNotIn(name, header)


class ExecuteThreeHundredAckWaitTest(unittest.TestCase):
    """execute_300 must not close a ticket because another ACK came first."""

    def test_foreign_ack_first_does_not_time_out_the_current_ticket(self) -> None:
        coordinator = __import__(
            "rl_agent.splitfusion_run4b5b_live_isolation_v1.branch_evidence_v1",
            fromlist=["x"]).TailOutputBranchCoordinatorV1()
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(Path(directory), F.RUN4B_VARIANT)
            request = S.build_ue_request_300(cfg, FX.actor(F.RUN4B_VARIANT))
            request = dataclasses.replace(
                request, output_root=Path(directory) / "out",
                evidence_root=Path(directory) / "ev")
            proc = make_processor(request, Path(directory) / "t.jsonl")
            stack = contextlib.ExitStack()
            self.addCleanup(stack.close)
            stack.enter_context(mock.patch.object(S.BASE, "_actor_action",
                                                  _fake_actor))
            sent_frames = []

            class Pipeline:
                variant = request.variant
                feature_schema_sha256 = request.feature_schema_sha256
                actor_boundary_sha256 = request.actor_boundary_sha256

                def transmit_next(self, index, previous):
                    opened = CL._now()
                    item = V4.RouteOpportunityV4(
                        sequence=index, frame_id=index + 1,
                        capture_timestamp_ns=opened - 1,
                        action_open_monotonic_raw_ns=opened, submit_kwargs={})
                    sent = proc(item, previous)
                    sent_frames.append(sent)
                    return sent

                def close(self):
                    pass

            class Receiver(_Receiver):
                def __init__(self):
                    super().__init__(SimpleNamespace(ack=lambda identity: A.encode_ack(
                        coordinator.publish(identity, b"t" + identity.exact_sha256().encode()).ack)))

            UE.execute_300(request, Pipeline(), Receiver())
            report = json.loads((request.output_root / UE.REPORT_NAME).read_text())
            decisions = report["policy_decisions"]
            self.assertEqual(decisions, 150)
            self.assertEqual(report["operational_timeouts"],
                             sum(1 for k in range(decisions) if k % 3 == 2))


if __name__ == "__main__":
    unittest.main()
