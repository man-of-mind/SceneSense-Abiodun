"""Reproduce the retained single-source integration blockers, without services.

This invokes the REAL GT sender, receiver, prediction serializer, memfd sealing,
feedback encoder and context/protocol validators. Only socket I/O and thread
startup are replaced. Predictions are explicitly synthetic CPU arrays, not
inference or quality evidence. A reproduced blocker is NOT a qualification pass.

Exit 2 means BLOCKED. No smoke/capture/registration action is implemented here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid


def gt_same_frame_probe():
    import numpy as np
    from rl_agent.splitfusion_exact_gt_live_v2 import edge as G
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import continuous_execution_v2 as X
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    from rl_agent.splitfusion_live_dispatch_v1 import dynamic_execution_contract as DC
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1

    contract = DC.load_dynamic_execution_contract()
    profile = contract.resolve_q_e4(11, 4321)
    labels = np.zeros((720, 1280), dtype=np.uint8)
    tickets = []
    for index, stream in enumerate(("m4_fixture_ego_a", "m4_fixture_ego_b")):
        context = build_frame_context_v1(stream_id=stream, frame_id=4172, sequence_id=1,
            capture_timestamp_ns=1_791_800_000_000_000_000,
            ego_world_x=15.0 * index, ego_world_y=0., ego_world_z=1.,
            ego_world_pitch=0., ego_world_yaw=0., ego_world_roll=0.)
        payload = b"synthetic-no-inference"
        envelope = X.ExecutionEnvelopeV3(mode_id=11, q_e4=4321,
            keep_count=profile.keep_count, anchor_action_id=profile.action_id,
            reward_requested=True, session_uuid=str(uuid.UUID(int=101 + index)),
            controller_lineage_sha256="1" * 64, decision_seq=0, ticket_seq=0,
            frame_id=context.frame_id, tensor_seq=context.sequence_id,
            capture_timestamp_ns=context.capture_timestamp_ns,
            execution_bundle_sha256=profile.execution_bundle_sha256,
            inner_payload_sha256=hashlib.sha256(payload).hexdigest(), inner_payload=payload)
        MP.validate_run4_identity(MP.run4_identity(envelope), contract=contract)
        X.pack_envelope_v3(envelope)
        tickets.append(G.E.EvaluationTicketV2(envelope=envelope, context=context,
            records=(), predicted_mask=labels,
            gt_identity=MP.gt_identity(run_id="m4_offline_fixture", cell_id="cell",
                stream_id=stream, frame_id=context.frame_id,
                anchor_action_id=profile.action_id, anchor_profile_id=profile.profile_id,
                capture_timestamp_ns=context.capture_timestamp_ns), enqueued_wall_ns=time.time_ns()))

    sent, feedback = [], []
    # This context only sets in-process bindings. No evaluator.start(), socket,
    # subprocess, model load, CARLA connection or GT worker is invoked.
    with G.install_early_gt_edge(socket_path="/nonexistent/m4-offline.sock",
                                capability_path="/nonexistent/m4-offline-capability.json"):
        evaluator = G.EarlyGtEvaluatorV1(spec=None, read_ground_truth=None,
            send=feedback.append, match_distance_m=2., gt_timeout_s=.01)
        evaluator._connect = lambda: None
        evaluator._receiver = SimpleNamespace(start=lambda: None)
        evaluator._send_service = lambda value, fd=None: sent.append(copy.deepcopy(value))
        for ticket in tickets:
            evaluator._materialized[ticket.context.frame_id] = G.raw()
            evaluator.submit(ticket)
        evaluator._queue.put(None)
        evaluator._send_loop()
        assert evaluator._failure is None, evaluator._failure
        predictions = [row for row in sent if row["op"] == "PREDICTION"]
        pending = [ticket.context.stream_id for ticket in evaluator._pending.values()]
        assert len(predictions) == 2 and pending == [tickets[1].context.stream_id]
        assert {row["frame_context"]["stream_id"] for row in predictions} == {
            ticket.context.stream_id for ticket in tickets}

        # The actual service UNSCORED response has only frame_id/reason, not a
        # source or session. Model the first request's response arriving first.
        response = dict(op="UNSCORED", frame_id=4172, reason="M4_OFFLINE_FIXTURE")
        with patch.object(G.CH, "receive", side_effect=[(response, None), G.CH.ChannelError("fixture EOF")]):
            evaluator._receive_loop()
        assert evaluator._failure is None, evaluator._failure
        assert len(feedback) == 1 and len(evaluator.records) == 1
        emitted_session = evaluator.records[0]["run4_identity"]["session_uuid"]
        assert emitted_session == tickets[1].envelope.session_uuid
        assert not evaluator._pending

        for ticket in tickets:
            evaluator.notify_terminal(ticket.context.frame_id, "DROPPED", "M4_FIXTURE")
        terminals = [evaluator._control.get_nowait() for _ in tickets]
        assert all(set(row) == {"op", "frame_id", "outcome", "stage", "raw_ns"} for row in terminals)
        assert terminals[0]["frame_id"] == terminals[1]["frame_id"]
        try:
            with G.install_early_gt_edge(socket_path="/nonexistent/second.sock",
                                        capability_path="/nonexistent/second.json"):
                raise AssertionError("nested binding unexpectedly admitted")
        except RuntimeError as error:
            assert str(error) == "nested early GT edge installation"
            nested_error = str(error)

    update = MP.build_run4_map_update(run_id="m4_offline_fixture", cell_id="cell",
        stream_id=tickets[0].context.stream_id, frame_id=4172, sequence_id=1,
        identity=MP.run4_identity(tickets[0].envelope), decoder_identity=profile.decoder_identity,
        capture_timestamp_ns=tickets[0].context.capture_timestamp_ns, records=[],
        service_deadline_at=1_791_800_000.17, ack_timeout_at=1_791_800_000.5,
        edge_timing={}, segmentation={"available": False})
    MP.validate_run4_map_update(update, contract=contract)
    return dict(status="COUNTEREXAMPLE_REPRODUCED", sent_prediction_count=len(predictions),
        submitted_streams=[t.context.stream_id for t in tickets],
        submitted_sessions=[t.envelope.session_uuid for t in tickets],
        shared_frame_id=4172, pending_after_two_sends=pending,
        pending_count_after_two_sends=len(pending),
        first_request_response_emitted_for_session=emitted_session,
        lost_first_source=True, terminal_messages_have_source=False,
        nested_binding_refusal=nested_error,
        limits="Synthetic predictions; transport arrival simulated; no live failure claimed."), update, contract


def map_probe(update, contract):
    from spatial_map_coop.coop_input_v1.adapter_direct_v1 import from_direct_object_map_update
    from spatial_map_coop.coop_input_v1.identity_config import CoopIdentityConfig
    from spatial_map_coop.coop_input_v1.normalized import CoopInputError
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP

    identity = CoopIdentityConfig.from_documents([dict(run_id="m4_offline_fixture", cell_id="cell",
        stream_id="m4_fixture_ego_a", ue_id="ue0", session_id="m4_fixture_session")])
    validated = []
    def upstream(document):
        MP.validate_run4_map_update(document, contract=contract)
        validated.append(True)
    try:
        from_direct_object_map_update(update, identity=identity, upstream_validator=upstream)
        raise AssertionError("legacy adapter unexpectedly accepted Run4")
    except CoopInputError as error:
        assert "schema drift" in str(error), str(error)
        assert validated == [True]
        refusal = str(error)
    try:
        identity.resolve(run_id="m4_offline_fixture", cell_id="cell", stream_id="m4_fixture_ego_b")
        raise AssertionError("unregistered source unexpectedly accepted")
    except CoopInputError:
        pass
    return dict(status="INCOMPATIBILITY_REPRODUCED", production_run4_validation="PASS",
        legacy_adapter_refusal=refusal, explicit_identity_wrong_source_rejection="PASS",
        continuous_q_e4=update["run4_identity"]["q_e4"],
        anchor_action_id=update["run4_identity"]["anchor_action_id"],
        wire_unchanged=True)


def telemetry_proofs():
    prefix = "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_ue_telemetry_provider_v2."
    names = [prefix + name for name in (
        "DciTest.test_rejected_classes_never_enter_cache",
        "RlcTest.test_cross_ue_rlc_rows_ignored_and_do_not_close_ticks",
        "DecisionTest.test_cross_session_and_cross_ue")]
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(names))
    assert result.wasSuccessful(), log.getvalue()
    return dict(status="PASS", tests=names, log=log.getvalue(),
        scope="Bound-provider CPU fixtures only, not two-UE attach/RNTI discovery or tracer fan-out.")


def retained_bindings(root):
    deployment = root.parent / "abiodun_run4_single_gpu_profiler_v1"
    registration = deployment / ("rl_agent/experiments/splitfusion_single_gpu_profiler_v1/"
        "20261012_m3_route_b_v20/good_all_live_view_01/registration.json")
    document = json.loads(registration.read_bytes())
    paths = ["rl_agent/splitfusion_exact_gt_live_v2/edge.py",
        "rl_agent/splitfusion_exact_gt_live_v2/service.py",
        "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/phase6_edge_runtime_v2.py",
        "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_map_protocol_v2.py",
        "rl_agent/splitfusion_hybrid_sac_live_route_b_v2/ue_telemetry_provider_v2.py",
        "rl_agent/splitfusion_run5_full_live_v1/runtime_policy.py"]
    rows = {}
    for path in paths:
        expected = set()
        if path in document["exact_gt_sources"]:
            expected.add(document["exact_gt_sources"][path])
        expected.update(row["sha256"] for row in document["experiment_binds"].values()
                        if row.get("path") == str(deployment / path))
        assert len(expected) == 1, f"missing or conflicting retained binding: {path}"
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        deployed = hashlib.sha256((deployment / path).read_bytes()).hexdigest()
        assert actual == deployed and actual in expected, f"retained source drift: {path}"
        rows[path] = actual
    return dict(registration=str(registration),
        registration_file_sha256=hashlib.sha256(registration.read_bytes()).hexdigest(),
        checked_sources=rows, status="SELECTED_RUNTIME_SOURCES_MATCH",
        scope="Not full prospective inventory or prerequisite verification; no M4 registration exists.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Create a NEW JSON file; existing evidence is never overwritten.")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new filename")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("run with CUDA_VISIBLE_DEVICES='' (CPU-only)")
    root = Path(__file__).resolve().parents[2]
    started = time.monotonic()
    bindings = retained_bindings(root)
    gt, update, contract = gt_same_frame_probe()
    result = dict(schema="splitfusion.m4_two_ue.boundary_probe.v1", status="BLOCKED",
        retained_binding=bindings, gt=gt, map=map_probe(update, contract), telemetry=telemetry_proofs())
    import torch
    assert not torch.cuda.is_initialized()
    free = shutil.disk_usage(root).free
    result.update(disk=dict(free_bytes=free, unchanged_launch_gate_bytes=16 * 1024**3,
                           shortage_bytes=max(0, 16 * 1024**3 - free)),
        cuda_initialized=False, live_services_started=False,
        duration_s=time.monotonic() - started,
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(Path(__file__).parent.glob("*.py"))},
        unqualified=["two-ego world ownership and both-stream drain", "independent live actor contexts",
                     "source-safe GT IPC and service", "source-routed feedback and map ingestion",
                     "two-UE attach, radio identity and achieved SNR", "two-stream analysis and launcher"])
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
