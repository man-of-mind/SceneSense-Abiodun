"""Real-process, multi-frame CPU rehearsal of the prospective live adapters.

Retained sensor/prediction bytes; actual current RAW owner events, private IPC,
named same-UID inbox, retained feedback encoder and fsynced archives. No radio,
CARLA server, models or GPU. This does not establish live latency performance.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import time
from types import SimpleNamespace

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import replay as H, fixtures as F
from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import KEYS, quality_bridge
from rl_agent.splitfusion_live_parity_runtime_v1 import pose as PO
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from . import edge_live as L, ipc as I
from .process import Client, LIVE_PROCESS_CONTRACT
from .staging import encode_stage
from .live_clock import read_boot_id
from .replay import inventory, OUTPUT_ROOT


def event(role, name, instant, pid=None):
    return dict(role=role, name=name, raw_ns=instant,
                pid=os.getpid() if pid is None else pid, boot_id=read_boot_id())


def run(output):
    output = Path(output).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(OUTPUT_ROOT),
              "fresh owned rehearsal output required")
    output.mkdir(parents=True, exist_ok=False)
    data = H.load_inputs(4)
    sources = inventory(data)
    binding = C.sha256(C.canonical(dict(scope="CPU_LIVE_ADAPTER_REHEARSAL_NOT_RADIO", sources=sources)))
    H.write(output / "PLAN.json", dict(source_sha256=sources, retained_input_sha256=data["inputs"],
        frames=4, max_retained_jobs=2, clocks="actual RAW harness events; no radio simulation claim",
        live_services_started=False, performance_claim=False))
    producer = scorer = archive = evaluator = None
    rows = []
    try:
        producer = Client("producer", dict(run_binding=binding,
            producer_binding=data["startup"]["binding_sha256"],
            prediction_contract=data["prediction_contract"], max_frames=4,
            producer_args={key: str(value) if isinstance(value, Path) else value
                           for key, value in K.producer_arguments(data["cases"][0]).items()},
            offline_replay=False, live_process_contract=LIVE_PROCESS_CONTRACT),
            directory=output / "producer", source_sha256=sources, cpu=4)
        scorer = Client("scorer", dict(run_binding=binding,
            producer_binding=data["startup"]["binding_sha256"], max_jobs=2,
            max_bytes=128*1024*1024), directory=output / "scorer", source_sha256=sources, cpu=5)
        archive_root, sidecars = output / "archive", output / "sidecars"
        archive_root.mkdir(); sidecars.mkdir()
        archive = ArchiveClient(roots=[archive_root], max_jobs=2,
            max_bytes=128*1024*1024, max_frames=4,
            binding=dict(numeric_run_binding=binding,
                         numeric_worker_source_digest=C.sha256(C.canonical(sources))))
        payloads = queue.Queue()
        socket_path = output / "numeric_gt.sock"
        with L.install_numeric_edge(socket_path=socket_path, scorer_client=scorer,
                archive_client=archive, archive_root_index=0, sidecar_directory=sidecars,
                max_frames=4, terminal_timeout_s=5.) as edge_state:
            spec, _ = quality_bridge()
            evaluator = E.Run4EvaluatorV2(spec=spec, read_ground_truth=None,
                send=payloads.put, match_distance_m=2., gt_timeout_s=2.)
            evaluator.start()
            for job, case in enumerate(data["cases"]):
                observed = I.raw()
                pred = data["predictions"][case.frame_id]
                expected = H.expectation(data, case, binding)
                plan = F.plan_for(case, pred, expected["frame_context"])
                chunks, extras = encode_stage({key: case.kwargs[key] for key in KEYS},
                                              case.kwargs["policy_timestamp_s"])
                staged = producer.call("STAGE_SOURCE", dict(job=job,
                    source_packet=b"".join(chunks), stage_extras=extras))
                opened = I.raw()
                prepared = producer.call("PREPARE_STAGED", dict(job=job, plan_bytes=plan,
                    capture_identity=case.kwargs["identity"],
                    frame_context=PO.context_document(expected["frame_context"]),
                    population=case.evidence["dynamic_snapshot_evidence"],
                    archive_clocks=None, payload_filenames=None, prescore_filename=None))
                # Harness handoff boundary, not a claim of a transmitted radio frame.
                sent = I.raw()
                producer.call("CONFIRM_SEND", dict(job=job, plan_bytes=plan))
                packet = producer.call("PUBLISH_READY", dict(job=job))
                note = dict(job=job, identity_bytes=prepared["identity_bytes"],
                            semantic_sha256=prepared["semantic_sha256"])
                source = producer.call("EXPORT_LIVE_SOURCE", note)
                L.NumericEdgeInbox.publish_target(socket_path, identity_bytes=note["identity_bytes"],
                    packet=packet, semantic_sha256=note["semantic_sha256"])
                if job == 3:
                    # Exact retained processor terminal path: a valid SFD4
                    # reward frame is superseded before inference. No ticket,
                    # score, quality ACK or feedback receipt is invented.
                    action = expected["expected_action_identity"]
                    contract = E.dec.load_dynamic_execution_contract()
                    profile = contract.resolve_q_e4(action["mode_id"], action["q_e4"])
                    inner = b"\x00"
                    C.require(profile.execution_bundle_sha256 == action["execution_bundle_sha256"]
                              and profile.action_id == action["anchor_action_id"],
                              "retained action differs from current dynamic execution binding")
                    envelope = E.X.ExecutionEnvelopeV3(mode_id=action["mode_id"],
                        q_e4=action["q_e4"], keep_count=profile.keep_count,
                        anchor_action_id=profile.action_id, reward_requested=True,
                        session_uuid=action["session_uuid"],
                        controller_lineage_sha256=action["controller_lineage_sha256"],
                        decision_seq=action["decision_seq"], ticket_seq=action["ticket_seq"],
                        frame_id=action["frame_id"], tensor_seq=action["tensor_seq"],
                        capture_timestamp_ns=case.kwargs["identity"]["capture_timestamp_ns"],
                        execution_bundle_sha256=profile.execution_bundle_sha256,
                        inner_payload_sha256=C.sha256(inner), inner_payload=inner)
                    frame = W.pack_sfd4(envelope, expected["frame_context"])
                    processor = E.Run4EdgeProcessorV2(contract=contract,
                        run_id=case.kwargs["identity"]["run_id"],
                        cell_id=case.kwargs["identity"]["cell_id"],
                        service_deadline_s=.2, ack_timeout_s=.4,
                        compute=lambda **_: (_ for _ in ()).throw(
                            AssertionError("superseded frame cannot run edge compute")))
                    edge_terminal = processor.terminal(frame,
                        outcome="SUPERSEDED_PENDING", stage="EDGE_PENDING_REPLACED",
                        superseded_by_frame_id=case.frame_id+1)
                    C.require(edge_terminal["frame_id"] == case.frame_id,
                              "actual processor terminal identity differs")
                    while I.raw() - opened <= 170_000_000:
                        time.sleep(.001)
                    terminal = event("ue", "feedback_timeout", I.raw())
                    events = [event("collector", "owned_tick_snapshot", observed),
                        event("producer", "source_stage_received",
                              staged["source_stage_received_raw_ns"],
                              staged["source_stage_owner_pid"]),
                        event("ue", "action_open", opened),
                        event("ue", "last_datagram_sent", sent), terminal]
                    L.NumericEdgeInbox.publish_terminal(socket_path,
                        identity_bytes=note["identity_bytes"],
                        ue_events_bytes=C.canonical(dict(
                            schema=L.OBSERVED_EVENTS_SCHEMA, events=events)),
                        source_export=source)
                    requested = L.NumericEdgeInbox.await_completion(socket_path,
                        identity_bytes=note["identity_bytes"], timeout_s=10.)
                    C.require(requested["status"] == "UNSCORED_REQUIRED"
                              and requested["reason"] == "FEEDBACK_TIMEOUT"
                              and requested["edge_terminal"] == edge_terminal,
                              "actual superseded frame was not admitted as unscored")
                    prepared_terminal = producer.call("PREPARE_LIVE_UNSCORED_TERMINAL",
                        dict(note, clock_descriptor_bytes=requested["clock_descriptor_bytes"],
                             reason=requested["reason"]))
                    L.NumericEdgeInbox.publish_unscored_files(socket_path,
                        identity_bytes=note["identity_bytes"], files=prepared_terminal["files"])
                    final = L.NumericEdgeInbox.await_completion(socket_path,
                        identity_bytes=note["identity_bytes"], timeout_s=10.)
                    C.require(final["status"] == "UNSCORED_DURABLE"
                              and final["clock_descriptor_sha256"] == C.sha256(
                                  requested["clock_descriptor_bytes"]),
                              "actual unscored ArchiveClient receipt missing")
                    producer.call("COMPLETE_LIVE_UNSCORED", dict(note,
                        terminal_sha256=prepared_terminal["terminal_sha256"],
                        receipt=final["receipt"], relative_prefix=final["relative_prefix"]))
                    ps, ss = producer.call("SNAPSHOT"), scorer.call("SNAPSHOT")
                    C.require(ps["valid"] and ss["valid"]
                              and ps["retained_jobs"] == ss["retained_jobs"] == 0,
                              "unscored producer retirement did not happen during route")
                    rows.append(dict(frame_id=case.frame_id, q_perc=None,
                        quality_exact=False, archive_status=final["status"],
                        clock_terminal="FEEDBACK_TIMEOUT", edge_terminal=edge_terminal,
                        producer=ps, scorer=ss))
                    print(json.dumps(dict(rehearsal_frames=len(rows), frame_id=case.frame_id)), flush=True)
                    continue
                envelope = SimpleNamespace(**expected["expected_action_identity"],
                    capture_timestamp_ns=case.kwargs["identity"]["capture_timestamp_ns"])
                ticket = E.EvaluationTicketV2(envelope=envelope, context=expected["frame_context"],
                    records=tuple(pred["records"]), predicted_mask=pred["labels"],
                    gt_identity=case.kwargs["identity"], enqueued_wall_ns=time.time_ns())
                evaluator.submit(ticket)
                raw = payloads.get(timeout=10.)
                # Wait for the sender's actual post-send stamp; receipt is then
                # observed by this harness on its own clock, never backdated.
                deadline = time.monotonic()+2.
                while evaluator._key(ticket) not in evaluator._emitted:
                    C.require(time.monotonic() < deadline, "actual feedback emission stamp missing")
                    time.sleep(.001)
                received = I.raw()
                feedback, reason = W.decode_feedback(raw)
                C.require(feedback.q_perc == pred["q_perc"], "retained numerical quality differs")
                terminal = (event("ue", "feedback_received", received) if received-opened <= 170_000_000
                            else event("ue", "feedback_timeout", received))
                events = [event("collector", "owned_tick_snapshot", observed),
                    event("producer", "source_stage_received", staged["source_stage_received_raw_ns"],
                          staged["source_stage_owner_pid"]),
                    event("ue", "action_open", opened), event("ue", "last_datagram_sent", sent), terminal]
                values = dict(schema=L.OBSERVED_EVENTS_SCHEMA, events=events)
                if terminal["name"] == "feedback_received":
                    values["feedback"] = dict(packet_sha256=C.sha256(raw), kind=feedback.kind,
                        reason=reason.name, q_perc=feedback.q_perc)
                L.NumericEdgeInbox.publish_terminal(socket_path, identity_bytes=note["identity_bytes"],
                    ue_events_bytes=C.canonical(values), source_export=source)
                final = L.NumericEdgeInbox.await_completion(socket_path,
                    identity_bytes=note["identity_bytes"], timeout_s=20.)
                C.require(final["status"] == "SCORED_DURABLE", "actual frame archive failed: "+str(final))
                producer.call("SCORE_COMPLETED", note)
                producer.call("COMPLETE_LIVE_SCORED", dict(note,
                    scorer_proof=final["scorer_durable_proof"]))
                ps, ss = producer.call("SNAPSHOT"), scorer.call("SNAPSHOT")
                C.require(ps["valid"] and ss["valid"] and ps["retained_jobs"] == ss["retained_jobs"] == 0,
                          "multi-frame producer/scorer ownership leaked")
                rows.append(dict(frame_id=case.frame_id, q_perc=feedback.q_perc,
                    quality_exact=True, archive_status=final["status"],
                    clock_terminal=final["terminal"], producer=ps, scorer=ss))
                print(json.dumps(dict(rehearsal_frames=len(rows), frame_id=case.frame_id)), flush=True)
            edge_report = evaluator.close(); evaluator = None
        closes = dict(producer=producer.close(), scorer=scorer.close(), archive=archive.close())
        C.require(inventory(data) == sources, "source changed during rehearsal")
        result = dict(status="FOUR_FRAME_LIVE_ADAPTER_CPU_REHEARSAL_PASS", frames=rows,
            edge_report=edge_report, edge_scope=edge_state, closes=closes,
            all_children_exit_zero=all(client.process.returncode == 0 for client in (producer, scorer, archive)),
            sources_unchanged=True, live_services_started=False, performance_claim=False)
        C.require(result["all_children_exit_zero"], "owned child exit failure")
        H.write(output / "RESULT.json", result)
        return result
    except BaseException as error:
        H.write(output / "FAILURE.json", dict(error=f"{type(error).__name__}: {error}", completed=rows,
            edge_failure=None if evaluator is None else evaluator._failure,
            live_services_started=False))
        raise
    finally:
        for client in (producer, scorer, archive):
            if client is not None and client.process.poll() is None:
                client.abort()
        H.write(output / "OUTPUT_SHA256.json", {str(path.relative_to(output)): H.sha(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "OUTPUT_SHA256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)["status"]))
