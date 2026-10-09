"""Five-frame CPU completion rehearsal with real producer/scorer/archive children.

Two retained-source frames receive explicit never-scored edge terminals, one
before TARGET and one after TARGET. Their actual observed UE timeouts and
durable archives must release both producer credits for two later scored
frames. A fifth frame with no edge processing or edge terminal retires after
the existing edge admission horizon, with a separate durable no-processing
proof. Publication and archive receipt gates belong only to this CPU harness;
the retained arrays, action/send notifications and edge terminals are fixtures,
not a CARLA capture, radio transmission or live-performance measurement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace as NS

from rl_agent.splitfusion_numeric_gt_integration_v1 import contract as C
from rl_agent.splitfusion_numeric_gt_integration_v2 import replay as H
from rl_agent.splitfusion_numeric_gt_producer_v1 import cases as K
from rl_agent.splitfusion_numeric_gt_producer_v1.replay_v11 import quality_bridge
from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP

from . import edge_live as L, ue_live as U
from .async_live_rehearsal import _owned_collector, _planned, _pose, _verify_archive
from .live_clock import ASYNC_HANDOFF_ID, ASYNC_SCHEMA, DEADLINE_NS
from .process import Client, LIVE_PROCESS_CONTRACT
from .replay import inventory, OUTPUT_ROOT
from .worker_owner import OwnedClient


FRAMES = 5
OVERLAP_FRAMES = 4
STATUS = "FIVE_FRAME_COMPLETION_REAL_CHILD_CPU_REHEARSAL_PASS"
SCOPE = dict(cpu_only=True, fixture_numerics=True, actual_ue_handoff=True,
    feedback_scope="HARNESS_CALLBACK_NOT_RADIO",
    edge_terminal_scope="HARNESS_CONTROL_TERMINAL_NOT_LIVE_TRANSPORT",
    live_services_started=False, live_capture_claim=False, live_qualified=False,
    deadline_claim=False, performance_claim=False)


def _await(predicate, message, *, handoff, evaluator, timeout_s=30.):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        C.require(not handoff.errors, "UE completion failed: " + repr(handoff.errors))
        C.require(evaluator._failure is None,
                  "numeric evaluator failed: " + str(evaluator._failure))
        C.require(time.monotonic() < deadline, message)
        time.sleep(.002)


def _observe_timeout(handoff, row):
    """Observe a real elapsed RAW deadline, then exercise the UE poll callback."""
    while True:
        observed = U.raw()
        remaining = DEADLINE_NS - (observed - row["action_open"])
        if remaining < 0:
            break
        time.sleep(min(.01, max(.0001, (remaining + 1) / 1e9)))
    handoff.on_controller_poll(NS(resolution=None), None, observed,
        NS(resolution=object(), resolved_by="TIMEOUT",
           reward_frame=NS(frame_id=row["case"].frame_id)))
    terminal = row["ticket"].terminal
    C.require(terminal is not None and terminal["name"] == "feedback_timeout"
              and terminal["raw_ns"] == observed
              and observed - row["action_open"] > DEADLINE_NS,
              "actual strictly-expired UE timeout observation required")


def _edge_terminal(row, superseded_by_frame_id):
    capture, action = row["capture"], row["action"]
    observed = time.time()
    return MP.build_run4_edge_terminal(run_id=capture["run_id"],
        cell_id=capture["cell_id"], stream_id=capture["stream_id"], identity=action,
        capture_timestamp_ns=capture["capture_timestamp_ns"],
        service_deadline_at=observed, ack_timeout_at=observed,
        outcome="SUPERSEDED_PENDING", stage="EDGE_PENDING_REPLACED",
        age_ms=max(0., (time.time_ns() - capture["capture_timestamp_ns"]) / 1e6),
        emit_at=observed, superseded_by_frame_id=superseded_by_frame_id)


def _verify_unscored(archive_root, row, receipt):
    frame_id = row["case"].frame_id
    directory = archive_root / "numeric_gt" / f"frame_{frame_id:010d}"
    C.require({path.name for path in directory.iterdir()} ==
              {"live_clock.json", "unscored_terminal.json"},
              "unscored frame must contain exactly clock and no-score terminal")
    clock_bytes = (directory / "live_clock.json").read_bytes()
    terminal_bytes = (directory / "unscored_terminal.json").read_bytes()
    clock, terminal = C._parse(clock_bytes), C._parse(terminal_bytes)
    key = row["ticket"].prepared["identity_bytes"]
    C.require(clock["schema"] == ASYNC_SCHEMA and clock["handoff_id"] == ASYNC_HANDOFF_ID
              and clock["ready_identity_sha256"] == C.sha256(key)
              and clock["scoring_state"] == "NOT_SCORED_AT_TIMEOUT"
              and terminal["ready_identity"] == C._parse(key)
              and terminal["clock_descriptor_sha256"] == C.sha256(clock_bytes)
              and terminal["score_exists"] is False
              and terminal["feedback_receipt_claim"] is False
              and terminal["quality_feedback_claim"] is False,
              "unscored archive must bind exact identity and never invent quality")
    deadline = clock["deadline"]
    C.require(deadline["terminal"] == "FEEDBACK_TIMEOUT"
              and deadline["feedback_received_raw_ns"] is None
              and deadline["timeout_observed_raw_ns"] - deadline["action_open_raw_ns"] > DEADLINE_NS,
              "unscored clock lacks actual observed UE timeout")
    events = {(event["role"], event["name"]): event["raw_ns"] for event in clock["events"]}
    C.require(not any(role in ("scorer", "coordinator") for role, _ in events)
              and events["collector", "owned_tick_snapshot"] <= events["ue", "source_owned"]
              <= events["ue", "action_open"] <= events["ue", "last_datagram_sent"]
              <= events["producer", "source_stage_received"],
              "unscored owner clocks invented scoring or violate after-send ownership")
    admission_expired = row["terminal_order"] == "NO_EDGE_PROCESSING_OR_TERMINAL"
    C.require(receipt["status"] == "COMPLETE" and receipt["durable"] is True
              and receipt["error"] is None and receipt["identity"] == row["capture"]
              and len(receipt["written"]) == (3 if admission_expired else 2),
              "actual durable unscored archive receipt required")
    for item in receipt["written"]:
        raw = (archive_root / item["relative"]).read_bytes()
        C.require(item["root"] == 0 and item["ready"] is False
                  and len(raw) == item["bytes"] and C.sha256(raw) == item["sha256"],
                  "unscored archive readback differs from child fsync receipt")
    admission = None
    if admission_expired:
        relative = f"numeric_gt_admission/frame_{frame_id:010d}.json"
        raw = (archive_root / relative).read_bytes()
        proof = C._parse(raw)
        C.require(relative in {item["relative"] for item in receipt["written"]}
                  and proof["schema"] == "splitfusion.numeric_gt.no_processing_after_horizon.v1"
                  and proof["ready_identity_sha256"] == C.sha256(key)
                  and proof["ready_identity"] == C._parse(key)
                  and proof["semantic_sha256"] == row["ticket"].prepared["semantic_sha256"]
                  and proof["verified_processing_admitted"] is False
                  and proof["evaluation_submitted"] is False
                  and proof["score_exists"] is False and proof["future_processing_refused"] is True
                  and proof["observed_wall_s"] > proof["capture_deadline_wall_s"]
                  and proof["processing_horizon"] == dict(run_id=row["capture"]["run_id"],
                      cell_id=row["capture"]["cell_id"], ack_timeout_s=.5),
                  "lost-frame retirement lacks durable bound no-processing admission proof")
        admission = dict(relative=relative, sha256=C.sha256(raw), document=proof)
    return dict(frame_id=frame_id, completion="UNSCORED_DURABLE", score_exists=False,
        numerical_q_perc=None, terminal=deadline["terminal"], files=2,
        clock_schema=clock["schema"], terminal_order=row["terminal_order"],
        durable_files=len(receipt["written"]), admission_expiry=admission,
        timeout_elapsed_ns=deadline["timeout_observed_raw_ns"] - deadline["action_open_raw_ns"],
        clock_descriptor_sha256=C.sha256(clock_bytes),
        unscored_terminal_sha256=C.sha256(terminal_bytes))


def run(output):
    output = Path(output).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(OUTPUT_ROOT)
              and not any(path.is_symlink() for path in output.parents),
              "fresh owned completion rehearsal output required")
    C.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "",
              "CPU rehearsal requires CUDA_VISIBLE_DEVICES empty")
    output.mkdir(parents=True, exist_ok=False)
    producer = scorer = archive = handoff = evaluator = None
    archive_gate = threading.Event()
    target_gate = threading.Event()
    completed, orders = [], []
    try:
        data = H.load_inputs(FRAMES)
        sources = inventory(data)
        cases = data["cases"][:FRAMES]
        binding = C.sha256(C.canonical(dict(scope="CPU_COMPLETION_REHEARSAL_NOT_RADIO",
                                           sources=sources)))
        H.write(output / "PLAN.json", dict(scope="CPU_COMPLETION_REAL_CHILD_REHEARSAL",
            frames=FRAMES, unscored_frames=3, scored_frames=2,
            edge_terminal_frames=2, no_processing_or_terminal_frames=1,
            source_sha256=sources, input_sha256=data["inputs"],
            run_binding=binding, handoff_id=ASYNC_HANDOFF_ID,
            gate_scope="CPU_REHEARSAL_ONLY", **SCOPE))
        ue_dir, archive_root, sidecars = output / "ue", output / "archive", output / "sidecars"
        for directory in (ue_dir, archive_root, sidecars):
            directory.mkdir()
        cpu = sorted(os.sched_getaffinity(0))
        C.require(len(cpu) >= 3, "three permitted CPU cores required")
        target_gate_entered = threading.Event()

        class Publisher(L.NumericEdgeInbox):
            @staticmethod
            def publish_target(socket_path, **kwargs):
                frame_id = C._parse(kwargs["identity_bytes"])["capture_identity"]["frame_id"]
                if frame_id == cases[0].frame_id:
                    target_gate_entered.set()
                    C.require(target_gate.wait(timeout=15.),
                              "bounded CPU TARGET gate was not released")
                reply = L.NumericEdgeInbox.publish_target(socket_path, **kwargs)
                orders.append(dict(frame_id=frame_id, event="TARGET_ADMITTED", raw_ns=U.raw()))
                return reply

        handoff = U.NumericUeHandoff(attempt_dir=ue_dir,
            registration=dict(numeric_limits=dict(max_frames=FRAMES, retained_jobs=2,
                pending_reward_jobs=2, retained_bytes=128 * 1024 * 1024,
                process_timeout_s=30.), numeric_worker_sources=sources), publisher=Publisher)
        handoff.collector = _owned_collector(cases)
        handoff.socket_path = output / "numeric_gt.sock"
        producer_args = {key: str(value) if isinstance(value, Path) else value
                         for key, value in K.producer_arguments(cases[0]).items()}
        producer = OwnedClient("producer", dict(run_binding=binding,
            producer_binding=data["startup"]["binding_sha256"],
            prediction_contract=data["prediction_contract"], max_frames=FRAMES,
            producer_args=producer_args, offline_replay=False,
            max_retained_jobs=2, max_retained_bytes=128 * 1024 * 1024,
            live_process_contract=LIVE_PROCESS_CONTRACT),
            directory=handoff.directory / "producer_process", source_sha256=sources,
            cpu=cpu[0], timeout_s=30., writable_attempt_root=ue_dir)
        handoff.client = producer
        handoff._start_client = lambda _source: producer
        handoff._write_created("PROCESS_READY.json", C.canonical(dict(producer=producer.ready)))
        C.require(producer.owner_thread_alive, "producer owner exited before first frame")
        scorer = Client("scorer", dict(run_binding=binding,
            producer_binding=data["startup"]["binding_sha256"], max_jobs=2,
            max_bytes=128 * 1024 * 1024), directory=output / "scorer",
            source_sha256=sources, cpu=cpu[1])
        archive = ArchiveClient(roots=[archive_root], max_jobs=2,
            max_bytes=128 * 1024 * 1024, max_frames=FRAMES,
            binding=dict(numeric_run_binding=binding,
                         numeric_worker_source_digest=C.sha256(C.canonical(sources))))
        H.write(output / "PROCESS_READY.json", dict(producer=producer.ready,
            scorer=scorer.ready, archive=archive.ready))
        archive_gate_entered, archive_gate_armed = threading.Event(), threading.Event()
        original_poll, original_submit = archive.poll, archive.submit

        def gated_poll():
            if archive_gate_armed.is_set() and not archive_gate.is_set():
                archive_gate_entered.set()
                C.require(archive_gate.wait(timeout=15.),
                          "bounded CPU archive receipt gate was not released")
            return original_poll()

        def gated_submit(*args, **kwargs):
            job = original_submit(*args, **kwargs)
            archive_gate_armed.set()
            return job

        archive.poll, archive.submit = gated_poll, gated_submit
        payloads, admitted = queue.Queue(), []
        with L.install_numeric_edge(socket_path=handoff.socket_path,
                scorer_client=scorer, archive_client=archive, archive_root_index=0,
                sidecar_directory=sidecars, max_frames=FRAMES,
                terminal_timeout_s=10.) as edge_state:
            spec, _ = quality_bridge()
            evaluator = E.Run4EvaluatorV2(spec=spec, read_ground_truth=None,
                send=payloads.put, match_distance_m=2., gt_timeout_s=2.)
            evaluator.start()

            def wait_for(predicate, message):
                _await(predicate, message, handoff=handoff, evaluator=evaluator)

            def admit(case):
                expected = H.expectation(data, case, binding)
                action, capture, context = (expected["expected_action_identity"],
                    case.kwargs["identity"], expected["frame_context"])
                runtime = NS(pipeline=NS(run_id=capture["run_id"], cell_id=capture["cell_id"],
                    stream_id=capture["stream_id"], decisions=[]),
                    engine=NS(lineage=action["controller_lineage_sha256"]),
                    _numeric_current_ego_pose=_pose(context), last_datagram={})
                ticket = handoff.stage_before_action(runtime, frame_id=case.frame_id,
                    carla_timestamp=case.kwargs["policy_timestamp_s"],
                    capture_timestamp_ns=capture["capture_timestamp_ns"], ego_pose=_pose(context))
                C.require(ticket.stage_event is None, "producer stage began before send")
                opened = U.raw()
                runtime.pipeline.decisions.append(dict(frame_id=case.frame_id,
                    action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=opened)))
                handoff.on_plan(runtime, _planned(action, capture, context))
                runtime.last_datagram[case.frame_id] = U.raw()
                begin = time.monotonic()
                handoff.after_submit(runtime, case.frame_id, {"sent": True})
                C.require(time.monotonic() - begin < .5,
                          "pending admission blocked UE submit")
                row = dict(case=case, ticket=ticket, action=action, capture=capture,
                    context=context, action_open=opened,
                    prediction=data["predictions"][case.frame_id])
                admitted.append(row)
                return row

            def observe_terminal(row, order, replacement):
                document = _edge_terminal(row, replacement)
                C.require(evaluator.inbox.observe_edge_terminal(document),
                          "reward-bearing edge terminal was not retained")
                row["terminal_order"] = order
                orders.append(dict(frame_id=row["case"].frame_id,
                    event="EDGE_TERMINAL_OBSERVED", order=order,
                    raw_ns=U.raw(), document=document))
                _observe_timeout(handoff, row)

            first = admit(cases[0])
            wait_for(target_gate_entered.is_set, "first TARGET did not reach CPU gate")
            C.require(not evaluator.inbox.rows_snapshot() and not first["ticket"].published,
                      "early terminal rehearsal accidentally admitted TARGET first")
            observe_terminal(first, "BEFORE_TARGET", cases[1].frame_id)
            target_gate.set()
            wait_for(archive_gate_entered.is_set, "unscored archive did not reach receipt gate")
            second = admit(cases[1])
            wait_for(lambda: second["ticket"].published, "second TARGET did not publish")
            observe_terminal(second, "AFTER_TARGET", cases[2].frame_id)
            third, fourth = admit(cases[2]), admit(cases[3])
            before = producer.call("SNAPSHOT")
            active = [row for row in admitted if row["ticket"].producer_admitted
                      and not row["ticket"].retired]
            pending = [row for row in (third, fourth) if not row["ticket"].producer_admitted
                       and row["ticket"].thread.is_alive()]
            C.require(len(active) == len(pending) == before["retained_jobs"] == 2
                      and before["retired"] == 0 and not handoff.errors,
                      "four-frame admission did not retain two active and two pending credits")
            overlap = dict(admitted=OVERLAP_FRAMES, active_producer_jobs=2, pending_reward_jobs=2,
                producer_retained_jobs=2, archive_receipt_gate_entered=True,
                gate_scope="CPU_REHEARSAL_ONLY")
            H.write(output / "OVERLAP.json", dict(overlap=overlap, producer_snapshot=before))
            archive_gate.set()
            # Neither later frame is submitted for scoring until both targets
            # exist. With two credits this requires both unscored retirements.
            wait_for(lambda: third["ticket"].published and fourth["ticket"].published,
                     "durable unscored completions did not release both producer credits")
            C.require(first["ticket"].retired and second["ticket"].retired
                      and not third["ticket"].retired and not fourth["ticket"].retired,
                      "producer credits were reused before unscored durable retirement")
            after = producer.call("SNAPSHOT")
            C.require(after["retained_jobs"] == after["retired"] == 2
                      and after["issued"] == OVERLAP_FRAMES,
                      "producer retained/retired accounting differs after credit reuse")
            credit_reuse = dict(unscored_retired_before_scoring=2,
                later_targets_published_before_scoring=2,
                retained_jobs_after_reuse=2, producer_snapshot=after)
            H.write(output / "CREDIT_REUSE.json", credit_reuse)
            for row in (third, fourth):
                frame_id, capture = row["case"].frame_id, row["capture"]
                prediction = row["prediction"]
                evaluation = E.EvaluationTicketV2(
                    envelope=NS(**row["action"], capture_timestamp_ns=capture["capture_timestamp_ns"]),
                    context=row["context"], records=tuple(prediction["records"]),
                    predicted_mask=prediction["labels"], gt_identity=capture,
                    enqueued_wall_ns=time.time_ns())
                evaluator.submit(evaluation)
                packet = payloads.get(timeout=30.)
                wait_for(lambda: evaluator._key(evaluation) in evaluator._emitted,
                         "actual scorer feedback emission missing")
                received = U.raw()
                feedback, reason = W.decode_feedback(packet)
                C.require(feedback.frame_id == frame_id
                          and feedback.q_perc == prediction["q_perc"],
                          "actual numerical scorer differs from retained expected quality")
                if received - row["action_open"] <= DEADLINE_NS:
                    handoff.collector.live._feedback_rows.append(dict(frame_id=frame_id,
                        receipt=dict(domain="CLOCK_MONOTONIC_RAW", ns=received),
                        sha256=C.sha256(packet), kind=feedback.kind, reason=reason.name,
                        q_perc=feedback.q_perc, **{"class": "ACCEPTED"}))
                    handoff.on_feedback(frame_id, received)
                else:
                    _observe_timeout(handoff, row)
            for row in admitted:
                wait_for(lambda: row["ticket"].retired and not row["ticket"].thread.is_alive(),
                         "actual UE durable retirement failed")
                if "terminal_order" in row:
                    receipts = [item for item in archive.completions
                                if item["identity"]["frame_id"] == row["case"].frame_id]
                    C.require(len(receipts) == 1, "one durable archive per unscored frame required")
                    result = _verify_unscored(archive_root, row, receipts[0])
                    key = row["ticket"].prepared["identity_bytes"]
                    stored = next(item for item in evaluator.inbox.rows_snapshot() if item[0] == key)
                    C.require(stored[1]["edge_terminal"] == next(item["document"] for item in orders
                              if item["frame_id"] == row["case"].frame_id
                              and item["event"] == "EDGE_TERMINAL_OBSERVED"),
                              "edge terminal changed between observation and target binding")
                else:
                    result = dict(_verify_archive(archive_root / "numeric_gt", row["case"].frame_id,
                        row["prediction"]["q_perc"]), completion="SCORED_DURABLE", score_exists=True)
                completed.append(result)
                print(json.dumps(dict(completion_rehearsal_frames=len(completed),
                                      frame_id=row["case"].frame_id)), flush=True)
            # Model an incomplete/lost uplink: this exact reward frame never
            # enters edge processing and emits no edge terminal. The binding
            # uses the retained processor's 500-ms capture horizon. Retained
            # capture wall times are old; the fresh UE RAW timeout is still
            # actually observed. No fixture timestamp is rewritten.
            capture = cases[4].kwargs["identity"]
            evaluator.inbox.bind_processing_horizon(run_id=capture["run_id"],
                cell_id=capture["cell_id"], ack_timeout_s=.5)
            lost = admit(cases[4])
            lost["terminal_order"] = "NO_EDGE_PROCESSING_OR_TERMINAL"
            wait_for(lambda: lost["ticket"].published, "lost-frame TARGET did not publish")
            _observe_timeout(handoff, lost)
            wait_for(lambda: lost["ticket"].retired and not lost["ticket"].thread.is_alive(),
                     "lost-frame admission expiry did not durably retire")
            key = lost["ticket"].prepared["identity_bytes"]
            stored = next(item for item in evaluator.inbox.rows_snapshot() if item[0] == key)
            C.require(stored[1]["edge_terminal"] is None
                      and not stored[1]["target_claimed"]
                      and stored[2]["status"] == "UNSCORED_DURABLE",
                      "lost-frame rehearsal accidentally used processing or edge terminal")
            receipts = [item for item in archive.completions
                        if item["identity"]["frame_id"] == lost["case"].frame_id]
            C.require(len(receipts) == 1, "one durable archive for lost frame required")
            completed.append(_verify_unscored(archive_root, lost, receipts[0]))
            drained = producer.call("SNAPSHOT")
            C.require(drained["retained_jobs"] == 0 and drained["issued"] == drained["retired"] == FRAMES,
                      "lost-frame durable retirement did not release producer credit")
            H.write(output / "LOST_FRAME_DRAIN.json", dict(frame_id=lost["case"].frame_id,
                producer_snapshot=drained, admission_expiry=completed[-1]["admission_expiry"],
                capture_horizon_scope="RETAINED_FIXTURE_WALL_TIMESTAMP_ALREADY_EXPIRED", **SCOPE))
            print(json.dumps(dict(completion_rehearsal_frames=len(completed),
                                  frame_id=lost["case"].frame_id)), flush=True)
            H.write(output / "EVENT_ORDER.json", orders)
            edge_report = evaluator.close()
            evaluator = None
        closes = dict(producer_handoff=handoff.close())
        closes["producer"] = C._parse((handoff.directory / "PROCESS_CLOSE.json").read_bytes())["producer"]
        closes["scorer"] = scorer.close()
        closes["archive"] = archive.close()
        C.require(not producer.owner_thread_alive and all(client.process.returncode == 0
                  for client in (producer, scorer, archive)),
                  "all real children and the producer owner must exit cleanly")
        C.require(inventory(data) == sources, "retained fixture or runtime sources changed")
        outcomes = [event.get("outcome") for event in closes["producer"]["events"].values()]
        C.require(outcomes.count("LIVE_UNSCORED_TIMEOUT_DURABLE") == 3
                  and outcomes.count("LIVE_SCORED_DURABLE") == 2
                  and all(closes[role]["closed"] is True and closes[role]["valid"] is True
                          and closes[role]["failure"] is None for role in ("producer", "scorer"))
                  and closes["producer"]["retained_jobs"] == closes["scorer"]["retained_jobs"] == 0
                  and closes["scorer"]["scored"] == closes["scorer"]["released"] == 2
                  and len(closes["archive"]["completions"]) == FRAMES
                  and closes["archive"]["submitted"] == closes["archive"]["completed"] == FRAMES
                  and closes["archive"]["outstanding_jobs"] == 0
                  and closes["archive"]["failed"] is False and closes["archive"]["fault"] is None,
                  "actual child CLOSE reports did not reconcile both completion paths")
        result = dict(status=STATUS, frames=completed, overlap=overlap,
            credit_reuse=credit_reuse, closes=closes, edge_report=edge_report,
            lost_frame_retained_jobs_after_completion=drained["retained_jobs"],
            edge_scope=edge_state, observed_event_order=orders,
            handoff_id=ASYNC_HANDOFF_ID, all_children_exit_zero=True,
            child_exitcodes={name: client.process.returncode for name, client in
                             (("producer", producer), ("scorer", scorer), ("archive", archive))},
            producer_owner_thread_alive_after_close=False,
            retained_sources_unchanged=True, sources_unchanged=True, **SCOPE)
        H.write(output / "RESULT.json", result)
        return result
    except BaseException as exc:
        H.write(output / "FAILURE.json", dict(error=f"{type(exc).__name__}: {exc}",
            completed=completed, observed_event_order=orders,
            edge_failure=None if evaluator is None else evaluator._failure, **SCOPE))
        raise
    finally:
        target_gate.set()
        archive_gate.set()
        if handoff is not None and producer is not None and not producer.closed:
            handoff.abort(RuntimeError("completion rehearsal cleanup"))
        for client in (producer, scorer, archive):
            if client is not None and client.process is not None and client.process.poll() is None:
                client.abort()
        H.write(output / "OUTPUT_SHA256.json", {str(path.relative_to(output)): H.sha(path)
            for path in sorted(output.rglob("*")) if path.is_file()
            and path.name != "OUTPUT_SHA256.json"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)["status"]))
