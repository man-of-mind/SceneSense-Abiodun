"""Four-frame real-child CPU rehearsal of the asynchronous numeric UE handoff.

The native arrays and expected scores are retained v11 fixtures. A fresh RAW
stamp records when this harness owns each fixture snapshot; there is no CARLA,
radio, route, actor, GPU, or live-performance claim. The producer child is
created by a persistent OwnedClient thread before the first frame, exactly to
exercise the revised after-send ownership and shutdown boundary.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

from . import edge_live as L, ue_live as U
from .live_clock import ASYNC_HANDOFF_ID, ASYNC_SCHEMA
from .process import Client, LIVE_PROCESS_CONTRACT
from .replay import inventory, OUTPUT_ROOT
from .worker_owner import OwnedClient


FRAMES = 4


def _owned_collector(cases):
    """Fixture sensor owner; numerical bytes stay retained, RAW ownership is new."""
    worlds, rgbs, semantics, depths, radars, completed, points = {}, {}, {}, {}, {}, {}, {}
    for case in cases:
        kwargs = case.kwargs
        frame = case.frame_id
        world = copy.copy(kwargs["frozen_world"])
        world.frozen_observed_raw_ns = U.raw()
        worlds[frame] = world
        rgb = dict(kwargs["rgb"])
        rgbs[frame] = rgb
        semantics[frame] = (kwargs["semantic_image"], rgb["semantic_callback"])
        depths[frame] = (kwargs["depth_image"], rgb["depth_callback"])
        radars[frame] = rgb["radar_callback"]
        completed[frame] = {"rgb", "semantic", "depth", "radar"}
        points[float(kwargs["policy_timestamp_s"])] = {"world_xyz": kwargs["radar_world_xyz"]}
    return NS(_quality_scene_lock=threading.Lock(), _quality_scenes=worlds,
        _avo_condition=threading.Condition(), _avo_rgb=rgbs,
        _avo_semantic=semantics, _avo_depth=depths, _avo_radar=radars,
        _avo_callbacks_complete=completed, _numeric_radar_lock=threading.Lock(),
        _numeric_radar=points, live=NS(_feedback_rows=[]))


def _planned(action, capture, context):
    frame = NS(session_uuid=action["session_uuid"], decision_seq=action["decision_seq"],
        ticket_seq=action["ticket_seq"], reward_requested=True)
    profile = NS(action_id=action["anchor_action_id"],
        profile_id=capture["profile_id"], mode_id=action["mode_id"],
        q_e4=action["q_e4"], execution_bundle_sha256=action["execution_bundle_sha256"])
    plan = NS(frame_identity=frame, frame_id=action["frame_id"],
        tensor_seq=action["tensor_seq"], capture_wall_ns=capture["capture_timestamp_ns"],
        profile=profile, kind=NS(value="POLICY_DECISION"))
    return NS(plan=plan, frame_id=action["frame_id"], tensor_seq=action["tensor_seq"],
              capture_wall_ns=capture["capture_timestamp_ns"])


def _pose(context):
    value = context.ego_world
    return (value.x, value.y, value.z, value.pitch, value.yaw, value.roll)


def _verify_archive(directory, frame_id, expected_quality):
    frame_dir = directory / f"frame_{frame_id:010d}"
    manifest_path = frame_dir / "live_archive.json"
    manifest = C._parse(manifest_path.read_bytes())
    clock = C._parse((frame_dir / "live_clock.json").read_bytes())
    score = C._parse((frame_dir / "numerical_score.json").read_bytes())
    C.require(clock["schema"] == ASYNC_SCHEMA and clock["handoff_id"] == ASYNC_HANDOFF_ID,
              "rehearsal did not exercise actual async V2 clock")
    C.require(score["q_perc"] == expected_quality,
              "rehearsal numeric score differs from retained expectation")
    C.require(manifest["q_perc"] == expected_quality,
              "rehearsal manifest quality differs")
    for filename, evidence in manifest["files"].items():
        data = (frame_dir / filename).read_bytes()
        C.require(len(data) == evidence["bytes"] and C.sha256(data) == evidence["sha256"],
                  "rehearsal archive bytes differ: " + filename)
    events = {(event["role"], event["name"]): event["raw_ns"] for event in clock["events"]}
    C.require(events["collector", "owned_tick_snapshot"] <= events["ue", "source_owned"]
              <= events["ue", "action_open"] <= events["ue", "last_datagram_sent"]
              <= events["producer", "source_stage_received"],
              "async after-send owner clocks not observed in order")
    C.require(manifest["deadline"]["terminal"] in ("FEEDBACK_ON_TIME", "FEEDBACK_TIMEOUT"),
              "actual rehearsal terminal missing")
    return dict(frame_id=frame_id, numerical_q_perc=score["q_perc"],
                terminal=manifest["deadline"]["terminal"],
                files=len(manifest["files"]), clock_schema=clock["schema"],
                post_send_stage_ms=(events["producer", "source_stage_received"] -
                                    events["ue", "last_datagram_sent"]) / 1e6)


def run(output):
    output = Path(output).absolute()
    C.require(not output.exists() and output.resolve().is_relative_to(OUTPUT_ROOT),
              "fresh owned async rehearsal output required")
    output.mkdir(parents=True, exist_ok=False)
    data = H.load_inputs(FRAMES)
    sources = inventory(data)
    binding = C.sha256(C.canonical(dict(scope="CPU_ASYNC_UE_REHEARSAL_NOT_RADIO", sources=sources)))
    cases = data["cases"][:FRAMES]
    H.write(output / "PLAN.json", dict(scope="CPU_ASYNC_UE_REAL_CHILD_REHEARSAL",
        frames=FRAMES, retained_sources=sources, source_sha256=sources,
        handoff_id=ASYNC_HANDOFF_ID,
        actual_ue_handoff=True, feedback_scope="HARNESS_CALLBACK_NOT_RADIO",
        live_services_started=False, fixture_numerics=True,
        deadline_claim=False, performance_claim=False))
    producer = scorer = archive = evaluator = handoff = None
    close_results = {}
    completed = []
    try:
        ue_dir = output / "ue"
        ue_dir.mkdir()
        archive_root, sidecars = output / "archive", output / "sidecars"
        archive_root.mkdir(); sidecars.mkdir()
        cpu = sorted(os.sched_getaffinity(0))
        C.require(len(cpu) >= 3, "three permitted CPU cores required")
        args = {key: str(value) if isinstance(value, Path) else value
                for key, value in K.producer_arguments(cases[0]).items()}
        producer_config = dict(run_binding=binding,
            producer_binding=data["startup"]["binding_sha256"],
            prediction_contract=data["prediction_contract"], max_frames=FRAMES,
            producer_args=args, offline_replay=False,
            max_retained_jobs=2, max_retained_bytes=128 * 1024 * 1024,
            live_process_contract=LIVE_PROCESS_CONTRACT)
        handoff = U.NumericUeHandoff(attempt_dir=ue_dir,
            registration=dict(numeric_limits=dict(max_frames=FRAMES, retained_jobs=2,
                pending_reward_jobs=2,
                retained_bytes=128 * 1024 * 1024, process_timeout_s=30.),
                numeric_worker_sources=sources), publisher=L.NumericEdgeInbox)
        handoff.collector = _owned_collector(cases)
        handoff.socket_path = output / "numeric_gt.sock"
        # The fixture has no live CARLA catalogue to register. Bind the exact
        # retained producer arguments and start its real persistent owner here.
        producer = OwnedClient("producer", producer_config,
            directory=handoff.directory / "producer_process", source_sha256=sources,
            cpu=cpu[0], timeout_s=30., writable_attempt_root=ue_dir)
        handoff.client = producer
        handoff._write_created("PROCESS_READY.json", C.canonical(dict(producer=producer.ready)))
        handoff._start_client = lambda _source: producer
        C.require(producer.owner_thread_alive, "producer child creator thread exited early")
        scorer = Client("scorer", dict(run_binding=binding,
            producer_binding=data["startup"]["binding_sha256"], max_jobs=2,
            max_bytes=128 * 1024 * 1024), directory=output / "scorer",
            source_sha256=sources, cpu=cpu[1])
        archive = ArchiveClient(roots=[archive_root], max_jobs=2,
            max_bytes=128 * 1024 * 1024, max_frames=FRAMES,
            binding=dict(numeric_run_binding=binding,
                         numeric_worker_source_digest=C.sha256(C.canonical(sources))))
        archive_gate = threading.Event()
        archive_gate_entered = threading.Event()
        archive_gate_armed = threading.Event()
        original_submit, original_poll = archive.submit, archive.poll
        def gated_poll():
            if archive_gate_armed.is_set() and not archive_gate.is_set():
                archive_gate_entered.set()
                C.require(archive_gate.wait(timeout=15.),
                          "bounded rehearsal archive gate was not released")
            return original_poll()
        def arm_after_first_submit(*args, **kwargs):
            job = original_submit(*args, **kwargs)
            archive_gate_armed.set()
            return job
        archive.poll = gated_poll
        archive.submit = arm_after_first_submit
        payloads = queue.Queue()
        admitted = []
        overlap = None
        with L.install_numeric_edge(socket_path=handoff.socket_path,
                scorer_client=scorer, archive_client=archive,
                archive_root_index=0, sidecar_directory=sidecars,
                max_frames=FRAMES, terminal_timeout_s=10.) as edge_state:
            spec, _ = quality_bridge()
            evaluator = E.Run4EvaluatorV2(spec=spec, read_ground_truth=None,
                send=payloads.put, match_distance_m=2., gt_timeout_s=2.)
            evaluator.start()
            def admit(case):
                frame_id = case.frame_id
                prediction = data["predictions"][frame_id]
                expected = H.expectation(data, case, binding)
                action = expected["expected_action_identity"]
                capture = case.kwargs["identity"]
                context = expected["frame_context"]
                runtime = NS(pipeline=NS(run_id=capture["run_id"], cell_id=capture["cell_id"],
                    stream_id=capture["stream_id"], decisions=[]),
                    engine=NS(lineage=action["controller_lineage_sha256"]),
                    _numeric_current_ego_pose=_pose(context), last_datagram={})
                ticket = handoff.stage_before_action(runtime, frame_id=frame_id,
                    carla_timestamp=case.kwargs["policy_timestamp_s"],
                    capture_timestamp_ns=capture["capture_timestamp_ns"], ego_pose=_pose(context))
                C.require(ticket.stage_event is None and handoff.client is producer,
                          "producer stage began before actual send")
                action_open = U.raw()
                runtime.pipeline.decisions.append(dict(frame_id=frame_id,
                    action_open=dict(domain="CLOCK_MONOTONIC_RAW", ns=action_open)))
                handoff.on_plan(runtime, _planned(action, capture, context))
                runtime.last_datagram[frame_id] = U.raw()
                sent_begin = time.monotonic()
                handoff.after_submit(runtime, frame_id, {"sent": True})
                C.require(time.monotonic() - sent_begin < .5,
                          "pending reward admission blocked retained UE submit")
                row = dict(case=case, ticket=ticket, action=action,
                           context=context, capture=capture, prediction=prediction,
                           action_open=action_open)
                admitted.append(row)
                return row

            def score_and_terminal(row):
                case, ticket = row["case"], row["ticket"]
                frame_id = case.frame_id
                action, context, capture = row["action"], row["context"], row["capture"]
                prediction, action_open = row["prediction"], row["action_open"]
                deadline = time.monotonic() + 30.
                while not ticket.published and ticket.fault is None:
                    C.require(time.monotonic() < deadline, "async target publication timed out")
                    time.sleep(.001)
                C.require(ticket.fault is None, ticket.fault or "async worker failed")
                envelope = NS(**action, capture_timestamp_ns=capture["capture_timestamp_ns"])
                eval_ticket = E.EvaluationTicketV2(envelope=envelope, context=context,
                    records=tuple(prediction["records"]), predicted_mask=prediction["labels"],
                    gt_identity=capture, enqueued_wall_ns=time.time_ns())
                evaluator.submit(eval_ticket)
                raw = payloads.get(timeout=30.)
                until = time.monotonic() + 5.
                while evaluator._key(eval_ticket) not in evaluator._emitted:
                    C.require(time.monotonic() < until, "actual feedback emission stamp missing")
                    time.sleep(.001)
                received = U.raw()
                feedback, reason = W.decode_feedback(raw)
                C.require(feedback.q_perc == prediction["q_perc"],
                          "actual numerical scorer differs from retained quality")
                if received - action_open <= 170_000_000:
                    handoff.collector.live._feedback_rows.append(dict(frame_id=frame_id,
                        receipt=dict(domain="CLOCK_MONOTONIC_RAW", ns=received),
                        sha256=C.sha256(raw), kind=feedback.kind,
                        reason=reason.name, q_perc=feedback.q_perc,
                        **{"class": "ACCEPTED"}))
                    handoff.on_feedback(frame_id, received)
                else:
                    handoff.on_controller_poll(NS(resolution=None), None, received,
                        NS(resolution=object(), resolved_by="TIMEOUT",
                           reward_frame=NS(frame_id=frame_id)))

            # The first archive is durably written by the real child, but its
            # receipt is withheld from the edge until four UE tickets have
            # been admitted. This is a CPU harness gate, not a live pipeline
            # setting or a latency measurement.
            score_and_terminal(admit(cases[0]))
            C.require(archive_gate_entered.wait(timeout=10.),
                      "first archive did not reach bounded receipt gate")
            score_and_terminal(admit(cases[1]))
            third, fourth = admit(cases[2]), admit(cases[3])
            snapshot = producer.call("SNAPSHOT")
            active = [row for row in admitted if row["ticket"].producer_admitted
                      and not row["ticket"].retired]
            pending = [row for row in (third, fourth)
                       if not row["ticket"].producer_admitted and row["ticket"].thread.is_alive()]
            C.require(len(admitted) == FRAMES and len(active) == 2 and len(pending) == 2
                      and snapshot["retained_jobs"] == 2 and not handoff.errors,
                      "four-ticket overlap did not respect two active producer credits")
            overlap = dict(admitted=len(admitted), active_producer_jobs=len(active),
                pending_reward_jobs=len(pending), producer_retained_jobs=snapshot["retained_jobs"],
                archive_receipt_gate_entered=True, gate_scope="CPU_REHEARSAL_ONLY")
            archive_gate.set()
            score_and_terminal(third)
            score_and_terminal(fourth)
            for row in admitted:
                ticket = row["ticket"]
                frame_id = row["case"].frame_id
                until = time.monotonic() + 30.
                while ticket.thread.is_alive() and evaluator._failure is None and time.monotonic() < until:
                    ticket.thread.join(timeout=.05)
                C.require(evaluator._failure is None,
                          "numeric edge archive worker failed: " + str(evaluator._failure))
                C.require(not ticket.thread.is_alive() and ticket.retired and not handoff.errors,
                          "async UE durable retirement failed: " + repr(handoff.errors))
                row = _verify_archive(archive_root / "numeric_gt", frame_id,
                                      row["prediction"]["q_perc"])
                completed.append(row)
                print(json.dumps(dict(async_rehearsal_frames=len(completed), frame_id=frame_id)),
                      flush=True)
            edge_report = evaluator.close(); evaluator = None
        close_results["producer"] = handoff.close()
        C.require(not producer.owner_thread_alive and producer.process.returncode == 0,
                  "producer creator thread or child survived clean CLOSE")
        close_results["scorer"] = scorer.close()
        close_results["archive"] = archive.close()
        C.require(inventory(data) == sources, "retained fixture inputs changed")
        C.require(len(completed) == FRAMES and all(client.process.returncode == 0
                  for client in (producer, scorer, archive)), "real worker close incomplete")
        result = dict(status="FOUR_FRAME_ASYNC_UE_REAL_CHILD_CPU_REHEARSAL_PASS",
            frames=completed, edge_report=edge_report, edge_scope=edge_state,
            overlap=overlap,
            closes=close_results, producer_owner_thread_alive_after_close=False,
            handoff_id=ASYNC_HANDOFF_ID, actual_ue_handoff=True,
            feedback_scope="HARNESS_CALLBACK_NOT_RADIO", deadline_claim=False,
            retained_sources_unchanged=True, sources_unchanged=True,
            all_children_exit_zero=True, live_services_started=False,
            performance_claim=False)
        H.write(output / "RESULT.json", result)
        return result
    except BaseException as exc:
        H.write(output / "FAILURE.json", dict(error=f"{type(exc).__name__}: {exc}",
            completed=completed, edge_failure=None if evaluator is None else evaluator._failure,
            edge_scope_error=None if "edge_state" not in locals() else edge_state.get("error"),
            live_services_started=False, performance_claim=False))
        raise
    finally:
        if "archive_gate" in locals():
            archive_gate.set()
        if handoff is not None and producer is not None and not producer.closed:
            handoff.abort(RuntimeError("rehearsal cleanup"))
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
