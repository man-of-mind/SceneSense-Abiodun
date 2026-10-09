"""CPU-only synthetic decision streams and real-archive-shaped records for the v2 tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS
import uuid

import torch

from rl_agent.splitfusion_live_dispatch_v1.frame_context import build_frame_context_v1
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.continuous_execution_v2 import ExecutionEnvelopeV3
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_live_wire_v2 as W
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP

from .plan import INPUT_SHAPE

NEU = Path(__file__).resolve().parents[3]
VALID_B1000_CELL = (NEU / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1"
                    "/20261006_run5_full_live_v1/valid_b1000/capture/cells/a71__favorable_stable")
VALID_B1000_ARCHIVE = VALID_B1000_CELL / "numeric_gt/service/archive"
PROFILER_WORKTREE = NEU / "abiodun_run4_single_gpu_profiler_v1"
STREAM = "synthetic-retention-stream"
RUN_ID, CELL_ID = "synthetic/run", "synthetic_cell"


def shared_input() -> torch.Tensor:
    return torch.zeros(INPUT_SHAPE, dtype=torch.float32)


def make_decision(*, frame_id, tensor_seq, decision_seq, session_uuid, q_e4=5000, mode_id=11,
                  anchor_action_id=None, payload=b"inner", stream_id=STREAM, ego=(1., 2., 3., 0., 10., 0.),
                  camera=None, lineage="a" * 64, bundle="b" * 64, capture_ns=None, gt_override=None,
                  keep_count=1000):
    capture_ns = 1_700_000_000_000_000_000 + frame_id * 50_000_000 if capture_ns is None else capture_ns
    kwargs = {} if camera is None else dict(camera_model_sha256=camera[0], camera_mount_sha256=camera[1])
    context = build_frame_context_v1(stream_id=stream_id, frame_id=frame_id, sequence_id=tensor_seq,
        capture_timestamp_ns=capture_ns, ego_world_x=ego[0], ego_world_y=ego[1], ego_world_z=ego[2],
        ego_world_pitch=ego[3], ego_world_yaw=ego[4], ego_world_roll=ego[5], **kwargs)
    envelope = ExecutionEnvelopeV3(mode_id=mode_id, q_e4=q_e4, keep_count=keep_count,
        anchor_action_id=anchor_action_id, reward_requested=True, session_uuid=session_uuid,
        controller_lineage_sha256=lineage, decision_seq=decision_seq, ticket_seq=decision_seq,
        frame_id=frame_id, tensor_seq=tensor_seq, capture_timestamp_ns=capture_ns,
        execution_bundle_sha256=bundle, inner_payload_sha256=hashlib.sha256(payload).hexdigest(),
        inner_payload=payload)
    planned = NS(reward_requested=True, session_uuid=session_uuid, frame_id=frame_id, tensor_seq=tensor_seq,
                 decision_seq=decision_seq, ticket_seq=decision_seq, capture_wall_ns=capture_ns, mode_id=mode_id,
                 q_e4=q_e4, carla_timestamp=frame_id * 0.05, radar_window_sha256="c" * 64,
                 plan=NS(kind=NS(value="POLICY_DECISION")))
    gt = MP.gt_identity(run_id=RUN_ID, cell_id=CELL_ID, stream_id=stream_id, frame_id=frame_id,
                        anchor_action_id=anchor_action_id,
                        anchor_profile_id=None if anchor_action_id is None else f"anchor_{anchor_action_id}",
                        capture_timestamp_ns=capture_ns)
    if gt_override is not None:
        gt = gt_override
    prepared = NS(wire=W.pack_sfd4(envelope, context), identity=MP.run4_identity(envelope), gt_identity=gt)
    return planned, prepared


def hold(frame_id):
    return NS(reward_requested=False, frame_id=frame_id, plan=NS(kind=NS(value="POLICY_HOLD")))


def fallback(frame_id):
    return NS(reward_requested=False, frame_id=frame_id, plan=NS(kind=NS(value="FALLBACK")))


def route(frames=999, *, seed=7, fallback_rate=0.01, session_every=14):
    """Frame-by-frame stream: policy decision / k_min2 hold alternation, rare fallbacks, session resets.

    Yields ("decision", planned, prepared, outcome) or ("hold"|"fallback", planned, None, None). The
    outcome (latency, Q_perc, payload) is random and never passed to the observer.
    """
    rng = random.Random(seed)
    frame, tensor, decision, session, sessions = 1592, 2, 0, None, 0
    need_decision = True
    for _ in range(frames):
        frame += 2
        if rng.random() < fallback_rate:
            yield ("fallback", fallback(frame), None, None)
            continue
        tensor += 1
        if need_decision:
            if session is None or decision >= session_every:
                session, decision, sessions = str(uuid.UUID(int=rng.getrandbits(128))), 0, sessions + 1
            outcome = {"latency_ms": rng.uniform(120, 260), "q_perc": rng.random(),
                       "payload_bytes": rng.randint(30_000, 120_000)}
            planned, prepared = make_decision(frame_id=frame, tensor_seq=tensor, decision_seq=decision,
                session_uuid=session, q_e4=rng.randint(3000, 8000), payload=rng.randbytes(64))
            decision += 1
            yield ("decision", planned, prepared, outcome)
        else:
            yield ("hold", hold(frame), None, None)
        need_decision = not need_decision


def drive(observer, events, tensor, *, device="cuda:0", fail=None):
    """Emulate materialize(): begin -> prepare_input -> observe_input -> end. ``fail(ordinal)``->exception."""
    decision_index = 0
    for kind, planned, prepared, _outcome in events:
        token = observer.begin(planned)
        if kind != "decision":
            if kind == "hold":       # holds still prepare an input; it must pass through untouched
                assert observer.observe_input(tensor, device) is tensor
            observer.end(token, None)
            continue
        produced = None
        try:
            if fail is not None and fail(decision_index):
                raise RuntimeError("injected production failure")
            assert observer.observe_input(tensor, device) is tensor
            produced = prepared
        except RuntimeError:
            pass
        finally:
            observer.end(token, produced)
            decision_index += 1


def worker_done():
    return [NS(is_alive=lambda: False, name="frame-worker")]


def archive_frames_by_tensor(archive=VALID_B1000_ARCHIVE):
    """Sorted (tensor_seq, frame_dir) of a numeric-GT archive = route-level policy-decision ordinal order."""
    rows = []
    for directory in sorted(Path(archive).glob("frame_*")):
        score = json.loads((directory / "scorer/numerical_score.json").read_bytes())
        rows.append((score["prediction_identity"]["action_identity"]["tensor_seq"], directory))
    return sorted(rows)


def decision_from_archive(frame_dir: Path, *, perturb_ego=False):
    """A synthetic retained decision whose wire/action/GT identities equal a real archived decision."""
    score = json.loads((frame_dir / "scorer/numerical_score.json").read_bytes())
    meta = json.loads((frame_dir / "scorer/target_metadata.json").read_bytes())
    anchor_path = frame_dir / "scorer/anchor_provenance.json"
    received = (json.loads(anchor_path.read_bytes())["prediction_identity_as_received"]
                if anchor_path.exists() else score["prediction_identity"])
    action, gt, ctx = received["action_identity"], received["gt_identity"], meta["frame_context"]
    ego = list(ctx["ego_world"])
    if perturb_ego:
        ego[0] += 0.5
    planned, prepared = make_decision(frame_id=action["frame_id"], tensor_seq=action["tensor_seq"],
        decision_seq=action["decision_seq"], session_uuid=action["session_uuid"], q_e4=action["q_e4"],
        mode_id=action["mode_id"], anchor_action_id=action["anchor_action_id"], stream_id=ctx["stream_id"],
        ego=tuple(ego), camera=(ctx["camera_model_sha256"], ctx["camera_mount_sha256"]),
        lineage=action["controller_lineage_sha256"], bundle=action["execution_bundle_sha256"],
        capture_ns=received["capture_timestamp_ns"], gt_override=gt)
    return planned, prepared
