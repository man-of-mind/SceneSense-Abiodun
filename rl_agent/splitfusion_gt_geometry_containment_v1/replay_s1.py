"""AVO-step replay of the saved S1 failure (14 dB session S1, frame 6146) -- a GEOMETRY-STEP replay, NOT a full
producer replay.  No producer/conflict history, owner or engine state of the live session is claimed restored.

Inputs are exact: ``freeze_failure_v1/SOURCE.bin`` is re-admitted through a fresh ``FreezeGate`` with its own REQUEST
descriptor (every buffer digest is re-verified by the gate); depth is decoded with the producer's bound decoder;
camera matrices are the owned source matrices.  ``numeric_avo`` needs nothing else (``splitfusion_avo_tilt_v1``
replay convention).  The S1 runtime's AVO step IS ``splitfusion_avo_tilt_v1.quality.numeric_avo`` (V16 registered
the tilt-aware switch), so re-running it must reproduce the live exception verbatim.

Checks: snapshot hashes; the failing frame/sequence from REQUEST (6146 / 1176 -- the UE's later connection reset
at frame 6160 is a consequence, not the failing request); live exception text reproduced exactly; the structured
diagnostics pass every classifier CONTENT check (TARGET, actor 106, footprint 0.0370927 m > 0.03 m); the
replayed error is NOT recognised by ``classify.recognise`` here only because the producer composition frame
(``composition.assemble``) is absent in an AVO-step replay (expected ``RAISE_SITE``).  The full composition path,
with the same pitch/roll on retained-corpus targets (identical footprint), is exercised by ``verify_engine`` and
``service_proof``.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -m rl_agent.splitfusion_gt_geometry_containment_v1.replay_s1 \\
    --snapshot <S1 capture root>/gt_gc_trace/freeze_failure_v1 --name <dir>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from rl_agent.splitfusion_avo_tilt_v1 import quality as Q, switch
from rl_agent.splitfusion_exact_gt_service_proto_v2 import freeze_codec as FZ
from rl_agent.splitfusion_numeric_gt_producer_v1 import originals as O
from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
from rl_agent.splitfusion_run5_live_avo_v1.quality_adapter import LiveGtContractError

from . import classify as CL

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "experiments/splitfusion_gt_geometry_containment_v1"
EXPECTED = dict(frame_id=6146, sequence=1176, actor="106", ue_reset_frame=6160)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(snapshot, name):
    snapshot = Path(snapshot).resolve(strict=True)
    ready = json.loads((snapshot / "READY.json").read_bytes())
    hashes_ok = {f: sha((snapshot / f).read_bytes()) == d for f, d in ready["sha256"].items()}
    error_record = json.loads((snapshot / "ERROR.json").read_bytes())
    request = json.loads((snapshot / "REQUEST.json").read_bytes())
    descriptor = json.loads(request["descriptor"])
    out = EVIDENCE / name
    out.mkdir(parents=True, exist_ok=False)
    if switch.installed():
        raise RuntimeError("the replay calls the tilt-aware step directly; the process switch must be off")
    fd, _ = FZ.seal([memoryview((snapshot / "SOURCE.bin").read_bytes())])
    gate = FZ.FreezeGate(descriptor["session_id"])
    gate.last_sequence, gate.last_frame = descriptor["sequence"] - 1, descriptor["frame_id"] - 1
    replay = dict(raised=False)
    try:
        with gate.admit(request["descriptor"].encode(), fd) as admitted:
            kw = admitted.kwargs
            depth = O.bound().visibility.decode_depth_bgra(
                np.frombuffer(kw["depth_image"].raw_data, np.uint8).reshape(720, 1280, 4).copy())
            try:
                Q.numeric_avo(identity=kw["identity"], rgb=kw["rgb"], semantic_image=kw["semantic_image"],
                              depth_image=kw["depth_image"], frozen_world=kw["frozen_world"],
                              camera_matrix=np.asarray(kw["camera_matrix"]),
                              camera_inverse=np.asarray(kw["camera_inverse"]), depth=depth, sync_validated=True)
            except LiveGtContractError as error:
                record, why = CL.content(error)
                full = CL.recognise(error)
                replay = dict(raised=True, type=type(error).__name__, message=str(error),
                              diagnostics=getattr(error, "geometry_diagnostics", None),
                              content_record=record, content_rejection=why,
                              recognise_in_avo_step_replay=dict(record=full[0], why=full[1]))
                traceback.clear_frames(error.__traceback__)      # release the borrowed views before unmapping
            del kw, depth
    finally:
        os.close(fd)
    offender = (replay.get("diagnostics") or {}).get("offender") or {}
    checks = dict(
        snapshot_hashes_verify=all(hashes_ok.values()),
        failing_request_is_frame_6146_sequence_1176=(descriptor["frame_id"], descriptor["sequence"])
            == (EXPECTED["frame_id"], EXPECTED["sequence"]),
        failing_frame_is_not_the_ue_reset_frame=descriptor["frame_id"] != EXPECTED["ue_reset_frame"],
        live_exception_reproduced_verbatim=replay.get("type") == error_record["exception_type"]
            and replay.get("message") == error_record["exception"],
        offender_is_actor_106_target=offender.get("gt_actor_id") == EXPECTED["actor"] and offender.get("role") == "TARGET",
        diagnostics_pass_every_content_check=replay.get("content_rejection") is None
            and (replay.get("content_record") or {}).get("reason") == CL.REASONS[CL.STATUS_TARGET],
        footprint_above_unchanged_envelope=(offender.get("footprint_tilt_m") or 0) > CL.core.GROUND_REJECT_MARGIN_M,
        avo_step_replay_lacks_composition_frame=replay.get("recognise_in_avo_step_replay", {}).get("why") == "RAISE_SITE")
    result = dict(schema=CL.VERSION + ".replay_s1", replay_kind="AVO_GEOMETRY_STEP_REPLAY", full_producer_replay=False,
                  historical_producer_state_restored=False, snapshot=str(snapshot), snapshot_ready_sha256=ready["sha256"],
                  frame_id=descriptor["frame_id"], sequence=descriptor["sequence"],
                  capture_template=descriptor["capture_template"],
                  live_exception=dict(type=error_record["exception_type"], message=error_record["exception"]),
                  replay=replay, checks=checks, status="PASS" if all(checks.values()) else "FAIL",
                  live_services_started=False)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    (out / "OUTPUT_SHA256.json").write_text(json.dumps({"RESULT.json": sha((out / "RESULT.json").read_bytes())},
                                                       indent=1) + "\n")
    print(json.dumps(dict(status=result["status"], checks=checks, offender=offender), indent=1, default=str))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.name)
