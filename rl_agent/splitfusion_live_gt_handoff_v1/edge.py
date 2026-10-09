"""Exact evaluator/pose composition with a bounded in-memory GT handoff.

This additive scope starts one local broker, not a route or model. It replaces
durable-file readiness only. The retained asynchronous evaluator, quality
formula, feedback encoder, queue/GT watchdogs and prediction archive run
unchanged. ``target_bundle_sha256`` in retained compatibility fields means the
explicitly labelled scoring-packet digest here, never a durable READY digest.
Future live registration/postflight must bind that packet to its final archive.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from . import broker as B, scoring_packet as P

SCHEMA = "splitfusion.gt_packet_edge_scope.v1"
TRANSPORT = "IMMUTABLE_SCORING_PACKET__DURABILITY_DEFERRED__EXACT_NUMERICS_V1"
REFERENCE_KIND = "SCORING_PACKET_SHA256_NOT_DURABLE_READY_SHA256"


def _require(value, message):
    if not value:
        raise B.BrokerError(message)


def _raw():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


@dataclass(frozen=True)
class _PacketBundle:
    """Private pose adapter; deliberately not the disk TargetBundle type."""
    identity: dict
    manifest_sha256: str
    input_metadata: dict
    gt: dict


def _packet_camera(bundle):
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    _require(isinstance(bundle, _PacketBundle), "packet pose requires a validated memory bundle")
    metadata = bundle.input_metadata
    P.T._validate_metadata(metadata, dict(bundle.identity))
    inverse = pose._matrix(json.loads(metadata["frame"]["camera_inverse_matrix_json"]), "target inverse")
    # Identical to the retained target-camera reconstruction; no new rounding
    # rule or source camera estimate is introduced.
    return pose._matrix(np.linalg.inv(inverse), "target camera")


def _private_pose_scope(runtime_proxy, target_proxy):
    """Retain every pose guard, changing only its private acquisition modules."""
    from rl_agent.splitfusion_live_parity_runtime_v1 import pose
    source = inspect.getsource(pose.install_edge_pose)
    before = "    from rl_agent.splitfusion_live_parity_repair_v1 import runtime as R, target_transport as T\n"
    _require(source.count(before) == 1, "retained pose import seam drift")
    original_sha = hashlib.sha256(source.encode()).hexdigest()
    source = source.replace(before, "    R, T = _handoff_runtime, _handoff_target\n", 1)

    class PacketPoseArchive(pose.PoseArchive):
        def add(self, kind, **fields):
            if kind == "EDGE_TARGET_RECONCILED":
                fields.update(target_reference_kind=REFERENCE_KIND,
                    scoring_packet_sha256=fields["target_manifest_sha256"],
                    gt_transport_semantics=TRANSPORT)
            return super().add(kind, **fields)

    namespace = dict(pose.__dict__, _handoff_runtime=runtime_proxy,
        _handoff_target=target_proxy, _verified_target_camera=_packet_camera,
        PoseArchive=PacketPoseArchive)
    exec(compile(source, "<private-packet-pose-composition>", "exec"), namespace)
    return namespace["install_edge_pose"], original_sha


@contextmanager
def install_edge_handoff(*, socket_path, binding, max_frames, archive_directory,
                         pose_archive_directory=None, publisher_uid=None, publisher_gid=None):
    """Install before evaluator construction; close evaluator before scope exit.

    Surround this scope with the retained prediction-contract scope. Do not
    also enter the old pose/evaluator scopes: both are composed privately here.
    Returned state remains available after exit and records bounded broker
    admission, consumption and genuine evaluator/prediction/pose close results.
    The existing evaluator's probe/GT timeout arguments are passed unchanged;
    they are not the agent's 170-ms reward deadline and this scope adds none.
    """
    from rl_agent.splitfusion_live_parity_repair_v1 import runtime as R
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2 as E
    from rl_agent.splitfusion_quality_feedback_probe_v1 import gt_evidence as G
    P.validate_binding(binding)
    _require(type(max_frames) is int and 0 < max_frames <= B.MAX_FRAMES, "bounded edge frame count required")
    archive_directory = Path(archive_directory)
    pose_directory = (archive_directory.parent / "pose_archive" if pose_archive_directory is None
                      else Path(pose_archive_directory))
    _require(not archive_directory.exists() and not pose_directory.exists(), "edge archives must be create-only")
    server = B.Server(socket_path, binding=binding, max_jobs=min(2, max_frames), max_frames=max_frames,
                      publisher_uid=publisher_uid, publisher_gid=publisher_gid)
    state = dict(schema=SCHEMA, binding=binding, transport_semantics=TRANSPORT,
        target_reference_kind=REFERENCE_KIND, max_frames=max_frames, errors=[], reads=[],
        complete=False, durability_claim=False, new_broker_threads=1,
        reward_deadline_changed=False, quality_formula_changed=False,
        publisher_credential_policy=dict(server.credential_policy))

    def packet_read(_directory, *, expected_identity, timeout_s):
        # The evaluator supplies its existing bounded probe or synchronous
        # wait. A not-ready error remains the retained evaluator's retry path.
        payload = server.take(expected_identity, timeout_s=timeout_s)
        observed = server.observe(expected_identity)
        _require(observed is not None and observed["consumed"], "consumed packet lacks admission evidence")
        try:
            gt = P.decode_packet(payload, expected_identity=expected_identity, binding=binding)
        except Exception as exc:
            state["errors"].append(f"packet decode: {type(exc).__name__}: {exc}")
            raise
        _require(gt["manifest_sha256"] == observed["packet_sha256"], "packet admission/read digest differs")
        gt.update(producer_gt_ready_wall_ns=gt["gt_ready_wall_ns"],
            producer_gt_ready_raw_ns=gt["gt_ready_raw_ns"],
            gt_ready_wall_ns=observed["admitted_wall_ns"], gt_ready_raw_ns=observed["admitted_raw_ns"],
            packet_decode_completed_raw_ns=_raw(), gt_transport_semantics=TRANSPORT,
            target_reference_kind=REFERENCE_KIND)
        state["reads"].append(dict(identity=dict(expected_identity), packet_sha256=gt["manifest_sha256"],
            prepared_content_sha256=gt["prepared"].content_sha256,
            input_content_sha256=gt["input_content_sha256"], avo_sha256=gt["avo_sidecar_sha256"],
            input_metadata_sha256=P.digest(P.T._canonical(gt["input_metadata"])),
            admitted_raw_ns=gt["gt_ready_raw_ns"], admitted_wall_ns=gt["gt_ready_wall_ns"],
            decoded_raw_ns=gt["packet_decode_completed_raw_ns"]))
        return _PacketBundle(dict(expected_identity), gt["manifest_sha256"], gt["input_metadata"], gt)

    target_proxy = SimpleNamespace(read_bundle=packet_read)

    @contextmanager
    def evaluation_scope(**kwargs):
        with R.install_edge_evaluation(**kwargs) as prediction_state:
            klass = E.Run4EvaluatorV2
            originals = (G.read_ground_truth, klass._observe_components, klass._record_read,
                         klass._evaluate_with, klass.close, klass.GT_COMPONENTS, klass.__init__)
            state["prediction_state"] = prediction_state
            evaluators, closed_evaluators = [], set()

            def initialize(evaluator, *args, **arguments):
                _require(not evaluators, "one retained evaluator per packet scope required")
                originals[6](evaluator, *args, **arguments)
                evaluators.append(evaluator)

            def component(observed):
                return dict(reference_kind=REFERENCE_KIND, packet_sha256=observed["packet_sha256"],
                    first_observed_wall_ns=observed["admitted_wall_ns"],
                    ready_raw_ns=observed["admitted_raw_ns"], size_bytes=observed["bytes"])

            def read(directory, *, expected_identity, timeout_s):
                # The pose scope wraps only our private proxy and verifies
                # packet metadata before returning these exact scoring inputs.
                return target_proxy.read_bundle(directory, expected_identity=expected_identity,
                    timeout_s=timeout_s).gt

            def observe(evaluator, ticket, timing):
                handoff = timing.setdefault("gt_handoff", dict(identity=dict(ticket.gt_identity),
                    transport_semantics=TRANSPORT, read_errors={}, components={}))
                observed = server.observe(ticket.gt_identity)
                if observed is not None:
                    handoff["components"]["scoring_packet"] = component(observed)

            def record_read(evaluator, timing, *, error=None):
                handoff = timing.get("gt_handoff")
                if handoff is None:
                    return
                if error is not None:
                    message = f"{type(error).__name__}: {error}"[:200]
                    errors = handoff["read_errors"]
                    errors[message] = errors.get(message, 0) + 1
                else:
                    handoff["read_success_wall_ns"] = time.time_ns()
                    observed = server.observe(handoff["identity"])
                    if observed is not None:
                        handoff["components"]["scoring_packet"] = component(observed)

            def evaluate(evaluator, ticket, gt, timing):
                timing.update(gt_transport_semantics=TRANSPORT, target_reference_kind=REFERENCE_KIND,
                    scoring_packet_sha256=gt["manifest_sha256"],
                    prepared_content_sha256=gt["prepared"].content_sha256,
                    input_content_sha256=gt["input_content_sha256"],
                    producer_gt_ready_raw_ns=gt["producer_gt_ready_raw_ns"],
                    producer_gt_ready_wall_ns=gt["producer_gt_ready_wall_ns"],
                    packet_admitted_raw_ns=gt["gt_ready_raw_ns"],
                    packet_admitted_wall_ns=gt["gt_ready_wall_ns"],
                    packet_decode_completed_raw_ns=gt["packet_decode_completed_raw_ns"])
                return originals[3](evaluator, ticket, gt, timing)

            def close(evaluator, timeout_s=None):
                result = originals[4](evaluator, timeout_s)
                state["evaluator_close"] = dict(result)
                if result.get("worker_alive") is False:
                    closed_evaluators.add(id(evaluator))
                return result

            G.read_ground_truth = read
            klass._observe_components, klass._record_read = observe, record_read
            klass._evaluate_with, klass.close = evaluate, close
            klass.GT_COMPONENTS = ("immutable_scoring_packet",)
            klass.__init__ = initialize
            try:
                yield prediction_state
            finally:
                # On failures, drain before restoring scorer/pose globals; no
                # live evaluator may resume against a different GT reader.
                for evaluator in evaluators:
                    if id(evaluator) not in closed_evaluators:
                        try:
                            evaluator.close()
                        except Exception as exc:
                            state["errors"].append(f"evaluator cleanup: {type(exc).__name__}: {exc}")
                (G.read_ground_truth, klass._observe_components, klass._record_read,
                 klass._evaluate_with, klass.close, klass.GT_COMPONENTS, klass.__init__) = originals

    runtime_proxy = SimpleNamespace(install_edge_evaluation=evaluation_scope)
    pose_scope, pose_sha = _private_pose_scope(runtime_proxy, target_proxy)
    state["retained_pose_function_sha256"] = pose_sha
    failed = False
    try:
        server.start()
        with pose_scope(archive_directory=pose_directory, max_frames=max_frames) as pose_state:
            state["pose_state"] = pose_state
            with runtime_proxy.install_edge_evaluation(archive_directory=archive_directory,
                                                     max_frames=max_frames):
                yield state
    except BaseException as exc:
        failed = True
        state["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            state["broker"] = server.close()
            closed = state.get("evaluator_close", {})
            pose_state = state.get("pose_state", {})
            prediction = state.get("prediction_state", {})
            state["complete"] = (not state["errors"] and state["broker"]["failure"] is None
                and state["broker"]["all_consumed"] and not state["broker"]["worker_alive"]
                and closed.get("worker_alive") is False and closed.get("parity_archive_complete") is True
                and prediction.get("flushed") is True and not prediction.get("archive_errors")
                and pose_state.get("flushed") is True and not pose_state.get("errors"))
            # Retain bounded primitive proof only. Archive instances are live
            # implementation owners, not serializable run evidence.
            state["prediction_state"] = {key: value for key, value in prediction.items() if key != "archive"}
            state["pose_state"] = {key: value for key, value in pose_state.items() if key != "archive"}
            if not failed and not state["complete"]:
                raise B.BrokerError("edge packet scope did not close/drain/preserve cleanly")
        except BaseException as exc:
            state["errors"].append(f"{type(exc).__name__}: {exc}")
            if not failed:
                raise
