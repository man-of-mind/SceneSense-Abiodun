"""W10275 process boundary for isolated Run-4B/Run-5B validation.

The process owns the UE-clock operational ledger and the local evidence roots.
It accepts final frame/action identities from an injected B pipeline (the
composition point for the proven CARLA/RAN/sensor/front/SFD sender seams),
receives the compact tail-output ACK, and records exactly one operational
outcome for every policy decision.  CARLA ground truth is optional local-only
post-run evidence and can never close a ticket or affect the live state.

Production preflight is intentionally blocked until a final joint-channel
actor artifact manifest and its weight file are supplied beside the selected
live manifest.  ``--offline-fake`` exercises all 300-frame accounting and
durable-evidence mechanics without launching CARLA, OAI, a socket, or CUDA.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Optional, Protocol, Sequence
import uuid

import numpy as np

from . import b_validation_runner_v1 as V
from . import frozen_actor_loader_v1 as F
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A
from . import operational_trace_v1 as T
from . import postrun_artifact_v1 as G


REQUEST_SCHEMA = "scenesense.splitfusion.run4b5b.process_request.v1"
RESULT_SCHEMA = "scenesense.splitfusion.run4b5b.b_ue_process_result.v1"
REPORT_SCHEMA = "scenesense.splitfusion.run4b5b.b_ue_process_report.v1"
EXECUTE_TOKEN = "SPLITFUSION_RUN4B5B_LIVE_VALIDATION_V1_EXECUTE"
ROLE = "UE_FRONT"
TRANSMITTED_BUDGET = 300
FINAL_ARTIFACT_MANIFEST_NAME = "FINAL_JOINT_CHANNEL_ACTOR_ARTIFACT_MANIFEST.json"
WEIGHTS_NAME = "actor_state_dict.pt"
RESULT_NAME = "B_UE_RESULT.json"
REPORT_NAME = "B_UE_REPORT.json"

REQUIRED_AUTHORITIES = (
    "rl_agent.splitfusion_run4_split_host_l10319_v1.contract",
    "rl_agent.splitfusion_run4_split_host_l10319_v1.local_ran_lifecycle_v1",
    "rl_agent.splitfusion_run4_split_host_l10319_v1.local_ran_executor_v1",
)


class BUEProcessError(RuntimeError):
    """The UE process request, binding, runtime, or evidence is unsafe."""


class FinalActorUnavailableError(BUEProcessError):
    """The final joint-channel actor authority has not been supplied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BUEProcessError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise BUEProcessError("value is not canonicalizable") from exc


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _decode_request(value: str) -> Mapping[str, Any]:
    _require(type(value) is str and bool(value), "request-b64 is empty")
    try:
        payload = base64.b64decode(value.encode("ascii"), altchars=b"-_",
                                   validate=True)
        raw = json.loads(payload.decode("ascii"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise BUEProcessError("request-b64 is invalid") from exc
    _require(type(raw) is dict and payload == _canonical(raw),
             "request is not canonical")
    return raw


@dataclass(frozen=True, slots=True)
class BUEProcessRequestV1:
    run_id: str
    variant: L.ActorVariant
    config_binding_sha256: str
    actor_boundary_sha256: str
    feature_schema_sha256: str
    transmitted_budget: int
    deadline_ns: int
    ack_semantics: str
    postrun_semantics: str
    clock_domain: str
    split_host: V.SplitHostBindingV1
    output_root: Path
    evidence_root: Path
    actor_manifest_path: Path
    required_authority_modules: tuple[str, ...]

    @classmethod
    def from_b64(cls, value: str) -> "BUEProcessRequestV1":
        raw = _decode_request(value)
        fields = {
            "schema", "role", "run_id", "variant",
            "config_binding_sha256", "actor_boundary_sha256",
            "feature_schema_sha256", "transmitted_budget", "deadline_ns",
            "ack_semantics", "postrun_semantics", "clock_domain",
            "split_host", "output_root", "evidence_root",
            "actor_manifest_path", "remote_attempt_root",
            "required_authority_modules", "old_live_quality_runtime_permitted",
        }
        _require(set(raw) == fields, "request fields are incomplete or foreign")
        _require(raw["schema"] == REQUEST_SCHEMA and raw["role"] == ROLE,
                 "request schema/role drift")
        _require(raw["remote_attempt_root"] is None,
                 "UE request carries a remote attempt root")
        _require(raw["old_live_quality_runtime_permitted"] is False,
                 "old live-quality runtime was authorized")
        try:
            variant = L.ActorVariant(raw["variant"])
        except (TypeError, ValueError) as exc:
            raise BUEProcessError("unknown B actor variant") from exc
        split = V.SplitHostBindingV1.from_mapping(raw["split_host"])
        for digest in (raw["config_binding_sha256"],
                       raw["actor_boundary_sha256"],
                       raw["feature_schema_sha256"]):
            _require(type(digest) is str and len(digest) == 64
                     and all(ch in "0123456789abcdef" for ch in digest),
                     "request contains a non-SHA-256 binding")
        for field in ("output_root", "evidence_root", "actor_manifest_path"):
            path = Path(raw[field])
            _require(path.is_absolute() and ".." not in path.parts,
                     f"{field} must be absolute without parent traversal")
        request = cls(
            run_id=raw["run_id"], variant=variant,
            config_binding_sha256=raw["config_binding_sha256"],
            actor_boundary_sha256=raw["actor_boundary_sha256"],
            feature_schema_sha256=raw["feature_schema_sha256"],
            transmitted_budget=raw["transmitted_budget"],
            deadline_ns=raw["deadline_ns"], ack_semantics=raw["ack_semantics"],
            postrun_semantics=raw["postrun_semantics"],
            clock_domain=raw["clock_domain"], split_host=split,
            output_root=Path(raw["output_root"]),
            evidence_root=Path(raw["evidence_root"]),
            actor_manifest_path=Path(raw["actor_manifest_path"]),
            required_authority_modules=tuple(raw["required_authority_modules"]),
        )
        request.validate()
        return request

    def validate(self) -> None:
        _require(self.transmitted_budget == TRANSMITTED_BUDGET,
                 "UE process budget is not exactly 300")
        _require(self.deadline_ns == A.ACK_DEADLINE_NS,
                 "operational deadline drift")
        _require(self.ack_semantics == V.ACK_SEMANTICS
                 and self.postrun_semantics == V.POSTRUN_SEMANTICS,
                 "ACK/post-run semantics drift")
        _require(self.clock_domain == A.CLOCK_DOMAIN,
                 "UE process clock is not CLOCK_MONOTONIC_RAW")
        _require(self.required_authority_modules == REQUIRED_AUTHORITIES,
                 "UE lifecycle authority set drift")
        _require(self.output_root != self.evidence_root,
                 "output and evidence roots overlap")
        _require(self.split_host.ue_host == "W10275.idcc.lab"
                 and self.split_host.ack_receiver_host == self.split_host.ue_host,
                 "UE/ACK host differs from W10275")
        expected_order = L.expected_feature_order(self.variant)
        _require(self.feature_schema_sha256
                 == L.feature_schema_sha256(self.variant, expected_order),
                 "request feature schema differs from exact B order")


def load_selected_live_manifest(request: BUEProcessRequestV1,
                                ) -> L.BActorManifestV1:
    path = request.actor_manifest_path
    _require(path.is_file() and not path.is_symlink(),
             "selected live actor manifest is absent or a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BUEProcessError("selected live actor manifest is unreadable") from exc
    manifest = L.BActorManifestV1.from_mapping(raw)
    _require(manifest.variant is request.variant,
             "selected manifest variant drift")
    _require(manifest.actor_boundary_sha256 == request.actor_boundary_sha256
             and manifest.feature_schema_sha256
             == request.feature_schema_sha256,
             "selected manifest binding drift")
    return manifest


def load_final_actor(request: BUEProcessRequestV1,
                     ) -> F.LoadedFrozenActorV1:
    """Load only a separately supplied final joint-channel actor authority."""
    selected = load_selected_live_manifest(request)
    manifest_path = request.actor_manifest_path.with_name(
        FINAL_ARTIFACT_MANIFEST_NAME)
    weights_path = request.actor_manifest_path.with_name(WEIGHTS_NAME)
    if not manifest_path.is_file() or not weights_path.is_file():
        raise FinalActorUnavailableError(
            "final joint-channel actor manifest/weights are not supplied")
    artifact = F.load_manifest(manifest_path)
    expected_variant = (F.ActorVariant.RUN4B
                        if request.variant is L.ActorVariant.RUN4B
                        else F.ActorVariant.RUN5B)
    _require(artifact.variant is expected_variant,
             "final artifact variant differs")
    _require(artifact.actor_boundary_sha256
             == selected.actor_boundary_sha256,
             "final artifact boundary differs from selected live manifest")
    _require(artifact.feature_schema_sha256
             == selected.feature_schema_sha256,
             "final artifact feature schema differs")
    _require(artifact.weights_file_sha256 == selected.weights_file_sha256,
             "final artifact weight digest differs")
    return F.load_frozen_actor(weights_path, artifact,
                               expected_variant=expected_variant)


@dataclass(frozen=True, slots=True)
class BTransmissionV1:
    """One already-sent SFD frame returned by the injected B pipeline."""

    identity: A.FrameActionIdentityV1
    action_open_monotonic_raw_ns: int
    payload_bytes: int
    decision_frame: bool
    gt_objects: Optional[tuple[Mapping[str, Any], ...]] = None
    gt_semantic_mask: Optional[np.ndarray] = None
    gt_recorded_monotonic_raw_ns: Optional[int] = None

    def __post_init__(self) -> None:
        _require(type(self.identity) is A.FrameActionIdentityV1,
                 "transmission identity is foreign")
        _require(type(self.action_open_monotonic_raw_ns) is int
                 and self.action_open_monotonic_raw_ns >= 0,
                 "action-open stamp is invalid")
        _require(type(self.payload_bytes) is int and self.payload_bytes > 0,
                 "payload size is invalid")
        _require(type(self.decision_frame) is bool,
                 "decision_frame must be exact bool")
        supplied = (self.gt_objects is not None,
                    self.gt_semantic_mask is not None,
                    self.gt_recorded_monotonic_raw_ns is not None)
        _require(len(set(supplied)) == 1,
                 "GT evidence must be wholly present or absent")


class BUEPipelineV1(Protocol):
    """Composition seam for proven sensor/state/actor/front/SFD components."""

    variant: L.ActorVariant
    feature_schema_sha256: str
    actor_boundary_sha256: str

    def transmit_next(
        self, frame_index: int, previous: Optional[A.OperationalOutcomeV1]
    ) -> BTransmissionV1: ...

    def close(self) -> None: ...


class AckReceiverV1(Protocol):
    def receive_until(self, identity: A.FrameActionIdentityV1,
                      deadline_monotonic_raw_ns: int,
                      ) -> Optional[tuple[bytes, int]]: ...
    def close(self) -> None: ...


class UdpOperationalAckReceiverV1:
    """UE-side compact-ACK endpoint stamped only on CLOCK_MONOTONIC_RAW."""

    def __init__(self, host: str, port: int) -> None:
        _require(type(host) is str and bool(host), "ACK bind host is empty")
        _require(type(port) is int and 1024 <= port <= 65535,
                 "ACK bind port is invalid")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))

    def receive_until(self, identity: A.FrameActionIdentityV1,
                      deadline_monotonic_raw_ns: int,
                      ) -> Optional[tuple[bytes, int]]:
        while True:
            now = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            remaining = deadline_monotonic_raw_ns - now
            if remaining < 0:
                return None
            self._socket.settimeout(remaining / 1_000_000_000)
            try:
                packet, _peer = self._socket.recvfrom(1 << 20)
            except socket.timeout:
                return None
            receipt = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            ack = A.decode_ack(packet)
            # Unknown ACKs are still returned so the durable ledger, not this
            # transport, owns orphan classification.
            if ack.identity == identity:
                return packet, receipt
            return packet, receipt

    def close(self) -> None:
        self._socket.close()


def execute_300(
    request: BUEProcessRequestV1, pipeline: BUEPipelineV1,
    receiver: AckReceiverV1,
) -> dict[str, Any]:
    """Execute exactly 300 transmitted frames and seal durable UE evidence."""
    _require(pipeline.variant is request.variant,
             "pipeline variant differs from request")
    _require(pipeline.feature_schema_sha256 == request.feature_schema_sha256,
             "pipeline feature schema differs")
    _require(pipeline.actor_boundary_sha256 == request.actor_boundary_sha256,
             "pipeline actor boundary differs")
    _require(not request.output_root.exists() and not request.evidence_root.exists(),
             "UE output/evidence roots must be create-only")
    request.output_root.mkdir(parents=True)
    request.evidence_root.mkdir(parents=True)
    operational_store = A.OperationalEvidenceStoreV1.create(
        request.evidence_root / "operational_evidence")
    trace_store = T.OperationalTraceStoreV1.create(
        request.evidence_root / "operational_trace")
    gt_store = G.GroundTruthEvidenceStoreV1.create(
        request.evidence_root / "carla_gt")
    ledger = A.OperationalAckLedgerV1(evidence_store=operational_store)
    previous: Optional[A.OperationalOutcomeV1] = None
    decisions = successes = timeouts = gt_records = 0
    primary: Optional[BaseException] = None
    try:
        for index in range(request.transmitted_budget):
            sent = pipeline.transmit_next(index, previous)
            _require(type(sent) is BTransmissionV1,
                     "pipeline returned a foreign transmission")
            _require(sent.identity.run_id == request.run_id,
                     "transmission run identity drift")
            if sent.gt_objects is not None:
                gt_store.write(
                    identity=sent.identity, eligible_objects=sent.gt_objects,
                    semantic_mask=sent.gt_semantic_mask,
                    recorded_monotonic_raw_ns=sent.gt_recorded_monotonic_raw_ns)
                gt_records += 1
            if not sent.decision_frame:
                continue
            decisions += 1
            ledger.open(sent.identity, sent.action_open_monotonic_raw_ns)
            deadline = sent.action_open_monotonic_raw_ns + A.ACK_DEADLINE_NS
            received = receiver.receive_until(sent.identity, deadline)
            if received is None:
                ledger.poll(deadline + 1)
            else:
                packet, receipt = received
                ledger.receive(packet, receipt)
                # A receiver can return an unknown packet first.  Close this
                # exact ticket at the inclusive boundary if it remains open.
                if ledger.outcome(sent.identity) is None:
                    ledger.poll(deadline + 1)
            outcome = ledger.outcome(sent.identity)
            _require(type(outcome) is A.OperationalOutcomeV1,
                     "decision did not reach an operational terminal")
            trace_store.write(outcome, sent.payload_bytes)
            previous = outcome
            if outcome.success:
                successes += 1
            else:
                timeouts += 1
    except BaseException as exc:
        primary = exc
        raise
    finally:
        errors: list[BaseException] = []
        for resource in (receiver, pipeline):
            try:
                resource.close()
            except BaseException as exc:
                errors.append(exc)
        if primary is None and errors:
            raise BUEProcessError(
                "UE process cleanup failed: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors))

    snapshot = operational_store.verify_all(require_all_resolved=True)
    traces = trace_store.verify_all()
    ground_truth = gt_store.verify_all()
    _require(len(snapshot.outcomes) == decisions == len(traces),
             "operational decision evidence does not reconcile")
    _require(successes + timeouts == decisions,
             "operational terminals do not reconcile")
    _require(len(ground_truth) == gt_records,
             "ground-truth evidence does not reconcile")
    report = {
        "schema": REPORT_SCHEMA, "run_id": request.run_id,
        "variant": request.variant.value,
        "transmitted_frames": request.transmitted_budget,
        "policy_decisions": decisions, "operational_successes": successes,
        "operational_timeouts": timeouts, "ground_truth_records": gt_records,
        "live_qperc_computed": False, "live_reward_computed": False,
        "gt_used_for_ack_or_state": False,
        "clock_domain": A.CLOCK_DOMAIN,
        "config_binding_sha256": request.config_binding_sha256,
        "actor_boundary_sha256": request.actor_boundary_sha256,
    }
    report_bytes = _canonical(report) + b"\n"
    report_path = request.output_root / REPORT_NAME
    with report_path.open("xb") as handle:
        handle.write(report_bytes)
    result = {
        "schema": RESULT_SCHEMA, "run_id": request.run_id,
        "variant": request.variant.value,
        "transmitted_frames": request.transmitted_budget,
        "terminal_status": "COMPLETE",
        "result_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "config_binding_sha256": request.config_binding_sha256,
        "actor_boundary_sha256": request.actor_boundary_sha256,
    }
    with (request.output_root / RESULT_NAME).open("xb") as handle:
        handle.write(_canonical(result) + b"\n")
    return result


class _OfflineFakePipeline:
    """Deterministic mechanics fixture; never a performance simulation."""

    def __init__(self, request: BUEProcessRequestV1) -> None:
        self.request = request
        self.variant = request.variant
        self.feature_schema_sha256 = request.feature_schema_sha256
        self.actor_boundary_sha256 = request.actor_boundary_sha256
        self.sent: dict[str, int] = {}

    def transmit_next(self, frame_index: int,
                      previous: Optional[A.OperationalOutcomeV1]
                      ) -> BTransmissionV1:
        opened = 10_000_000_000 + frame_index * 250_000_000
        identity = A.FrameActionIdentityV1(
            run_id=self.request.run_id, cell_id="offline_fake",
            stream_id="ue0_route_b",
            session_uuid=str(uuid.UUID(int=1)),
            controller_lineage_sha256=hashlib.sha256(b"offline").hexdigest(),
            decision_seq=frame_index, ticket_seq=frame_index,
            frame_id=frame_index, tensor_seq=frame_index,
            capture_timestamp_ns=opened - 10_000_000,
            mode_id=11, q_e4=3000, keep_count=7000,
            anchor_action_id=67, profile_id="split_ae32_uint4_q3000",
            execution_bundle_sha256=hashlib.sha256(b"bundle").hexdigest())
        self.sent[identity.exact_sha256()] = opened
        return BTransmissionV1(
            identity=identity, action_open_monotonic_raw_ns=opened,
            payload_bytes=177_000, decision_frame=True,
            gt_objects=(), gt_semantic_mask=np.zeros((1, 1), dtype=np.uint8),
            gt_recorded_monotonic_raw_ns=opened + 1)

    def close(self) -> None: return None


class _OfflineFakeReceiver:
    def __init__(self, pipeline: _OfflineFakePipeline) -> None:
        self.pipeline = pipeline

    def receive_until(self, identity: A.FrameActionIdentityV1,
                      deadline_monotonic_raw_ns: int,
                      ) -> Optional[tuple[bytes, int]]:
        opened = self.pipeline.sent[identity.exact_sha256()]
        if identity.frame_id % 10 == 0:
            return None
        ack = A.TailOutputAckV1.success(identity, b"offline-tail-output")
        return A.encode_ack(ack), opened + 100_000_000

    def close(self) -> None: return None


def production_preflight(request: BUEProcessRequestV1) -> dict[str, Any]:
    manifest = load_selected_live_manifest(request)
    actor = load_final_actor(request)
    return {
        "status": "B_UE_PRODUCTION_PREFLIGHT_PASS",
        "run_id": request.run_id, "variant": request.variant.value,
        "budget": request.transmitted_budget,
        "actor_boundary_sha256": manifest.actor_boundary_sha256,
        "loaded_actor_boundary_sha256": actor.manifest.actor_boundary_sha256,
        "services_launched": False,
    }


def offline_fake(request: BUEProcessRequestV1) -> dict[str, Any]:
    load_selected_live_manifest(request)
    pipeline = _OfflineFakePipeline(request)
    return execute_300(request, pipeline, _OfflineFakeReceiver(pipeline))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", nargs="?",
                        choices=("preflight", "offline-fake", "run", "stop"))
    operations = parser.add_mutually_exclusive_group()
    operations.add_argument("--preflight", action="store_true")
    operations.add_argument("--offline-fake", action="store_true")
    operations.add_argument("--run", action="store_true")
    operations.add_argument("--stop", action="store_true")
    parser.add_argument("--request-b64", required=True)
    parser.add_argument("--execute")
    args = parser.parse_args(argv)
    selected = [name for name in ("preflight", "offline_fake", "run", "stop")
                if getattr(args, name)]
    if args.operation is not None:
        selected.append(args.operation.replace("-", "_"))
    _require(len(selected) == 1, "select exactly one UE process operation")
    operation = selected[0]
    request = BUEProcessRequestV1.from_b64(args.request_b64)
    if operation == "preflight":
        print(json.dumps(production_preflight(request), sort_keys=True))
        return 0
    if operation == "offline_fake":
        result = offline_fake(request)
        print(json.dumps(result, sort_keys=True))
        return 0
    _require(args.execute == EXECUTE_TOKEN, "production execute token missing")
    if operation == "stop":
        # The real pipeline must own process-scoped teardown.  With no live
        # runtime started by this module yet, stop is safely idempotent.
        print(json.dumps({"status": "B_UE_NOT_RUNNING",
                          "services_stopped": 0}, sort_keys=True))
        return 0
    # Do not silently reuse the old live-quality child.  Actor authority is
    # checked first, then the missing final production pipeline is explicit.
    production_preflight(request)
    raise BUEProcessError(
        "B UE production CARLA/RAN/sensor/front/SFD pipeline is not bound")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUEST_SCHEMA", "RESULT_SCHEMA", "REPORT_SCHEMA", "EXECUTE_TOKEN",
    "FINAL_ARTIFACT_MANIFEST_NAME", "WEIGHTS_NAME", "BUEProcessError",
    "FinalActorUnavailableError", "BUEProcessRequestV1", "BTransmissionV1",
    "BUEPipelineV1", "AckReceiverV1", "UdpOperationalAckReceiverV1",
    "load_selected_live_manifest", "load_final_actor", "execute_300",
    "production_preflight", "offline_fake", "main",
]
