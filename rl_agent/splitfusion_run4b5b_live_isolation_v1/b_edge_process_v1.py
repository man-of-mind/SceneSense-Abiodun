"""B-variant CN/edge entry point and exact post-compute seam.

The qualified Phase-6 edge owns feature reassembly, dynamic-profile
verification, dequantization, optional AE decoding and the frozen tail.  This
module does not duplicate that machinery.  It defines the one additive seam
that the qualified runtime must call after it has produced usable object
records and a semantic mask:

``usable output -> synchronous operational ACK -> map/prediction queues``.

Ground truth, Q_perc, reward calculation, map completion and the historical
quality-feedback wire are deliberately absent.  Importing this module opens no
socket, starts no thread and imports neither torch nor the live edge runtime.

The current qualified runtime has no injectable post-compute hook and embeds
``map -> GT evaluator`` in its private loop.  Consequently ``start`` refuses
with :class:`RuntimeSeamRequired` until that small hook exists.  ``preflight``
still validates the exact process request and authorities, and
``offline-fake`` executes this module's complete ordering/evidence path
without network, CUDA or services.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import branch_evidence_v1 as B
from . import live_adapters_v1 as L
from . import operational_ack_v1 as A


REQUEST_SCHEMA = "scenesense.splitfusion.run4b5b.process_request.v1"
PREFLIGHT_SCHEMA = "scenesense.splitfusion.run4b5b.b_edge_preflight.v1"
FAKE_RESULT_SCHEMA = "scenesense.splitfusion.run4b5b.b_edge_offline_fake.v1"
EXECUTE_TOKEN = "SPLITFUSION_RUN4B5B_LIVE_VALIDATION_V1_EXECUTE"
ROLE = "CN_EDGE"
REMOTE_HOST = "L10319.idcc.lab"
LOCAL_HOST = "W10275.idcc.lab"
REQUIRED_AUTHORITIES = (
    "rl_agent.splitfusion_run4_split_host_l10319_v1.contract",
)
PROVEN_EDGE_MODULE = (
    "rl_agent.splitfusion_hybrid_sac_live_route_b_v2.phase6_edge_runtime_v2"
)
PROVEN_PROCESSOR = "Run4EdgeProcessorV2"
PROVEN_COMPUTE = "run4_compute_on_detached_runtime"
PROVEN_MAP_PUBLISHER = "Run4MapPublisherV2"
REQUIRED_LIVE_HOOK = "run_b_operational_edge_service_v1"
ACK_SEMANTICS = "TAIL_OUTPUT_ACK_BEFORE_MAP_AND_POSTRUN_EVIDENCE"
POSTRUN_SEMANTICS = "QUALITY_FROM_CREATE_ONLY_PREDICTION_AND_CARLA_GT_AFTER_RUN"
CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
DEADLINE_NS = 170_000_000


class BEdgeProcessError(RuntimeError):
    """The B edge request or post-compute contract was violated."""


class RequestError(BEdgeProcessError):
    """The lifecycle request is incomplete, foreign or unsafe."""


class MissingRuntimeAuthority(BEdgeProcessError):
    """A required proven runtime authority cannot be resolved."""


class MissingAckEndpoint(BEdgeProcessError):
    """The exact UE-side operational-ACK endpoint is absent or invalid."""


class RuntimeSeamRequired(BEdgeProcessError):
    """The qualified edge loop lacks the required additive callback seam."""


def _require(condition: bool, message: str,
             error: type[BEdgeProcessError] = BEdgeProcessError) -> None:
    if not condition:
        raise error(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RequestError("value is not canonically serializable") from exc


def _decode_request(encoded: str) -> dict[str, Any]:
    _require(type(encoded) is str and bool(encoded), "request is empty",
             RequestError)
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = base64.b64decode(encoded + padding, altchars=b"-_",
                                   validate=True)
        raw = json.loads(payload.decode("ascii"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise RequestError("request is not canonical urlsafe-base64 JSON") from exc
    expected = {
        "schema", "role", "run_id", "variant", "config_binding_sha256",
        "actor_boundary_sha256", "feature_schema_sha256",
        "transmitted_budget", "deadline_ns", "ack_semantics",
        "postrun_semantics", "clock_domain", "split_host", "output_root",
        "evidence_root", "actor_manifest_path", "remote_attempt_root",
        "required_authority_modules", "old_live_quality_runtime_permitted",
    }
    _require(type(raw) is dict and set(raw) == expected,
             "request fields are incomplete or foreign", RequestError)
    _require(_canonical(raw) == payload,
             "request JSON is not canonical", RequestError)
    return raw


def _sha256(value: Any, field: str) -> str:
    _require(type(value) is str and len(value) == 64
             and all(ch in "0123456789abcdef" for ch in value),
             f"{field} is not a lowercase SHA-256", RequestError)
    return value


def _validate_request(raw: Mapping[str, Any]) -> dict[str, Any]:
    _require(raw["schema"] == REQUEST_SCHEMA, "request schema drift",
             RequestError)
    _require(raw["role"] == ROLE, "request is not for the CN/edge role",
             RequestError)
    _require(type(raw["run_id"]) is str and bool(raw["run_id"]),
             "run_id is empty", RequestError)
    _require(raw["variant"] in {item.value for item in L.ActorVariant},
             "unknown B actor variant", RequestError)
    for field in ("config_binding_sha256", "actor_boundary_sha256",
                  "feature_schema_sha256"):
        _sha256(raw[field], field)
    _require(raw["transmitted_budget"] == 300,
             "transmitted budget must be exactly 300", RequestError)
    _require(raw["deadline_ns"] == DEADLINE_NS,
             "operational deadline drift", RequestError)
    _require(raw["ack_semantics"] == ACK_SEMANTICS,
             "operational ACK semantics drift", RequestError)
    _require(raw["postrun_semantics"] == POSTRUN_SEMANTICS,
             "post-run semantics drift", RequestError)
    _require(raw["clock_domain"] == CLOCK_DOMAIN,
             "clock domain drift", RequestError)
    _require(raw["output_root"] is None and raw["evidence_root"] is None
             and raw["actor_manifest_path"] is None,
             "UE-only paths are forbidden in the edge request", RequestError)
    _require(raw["old_live_quality_runtime_permitted"] is False,
             "legacy live-quality runtime was enabled", RequestError)
    _require(raw["required_authority_modules"] == list(REQUIRED_AUTHORITIES),
             "runtime authority list drift", RequestError)

    split = raw["split_host"]
    expected_split = {"carla_host", "ue_host", "cn_host", "edge_host",
                      "ext_dn_host", "ack_receiver_host",
                      "ack_receiver_port"}
    _require(type(split) is dict and set(split) == expected_split,
             "split-host fields are incomplete or foreign", RequestError)
    _require(split["carla_host"] == split["ue_host"] == LOCAL_HOST,
             "CARLA/UE host drift", RequestError)
    _require(split["cn_host"] == split["edge_host"]
             == split["ext_dn_host"] == REMOTE_HOST,
             "CN/edge/ext-DN host drift", RequestError)
    _require(split["ack_receiver_host"] == LOCAL_HOST,
             "ACK receiver must be UE-side", MissingAckEndpoint)
    _require(type(split["ack_receiver_port"]) is int
             and 1024 <= split["ack_receiver_port"] <= 65535,
             "ACK receiver port is absent or invalid", MissingAckEndpoint)
    attempt = raw["remote_attempt_root"]
    _require(type(attempt) is str and Path(attempt).is_absolute()
             and str(attempt) not in {"/", "/tmp"}
             and ".." not in Path(attempt).parts,
             "remote attempt root is unsafe", RequestError)
    return dict(raw)


def _find_module(module: str) -> Any:
    try:
        return importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError) as exc:
        raise MissingRuntimeAuthority(
            f"runtime authority cannot be resolved: {module}") from exc


def preflight_request(encoded: str, *,
                      find_module: Callable[[str], Any] = _find_module,
                      edge_symbols: Mapping[str, Any] | None = None,
                      ) -> dict[str, Any]:
    """Validate a CN/edge request without importing CUDA/runtime modules."""
    raw = _validate_request(_decode_request(encoded))
    for module in (*REQUIRED_AUTHORITIES, PROVEN_EDGE_MODULE):
        _require(find_module(module) is not None,
                 f"runtime authority is missing: {module}",
                 MissingRuntimeAuthority)
    required_symbols = (PROVEN_PROCESSOR, PROVEN_COMPUTE,
                        PROVEN_MAP_PUBLISHER)
    symbol_status = {name: True for name in required_symbols}
    hook_present = False
    if edge_symbols is not None:
        symbol_status = {name: name in edge_symbols for name in required_symbols}
        missing = [name for name, present in symbol_status.items() if not present]
        _require(not missing, "proven edge symbols missing: " + ", ".join(missing),
                 MissingRuntimeAuthority)
        hook_present = REQUIRED_LIVE_HOOK in edge_symbols
    return {
        "schema": PREFLIGHT_SCHEMA,
        "run_id": raw["run_id"],
        "variant": raw["variant"],
        "config_binding_sha256": raw["config_binding_sha256"],
        "ack_receiver_host": raw["split_host"]["ack_receiver_host"],
        "ack_receiver_port": raw["split_host"]["ack_receiver_port"],
        "authority_modules": [*REQUIRED_AUTHORITIES, PROVEN_EDGE_MODULE],
        "proven_symbols": symbol_status,
        "required_live_hook": REQUIRED_LIVE_HOOK,
        "live_hook_present": hook_present,
        "offline_postcompute_ready": True,
        "live_start_ready": hook_present,
    }


@dataclass(frozen=True, slots=True)
class UsableTailOutputV1:
    """Exact result handed across the additive post-compute seam."""

    identity: A.FrameActionIdentityV1
    reward_requested: bool
    object_records: tuple[Mapping[str, Any], ...]
    semantic_mask: Any
    map_document: Mapping[str, Any]
    tail_ready_monotonic_raw_ns: int

    def __post_init__(self) -> None:
        _require(type(self.identity) is A.FrameActionIdentityV1,
                 "usable output identity is foreign")
        _require(type(self.reward_requested) is bool,
                 "reward_requested must be bool")
        _require(type(self.object_records) is tuple,
                 "object_records must be an exact tuple")
        _require(isinstance(self.map_document, Mapping),
                 "map document is not a mapping")
        _require(type(self.tail_ready_monotonic_raw_ns) is int
                 and self.tail_ready_monotonic_raw_ns >= 0,
                 "tail-ready timestamp is invalid")


class EdgePostComputeSeamV1:
    """Bind usable tail output to the immediate-ACK branch dispatcher."""

    def __init__(self, *, dispatcher: L.TailOutputDispatchV1,
                 register_map_document: Callable[
                     [A.FrameActionIdentityV1, Mapping[str, Any]], None]) -> None:
        _require(type(dispatcher) is L.TailOutputDispatchV1,
                 "dispatcher must be exactly TailOutputDispatchV1")
        _require(callable(register_map_document),
                 "map-document registrar is not callable")
        self._dispatcher = dispatcher
        self._register_map_document = register_map_document

    def emit(self, output: UsableTailOutputV1) -> L.DispatchReceiptV1 | bool:
        _require(type(output) is UsableTailOutputV1,
                 "usable output has a foreign type")
        self._register_map_document(output.identity,
                                    dict(output.map_document))
        if not output.reward_requested:
            marker = _canonical({
                "identity_sha256": output.identity.exact_sha256(),
                "map_only": True,
            })
            return self._dispatcher.dispatch_map_only(
                output.identity, marker,
                tail_ready_monotonic_raw_ns=output.tail_ready_monotonic_raw_ns)
        prediction = L.encode_prediction_tail_output(
            identity=output.identity,
            objects=output.object_records,
            semantic_mask=output.semantic_mask,
        )
        return self._dispatcher.dispatch_decision(
            output.identity, prediction,
            tail_ready_monotonic_raw_ns=output.tail_ready_monotonic_raw_ns)


class MapDocumentRegistryV1:
    """One-use exact-identity bridge to the proven direct-map publisher."""

    def __init__(self, publish: Callable[[Mapping[str, Any]], Any]) -> None:
        _require(callable(publish), "map publisher is not callable")
        self._publish = publish
        self._documents: dict[str, tuple[A.FrameActionIdentityV1,
                                         Mapping[str, Any]]] = {}

    def register(self, identity: A.FrameActionIdentityV1,
                 document: Mapping[str, Any]) -> None:
        exact = identity.exact_sha256()
        prior = self._documents.get(exact)
        if prior is not None:
            _require(prior[0] == identity and dict(prior[1]) == dict(document),
                     "map document conflicts with registered identity")
            raise BEdgeProcessError("map document was already registered")
        self._documents[exact] = (identity, dict(document))

    def callback(self, work: L.ImmutableTailWorkV1
                 ) -> L.BranchCallbackResultV1:
        exact = work.identity.exact_sha256()
        row = self._documents.pop(exact, None)
        _require(row is not None and row[0] == work.identity,
                 "map work has no exact registered document")
        result = self._publish(row[1])
        digest = hashlib.sha256(_canonical({
            "identity_sha256": exact, "publisher_result": result,
        })).hexdigest()
        return L.BranchCallbackResultV1(
            status=B.BranchStatus.SUCCEEDED,
            detail_code="MAP_PUBLISHED", evidence_sha256=digest)


def _fake_identity() -> A.FrameActionIdentityV1:
    return A.FrameActionIdentityV1(
        run_id="offline_fake_run", cell_id="offline_fake_cell",
        stream_id="ego_rgb", session_uuid="00000000-0000-4000-8000-000000000001",
        controller_lineage_sha256=hashlib.sha256(b"controller").hexdigest(),
        decision_seq=0, ticket_seq=0, frame_id=1, tensor_seq=1,
        capture_timestamp_ns=1, mode_id=11, q_e4=3000, keep_count=2,
        anchor_action_id=67, profile_id="split_ae32_uint4_q3000",
        execution_bundle_sha256=hashlib.sha256(b"bundle").hexdigest())


def offline_fake(encoded: str, *, root: Path) -> dict[str, Any]:
    """Exercise ordering and create-only evidence without external systems."""
    preflight = preflight_request(encoded)
    _require(not Path(root).exists(), "offline fake root must be create-only")
    Path(root).mkdir(parents=True)
    events: list[str] = []
    ack_packets: list[bytes] = []

    def ack_sender(packet: bytes) -> None:
        A.decode_ack(packet)
        events.append("ACK")
        ack_packets.append(bytes(packet))

    registry = MapDocumentRegistryV1(
        lambda document: events.append("MAP") or {
            "frame_id": document["frame_id"]})
    store = B.PredictionEvidenceStoreV1.create(Path(root) / "prediction")
    prediction_callback = L.prediction_store_callback(
        store, clock=lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
    dispatcher = L.TailOutputDispatchV1(
        ack_sender=ack_sender, queue_depth=4,
        map_callback=registry.callback,
        prediction_callback=lambda work: (
            events.append("PREDICTION") or prediction_callback(work)),
    )
    dispatcher.start()
    identity = _fake_identity()
    seam = EdgePostComputeSeamV1(
        dispatcher=dispatcher, register_map_document=registry.register)
    receipt = seam.emit(UsableTailOutputV1(
        identity=identity, reward_requested=True,
        object_records=({"class_name": "person", "world_x": 1.0,
                         "world_y": 2.0},),
        semantic_mask=[[0, 1], [0, 0]],
        map_document={"frame_id": identity.frame_id},
        tail_ready_monotonic_raw_ns=1,
    ))
    dispatcher.stop()
    _require(type(receipt) is L.DispatchReceiptV1,
             "offline decision produced no dispatch receipt")
    _require(events and events[0] == "ACK",
             "operational ACK was not synchronous and first")
    ack = A.decode_ack(ack_packets[0])
    _require(ack.identity == identity,
             "offline ACK identity drift")
    record_paths = sorted((Path(root) / "prediction" / "records").glob("*.json"))
    _require(len(record_paths) == 1,
             "offline prediction evidence count drift")
    return {
        "schema": FAKE_RESULT_SCHEMA,
        "preflight": preflight,
        "identity_sha256": identity.exact_sha256(),
        "tail_output_sha256": ack.tail_output_sha256,
        "ack_packet_sha256": receipt.ack_packet_sha256,
        "events": events,
        "map_enqueued": receipt.map_enqueued,
        "prediction_enqueued": receipt.prediction_enqueued,
        "prediction_record_count": len(record_paths),
        "contains_gt_qperc_or_reward": False,
    }


def _edge_symbols_without_import() -> Mapping[str, Any]:
    """Read symbol names without importing the CUDA-capable edge module."""
    spec = _find_module(PROVEN_EDGE_MODULE)
    _require(spec is not None and spec.origin is not None,
             "proven edge source is unavailable", MissingRuntimeAuthority)
    source = Path(spec.origin).read_text(encoding="utf-8")
    names = {name: True for name in (
        PROVEN_PROCESSOR, PROVEN_COMPUTE, PROVEN_MAP_PUBLISHER,
        REQUIRED_LIVE_HOOK,
    ) if (f"class {name}" in source or f"def {name}" in source)}
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "offline-fake",
                                            "start", "stop"))
    parser.add_argument("--request-b64", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "preflight":
        result = preflight_request(
            args.request_b64, edge_symbols=_edge_symbols_without_import())
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "offline-fake":
        _require(args.output_root is not None,
                 "offline-fake requires --output-root")
        result = offline_fake(args.request_b64, root=args.output_root)
        print(json.dumps(result, sort_keys=True))
        return 0
    _require(args.execute == EXECUTE_TOKEN,
             "live command requires the exact execution token")
    preflight = preflight_request(
        args.request_b64, edge_symbols=_edge_symbols_without_import())
    if args.command == "stop":
        # Nothing is launched by this additive module until the hook exists.
        print(json.dumps({"stopped": True, "run_id": preflight["run_id"]},
                         sort_keys=True))
        return 0
    _require(preflight["live_hook_present"],
             "qualified edge runtime lacks additive post-compute hook "
             f"{REQUIRED_LIVE_HOOK}; copy/monkey-patch fallback is forbidden",
             RuntimeSeamRequired)
    raise RuntimeSeamRequired(
        "live hook was detected but binding it requires a reviewed authority")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BEdgeProcessError as exc:
        print(f"B_EDGE_PROCESS_REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
