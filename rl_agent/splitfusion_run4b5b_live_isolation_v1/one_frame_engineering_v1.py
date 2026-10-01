"""One-frame production engineering gate for the isolated B live path.

This is deliberately *not* a reduced qualification configuration.  It has a
different schema, an exact one-frame/one-decision budget and an explicit
``policy_performance_claim = false`` boundary.  A successful attempt proves
only that the selected final actor, UE/front/SFD path, remote GT-free tail,
compact operational ACK and evidence paths compose once across W10275 and
L10319.  The frozen 300-frame contract remains unchanged.

The host-specific factory is loaded lazily.  Importing this module performs no
I/O, imports no torch/CUDA module and launches no process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Mapping, Protocol, Sequence

from . import final_actor_gate_v2 as F
from . import production_lifecycle_adapter_v1 as P


CONFIG_SCHEMA = "scenesense.splitfusion.run4b5b.one_frame_engineering.v1"
SEAL_SCHEMA = "scenesense.splitfusion.run4b5b.one_frame_engineering_seal.v1"
RESULT_SCHEMA = "scenesense.splitfusion.run4b5b.one_frame_engineering_result.v1"
PURPOSE = "ENGINEERING_HANDSHAKE_ONLY__NEVER_300_FRAME_QUALIFICATION"
EXECUTE_TOKEN = "SPLITFUSION_RUN4B5B_ONE_FRAME_ENGINEERING_V1_EXECUTE"
FACTORY_MODULE = (
    "rl_agent.splitfusion_run4b5b_live_isolation_v1."
    "b_one_frame_production_factory_v1"
)
FACTORY_SYMBOL = "build_one_frame_lifecycle_v1"

TRANSMITTED_BUDGET = 1
POLICY_DECISION_BUDGET = 1
DEADLINE_NS = 170_000_000
CLOCK_DOMAIN = "CLOCK_MONOTONIC_RAW"
ACK_SEMANTICS = "TAIL_OUTPUT_READY__GT_FREE__SYNCHRONOUS_BEFORE_MAP"
POSTRUN_SEMANTICS = "RAW_CARLA_GT_SPOOLED_LIVE__QPERC_MATERIALIZED_POSTRUN"

LOCAL_HOST = P.LOCAL_HOST
REMOTE_HOST = P.REMOTE_HOST
REMOTE_SSH = P.REMOTE_SSH
LOCAL_LAN_IP = "10.21.16.222"
REMOTE_LAN_IP = "10.21.16.162"
CN_SUBNET = "192.168.70.128/26"
EDGE_IP = "192.168.70.140"
EXT_DN_IP = "192.168.70.135"
UE_TUNNEL_IP = "10.0.0.2"
UE_TUNNEL_INTERFACE = "oaitun_ue1"
UE_POLICY_TABLE = 9999
EDGE_ROUTE = "192.168.70.140/32"
EDGE_FEATURE_PORT = 51002
ACK_PORT = 51014
DIRECT_MAP_PORT = 39320
CARLA_RPC_PORT = 2000

_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}")
_CELL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_SHA = re.compile(r"[0-9a-f]{64}")


class OneFrameEngineeringError(RuntimeError):
    """The engineering config or lifecycle violated its narrow contract."""


class ProductionFactoryUnavailable(OneFrameEngineeringError):
    """The production UE/edge composition factory has not landed."""


def _require(value: bool, message: str) -> None:
    if not value:
        raise OneFrameEngineeringError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise OneFrameEngineeringError("value is not canonical JSON") from exc


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            value.update(block)
    return value.hexdigest()


def _absolute(value: Any, label: str) -> Path:
    _require(type(value) is str and bool(value), f"{label} is empty")
    pure = PurePath(value)
    _require(pure.is_absolute() and ".." not in pure.parts,
             f"{label} must be absolute without parent traversal")
    path = Path(value).resolve(strict=False)
    _require(str(path) not in {"/", "/home", "/tmp"},
             f"{label} is too broad")
    return path


def _sha(value: Any, label: str) -> str:
    _require(type(value) is str and bool(_SHA.fullmatch(value)),
             f"{label} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class NetworkBindingV1:
    local_host: str
    remote_host: str
    remote_ssh: str
    local_lan_ip: str
    remote_lan_ip: str
    cn_subnet: str
    edge_ip: str
    ext_dn_ip: str
    ue_tunnel_ip: str
    ue_tunnel_interface: str
    ue_policy_table: int
    edge_route: str
    edge_feature_port: int
    ack_port: int
    direct_map_host: str
    direct_map_port: int
    carla_rpc_host: str
    carla_rpc_port: int

    def __post_init__(self) -> None:
        exact = {
            "local_host": LOCAL_HOST, "remote_host": REMOTE_HOST,
            "remote_ssh": REMOTE_SSH, "local_lan_ip": LOCAL_LAN_IP,
            "remote_lan_ip": REMOTE_LAN_IP, "cn_subnet": CN_SUBNET,
            "edge_ip": EDGE_IP, "ext_dn_ip": EXT_DN_IP,
            "ue_tunnel_ip": UE_TUNNEL_IP,
            "ue_tunnel_interface": UE_TUNNEL_INTERFACE,
            "ue_policy_table": UE_POLICY_TABLE, "edge_route": EDGE_ROUTE,
            "edge_feature_port": EDGE_FEATURE_PORT, "ack_port": ACK_PORT,
            "direct_map_host": LOCAL_LAN_IP,
            "direct_map_port": DIRECT_MAP_PORT,
            "carla_rpc_host": "127.0.0.1",
            "carla_rpc_port": CARLA_RPC_PORT,
        }
        for field, expected in exact.items():
            _require(getattr(self, field) == expected,
                     f"network binding drift: {field}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NetworkBindingV1":
        fields = set(cls.__dataclass_fields__)
        _require(type(raw) is dict and set(raw) == fields,
                 "network binding fields are incomplete or foreign")
        return cls(**{name: raw[name] for name in fields})

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class OneFrameConfigV1:
    run_id: str
    cell_id: str
    variant: str
    actor_manifest_path: Path
    actor_manifest_sha256: str
    actor_weights_path: Path
    actor_weights_sha256: str
    actor_evidence_root: Path
    local_repository: Path
    remote_repository: Path
    local_attempt_root: Path
    remote_attempt_root: Path
    edge_campaign_config: Path
    route_config: Path
    network: NetworkBindingV1
    transmitted_budget: int
    policy_decision_budget: int
    deadline_ns: int
    clock_domain: str
    ack_semantics: str
    postrun_semantics: str
    purpose: str
    policy_performance_claim: bool
    factory_module: str

    def __post_init__(self) -> None:
        _require(type(self.run_id) is str and bool(_RUN.fullmatch(self.run_id)),
                 "run_id is unsafe")
        _require(type(self.cell_id) is str and bool(_CELL.fullmatch(self.cell_id)),
                 "cell_id is unsafe")
        _require(self.variant in {F.RUN4B_VARIANT, F.RUN5B_VARIANT},
                 "variant is not an exact final B actor")
        for field in (
            "actor_manifest_path", "actor_weights_path", "actor_evidence_root",
            "local_repository", "remote_repository", "local_attempt_root",
            "remote_attempt_root", "edge_campaign_config", "route_config",
        ):
            object.__setattr__(self, field,
                               _absolute(str(getattr(self, field)), field))
        _sha(self.actor_manifest_sha256, "actor_manifest_sha256")
        _sha(self.actor_weights_sha256, "actor_weights_sha256")
        _require(self.actor_manifest_path.name.endswith("MANIFEST_V2.json"),
                 "actor manifest filename is not the final V2 authority")
        _require(self.actor_weights_path.name == "actor_state_dict.pt",
                 "actor weights filename drift")
        _require(self.local_attempt_root != self.remote_attempt_root,
                 "local and remote attempt roots overlap")
        _require(self.local_repository != self.local_attempt_root
                 and self.remote_repository != self.remote_attempt_root,
                 "attempt root equals a repository")
        _require(type(self.network) is NetworkBindingV1,
                 "network binding has a foreign type")
        _require(self.transmitted_budget == TRANSMITTED_BUDGET,
                 "engineering budget must be exactly one transmitted frame")
        _require(self.policy_decision_budget == POLICY_DECISION_BUDGET,
                 "engineering budget must be exactly one policy decision")
        _require(self.deadline_ns == DEADLINE_NS,
                 "operational ACK deadline drift")
        _require(self.clock_domain == CLOCK_DOMAIN,
                 "clock domain drift")
        _require(self.ack_semantics == ACK_SEMANTICS,
                 "engineering ACK semantics drift")
        _require(self.postrun_semantics == POSTRUN_SEMANTICS,
                 "engineering postrun semantics drift")
        _require(self.purpose == PURPOSE,
                 "engineering purpose/qualification separation drift")
        _require(self.policy_performance_claim is False,
                 "one-frame engineering may not claim policy performance")
        _require(self.factory_module == FACTORY_MODULE,
                 "production factory module drift")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OneFrameConfigV1":
        fields = {
            "schema", "run_id", "cell_id", "variant",
            "actor_manifest_path", "actor_manifest_sha256",
            "actor_weights_path", "actor_weights_sha256",
            "actor_evidence_root", "local_repository", "remote_repository",
            "local_attempt_root", "remote_attempt_root",
            "edge_campaign_config", "route_config", "network",
            "transmitted_budget", "policy_decision_budget", "deadline_ns",
            "clock_domain", "ack_semantics", "postrun_semantics", "purpose",
            "policy_performance_claim", "factory_module",
        }
        _require(type(raw) is dict and set(raw) == fields,
                 "one-frame config fields are incomplete or foreign")
        _require(raw["schema"] == CONFIG_SCHEMA, "one-frame schema drift")
        lowered = _canonical(raw).decode("ascii").lower()
        for forbidden in ("quality_ack", "live_qperc", "gt_feedback",
                          "reward_ticket", "map_install_ack"):
            _require(forbidden not in lowered,
                     f"retired live-quality path present: {forbidden}")
        values = dict(raw)
        values.pop("schema")
        values["network"] = NetworkBindingV1.from_mapping(values["network"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"schema": CONFIG_SCHEMA}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, NetworkBindingV1):
                value = value.as_dict()
            result[name] = value
        return result

    def binding_sha256(self) -> str:
        return hashlib.sha256(_canonical(self.as_dict())).hexdigest()

    def lifecycle_settings(self) -> P.ProductionLifecycleSettingsV1:
        return P.ProductionLifecycleSettingsV1(
            local_repository=self.local_repository,
            remote_repository=self.remote_repository,
            remote_attempt_base=self.remote_attempt_root.parent,
        )


def seal(config: OneFrameConfigV1) -> dict[str, Any]:
    _require(type(config) is OneFrameConfigV1, "config type is foreign")
    return {"schema": SEAL_SCHEMA, "binding_sha256": config.binding_sha256(),
            "config": config.as_dict()}


def load_config(path: Path) -> OneFrameConfigV1:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             "sealed one-frame config is absent or a symlink")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OneFrameEngineeringError("sealed config is unreadable") from exc
    _require(type(raw) is dict
             and set(raw) == {"schema", "binding_sha256", "config"},
             "sealed config envelope is incomplete or foreign")
    _require(raw["schema"] == SEAL_SCHEMA, "sealed config schema drift")
    config = OneFrameConfigV1.from_mapping(raw["config"])
    _require(raw["binding_sha256"] == config.binding_sha256(),
             "sealed config binding differs")
    return config


def _load_selected_actor(config: OneFrameConfigV1) -> F.LoadedFinalActorV2:
    _require(config.actor_manifest_path.is_file()
             and not config.actor_manifest_path.is_symlink(),
             "actor manifest is absent or a symlink")
    _require(_digest(config.actor_manifest_path) == config.actor_manifest_sha256,
             "actor manifest byte digest differs")
    _require(config.actor_weights_path.is_file()
             and not config.actor_weights_path.is_symlink(),
             "actor weights are absent or a symlink")
    _require(_digest(config.actor_weights_path) == config.actor_weights_sha256,
             "actor weights byte digest differs")
    actor = F.verify_and_load_final_actor(
        config.actor_manifest_path, config.actor_weights_path,
        evidence_root=config.actor_evidence_root)
    _require(actor.identity.variant == config.variant,
             "loaded final actor variant differs")
    return actor


def require_create_only_targets(config: OneFrameConfigV1) -> None:
    for path, label in ((config.local_attempt_root, "local attempt root"),
                        (config.remote_attempt_root, "remote attempt root")):
        _require(not path.exists(), f"{label} is not create-only")


@dataclass(frozen=True, slots=True)
class OneFrameExecutionV1:
    run_id: str
    variant: str
    transmitted_frames: int
    policy_decisions: int
    operational_successes: int
    operational_timeouts: int
    observed_latency_ns: int
    ack_before_map_offer: bool
    prediction_evidence_written: bool
    live_qperc_computed: bool
    exact_identity_sha256: str
    result_sha256: str

    def validate(self, config: OneFrameConfigV1) -> None:
        _require(self.run_id == config.run_id and self.variant == config.variant,
                 "execution run/variant drift")
        _require(self.transmitted_frames == 1 and self.policy_decisions == 1,
                 "one-frame execution count drift")
        _require(self.operational_successes == 1
                 and self.operational_timeouts == 0,
                 "one-frame handshake did not receive a timely ACK")
        _require(type(self.observed_latency_ns) is int
                 and 0 <= self.observed_latency_ns <= config.deadline_ns,
                 "one-frame ACK exceeded the inclusive 170-ms deadline")
        _require(self.ack_before_map_offer is True,
                 "operational ACK was not synchronous before map offer")
        _require(self.prediction_evidence_written is True,
                 "tail prediction evidence was not written")
        _require(self.live_qperc_computed is False,
                 "live Qperc/GT scoring was executed")
        for value, label in ((self.exact_identity_sha256, "identity digest"),
                             (self.result_sha256, "result digest")):
            _sha(value, label)


class OneFrameLifecycleV1(Protocol):
    def preflight(self, config: OneFrameConfigV1,
                  actor: F.LoadedFinalActorV2) -> None: ...
    def start(self, config: OneFrameConfigV1,
              actor: F.LoadedFinalActorV2) -> None: ...
    def execute(self, config: OneFrameConfigV1,
                actor: F.LoadedFinalActorV2) -> OneFrameExecutionV1: ...
    def stop(self, config: OneFrameConfigV1) -> None: ...


def load_production_lifecycle(config: OneFrameConfigV1) -> OneFrameLifecycleV1:
    try:
        module = importlib.import_module(config.factory_module)
    except (ImportError, AttributeError) as exc:
        raise ProductionFactoryUnavailable(
            f"production factory unavailable: {config.factory_module}") from exc
    factory = getattr(module, FACTORY_SYMBOL, None)
    _require(callable(factory),
             f"production factory lacks {FACTORY_SYMBOL}")
    lifecycle = factory(config, config.lifecycle_settings())
    for method in ("preflight", "start", "execute", "stop"):
        _require(callable(getattr(lifecycle, method, None)),
                 f"production lifecycle lacks {method}")
    return lifecycle


def preflight(config: OneFrameConfigV1,
              lifecycle: OneFrameLifecycleV1) -> F.LoadedFinalActorV2:
    require_create_only_targets(config)
    actor = _load_selected_actor(config)
    lifecycle.preflight(config, actor)
    return actor


def run(config: OneFrameConfigV1,
        lifecycle: OneFrameLifecycleV1) -> OneFrameExecutionV1:
    actor = preflight(config, lifecycle)
    started = False
    primary: BaseException | None = None
    try:
        lifecycle.start(config, actor)
        started = True
        execution = lifecycle.execute(config, actor)
        _require(type(execution) is OneFrameExecutionV1,
                 "lifecycle returned a foreign execution record")
        execution.validate(config)
        return execution
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if started:
            try:
                lifecycle.stop(config)
            except BaseException:
                if primary is None:
                    raise


class _OfflineFakeLifecycle:
    """Mechanics fixture only; never a performance or live readiness claim."""

    def __init__(self) -> None:
        self.started = False

    def preflight(self, config: OneFrameConfigV1,
                  actor: F.LoadedFinalActorV2) -> None:
        return None

    def start(self, config: OneFrameConfigV1,
              actor: F.LoadedFinalActorV2) -> None:
        config.local_attempt_root.mkdir(parents=True)
        self.started = True

    def execute(self, config: OneFrameConfigV1,
                actor: F.LoadedFinalActorV2) -> OneFrameExecutionV1:
        _require(self.started, "offline lifecycle was not started")
        identity = hashlib.sha256(_canonical({
            "run": config.run_id, "variant": config.variant,
            "actor": actor.identity.actor_state_dict_sha256,
        })).hexdigest()
        result = hashlib.sha256(_canonical({
            "identity": identity, "latency_ns": 100_000_000,
        })).hexdigest()
        return OneFrameExecutionV1(
            run_id=config.run_id, variant=config.variant,
            transmitted_frames=1, policy_decisions=1,
            operational_successes=1, operational_timeouts=0,
            observed_latency_ns=100_000_000, ack_before_map_offer=True,
            prediction_evidence_written=True, live_qperc_computed=False,
            exact_identity_sha256=identity, result_sha256=result)

    def stop(self, config: OneFrameConfigV1) -> None:
        self.started = False


def _result(value: OneFrameExecutionV1, config: OneFrameConfigV1,
            status: str) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA, "status": status,
        "config_binding_sha256": config.binding_sha256(),
        "purpose": config.purpose, "policy_performance_claim": False,
        **{name: getattr(value, name)
           for name in value.__dataclass_fields__},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--config-check", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--offline-fake", action="store_true")
    modes.add_argument("--execute")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    if args.config_check:
        print(json.dumps({
            "schema": RESULT_SCHEMA, "status": "CONFIG_VALID",
            "config_binding_sha256": config.binding_sha256(),
            "purpose": PURPOSE, "services_launched": False,
        }, sort_keys=True))
        return 0
    lifecycle = (_OfflineFakeLifecycle() if args.offline_fake
                 else load_production_lifecycle(config))
    if args.preflight:
        actor = preflight(config, lifecycle)
        print(json.dumps({
            "schema": RESULT_SCHEMA, "status": "PREFLIGHT_PASS",
            "config_binding_sha256": config.binding_sha256(),
            "variant": config.variant,
            "actor_state_dict_sha256": actor.identity.actor_state_dict_sha256,
            "purpose": PURPOSE, "services_launched": False,
        }, sort_keys=True))
        return 0
    if args.execute is not None:
        _require(args.execute == EXECUTE_TOKEN,
                 "production execution token differs")
    value = run(config, lifecycle)
    print(json.dumps(_result(
        value, config,
        "OFFLINE_FAKE_PASS" if args.offline_fake else "ONE_FRAME_PASS"),
        sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - production CLI
    try:
        raise SystemExit(main())
    except OneFrameEngineeringError as exc:
        print(f"ONE_FRAME_ENGINEERING_REFUSED: {exc}")
        raise SystemExit(2)

