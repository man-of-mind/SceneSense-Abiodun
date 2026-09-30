"""Strict, non-executing contract for the Run-4 two-host qualification.

This module is deliberately pure.  It neither opens a socket nor invokes
``ip``, ``iptables``, ``sysctl``, Docker, CUDA, OAI, or CARLA.  It describes
the exact topology, validates facts collected independently on L10319, and
produces argv vectors for a later reviewed executor.

The existing CN/container addresses are preserved.  W10275 reaches the
remote Docker subnet through L10319; there is no NAT and no blanket firewall
change.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SCHEMA = "scenesense.run4.split_host_l10319.v1"
RULE_TAG = "scenesense-run4-l10319-v1"
EDGE_IMAGE_TAG = "oai-perception-rx:latest"
EDGE_IMAGE_CONFIG_DIGEST = (
    "sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba"
)
EDGE_IMAGE_MANIFEST_DIGEST = (
    "sha256:ac1437601cb1b4a52c761d762ba533fcb431e46f364073516697a89155cd901c"
)
EDGE_IMAGE_CANONICAL_INSPECT_SHA256 = (
    "7f8a14571eb00426d98b7084175de1180d4a50300c01150a5c2390c55bba3d91"
)
REMOTE_IMAGE_ID = EDGE_IMAGE_MANIFEST_DIGEST
REMOTE_CONTAINER_IMAGE_ID = EDGE_IMAGE_MANIFEST_DIGEST
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SplitHostContractError(ValueError):
    """A prospective split-host value violates the registered contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitHostContractError(message)


@dataclass(frozen=True)
class ArtifactIdentity:
    name: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.name), "artifact name is empty")
        path = Path(self.relative_path)
        _require(not path.is_absolute() and ".." not in path.parts,
                 f"artifact path is not repository-relative: {self.relative_path}")
        _require(bool(SHA256_RE.fullmatch(self.sha256)),
                 f"artifact SHA-256 is invalid: {self.name}")


# The actor stays on W10275 and is deliberately not transferred to the edge host.
LOCAL_ACTOR_ARTIFACT = ArtifactIdentity(
    "actor_seed43_update10000",
    "rl_agent/experiments/splitfusion_hybrid_sac_live_route_b_v2/"
    "20260929_seed43_update10000_actor_export/actor_state_dict.pt",
    "d064013d011b67dcd2c7c23acc3c396afe6750be0d43ef0204f2fbecbb9b8e29",
)

# Exactly seven ignored binaries required on L10319 by the frozen codec/edge path.
ARTIFACTS = (
    ArtifactIdentity(
        "perception",
        "experiments/route_b_v3_1_splitfusion_fcos_r50_fpn_p2_p7_v1_"
        "numerical_recovery_v1/20260830_recovered_epoch10_gate_v1/"
        "checkpoints/epoch_026.pt",
        "da14d21edbd374c1c3abce02ca4674b9f4097becfba9759aba945cea160a297f",
    ),
    ArtifactIdentity(
        "ranker",
        "experiments/splitfusion_fcos_hybrid_q_v1/"
        "20260901_185725_phase5_ranker_training/checkpoints/ranker_epoch_04.pt",
        "07781c56a4c0f306f16d332f64627ce6b9458e154f40ab9fef89f89909b79cb5",
    ),
    ArtifactIdentity(
        "ae128",
        "experiments/splitfusion_fcos_ae_v1/"
        "20260902_220623_phase9c_ae128_training/checkpoints/ae128_epoch_08.pt",
        "0c2ba3a495684c0f8222492f554eb3de7c7a76181e0bd4b4a83529897db30f72",
    ),
    ArtifactIdentity(
        "ae64",
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase10_ae64_training/checkpoints/ae64_epoch_12.pt",
        "dd7c5124e27114584ab2083e59160a3ff2a2d040d0a37d22564ac98c838aa8e0",
    ),
    ArtifactIdentity(
        "ae32",
        "experiments/splitfusion_fcos_ae_v1/"
        "20260903_phase10_ae32_training/checkpoints/ae32_epoch_08.pt",
        "e2f867757e8db0620316c092264ac7eb53d12bb5ef66ed14475eb40693d1f271",
    ),
    ArtifactIdentity(
        "torchvision_fcos",
        "experiments/splitfusion_phase15_runtime_cache_v1/torch/hub/checkpoints/"
        "fcos_resnet50_fpn_coco-99b0c9b7.pth",
        "99b0c9b7cfb1527d782db86b91d207f00547c792fb4103fc612b651d0a07b9e7",
    ),
    ArtifactIdentity(
        "compose_fusion_checkpoint",
        "checkpoints/fusion_object_best.pt",
        "6a9d6a1041055e4a6461836d9a95a9a2d6c0b30b7e67984e87426456460478a5",
    ),
)


@dataclass(frozen=True)
class SplitHostTopology:
    local_name: str
    local_lan_ip: str
    remote_name: str
    remote_lan_ip: str
    cn_subnet: str
    amf_ip: str
    upf_ip: str
    ext_dn_ip: str
    edge_ip: str
    map_ip: str
    map_port: int
    local_prefix_length: int

    def validate(self) -> "SplitHostTopology":
        local = ipaddress.ip_address(self.local_lan_ip)
        remote = ipaddress.ip_address(self.remote_lan_ip)
        subnet = ipaddress.ip_network(self.cn_subnet, strict=True)
        _require(local.version == remote.version == subnet.version == 4,
                 "the split-host contract is IPv4-only")
        local_lan = ipaddress.ip_network(
            f"{local}/{self.local_prefix_length}", strict=False)
        _require(remote in local_lan and local != remote,
                 "local and remote hosts must be distinct peers on the local LAN")
        _require(local not in subnet and remote not in subnet,
                 "host LAN addresses must not overlap the routed CN subnet")
        roles = {
            "amf": ipaddress.ip_address(self.amf_ip),
            "upf": ipaddress.ip_address(self.upf_ip),
            "ext_dn": ipaddress.ip_address(self.ext_dn_ip),
            "edge": ipaddress.ip_address(self.edge_ip),
        }
        _require(len(set(roles.values())) == len(roles),
                 "CN and edge role addresses must be unique")
        for role, value in roles.items():
            _require(value in subnet, f"{role} is outside the routed CN subnet")
            _require(value not in (subnet.network_address, subnet.broadcast_address),
                     f"{role} uses a reserved subnet address")
        _require(self.map_ip == self.local_lan_ip,
                 "the direct map must stay on W10275's LAN address")
        _require(1 <= self.map_port <= 65535, "map port is invalid")
        _require((self.local_name, self.remote_name) == ("W10275", "L10319"),
                 "host identity drift")
        return self


def default_topology() -> SplitHostTopology:
    """Return the sole registered topology; no address is inferred at runtime."""
    return SplitHostTopology(
        local_name="W10275",
        local_lan_ip="10.21.16.222",
        remote_name="L10319",
        remote_lan_ip="10.21.16.162",
        cn_subnet="192.168.70.128/26",
        amf_ip="192.168.70.132",
        upf_ip="192.168.70.134",
        ext_dn_ip="192.168.70.135",
        edge_ip="192.168.70.140",
        map_ip="10.21.16.222",
        map_port=39320,
        local_prefix_length=24,
    ).validate()


@dataclass(frozen=True)
class Command:
    host: str
    purpose: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(self.host in ("W10275", "L10319"), "unknown command host")
        _require(bool(self.argv) and all(bool(x) for x in self.argv),
                 "empty command argument")


@dataclass(frozen=True)
class TaggedRule:
    description: str
    rule_args: tuple[str, ...]

    def check(self) -> Command:
        return Command("L10319", f"check {self.description}",
                       ("sudo", "iptables", "-C", "DOCKER-USER") + self.rule_args)

    def add(self) -> Command:
        return Command("L10319", f"add {self.description}",
                       ("sudo", "iptables", "-I", "DOCKER-USER", "1")
                       + self.rule_args)

    def remove(self) -> Command:
        return Command("L10319", f"remove {self.description}",
                       ("sudo", "iptables", "-D", "DOCKER-USER") + self.rule_args)


@dataclass(frozen=True)
class ReversibleNetworkPlan:
    probes: tuple[Command, ...]
    local_apply: Command
    local_rollback: Command
    forwarding_apply: Command
    forwarding_rollback: Command
    rules: tuple[TaggedRule, ...]


def _route_rollback(topology: SplitHostTopology,
                    previous_route_argv: Optional[Sequence[str]]) -> Command:
    if previous_route_argv is None:
        return Command(
            "W10275", "remove newly-added CN route",
            ("sudo", "ip", "route", "del", topology.cn_subnet,
             "via", topology.remote_lan_ip),
        )
    previous = tuple(str(value) for value in previous_route_argv)
    _require(previous[:2] == ("ip", "route") and len(previous) >= 4,
             "previous route must be an audited `ip route replace ...` argv")
    _require(previous[2] == "replace" and previous[3] == topology.cn_subnet,
             "previous route restores a different destination")
    return Command("W10275", "restore previous CN route", ("sudo",) + previous)


def network_plan(*, previous_ip_forward: int,
                 previous_route_argv: Optional[Sequence[str]]) -> ReversibleNetworkPlan:
    """Build an exact reversible plan from separately audited prior state.

    Execution semantics are intentionally not implemented here.  A later
    executor must run each rule's ``check`` first, add only if absent, record
    which rules it added, and remove only those rules during teardown.
    """
    topology = default_topology()
    _require(previous_ip_forward in (0, 1),
             "previous net.ipv4.ip_forward must be measured as 0 or 1")
    common = (
        "-m", "comment", "--comment", RULE_TAG,
        "-j", "ACCEPT",
    )
    rules = (
        TaggedRule(
            "W10275 to routed CN/edge subnet",
            ("-s", f"{topology.local_lan_ip}/32", "-d", topology.cn_subnet) + common,
        ),
        TaggedRule(
            "routed CN/edge subnet to W10275",
            ("-s", topology.cn_subnet, "-d", f"{topology.local_lan_ip}/32") + common,
        ),
    )
    return ReversibleNetworkPlan(
        probes=(
            Command("W10275", "record prior CN route",
                    ("ip", "-j", "route", "show", topology.cn_subnet)),
            Command("L10319", "record prior forwarding state",
                    ("sysctl", "-n", "net.ipv4.ip_forward")),
            Command("L10319", "record Docker forward chain",
                    ("sudo", "iptables", "-S", "DOCKER-USER")),
        ),
        local_apply=Command(
            "W10275", "route CN subnet through L10319",
            ("sudo", "ip", "route", "replace", topology.cn_subnet,
             "via", topology.remote_lan_ip, "src", topology.local_lan_ip),
        ),
        local_rollback=_route_rollback(topology, previous_route_argv),
        forwarding_apply=Command(
            "L10319", "enable IPv4 forwarding",
            ("sudo", "sysctl", "-w", "net.ipv4.ip_forward=1"),
        ),
        forwarding_rollback=Command(
            "L10319", "restore IPv4 forwarding",
            ("sudo", "sysctl", "-w",
             f"net.ipv4.ip_forward={previous_ip_forward}"),
        ),
        rules=rules,
    )


_AMF_RE = re.compile(
    r'(?m)^(?P<prefix>\s*amf_ip_address\s*=\s*\(\{\s*ipv4\s*=\s*)'
    r'"[^"]+"(?P<suffix>\s*;\s*\}\)\s*;\s*)$'
)
_N2_RE = re.compile(
    r'(?m)^(?P<prefix>\s*GNB_IPV4_ADDRESS_FOR_NG_AMF\s*=\s*)'
    r'"[^"]+"(?P<suffix>\s*;\s*)$'
)
_N3_RE = re.compile(
    r'(?m)^(?P<prefix>\s*GNB_IPV4_ADDRESS_FOR_NGU\s*=\s*)'
    r'"[^"]+"(?P<suffix>\s*;\s*)$'
)


def rewrite_runtime_gnb_config(source: str) -> str:
    """Return a runtime-only gNB config for remote AMF/UPF routing."""
    topology = default_topology()
    replacements = (
        (_AMF_RE, topology.amf_ip, "AMF"),
        (_N2_RE, f"{topology.local_lan_ip}/{topology.local_prefix_length}", "N2"),
        (_N3_RE, f"{topology.local_lan_ip}/{topology.local_prefix_length}", "N3"),
    )
    result = source
    for pattern, value, label in replacements:
        result, count = pattern.subn(
            lambda match: f'{match.group("prefix")}"{value}"{match.group("suffix")}',
            result,
        )
        _require(count == 1, f"expected exactly one {label} address, found {count}")
    assert_runtime_gnb_config(result)
    return result


def assert_runtime_gnb_config(text: str) -> None:
    topology = default_topology()
    expected = (
        (_AMF_RE, topology.amf_ip, "AMF"),
        (_N2_RE, f"{topology.local_lan_ip}/{topology.local_prefix_length}", "N2"),
        (_N3_RE, f"{topology.local_lan_ip}/{topology.local_prefix_length}", "N3"),
    )
    for pattern, value, label in expected:
        matches = tuple(pattern.finditer(text))
        _require(len(matches) == 1, f"runtime {label} address is missing or duplicated")
        quoted = re.findall(r'"([^"]+)"', matches[0].group(0))
        _require(quoted == [value], f"runtime {label} address drift")


@dataclass(frozen=True)
class RemoteGpuIdentity:
    model: str
    uuid: str
    memory_total_mib: int
    driver_version: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RemoteGpuIdentity":
        required = {"model", "uuid", "memory_total_mib", "driver_version"}
        _require(set(raw) == required, "remote GPU fact fields are incomplete or foreign")
        result = cls(
            model=str(raw["model"]).strip(), uuid=str(raw["uuid"]).strip(),
            memory_total_mib=int(raw["memory_total_mib"]),
            driver_version=str(raw["driver_version"]).strip(),
        )
        _require(bool(result.model) and bool(result.uuid) and bool(result.driver_version),
                 "remote GPU facts must be measured, not blank")
        _require(result.memory_total_mib > 0, "remote GPU VRAM must be measured")
        return result


@dataclass(frozen=True)
class RemoteRuntimeBinding:
    schema: str
    hostname: str
    host_ipv4: str
    gpu: RemoteGpuIdentity
    image_tag: str
    image_manifest_digest: str
    image_config_digest: str
    remote_image_id: str
    remote_container_image_id: str
    canonical_inspect_fields_sha256: str
    artifacts: tuple[ArtifactIdentity, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RemoteRuntimeBinding":
        required = {
            "schema", "hostname", "host_ipv4", "gpu", "image_tag",
            "image_manifest_digest", "image_config_digest", "remote_image_id",
            "remote_container_image_id", "canonical_inspect_fields_sha256",
            "artifacts",
        }
        _require(set(raw) == required, "remote runtime binding fields are incomplete or foreign")
        _require(isinstance(raw["gpu"], Mapping), "gpu facts must be an object")
        _require(isinstance(raw["artifacts"], list), "artifacts must be a list")
        artifacts = tuple(
            ArtifactIdentity(
                name=str(item["name"]), relative_path=str(item["relative_path"]),
                sha256=str(item["sha256"]),
            ) if isinstance(item, Mapping) and set(item) == {
                "name", "relative_path", "sha256"
            } else (_ for _ in ()).throw(
                SplitHostContractError("artifact fields are incomplete or foreign")
            )
            for item in raw["artifacts"]
        )
        result = cls(
            schema=str(raw["schema"]), hostname=str(raw["hostname"]),
            host_ipv4=str(raw["host_ipv4"]),
            gpu=RemoteGpuIdentity.from_mapping(raw["gpu"]),
            image_tag=str(raw["image_tag"]),
            image_manifest_digest=str(raw["image_manifest_digest"]),
            image_config_digest=str(raw["image_config_digest"]),
            remote_image_id=str(raw["remote_image_id"]),
            remote_container_image_id=str(raw["remote_container_image_id"]),
            canonical_inspect_fields_sha256=str(raw["canonical_inspect_fields_sha256"]),
            artifacts=artifacts,
        )
        result.validate()
        return result

    def validate(self) -> "RemoteRuntimeBinding":
        topology = default_topology()
        _require(self.schema == "scenesense.run4.remote_runtime_binding.v1",
                 "remote runtime binding schema drift")
        _require((self.hostname, self.host_ipv4)
                 == (topology.remote_name, topology.remote_lan_ip),
                 "remote host identity drift")
        _require(self.image_tag == EDGE_IMAGE_TAG, "edge image tag drift")
        _require(self.image_manifest_digest == EDGE_IMAGE_MANIFEST_DIGEST,
                 "edge OCI manifest digest drift")
        _require(self.image_config_digest == EDGE_IMAGE_CONFIG_DIGEST,
                 "edge OCI config digest drift")
        _require(self.remote_image_id == REMOTE_IMAGE_ID,
                 "remote image inspect ID drift")
        _require(self.remote_container_image_id == REMOTE_CONTAINER_IMAGE_ID,
                 "remote container image ID drift")
        _require(self.canonical_inspect_fields_sha256
                 == EDGE_IMAGE_CANONICAL_INSPECT_SHA256,
                 "portable canonical image metadata drift")
        _require(self.artifacts == ARTIFACTS,
                 "remote artifact set/order/path/hash drift")
        return self


def verify_artifact_files(root: Path) -> None:
    """Hash all seven transferred files; missing files never become defaults."""
    root = Path(root)
    for artifact in ARTIFACTS:
        path = root / artifact.relative_path
        _require(path.is_file(), f"remote artifact is missing: {artifact.name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(digest == artifact.sha256,
                 f"remote artifact hash drift: {artifact.name}")


def readiness_report(*, remote_binding: Optional[RemoteRuntimeBinding],
                     remote_connectivity_qualified: bool,
                     gt_lifecycle_seam_implemented: bool) -> Mapping[str, Any]:
    """Report readiness without converting missing evidence into success."""
    blockers = []
    if remote_binding is None:
        blockers.append("REMOTE_HARDWARE_IMAGE_AND_ARTIFACT_FACTS_NOT_BOUND")
    else:
        remote_binding.validate()
    if not remote_connectivity_qualified:
        blockers.append("SPLIT_HOST_ROUTE_AND_CONTAINER_CONNECTIVITY_NOT_QUALIFIED")
    if not gt_lifecycle_seam_implemented:
        blockers.append("PHASE6_GT_AND_SERVICE_LIFECYCLE_SEAM_NOT_IMPLEMENTED")
    return {
        "schema": "scenesense.run4.split_host_readiness.v1",
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "live_run_authorized": False,
        "note": "This scaffold never authorizes or launches a live run.",
    }
