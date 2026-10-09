"""Immutable startup registry for the locked SplitFusion action catalog."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_BINDING = Path(__file__).resolve().with_name("runtime_binding.json")
RUNTIME_BINDING_SCHEMA = "scenesense.splitfusion_live_dispatch_runtime_binding.v1"
RUNTIME_BINDING_STATUS = "IMPLEMENTATION_READY_NOT_GPU_OR_LIVE_QUALIFIED"
CATALOG_SCHEMA = "splitfusion_72_action_catalog_v1"
CAMPAIGN_BINDING_SCHEMA = "scenesense.splitfusion_72x4_campaign_binding.v1"
CATALOG_TERMINAL = "SPLITFUSION_72_PROFILE_ACTION_CATALOG_LOCKED"
CAMPAIGN_TERMINAL = "SPLITFUSION_72X4_UE_CAMPAIGN_CONTRACT_BOUND"
FAMILIES = ("noAE", "AE128", "AE64", "AE32")
QUANTIZERS = ("UINT8", "UINT6", "UINT4")
Q_E4 = (0, 3000, 5000, 7000, 9000, 9800)
SPATIAL_CELLS = 21504


class DispatchContractError(RuntimeError):
    """A startup binding or per-frame dispatch identity failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_path(relative: str) -> Path:
    path = (REPOSITORY_ROOT / relative).resolve(strict=True)
    try:
        path.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise DispatchContractError(f"bound path escapes repository: {relative}") from exc
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _decode_json(payload: bytes, path: Path) -> dict[str, Any]:
    value = json.loads(payload.decode("utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


@dataclass(frozen=True)
class CheckpointBinding:
    path: str
    sha256: str


@dataclass(frozen=True)
class WireIdentity:
    magic_ascii: str
    version: int
    codec_id: int
    layout: str


@dataclass(frozen=True)
class ActionProfile:
    action_id: int
    profile_id: str
    execution_mode: str
    family: str
    family_id: int
    quantizer: str
    bit_width: int
    q: float
    q_e4: int
    keep_count: int
    drop_count: int
    transported_channels: int
    latent_width: int | None
    routing_tag: int
    decoder_identity: str
    zstd_level: int
    wire: WireIdentity
    perception_checkpoint: CheckpointBinding
    ae_checkpoint: CheckpointBinding | None
    ranker_checkpoint: CheckpointBinding | None
    ranker_bypassed: bool
    segmentation_installable: bool
    segmentation_behavior: str


@dataclass(frozen=True)
class RegistryStartupAudit:
    runtime_binding_reads: int
    catalog_reads: int
    catalog_terminal_reads: int
    campaign_binding_reads: int
    campaign_terminal_reads: int
    artifact_hash_operations: int
    verified_runtime_artifacts: bool
    action_count: int


def _checkpoint(value: Any, *, name: str, optional: bool = False) -> CheckpointBinding | None:
    if value is None and optional:
        return None
    require(isinstance(value, dict), f"{name} checkpoint binding is missing")
    path = str(value.get("path", ""))
    digest = str(value.get("sha256", ""))
    require(path != "" and len(digest) == 64, f"{name} checkpoint binding is invalid")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise DispatchContractError(f"{name} checkpoint SHA-256 is not hexadecimal") from exc
    return CheckpointBinding(path=path, sha256=digest)


def _action_profile(row: Mapping[str, Any]) -> ActionProfile:
    capabilities = row.get("capabilities", {})
    ranker = row.get("ranker", {})
    wire = row.get("wire", {})
    require(isinstance(capabilities, dict), "catalog action capabilities are invalid")
    require(isinstance(ranker, dict), "catalog action ranker binding is invalid")
    require(isinstance(wire, dict), "catalog action wire binding is invalid")
    profile = ActionProfile(
        action_id=int(row["action_id"]),
        profile_id=str(row["profile_id"]),
        execution_mode=str(row["execution_mode"]),
        family=str(row["family"]),
        family_id=int(row["family_id"]),
        quantizer=str(row["quantizer"]),
        bit_width=int(row["bit_width"]),
        q=float(row["q"]),
        q_e4=int(row["q_e4"]),
        keep_count=int(row["keep_count"]),
        drop_count=int(row["drop_count"]),
        transported_channels=int(row["transported_channels"]),
        latent_width=(None if row.get("latent_width") is None else int(row["latent_width"])),
        routing_tag=int(row["routing_tag"]),
        decoder_identity=str(row["decoder_identity"]),
        zstd_level=int(row["zstd_level"]),
        wire=WireIdentity(
            magic_ascii=str(wire["magic_ascii"]),
            version=int(wire["version"]),
            codec_id=int(wire["codec_id"]),
            layout=str(wire["layout"]),
        ),
        perception_checkpoint=_checkpoint(row.get("perception_checkpoint"), name="perception"),
        ae_checkpoint=_checkpoint(row.get("ae_checkpoint"), name="AE", optional=True),
        ranker_checkpoint=_checkpoint(ranker.get("checkpoint"), name="ranker", optional=True),
        ranker_bypassed=bool(ranker.get("bypassed")),
        segmentation_installable=bool(capabilities.get("segmentation_installable")),
        segmentation_behavior=str(capabilities.get("segmentation_behavior", "")),
    )
    require(capabilities.get("transport_valid") is True, f"action {profile.action_id} is transport-invalid")
    require(capabilities.get("agent_action_enabled") is True, f"action {profile.action_id} is disabled")
    return profile


def _validate_profile(profile: ActionProfile) -> None:
    require(profile.execution_mode == "SPLIT", f"action {profile.action_id} is not SPLIT")
    require(profile.family in FAMILIES, f"action {profile.action_id} family is unregistered")
    require(profile.quantizer in QUANTIZERS, f"action {profile.action_id} quantizer is unregistered")
    require(profile.q_e4 in Q_E4 and round(profile.q * 10000) == profile.q_e4, f"action {profile.action_id} q drift")
    expected_drop = math.floor(profile.q * SPATIAL_CELLS + 0.5)
    require(profile.drop_count == expected_drop, f"action {profile.action_id} drop-count drift")
    require(profile.keep_count == SPATIAL_CELLS - expected_drop, f"action {profile.action_id} keep-count drift")
    require(profile.bit_width == int(profile.quantizer.removeprefix("UINT")), f"action {profile.action_id} bit-width drift")
    require(profile.zstd_level == 1, f"action {profile.action_id} zstd level drift")
    require(profile.wire.layout == "CURRENT_CELL_MAJOR", f"action {profile.action_id} layout drift")
    if profile.family == "noAE":
        require(profile.family_id == 0, f"action {profile.action_id} noAE family-id drift")
        require(profile.latent_width is None and profile.transported_channels == 256, f"action {profile.action_id} noAE width drift")
        require(profile.routing_tag == 0 and profile.ae_checkpoint is None, f"action {profile.action_id} noAE routing drift")
    else:
        expected = {"AE128": (1, 128), "AE64": (2, 64), "AE32": (3, 32)}[profile.family]
        require((profile.family_id, profile.latent_width, profile.transported_channels) == (expected[0], expected[1], expected[1]), f"action {profile.action_id} AE family/width drift")
        require(profile.ae_checkpoint is not None, f"action {profile.action_id} AE checkpoint is missing")
        require(profile.routing_tag == int(profile.ae_checkpoint.sha256[:8], 16) != 0, f"action {profile.action_id} routing-tag drift")
    if profile.q_e4 == 0:
        require(profile.ranker_bypassed and profile.ranker_checkpoint is None, f"action {profile.action_id} q=0 ranker binding drift")
    else:
        require(not profile.ranker_bypassed and profile.ranker_checkpoint is not None, f"action {profile.action_id} ranker binding drift")


class SplitActionRegistry:
    """One immutable action-id lookup built from files read once at startup."""

    def __init__(self, profiles: tuple[ActionProfile, ...], audit: RegistryStartupAudit) -> None:
        self._profiles = profiles
        self._by_id = MappingProxyType({profile.action_id: profile for profile in profiles})
        self._audit = audit

    @classmethod
    def from_runtime_binding(
        cls,
        path: Path = DEFAULT_RUNTIME_BINDING,
        *,
        verify_runtime_artifacts: bool = True,
    ) -> "SplitActionRegistry":
        binding_path = path.resolve(strict=True)
        runtime_binding = _read_json(binding_path)
        require(runtime_binding.get("schema") == RUNTIME_BINDING_SCHEMA, "runtime-binding schema drift")
        require(runtime_binding.get("status") == RUNTIME_BINDING_STATUS, "runtime-binding status drift")
        inputs = runtime_binding.get("inputs", {})
        require(isinstance(inputs, dict), "runtime-binding inputs are missing")
        hash_operations = 0

        def verified_input(name: str) -> tuple[Path, Mapping[str, Any], bytes]:
            nonlocal hash_operations
            item = inputs.get(name)
            require(isinstance(item, dict), f"runtime-binding input is missing: {name}")
            source = _repository_path(str(item.get("path", "")))
            payload = source.read_bytes()
            observed = hashlib.sha256(payload).hexdigest()
            hash_operations += 1
            require(observed == item.get("sha256"), f"runtime-binding input hash drift: {name}")
            return source, item, payload

        catalog_path, catalog_item, catalog_bytes = verified_input("action_catalog")
        terminal_path, _, terminal_bytes = verified_input("action_catalog_terminal")
        campaign_path, campaign_item, campaign_bytes = verified_input("campaign_binding")
        campaign_terminal_path, _, campaign_terminal_bytes = verified_input(
            "campaign_binding_terminal"
        )
        catalog = _decode_json(catalog_bytes, catalog_path)
        campaign = _decode_json(campaign_bytes, campaign_path)
        require(catalog.get("schema") == CATALOG_SCHEMA, "catalog schema drift")
        require(campaign.get("schema") == CAMPAIGN_BINDING_SCHEMA, "campaign-binding schema drift")
        require(
            campaign.get("status") == "BOUND_CONFIGURATION_ONLY_REAL_LAUNCH_BLOCKED",
            "campaign-binding status drift",
        )
        require(campaign.get("catalog", {}).get("sha256") == catalog_item.get("sha256"), "campaign/catalog hash disagreement")
        require(campaign.get("catalog", {}).get("commit") == catalog_item.get("commit"), "campaign/catalog commit disagreement")
        require(
            terminal_bytes.decode("utf-8").strip()
            == f"{CATALOG_TERMINAL} {catalog_item['sha256']}",
            f"catalog terminal drift: {terminal_path}",
        )
        require(
            campaign_terminal_bytes.decode("utf-8").strip()
            == f"{CAMPAIGN_TERMINAL} {campaign_item['sha256']}",
            f"campaign terminal drift: {campaign_terminal_path}",
        )

        artifacts = runtime_binding.get("startup_artifacts", [])
        require(isinstance(artifacts, list), "runtime startup_artifacts must be a list")
        if verify_runtime_artifacts:
            seen: set[str] = set()
            for item in artifacts:
                require(isinstance(item, dict), "runtime startup artifact is invalid")
                artifact_path = str(item.get("path", ""))
                if artifact_path in seen:
                    continue
                seen.add(artifact_path)
                source = _repository_path(artifact_path)
                hash_operations += 1
                require(sha256_file(source) == item.get("sha256"), f"startup artifact hash drift: {artifact_path}")

        rows = catalog.get("profiles")
        require(isinstance(rows, list) and len(rows) == 72, "catalog must contain exactly 72 actions")
        profiles = tuple(_action_profile(row) for row in rows)
        require([profile.action_id for profile in profiles] == list(range(72)), "catalog action IDs are not contiguous 0..71")
        require(len({profile.profile_id for profile in profiles}) == 72, "catalog profile IDs are not unique")
        require(
            {(p.family, p.quantizer, p.q_e4) for p in profiles}
            == {(f, quantizer, q) for f in FAMILIES for quantizer in QUANTIZERS for q in Q_E4},
            "catalog family/quantizer/q Cartesian coverage drift",
        )
        for profile in profiles:
            _validate_profile(profile)
        selected = runtime_binding.get("selected_checkpoints", {})
        require(isinstance(selected, dict), "runtime selected_checkpoints are missing")
        reference = {profile.family: profile for profile in profiles if profile.q_e4 == 0 and profile.quantizer == "UINT8"}
        expected_checkpoints = {
            "perception": reference["noAE"].perception_checkpoint,
            "ranker": next(profile.ranker_checkpoint for profile in profiles if profile.ranker_checkpoint is not None),
            "AE128": reference["AE128"].ae_checkpoint,
            "AE64": reference["AE64"].ae_checkpoint,
            "AE32": reference["AE32"].ae_checkpoint,
        }
        require(
            selected
            == {
                name: {"path": checkpoint.path, "sha256": checkpoint.sha256}
                for name, checkpoint in expected_checkpoints.items()
                if checkpoint is not None
            },
            "runtime selected-checkpoint bindings disagree with catalog",
        )
        routes = runtime_binding.get("family_routes", {})
        require(isinstance(routes, dict), "runtime family_routes are missing")
        require(
            routes
            == {
                family: {
                    "family_id": profile.family_id,
                    "transported_channels": profile.transported_channels,
                    "latent_width": profile.latent_width,
                    "routing_tag": profile.routing_tag,
                    "decoder_identity": profile.decoder_identity,
                }
                for family, profile in reference.items()
            },
            "runtime family routing identities disagree with catalog",
        )
        inventory = campaign.get("inventory", {})
        require(
            (inventory.get("split_actions"), inventory.get("network_profiles"), inventory.get("campaign_cells"), inventory.get("pilot_cells"))
            == (72, 4, 288, 16),
            "Phase-12B campaign inventory drift",
        )
        return cls(
            profiles,
            RegistryStartupAudit(
                runtime_binding_reads=1,
                catalog_reads=1,
                catalog_terminal_reads=1,
                campaign_binding_reads=1,
                campaign_terminal_reads=1,
                artifact_hash_operations=hash_operations,
                verified_runtime_artifacts=verify_runtime_artifacts,
                action_count=72,
            ),
        )

    @property
    def profiles(self) -> tuple[ActionProfile, ...]:
        return self._profiles

    @property
    def startup_audit(self) -> RegistryStartupAudit:
        return self._audit

    def resolve(self, action_id: int) -> ActionProfile:
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise DispatchContractError("action_id must be an integer")
        profile = self._by_id.get(action_id)
        if profile is None:
            raise DispatchContractError(f"unregistered action_id {action_id}; expected 0..71")
        if profile.execution_mode != "SPLIT":
            raise DispatchContractError(f"action_id {action_id} is not enabled for SPLIT")
        return profile

    def find(self, family: str, quantizer: str, q_e4: int) -> ActionProfile:
        matches = [
            profile
            for profile in self._profiles
            if (profile.family, profile.quantizer, profile.q_e4)
            == (family, quantizer, q_e4)
        ]
        require(len(matches) == 1, "catalog dispatch tuple is not unique")
        return matches[0]
