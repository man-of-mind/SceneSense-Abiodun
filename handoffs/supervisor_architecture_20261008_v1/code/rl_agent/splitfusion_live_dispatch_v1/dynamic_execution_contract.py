"""Fail-closed Phase-1 contract for continuous-q SplitFusion dispatch.

This module bridges the frozen Hybrid-SAC action contract to the artifacts
already qualified by :mod:`splitfusion_live_dispatch_v1`.  It deliberately
does *not* change the SFD1 envelope or either runtime.  Its only responsibility
is to resolve an authoritative ``(mode_id, q_e4)`` pair into an immutable,
artifact-bound execution description.

Important boundaries
--------------------

* ``q_e4`` is authoritative.  This module accepts no floating-point ``q`` and
  never performs a second policy-to-wire quantization.
* A catalog ``action_id`` / ``profile_id`` is retained only at one of the six
  exact measured anchors.  Off-anchor actions carry both as ``None``; no
  nearest anchor, interpolation, or synthetic identity is permitted.
* Loading is explicit and verifies the exact catalog, action-contract source,
  reviewed behavioral-source binding, runtime binding, checkpoints, codecs,
  and startup source artifacts.  A normal dotted import first executes this
  package's eager ``__init__`` and may therefore import :mod:`torch`; it must
  not initialize CUDA, read runtime/evidence artifacts, run inference, or
  launch a service.
* Off-anchor actions are executable descriptions, not measured scientific
  claims.  They are labelled ``UNMEASURED_OFF_ANCHOR``.

SFD1 v3, UE/edge runtime integration, map publication, and live qualification
are intentionally outside this phase.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

from rl_agent.splitfusion_hybrid_sac_v1 import action_contract as ac

from . import registry as legacy_registry

__all__ = [
    "DynamicExecutionContractError",
    "DYNAMIC_EXECUTION_PROFILE_SCHEMA",
    "EXECUTION_BUNDLE_SCHEMA",
    "MODE_INVARIANT_PROOF_SCHEMA",
    "BEHAVIORAL_SOURCE_BINDING_SCHEMA",
    "BEHAVIORAL_SOURCE_BINDING_STATUS",
    "BEHAVIORAL_SOURCE_BINDING_RELATIVE_PATH",
    "EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS",
    "RUNTIME_BINDING_SHA256",
    "ACTION_CONTRACT_SOURCE_SHA256",
    "MEASURED_ANCHOR",
    "UNMEASURED_OFF_ANCHOR",
    "ArtifactBinding",
    "CheckpointBinding",
    "WireExecutionIdentity",
    "ModeInvariantProof",
    "ModeExecutionBinding",
    "ExecutionBundleDescriptor",
    "ExecutableDispatchProfile",
    "DynamicExecutionContract",
    "load_dynamic_execution_contract",
]


DYNAMIC_EXECUTION_PROFILE_SCHEMA = (
    "scenesense.splitfusion_dynamic_execution_profile.v1"
)
EXECUTION_BUNDLE_SCHEMA = "scenesense.splitfusion_execution_bundle.v1"
MODE_INVARIANT_PROOF_SCHEMA = "scenesense.splitfusion_mode_invariant_proof.v1"
BEHAVIORAL_SOURCE_BINDING_SCHEMA = (
    "scenesense.splitfusion_dynamic_execution_behavioral_source_binding.v1"
)
BEHAVIORAL_SOURCE_BINDING_STATUS = "CONTRACT_ONLY_NOT_LIVE_RUNTIME_INTEGRATED"
BEHAVIORAL_SOURCE_BINDING_RELATIVE_PATH = (
    "rl_agent/splitfusion_live_dispatch_v1/"
    "dynamic_execution_source_binding.json"
)
_BEHAVIORAL_SOURCE_TRUST_AUTHORITY = "REVIEWED_VERSION_CONTROLLED_TRUST_ANCHOR"
_BEHAVIORAL_SOURCE_TRUST_RULE = (
    "The manifest is the acyclic review root: it hashes the dynamic contract "
    "and every listed behavioral source, while issued execution bundles hash "
    "the manifest. Manifest and source changes require joint code review."
)

# This exact role/path set is part of the reviewed contract.  The external
# binding JSON is an intentionally acyclic version-controlled trust anchor: it
# hashes this module and every lower-level implementation that gives mode/q an
# execution meaning.  The module therefore does not embed the JSON's hash
# (which would create an impossible self-hash cycle).  Instead, loading proves
# the exact set and all source bytes, and every execution-bundle digest carries
# the binding JSON's own SHA-256.  A reviewer must approve the binding JSON and
# source changes together; an already-issued bundle detects either change.
EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "dynamic_execution_contract_source": (
            "rl_agent/splitfusion_live_dispatch_v1/"
            "dynamic_execution_contract.py"
        ),
        "hybrid_q_contract_source": (
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/contract.py"
        ),
        "hybrid_q_selection_source": (
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/selection.py"
        ),
        "hybrid_q_sparse_codec_source": (
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/codec.py"
        ),
        "ae_composition_source": (
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_composition.py"
        ),
    }
)

# This Phase-1 adapter is reviewed against one exact pre-existing runtime
# binding.  A changed binding must produce a new reviewed contract rather than
# silently changing an execution bundle beneath a learned policy.
RUNTIME_BINDING_SHA256 = (
    "604450c7f0f791a480ba4e5ed6b286a2b6f68497735372a7a63ed92e93585551"
)

ACTION_CONTRACT_SOURCE_RELATIVE_PATH = (
    "rl_agent/splitfusion_hybrid_sac_v1/action_contract.py"
)
ACTION_CONTRACT_SOURCE_SHA256 = (
    "72ba342cb9c4772c152f56ae00a490bc5048c154047e9f7944c4614cb1a9d6fe"
)

MEASURED_ANCHOR = "MEASURED_ANCHOR"
UNMEASURED_OFF_ANCHOR = "UNMEASURED_OFF_ANCHOR"

# These are the fields this adapter declares to be invariant across the six
# catalog rows of one (family, quantizer) mode.  Ranker use is intentionally
# absent: q=0 bypasses it while every q>0 anchor uses the same frozen ranker.
_DIRECT_MODE_INVARIANT_FIELDS = (
    "execution_mode",
    "family",
    "family_id",
    "quantizer",
    "bit_width",
    "latent_width",
    "transported_channels",
    "decoder_identity",
    "routing_tag",
    "zstd_level",
    "checkpoint_sha256",
)
_NESTED_MODE_INVARIANT_FIELDS = (
    ("perception_checkpoint", "path"),
    ("perception_checkpoint", "sha256"),
    ("ae_checkpoint", "path"),
    ("ae_checkpoint", "sha256"),
    ("wire", "magic_ascii"),
    ("wire", "version"),
    ("wire", "codec_id"),
    ("wire", "layout"),
    ("wire", "codec_source"),
)

# Exact deployed codec identity, independently checked after catalog/runtime
# reconciliation.  UINT8 has distinct noAE and AE framings; UINT6/UINT4 share
# the registered low-bit framing but retain distinct quantizer/bit-width modes.
_EXPECTED_CODEC_IDENTITY_BY_FAMILY_QUANTIZER: Mapping[
    tuple[str, str], tuple[int, str, str]
] = MappingProxyType(
    {
        ("noAE", "UINT8"): (
            1,
            "HQ8\\0",
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1/uint8_codec.py",
        ),
        ("noAE", "UINT6"): (
            3,
            "HQLB",
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_transport.py",
        ),
        ("noAE", "UINT4"): (
            3,
            "HQLB",
            "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
            "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_transport.py",
        ),
        **{
            (family, "UINT8"): (
                2,
                "AE8\\0",
                "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
                "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/ae_uint8_transport.py",
            )
            for family in ("AE128", "AE64", "AE32")
        },
        **{
            (family, quantizer): (
                3,
                "HQLB",
                "pole_lraspp_multimodal_fusion/object_head_pilot_v1/"
                "splitfusion_fcos_r50_fpn_p2_p7_ae_v1/lowbit_transport.py",
            )
            for family in ("AE128", "AE64", "AE32")
            for quantizer in ("UINT6", "UINT4")
        },
    }
)
_EXPECTED_BIT_WIDTH_BY_QUANTIZER: Mapping[str, int] = MappingProxyType(
    {"UINT8": 8, "UINT6": 6, "UINT4": 4}
)


class DynamicExecutionContractError(RuntimeError):
    """A dynamic dispatch identity or one of its frozen bindings drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DynamicExecutionContractError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    """Return the single canonical representation used for every digest."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _read_json_object(path: Path, *, label: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DynamicExecutionContractError(
            f"cannot read {label} as UTF-8 JSON object at {path}: {exc}"
        ) from exc
    _require(isinstance(value, Mapping), f"{label} root must be a JSON object")
    return payload, value


def _repository_path(relative: str) -> Path:
    _require(isinstance(relative, str) and relative != "", "bound path is empty")
    try:
        path = (legacy_registry.REPOSITORY_ROOT / relative).resolve(strict=True)
    except OSError as exc:
        raise DynamicExecutionContractError(
            f"bound repository path does not exist: {relative}: {exc}"
        ) from exc
    try:
        path.relative_to(legacy_registry.REPOSITORY_ROOT)
    except ValueError as exc:
        raise DynamicExecutionContractError(
            f"bound path escapes repository root: {relative}"
        ) from exc
    return path


def _action_contract_source_path() -> Path:
    """Separated for a no-mutation source-drift regression test."""
    return _repository_path(ACTION_CONTRACT_SOURCE_RELATIVE_PATH)


def _behavioral_source_binding_path() -> Path:
    """Separated for fail-closed binding/tamper regression tests."""
    return _repository_path(BEHAVIORAL_SOURCE_BINDING_RELATIVE_PATH)


def _strict_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise DynamicExecutionContractError(
            f"{name} must be an integer, got {type(value).__name__}: {value!r}"
        )
    integer = int(value)
    if not minimum <= integer <= maximum:
        raise DynamicExecutionContractError(
            f"{name} {integer} is outside [{minimum}, {maximum}]"
        )
    return integer


def _checkpoint_from_mapping(value: Any, *, label: str) -> "CheckpointBinding":
    _require(isinstance(value, Mapping), f"{label} must be a checkpoint mapping")
    path = value.get("path")
    digest = value.get("sha256")
    _require(isinstance(path, str) and path != "", f"{label}.path is invalid")
    _require(
        isinstance(digest, str) and len(digest) == 64,
        f"{label}.sha256 is invalid",
    )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise DynamicExecutionContractError(
            f"{label}.sha256 is not hexadecimal"
        ) from exc
    return CheckpointBinding(path=path, sha256=digest)


def _optional_checkpoint_from_mapping(
    value: Any, *, label: str
) -> Optional["CheckpointBinding"]:
    if value is None:
        return None
    return _checkpoint_from_mapping(value, label=label)


def _nested_value(row: Mapping[str, Any], outer: str, inner: str) -> Any:
    value = row.get(outer)
    if value is None and outer == "ae_checkpoint":
        return None
    _require(isinstance(value, Mapping), f"{outer} must be a mapping")
    _require(inner in value, f"{outer}.{inner} is missing")
    return value[inner]


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """One source or checkpoint hash verified at contract-load time."""

    role: str
    path: str
    sha256: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"path": self.path, "role": self.role, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class CheckpointBinding:
    """A frozen model checkpoint identity."""

    path: str
    sha256: str

    def to_canonical_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class WireExecutionIdentity:
    """The inner sparse-codec identity for a joint mode."""

    magic_ascii: str
    version: int
    codec_id: int
    layout: str
    codec_source: ArtifactBinding

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "codec_id": self.codec_id,
            "codec_source": self.codec_source.to_canonical_dict(),
            "layout": self.layout,
            "magic_ascii": self.magic_ascii,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ModeInvariantProof:
    """Durable proof that all six catalog anchors share one mode identity."""

    schema: str
    anchor_q_e4: Tuple[int, ...]
    anchor_action_ids: Tuple[int, ...]
    anchor_profile_ids: Tuple[str, ...]
    invariant_descriptor_json: str
    invariant_sha256: str


@dataclass(frozen=True, slots=True)
class ModeExecutionBinding:
    """All q-invariant execution fields for one of the 12 joint modes."""

    mode_id: int
    family: str
    family_id: int
    quantizer: str
    bit_width: int
    latent_width: Optional[int]
    transported_channels: int
    decoder_identity: str
    routing_tag: int
    zstd_level: int
    wire: WireExecutionIdentity
    perception_checkpoint: CheckpointBinding
    ae_checkpoint: Optional[CheckpointBinding]
    ranker_checkpoint: CheckpointBinding
    invariant_proof: ModeInvariantProof

    @property
    def canonical_mode(self) -> str:
        return f"SPLIT/{self.family}/{self.quantizer}"

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "ae_checkpoint": (
                None
                if self.ae_checkpoint is None
                else self.ae_checkpoint.to_canonical_dict()
            ),
            "bit_width": self.bit_width,
            "decoder_identity": self.decoder_identity,
            "family": self.family,
            "family_id": self.family_id,
            "invariant_proof_sha256": self.invariant_proof.invariant_sha256,
            "latent_width": self.latent_width,
            "mode_id": self.mode_id,
            "perception_checkpoint": self.perception_checkpoint.to_canonical_dict(),
            "quantizer": self.quantizer,
            "ranker_checkpoint": self.ranker_checkpoint.to_canonical_dict(),
            "routing_tag": self.routing_tag,
            "transported_channels": self.transported_channels,
            "wire": self.wire.to_canonical_dict(),
            "zstd_level": self.zstd_level,
        }


@dataclass(frozen=True, slots=True)
class ExecutionBundleDescriptor:
    """Canonical provenance descriptor for one exact mode/q execution."""

    schema: str
    profile_schema: str
    catalog_schema: str
    catalog_sha256: str
    action_contract_source_path: str
    action_contract_source_sha256: str
    behavioral_source_binding_path: str
    behavioral_source_binding_schema: str
    behavioral_source_binding_status: str
    behavioral_source_binding_sha256: str
    behavioral_source_trust_authority: str
    behavioral_source_trust_rule: str
    behavioral_sources: Tuple[ArtifactBinding, ...]
    runtime_binding_schema: str
    runtime_binding_sha256: str
    runtime_source_base_commit: str
    mode: ModeExecutionBinding
    q_e4: int
    keep_count: int
    drop_count: int
    ranker_bypassed: bool
    active_ranker_checkpoint: Optional[CheckpointBinding]
    action_id: Optional[int]
    profile_id: Optional[str]
    measurement_status: str
    startup_artifacts: Tuple[ArtifactBinding, ...]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "action_contract_source": {
                "path": self.action_contract_source_path,
                "sha256": self.action_contract_source_sha256,
            },
            "anchor_identity": (
                None
                if self.action_id is None
                else {
                    "action_id": self.action_id,
                    "profile_id": self.profile_id,
                }
            ),
            "behavioral_source_binding": {
                "path": self.behavioral_source_binding_path,
                "schema": self.behavioral_source_binding_schema,
                "sha256": self.behavioral_source_binding_sha256,
                "sources": [
                    item.to_canonical_dict() for item in self.behavioral_sources
                ],
                "status": self.behavioral_source_binding_status,
                "trust_bootstrap": {
                    "authority": self.behavioral_source_trust_authority,
                    "rule": self.behavioral_source_trust_rule,
                },
            },
            "catalog": {
                "schema": self.catalog_schema,
                "sha256": self.catalog_sha256,
            },
            "measurement_status": self.measurement_status,
            "mode": self.mode.to_canonical_dict(),
            "profile_schema": self.profile_schema,
            "quality": {
                "drop_count": self.drop_count,
                "keep_count": self.keep_count,
                "q_e4": self.q_e4,
                "q_exec_decimal": f"{self.q_e4 / ac.Q_E4_SCALE:.4f}",
            },
            "ranker": {
                "bypassed": self.ranker_bypassed,
                "checkpoint": (
                    None
                    if self.active_ranker_checkpoint is None
                    else self.active_ranker_checkpoint.to_canonical_dict()
                ),
            },
            "runtime_binding": {
                "schema": self.runtime_binding_schema,
                "sha256": self.runtime_binding_sha256,
                "source_base_commit": self.runtime_source_base_commit,
            },
            "schema": self.schema,
            "startup_artifacts": [
                item.to_canonical_dict() for item in self.startup_artifacts
            ],
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_canonical_dict())

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.canonical_json.encode("ascii"))


@dataclass(frozen=True, slots=True)
class ExecutableDispatchProfile:
    """Immutable live-dispatch description for one exact continuous-q action."""

    schema: str
    mode_id: int
    family: str
    family_id: int
    quantizer: str
    bit_width: int
    latent_width: Optional[int]
    transported_channels: int
    decoder_identity: str
    routing_tag: int
    zstd_level: int
    wire: WireExecutionIdentity
    q_e4: int
    q_exec: float
    keep_count: int
    drop_count: int
    ranker_bypassed: bool
    perception_checkpoint: CheckpointBinding
    ae_checkpoint: Optional[CheckpointBinding]
    ranker_checkpoint: Optional[CheckpointBinding]
    action_id: Optional[int]
    profile_id: Optional[str]
    measurement_status: str
    catalog_schema: str
    catalog_sha256: str
    runtime_binding_schema: str
    runtime_binding_sha256: str
    contract_status: str
    execution_bundle: ExecutionBundleDescriptor
    execution_bundle_sha256: str

    @property
    def is_registered_anchor(self) -> bool:
        return self.action_id is not None


@dataclass(frozen=True, slots=True)
class DynamicExecutionContract:
    """Bound resolver for all 12 modes and all 9,801 wire qualities."""

    action_contract: ac.SplitActionContract
    runtime_binding_path: Path
    runtime_binding_schema: str
    runtime_binding_sha256: str
    runtime_source_base_commit: str
    behavioral_source_binding_path: Path
    behavioral_source_binding_sha256: str
    behavioral_source_trust_authority: str
    behavioral_source_trust_rule: str
    behavioral_sources: Tuple[ArtifactBinding, ...]
    modes: Tuple[ModeExecutionBinding, ...]
    startup_artifacts: Tuple[ArtifactBinding, ...]
    _modes_by_id: Mapping[int, ModeExecutionBinding]

    def mode(self, mode_id: int) -> ModeExecutionBinding:
        mode_id = _strict_int(
            mode_id,
            name="mode_id",
            minimum=0,
            maximum=ac.EXPECTED_MODE_COUNT - 1,
        )
        try:
            return self._modes_by_id[mode_id]
        except KeyError as exc:  # defensive: construction proves full coverage
            raise DynamicExecutionContractError(
                f"mode_id {mode_id} is absent from the bound mode inventory"
            ) from exc

    def resolve_q_e4(self, mode_id: int, q_e4: int) -> ExecutableDispatchProfile:
        """Resolve an already-quantized wire action; never quantize a float."""
        mode = self.mode(mode_id)
        q_e4 = _strict_int(
            q_e4,
            name="q_e4",
            minimum=ac.Q_E4_MIN,
            maximum=ac.Q_E4_MAX,
        )
        keep_count, drop_count = ac.keep_drop_counts(q_e4)
        anchor = self.action_contract.find_anchor(
            mode.family, mode.quantizer, q_e4
        )
        action_id = None if anchor is None else anchor.action_id
        profile_id = None if anchor is None else anchor.profile_id
        measurement_status = (
            UNMEASURED_OFF_ANCHOR if anchor is None else MEASURED_ANCHOR
        )
        ranker_bypassed = q_e4 == 0
        active_ranker = None if ranker_bypassed else mode.ranker_checkpoint

        bundle = ExecutionBundleDescriptor(
            schema=EXECUTION_BUNDLE_SCHEMA,
            profile_schema=DYNAMIC_EXECUTION_PROFILE_SCHEMA,
            catalog_schema=self.action_contract.schema,
            catalog_sha256=self.action_contract.catalog_sha256,
            action_contract_source_path=ACTION_CONTRACT_SOURCE_RELATIVE_PATH,
            action_contract_source_sha256=ACTION_CONTRACT_SOURCE_SHA256,
            behavioral_source_binding_path=(
                BEHAVIORAL_SOURCE_BINDING_RELATIVE_PATH
            ),
            behavioral_source_binding_schema=BEHAVIORAL_SOURCE_BINDING_SCHEMA,
            behavioral_source_binding_status=BEHAVIORAL_SOURCE_BINDING_STATUS,
            behavioral_source_binding_sha256=(
                self.behavioral_source_binding_sha256
            ),
            behavioral_source_trust_authority=(
                self.behavioral_source_trust_authority
            ),
            behavioral_source_trust_rule=self.behavioral_source_trust_rule,
            behavioral_sources=self.behavioral_sources,
            runtime_binding_schema=self.runtime_binding_schema,
            runtime_binding_sha256=self.runtime_binding_sha256,
            runtime_source_base_commit=self.runtime_source_base_commit,
            mode=mode,
            q_e4=q_e4,
            keep_count=keep_count,
            drop_count=drop_count,
            ranker_bypassed=ranker_bypassed,
            active_ranker_checkpoint=active_ranker,
            action_id=action_id,
            profile_id=profile_id,
            measurement_status=measurement_status,
            startup_artifacts=self.startup_artifacts,
        )
        return ExecutableDispatchProfile(
            schema=DYNAMIC_EXECUTION_PROFILE_SCHEMA,
            mode_id=mode.mode_id,
            family=mode.family,
            family_id=mode.family_id,
            quantizer=mode.quantizer,
            bit_width=mode.bit_width,
            latent_width=mode.latent_width,
            transported_channels=mode.transported_channels,
            decoder_identity=mode.decoder_identity,
            routing_tag=mode.routing_tag,
            zstd_level=mode.zstd_level,
            wire=mode.wire,
            q_e4=q_e4,
            q_exec=q_e4 / ac.Q_E4_SCALE,
            keep_count=keep_count,
            drop_count=drop_count,
            ranker_bypassed=ranker_bypassed,
            perception_checkpoint=mode.perception_checkpoint,
            ae_checkpoint=mode.ae_checkpoint,
            ranker_checkpoint=active_ranker,
            action_id=action_id,
            profile_id=profile_id,
            measurement_status=measurement_status,
            catalog_schema=self.action_contract.schema,
            catalog_sha256=self.action_contract.catalog_sha256,
            runtime_binding_schema=self.runtime_binding_schema,
            runtime_binding_sha256=self.runtime_binding_sha256,
            contract_status=BEHAVIORAL_SOURCE_BINDING_STATUS,
            execution_bundle=bundle,
            execution_bundle_sha256=bundle.sha256,
        )

    def verify_profile(self, profile: ExecutableDispatchProfile) -> None:
        """Require a profile to equal this contract's authoritative resolution.

        Frozen dataclasses prevent ordinary mutation, but Python callers can
        still manufacture or ``dataclasses.replace`` a value object.  Runtime
        boundaries must therefore verify the complete record rather than trust
        its type or its self-reported digest.
        """
        if type(profile) is not ExecutableDispatchProfile:
            raise DynamicExecutionContractError(
                "profile must be an exact ExecutableDispatchProfile, got "
                f"{type(profile).__name__}"
            )
        expected = self.resolve_q_e4(profile.mode_id, profile.q_e4)
        _require(
            profile == expected,
            "executable dispatch profile disagrees with the authoritative "
            f"resolution for mode_id={profile.mode_id}, q_e4={profile.q_e4}",
        )
        _require(
            profile.execution_bundle_sha256
            == profile.execution_bundle.sha256,
            "execution-bundle digest disagrees with its canonical descriptor",
        )


def _artifacts_from_binding(
    document: Mapping[str, Any],
) -> Tuple[ArtifactBinding, ...]:
    values = document.get("startup_artifacts")
    _require(isinstance(values, Sequence), "startup_artifacts must be a sequence")
    artifacts = []
    paths: set[str] = set()
    roles: set[str] = set()
    for index, value in enumerate(values):
        _require(
            isinstance(value, Mapping),
            f"startup_artifacts[{index}] must be a mapping",
        )
        role = value.get("role")
        path = value.get("path")
        digest = value.get("sha256")
        _require(isinstance(role, str) and role != "", f"artifact {index} role invalid")
        _require(isinstance(path, str) and path != "", f"artifact {index} path invalid")
        _require(
            isinstance(digest, str) and len(digest) == 64,
            f"artifact {index} SHA-256 invalid",
        )
        _require(path not in paths, f"duplicate startup artifact path: {path}")
        _require(role not in roles, f"duplicate startup artifact role: {role}")
        paths.add(path)
        roles.add(role)
        artifacts.append(ArtifactBinding(role=role, path=path, sha256=digest))
    _require(artifacts, "startup_artifacts is empty")
    return tuple(sorted(artifacts, key=lambda item: (item.path, item.role)))


def _load_behavioral_source_binding(
    path: Path,
) -> tuple[str, str, str, Tuple[ArtifactBinding, ...]]:
    """Verify the acyclic reviewed source manifest and every bound source.

    The returned tuple is ``(manifest_sha256, trust_authority, trust_rule,
    sources)``.  The manifest is deliberately external: embedding its digest
    in the module that it hashes would make the trust graph cyclic.
    """
    payload, document = _read_json_object(path, label="behavioral source binding")
    _require(
        set(document) == {"schema", "sources", "status", "trust_bootstrap"},
        "behavioral source-binding has unexpected or missing top-level fields",
    )
    _require(
        document.get("schema") == BEHAVIORAL_SOURCE_BINDING_SCHEMA,
        "behavioral source-binding schema drift",
    )
    _require(
        document.get("status") == BEHAVIORAL_SOURCE_BINDING_STATUS,
        "behavioral source-binding status drift",
    )
    trust = document.get("trust_bootstrap")
    _require(
        isinstance(trust, Mapping),
        "behavioral source-binding trust_bootstrap must be a mapping",
    )
    _require(
        set(trust) == {"authority", "rule"},
        "behavioral source-binding trust_bootstrap fields drift",
    )
    _require(
        trust.get("authority") == _BEHAVIORAL_SOURCE_TRUST_AUTHORITY,
        "behavioral source-binding trust authority drift",
    )
    _require(
        trust.get("rule") == _BEHAVIORAL_SOURCE_TRUST_RULE,
        "behavioral source-binding trust rule drift",
    )

    values = document.get("sources")
    _require(
        isinstance(values, list),
        "behavioral source-binding sources must be a JSON array",
    )
    sources: list[ArtifactBinding] = []
    observed_role_paths: dict[str, str] = {}
    for index, value in enumerate(values):
        _require(
            isinstance(value, Mapping),
            f"behavioral source {index} must be a mapping",
        )
        _require(
            set(value) == {"path", "role", "sha256"},
            f"behavioral source {index} has unexpected or missing fields",
        )
        role = value.get("role")
        relative_path = value.get("path")
        expected_sha256 = value.get("sha256")
        _require(
            isinstance(role, str) and role != "",
            f"behavioral source {index} role is invalid",
        )
        _require(
            isinstance(relative_path, str) and relative_path != "",
            f"behavioral source {index} path is invalid",
        )
        _require(
            isinstance(expected_sha256, str) and len(expected_sha256) == 64,
            f"behavioral source {index} SHA-256 is invalid",
        )
        try:
            int(expected_sha256, 16)
        except ValueError as exc:
            raise DynamicExecutionContractError(
                f"behavioral source {index} SHA-256 is not hexadecimal"
            ) from exc
        _require(
            role not in observed_role_paths,
            f"duplicate behavioral source role: {role}",
        )
        observed_role_paths[role] = relative_path
        observed_sha256 = _sha256_file(_repository_path(relative_path))
        _require(
            observed_sha256 == expected_sha256,
            "behavioral source hash drift for "
            f"{role} ({relative_path}): expected {expected_sha256}, "
            f"got {observed_sha256}",
        )
        sources.append(
            ArtifactBinding(
                role=role,
                path=relative_path,
                sha256=expected_sha256,
            )
        )

    _require(
        observed_role_paths == dict(EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS),
        "behavioral source role/path closure drift: expected "
        f"{dict(EXPECTED_BEHAVIORAL_SOURCE_ROLE_PATHS)!r}, got "
        f"{observed_role_paths!r}",
    )
    sources.sort(key=lambda item: item.role)
    return (
        _sha256_bytes(payload),
        str(trust["authority"]),
        str(trust["rule"]),
        tuple(sources),
    )


def _catalog_rows(document: Mapping[str, Any]) -> Mapping[tuple[str, str, int], Mapping[str, Any]]:
    rows = document.get("profiles")
    _require(isinstance(rows, Sequence), "catalog profiles must be a sequence")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"catalog profile {index} is invalid")
        key = (str(row.get("family")), str(row.get("quantizer")), int(row.get("q_e4")))
        _require(key not in indexed, f"duplicate catalog profile identity: {key}")
        indexed[key] = row
    return MappingProxyType(indexed)


def _prove_mode_invariants(
    *,
    mode: ac.JointMode,
    anchor_q_e4: Tuple[int, ...],
    rows: Sequence[Mapping[str, Any]],
) -> ModeInvariantProof:
    _require(len(rows) == ac.EXPECTED_Q_ANCHOR_COUNT, "mode does not have six anchors")
    _require(
        tuple(int(row["q_e4"]) for row in rows) == anchor_q_e4,
        f"{mode.canonical} anchor order drift",
    )
    invariant: dict[str, Any] = {}
    for field in _DIRECT_MODE_INVARIANT_FIELDS:
        values = [row.get(field) for row in rows]
        _require(
            all(value == values[0] for value in values[1:]),
            f"{mode.canonical} contradicts mode-invariant field {field}",
        )
        invariant[field] = values[0]
    for outer, inner in _NESTED_MODE_INVARIANT_FIELDS:
        values = [_nested_value(row, outer, inner) for row in rows]
        _require(
            all(value == values[0] for value in values[1:]),
            f"{mode.canonical} contradicts mode-invariant field {outer}.{inner}",
        )
        invariant[f"{outer}.{inner}"] = values[0]

    # The q-dependent ranker rule is also proved across all six anchors.
    for row in rows:
        ranker = row.get("ranker")
        _require(isinstance(ranker, Mapping), "catalog ranker binding is invalid")
        if int(row["q_e4"]) == 0:
            _require(
                ranker.get("bypassed") is True and ranker.get("checkpoint") is None,
                f"{mode.canonical} q=0 ranker bypass drift",
            )
        else:
            _require(
                ranker.get("bypassed") is False
                and isinstance(ranker.get("checkpoint"), Mapping),
                f"{mode.canonical} q>0 ranker binding drift",
            )
    positive_rankers = [
        row["ranker"]["checkpoint"] for row in rows if int(row["q_e4"]) > 0
    ]
    _require(
        all(value == positive_rankers[0] for value in positive_rankers[1:]),
        f"{mode.canonical} has contradictory positive-q ranker checkpoints",
    )
    invariant["ranker.q0"] = {"bypassed": True, "checkpoint": None}
    invariant["ranker.positive_q_checkpoint"] = positive_rankers[0]

    document = {
        "anchor_action_ids": [int(row["action_id"]) for row in rows],
        "anchor_profile_ids": [str(row["profile_id"]) for row in rows],
        "anchor_q_e4": list(anchor_q_e4),
        "invariant": invariant,
        "mode": {
            "family": mode.family,
            "mode_id": mode.mode_id,
            "quantizer": mode.quantizer,
        },
        "schema": MODE_INVARIANT_PROOF_SCHEMA,
    }
    canonical = _canonical_json(document)
    return ModeInvariantProof(
        schema=MODE_INVARIANT_PROOF_SCHEMA,
        anchor_q_e4=anchor_q_e4,
        anchor_action_ids=tuple(int(row["action_id"]) for row in rows),
        anchor_profile_ids=tuple(str(row["profile_id"]) for row in rows),
        invariant_descriptor_json=canonical,
        invariant_sha256=_sha256_bytes(canonical.encode("ascii")),
    )


def _build_mode_bindings(
    *,
    contract: ac.SplitActionContract,
    registry: legacy_registry.SplitActionRegistry,
    catalog_document: Mapping[str, Any],
    runtime_document: Mapping[str, Any],
    artifacts: Tuple[ArtifactBinding, ...],
) -> Tuple[ModeExecutionBinding, ...]:
    rows_by_key = _catalog_rows(catalog_document)
    artifact_by_path = {item.path: item for item in artifacts}
    selected = runtime_document.get("selected_checkpoints")
    routes = runtime_document.get("family_routes")
    _require(isinstance(selected, Mapping), "selected_checkpoints is invalid")
    _require(isinstance(routes, Mapping), "family_routes is invalid")
    ranker_checkpoint = _checkpoint_from_mapping(
        selected.get("ranker"), label="selected_checkpoints.ranker"
    )
    perception_checkpoint = _checkpoint_from_mapping(
        selected.get("perception"), label="selected_checkpoints.perception"
    )

    modes = []
    for action_mode in contract.modes:
        anchors = contract.anchors_for_mode(action_mode.mode_id)
        rows = tuple(
            rows_by_key[(action_mode.family, action_mode.quantizer, anchor.q_e4)]
            for anchor in anchors
        )
        proof = _prove_mode_invariants(
            mode=action_mode,
            anchor_q_e4=contract.q_anchor_order,
            rows=rows,
        )
        reference = rows[0]
        wire = reference["wire"]
        codec_source = wire["codec_source"]
        codec_path = str(codec_source["path"])
        _require(codec_path in artifact_by_path, f"codec source is not a startup artifact: {codec_path}")
        _require(
            artifact_by_path[codec_path].sha256 == str(codec_source["sha256"]),
            f"codec source hash disagrees with startup artifact: {codec_path}",
        )
        expected_codec = _EXPECTED_CODEC_IDENTITY_BY_FAMILY_QUANTIZER.get(
            (action_mode.family, action_mode.quantizer)
        )
        _require(
            expected_codec is not None,
            f"no reviewed codec identity for {action_mode.canonical}",
        )
        _require(
            (
                int(wire["codec_id"]),
                str(wire["magic_ascii"]),
                codec_path,
            )
            == expected_codec,
            f"{action_mode.canonical} codec identity drift",
        )
        _require(
            int(action_mode.bit_width)
            == _EXPECTED_BIT_WIDTH_BY_QUANTIZER.get(action_mode.quantizer),
            f"{action_mode.canonical} bit-width/quantizer drift",
        )
        _require(
            int(wire["version"]) == 1
            and str(wire["layout"]) == "CURRENT_CELL_MAJOR",
            f"{action_mode.canonical} wire version/layout drift",
        )

        ae_checkpoint = _optional_checkpoint_from_mapping(
            reference.get("ae_checkpoint"), label=f"{action_mode.canonical}.ae"
        )
        selected_ae = selected.get(action_mode.family)
        if action_mode.family == "noAE":
            _require(ae_checkpoint is None and selected_ae is None, "noAE unexpectedly binds an AE checkpoint")
        else:
            _require(
                ae_checkpoint
                == _checkpoint_from_mapping(
                    selected_ae, label=f"selected_checkpoints.{action_mode.family}"
                ),
                f"{action_mode.canonical} AE checkpoint disagrees with runtime binding",
            )

        reference_profile = registry.find(
            action_mode.family, action_mode.quantizer, contract.q_anchor_order[0]
        )
        route = routes.get(action_mode.family)
        _require(isinstance(route, Mapping), f"missing family route {action_mode.family}")
        _require(
            dict(route)
            == {
                "family_id": action_mode.family_id,
                "transported_channels": action_mode.transported_channels,
                "latent_width": action_mode.latent_width,
                "routing_tag": action_mode.routing_tag,
                "decoder_identity": action_mode.decoder_identity,
            },
            f"{action_mode.canonical} route disagrees with action contract",
        )
        _require(
            reference_profile.perception_checkpoint.sha256
            == perception_checkpoint.sha256,
            f"{action_mode.canonical} perception checkpoint drift",
        )
        for anchor in anchors:
            legacy = registry.resolve(anchor.action_id)
            _require(
                (legacy.family, legacy.quantizer, legacy.q_e4, legacy.profile_id)
                == (
                    action_mode.family,
                    action_mode.quantizer,
                    anchor.q_e4,
                    anchor.profile_id,
                ),
                f"legacy registry/action contract disagreement at action {anchor.action_id}",
            )
            if anchor.q_e4 > 0:
                _require(
                    legacy.ranker_checkpoint is not None
                    and legacy.ranker_checkpoint.sha256 == ranker_checkpoint.sha256,
                    f"action {anchor.action_id} ranker checkpoint drift",
                )

        modes.append(
            ModeExecutionBinding(
                mode_id=action_mode.mode_id,
                family=action_mode.family,
                family_id=action_mode.family_id,
                quantizer=action_mode.quantizer,
                bit_width=int(action_mode.bit_width),
                latent_width=action_mode.latent_width,
                transported_channels=action_mode.transported_channels,
                decoder_identity=action_mode.decoder_identity,
                routing_tag=action_mode.routing_tag,
                zstd_level=action_mode.zstd_level,
                wire=WireExecutionIdentity(
                    magic_ascii=str(wire["magic_ascii"]),
                    version=int(wire["version"]),
                    codec_id=int(wire["codec_id"]),
                    layout=str(wire["layout"]),
                    codec_source=artifact_by_path[codec_path],
                ),
                perception_checkpoint=perception_checkpoint,
                ae_checkpoint=ae_checkpoint,
                ranker_checkpoint=ranker_checkpoint,
                invariant_proof=proof,
            )
        )
    _require(len(modes) == ac.EXPECTED_MODE_COUNT, "did not build exactly 12 modes")
    _require(
        tuple(mode.mode_id for mode in modes) == tuple(range(ac.EXPECTED_MODE_COUNT)),
        "mode IDs are not the declared contiguous order",
    )
    return tuple(modes)


def load_dynamic_execution_contract(
    runtime_binding_path: Optional[Path] = None,
    behavioral_source_binding_path: Optional[Path] = None,
) -> DynamicExecutionContract:
    """Load and verify the exact Phase-1 dynamic execution contract.

    This operation hashes the bound checkpoints and sources but never loads a
    model, initializes CUDA, runs inference, or starts any service.  Torch may
    already have been imported by this package's eager ``__init__`` before the
    operation is called.
    """
    binding_path = (
        legacy_registry.DEFAULT_RUNTIME_BINDING
        if runtime_binding_path is None
        else Path(runtime_binding_path)
    ).resolve(strict=True)
    binding_bytes, runtime_document = _read_json_object(
        binding_path, label="runtime binding"
    )
    binding_sha256 = _sha256_bytes(binding_bytes)
    _require(
        binding_sha256 == RUNTIME_BINDING_SHA256,
        "runtime-binding SHA-256 drift: "
        f"expected {RUNTIME_BINDING_SHA256}, got {binding_sha256}",
    )
    _require(
        runtime_document.get("schema") == legacy_registry.RUNTIME_BINDING_SCHEMA,
        "runtime-binding schema drift",
    )
    _require(
        runtime_document.get("status") == legacy_registry.RUNTIME_BINDING_STATUS,
        "runtime-binding status drift",
    )

    reviewed_source_binding_path = _behavioral_source_binding_path()
    source_binding_path = (
        reviewed_source_binding_path
        if behavioral_source_binding_path is None
        else Path(behavioral_source_binding_path).resolve(strict=True)
    )
    (
        behavioral_source_binding_sha256,
        behavioral_source_trust_authority,
        behavioral_source_trust_rule,
        behavioral_sources,
    ) = _load_behavioral_source_binding(source_binding_path)
    _require(
        source_binding_path == reviewed_source_binding_path,
        "behavioral source binding was loaded from an unreviewed path: "
        f"expected {reviewed_source_binding_path}, got {source_binding_path}",
    )

    action_source = _action_contract_source_path()
    observed_action_source_sha = _sha256_file(action_source)
    _require(
        observed_action_source_sha == ACTION_CONTRACT_SOURCE_SHA256,
        "Hybrid-SAC action-contract source drift: "
        f"expected {ACTION_CONTRACT_SOURCE_SHA256}, got {observed_action_source_sha}",
    )

    # The established registry supplies the existing campaign/terminal checks,
    # hashes every startup artifact, and proves its 72-anchor view.  We then
    # reconcile that independent view with the Hybrid-SAC action contract.
    try:
        registry = legacy_registry.SplitActionRegistry.from_runtime_binding(
            binding_path, verify_runtime_artifacts=True
        )
    except (legacy_registry.DispatchContractError, OSError, ValueError) as exc:
        raise DynamicExecutionContractError(
            f"legacy runtime/artifact binding failed closed: {exc}"
        ) from exc
    _require(
        registry.startup_audit.verified_runtime_artifacts is True,
        "runtime artifacts were not hash-verified",
    )

    try:
        contract = ac.load_contract()
    except ac.ActionContractError as exc:
        raise DynamicExecutionContractError(
            f"Hybrid-SAC action catalog failed closed: {exc}"
        ) from exc

    inputs = runtime_document.get("inputs")
    _require(isinstance(inputs, Mapping), "runtime-binding inputs are invalid")
    catalog_input = inputs.get("action_catalog")
    _require(isinstance(catalog_input, Mapping), "action_catalog input is missing")
    _require(
        catalog_input.get("path") == ac.CATALOG_RELATIVE_PATH
        and catalog_input.get("schema") == ac.CATALOG_SCHEMA
        and catalog_input.get("sha256") == ac.CATALOG_SHA256,
        "runtime binding and Hybrid-SAC action contract disagree on the catalog",
    )
    catalog_bytes, catalog_document = _read_json_object(
        contract.catalog_path, label="frozen action catalog"
    )
    _require(
        _sha256_bytes(catalog_bytes) == contract.catalog_sha256 == ac.CATALOG_SHA256,
        "catalog bytes disagree after action-contract verification",
    )

    artifacts = _artifacts_from_binding(runtime_document)
    # Every selected checkpoint must be among the independently verified
    # startup artifacts, preventing a selected-but-unverified model identity.
    artifact_pairs = {(item.path, item.sha256) for item in artifacts}
    selected = runtime_document.get("selected_checkpoints")
    _require(isinstance(selected, Mapping), "selected_checkpoints is invalid")
    for name, checkpoint in selected.items():
        parsed = _checkpoint_from_mapping(checkpoint, label=f"selected.{name}")
        _require(
            (parsed.path, parsed.sha256) in artifact_pairs,
            f"selected checkpoint {name} is absent from verified startup artifacts",
        )

    modes = _build_mode_bindings(
        contract=contract,
        registry=registry,
        catalog_document=catalog_document,
        runtime_document=runtime_document,
        artifacts=artifacts,
    )
    return DynamicExecutionContract(
        action_contract=contract,
        runtime_binding_path=binding_path,
        runtime_binding_schema=str(runtime_document["schema"]),
        runtime_binding_sha256=binding_sha256,
        runtime_source_base_commit=str(runtime_document["source_base_commit"]),
        behavioral_source_binding_path=source_binding_path,
        behavioral_source_binding_sha256=behavioral_source_binding_sha256,
        behavioral_source_trust_authority=behavioral_source_trust_authority,
        behavioral_source_trust_rule=behavioral_source_trust_rule,
        behavioral_sources=behavioral_sources,
        modes=modes,
        startup_artifacts=artifacts,
        _modes_by_id=MappingProxyType({mode.mode_id: mode for mode in modes}),
    )
