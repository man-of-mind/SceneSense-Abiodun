"""Fail-closed final-actor gate for paired Run-4B-Joint and Run-5B.

Import is standard-library-only. Torch and model code are loaded only after
manifest/evidence/weight byte identities pass. The SNR-free Run-4B pilot is
rejected before tensor deserialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


MANIFEST_SCHEMA = "splitfusion.final_actor_validation_manifest.v2"
RUN4B_VARIANT = "RUN4B_JOINT_CHANNEL_COMPARATOR_V1"
RUN5B_VARIANT = "RUN5B_NO_LIVE_QPERC_V1"
SCIENTIFIC_CHANNEL_SHA256 = (
    "870bf3558722291f51fac594eda41fba6aca3d9225165fb645de636287518e63"
)
RUN5B_ONLY_AUTHORITY_SHA256 = (
    "14a192f653c289d94d254823dca9e3145410c677604b9bdcd9a278bd4dac17d7"
)
PILOT_ACTOR_SHA256 = (
    "b4337f000be9acbd16113d0891ed40fdac28c4f0bcb31c5b970a52d901c2ebc2"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_SHA = re.compile(r"[0-9a-f]{40}")

RUN4B_FEATURE_ORDER = (
    "camera_si_scaled",
    "radar_p40",
    "prior_ul_mcs_normalized",
    "pre_action_rlc_backlog_log1p_scaled",
    *(f"prev_joint_mode_{index}_one_hot" for index in range(12)),
    "prev_q_normalized",
    "prev_operational_latency_normalized",
    "prev_present",
    "prev_operational_success",
)
RUN5B_FEATURE_ORDER = (
    *RUN4B_FEATURE_ORDER,
    "effective_external_ul_snr_proxy_scaled",
)

_FIELDS = {
    "actor_export_relative_path", "actor_export_schema",
    "actor_export_sha256", "actor_state_dict_sha256",
    "actor_tree_sha256", "campaign_code_commit",
    "campaign_complete_relative_path", "campaign_complete_sha256",
    "feature_count", "feature_order", "feature_order_sha256",
    "feature_schema_id", "feature_schema_sha256",
    "fixture_outputs_sha256", "forbidden_actor_sha256s",
    "model_binding_sha256", "operational_latency_provider_sha256",
    "payload_index_relative_path", "payload_index_sha256",
    "preregistered_live_actor", "registration_relative_path",
    "registration_sha256", "reward_schema_sha256",
    "run5b_only_authority_sha256", "runner_binding_sha256", "schema",
    "scientific_channel_sha256", "selected_seed", "selected_update",
    "source_actor_relative_path", "source_git_branch",
    "source_git_head", "variant",
}

_COMMON = {
    "schema": MANIFEST_SCHEMA,
    "scientific_channel_sha256": SCIENTIFIC_CHANNEL_SHA256,
    "operational_latency_provider_sha256":
        "3cc1e6e36ae6e4f43dc8110077c62efbf5c719aaa179935b0f6de4c5bb1f6c29",
    "reward_schema_sha256":
        "a065a5c1e6bc69c276c967b17d0a14bfcd4cabf78cefe4b97737cc1a0864a0b6",
    "selected_seed": 43,
    "selected_update": 10000,
    "preregistered_live_actor": True,
    "forbidden_actor_sha256s": [PILOT_ACTOR_SHA256],
}

_EXPECTED = {
    RUN4B_VARIANT: {
        **_COMMON,
        "variant": RUN4B_VARIANT,
        "source_git_branch": "run4b-joint-channel-comparator-v1",
        "source_git_head":
            "0d73deaa18d593e4f51d92f873c389057597313e",
        "campaign_code_commit":
            "7598e4d853aef320a498ba8da5d4b9fdde1f9027",
        "source_actor_relative_path":
            "rl_agent/experiments/splitfusion_hybrid_sac_run4b_v1/"
            "20261001_7598e4d_joint_channel_comparator_v1/campaign/"
            "seed_43/final_actor/actor_state_dict.pt",
        "actor_state_dict_sha256":
            "643cb7adee66bc7e82665bf5cdbbcead39d6690c7af4fe988bd48be730753c3c",
        "actor_tree_sha256":
            "998a519f2f2039b1779c381ce247fbaeeb42f0edd138def75efd66d1932c6fc9",
        "fixture_outputs_sha256":
            "118060041d337dff32c12f758891cefeabfcb2543db1d64c84c2b20875e7b8a0",
        "actor_export_schema":
            "splitfusion.run4b_joint_channel.actor_export.v1",
        "actor_export_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run4b_v1/"
            "evidence_joint_channel_v1/seed_43_ACTOR_EXPORT.json",
        "actor_export_sha256":
            "cb508a743b9b9ffd6f52c1fe2ea924c85050afb5b43e02f9e24cf434fbad5c7f",
        "feature_schema_id": "splitfusion_run4b_policy_features_v1",
        "feature_schema_sha256":
            "217ed502dfbbd02807e55d9261a38570b04d9f805c67e1ca548c1aa7254a8e0d",
        "feature_order_sha256":
            "236624a8bc79626c08c216e34a2f4f45191a949ae435f9c5dbbb558e08d17818",
        "feature_count": 20,
        "feature_order": list(RUN4B_FEATURE_ORDER),
        "model_binding_sha256":
            "ed44a5c6c8cef661865ab64fbf821797e35fc072c83d78dd9bee4b28688cf992",
        "runner_binding_sha256":
            "3c014497ec599d0073e74b28b7ac57b6980b605b98b684c75b96e691ed7fb69f",
        "run5b_only_authority_sha256": None,
        "registration_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run4b_v1/"
            "RUN4B_JOINT_CHANNEL_COMPARATOR_V1_REGISTRATION.json",
        "registration_sha256":
            "5c85349bfad2fcd8ea1252026289f8e41c9c7978c8eb17f75ff08959e8b6ca9f",
        "campaign_complete_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run4b_v1/"
            "evidence_joint_channel_v1/CAMPAIGN_COMPLETE.json",
        "campaign_complete_sha256":
            "13f9246e821a23dabc29893ca3b4d6c94f7cb5d9f7b40273b93f9a05eef8cea7",
        "payload_index_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run4b_v1/"
            "evidence_joint_channel_v1/PAYLOAD_INDEX.json",
        "payload_index_sha256":
            "498f2a46928c085737c8e7bf4e0649fa7ac60280bb8fdfbeba09df0aec2027cf",
    },
    RUN5B_VARIANT: {
        **_COMMON,
        "variant": RUN5B_VARIANT,
        "source_git_branch": "run5b-no-qperc-v1",
        "source_git_head":
            "123c54bb179b77b33998afcbf41fddf837662136",
        "campaign_code_commit":
            "d5d501bd54167bd1ceaf1bd95249fcd4a4c06ce4",
        "source_actor_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run5b_v1/campaign_runs/"
            "campaign_three_seed_10000_v1/seed_43/final_actor/"
            "actor_state_dict.pt",
        "actor_state_dict_sha256":
            "10e5332d00a89a5a8138eaf5e835eb853166698f547559c7fe93206d59592e8e",
        "actor_tree_sha256":
            "258379e575c63c7468ca3dfef2b090afd7d0b08e2cfd339b9205886374f51dfd",
        "fixture_outputs_sha256":
            "7071c6538fe1be7d320708fd7c69bd25c307558f9104e49cbd27c4d073b1c582",
        "actor_export_schema": "splitfusion.run5b.actor_export.v1",
        "actor_export_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run5b_v1/"
            "evidence/seed_43_ACTOR_EXPORT.json",
        "actor_export_sha256":
            "25862371c715a5cd89ad9f2f9e1a0868a172863fe956284752751c843f5ca91b",
        "feature_schema_id": "splitfusion_run5b_policy_features_v2",
        "feature_schema_sha256":
            "5fd799df4cd84837fe2554c71056c31d386e791438f85a601d5924e75b1a57f7",
        "feature_order_sha256":
            "987bc7997230ea3867d0023bb7af95b817778ff5d28ffa3944ee5d6e66c24a47",
        "feature_count": 21,
        "feature_order": list(RUN5B_FEATURE_ORDER),
        "model_binding_sha256":
            "1fa190d8f965c49888de19c49ced819955d57fde0fe7db541fcbc3f5d85191a1",
        "runner_binding_sha256":
            "6f4f8fe3c54f0e65ab05d2f263e2ca33be8a012cf6d732b706061fcc0483c937",
        "run5b_only_authority_sha256": RUN5B_ONLY_AUTHORITY_SHA256,
        "registration_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run5b_v1/RUN5B_REGISTRATION.json",
        "registration_sha256":
            "ffa13b51a679fa61787d57bd84bd5d6b0a1ada96ca44737e84927622d84e483f",
        "campaign_complete_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run5b_v1/"
            "evidence/CAMPAIGN_COMPLETE.json",
        "campaign_complete_sha256":
            "c5a91a82685c7eee3a3774ae5c9be48b537994dded382dd18de76d7f124da9d2",
        "payload_index_relative_path":
            "rl_agent/splitfusion_hybrid_sac_run5b_v1/evidence/PAYLOAD_INDEX.json",
        "payload_index_sha256":
            "6a5483882d3fff9d2110ce1ff129873192306c1ea33ff42e0e65fc1d35d8b269",
    },
}


class FinalActorGateError(RuntimeError):
    """A final actor or one of its authorities differed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalActorGateError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(),
             f"{label} must be a regular non-symlink file")
    return path


def _exact(observed: Any, expected: Any, path: str = "manifest") -> None:
    _require(type(observed) is type(expected), f"{path} has a foreign type")
    if type(expected) is dict:
        _require(set(observed) == set(expected),
                 f"{path} fields are incomplete or foreign")
        for key in expected:
            _exact(observed[key], expected[key], f"{path}.{key}")
    elif type(expected) is list:
        _require(len(observed) == len(expected), f"{path} length differs")
        for index, (left, right) in enumerate(zip(observed, expected)):
            _exact(left, right, f"{path}[{index}]")
    else:
        _require(observed == expected, f"{path} differs")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    path = _regular(path, label)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalActorGateError(f"{label} is unreadable") from exc
    _require(type(raw) is dict, f"{label} root is not an exact object")
    return raw


@dataclass(frozen=True, slots=True)
class FinalActorIdentityV2:
    variant: str
    actor_state_dict_sha256: str
    actor_tree_sha256: str
    feature_schema_id: str
    feature_schema_sha256: str
    feature_order_sha256: str
    feature_order: tuple[str, ...]
    model_binding_sha256: str
    operational_latency_provider_sha256: str
    scientific_channel_sha256: str
    run5b_only_authority_sha256: str | None
    selected_seed: int
    selected_update: int


@dataclass(frozen=True, slots=True)
class LoadedFinalActorV2:
    identity: FinalActorIdentityV2
    _module: Any

    @property
    def module(self) -> Any:
        return self._module


def load_manifest(path: Path) -> tuple[FinalActorIdentityV2, Mapping[str, Any]]:
    raw = _read_json(path, "final actor manifest")
    _require(set(raw) == _FIELDS,
             "final actor manifest fields are incomplete or foreign")
    variant = raw.get("variant")
    _require(type(variant) is str and variant in _EXPECTED,
             "final actor variant is unknown")
    _exact(raw, _EXPECTED[variant])
    for key, value in raw.items():
        if key.endswith("_sha256") and value is not None:
            _require(type(value) is str and bool(_SHA256.fullmatch(value)),
                     f"{key} is not a lowercase SHA-256")
    _require(bool(_GIT_SHA.fullmatch(raw["source_git_head"])),
             "source_git_head is not a Git object identity")
    _require(bool(_GIT_SHA.fullmatch(raw["campaign_code_commit"])),
             "campaign_code_commit is not a Git object identity")
    identity = FinalActorIdentityV2(
        variant=variant,
        actor_state_dict_sha256=raw["actor_state_dict_sha256"],
        actor_tree_sha256=raw["actor_tree_sha256"],
        feature_schema_id=raw["feature_schema_id"],
        feature_schema_sha256=raw["feature_schema_sha256"],
        feature_order_sha256=raw["feature_order_sha256"],
        feature_order=tuple(raw["feature_order"]),
        model_binding_sha256=raw["model_binding_sha256"],
        operational_latency_provider_sha256=
            raw["operational_latency_provider_sha256"],
        scientific_channel_sha256=raw["scientific_channel_sha256"],
        run5b_only_authority_sha256=raw["run5b_only_authority_sha256"],
        selected_seed=raw["selected_seed"],
        selected_update=raw["selected_update"],
    )
    return identity, MappingProxyType(raw)


def _verify_evidence(document: Mapping[str, Any],
                     evidence_root: Path) -> None:
    root = Path(evidence_root)
    checks = (
        ("actor_export_relative_path", "actor_export_sha256", "actor export"),
        ("registration_relative_path", "registration_sha256", "registration"),
        ("campaign_complete_relative_path", "campaign_complete_sha256",
         "campaign completion"),
        ("payload_index_relative_path", "payload_index_sha256",
         "payload index"),
    )
    for path_key, hash_key, label in checks:
        relative = Path(document[path_key])
        _require(not relative.is_absolute() and ".." not in relative.parts,
                 f"{label} path escapes evidence root")
        path = _regular(root / relative, label)
        _require(_sha_file(path) == document[hash_key],
                 f"{label} hash differs")

    export = _read_json(root / document["actor_export_relative_path"],
                        "actor export")
    required_export = {
        "actor_state_dict_sha256": document["actor_state_dict_sha256"],
        "actor_tree_sha256": document["actor_tree_sha256"],
        "fixture_outputs_sha256": document["fixture_outputs_sha256"],
        "feature_order": document["feature_order"],
        "feature_schema_sha256": document["feature_schema_sha256"],
        "model_binding_sha256": document["model_binding_sha256"],
        "operational_latency_provider_sha256":
            document["operational_latency_provider_sha256"],
        "preregistered_live_actor": True,
        "runner_binding_sha256": document["runner_binding_sha256"],
        "schema": document["actor_export_schema"],
        "seed": 43,
        "update": 10000,
    }
    for key, expected in required_export.items():
        _require(key in export, f"actor export lacks {key}")
        _exact(export[key], expected, f"actor_export.{key}")
    if document["variant"] == RUN5B_VARIANT:
        _require(export.get("feature_order_sha256")
                 == document["feature_order_sha256"],
                 "Run-5B export feature-order hash differs")
        _require(export.get("feature_schema_id")
                 == document["feature_schema_id"],
                 "Run-5B export feature schema id differs")
        _require(export.get("joint_channel_binding_sha256")
                 == RUN5B_ONLY_AUTHORITY_SHA256,
                 "Run-5B export authority differs")
        _require(export.get("registration_sha256")
                 == document["registration_sha256"],
                 "Run-5B export registration differs")
    else:
        _require("joint_channel_binding_sha256" not in export,
                 "Run-4B export falsely claims the Run-5B authority")


def _load_tensor_state(weights_path: Path, identity: FinalActorIdentityV2,
                       observed_file_sha256: str) -> Any:
    import torch
    from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
        _tree_sha256,
    )
    from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import (
        HybridSacModelConfig,
        build_actor,
    )
    from rl_agent.splitfusion_hybrid_sac_v1.modeled_smoke_support import (
        MODELED_SMOKE_SUPPORT,
    )

    _require(observed_file_sha256 == identity.actor_state_dict_sha256,
             "actor file hash differs")
    rng_before = torch.random.get_rng_state().clone()
    cuda_before = torch.cuda.is_initialized()
    try:
        state = torch.load(
            weights_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FinalActorGateError("weights-only CPU load failed") from exc
    _require(isinstance(state, Mapping),
             "actor payload is not a state mapping")
    for key, tensor in state.items():
        _require(type(key) is str and type(tensor) is torch.Tensor,
                 "actor payload is not an exact tensor state_dict")
        _require(tensor.device.type == "cpu", f"{key} escaped CPU")
        if tensor.is_floating_point():
            _require(tensor.dtype is torch.float32, f"{key} is not float32")
            _require(bool(torch.isfinite(tensor).all()),
                     f"{key} is non-finite")
    tree = _tree_sha256(dict(state))
    _require(tree == identity.actor_tree_sha256,
             "actor tensor-tree identity differs")
    input_weight = state.get("encoder.0.weight")
    _require(type(input_weight) is torch.Tensor
             and input_weight.ndim == 2
             and int(input_weight.shape[1]) == len(identity.feature_order),
             "actor input width differs")
    config = HybridSacModelConfig(
        state_dim=len(identity.feature_order),
        mode_count=12,
        dtype=torch.float32,
        modeled_smoke_support=MODELED_SMOKE_SUPPORT,
    )
    actor = build_actor(config, seed=0)
    try:
        actor.load_state_dict(dict(state), strict=True)
    except (RuntimeError, ValueError) as exc:
        raise FinalActorGateError(
            "actor state does not fit the exact architecture") from exc
    actor.eval()
    actor.requires_grad_(False)
    _require(_tree_sha256(actor.state_dict()) == identity.actor_tree_sha256,
             "loaded actor tensor tree differs")
    _require(torch.equal(rng_before, torch.random.get_rng_state()),
             "final actor gate changed global torch RNG")
    _require(torch.cuda.is_initialized() == cuda_before,
             "final actor gate initialized CUDA")
    return actor


def verify_and_load_final_actor(
        manifest_path: Path, weights_path: Path, *,
        evidence_root: Path) -> LoadedFinalActorV2:
    identity, document = load_manifest(manifest_path)
    _verify_evidence(document, evidence_root)
    weights = _regular(Path(weights_path), "actor weights")
    _require(weights.name == "actor_state_dict.pt",
             "actor weights filename differs")
    observed = _sha_file(weights)
    _require(observed not in document["forbidden_actor_sha256s"],
             "quarantined SNR-free Run-4B pilot actor refused")
    actor = _load_tensor_state(weights, identity, observed)
    return LoadedFinalActorV2(identity=identity, _module=actor)


def bundled_manifest_path(variant: str) -> Path:
    _require(type(variant) is str and variant in _EXPECTED,
             "final actor variant is unknown")
    name = (
        "RUN4B_JOINT_FINAL_ACTOR_MANIFEST_V2.json"
        if variant == RUN4B_VARIANT
        else "RUN5B_JOINT_FINAL_ACTOR_MANIFEST_V2.json"
    )
    return Path(__file__).resolve().with_name(name)
