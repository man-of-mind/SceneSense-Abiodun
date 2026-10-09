"""Policy-neutral identity: record the actor, state schema and checkpoint in use.

Nothing here names a particular policy.  The caller supplies a binding manifest
path and the weights file it pins; the profiler records what those files *say*
and what the bytes on disk *are*, plus what the live actor object reports.  A
Run-4, Run-5 or any other binding is handled the same way, as long as it is a
JSON document.  Known key paths are tried in order; anything absent is
recorded as ``None``, never filled with a default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SCHEMA = "scenesense.single_gpu_profiler.policy_identity.v1"

# (output field, candidate dotted key paths in the binding document)
BINDING_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("binding_schema", ("schema",)),
    ("manifest_sha256", ("manifest_sha256",)),
    ("actor_boundary_sha256", ("actor.boundary_sha256", "actor_boundary_sha256")),
    ("weights_file_name", ("actor.weights_file_name", "weights_file_name")),
    ("weights_file_sha256_declared", ("actor.weights_file_sha256", "weights_file_sha256",
                                      "actor_state_dict_sha256")),
    ("feature_schema_id", ("binding.feature_schema_id", "feature_schema_id")),
    ("feature_schema_sha256", ("binding.feature_schema_sha256", "feature_schema_sha256")),
    ("state_dim", ("binding.policy_feature_count", "policy_feature_count", "state_dim")),
    ("feature_order", ("binding.policy_feature_order", "policy_feature_order")),
    ("contract_schema_sha256", ("contract_schema_sha256", "binding.*contract_schema_sha256")),
    ("model_binding_sha256", ("model_binding_sha256", "binding.*model_binding_sha256")),
    ("deployment_rule", ("binding.deployment_rule", "deployment_rule")),
    ("seed", ("selected.seed", "seed")),
    ("update", ("selected.update", "actor.restored_update_count", "update")),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lookup(document: Mapping[str, Any], dotted: str) -> Any:
    """Dotted key path; a final ``*suffix`` part matches the one key ending in it."""
    value: Any = document
    for part in dotted.split("."):
        if not isinstance(value, Mapping):
            return None
        if part.startswith("*"):
            matches = [k for k in value if str(k).endswith(part[1:])]
            if len(matches) != 1:
                return None
            part = matches[0]
        if part not in value:
            return None
        value = value[part]
    return value


def binding_identity(binding_path: Path, weights_path: Optional[Path] = None) -> dict[str, Any]:
    """Identity of a binding manifest and the weights it pins (read-only)."""
    binding_path = Path(binding_path)
    document = json.loads(binding_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"schema": SCHEMA, "binding_path": str(binding_path),
                           "binding_file_sha256": sha256_file(binding_path)}
    for name, paths in BINDING_FIELDS:
        out[name] = next((v for v in (_lookup(document, p) for p in paths) if v is not None),
                         None)
    if out["feature_order"] is not None and out["state_dim"] is None:
        out["state_dim"] = len(out["feature_order"])
    if weights_path is None and out["weights_file_name"]:
        weights_path = binding_path.parent / str(out["weights_file_name"])
    if weights_path is not None:
        weights_path = Path(weights_path)
        out["weights_path"] = str(weights_path)
        out["weights_file_sha256_observed"] = (sha256_file(weights_path)
                                               if weights_path.is_file() else None)
        out["weights_match_declaration"] = (
            out["weights_file_sha256_declared"] is not None
            and out["weights_file_sha256_observed"] == out["weights_file_sha256_declared"])
    return out


def runtime_actor_identity(actor: Any) -> dict[str, Any]:
    """What the live actor object reports about itself (no call is made)."""
    out: dict[str, Any] = {"actor_class": f"{type(actor).__module__}.{type(actor).__qualname__}"}
    for attr in ("boundary_sha256", "actor_boundary_sha256", "state_dim", "feature_count"):
        try:
            value = getattr(actor, attr)
        except Exception:  # noqa: BLE001
            continue
        if callable(value):
            continue
        out[attr] = value
    return out


def decision_vector_identity(features: Any) -> dict[str, Any]:
    """Length and schema of one state vector the actor was called with."""
    out: dict[str, Any] = {"vector_type": f"{type(features).__module__}."
                                          f"{type(features).__qualname__}"}
    try:
        values: Sequence[Any] = (features.as_tuple() if hasattr(features, "as_tuple")
                                 else tuple(features))
        out["state_dim"] = len(values)
    except Exception:  # noqa: BLE001
        out["state_dim"] = None
    for attr in ("schema_id", "feature_schema_id", "SCHEMA_ID"):
        value = getattr(features, attr, None)
        if isinstance(value, str):
            out["feature_schema_id"] = value
            break
    return out
