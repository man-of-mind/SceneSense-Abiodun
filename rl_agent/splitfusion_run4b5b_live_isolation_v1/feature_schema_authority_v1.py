"""Exact training feature-schema authorities for the B live adapters.

The actor manifests carry the digest of the complete training schema, not a
deployment-specific wrapper around its feature names.  This module binds each
live variant to that authoritative digest and independently guards the
feature count/order plus the scaling, prior and excluded-information
semantics.  It is pure: importing or calling it performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as R4B
from rl_agent.splitfusion_hybrid_sac_run5b_v1 import (
    run5b_state_contract as R5B,
)


RUN4B_VARIANT = "RUN4B_MCS_BACKLOG_NO_QPERC"
RUN5B_VARIANT = "RUN5B_MCS_BACKLOG_SNR_NO_QPERC"
RUN4B_SCHEMA_SHA256 = (
    "217ed502dfbbd02807e55d9261a38570b04d9f805c67e1ca548c1aa7254a8e0d"
)
RUN5B_SCHEMA_SHA256 = (
    "5fd799df4cd84837fe2554c71056c31d386e791438f85a601d5924e75b1a57f7"
)

_RUN4_KEYS = {
    "schema_id", "schema_version", "feature_count", "feature_order",
    "scaling", "prior_semantics", "excluded",
}
_RUN5_KEYS = {
    "schema_id", "schema_version", "feature_count", "feature_order",
    "positions_0_19", "position_20", "q_perc",
    "forbidden_feature_tokens",
}
_SEMANTIC_SHA256 = {
    RUN4B_VARIANT: {
        "scaling": "2f612ee5336e4701fb8a49c7d63bf32be23b69fe37f79cc85438cfc3abbca9d0",
        "prior_semantics": "d972760e5c808fac5f655f16de0084a7f27cb43c63a46a0d1fcb363b19874ece",
        "excluded": "9012c2431fa897b224cf37c62bf4c398da1136eb6bb67f765bdeb115dff326ea",
    },
    RUN5B_VARIANT: {
        "positions_0_19": "28b392b332efe46ba3089fb0abbee02c34316784e343647215a4cea99ca6ae95",
        "position_20": "f0716d71cd69c401c83c6c0f801285ed5a65b37bfb1ff4c3b05248f6bdaed446",
        "q_perc": "bb33b1eddbeb5ae69ebc48de93c41acb968719446b770c7158e56f118bd48240",
        "forbidden_feature_tokens": "f26734f18f31f4e837d459497e1702b5c5d85263c3408c4a913d135a3e4a8ba3",
    },
}


class FeatureSchemaAuthorityError(ValueError):
    """A live feature schema differs from its frozen training authority."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise FeatureSchemaAuthorityError(
            "feature schema is not canonical JSON") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise FeatureSchemaAuthorityError(message)


def _authority(variant: str) -> tuple[Mapping[str, Any], str, set[str], str, int]:
    if variant == RUN4B_VARIANT:
        return (R4B.FEATURE_SCHEMA, RUN4B_SCHEMA_SHA256, _RUN4_KEYS,
                "splitfusion_run4b_policy_features_v1", 1)
    if variant == RUN5B_VARIANT:
        return (R5B.FEATURE_SCHEMA, RUN5B_SCHEMA_SHA256, _RUN5_KEYS,
                "splitfusion_run5b_policy_features_v2", 2)
    raise FeatureSchemaAuthorityError("unknown live actor variant")


def _verify(variant: str, order: Sequence[str],
            schema: Mapping[str, Any]) -> str:
    _original, expected_digest, keys, schema_id, version = _authority(variant)
    _require(type(schema) is dict and set(schema) == keys,
             "authoritative feature schema fields drifted")
    exact = tuple(order)
    _require(all(type(name) is str and name for name in exact),
             "live feature order contains a foreign item")
    _require(schema.get("schema_id") == schema_id,
             "authoritative feature schema id drifted")
    _require(schema.get("schema_version") == version,
             "authoritative feature schema version drifted")
    _require(schema.get("feature_count") == len(exact),
             "authoritative feature count differs from live order")
    _require(tuple(schema.get("feature_order", ())) == exact,
             "authoritative feature order differs element-wise")
    for field, expected in _SEMANTIC_SHA256[variant].items():
        _require(field in schema and _sha(schema[field]) == expected,
                 f"authoritative {field} semantics drifted")
    digest = _sha(schema)
    _require(digest == expected_digest,
             "authoritative complete feature schema digest drifted")
    return digest


def feature_schema_sha256(
        variant: str, order: Sequence[str], *,
        schema_override: Optional[Mapping[str, Any]] = None,
        ) -> str:
    """Return the verified complete training-schema digest for ``variant``.

    ``schema_override`` exists for pure refusal tests.  Production callers
    omit it and always read the imported frozen training authority.
    """
    schema, _digest_value, _keys, _id, _version = _authority(variant)
    selected = schema if schema_override is None else schema_override
    if variant == RUN5B_VARIANT:
        # Run-5B explicitly inherits the entire Run-4B prefix contract; verify
        # that authority independently rather than trusting only its digest
        # recorded inside the Run-5B schema.
        _verify(RUN4B_VARIANT, tuple(R4B.FEATURE_ORDER), R4B.FEATURE_SCHEMA)
    return _verify(variant, order, selected)


__all__ = [
    "RUN4B_VARIANT", "RUN5B_VARIANT", "RUN4B_SCHEMA_SHA256",
    "RUN5B_SCHEMA_SHA256", "FeatureSchemaAuthorityError",
    "feature_schema_sha256",
]
