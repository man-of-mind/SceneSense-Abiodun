"""Distinct one-frame request contract for split-host engineering only.

The frozen 300-frame qualification request is intentionally untouched.  This
schema cannot be confused with it: it requires budget one, an explicit
engineering purpose and the actual UE tunnel ACK endpoint ``10.0.0.2:51014``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping

from . import b_edge_process_v1 as E
from . import live_adapters_v1 as L


SCHEMA = "scenesense.splitfusion.run4b5b.edge_engineering_request.v2"
PURPOSE = "ONE_FRAME_SPLIT_HOST_ENGINEERING"
CLAIM_SCOPE = "ENGINEERING_HANDSHAKE_ONLY__NOT_QUALIFICATION"
TRANSMITTED_BUDGET = 1
ACK_RECEIVER_HOST = "10.0.0.2"
ACK_RECEIVER_PORT = 51014


class EngineeringRequestError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EngineeringRequestError(message)


def _canonical(raw: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(dict(raw), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise EngineeringRequestError(
            "engineering request is not canonicalizable") from exc


def decode_and_validate(encoded: str) -> dict[str, Any]:
    _require(type(encoded) is str and bool(encoded), "request is empty")
    try:
        padding = "=" * (-len(encoded) % 4)
        payload = base64.b64decode(encoded + padding, altchars=b"-_",
                                   validate=True)
        raw = json.loads(payload.decode("ascii"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise EngineeringRequestError(
            "request is not canonical urlsafe-base64 JSON") from exc
    fields = {
        "schema", "purpose", "claim_scope", "role", "run_id", "variant",
        "config_binding_sha256", "actor_boundary_sha256",
        "feature_schema_sha256", "transmitted_budget", "deadline_ns",
        "ack_semantics", "postrun_semantics", "clock_domain", "split_host",
        "output_root", "evidence_root", "actor_manifest_path",
        "remote_attempt_root", "required_authority_modules",
        "old_live_quality_runtime_permitted",
    }
    _require(type(raw) is dict and set(raw) == fields,
             "engineering request fields are incomplete or foreign")
    _require(_canonical(raw) == payload, "engineering request is noncanonical")
    _require(raw["schema"] == SCHEMA, "engineering request schema drift")
    _require(raw["purpose"] == PURPOSE, "engineering purpose drift")
    _require(raw["claim_scope"] == CLAIM_SCOPE,
             "engineering claim scope drift")
    _require(raw["role"] == E.ROLE, "request is not for the CN/edge role")
    _require(type(raw["run_id"]) is str and bool(raw["run_id"]),
             "run_id is empty")
    _require(raw["variant"] in {item.value for item in L.ActorVariant},
             "unknown B actor variant")
    for field in ("config_binding_sha256", "actor_boundary_sha256",
                  "feature_schema_sha256"):
        value = raw[field]
        _require(type(value) is str and len(value) == 64
                 and all(ch in "0123456789abcdef" for ch in value),
                 f"{field} is not a lowercase SHA-256")
    _require(raw["transmitted_budget"] == TRANSMITTED_BUDGET,
             "engineering budget must be exactly one")
    _require(raw["deadline_ns"] == E.DEADLINE_NS,
             "operational deadline drift")
    _require(raw["ack_semantics"] == E.ACK_SEMANTICS,
             "operational ACK semantics drift")
    _require(raw["postrun_semantics"] == E.POSTRUN_SEMANTICS,
             "post-run semantics drift")
    _require(raw["clock_domain"] == E.CLOCK_DOMAIN, "clock domain drift")
    _require(raw["output_root"] is None and raw["evidence_root"] is None
             and raw["actor_manifest_path"] is None,
             "UE-only paths are forbidden in the edge request")
    _require(raw["old_live_quality_runtime_permitted"] is False,
             "legacy live-quality runtime was enabled")
    _require(raw["required_authority_modules"] == list(E.REQUIRED_AUTHORITIES),
             "runtime authority list drift")
    split = raw["split_host"]
    split_fields = {"carla_host", "ue_host", "cn_host", "edge_host",
                    "ext_dn_host", "ack_receiver_host", "ack_receiver_port"}
    _require(type(split) is dict and set(split) == split_fields,
             "split-host fields are incomplete or foreign")
    _require(split["carla_host"] == split["ue_host"] == E.LOCAL_HOST,
             "CARLA/UE host drift")
    _require(split["cn_host"] == split["edge_host"]
             == split["ext_dn_host"] == E.REMOTE_HOST,
             "CN/edge/ext-DN host drift")
    _require(split["ack_receiver_host"] == ACK_RECEIVER_HOST
             and split["ack_receiver_port"] == ACK_RECEIVER_PORT,
             "engineering ACK must target the exact UE tunnel endpoint")
    attempt = raw["remote_attempt_root"]
    _require(type(attempt) is str and Path(attempt).is_absolute()
             and str(attempt) not in {"/", "/tmp"}
             and ".." not in Path(attempt).parts,
             "remote attempt root is unsafe")
    return dict(raw)
