#!/usr/bin/env python3
"""Durable, one-use authorization for the single live production-domain capture.

The grant is bound to the exact output path, execution commit, config digest
and source inventory.  Consumption is an ``O_EXCL`` create plus ``fsync`` and
happens before any RAN/container/network mutation, so a crash spends the grant
rather than silently making it reusable.  Exactly one live attempt is
authorized.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import config as CFG
from . import contract as C


AUTH_SCHEMA = "scenesense.ue_production_queue_capture_authorization.v1"
STAGE = "production_queue_capture_v1"
TOKEN = "AUTHORIZE_PRODUCTION_QUEUE_CAPTURE_V1"
MAX_LIFETIME_SECONDS = 28_800
CONSUMPTION_DIRNAME = ".authorization_consumed"
REQUIRED_FIELDS = {
    "schema", "stage", "token", "grant_id", "granted_by", "granted_utc",
    "expires_utc", "execution_head", "source_inventory_sha256",
    "config_sha256", "contract_sha256", "output_path",
}


class AuthorizationError(RuntimeError):
    """The grant is stale, reusable, ambiguous, or not bound to this run."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationError(message)


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(repo_root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    require(completed.returncode == 0,
            f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def execution_identity(repo_root: Path = C.ROOT) -> dict[str, Any]:
    return {
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "package_status": _git(
            repo_root, "status", "--porcelain", "--", C.PACKAGE_RELPATH),
    }


def _parse_utc(value: str, label: str) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"),
            f"{label} must be a UTC instant ending in Z")
    return datetime.fromisoformat(value[:-1]).replace(tzinfo=timezone.utc)


def validate_grant(
    grant: Mapping[str, Any], *, output_path: Path, repo_root: Path = C.ROOT,
) -> dict[str, Any]:
    require(set(grant) == REQUIRED_FIELDS,
            f"grant fields drifted: {sorted(set(grant) ^ REQUIRED_FIELDS)}")
    require(grant["schema"] == AUTH_SCHEMA, "grant schema drifted")
    require(grant["stage"] == STAGE, "grant stage drifted")
    require(grant["token"] == TOKEN, "grant token drifted")
    require(isinstance(grant["grant_id"], str) and grant["grant_id"],
            "grant id must be a non-empty string")
    granted = _parse_utc(grant["granted_utc"], "granted_utc")
    expires = _parse_utc(grant["expires_utc"], "expires_utc")
    now = datetime.now(timezone.utc)
    require(granted <= now, "grant is not yet valid")
    require(now < expires, "grant has expired")
    lifetime = (expires - granted).total_seconds()
    require(0 < lifetime <= MAX_LIFETIME_SECONDS,
            f"grant lifetime {lifetime}s exceeds the registered maximum")
    identity = execution_identity(repo_root)
    require(grant["execution_head"] == identity["head"],
            "grant is not bound to the current execution commit")
    require(grant["contract_sha256"] == C.CONTRACT_SHA256,
            "grant is not bound to the frozen contract document")
    config_sha = C.sha256_file(CFG.DEFAULT_CONFIG)
    require(grant["config_sha256"] == config_sha,
            "grant is not bound to this capture config")
    inventory = CFG.source_inventory(repo_root)
    require(grant["source_inventory_sha256"] == inventory["inventory_sha256"],
            "grant is not bound to this source inventory")
    require(Path(grant["output_path"]) == output_path,
            "grant is not bound to this exact output path")
    return {"identity": identity, "inventory": inventory,
            "config_sha256": config_sha}


def consume(
    grant_path: Path, *, output_path: Path, repo_root: Path = C.ROOT,
) -> dict[str, Any]:
    """Validate and durably spend the grant before any live mutation."""
    grant = json.loads(Path(grant_path).read_text(encoding="utf-8"))
    context = validate_grant(grant, output_path=output_path, repo_root=repo_root)
    directory = Path(grant_path).parent / CONSUMPTION_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{grant['grant_id']}.json"
    record = {
        "schema": AUTH_SCHEMA + ".consumption",
        "grant_id": grant["grant_id"], "stage": STAGE,
        "consumed_utc": datetime.now(timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "consumption_uuid": str(uuid.uuid4()),
        "output_path": str(output_path),
        "execution_head": context["identity"]["head"],
        "source_inventory_sha256":
            context["inventory"]["inventory_sha256"],
        "config_sha256": context["config_sha256"],
        "contract_sha256": C.CONTRACT_SHA256,
    }
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise AuthorizationError(
            f"grant {grant['grant_id']} was already consumed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return record
