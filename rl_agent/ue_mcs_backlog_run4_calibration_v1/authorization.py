"""Durable, one-use authorization for the live calibration campaign.

The grant is operator-created after the implementation commit.  It is bound to
the exact output path, execution commit and parent, config, source inventory,
and sealed amendment.  Consumption is an ``O_EXCL`` create plus ``fsync`` and
happens before any RAN/container/network mutation.  A crash therefore spends
the grant; it never silently makes it reusable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import contract as C


AUTH_SCHEMA = "scenesense.ue_mcs_backlog_run4_calibration_authorization.v1"
STAGE = "run4_physical_queue_calibration"
TOKEN = "AUTHORIZE_RUN4_PHYSICAL_QUEUE_CALIBRATION"
MAX_LIFETIME_SECONDS = 21_600
CONSUMPTION_DIRNAME = ".authorization_consumed"
REQUIRED_FIELDS = {
    "schema", "stage", "token", "grant_id", "granted_by", "granted_utc",
    "expires_utc", "execution_head", "execution_parent_head",
    "source_inventory_sha256", "config_sha256", "amendment_sha256",
    "output_path",
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
    head = _git(repo_root, "rev-parse", "HEAD")
    parent = _git(repo_root, "rev-parse", "HEAD^")
    status = _git(repo_root, "status", "--porcelain", "--", C.PACKAGE_RELPATH)
    return {"head": head, "parent_head": parent,
            "package_sources_clean": not bool(status),
            "package_status": status.splitlines() if status else []}


def _parse_utc(value: Any, *, field: str) -> datetime:
    require(type(value) is str and bool(value), f"{field} must be a UTC timestamp")
    require(value.endswith("Z"), f"{field} must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise AuthorizationError(f"{field} is not ISO-8601: {value!r}") from exc
    require(parsed.tzinfo is not None, f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def authorization_template(
    output_path: Path, *, granted_by: str,
    granted_utc: str, expires_utc: str, repo_root: Path = C.ROOT,
) -> dict[str, Any]:
    """Return an operator-reviewable template; never write or consume it."""
    identity = execution_identity(repo_root)
    require(identity["package_sources_clean"],
            "commit the package before creating an authorization")
    inventory = C.source_inventory(repo_root)
    return {
        "schema": AUTH_SCHEMA, "stage": STAGE, "token": TOKEN,
        "grant_id": str(uuid.uuid4()), "granted_by": granted_by,
        "granted_utc": granted_utc, "expires_utc": expires_utc,
        "execution_head": identity["head"],
        "execution_parent_head": identity["parent_head"],
        "source_inventory_sha256": inventory["inventory_sha256"],
        "config_sha256": C.sha256_file(repo_root / C.CONFIG_RELPATH),
        "amendment_sha256": C.AMENDMENT_SHA256,
        "output_path": str(output_path.resolve()),
    }


def _load(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"authorization file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorizationError(f"cannot parse authorization: {exc}") from exc
    require(type(value) is dict, "authorization must be a JSON object")
    unknown = set(value) - REQUIRED_FIELDS
    missing = REQUIRED_FIELDS - set(value)
    require(not unknown, f"authorization has unknown fields: {sorted(unknown)}")
    require(not missing, f"authorization lacks fields: {sorted(missing)}")
    return value, C.sha256_file(path)


def validate_authorization(
    path: Path, output_path: Path, *, repo_root: Path = C.ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    value, digest = _load(path)
    require(value["schema"] == AUTH_SCHEMA and value["stage"] == STAGE
            and value["token"] == TOKEN, "authorization identity mismatch")
    try:
        grant_id = str(uuid.UUID(str(value["grant_id"])))
    except (ValueError, AttributeError) as exc:
        raise AuthorizationError("grant_id must be a canonical UUID") from exc
    require(grant_id == value["grant_id"], "grant_id is not canonical")
    require(type(value["granted_by"]) is str and value["granted_by"].strip(),
            "granted_by must name the authorizing operator")
    granted = _parse_utc(value["granted_utc"], field="granted_utc")
    expires = _parse_utc(value["expires_utc"], field="expires_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    lifetime = (expires - granted).total_seconds()
    require(0 < lifetime <= MAX_LIFETIME_SECONDS,
            f"authorization lifetime must be in (0,{MAX_LIFETIME_SECONDS}] seconds")
    require(granted <= current <= expires,
            "authorization is not currently valid (not-yet-valid or expired)")
    require(value["output_path"] == str(output_path.resolve()),
            "authorization is bound to a different output path")
    require(not output_path.exists(), "create-only output path already exists")
    identity = execution_identity(repo_root)
    require(identity["package_sources_clean"],
            f"package sources are dirty: {identity['package_status']}")
    require(value["execution_head"] == identity["head"],
            "authorization execution HEAD differs from current HEAD")
    require(value["execution_parent_head"] == identity["parent_head"],
            "authorization execution parent differs from current parent")
    inventory = C.source_inventory(repo_root)
    require(value["source_inventory_sha256"] == inventory["inventory_sha256"],
            "authorization source-inventory binding drifted")
    require(value["config_sha256"] == C.sha256_file(repo_root / C.CONFIG_RELPATH),
            "authorization config binding drifted")
    require(value["amendment_sha256"] == C.AMENDMENT_SHA256,
            "authorization amendment binding drifted")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)),
            "authorization digest is invalid")
    return {**value, "authorization_sha256": digest,
            "validated_utc": current.isoformat().replace("+00:00", "Z")}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def consume_authorization(
    path: Path, output_path: Path, *, output_root: Path,
    repo_root: Path = C.ROOT, now: datetime | None = None,
) -> dict[str, Any]:
    """Validate then durably spend exactly one grant before live mutation."""
    output_path = output_path.resolve()
    output_root = output_root.resolve()
    require(output_path.parent == output_root,
            "authorized run must be one direct create-only child of output root")
    validated = validate_authorization(
        path, output_path, repo_root=repo_root, now=now)
    consumed_dir = output_root / CONSUMPTION_DIRNAME
    consumed_dir.mkdir(parents=True, exist_ok=True)
    for directory in (output_root.parent, output_root, consumed_dir):
        _fsync_directory(directory)
    marker = consumed_dir / f"{validated['grant_id']}.json"
    record = {
        "schema": "scenesense.ue_mcs_backlog_run4_calibration_consumption.v1",
        "grant_id": validated["grant_id"],
        "authorization_sha256": validated["authorization_sha256"],
        "output_path": str(output_path),
        "execution_head": validated["execution_head"],
        "source_inventory_sha256": validated["source_inventory_sha256"],
        "consumed_utc": (now or datetime.now(timezone.utc)).astimezone(
            timezone.utc).isoformat().replace("+00:00", "Z"),
        "semantics": "DURABLE_ONE_USE_PRE_LIVE_MUTATION_NO_AUTOMATIC_REFUND",
    }
    payload = (json.dumps(record, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise AuthorizationError(
            f"grant {validated['grant_id']} was already consumed at {marker}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(consumed_dir)
    except Exception:
        # Never delete a partial marker: conservative one-use semantics mean a
        # failed durable write still requires explicit operator review/new grant.
        raise
    return {**validated, "consumption_marker": str(marker),
            "consumption_marker_sha256": C.sha256_file(marker)}


def lineage_record(consumed: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "scenesense.ue_mcs_backlog_run4_calibration_lineage.v1",
        "stage": STAGE,
        "grant_id": consumed["grant_id"],
        "authorization_sha256": consumed["authorization_sha256"],
        "consumption_marker": consumed["consumption_marker"],
        "consumption_marker_sha256": consumed["consumption_marker_sha256"],
        "execution_head": consumed["execution_head"],
        "execution_parent_head": consumed["execution_parent_head"],
        "source_inventory_sha256": consumed["source_inventory_sha256"],
        "output_path": consumed["output_path"],
        "one_use_rule": (
            "The expiring operator grant is bound to one exact output path and "
            "is durably consumed before any live mutation. Failure never refunds it."
        ),
    }
