"""One-attempt authorization and lineage for Run 4.

The registered retry policy is one attempt per campaign. A failed attempt is
preserved in place; only a proven engineering defect may be repaired, and the
repair runs as a new timestamped root under a single code revision. That policy
is worth nothing if it is only prose, so it is enforced here: a second attempt
cannot start unless an explicit authorization record names the attempt it
supersedes and the defect that was repaired.

Lineage is recorded both ways -- each run names its parent, and the
authorization names the child it permits -- so an orphan run, or a run spliced
onto a different code revision, is detectable after the fact.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]

SCIENTIFIC_STAGE = "near_capacity_scientific_cells"
CAPACITY_STAGE = "near_capacity_capacity_qualification"
STAGES = (CAPACITY_STAGE, SCIENTIFIC_STAGE)

#: Written by the operator to permit a run. Never created by this code.
AUTHORIZATION_FILENAME = "AUTHORIZATION.json"

MAX_ATTEMPTS_WITHOUT_NEW_AUTHORIZATION = 1


class AuthorizationError(RuntimeError):
    """Raised when a run is not authorized, or would violate one-attempt."""


@dataclass(frozen=True)
class Authorization:
    stage: str
    token: str
    granted_by: str
    granted_utc: str
    supersedes: str | None
    repaired_defect: str | None

    @classmethod
    def load(cls, path: Path) -> "Authorization":
        data = json.loads(path.read_text())
        unknown = set(data) - {
            "stage", "token", "granted_by", "granted_utc", "supersedes",
            "repaired_defect"}
        if unknown:
            raise AuthorizationError(
                f"authorization has unknown field(s) {sorted(unknown)}; "
                f"refusing rather than ignoring them")
        missing = {"stage", "token", "granted_by", "granted_utc"} - set(data)
        if missing:
            raise AuthorizationError(
                f"authorization is missing required field(s) {sorted(missing)}")
        return cls(stage=str(data["stage"]), token=str(data["token"]),
                   granted_by=str(data["granted_by"]),
                   granted_utc=str(data["granted_utc"]),
                   supersedes=data.get("supersedes"),
                   repaired_defect=data.get("repaired_defect"))


def source_commit(repo_root: Path | None = None) -> dict[str, Any]:
    """The exact starting commit and whether the tree was dirty."""
    root = repo_root or ROOT
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=str(root), text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL).stdout.strip()
    status = git("status", "--porcelain")
    owned_prefix = "rl_agent/ue_mcs_backlog_near_capacity_v1/"
    dirty = [line[3:] for line in status.splitlines() if line[3:].strip()]
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_paths": dirty,
        "dirty_outside_this_task": [p for p in dirty
                                    if not p.startswith(owned_prefix)],
        "tree_clean_for_this_task": not any(
            p.startswith(owned_prefix) for p in dirty),
    }


def existing_attempts(output_root: Path) -> list[str]:
    """Completed or partial attempt roots already present. Never deleted."""
    if not output_root.is_dir():
        return []
    return sorted(child.name for child in output_root.iterdir()
                  if child.is_dir() and not child.name.startswith("."))


def require_authorization(
    stage: str, output_root: Path, *, expected_token: str,
    repo_root: Path | None = None, require_clean_sources: bool = True,
) -> dict[str, Any]:
    """Refuse unless this attempt is explicitly authorized.

    The authorization lives beside the output root, not inside the run it
    permits, so a run cannot authorize itself.
    """
    if stage not in STAGES:
        raise AuthorizationError(f"unknown stage {stage!r}")

    # Looked for beside the campaign root, so a run cannot authorize itself.
    path = output_root.parent / AUTHORIZATION_FILENAME
    if not path.is_file():
        raise AuthorizationError(
            f"stage {stage!r} is not authorized: no {path} . Phase B requires an "
            f"explicit operator authorization; this code never creates one.")

    auth = Authorization.load(path)
    if auth.stage != stage:
        raise AuthorizationError(
            f"authorization is for stage {auth.stage!r}, not {stage!r}")
    if auth.token != expected_token:
        raise AuthorizationError(
            f"authorization token {auth.token!r} != expected {expected_token!r}")

    prior = existing_attempts(output_root)
    if len(prior) >= MAX_ATTEMPTS_WITHOUT_NEW_AUTHORIZATION:
        if not auth.supersedes:
            raise AuthorizationError(
                f"{len(prior)} attempt(s) already exist ({prior}) and the "
                f"authorization does not name one it supersedes. One attempt per "
                f"campaign; a repair needs a new authorization naming the failed "
                f"attempt and the proven defect.")
        if auth.supersedes not in prior:
            raise AuthorizationError(
                f"authorization supersedes {auth.supersedes!r}, which is not an "
                f"existing attempt {prior}")
        if not auth.repaired_defect:
            raise AuthorizationError(
                "a superseding authorization must state the proven engineering "
                "defect that was repaired")

    commit = source_commit(repo_root)
    if require_clean_sources and not commit["tree_clean_for_this_task"]:
        raise AuthorizationError(
            "this task's own sources are dirty; a scientific run must execute a "
            "single committed code revision. Commit or revert "
            f"{[p for p in commit['dirty_paths'] if 'near_capacity' in p]}")

    return {
        "stage": stage,
        "authorization_path": str(path),
        "granted_by": auth.granted_by,
        "granted_utc": auth.granted_utc,
        "supersedes": auth.supersedes,
        "repaired_defect": auth.repaired_defect,
        "prior_attempts": prior,
        "source_commit": commit,
    }


def lineage_record(
    stage: str, *, run_id: str, parent: Mapping[str, Any] | None,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """The lineage this run writes into its own root."""
    return {
        "stage": stage,
        "run_id": run_id,
        "parent": dict(parent) if parent else None,
        "authorization": dict(authorization),
        "one_attempt_policy": (
            "One attempt per campaign. A failed attempt is preserved in place. "
            "A repair requires a new authorization naming the superseded attempt "
            "and the proven defect, and runs as a new timestamped root under a "
            "single committed code revision. Cells are never spliced across "
            "revisions and no cell is rerun into an existing root."),
    }
