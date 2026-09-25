"""Guard the completed, superseded-analysis Run-3 campaign against mutation.

``20260924_131015`` is a finished campaign whose verdict is ``INCONCLUSIVE``.
Run 4 must neither rerun nor reinterpret it.  Its own integrity amendment
records that a *concurrent* process silently rewrote 15 derived files in place
while the campaign was sealing its manifest, so "nobody would edit it" is not
an assumption this repository has earned.  These digests are therefore checked
before Run 4 starts and again after it finishes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

#: Repository root: .../abiodun
ROOT = Path(__file__).resolve().parents[2]

PROTECTED_RUN_RELPATH = (
    "rl_agent/experiments/ue_mcs_backlog_calibration_v1/20260924_131015"
)

#: Recorded 2026-09-24 during the Run-4 Phase-A audit, before anything was
#: created.  Any difference is a stop condition, never a thing to "repair".
PROTECTED_SHA256: Mapping[str, str] = {
    "manifest.json":
        "219db3ed7dacc975ab3bfc56164ad527724989ba12d6a1f0d83f97af382595a1",
    "INTEGRITY_AMENDMENT_DERIVED_V1_SUPERSEDED.md":
        "d68c665d061452fe663d227b6658ed3650244a9af963539e7ed6f891f1b5a00b",
    "VERIFIER_INPUT_MANIFEST_V2.json":
        "c2b796bcf8991891f79d407751c5edd0c8b5774d425513f82bdd528ad4eba640",
    "analysis_v2.json":
        "4c48df847a1be13e29ca325c886412584de6b90692d8063090245867e71be999",
    "decisions_v2.csv":
        "cfddd8b9623b055b80a220f48f71e1619c6e8689ce8caf20c2174da95b7af799",
}

#: The Run-3 verdict, restated so Run 4 can never be read as upgrading it.
PROTECTED_VERDICT = "INCONCLUSIVE"
PROTECTED_BOUND_RESULT = "4/7 scientific checks; 12/13 structural gates"


class ProtectedEvidenceError(RuntimeError):
    """Raised when protected Run-3 evidence is missing or altered."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(repo_root: Path | None = None) -> dict[str, Any]:
    """Recompute every protected digest. Pure: reads only, writes nothing."""
    root = (repo_root or ROOT) / PROTECTED_RUN_RELPATH
    files: dict[str, Any] = {}
    for name, expected in sorted(PROTECTED_SHA256.items()):
        path = root / name
        if not path.is_file():
            files[name] = {"present": False, "expected_sha256": expected,
                           "observed_sha256": None, "unchanged": False}
            continue
        observed = sha256_file(path)
        files[name] = {"present": True, "expected_sha256": expected,
                       "observed_sha256": observed,
                       "size_bytes": path.stat().st_size,
                       "unchanged": observed == expected}
    return {
        "protected_run": PROTECTED_RUN_RELPATH,
        "verdict_preserved_as": PROTECTED_VERDICT,
        "bound_result": PROTECTED_BOUND_RESULT,
        "files": files,
        "all_unchanged": all(item["unchanged"] for item in files.values()),
    }


def require_unchanged(stage: str, repo_root: Path | None = None) -> dict[str, Any]:
    """Audit and refuse to continue if anything moved."""
    report = audit(repo_root)
    if not report["all_unchanged"]:
        broken = sorted(name for name, item in report["files"].items()
                        if not item["unchanged"])
        raise ProtectedEvidenceError(
            f"protected Run-3 evidence changed at stage {stage!r}: "
            f"{', '.join(broken)}. Refusing to continue; do not repair.")
    report["stage"] = stage
    return report


def assert_outside_protected_run(path: Path, repo_root: Path | None = None) -> None:
    """Refuse any write target inside the protected campaign."""
    protected = ((repo_root or ROOT) / PROTECTED_RUN_RELPATH).resolve()
    candidate = path.resolve()
    if candidate == protected or protected in candidate.parents:
        raise ProtectedEvidenceError(
            f"{candidate} is inside protected run {protected}; refusing to write")


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import json
    import sys
    result = audit()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["all_unchanged"] else 1)
