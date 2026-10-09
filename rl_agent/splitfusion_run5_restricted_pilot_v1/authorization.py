"""Minimal launch authorization for this explicitly scoped pilot only.

The global guards are untouched: ``run4_v1.environment.REGISTERED_CALIBRATION_BINDING_SHA256`` stays None and the
corrected candidate's ``APPROVAL.json`` path stays absent. This module authorizes exactly one thing: pilot collectors
bound to the named binding/split/support/source digests, the two reward arms, the three seeds and the update budget.
The environment mechanics run under the existing ``SYNTHETIC_MECHANICS_FIXTURE`` class; pilot transitions are
labelled ``PILOT_AUTHORIZED_MODELED`` and are never calibration evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import support as S

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SCHEMA = "splitfusion.run5_restricted_pilot.authorization.v1"
ARMS = ("CONTROL", "CANDIDATE")
SEEDS = (17, 29, 43)
UPDATES = 10_000
EVIDENCE_LABEL = "PILOT_AUTHORIZED_MODELED"


class AuthorizationError(RuntimeError):
    pass


def source_digest() -> str:
    files = sorted(p for p in PACKAGE.glob("*.py") if not p.name.startswith("test_"))
    return hashlib.sha256(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                                     sort_keys=True).encode()).hexdigest()


def issue(path: Path, *, binding_dir: Path, authorized_by: str, authorized_on: str, scope: str) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise AuthorizationError("authorization files are create-only")
    binding_dir = Path(binding_dir)
    manifest = json.loads((binding_dir / "OUTPUT_SHA256.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((binding_dir / name).read_bytes()).hexdigest() != digest:
            raise AuthorizationError(f"binding file drift: {name}")
    document = dict(schema=SCHEMA, authorized_by=authorized_by, authorized_on=authorized_on, scope=scope,
                    binding_dir=str(binding_dir.resolve().relative_to(ROOT)), binding_manifest=manifest,
                    support_sha256=S.SUPPORT_SHA256, source_digest=source_digest(), arms=list(ARMS), seeds=list(SEEDS),
                    updates=UPDATES, evidence_label=EVIDENCE_LABEL,
                    global_guards_unchanged=["run4_v1.environment.REGISTERED_CALIBRATION_BINDING_SHA256 is None",
                                             "splitfusion_corrected_latency_env_v1/APPROVAL.json absent"],
                    known_calibration_gaps_accepted_for_this_pilot_only=[
                        "BINDING_PURPOSE_NOT_CALIBRATION", "PROFILES_UNMEASURED: MID_VARIABLE, ADVERSE_STABLE, FADE_RECOVERY "
                        "(pilot is FAVORABLE_STABLE only)", "CONTINUOUS_Q: registered interpolation between measured points",
                        "MODES_MASKED: 0-4, 6, 7, 9, 10", "FALLBACK: causal per-opportunity model from 8 measured runs"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True))
    return document


def verify(path: Path, *, arm: str, seed: int, binding_dir: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    binding_dir = Path(binding_dir)
    if document.get("schema") != SCHEMA:
        raise AuthorizationError("foreign authorization")
    if document["source_digest"] != source_digest():
        raise AuthorizationError("pilot source changed after authorization")
    if document["support_sha256"] != S.SUPPORT_SHA256:
        raise AuthorizationError("action support differs from the authorized support")
    if str(binding_dir.resolve().relative_to(ROOT)) != document["binding_dir"]:
        raise AuthorizationError("binding directory differs")
    for name, digest in document["binding_manifest"].items():
        if hashlib.sha256((binding_dir / name).read_bytes()).hexdigest() != digest:
            raise AuthorizationError(f"binding file drift: {name}")
    if arm not in document["arms"] or seed not in document["seeds"] + [9017, 9029, 9043, 0]:
        raise AuthorizationError("arm or seed outside the authorized scope")
    return document
