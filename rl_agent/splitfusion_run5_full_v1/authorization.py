"""Launch authorization for the full Run-5 training (this scope only).

Records Abiodun's 2026-10-06 instruction (full 12-mode Run-5 agent, original continuous-q support, fresh actors,
agreed reward, seeds 17/29/43 to 10,000 updates, original SAC settings, existing evidence only). Global guards are
untouched: ``run4_v1.environment.REGISTERED_CALIBRATION_BINDING_SHA256`` stays None and the corrected candidate's
``APPROVAL.json`` stays absent; transitions are labelled ``FULL_RUN5_MODELED_TRAINING``, never calibration evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import quality as Q
from . import support as S

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SCHEMA = "splitfusion.run5_full.authorization.v1"
SEEDS = (17, 29, 43)
UPDATES = 10_000
EVIDENCE_LABEL = "FULL_RUN5_MODELED_TRAINING"


class AuthorizationError(RuntimeError):
    pass


def source_digest() -> str:
    files = sorted(p for p in PACKAGE.glob("*.py") if not p.name.startswith("test_") and p.name not in
                   ("replay_plot.py", "validate.py", "export_actor.py"))
    return hashlib.sha256(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                                     sort_keys=True).encode()).hexdigest()


def _manifest_ok(binding_dir: Path) -> dict[str, str]:
    manifest = json.loads((binding_dir / "OUTPUT_SHA256.json").read_text())
    for name, digest in manifest.items():
        if hashlib.sha256((binding_dir / name).read_bytes()).hexdigest() != digest:
            raise AuthorizationError(f"binding file drift: {name}")
    return manifest


def issue(path: Path, *, binding_dir: Path, authorized_by: str, authorized_on: str, scope: str) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise AuthorizationError("authorization files are create-only")
    binding_dir = Path(binding_dir)
    document = dict(schema=SCHEMA, authorized_by=authorized_by, authorized_on=authorized_on, scope=scope,
                    binding_dir=str(binding_dir.resolve().relative_to(ROOT)), binding_manifest=_manifest_ok(binding_dir),
                    support_sha256=S.SUPPORT_SHA256, reward_sha256=Q.REWARD_SHA256, source_digest=source_digest(),
                    seeds=list(SEEDS), updates=UPDATES, evidence_label=EVIDENCE_LABEL,
                    global_guards_unchanged=["run4_v1.environment.REGISTERED_CALIBRATION_BINDING_SHA256 is None",
                                             "splitfusion_corrected_latency_env_v1/APPROVAL.json absent"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True))
    return document


def verify(path: Path, *, seed: int, binding_dir: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    binding_dir = Path(binding_dir)
    if document.get("schema") != SCHEMA:
        raise AuthorizationError("foreign authorization")
    if document["source_digest"] != source_digest():
        raise AuthorizationError("training source changed after authorization")
    if document["support_sha256"] != S.SUPPORT_SHA256 or document["reward_sha256"] != Q.REWARD_SHA256:
        raise AuthorizationError("support or reward differs from the authorized definition")
    if str(binding_dir.resolve().relative_to(ROOT)) != document["binding_dir"]:
        raise AuthorizationError("binding directory differs")
    if _manifest_ok(binding_dir) != document["binding_manifest"]:
        raise AuthorizationError("binding manifest differs")
    if seed not in document["seeds"]:
        raise AuthorizationError("seed outside the authorized scope")
    return document
