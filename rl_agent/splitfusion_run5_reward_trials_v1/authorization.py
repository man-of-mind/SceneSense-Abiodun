"""Launch authorization for the bounded Run-5 reward comparison (this scope only).

Records Abiodun's 2026-10-06 instruction: exactly three reward arms (C0/C1/C2, fixed coefficients), seeds 17/29/43,
10,000 updates each (nine runs, no sweep), one frozen latency binding (binding_02, TRAIN_GCV2 pool) for every arm,
original Run-5 SAC/warm-up/support/state, no live capture or deployment. Global guards untouched:
``run4_v1.environment.REGISTERED_CALIBRATION_BINDING_SHA256`` stays None, the corrected candidate's APPROVAL.json
stays absent; transitions are labelled ``RUN5_REWARD_TRIAL_MODELED``, never calibration evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rl_agent.splitfusion_run5_full_v1 import support as S

from . import rewards as R

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
SCHEMA = "splitfusion.run5_reward_trials.authorization.v1"
ARMS = R.ARMS
SEEDS = (17, 29, 43)
UPDATES = 10_000
EVIDENCE_LABEL = "RUN5_REWARD_TRIAL_MODELED"
TRAINING_SOURCES = ("__init__.py", "rewards.py", "catalog.py", "evidence.py", "latency.py", "collector.py",
                    "trainer.py", "campaign.py", "authorization.py")


class AuthorizationError(RuntimeError):
    pass


def source_digest() -> str:
    return hashlib.sha256(json.dumps({n: hashlib.sha256((PACKAGE / n).read_bytes()).hexdigest()
                                      for n in TRAINING_SOURCES}, sort_keys=True).encode()).hexdigest()


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
                    support_sha256=S.SUPPORT_SHA256, arms={a: dict(equation=R.EQUATIONS[a], reward_sha256=R.reward_sha256(a))
                                                           for a in ARMS},
                    source_digest=source_digest(), seeds=list(SEEDS), updates=UPDATES, training_pool="TRAIN_GCV2",
                    scene_split="fit", evidence_label=EVIDENCE_LABEL,
                    global_guards_unchanged=["run4_v1.environment.REGISTERED_CALIBRATION_BINDING_SHA256 is None",
                                             "splitfusion_corrected_latency_env_v1/APPROVAL.json absent"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True))
    return document


def verify(path: Path, *, arm: str, seed: int, binding_dir: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text())
    binding_dir = Path(binding_dir)
    if document.get("schema") != SCHEMA:
        raise AuthorizationError("foreign authorization")
    if document["source_digest"] != source_digest():
        raise AuthorizationError("training source changed after authorization")
    if document["support_sha256"] != S.SUPPORT_SHA256:
        raise AuthorizationError("support differs from the authorized definition")
    if arm not in document["arms"] or document["arms"][arm]["reward_sha256"] != R.reward_sha256(arm):
        raise AuthorizationError("arm or reward differs from the authorized definition")
    if str(binding_dir.resolve().relative_to(ROOT)) != document["binding_dir"]:
        raise AuthorizationError("binding directory differs")
    if _manifest_ok(binding_dir) != document["binding_manifest"]:
        raise AuthorizationError("binding manifest differs")
    if seed not in document["seeds"]:
        raise AuthorizationError("seed outside the authorized scope")
    return document
