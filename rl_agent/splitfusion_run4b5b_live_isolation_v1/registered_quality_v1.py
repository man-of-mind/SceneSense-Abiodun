"""Load the already registered Run-4/5 Q_perc definition by pinned bytes."""

from __future__ import annotations

from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run4_v1.scientific_basis import (
    QUALITY_SOURCE_FILE_SHA256,
    QUALITY_SOURCE_RELATIVE_PATH,
    verify_quality_source,
)
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.quality import (
    load_reward_spec,
)
from rl_agent.splitfusion_hybrid_sac_v1.state_reward_transition_contract import (
    RewardSpecV1,
)


class RegisteredQualityError(RuntimeError):
    """The pinned Q_perc source cannot be loaded exactly."""


def load_registered_quality_spec(repo_root: Path) -> RewardSpecV1:
    """Verify source bytes and return the exact existing RewardSpecV1.

    Only the quality fields of this source are approved for the B variants;
    their operational reward still uses the separate 170-ms contract.  The
    post-run evaluator likewise consumes this object only through the existing
    protocol-v2 Q_perc derivation.
    """

    root = Path(repo_root)
    if not root.is_absolute():
        raise RegisteredQualityError("repo_root must be absolute")
    observed = verify_quality_source(root)
    if observed != QUALITY_SOURCE_FILE_SHA256:  # pragma: no cover - verifier owns it
        raise RegisteredQualityError("registered quality source digest drifted")
    spec = load_reward_spec(
        root / QUALITY_SOURCE_RELATIVE_PATH,
        QUALITY_SOURCE_FILE_SHA256,
    )
    if type(spec) is not RewardSpecV1:
        raise RegisteredQualityError("quality loader returned a foreign type")
    return spec


__all__ = ["RegisteredQualityError", "load_registered_quality_spec"]
