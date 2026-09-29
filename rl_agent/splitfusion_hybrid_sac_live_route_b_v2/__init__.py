"""Additive Run-4 live-qualification seams for the selected Hybrid-SAC actor.

``RUN4_FROZEN_POLICY_LIVE_QUALIFICATION_V2__NOT_ONLINE_LEARNING``

This package prepares the post-hoc Run-4 candidate (seed 43, update 10,000)
for a bounded 300-frame Route-B live qualification.  It is deployment
validation, not learning: no optimizer, critic, replay buffer or parameter
update is constructed, and the actor is loaded read-only.

Every pinned production module is imported unchanged.  Nothing in
``splitfusion_hybrid_sac_live_route_b_v1`` (the Run-3 package) is modified or
reused as live authority.

Importing this package performs no filesystem, network, CARLA, OAI, Docker or
CUDA work.  Submodules are imported explicitly so that importing the package
never pulls in ``torch``.

Phases: 0 reconciliation, 1 frozen actor, 2 live state and UE telemetry,
3 continuous (mode_id, q_e4) execution, 4 action hold and 170-ms reward
ticket, 5 composed offline readiness.  See each ``PHASE*_REPORT.md``.
"""

from __future__ import annotations

PACKAGE_LABEL = "RUN4_FROZEN_POLICY_LIVE_QUALIFICATION_V2__NOT_ONLINE_LEARNING"

__all__ = ["PACKAGE_LABEL"]
