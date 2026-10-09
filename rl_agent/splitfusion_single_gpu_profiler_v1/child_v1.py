#!/usr/bin/env python3
"""Phase-6 child entry point with the profiler seams (UE process).

Runs ``phase6_live_child_nobuild_v2.main`` with the same argv.  The profiler
configuration is the create-only ``PROFILER_CONFIG.json`` the runner writes
into the output root (``<attempt-dir>/../..``).  If it is absent, or
``enabled`` is false, nothing is installed and the child is exactly the
Phase-6 no-build child.

When enabled, the profiler seams are installed immediately after the unchanged
Phase-6 seams (``install_run4_seams_nobuild``), before the map, edge, CARLA
client or route exist.  The policy identity recorded is read from the binding
named in the configuration, at start and again at close.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import identity_v1 as IV
from . import recorder_v1 as RV

CONFIG_NAME = "PROFILER_CONFIG.json"
CONFIG_SCHEMA = "scenesense.single_gpu_profiler.config.v1"


class ProfilerConfigError(RuntimeError):
    pass


def attempt_dir_from_argv(argv: Sequence[str]) -> Optional[Path]:
    items = list(argv)
    for index, item in enumerate(items):
        if item == "--attempt-dir" and index + 1 < len(items):
            return Path(items[index + 1])
    return None


def load_config(attempt_dir: Optional[Path]) -> Optional[dict[str, Any]]:
    if attempt_dir is None:
        return None
    path = Path(attempt_dir).resolve().parent.parent / CONFIG_NAME
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ProfilerConfigError(f"foreign profiler config schema in {path}")
    return config if config.get("enabled") is True else None


def policy_identity(config: dict[str, Any], root: Path) -> dict[str, Any]:
    policy = config.get("policy") or {}
    binding = policy.get("binding_path")
    if not binding:
        return {"binding": None}
    weights = policy.get("weights_path")
    return IV.binding_identity(root / binding, None if not weights else root / weights)


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - live
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
        phase6_live_child_nobuild_v2 as NB,
    )

    from . import seams_ue_v1 as SU

    args = list(sys.argv[1:] if argv is None else argv)
    attempt_dir = attempt_dir_from_argv(args)
    config = load_config(attempt_dir)
    if config is None:
        return NB.main(args)
    root = Path(__file__).resolve().parents[2]
    identity = policy_identity(config, root)
    rec = RV.ProfileRecorderV1(Path(attempt_dir) / "profiler" / "ue.profile.jsonl",
                               role="ue", run_label=str(config.get("run_label", "")),
                               identity={"policy": identity,
                                         "profiler_config_sha256": config.get("sha256")},
                               gauge_interval_s=float(config.get("gauge_interval_s", 0.1)),
                               flush_interval_s=float(config.get("flush_interval_s", 0.2)))
    rec.start()
    RV.activate(rec)
    original = NB.install_run4_seams_nobuild

    def install(campaign: Any, *, attempt_dir: Path, **kwargs: Any) -> Any:
        seams = original(campaign, attempt_dir=attempt_dir, **kwargs)
        SU.install_ue_seams(rec, attempt_dir=Path(attempt_dir), policy_identity=identity)
        return seams

    NB.install_run4_seams_nobuild = install
    try:
        return NB.main(args)
    finally:
        try:
            rec.note("policy_identity_at_close", **policy_identity(config, root))
        finally:
            rec.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
