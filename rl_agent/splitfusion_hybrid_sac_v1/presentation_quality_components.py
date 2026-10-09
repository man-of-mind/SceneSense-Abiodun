"""Read-only quality-component join for Hybrid-SAC presentation figures.

Checkpoint evaluators persisted the registered joint quality coordinate
(``q_perc``), but not its ``q_loc`` and ``q_seg`` constituents. This helper
replays only the already-bound empirical quality-surface lookup for each saved
sample/action identity. It does not execute a policy, model, CARLA, CUDA, or a
network service.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

from .corrected_p40_sidecar import load_exact_corrected_p40_sidecar
from .empirical_quality_surface import load_empirical_quality_surface


class PresentationQualityError(RuntimeError):
    """A saved validation row cannot be reconciled to the quality surface."""


def quality_components_for_rows(
    rows: Sequence[Mapping[str, str]], *, project_root: Path
) -> Tuple[Mapping[str, float], ...]:
    """Return ``q_loc``, ``q_seg`` and ``q_perc`` in input-row order."""

    root = Path(project_root).resolve(strict=True)
    sidecar = load_exact_corrected_p40_sidecar(root=root)
    cache: Dict[Tuple[str, int, int], Mapping[str, float]] = {}
    output = []
    with load_empirical_quality_surface(sidecar, project_root=root) as surface:
        for row in rows:
            key = (
                str(row["sample_id"]),
                int(row["executed_mode_id"]),
                int(row["executed_q_e4"]),
            )
            values = cache.get(key)
            if values is None:
                result = surface.query_fit_q_e4(key[0], key[1], key[2])
                derived: Dict[str, float] = {}
                for name in ("q_loc", "q_seg", "q_perc"):
                    component = result.policy.component(name)
                    if not component.valid or component.value is None:
                        raise PresentationQualityError(
                            f"{key}: quality component {name} is unavailable"
                        )
                    value = float(component.value)
                    if not math.isfinite(value):
                        raise PresentationQualityError(
                            f"{key}: quality component {name} is non-finite"
                        )
                    derived[name] = value
                values = derived
                cache[key] = values
            persisted = row.get("q_perc")
            if persisted not in (None, "") and not math.isclose(
                float(persisted), values["q_perc"], rel_tol=0.0, abs_tol=1e-12
            ):
                raise PresentationQualityError(
                    f"{key}: persisted q_perc does not match the bound surface"
                )
            output.append(values)
    return tuple(output)
