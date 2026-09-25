"""Re-derive the per-channel uplink capacity from Run-3 evidence, read-only.

The Run-3 preregistration sited its load tiers against "a ~6 Mbps uplink".
Run-3's own measurements contradict that figure, and Run 4's tier placement
depends on getting it right, so the derivation is code rather than a remembered
number.

Method: over each consecutive decision pair inside one block, service equals
``payload_enqueued - (backlog_next - backlog_now)`` per unit time. Restricted to
intervals where the queue was *already* deep (> ``MIN_BACKLOG_BYTES``) and both
endpoints sit below ``CEILING_FRACTION`` of the observed ceiling, the UE was
continuously backlogged, so the measured rate is the link's capacity rather than
the offered load, and neither endpoint is censored by the buffer limit.

This module only reads the protected campaign. It writes nothing.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rl_agent.ue_mcs_backlog_near_capacity_v1 import protected_evidence as PE

ROOT = PE.ROOT
DECISIONS_RELPATH = f"{PE.PROTECTED_RUN_RELPATH}/decisions_v2.csv"

#: Observed RLC buffer ceiling in the Run-3 campaign.
CEILING_BYTES = 49_984_583
CEILING_FRACTION = 0.85
MIN_BACKLOG_BYTES = 2_000_000
MIN_DT_S = 0.05
MAX_DT_S = 0.50


def rederive(repo_root: Path | None = None) -> dict[str, Any]:
    """Return per-channel capacity percentiles. Reads only; never writes."""
    root = repo_root or ROOT
    PE.require_unchanged("capacity_rederivation", root)
    path = root / DECISIONS_RELPATH

    blocks: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            blocks[(row["cell_id"], row["block_index"])].append(row)

    samples: dict[str, list[float]] = defaultdict(list)
    for rows in blocks.values():
        rows.sort(key=lambda r: int(r["decision_index"]))
        profile = rows[0]["profile_id"]
        payload = int(rows[0]["payload_bytes"])
        for now, nxt in zip(rows, rows[1:]):
            try:
                backlog = int(now["pre_enqueue_backlog_bytes"])
                backlog_next = int(nxt["pre_enqueue_backlog_bytes"])
            except (ValueError, KeyError):
                continue  # missing stays missing
            if backlog < MIN_BACKLOG_BYTES:
                continue
            if max(backlog, backlog_next) > CEILING_FRACTION * CEILING_BYTES:
                continue
            dt = (int(nxt["decision_monotonic_ns"])
                  - int(now["decision_monotonic_ns"])) / 1e9
            if not MIN_DT_S < dt <= MAX_DT_S:
                continue
            samples[profile].append(
                (payload - (backlog_next - backlog)) / dt * 8 / 1e6)

    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {
        "source": DECISIONS_RELPATH,
        "method": ("service = (payload - delta_backlog)/dt over continuously "
                   "backlogged, uncensored intervals"),
        "filters": {"min_backlog_bytes": MIN_BACKLOG_BYTES,
                    "ceiling_bytes": CEILING_BYTES,
                    "ceiling_fraction": CEILING_FRACTION,
                    "dt_s": [MIN_DT_S, MAX_DT_S]},
        "capacity_mbps": {
            profile: {"p10": round(pct(values, 0.10), 2),
                      "p50": round(statistics.median(values), 2),
                      "p90": round(pct(values, 0.90), 2),
                      "n": len(values)}
            for profile, values in sorted(samples.items())},
    }


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import json
    print(json.dumps(rederive(), indent=2))
