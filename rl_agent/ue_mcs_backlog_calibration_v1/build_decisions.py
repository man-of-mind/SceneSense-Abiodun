#!/usr/bin/env python3
"""Turn one campaign's raw cell evidence into causal per-decision records.

Offline and read-only with respect to the evidence tree. One row per
application decision, with the two candidate state features attached only from
observations that already existed when the decision was made.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_agent.ue_mcs_backlog_calibration_v1 import contract as C  # noqa: E402
from rl_agent.ue_mcs_backlog_calibration_v1 import decision_join as J  # noqa: E402


def require_create_only_targets(run_dir: Path) -> tuple[Path, Path]:
    """Refuse to overwrite either immutable v2 decision artifact."""
    targets = (run_dir / "decisions_v2.csv", run_dir / "decisions_build_v2.json")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"v2 decision output already exists: {existing}")
    return targets


def build_cell(cell_dir: Path) -> dict[str, Any]:
    """Join one cell, or explain precisely why it cannot be joined."""
    record = json.loads((cell_dir / "cell_record.json").read_text())
    out: dict[str, Any] = {"cell_tag": record["cell_tag"],
                           "cell_status": record["status"]}

    sender_csv = cell_dir / "sender_decisions.csv"
    ue_csv = cell_dir / "ttracer/ue/csv"
    needed = {
        "sender": sender_csv,
        "dci": ue_csv / "NRUE_MAC_DCI_GRANT.csv",
        "rlc": ue_csv / "NRUE_MAC_RLC_BUFFER_STATUS.csv",
        "gnb_mcs": cell_dir / "ttracer/gnb/csv/GNB_MAC_UL_MCS_DECISION.csv",
    }
    missing = [name for name, path in needed.items() if not path.is_file()]
    if missing:
        out["joined"] = False
        out["reason"] = f"missing evidence: {missing}"
        return out

    with sender_csv.open(newline="", encoding="utf-8") as handle:
        sender_rows = list(csv.DictReader(handle))

    # Prefer the same-event PDCP bridge; fall back to the sender's own
    # same-instant wall/monotonic pairs. Both are measured; which one was used
    # is recorded so the join precision can be judged.
    pdcp_path = ue_csv / "NR_PDCP_TX_SDU.csv"
    pdcp_rows = (J.read_exact_csv(pdcp_path, C.PDCP_TX_SDU_HEADER)
                 if pdcp_path.is_file() else [])
    if pdcp_rows:
        bridge = J.build_clock_bridge(pdcp_rows)
        bridge_source = "NR_PDCP_TX_SDU_SAME_EVENT"
    else:
        bridge = J.build_clock_bridge_from_sender(sender_rows)
        bridge_source = "SENDER_SAME_INSTANT_WALL_MONOTONIC_PAIRS"
    rlc_rows = J.read_exact_csv(needed["rlc"], C.RLC_BUFFER_HEADER)
    dci_rows = J.read_exact_csv(needed["dci"], C.DCI_GRANT_HEADER)
    gnb_mcs_rows = J.read_exact_csv(
        needed["gnb_mcs"], C.GNB_MCS_DECISION_HEADER)
    window_audit = J.audit_bridge_window(bridge, rlc_rows, sender_rows)
    if not window_audit["verified"]:
        out["joined"] = False
        out["reason"] = f"clock bridge failed its window audit: {window_audit}"
        return out
    ticks = J.build_backlog_ticks(rlc_rows, bridge)
    grants, grant_counts = J.build_ul_grants(dci_rows, bridge)
    gnb_decisions = J.build_gnb_mcs_decisions(gnb_mcs_rows, bridge)

    arrivals_by_block: dict[int, dict[int, J.FrameArrival]] = {}
    for block in record["blocks"]:
        index = int(block["block_index"])
        events = cell_dir / f"receiver_block{index}_{block['tier']}_events.jsonl"
        arrivals_by_block[index] = (
            J.build_arrivals(events, int(block["chunks_per_frame"]))
            if events.is_file() else {})

    records = J.join_cell(
        cell_meta={"cell_id": record["cell_id"], "profile_id": record["profile_id"],
                   "order_index": record["order_index"],
                   "repetition": record["repetition"],
                   "sequence": tuple(record["sequence"])},
        sender_rows=sender_rows, ticks=ticks, grants=grants,
        arrivals_by_block=arrivals_by_block, budget_ms=C.AGENT_PATH_BUDGET_MS)
    used_grants = J.grants_used_by_records(records, grants)
    provenance = J.audit_ue_gnb_mcs_provenance(used_grants, gnb_decisions)
    provenance["scope"] = "UNIQUE_UE_GRANTS_ACTUALLY_USED_BY_DECISIONS"
    provenance["raw_ue_round0_grants_available"] = len(grants)

    bridge_json = bridge.to_json()
    bridge_json["source"] = bridge_source
    bridge_json["window_audit"] = window_audit
    out.update({
        "joined": True,
        "records": records,
        "clock_bridge": bridge_json,
        "grant_counts": grant_counts,
        "ue_gnb_mcs_provenance": provenance,
        "backlog_ticks": len(ticks),
        "pdcp_rows": len(pdcp_rows),
        "bridge_source": bridge_source,
        "causal_audit": J.causal_audit(records),
        "sender_decisions": len(sender_rows),
    })
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    out_csv, out_summary = require_create_only_targets(args.run_dir)

    cells = sorted(p for p in (args.run_dir / "cells").iterdir() if p.is_dir())
    all_records: list[dict[str, Any]] = []
    per_cell: list[dict[str, Any]] = []
    for cell_dir in cells:
        if not (cell_dir / "cell_record.json").is_file():
            per_cell.append({"cell_tag": cell_dir.name, "joined": False,
                             "reason": "no cell_record.json"})
            continue
        result = build_cell(cell_dir)
        if result.get("joined"):
            all_records.extend(result.pop("records"))
        per_cell.append(result)

    with out_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(J.DECISION_FIELDS))
        writer.writeheader()
        for row in all_records:
            writer.writerow({key: row.get(key) for key in J.DECISION_FIELDS})

    summary = {
        "run_dir": str(args.run_dir),
        "cells_total": len(cells),
        "cells_joined": sum(1 for c in per_cell if c.get("joined")),
        "decisions": len(all_records),
        "per_cell": per_cell,
    }
    with out_summary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("cells_total", "cells_joined", "decisions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
