#!/usr/bin/env python3
"""Offline analysis of a Route B population ledger's walker-controller churn.

Read-only. Groups ``controller_disowned`` events by walker body, reconstructs
each controller's lifetime in observed route ticks from its spawn/disown pair,
and reports whether the churn is concentrated in a small number of pathological
bodies or spread across the population.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PHASES = ("body_spawned", "controller_spawned", "controller_started",
          "controller_disowned", "body_lost", "orphan_controller_reaped")


def load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = load(args.ledger)
    report: dict[str, Any] = {
        "schema": "route_b_perception_v2.controller_churn_analysis.v1",
        "ledger": str(args.ledger.resolve()),
        "event_lines": len(rows),
    }

    # Controller lifetimes: pair each controller_spawned with the next event that
    # ends that controller (disown, body loss, or end of episode).
    spawn_tick: dict[int, int] = {}
    spawn_body: dict[int, int] = {}
    lifetimes: list[dict[str, Any]] = []
    last_tick = max((int(r.get("observed_tick") or 0) for r in rows), default=0)
    disowned_by_body: dict[int, int] = defaultdict(int)
    ended: set[int] = set()

    for row in rows:
        phase = row.get("phase")
        tick = int(row.get("observed_tick") or 0)
        controller_id = row.get("controller_id")
        body_id = row.get("body_id")
        if phase == "controller_spawned" and controller_id is not None:
            spawn_tick[int(controller_id)] = tick
            spawn_body[int(controller_id)] = int(body_id) if body_id is not None else -1
        elif phase in ("controller_disowned", "body_lost") and controller_id is not None:
            cid = int(controller_id)
            if phase == "controller_disowned" and body_id is not None:
                disowned_by_body[int(body_id)] += 1
            if cid in spawn_tick and cid not in ended:
                ended.add(cid)
                lifetimes.append({
                    "controller_id": cid,
                    "body_id": spawn_body.get(cid, -1),
                    "spawn_tick": spawn_tick[cid],
                    "end_tick": tick,
                    "lifetime_ticks": tick - spawn_tick[cid],
                    "end_reason": phase,
                })
    for cid, tick in spawn_tick.items():
        if cid not in ended:
            lifetimes.append({
                "controller_id": cid,
                "body_id": spawn_body.get(cid, -1),
                "spawn_tick": tick,
                "end_tick": last_tick,
                "lifetime_ticks": last_tick - tick,
                "end_reason": "survived_to_episode_end",
            })

    disown_counts = dict(sorted(
        ((int(k), int(v)) for k, v in disowned_by_body.items()),
        key=lambda item: (-item[1], item[0])))
    total_disowns = sum(disown_counts.values())
    bodies_with_disowns = len(disown_counts)
    top_body, top_count = (None, 0)
    if disown_counts:
        top_body, top_count = next(iter(disown_counts.items()))

    managed_bodies = {
        int(r["body_id"]) for r in rows
        if r.get("phase") == "body_spawned" and r.get("body_id") is not None
    }
    short = [row for row in lifetimes if row["end_reason"] == "controller_disowned"]
    survived = [row for row in lifetimes if row["end_reason"] == "survived_to_episode_end"]

    report["disowns_by_body"] = disown_counts
    report["totals"] = {
        "managed_bodies_seen": len(managed_bodies),
        "controllers_spawned": len(spawn_tick),
        "controller_disowned_events": total_disowns,
        "bodies_with_at_least_one_disown": bodies_with_disowns,
        "bodies_with_zero_disowns": len(managed_bodies) - bodies_with_disowns,
        "top_body": top_body,
        "top_body_disowns": top_count,
        "top_body_share_of_disowns": (
            round(top_count / total_disowns, 4) if total_disowns else None),
        "last_observed_tick": last_tick,
    }
    report["controller_lifetime_ticks"] = {
        "disowned": {
            "count": len(short),
            "min": min((r["lifetime_ticks"] for r in short), default=None),
            "median": (round(statistics.median(r["lifetime_ticks"] for r in short), 1)
                       if short else None),
            "max": max((r["lifetime_ticks"] for r in short), default=None),
        },
        "survived_to_episode_end": {
            "count": len(survived),
            "min": min((r["lifetime_ticks"] for r in survived), default=None),
            "median": (round(statistics.median(r["lifetime_ticks"] for r in survived), 1)
                       if survived else None),
            "max": max((r["lifetime_ticks"] for r in survived), default=None),
        },
    }
    report["lifetimes"] = sorted(lifetimes, key=lambda r: r["spawn_tick"])

    # Concentration verdict: is this one pathological body, or systematic?
    others = total_disowns - top_count
    report["verdict"] = {
        "single_body_concentration": bool(
            total_disowns and top_count / total_disowns >= 0.5),
        "disowns_excluding_top_body": others,
        "bodies_excluding_top_with_disowns": max(0, bodies_with_disowns - 1),
        "mean_disowns_per_other_affected_body": (
            round(others / (bodies_with_disowns - 1), 2)
            if bodies_with_disowns > 1 else None),
        "affected_body_fraction": (
            round(bodies_with_disowns / len(managed_bodies), 4)
            if managed_bodies else None),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k != "lifetimes"}
    print(json.dumps(printable, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
