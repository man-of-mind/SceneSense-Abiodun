#!/usr/bin/env python3
"""Mechanism view of the evaluator-ordering experiment (read-only over exports).

Per reward ticket, joined by exact identity on one raw clock, two **parallel**
branches are reported separately (never summed into one waterfall):

* evaluator branch: tail-compute end -> evaluator submission -> Q_perc start ->
  Q_perc end -> feedback emitted -> UE receipt;
* map branch: tail-compute end -> map update built -> map published ->
  (capture ->) map installed.

Plus outcome accounting (timely / late / missing with terminals), edge queue
and supersession counts, and the retrospective check of the "recovered
tickets" hypothesis against what actually happened.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import export_v1 as EX

SCHEMA = "scenesense.single_gpu_profiler.ordering_mechanism.v1"


def pct(values: Sequence[float], p: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(1, -(-int(p * len(ordered)) // 100)) - 1]


def dist(values: Iterable[Optional[float]]) -> dict[str, Any]:
    present = [float(v) for v in values if v is not None]
    return {"n": len(present), "p50": pct(present, 50), "p95": pct(present, 95),
            "p99": pct(present, 99), "max": max(present) if present else None,
            "mean": statistics.fmean(present) if present else None}


def _f(v: Any) -> Optional[float]:
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def arm_rows(root: Path) -> dict[str, Any]:
    export = Path(root) / "export_v1"
    assert EX.verify(export)["verified"], f"export not verified: {root}"
    frames = {int(r["frame_id"]): r for r in csv.DictReader((export / "frames.csv").open())}
    tickets = {int(r["frame_id"]): r for r in csv.DictReader((export / "tickets.csv").open())}
    marks: dict[int, dict[str, int]] = collections.defaultdict(dict)
    for r in csv.DictReader((export / "marks.csv").open()):
        if r["frame_id"]:
            marks[int(r["frame_id"])].setdefault(r["stage"], int(r["t_raw_ns"]))
    rows = []
    for f, t in tickets.items():
        fr = frames[f]
        m = marks[f]

        def start(stage: str) -> Optional[int]:
            v = _f(fr.get(f"{stage}__start_raw_ns"))
            return None if v is None else int(v)

        def end(stage: str) -> Optional[int]:
            s, d = start(stage), _f(fr.get(f"{stage}__dur_ms"))
            return None if s is None or d is None else s + int(d * 1e6)

        ao = int(fr["action_open_raw_ns"])
        compute_end = end("edge.compute")
        submit = m.get("edge.evaluator_submit")
        receipt = m.get("ue.feedback_receipt")
        ms = lambda a, b: None if a is None or b is None else (b - a) / 1e6  # noqa: E731
        total = ms(ao, receipt)
        rows.append({
            "frame_id": f, "terminal": t.get("resolution_terminal"),
            "cls": "MISSING" if receipt is None else ("TIMELY" if total <= 170 else "LATE"),
            "ao_to_receipt": total,
            # evaluator branch
            "compute_end_to_submit": ms(compute_end, submit),
            "submit_to_qperc_start": ms(submit, start("edge.qperc_evaluate")),
            "qperc": _f(fr.get("edge.qperc_evaluate__dur_ms")),
            "qperc_end_to_emit_end": ms(end("edge.qperc_evaluate"), end("edge.feedback_emit")),
            "emit_end_to_receipt": ms(end("edge.feedback_emit"), receipt),
            "compute_end_to_receipt": ms(compute_end, receipt),
            # map branch
            "compute_end_to_map_built": ms(compute_end, end("edge.map_update_build")),
            "compute_end_to_map_published": ms(compute_end, end("edge.map_publish")),
            "capture_to_map_install": _f(t.get("capture_to_map_install_ms")),
            "map_outcome": fr.get("map_outcome"),
            # context
            "ao_to_compute_end": ms(ao, compute_end),
            "submit_before_map_built": (None if submit is None or end("edge.map_update_build") is None
                                        else submit < end("edge.map_update_build")),
        })
    return {"root": str(root), "rows": rows}


def summarize(rows: Sequence[dict]) -> dict[str, Any]:
    keys = ("ao_to_receipt", "ao_to_compute_end", "compute_end_to_submit", "submit_to_qperc_start",
            "qperc", "qperc_end_to_emit_end", "emit_end_to_receipt", "compute_end_to_receipt",
            "compute_end_to_map_built", "compute_end_to_map_published", "capture_to_map_install")
    out: dict[str, Any] = {"classes": dict(collections.Counter(r["cls"] for r in rows)),
                           "terminals": dict(collections.Counter(r["terminal"] for r in rows)),
                           "map_outcomes": dict(collections.Counter(r["map_outcome"] for r in rows)),
                           "submit_before_map_built": sum(1 for r in rows if r["submit_before_map_built"]),
                           "by_class": {}}
    for cls in ("ALL", "TIMELY", "LATE"):
        subset = rows if cls == "ALL" else [r for r in rows if r["cls"] == cls]
        out["by_class"][cls] = {k: dist(r[k] for r in subset) for k in keys}
    late = [r for r in rows if r["cls"] == "LATE" and r["ao_to_receipt"] is not None]
    excess = sorted(r["ao_to_receipt"] - 170 for r in late)
    out["late_excess_ms"] = {"p25": pct(excess, 25), "p50": pct(excess, 50), "p75": pct(excess, 75),
                             "le_5": sum(1 for x in excess if x <= 5), "le_10": sum(1 for x in excess if x <= 10)}
    return out


def render(a: dict, b: dict, la: str, lb: str) -> str:
    lines = [f"# Edge evaluator ordering: {la} (serial) vs {lb} (early)", "",
             "Branches are reported separately; nothing is summed across branches.", "",
             f"| per reward ticket, ms | {la} ALL P50/P95 | {lb} ALL P50/P95 | {la} LATE P50/P95 | {lb} LATE P50/P95 |",
             "|---|---|---|---|---|"]
    def cell(d): return f"{d['p50']:.1f} / {d['p95']:.1f} (n={d['n']})" if d["p50"] is not None else "–"
    for k in ("ao_to_compute_end", "compute_end_to_submit", "submit_to_qperc_start", "qperc",
              "qperc_end_to_emit_end", "emit_end_to_receipt", "compute_end_to_receipt", "ao_to_receipt",
              "compute_end_to_map_built", "compute_end_to_map_published", "capture_to_map_install"):
        lines.append(f"| {k} | {cell(a['by_class']['ALL'][k])} | {cell(b['by_class']['ALL'][k])} | "
                     f"{cell(a['by_class']['LATE'][k])} | {cell(b['by_class']['LATE'][k])} |")
    lines += ["", f"- classes: {la} {a['classes']} · {lb} {b['classes']}",
              f"- terminals: {la} {a['terminals']} · {lb} {b['terminals']}",
              f"- map outcomes: {la} {a['map_outcomes']} · {lb} {b['map_outcomes']}",
              f"- tickets submitted before the map update was built: {la} {a['submit_before_map_built']} · {lb} {b['submit_before_map_built']}",
              f"- late excess over 170 ms P25/P50/P75, ≤5 ms, ≤10 ms: {la} {a['late_excess_ms']} · {lb} {b['late_excess_ms']}", ""]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--labels", default="C2,E")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    la, lb = args.labels.split(",")
    A, B = arm_rows(args.a), arm_rows(args.b)
    sa, sb = summarize(A["rows"]), summarize(B["rows"])
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "MECHANISM.json").open("x", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, la: {"root": A["root"], "summary": sa, "rows": A["rows"]},
                   lb: {"root": B["root"], "summary": sb, "rows": B["rows"]}}, handle, indent=1,
                  sort_keys=True, default=str)
    text = render(sa, sb, la, lb)
    (args.out / "MECHANISM.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
