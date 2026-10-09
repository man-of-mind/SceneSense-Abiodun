"""Deadline-aware localisation quality from the analyzer's per-decision timeline (offline, CPU only).

Same per-decision inputs and tolerances as the registered analyzer (vehicle 1.0 m, pedestrian 1.2 m, eligible GT incl.
misses); the only difference is the denominator/numerator rule, which makes timeliness part of quality:

  * GT-available decisions = ``quality_defined`` True (exact GT exists and was scored for that frame);
  * deadline-aware recall  = within-tolerance objects of decisions whose feedback was received <= 170 ms
                             / ALL eligible GT objects of GT-available decisions -- a late (or missing) prediction
                             keeps its eligible GT in the denominator and contributes zero, i.e. it is a deadline miss;
  * unavailable GT        = decisions with ``quality_defined`` False; they are counted separately and are neither
                             misses nor hits (no GT exists to score against).

The analyzer's own all-prediction recall (no deadline) and its timely-subset recall are reported alongside, so the three
views share every count.  Computed identically for every capture given (V11 and V12), descriptively.

    python3 -B -m rl_agent.splitfusion_m3_route_b_v12_analysis.deadline_quality \
        --timeline LABEL=<analysis dir>/DECISION_TIMELINE.csv [...] --out <new dir>
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CLASSES = (("vehicle", "vehicle_eligible", "vehicle_within_1p0", 1.0),
           ("person", "person_eligible", "person_within_1p2", 1.2))


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def truthy(value):
    return str(value).strip().lower() in ("true", "1")


def evaluate(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    decisions = len(rows)
    timely = [truthy(r["timely_170"]) for r in rows]
    defined = [truthy(r["quality_defined"]) for r in rows]
    out = dict(source=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               decisions=decisions, timely_170=sum(timely), timely_fraction=sum(timely) / decisions if decisions else None,
               timely_wilson95=wilson(sum(timely), decisions),
               gt_available_decisions=sum(defined), gt_unavailable_decisions=decisions - sum(defined),
               gt_available_and_timely=sum(d and t for d, t in zip(defined, timely)),
               gt_available_and_late=sum(d and not t for d, t in zip(defined, timely)),
               gt_unavailable_and_timely=sum((not d) and t for d, t in zip(defined, timely)),
               gt_unavailable_and_late=sum((not d) and (not t) for d, t in zip(defined, timely)), classes={})
    for name, eligible_key, within_key, tolerance in CLASSES:
        eligible = within = eligible_timely = within_timely = eligible_late = within_late = 0
        for row, d, t in zip(rows, defined, timely):
            if not d:
                continue
            e, w = int(float(row[eligible_key] or 0)), int(float(row[within_key] or 0))
            eligible += e
            within += w
            if t:
                eligible_timely += e
                within_timely += w
            else:
                eligible_late += e
                within_late += w
        out["classes"][name] = dict(
            tolerance_m=tolerance, eligible_gt_in_gt_available_decisions=eligible,
            all_predictions_recall=within / eligible if eligible else None,
            all_predictions_within=within,
            deadline_aware_recall=within_timely / eligible if eligible else None,
            deadline_aware_within=within_timely, deadline_aware_wilson95=wilson(within_timely, eligible),
            deadline_misses_eligible_gt=eligible_late,
            late_predictions_that_would_have_been_within=within_late,
            timely_subset_recall=within_timely / eligible_timely if eligible_timely else None,
            timely_subset_eligible_gt=eligible_timely)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", action="append", required=True, help="LABEL=<DECISION_TIMELINE.csv>")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema="splitfusion.m3_route_b_v12.deadline_quality.v1",
                  definitions=__doc__.split("\n\n")[1], captures={})
    for item in args.timeline:
        label, path = item.split("=", 1)
        result["captures"][label] = evaluate(path)
    output = args.out / "DEADLINE_QUALITY.json"
    output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    rows = []
    for label, value in result["captures"].items():
        for name, cls in value["classes"].items():
            rows.append(dict(capture=label, cls=name, decisions=value["decisions"], timely_170=value["timely_170"],
                             gt_available=value["gt_available_decisions"],
                             gt_unavailable=value["gt_unavailable_decisions"],
                             eligible_gt=cls["eligible_gt_in_gt_available_decisions"],
                             all_predictions_recall=cls["all_predictions_recall"],
                             deadline_aware_recall=cls["deadline_aware_recall"],
                             deadline_misses_eligible_gt=cls["deadline_misses_eligible_gt"],
                             timely_subset_recall=cls["timely_subset_recall"]))
    with (args.out / "DEADLINE_QUALITY.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.iterdir()) if p.is_file()}
    (args.out / "OUTPUT_SHA256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: {c: (v["classes"][c]["all_predictions_recall"], v["classes"][c]["deadline_aware_recall"])
                          for c in v["classes"]} for k, v in result["captures"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
