"""Adapter smoke feed: re-times ONE recorded capture's evidence into the exact live-view tap format at 1x.

Test tooling only.  Reads a sealed cell (installed-map journal, UE pose evidence and, where retained, the scorer's copy
of the producer records with the world-yaw pair) and appends ``live_view/ue_pose.jsonl`` and
``live_view/map_installs.jsonl`` in a NEW directory, at the recorded inter-arrival times, with every wall-clock field
shifted by one integer-ns offset so ages are current.  Hashes of the owned sensor source are unaffected (they do not
cover wall stamps).  Faults can be injected to exercise the viewer: a writer pause (feed stale / disconnect), duplicate
and out-of-order installs, and an install line from another session.  Never touches the source capture.
"""
import argparse
import json
from pathlib import Path
import time

TAP_MAP = "scenesense.live_view.map_install.v1"
TAP_POSE = "scenesense.live_view.ue_pose.v1"


def load(cell, start_s, duration_s):
    journal = []
    with (cell / "m3_map/installed_map_records.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d["outcome"] == "OK":
                journal.append(d["map_document"]["record"])
    journal.sort(key=lambda r: r["install_timestamp"])
    t0 = journal[0]["install_timestamp"] + start_s
    journal = [r for r in journal if t0 <= r["install_timestamp"] <= t0 + duration_s]
    frames = {r["frame_id"] for r in journal}
    wires, owned = {}, {}
    with (cell / "pose_archive/pose_evidence.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d["kind"] == "UE_WIRE" and d["context"]["frame_id"] in frames:
                wires[d["context"]["frame_id"]] = d
            elif d["kind"] == "OWNED_SOURCE":
                owned[d["source_sha256"]] = d
    ingest = {}
    import csv
    with (cell / "direct_edge_map/direct_map_ingest.csv").open() as f:
        for row in csv.DictReader(f):
            if row["outcome"] == "RESULT_INSTALLED" and int(row["frame_id"]) in frames:
                ingest[int(row["frame_id"])] = row
    return journal, wires, owned, ingest


def producer_records(cell, frame):
    path = cell / f"numeric_gt/service/archive/frame_{frame}/scorer/numerical_score.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text())["prediction_records"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new directory; live_view/ is created inside")
    p.add_argument("--start-s", type=float, default=30.0)
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--pause-at-s", type=float, default=None, help="stop writing for --pause-s (stale feed)")
    p.add_argument("--pause-s", type=float, default=3.0)
    p.add_argument("--duplicate-every", type=int, default=0)
    p.add_argument("--out-of-order-at", type=int, default=None, help="re-send an older install after this index")
    p.add_argument("--old-session-at", type=int, default=None, help="inject an install from another run after index")
    p.add_argument("--yaw-from-scorer", action="store_true", help="attach retained producer records (yaw pair)")
    args = p.parse_args(argv)
    journal, wires, owned, ingest = load(args.cell, args.start_s, args.duration_s)
    directory = args.out / "live_view"
    directory.mkdir(parents=True, exist_ok=False)
    # The first installation lands "now"; every other stamp keeps its recorded offset from it.
    shift = time.time_ns() + 150_000_000 - round(journal[0]["install_timestamp"] * 1e9)
    sh = lambda ns: int(ns) + shift  # noqa: E731 - reads the current shift
    pose_f = (directory / "ue_pose.jsonl").open("x", buffering=1)
    map_f = (directory / "map_installs.jsonl").open("x", buffering=1)
    seq = dict(pose=0, map=0)
    start_wall, start_install = time.monotonic() + 0.15, journal[0]["install_timestamp"]
    sent_lines = []
    paused = False

    def pose(kind, value):
        seq["pose"] += 1
        pose_f.write(json.dumps(dict(value, schema=TAP_POSE, kind=kind, seq=seq["pose"], tap_wall_ns=time.time_ns(),
                                     written_wall_ns=time.time_ns()), separators=(",", ":")) + "\n")

    def install(line):
        seq["map"] += 1
        line = dict(line, seq=seq["map"], tap_wall_ns=time.time_ns(), written_wall_ns=time.time_ns())
        map_f.write(json.dumps(line, separators=(",", ":")) + "\n")

    for index, record in enumerate(journal):
        due = start_wall + (record["install_timestamp"] - start_install)
        if args.pause_at_s is not None and not paused and record["install_timestamp"] - start_install >= args.pause_at_s:
            # The pipeline produces nothing for pause_s (stale feed); later stamps move with it, as they would live.
            paused = True
            due += args.pause_s
            start_wall += args.pause_s
            shift += round(args.pause_s * 1e9)
        frame = record["frame_id"]
        wire = wires.get(frame)
        capture_ns = sh(round(record["capture_timestamp"] * 1e9))
        # pose first (as in the pipeline: prepared/sent before installation), ~120 ms before the install
        while time.monotonic() < due - 0.12:
            time.sleep(0.002)
        if wire is not None:
            src = owned.get(wire["source_sha256"])
            if src is not None:
                pose("OWNED_SOURCE", {k: v for k, v in src.items() if k not in ("kind", "observed_raw_ns")})
            w = json.loads(json.dumps(wire))
            w["context"]["capture_timestamp_ns"] = sh(w["context"]["capture_timestamp_ns"])
            w["gt_identity"]["capture_timestamp_ns"] = sh(w["gt_identity"]["capture_timestamp_ns"])
            pose("UE_WIRE", {k: v for k, v in w.items() if k not in ("kind", "observed_raw_ns")})
        while time.monotonic() < due:
            time.sleep(0.002)
        r = json.loads(json.dumps(record))
        r["capture_timestamp"] = capture_ns / 1_000_000_000.0
        r["install_timestamp"] = record["install_timestamp"] + shift / 1e9
        row = ingest.get(frame, {})
        producer = []
        if args.yaw_from_scorer:
            for q in producer_records(args.cell, frame):
                q = dict(q, capture_timestamp_ns=sh(q["capture_timestamp_ns"]))
                producer.append(q)
        line = dict(schema=TAP_MAP, identity=dict(
            schema="splitfusion_direct_object_map_update.v1", run_id=row.get("run_id"), cell_id=row.get("cell_id"),
            stream_id=r["stream_id"], frame_id=frame, sequence_id=None, action_id=r.get("action_id"),
            profile_id=None, capture_timestamp_ns=capture_ns),
            install=dict(install_timestamp=r["install_timestamp"]), record=r, record_available=True,
            producer_records=producer, producer_record_count=len(producer), tap_counts={})
        install(line)
        sent_lines.append(line)
        if args.duplicate_every and index % args.duplicate_every == 0:
            install(line)                                   # exact duplicate frame
        if args.out_of_order_at is not None and index == args.out_of_order_at and index >= 5:
            install(sent_lines[index - 5])                  # delayed older frame
        if args.old_session_at is not None and index == args.old_session_at:
            other = json.loads(json.dumps(sent_lines[max(0, index - 3)]))
            other["identity"]["run_id"] = "OLD_SESSION/19700101T000000Z"
            install(other)
    pose_f.close()
    map_f.close()
    print(json.dumps(dict(status="DONE", installs=seq["map"], pose_lines=seq["pose"], frames=len(journal))))


if __name__ == "__main__":
    main()
