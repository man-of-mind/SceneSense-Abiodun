#!/usr/bin/env python3
"""Record live spatial-map snapshots to JSONL so the map/geometry/occlusion logic can be developed and
verified OFFLINE afterwards (no live CARLA needed). Run this alongside the live pipeline (ideally with
BOTH egos streaming) for a few minutes, then Ctrl+C.

  python3 record_trace.py --out recordings/two_ego_run1.jsonl --hz 5 --duration-s 300

Writes:
  <out>            one JSON snapshot per line (deduped: only when frame_id advances)
  <out>.static.json  the static town geometry (roads/buildings), captured once
"""
import argparse
import json
import os
import time
import urllib.request


def _get(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:35011")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--duration-s", type=float, default=0.0, help="0 = until Ctrl+C")
    a = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)

    try:
        static = _get(a.api + "/api/spatial_map/static_geometry")
        with open(a.out + ".static.json", "w") as f:
            json.dump(static, f)
        print(f"saved static geometry ({len(static.get('roads', []))} roads, "
              f"{len(static.get('buildings', []))} buildings) -> {a.out}.static.json")
    except Exception as e:
        print("WARN could not fetch static geometry:", e)

    dt = 1.0 / max(0.5, a.hz)
    t0 = time.time()
    n = 0
    last_frame = object()
    with open(a.out, "w") as f:
        try:
            while True:
                try:
                    snap = _get(a.api + "/api/spatial_map/latest")
                    fr = snap.get("frame_id")
                    if fr != last_frame:  # dedup identical (unchanged) snapshots
                        f.write(json.dumps({"t": time.time(), "snap": snap}) + "\n")
                        f.flush()
                        n += 1
                        last_frame = fr
                        if n % 20 == 0:
                            print(f"  {n} snapshots (frame {fr})")
                except Exception:
                    pass
                if a.duration_s and (time.time() - t0) >= a.duration_s:
                    break
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
    print(f"recorded {n} snapshots -> {a.out}")


if __name__ == "__main__":
    main()
