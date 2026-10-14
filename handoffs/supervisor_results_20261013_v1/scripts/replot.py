"""Recreate the 13 PNGs from packaged CSV/JSON data, without the experiment tree."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import prepare_plots as P
import transport_plots as T

PACKAGE = Path(__file__).resolve().parents[1]


def records(name):
    with (PACKAGE / "data" / name).open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if not v:
                r[k] = None
            elif v in ("True", "False"):
                r[k] = v == "True"
            elif k != "arm":
                try:
                    r[k] = int(v)
                except ValueError:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
    return rows


def load():
    stats = json.loads((PACKAGE / "data/METRICS.json").read_text())
    files = {"frames": "FRAMES.csv", "decisions": "DECISIONS.csv", "radio": "PUSCH.csv",
             "commands": "CHANNEL_COMMANDS.csv", "installs": "MAP_INSTALLS.csv", "segments": "AOI_SEGMENTS.csv"}
    tables = {key: records(name) for key, name in files.items()}
    return {a: {"summary": stats[a], **{key: [r for r in rows if r["arm"] == a]
                                       for key, rows in tables.items()}} for a in P.ARMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "plots_regenerated")
    args = parser.parse_args()
    output = args.output.resolve()
    assert output not in {PACKAGE, PACKAGE / "data", PACKAGE / "figures", PACKAGE / "raw"}
    output.mkdir(parents=True, exist_ok=True)
    P.OUT = output
    data = load()
    P.style()
    for plot in [P.completion_plots, P.payload_plot, P.payload_time_plot, P.radio_plot,
                 P.latency_time_plot, P.quality_plot, P.scene_plot, P.map_plot,
                 P.quality_time_plot, P.map_quantile_plot]:
        plot(data)
    stats = json.loads((PACKAGE / "data/TRANSPORT_METRICS.json").read_text())
    rates, queues = records("TRANSPORT_RATES_1S.csv"), records("BUFFER_OCCUPANCY_1S.csv")
    transport = {a: {"stats": stats[a], "rates": [r for r in rates if r["arm"] == a],
                     "queues": [r for r in queues if r["arm"] == a]} for a in P.ARMS}
    T.rate_plot(transport)
    T.queue_plot(transport)
    result = {"png_count": len(P.VALIDATION), "figures": P.VALIDATION,
              "matches_shared_png": {n: P.sha(output / n) == P.sha(PACKAGE / "figures" / n)
                                     for n in P.VALIDATION}}
    P.write_json("REPRODUCTION.json", result)
    P.write_json("CAPTIONS.json", P.CAPTIONS)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
