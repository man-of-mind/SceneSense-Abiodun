"""Reproduce transport CSVs from packaged compressed T-tracer recordings."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import tempfile

import prepare_plots as P
import transport_plots as T

PACKAGE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "transport_regenerated")
    args = parser.parse_args()
    output = args.output.resolve()
    assert output not in {PACKAGE, PACKAGE / "data", PACKAGE / "raw", PACKAGE / "figures"}
    output.mkdir(parents=True, exist_ok=True)
    P.OUT = output
    for name in ("METRICS.json", "FRAMES.csv"):
        shutil.copyfile(PACKAGE / "data" / name, output / name)
    # Temporary files are only these locally unpacked copies, not original captures.
    with tempfile.TemporaryDirectory(prefix="supervisor_transport_") as tmp:
        P.CAPTURES = Path(tmp)
        for arm in P.ARMS:
            dest = P.CAPTURES / arm / "capture/cells/a71__favorable_stable/ttracer/ue/ue.raw"
            dest.parent.mkdir(parents=True)
            with gzip.open(PACKAGE / "raw" / arm / "ue.raw.gz", "rb") as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)
            dest = P.CAPTURES / arm / "capture/m3_radio/ttracer/live_mcs_with_ingest.csv"
            dest.parent.mkdir(parents=True)
            with gzip.open(PACKAGE / "raw" / arm / "live_mcs_with_ingest.csv.gz", "rb") as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target)
        T.extract()
    names = ("TRANSPORT_RATES_1S.csv", "BUFFER_OCCUPANCY_1S.csv", "TRANSPORT_METRICS.json")
    result = {name: P.sha(output / name) == P.sha(PACKAGE / "data" / name) for name in names}
    print(json.dumps(result, indent=2))
    assert all(result.values())


if __name__ == "__main__":
    main()
