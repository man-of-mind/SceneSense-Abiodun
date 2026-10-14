"""Create a portable supervisor data handoff without modifying source evidence."""
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "handoffs/supervisor_results_20261013_v1"
PLOTS = HERE / "three_arm_plots_01"
CAPTURES = ROOT.parent / "abiodun_run4_single_gpu_profiler_v1/rl_agent/experiments/splitfusion_single_gpu_profiler_v1/20261009_static_modes_v13"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    assert not (OUT / "raw").exists(), "Refuse to rebuild over a populated handoff"
    manifest = []

    def copy(source, relative, compressed=False):
        source = source.resolve()
        dest = OUT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists(), str(dest)
        digest = sha(source)
        if compressed:
            with source.open("rb") as inp, dest.open("wb") as packed:
                with gzip.GzipFile(fileobj=packed, mode="wb", filename="", mtime=0, compresslevel=1) as out:
                    shutil.copyfileobj(inp, out)
            h = hashlib.sha256()
            with gzip.open(dest, "rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
            assert h.hexdigest() == digest
        else:
            shutil.copyfile(source, dest)
            assert sha(dest) == digest
        assert sha(source) == digest
        manifest.append({"source": str(source), "source_sha256": digest, "source_bytes": source.stat().st_size,
                         "packaged_path": relative, "packaged_sha256": sha(dest), "packaged_bytes": dest.stat().st_size,
                         "encoding": "gzip" if compressed else "byte-identical copy"})

    for f in sorted(PLOTS.iterdir()):
        if f.suffix == ".png":
            copy(f, "figures/" + f.name)
        elif f.suffix in {".csv", ".json"} and "SHA256" not in f.name and "VALIDATION" not in f.name:
            copy(f, "data/" + f.name)
        else:
            copy(f, "provenance/original_plot_bundle/" + f.name)
    for name in ("prepare_plots.py", "transport_plots.py"):
        copy(HERE / name, "scripts/" + name)
    for arm in ("agent", "mode03", "mode09"):
        capture = CAPTURES / arm / "capture"
        cell = capture / "cells/a71__favorable_stable"
        for name in ("run4_phase6/PHASE6_UE_EVIDENCE.json", "run4_phase6/edge_report.json",
                     "per_frame_metrics.csv", "perception_metrics.csv", "map_feedback.csv",
                     "quality_ack_oaitun_ue1.pcap", "quality_ack_tcpdump.log"):
            copy(cell / name, "raw/" + arm + "/" + Path(name).name)
        copy(cell / "numeric_gt/service/FRAMES.jsonl", "raw/" + arm + "/gt_service_frames.jsonl.gz", True)
        copy(cell / "ttracer/ue/ue.raw", "raw/" + arm + "/ue.raw.gz", True)
        for name in ("live_mcs_with_ingest.csv", "live_pusch_with_ingest.csv"):
            copy(capture / "m3_radio/ttracer" / name, "raw/" + arm + "/" + name + ".gz", True)
    for source in ("rl_agent/splitfusion_hybrid_sac_live_route_b_v2/run4_live_wire_v2.py",
                   "rl_agent/splitfusion_hybrid_sac_run4_v1/run4_contract.py",
                   "rl_agent/splitfusion_hybrid_sac_run4_v1/RUN4_CONTRACT.md",
                   "rl_agent/splitfusion_run5_full_v1/quality.py"):
        copy(ROOT / source, "source_reference/" + source)
    for name in ("CAMPAIGN_REPORT.json", "CAMPAIGN_TABLE.md", "REPORT.md"):
        copy(ROOT / "experiments/splitfusion_static_modes_v13/campaign_report_05" / name, "provenance/campaign/" + name)
    copy(Path(__file__), "source_reference/build_results_handoff.py")
    (OUT / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    versions = {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in ("numpy", "matplotlib")}}
    (OUT / "ENVIRONMENT.json").write_text(json.dumps(versions, indent=2) + "\n")
    sys.path.insert(0, str(OUT / "scripts"))
    import export_feedback
    import quantile_table
    export_feedback.export(OUT / "data")
    quantile_table.build(OUT / "data/QUANTILES_P50_P95_P99.csv")
    print(json.dumps({"package": str(OUT), "copied_files": len(manifest),
                      "bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
