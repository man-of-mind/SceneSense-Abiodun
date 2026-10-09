"""Networkless, GPU-free smoke of the actual pinned edge image and children."""
import argparse
import json
from pathlib import Path


def run(sources_path, output):
    from .live_contract import canonical
    from .process import Client, verify_sources
    from .profiled_startup import check
    from .live_entries import edge
    from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
    import torch
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    document = json.loads(Path(sources_path).read_bytes())
    sources = document["source_sha256"]
    verify_sources(sources)
    archive_root = output / "archive"
    archive_root.mkdir()
    scorer = archive = None
    try:
        scorer = Client("scorer", dict(run_binding="a"*64, producer_binding="b"*64,
            max_jobs=2, max_bytes=128*1024*1024), directory=output / "scorer",
            writable_attempt_root=output, source_sha256=sources, cpu=5)
        archive = ArchiveClient(roots=[archive_root], max_jobs=2, max_bytes=128*1024*1024,
            max_frames=1, binding=dict(numeric_worker_source_digest=canonical(sources)))
        composition = check(output / "profiled")
        closes = dict(scorer=scorer.close(), archive=archive.close())
        verify_sources(sources)
        passed = (not torch.cuda.is_initialized() and scorer.process.returncode == 0
                  and archive.process.returncode == 0 and composition["actual_profiled_numeric_class"])
        result = dict(passed=passed, actual_image_cpu_probe=True,
            source_sha256=sources, cuda_initialized=torch.cuda.is_initialized(),
            live_services_started=False, composition=composition, closes=closes,
            worker_ready=dict(scorer=scorer.ready, archive=archive.ready))
        with (output / "RESULT.json").open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        print(json.dumps(dict(passed=passed, cuda_initialized=torch.cuda.is_initialized())))
        return passed
    except BaseException as error:
        with (output / "FAILURE.json").open("x") as stream:
            json.dump(dict(error=f"{type(error).__name__}: {error}"), stream)
        raise
    finally:
        for client in (scorer, archive):
            if client is not None and client.process.poll() is None:
                client.abort()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.sources, args.output) else 1)
