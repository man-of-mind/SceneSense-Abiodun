"""M3.1 v7 GT worker trampoline: the FREEZE exception-lifetime repair in the prep role only, then the unchanged v4
worker trampoline (v3 implementation: role checks, GC policy, tracing, producer switch, capacity/pose substitutions).

    python -m rl_agent.splitfusion_m3_route_b_v7.worker_entry <the v3 worker_entry arguments>

prep:  ``splitfusion_gt_freeze_lifetime_v1.repair.install(<--trace-dir>)`` makes ``prep_core.PrepCore`` resolve to
       ``PrepCoreLifetime`` in this process, before the retained worker constructs (and traces) its core; restored on
       exit. A fatal FREEZE prints the chained exception and keeps ``<trace-dir>/freeze_failure_v1``.
score: nothing is installed; the retained trampoline runs alone.

Only ``--role`` and ``--trace-dir`` are read here; the retained parser validates every argument. The prep role writes
``prep_lifetime_binding.json`` (what was installed, and that it was restored) when it exits; a failure to write it is
reported on stderr and never replaces the worker's own exception.
"""
import os
if __name__ == "__main__":
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})

import argparse
import json
from pathlib import Path
import sys

from . import contract as C


def role_and_trace(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--role", choices=("prep", "score"), required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    return known.role, known.trace_dir


def _write_record(path, record):
    try:
        with open(path, "x") as stream:
            json.dump(record, stream, indent=1, sort_keys=True)
            stream.write("\n")
    except BaseException as error:          # diagnostics never replace the worker's own outcome
        try:
            print(f"PREP_LIFETIME_RECORD_WRITE_FAILED: {type(error).__name__}: {str(error)[:512]}",
                  file=sys.stderr, flush=True)
        except BaseException:
            pass


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    role, trace_dir = role_and_trace(args)
    from rl_agent.splitfusion_m3_route_b_v4 import worker_entry as retained
    if role != "prep":
        return retained.main(args)
    from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
    from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as R
    record = dict(schema="splitfusion.m3_route_b_v7.prep_lifetime_binding.v1", role=role, pid=os.getpid(),
                  entry_module=C.WORKER_ENTRY, contract_module=C.__name__, retained_trampoline=retained.__name__,
                  repair_module=R.__name__, repair_file=str(Path(R.__file__).resolve()),
                  repair_sha256=C.sha(Path(R.__file__)), failure_directory=str(trace_dir),
                  installed=None, restored=None, exit=None)
    try:
        with R.install(str(trace_dir)):
            record["installed"] = (PC.PrepCore is R.PrepCoreLifetime
                                   and R.PrepCoreLifetime.failure_directory == str(trace_dir))
            if not record["installed"]:
                raise RuntimeError("REFUSED: prep FREEZE lifetime repair is not the constructed core class")
            return retained.main(args)
    except BaseException as error:
        record["exit"] = f"{type(error).__name__}: {error}"[:500]
        raise
    finally:
        record["restored"] = PC.PrepCore is R.ORIGINAL_PREP_CORE and R.PrepCoreLifetime.failure_directory is None
        _write_record(trace_dir / C.PREP_BINDING_RECORD, record)


if __name__ == "__main__":
    raise SystemExit(main())
