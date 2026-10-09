"""Bounded startup evidence, never a retry or a replacement for an exception."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_x(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload if isinstance(payload, bytes) else
                     (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def edge_entry(argv=None):
    """Stdlib-only outer edge guard; preserve failures even before imports."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args.count("--ready-file") != 1:
        raise ValueError("explicit single ready-file required")
    ready = Path(args[args.index("--ready-file") + 1])
    if ready != Path("/work/torch_cache/ready.json"):
        raise ValueError("foreign edge startup evidence root")
    directory = ready.parent / "profiler" / "startup_v2"
    write_x(directory / "STARTED.json", {"pid": os.getpid(), "wall_ns": time.time_ns()})
    try:
        from .entries import edge
        result = edge(args)
    except BaseException as exc:
        # The original exception/exit status remains authoritative even if disk
        # evidence fails. Stderr is additionally captured by the host wrapper.
        try:
            write_x(directory / "FAILED.json", {"type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "wall_ns": time.time_ns()})
        except Exception as evidence_error:
            print(f"edge startup evidence failure: {evidence_error!r}", file=sys.stderr, flush=True)
        raise
    write_x(directory / "EXITED.json", {"returncode": result, "wall_ns": time.time_ns()})
    return result


def _preserve(attempt, temporary_dir, error, endpoint):
    directory = Path(attempt) / "parity_startup_diagnostics"
    errors = []
    log = Path(temporary_dir) / "edge_launcher.log"
    if log.is_file():
        try:
            write_x(directory / "edge_launcher.log", log.read_bytes())
        except Exception as exc:
            errors.append(f"launcher: {exc!r}")
    if error is not None:
        try:
            command = ["sudo", "-n", "docker", "logs", "--tail", "200", "--timestamps", "oai-perception-rx"]
            done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, timeout=10, check=False)
            write_x(directory / "edge_container.log", done.stdout)
            write_x(directory / "docker_log_result.json", {"command": command, "returncode": done.returncode})
        except Exception as exc:
            errors.append(f"docker: {exc!r}")
    scratch = endpoint.get("edge_scratch")
    if scratch:
        source = Path(scratch) / "profiler" / "startup_v2"
        for name in ("STARTED.json", "FAILED.json", "EXITED.json"):
            try:
                path = source / name
                if path.is_file() and not path.is_symlink():
                    write_x(directory / name, path.read_bytes())
            except Exception as exc:
                errors.append(f"edge_record:{name}: {exc!r}")
    write_x(directory / "HOST_STARTUP_RESULT.json", {
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "exception_chain": None if error is None else traceback.format_exc(),
        "edge_ready": error is None, "capture_errors": errors, "wall_ns": time.time_ns()})


@contextmanager
def preserve_startup(attempt):
    from rl_agent.splitfusion_direct_edge_map_v1 import adapter_direct_v1 as D
    original = D.start_direct_live_edge

    def start(campaign, cell, temporary_dir):
        try:
            result = original(campaign, cell, temporary_dir)
        except BaseException as original_error:
            try:
                _preserve(attempt, temporary_dir, original_error, D._ENDPOINT)
            except Exception as evidence_error:
                print(f"startup diagnostic preservation failed: {evidence_error!r}", file=sys.stderr, flush=True)
            raise
        _preserve(attempt, temporary_dir, None, D._ENDPOINT)
        return result

    D.start_direct_live_edge = start
    try:
        yield
    finally:
        D.start_direct_live_edge = original
