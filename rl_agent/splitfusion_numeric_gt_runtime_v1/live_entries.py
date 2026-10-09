"""Real Run-5 child and edge entry points for the numeric GT experiment."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from . import live_contract as C
from .live_host import write


_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_real_directory(path):
    """Open every directory component without following a symlink."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("absolute numeric evidence directory required")
    fd = os.open("/", _DIRECTORY_FLAGS)
    try:
        for name in path.parts[1:]:
            next_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _set_private_owner(fd, *, uid, gid, mode, process_uid, process_gid):
    before = os.fstat(fd)
    if (before.st_uid, before.st_gid) not in ((process_uid, process_gid), (uid, gid)):
        raise ValueError("numeric evidence has foreign owner")
    os.fchown(fd, uid, gid)
    os.fchmod(fd, mode)
    after = os.fstat(fd)
    if ((after.st_dev, after.st_ino, after.st_size) !=
            (before.st_dev, before.st_ino, before.st_size)
            or (after.st_uid, after.st_gid) != (uid, gid)
            or stat.S_IMODE(after.st_mode) != mode):
        raise ValueError("numeric evidence handoff changed inode/size or missed owner/mode")


def handoff_numeric_evidence(directory, scope_path, *, publisher_uid, publisher_gid):
    """Transfer only this launch's exact durable numeric evidence to its UE owner.

    Children are closed first. Every component is opened O_NOFOLLOW and must be
    a regular file or directory owned by this edge process or the registered
    publisher. No shared scratch ancestor is chown'd or made world-readable.
    """
    directory, scope_path = Path(directory), Path(scope_path)
    if (directory.name != "numeric_runtime" or scope_path.name != "numeric_edge_scope.json"
            or scope_path.parent != directory.parent
            or type(publisher_uid) is not int or type(publisher_gid) is not int
            or publisher_uid < 0 or publisher_gid < 0):
        raise ValueError("exact launch-owned numeric evidence and publisher required")
    process_uid, process_gid = os.geteuid(), os.getegid()
    count = {"files": 0, "directories": 0}

    def visit(fd):
        for name in os.listdir(fd):
            if name in ("", ".", ".."):
                raise ValueError("invalid numeric evidence child")
            observed = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(observed.st_mode):
                flags, mode = _DIRECTORY_FLAGS, 0o700
            elif stat.S_ISREG(observed.st_mode):
                flags, mode = _FILE_FLAGS, 0o600
            else:
                raise ValueError("numeric evidence contains symlink or special file")
            child = os.open(name, flags, dir_fd=fd)
            try:
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
                    raise ValueError("numeric evidence child changed during handoff")
                if mode == 0o700:
                    visit(child)
                    count["directories"] += 1
                else:
                    count["files"] += 1
                _set_private_owner(child, uid=publisher_uid, gid=publisher_gid,
                    mode=mode, process_uid=process_uid, process_gid=process_gid)
            finally:
                os.close(child)

    root = _open_real_directory(directory)
    try:
        visit(root)
        _set_private_owner(root, uid=publisher_uid, gid=publisher_gid,
            mode=0o700, process_uid=process_uid, process_gid=process_gid)
        count["directories"] += 1
    finally:
        os.close(root)
    parent = _open_real_directory(scope_path.parent)
    try:
        scope = os.open(scope_path.name, _FILE_FLAGS, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(scope).st_mode):
                raise ValueError("numeric edge scope is not a regular file")
            _set_private_owner(scope, uid=publisher_uid, gid=publisher_gid,
                mode=0o600, process_uid=process_uid, process_gid=process_gid)
        finally:
            os.close(scope)
    finally:
        os.close(parent)
    count["files"] += 1
    return count


def _normal_edge_exit(exc):
    return isinstance(exc, SystemExit) and exc.code in (None, 0)


def _run_edge_ordered(main, args):
    """The retained edge loop terminates normally via SystemExit(0) on SIGTERM."""
    try:
        return main(args)
    except SystemExit as exc:
        if _normal_edge_exit(exc):
            return 0
        raise


def _registration(path):
    value = json.loads(Path(path).read_bytes())
    if (value.get("id") != C.ID or value.get("transmitted_budget") != C.BUDGET
            or value.get("child_module") != C.CHILD or value.get("edge_module") != C.EDGE
            or value.get("live_metric_contract") != C.metrics()
            or value.get("registration_sha256") != C.canonical(
                {key: item for key, item in value.items() if key != "registration_sha256"})):
        raise ValueError("numeric live entry requires exact sealed registration")
    # The host registration already enumerates the complete worker closure.
    # Verify those exact bytes here without importing UE-only source geometry.
    from .process import verify_sources
    verify_sources(value["numeric_worker_sources"])
    for name in ("process.py", "edge_live.py", "scorer.py", "live_entries.py"):
        if C.PACKAGE + "/" + name not in value["numeric_worker_sources"]:
            raise ValueError("numeric registration omits live worker source")
    return value


@contextmanager
def preserve_numeric_evidence():
    """Copy after actual edge shutdown and before retained scratch teardown."""
    from rl_agent.splitfusion_single_gpu_profiler_v1 import seams_ue_v1 as S
    original = S.copy_edge_profile

    def copy(edge_scratch, destination):
        result = original(edge_scratch, destination)
        source = None if edge_scratch is None else Path(edge_scratch) / "profiler"
        if source is None or not source.is_dir():
            return dict(result, numeric_runtime_preserved=False, numeric_reason="edge not created")
        files = []
        numerical = source / "numeric_runtime"
        if numerical.is_dir():
            files.extend(p for p in numerical.rglob("*") if p.is_file())
        scope = source / "numeric_edge_scope.json"
        if scope.is_file():
            files.append(scope)
        if not files:
            return dict(result, numeric_runtime_preserved=False, numeric_reason="no numeric edge evidence")
        inventory = {}
        for path in files:
            if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                raise ValueError("symbolic numeric evidence refused")
            relative = path.relative_to(source)
            target = Path(destination) / relative
            digest = C.sha(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_symlink() or C.sha(target) != digest:
                    raise ValueError("existing copied numeric evidence differs")
            else:
                shutil.copyfile(path, target)
            if C.sha(target) != digest:
                raise ValueError("numeric evidence copy readback differs")
            inventory[str(relative)] = digest
        return dict(result, numeric_runtime_preserved=True, numeric_files=inventory)

    S.copy_edge_profile = copy
    try:
        yield
    finally:
        S.copy_edge_profile = original


def child(argv=None):
    from rl_agent.splitfusion_single_gpu_profiler_v1 import child_v1 as BASE
    from rl_agent.splitfusion_run5_live_avo_v1 import child_entry as CHILD
    from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
    from rl_agent.splitfusion_live_parity_repair_v1.runtime import install_ue_perception
    from rl_agent.splitfusion_live_parity_runtime_v1.pose import install_ue_pose
    from rl_agent.splitfusion_live_parity_runtime_v8.startup import preserve_startup
    from .ue_live import install_numeric_ue
    from .live_host import verify_prospective
    args = list(sys.argv[1:] if argv is None else argv)
    attempt = Path(BASE.attempt_dir_from_argv(args)).resolve(strict=True)
    config = BASE.load_config(attempt)
    if (config is None or config.get("qualification_id") != C.ID
            or config.get("sha256") != C.canonical({k: v for k, v in config.items() if k != "sha256"})):
        raise ValueError("sealed enabled numerical profiler configuration required")
    output = attempt.parent.parent.parent
    verify_prospective(output)
    registration = _registration(output / "registration.json")
    bind_legacy_namespace()
    original = CHILD.EDGE_MODULE
    CHILD.EDGE_MODULE = C.EDGE
    try:
        with preserve_numeric_evidence(), preserve_startup(attempt), install_ue_perception(), \
                install_ue_pose(attempt_dir=attempt, max_frames=C.BUDGET), \
                install_numeric_ue(attempt_dir=attempt, registration=registration):
            return CHILD.main(args)
    finally:
        CHILD.EDGE_MODULE = original


def edge(argv=None):
    from .process import Client
    from .edge_live import install_numeric_edge
    from rl_agent.splitfusion_live_gt_handoff_v1.archive import ArchiveClient
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1 as EP
    from rl_agent.splitfusion_single_gpu_profiler_v1 import edge_profiled_v1_ordering as ORDERED
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_prewarm_v2 as PW
    args = list(sys.argv[1:] if argv is None else argv)
    registration = _registration("/partition_output/registration.json")
    profile = EP.profile_path(args).parent
    directory = profile / "numeric_runtime"
    directory.mkdir(parents=True, exist_ok=False)
    archive_root = directory / "archive"
    archive_root.mkdir(exist_ok=False)
    sidecars = directory / "sidecars"
    sidecars.mkdir(exist_ok=False)
    source_digest = C.canonical(registration["numeric_worker_sources"])
    limits = registration["numeric_limits"]
    workers, state = [], None
    scorer = archive = None
    result = None
    error = None
    try:
        scorer = Client("scorer", dict(run_binding=registration["numeric_run_binding"],
            producer_binding=registration["numeric_producer_binding"],
            max_jobs=limits["retained_jobs"], max_bytes=limits["retained_bytes"]),
            directory=directory / "scorer_process", writable_attempt_root=directory,
            source_sha256=registration["numeric_worker_sources"], cpu=limits["scorer_cpu"],
            timeout_s=limits["process_timeout_s"])
        archive = ArchiveClient(roots=[archive_root], max_jobs=limits["retained_jobs"],
            max_bytes=limits["retained_bytes"], max_frames=C.BUDGET,
            binding=dict(numeric_run_binding=registration["numeric_run_binding"],
                         numeric_worker_source_digest=source_digest))
        sys.path.insert(0, "/cpu_partition_code")
        import edge_gate
        edge_gate.install("/partition_output", PW)
        socket_path = Path("/work/torch_cache/segmentation_evidence") / C.SOCKET_NAME
        with install_numeric_edge(socket_path=socket_path, scorer_client=scorer,
                archive_client=archive, archive_root_index=0,
                sidecar_directory=sidecars, max_frames=C.BUDGET,
                publisher_uid=registration["publisher_uid"], publisher_gid=registration["publisher_gid"]) as state:
            result = _run_edge_ordered(ORDERED.main, args)
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        failures = []
        for role, client in (("scorer", scorer), ("archive", archive)):
            if client is None:
                continue
            close = None
            try:
                close = client.close()
            except BaseException as exc:
                failures.append(f"{role}: {type(exc).__name__}: {exc}")
                # Each client stops only its own child. No process-name kills.
                if hasattr(client, "abort"):
                    client.abort()
            workers.append(dict(role=role, ready=client.ready, close=close,
                exitcode=client.process.returncode, source_sha256=registration["numeric_worker_sources"]))
            if role == "scorer":
                write(directory / "SCORER_PROCESS_TIMELINE.json", dict(events=client.events))
        write(profile / "numeric_edge_scope.json", dict(schema="splitfusion.numeric_gt.worker_scope.v1",
            workers=workers, state=state, returncode=result, error=error, close_errors=failures,
            numeric_worker_source_digest=source_digest, live_qualified=False))
        try:
            handoff_numeric_evidence(directory, profile / "numeric_edge_scope.json",
                publisher_uid=registration["publisher_uid"],
                publisher_gid=registration["publisher_gid"])
        except BaseException as exc:
            failures.append(f"numeric evidence ownership: {type(exc).__name__}: {exc}")
        if failures and error is None:
            raise RuntimeError("numerical edge shutdown failed: " + "; ".join(failures))
