"""UE numerical-thread cap experiment: the one scoped intervention and its evidence.

Arms:

* **A (baseline)** -- the UE child's environment is exactly ``lifecycle.child_env()``
  (``os.environ`` without ``PYTHONPATH``); ``OMP_NUM_THREADS`` and
  ``MKL_NUM_THREADS`` stay as they were (unset on this host).
* **B (treatment)** -- the same environment plus ``OMP_NUM_THREADS=1`` and
  ``MKL_NUM_THREADS=1``, present from ``exec`` so every numerical library in the
  UE child reads it at initialization.  Nothing else changes: no inter-op
  setting, affinity, wait policy, MPS, precision, rendering, radio, queue or
  scheduling change.

Scope.  The cap must reach the UE Python child only.  CARLA, OAI, the sampler
and the parent keep the parent's environment (they are launched by the parent
or by the lifecycle module's own ``child_env``).  Everything the UE child
launches itself -- the direct map server, the target-SNR actuator, the T-tracer
readers, the edge ``docker compose`` command and the ``sudo docker`` helpers --
goes through ``subprocess.Popen``; :func:`install_child_environment_restore`
restores the cap variables to their baseline values (deleting them when they
were absent) in every such launch, in both arms, so the two arms run the same
code path.

Evidence (identical in both arms): the UE child's effective settings before any
numerical import and after the run (PyTorch intra-/inter-op threads,
``parallel_info``, ``omp_get_max_threads`` from the loaded OpenMP runtime,
``threadpoolctl`` pool inventory, OpenCV threads, loaded numerical libraries,
affinity, thread count, launch counters), and a parent-side inventory of the
thread-relevant environment of every role process on the host.

Nothing here touches the qualified timing instrumentation.  Importing this
module initializes nothing.
"""

from __future__ import annotations

import collections
import ctypes
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

SCHEMA = "scenesense.single_gpu_profiler.threadcap.v1"
CAP_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
TREATMENT = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
# A/B: the thread-cap experiment.  C/T: the radar-bundle experiment, both with
# the UE-only cap retained; T additionally engages the optimized radar bundle.
ARMS: dict[str, dict[str, str]] = {"A": {}, "B": dict(TREATMENT), "C": dict(TREATMENT),
                                   "T": dict(TREATMENT), "C2": dict(TREATMENT),
                                   "E": dict(TREATMENT)}
BUNDLES: dict[str, str] = {"A": "production", "B": "production", "C": "production",
                           "T": "optimized", "C2": "production", "E": "production"}
# C2/E: the edge evaluator-submission ordering experiment (cap on, production radar).
ORDERINGS: dict[str, str] = {"A": "serial", "B": "serial", "C": "serial", "T": "serial",
                             "C2": "serial", "E": "early"}
ORDERING_ARMS = ("C2", "E")             # arms that route the edge through the ordering entry
BUNDLE_SNAPSHOT_ARMS = ("C", "T", "C2", "E")   # arms registered after the bundle existed must record it
ARM_ENV = "SCENESENSE_THREADCAP_ARM"
RESTORE_ENV = "SCENESENSE_THREADCAP_RESTORE_JSON"
BUNDLE_ENV = "SCENESENSE_RADAR_BUNDLE"
MARKER_VARIABLES = (ARM_ENV, RESTORE_ENV, BUNDLE_ENV)
OBSERVED_VARIABLES = CAP_VARIABLES + (
    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_PROC_BIND",
    "OMP_PLACES", "OMP_DYNAMIC", "KMP_AFFINITY", "KMP_BLOCKTIME", "GOMP_CPU_AFFINITY",
    "GOMP_SPINCOUNT", "MKL_DYNAMIC", "MKL_THREADING_LAYER", "TBB_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES", "CUDA_MPS_PIPE_DIRECTORY", "PYTHONPATH")
NUMERICAL_LIBRARY = re.compile(r"/(lib(gomp|iomp5|omp|openblas|blas|lapack|mkl|tbb|tbbmalloc)"
                               r"[^/]*\.so[^/ ]*)$")
EVIDENCE_LEAF = "threadcap"
UE_SETTINGS_NAME = "UE_THREAD_SETTINGS_{stage}.json"
INVENTORY_NAME = "PROCESS_ENV_INVENTORY.jsonl"


class ThreadCapError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ThreadCapError(message)


# ---------------------------------------------------------------------------
# arm environments (parent side)
# ---------------------------------------------------------------------------


def baseline_values(base: Mapping[str, str]) -> dict[str, Optional[str]]:
    """The cap variables as the baseline environment has them (``None`` = absent)."""
    return {key: base.get(key) for key in CAP_VARIABLES}


def arm_environment(base: Mapping[str, str], arm: str) -> dict[str, str]:
    """The UE child's environment for ``arm`` (pure).

    Both arms carry the arm marker and the JSON of the baseline values so the
    child restores exactly those values in everything it launches.
    """
    require(arm in ARMS, f"unknown arm {arm!r}")
    env = dict(base)
    env.update(ARMS[arm])
    env[ARM_ENV] = arm
    env[RESTORE_ENV] = json.dumps(baseline_values(base), sort_keys=True)
    env[BUNDLE_ENV] = BUNDLES[arm]
    return env


class LifecycleArmProxy:
    """The lifecycle module with ``child_env`` replaced by the arm environment.

    ``run_one_cell`` calls ``lifecycle.child_env()`` only for the UE child; the
    module's ``start_carla`` calls its own module-global ``child_env`` and is not
    affected.  Every other attribute forwards unchanged.
    """

    def __init__(self, module: Any, arm: str) -> None:
        self._threadcap_module = module
        self._threadcap_arm = str(arm)

    def child_env(self) -> dict[str, str]:
        return arm_environment(self._threadcap_module.child_env(), self._threadcap_arm)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._threadcap_module, name)


def install_lifecycle_arm(supervisor: Any, arm: str) -> Callable[..., Any]:
    """Route ``supervisor.import_lifecycle_helper`` through :class:`LifecycleArmProxy`."""
    original = getattr(supervisor.import_lifecycle_helper, "__wrapped__",
                       supervisor.import_lifecycle_helper)

    def import_lifecycle_helper(config: Mapping[str, Any]) -> Any:
        return LifecycleArmProxy(original(config), arm)

    import_lifecycle_helper.__wrapped__ = original  # type: ignore[attr-defined]
    import_lifecycle_helper.arm = arm  # type: ignore[attr-defined]
    supervisor.import_lifecycle_helper = import_lifecycle_helper
    return import_lifecycle_helper


# ---------------------------------------------------------------------------
# child side: restore baseline values in everything the UE child launches
# ---------------------------------------------------------------------------


class _RestoreState:
    def __init__(self, restore: Mapping[str, Optional[str]]) -> None:
        self.restore = dict(restore)
        self.lock = threading.Lock()
        self.launches = 0
        self.restored = 0
        self.argv0: "collections.Counter[str]" = collections.Counter()


_STATE: dict[str, Optional[_RestoreState]] = {"state": None}
_POPEN_ENV_POSITION = 9        # Popen(args, bufsize, executable, stdin, stdout, stderr,
#                                preexec_fn, close_fds, shell, cwd, env, ...)


def restored_environment(env: Optional[Mapping[str, str]],
                         restore: Mapping[str, Optional[str]]) -> tuple[dict[str, str], int]:
    """A child's environment with the cap variables at their baseline values."""
    out = dict(os.environ if env is None else env)
    changed = 0
    for key, value in restore.items():
        if value is None:
            if key in out:
                del out[key]
                changed += 1
        elif out.get(key) != value:
            out[key] = value
            changed += 1
    for key in MARKER_VARIABLES:
        out.pop(key, None)
    return out, changed


def install_child_environment_restore(restore: Mapping[str, Optional[str]]) -> _RestoreState:
    """Patch ``subprocess.Popen.__init__`` once so every launch gets baseline values."""
    if _STATE["state"] is not None:
        return _STATE["state"]
    state = _RestoreState(restore)
    real_init = subprocess.Popen.__init__

    def __init__(self: Any, args: Any, *positional: Any, **keywords: Any) -> None:
        positional = list(positional)
        if len(positional) > _POPEN_ENV_POSITION:
            env, changed = restored_environment(positional[_POPEN_ENV_POSITION], state.restore)
            positional[_POPEN_ENV_POSITION] = env
        else:
            env, changed = restored_environment(keywords.get("env"), state.restore)
            keywords["env"] = env
        with state.lock:
            state.launches += 1
            state.restored += changed
            try:
                first = args if isinstance(args, (str, bytes)) else list(args)[0]
                state.argv0[Path(str(first)).name[:40]] += 1
            except Exception:  # noqa: BLE001 - counting only
                state.argv0["?"] += 1
        real_init(self, args, *positional, **keywords)

    __init__.__wrapped__ = real_init  # type: ignore[attr-defined]
    subprocess.Popen.__init__ = __init__  # type: ignore[assignment]
    _STATE["state"] = state
    return state


def restore_counters() -> dict[str, Any]:
    state = _STATE["state"]
    if state is None:
        return {"installed": False}
    with state.lock:
        return {"installed": True, "restore": dict(state.restore), "launches": state.launches,
                "variables_restored": state.restored, "argv0": dict(state.argv0)}


# ---------------------------------------------------------------------------
# child side: effective settings
# ---------------------------------------------------------------------------


def _proc_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def loaded_numerical_libraries(pid: str = "self") -> list[str]:
    text = _proc_text(f"/proc/{pid}/maps") or ""
    found = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 6 and NUMERICAL_LIBRARY.search(parts[-1]):
            found.add(parts[-1])
    return sorted(found)


def _status_field(name: str, pid: str = "self") -> Optional[str]:
    for line in (_proc_text(f"/proc/{pid}/status") or "").splitlines():
        if line.startswith(name + ":"):
            return line.split(":", 1)[1].strip()
    return None


def _openmp_max_threads(libraries: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path in libraries:
        name = Path(path).name
        if not (name.startswith("libgomp") or name.startswith("libiomp") or name.startswith("libomp")):
            continue
        try:
            lib = ctypes.CDLL(path)      # already loaded: no new library is mapped
            out[name] = int(lib.omp_get_max_threads())
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            out[name] = f"unavailable: {type(exc).__name__}"
    return out


def _threadpool_info() -> Any:
    try:
        import threadpoolctl

        return [{k: info.get(k) for k in ("user_api", "internal_api", "num_threads",
                                          "filepath", "version", "threading_layer")}
                for info in threadpoolctl.threadpool_info()]
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}: {exc}"


def settings_snapshot(stage: str, *, arm: Optional[str]) -> dict[str, Any]:
    """Effective thread settings of this process (stdlib only before any import;
    whatever numerical modules are already loaded afterwards -- never imports them)."""
    libraries = loaded_numerical_libraries()
    snapshot: dict[str, Any] = {
        "schema": SCHEMA, "record": "ue_thread_settings", "stage": stage, "arm": arm,
        "pid": os.getpid(), "ppid": os.getppid(), "executable": sys.executable,
        "argv": list(sys.argv), "python": sys.version.split()[0],
        "wall_ns": time.time_ns(), "raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        "environment": {key: os.environ.get(key) for key in OBSERVED_VARIABLES},
        "markers": {key: os.environ.get(key) for key in MARKER_VARIABLES},
        "affinity_cpus": len(os.sched_getaffinity(0)), "cpu_count": os.cpu_count(),
        "threads_now": _status_field("Threads"),
        "voluntary_ctxt_switches": _status_field("voluntary_ctxt_switches"),
        "nonvoluntary_ctxt_switches": _status_field("nonvoluntary_ctxt_switches"),
        "numerical_libraries_loaded": libraries,
        "openmp_max_threads": _openmp_max_threads(libraries),
        "modules_present": {name: name in sys.modules
                            for name in ("torch", "numpy", "cv2", "threadpoolctl")},
        "launch_restore": restore_counters(),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            snapshot["torch"] = {
                "version": torch.__version__,
                "intra_op_threads": int(torch.get_num_threads()),
                "inter_op_threads": int(torch.get_num_interop_threads()),
                "cuda_initialized": bool(torch.cuda.is_initialized()),
                "parallel_info": torch.__config__.parallel_info()}
        except Exception as exc:  # noqa: BLE001
            snapshot["torch"] = f"unavailable: {type(exc).__name__}: {exc}"
    cv2 = sys.modules.get("cv2")
    if cv2 is not None:
        try:
            framework = [line.strip() for line in cv2.getBuildInformation().splitlines()
                         if "Parallel framework" in line]
            snapshot["cv2"] = {"threads": int(cv2.getNumThreads()), "parallel_framework": framework}
        except Exception as exc:  # noqa: BLE001
            snapshot["cv2"] = f"unavailable: {type(exc).__name__}: {exc}"
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        snapshot["numpy"] = {"version": numpy.__version__}
    if torch is not None or numpy is not None:
        snapshot["threadpools"] = _threadpool_info()
    snapshot["cap_effective"] = cap_effective(snapshot)
    return snapshot


def cap_effective(snapshot: Mapping[str, Any]) -> Optional[bool]:
    """True when every observed OpenMP/MKL/PyTorch pool reports one thread.

    ``None`` before numerical modules exist (nothing to observe yet).
    """
    observed: list[int] = []
    torch = snapshot.get("torch")
    if isinstance(torch, Mapping):
        observed.append(int(torch["intra_op_threads"]))
    for value in (snapshot.get("openmp_max_threads") or {}).values():
        if isinstance(value, int):
            observed.append(value)
    pools = snapshot.get("threadpools")
    if isinstance(pools, list):
        for pool in pools:
            if pool.get("internal_api") in ("openmp", "mkl") and pool.get("num_threads"):
                observed.append(int(pool["num_threads"]))
    if not observed:
        return None
    return all(value == 1 for value in observed)


def write_snapshot(directory: Path, snapshot: Mapping[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / UE_SETTINGS_NAME.format(stage=str(snapshot["stage"]).upper())
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(snapshot), handle, indent=1, sort_keys=True, default=str)
        handle.write("\n")
    return path


def child_arm() -> tuple[str, dict[str, Optional[str]]]:
    """The arm and baseline values the parent handed to this child."""
    arm = os.environ.get(ARM_ENV)
    require(arm in ARMS, f"{ARM_ENV} must be one of {sorted(ARMS)}; got {arm!r}")
    restore = json.loads(os.environ.get(RESTORE_ENV) or "{}")
    require(set(restore) == set(CAP_VARIABLES), f"{RESTORE_ENV} must name exactly {CAP_VARIABLES}")
    for key, value in ARMS[arm].items():
        require(os.environ.get(key) == value,
                f"arm {arm} requires {key}={value!r} at exec; got {os.environ.get(key)!r}")
    if arm == "A":
        for key, value in restore.items():
            require(os.environ.get(key) == value,
                    f"arm A requires baseline {key}={value!r}; got {os.environ.get(key)!r}")
    require(os.environ.get(BUNDLE_ENV, BUNDLES[arm]) == BUNDLES[arm],
            f"arm {arm} requires {BUNDLE_ENV}={BUNDLES[arm]!r}")
    return arm, restore


# ---------------------------------------------------------------------------
# parent side: environment inventory of every role process
# ---------------------------------------------------------------------------


def read_environ(pid: int, *, sudo: bool = True) -> tuple[Optional[dict[str, str]], str]:
    """``/proc/<pid>/environ`` as a dict; read-only, ``sudo -n cat`` for root processes."""
    path = f"/proc/{pid}/environ"
    try:
        raw = Path(path).read_bytes()
        source = "direct"
    except PermissionError:
        if not sudo:
            return None, "permission_denied"
        completed = subprocess.run(["sudo", "-n", "cat", path], stdin=subprocess.DEVNULL,
                                   capture_output=True, timeout=5.0, check=False)
        if completed.returncode != 0:
            return None, f"sudo_failed_rc{completed.returncode}"
        raw, source = completed.stdout, "sudo"
    except (FileNotFoundError, ProcessLookupError):
        return None, "exited"
    out: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, _, value = item.partition(b"=")
            out[key.decode(errors="replace")] = value.decode(errors="replace")
    return out, source


def _parent_of(pid: int) -> Optional[int]:
    text = _proc_text(f"/proc/{pid}/stat")
    if not text:
        return None
    try:
        return int(text.rpartition(")")[2].split()[1])
    except (IndexError, ValueError):
        return None


class ProcessEnvironmentInventory:
    """Poll ``/proc`` and record, once per PID, the thread-relevant environment of
    every role process (sampler role rules) and of every descendant of the runner.

    Read-only.  Rows are appended create-only as JSONL; :meth:`summary` reports,
    per role, the distinct values of the cap variables and every process outside
    the UE child whose cap variables differ from the baseline (a scope violation).
    """

    def __init__(self, path: Path, *, root_pid: int, baseline: Mapping[str, Optional[str]],
                 classify: Callable[[str, str], Optional[str]], interval_s: float = 1.0,
                 budget: int = 400, proc: str = "/proc") -> None:
        self.path = Path(path)
        self.root_pid = int(root_pid)
        self.baseline = dict(baseline)
        self._classify = classify
        self._interval = float(interval_s)
        self._budget = int(budget)
        self._proc = proc
        self._seen: set[int] = set()
        self._rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="threadcap-inventory", daemon=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        self.errors = 0

    def start(self) -> "ProcessEnvironmentInventory":
        self._thread.start()
        return self

    def _descends_from_root(self, pid: int) -> bool:
        current: Optional[int] = pid
        for _ in range(12):
            if current is None or current <= 1:
                return False
            if current == self.root_pid:
                return True
            current = _parent_of(current)
        return False

    def _examine(self, pid: int) -> None:
        comm = (_proc_text(f"{self._proc}/{pid}/comm") or "").strip()
        cmdline = (_proc_text(f"{self._proc}/{pid}/cmdline") or "").replace("\0", " ").strip()
        if not comm and not cmdline:
            return
        role = self._classify(comm, cmdline)
        descendant = self._descends_from_root(pid)
        if role is None and not descendant:
            return
        environ, source = read_environ(pid)
        nspid = _status_field("NSpid", str(pid)) or ""
        row = {"pid": pid, "ppid": _parent_of(pid), "role": role, "descendant_of_runner": descendant,
               "comm": comm, "cmdline": cmdline[:200], "environ_source": source,
               "in_container": len(nspid.split()) > 1,
               "t_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
               "variables": None if environ is None else {k: environ.get(k) for k in OBSERVED_VARIABLES},
               "markers": None if environ is None else {k: environ.get(k) for k in MARKER_VARIABLES}}
        self._rows.append(row)
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._handle.flush()

    def scan(self) -> None:
        try:
            pids = sorted(int(p) for p in os.listdir(self._proc) if p.isdigit())
        except OSError:
            return
        examined = 0
        for pid in pids:
            if pid in self._seen:
                continue
            self._seen.add(pid)
            try:
                self._examine(pid)
            except Exception:  # noqa: BLE001 - inventory never interferes
                self.errors += 1
            examined += 1
            if examined >= self._budget:
                break

    def _run(self) -> None:
        while not self._stop.is_set():
            self.scan()
            self._stop.wait(self._interval)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=15.0)
        self.scan()                                   # one last pass (late processes)
        summary = self.summary()
        self._handle.write(json.dumps({"summary": summary}, sort_keys=True) + "\n")
        self._handle.close()
        return summary

    def summary(self) -> dict[str, Any]:
        per_role: dict[str, dict[str, set]] = collections.defaultdict(
            lambda: {key: set() for key in CAP_VARIABLES})
        violations, unreadable = [], []
        ue_rows = []
        for row in self._rows:
            role = row["role"] or ("runner_descendant" if row["descendant_of_runner"] else "other")
            variables = row.get("variables")
            if variables is None:
                unreadable.append({"pid": row["pid"], "role": role, "comm": row["comm"],
                                   "reason": row["environ_source"]})
                continue
            for key in CAP_VARIABLES:
                per_role[role][key].add(variables.get(key))
            if row["role"] == "ue":
                ue_rows.append(row)
                continue
            for key in CAP_VARIABLES:
                if variables.get(key) != self.baseline.get(key):
                    violations.append({"pid": row["pid"], "role": role, "comm": row["comm"],
                                       "variable": key, "value": variables.get(key),
                                       "baseline": self.baseline.get(key)})
        return {"rows": len(self._rows), "errors": self.errors,
                "roles": {role: {key: sorted(map(str, values)) for key, values in keys.items()}
                          for role, keys in sorted(per_role.items())},
                "ue_processes": [{"pid": r["pid"], "variables": {k: r["variables"].get(k)
                                                                 for k in CAP_VARIABLES}}
                                 for r in ue_rows],
                "scope_violations": violations, "unreadable": unreadable,
                "scope_ok": not violations and bool(ue_rows)}
