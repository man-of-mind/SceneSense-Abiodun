"""Genuine handoff process receipts and one startup-only archive inspection."""
from contextlib import contextmanager
import json
import os
from pathlib import Path

from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H
from rl_agent.splitfusion_live_gt_handoff_v1.client import Client

FACTORY = "rl_agent.splitfusion_live_parity_runtime_v8.worker_observer.ObservedClient"


def archive_startup(parent_pid):
    """INIT already started its writer; inspect only that owned descendant."""
    children = Path(f"/proc/{parent_pid}/task/{parent_pid}/children").read_text().split()
    if len(children) != 1 or not children[0].isdigit():
        raise ValueError("one owned archive child must exist before sensors")
    pid = int(children[0])
    root = Path(f"/proc/{pid}")
    status = dict(line.split(":", 1) for line in (root / "status").read_text().splitlines() if ":" in line)
    if int(status["PPid"].strip()) != parent_pid:
        raise ValueError("archive process parent differs")
    environ = {}
    for item in (root / "environ").read_bytes().split(b"\0"):
        key, separator, value = item.partition(b"=")
        if separator and key.decode() in H.ENVIRONMENT:
            environ[key.decode()] = value.decode()
    if environ != H.ENVIRONMENT:
        raise ValueError("archive process effective CPU environment differs")
    command = (root / "cmdline").read_bytes().split(b"\0")
    if b"rl_agent.splitfusion_live_gt_handoff_v1.archive" not in command:
        raise ValueError("owned descendant is not registered archive writer")
    return dict(pid=pid, parent_pid=parent_pid, environment=environ,
        uid=[int(item) for item in status["Uid"].split()],
        gid=[int(item) for item in status["Gid"].split()],
        scheduling=dict(cpus=sorted(os.sched_getaffinity(pid)), nice=os.getpriority(os.PRIO_PROCESS, pid)),
        inspection="PROCFS_BEFORE_SENSOR_STARTUP", numerical_library_settings_observed=False)


class ObservedClient(Client):
    def __init__(self, *, _observation, **kwargs):
        self.observation = _observation
        self.close_record = self.archive_startup = None
        self.observation["clients"].append(self)
        try:
            super().__init__(**kwargs)
            self.archive_startup = archive_startup(self.process.pid)
        except BaseException:
            if getattr(self, "process", None) is not None and self.process.poll() is None:
                self.abort()
            raise

    def _observe(self, identity, result, error=None):
        state = self.observation
        try:
            if len(state["requests"]) >= state["max_frames"]:
                raise ValueError("CPU process observation bound exceeded")
            row = {"identity": dict(identity), "status": result.get("status", "FAILED_REQUEST"),
                "worker": result.get("worker"), "process_context": result.get("process_context"),
                "rows": result.get("rows"), "ipc": result.get("ipc"), "error": error}
            state["requests"].append(json.loads(W.canonical(row)))
        except Exception as exc:
            state["dropped"] += 1
            if len(state["errors"]) < 16:
                state["errors"].append(f"{type(exc).__name__}: {exc}")

    def run(self, **kwargs):
        try:
            result = super().run(**kwargs)
        except BaseException as exc:
            self._observe(kwargs.get("kwargs", {}).get("identity", {}),
                getattr(exc, "worker_result", {}) or {}, f"{type(exc).__name__}: {exc}")
            raise
        self._observe(result["identity"], result)
        return result

    def close(self):
        result = super().close()
        self.close_record = result
        return result


@contextmanager
def observe_cpu_worker(*, attempt_dir, max_frames, expected_sources,
                       expected_bootstrap_sources, handoff_binding, drain_check):
    if type(max_frames) is not int or not 1 <= max_frames <= 1000:
        raise ValueError("bounded worker observation required")
    if expected_bootstrap_sources != H.sources() or handoff_binding != H.binding():
        raise ValueError("handoff observer source binding differs")
    path = Path(attempt_dir) / "CPU_GT_PROCESS_TIMELINE.json"
    if path.exists():
        raise FileExistsError(path)
    state = dict(clients=[], requests=[], max_frames=max_frames, dropped=0, errors=[])
    def factory(**kwargs):
        if state["clients"]:
            raise ValueError("one observed CPU worker per scope")
        return ObservedClient(_observation=state, **kwargs)
    failed = False
    try:
        yield {"factory": factory, "state": state}
    except BaseException:
        failed = True
        raise
    finally:
        from rl_agent.splitfusion_live_parity_runtime_v1.entries import write_x
        try:
            drained = drain_check() is True
            clients = []
            stopped = bool(state["clients"])
            for client in state["clients"]:
                process = getattr(client, "process", None)
                exitcode = None if process is None else process.poll()
                stopped = stopped and process is not None and exitcode is not None
                clients.append(dict(pid=None if process is None else process.pid,
                    ready=getattr(client, "ready", None), close=client.close_record,
                    failure_close=getattr(client, "handoff_close", None) if client.close_record is None else None,
                    archive_startup=client.archive_startup, exitcode=exitcode,
                    failed=getattr(client, "failed", True)))
            archive = (clients[0]["close"] or {}).get("timing", {}).get("archive", {}) if clients else {}
            startup = (clients[0]["archive_startup"] or {}) if clients else {}
            complete = (not failed and drained and stopped and len(clients) == 1
                and clients[0]["exitcode"] == 0 and clients[0]["close"] is not None
                and not clients[0]["failed"] and not state["dropped"] and not state["errors"]
                and archive.get("child_pid") == startup.get("pid") and bool(startup)
                and archive.get("child_returncode") == 0 and archive.get("failed") is False
                and all(row["status"] == "OK" for row in state["requests"]))
            write_x(path, dict(schema="splitfusion.gt_handoff_process_timeline.v8", factory=FACTORY,
                sources=dict(expected_sources), bootstrap_sources=dict(expected_bootstrap_sources),
                handoff_binding=handoff_binding, max_frames=max_frames,
                clock="CLOCK_MONOTONIC_RAW", cpu_clock="thread_time_ns", complete=complete,
                worker_drained=drained, child_stopped=stopped, clients=clients,
                requests=state["requests"], dropped=state["dropped"], errors=state["errors"],
                new_threads=0, new_queues=0, counts_scope="OBSERVER_ONLY", observation_cost_included=True))
            if not failed and not complete:
                raise RuntimeError("GT handoff process timeline incomplete")
        except Exception:
            if not failed:
                raise
