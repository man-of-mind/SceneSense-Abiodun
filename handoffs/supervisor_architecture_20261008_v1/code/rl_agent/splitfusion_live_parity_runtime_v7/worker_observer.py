"""Bounded genuine child CPU/timeline evidence; no new thread or work queue."""
from contextlib import contextmanager
import json
from pathlib import Path

from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W

FACTORY = "rl_agent.splitfusion_live_parity_runtime_v7.worker_observer.ObservedClient"


class ObservedClient(W.Client):
    def __init__(self, *, _observation, **kwargs):
        self.observation = _observation
        self.close_record = None
        self.observation["clients"].append(self)
        super().__init__(**kwargs)

    def _observe(self, identity, result, error=None):
        state = self.observation
        try:
            if len(state["requests"]) >= state["max_frames"]:
                raise ValueError("CPU process observation bound exceeded")
            row = {"identity": dict(identity), "status": result.get("status", "FAILED_REQUEST"),
                "worker": result.get("worker"), "process_context": result.get("process_context"),
                "rows": result.get("rows"), "ipc": result.get("ipc"), "error": error}
            # Own only primitive timing/identity metadata, never image buffers.
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
def observe_cpu_worker(*, attempt_dir, max_frames, expected_sources, drain_check):
    if type(max_frames) is not int or not 1 <= max_frames <= 1000:
        raise ValueError("bounded worker observation required")
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
                    exitcode=exitcode, failed=getattr(client, "failed", True)))
            complete = (not failed and drained and stopped and len(clients) == 1
                and clients[0]["exitcode"] == 0 and clients[0]["close"] is not None
                and not clients[0]["failed"] and not state["dropped"] and not state["errors"]
                and all(row["status"] == "OK" for row in state["requests"]))
            write_x(path, dict(schema="splitfusion.cpu_gt_process_timeline.v7",
                factory=FACTORY, sources=dict(expected_sources), max_frames=max_frames,
                clock="CLOCK_MONOTONIC_RAW", cpu_clock="thread_time_ns",
                complete=complete, worker_drained=drained, child_stopped=stopped,
                clients=clients, requests=state["requests"], dropped=state["dropped"], errors=state["errors"],
                new_threads=0, new_queues=0, observation_cost_included=True))
            if not failed and not complete:
                raise RuntimeError("CPU worker process timeline incomplete")
        except Exception:
            if not failed:
                raise
