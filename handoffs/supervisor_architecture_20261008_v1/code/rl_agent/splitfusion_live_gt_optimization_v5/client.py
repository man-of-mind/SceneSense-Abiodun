"""Own only process startup; reuse frozen V4 request/IPC/drain semantics."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from . import contract as C


class Client(W.Client):
    def __init__(self, *, expected_bootstrap_sources, attempt_dir, catalogue_dir,
                 max_frames, expected_sources, timeout_s=30., allow_replay_clock=False,
                 source_attempt_name=None, semantic_attempt_dir=None):
        W.require(expected_bootstrap_sources == C.sources(), "CPU worker bootstrap binding differs")
        W.require(expected_sources == W.sources(), "CPU worker body binding differs")
        W.require(0 < timeout_s <= 120, "bounded worker timeout required")
        self.bootstrap_sources = dict(expected_bootstrap_sources)
        self.timeout = float(timeout_s)
        self.failed = self.closed = False
        self.count = 0
        self.lock = threading.Lock()
        self.events = []
        root = Path(attempt_dir)
        root.mkdir(parents=True, exist_ok=False)
        parent, child = socket.socketpair()
        self.socket = parent
        try:
            self.console = (root / "worker.log").open("xb")
        except BaseException:
            parent.close()
            child.close()
            raise
        env = dict(os.environ, **C.ENVIRONMENT, PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONPATH", None)
        try:
            self.process = subprocess.Popen([sys.executable, "-u", "-m", C.BOOTSTRAP,
                "--fd", str(child.fileno())], pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=self.console, stderr=subprocess.STDOUT,
                env=env, cwd=Path(__file__).resolve().parents[2])
        except BaseException:
            parent.close()
            child.close()
            self.console.close()
            raise
        child.close()
        try:
            self.expected_context = W.scheduling_context()
            self.ready = self._exchange(dict(command="INIT", attempt_dir=str(root / "engine"),
                catalogue_dir=str(catalogue_dir), max_frames=max_frames,
                expected_sources=expected_sources, allow_replay_clock=allow_replay_clock,
                source_attempt_name=source_attempt_name,
                semantic_attempt_dir=None if semantic_attempt_dir is None else str(semantic_attempt_dir),
                expected_process_context=self.expected_context))
        except BaseException:
            # The outer runtime does not own self yet if this constructor
            # raises. It is our responsibility to terminate and join here.
            self.abort()
            raise

    def _exchange(self, meta, body=b"", descriptor=None):
        try:
            result = super()._exchange(meta, body=body, descriptor=descriptor)
            context = (result.get("result", {}).get("process_context")
                       if meta.get("command") == "RUN" else result.get("process_context"))
            C.validate_context(context, self.bootstrap_sources)
            W.require(context.get("pid") == self.process.pid, "foreign CPU worker startup identity")
            if meta.get("command") == "INIT":
                W.require(result.get("pid") == self.process.pid
                    and result.get("cuda_initialized") is False,
                    "CPU worker READY identity/CUDA status differs")
            W.require(context.get("scheduling") == self.expected_context,
                      "CPU worker inherited scheduling differs")
            return result
        except BaseException:
            self.abort()
            raise
