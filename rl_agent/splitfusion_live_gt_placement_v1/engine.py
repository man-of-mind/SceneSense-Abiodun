"""Sealed v8 CONTROL engine with dedicated-child placement around requests."""
import hashlib
import os
import time

from rl_agent.splitfusion_live_gt_optimization_v8.engine import ControlEngine
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W
from rl_agent.splitfusion_live_parity_repair_v1 import target_transport as T
from .placement import Placement, process_threads, scheduling


class Engine(ControlEngine):
    def __init__(self, *, placement_cpu, placement_original_context, **kwargs):
        self.placement = None
        self.placement_scope_record = None
        self.native_pool_startup = None
        super().__init__(**kwargs)
        try:
            W.require(self.archiver is not None and self.archiver.process.poll() is None
                      and self.count == 0, "archive READY required before GT placement")
            self.placement = Placement(requested_cpu=placement_cpu,
                original_context=placement_original_context, archive_pid=self.archiver.process.pid,
                max_frames=self.max_frames, native_pool_startup=self.native_pool_startup)
            self.placement.start()
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def _start_archive(self, roots):
        super()._start_archive(roots)
        # OpenCV's TBB workers cache a full startup mask and otherwise appear
        # lazily during the first real GT request. Materialize that unchanged
        # pool before control GC and placement, symmetrically in both arms.
        # No saved frame, producer history or retained numerical result is used.
        import cv2
        import numpy as np
        import torch
        before = process_threads(os.getpid())
        original = scheduling()
        caps = dict(opencv_threads=cv2.getNumThreads(), torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads())
        start, cpu = W.raw(), time.thread_time_ns()
        thread_counts = []
        pool_ready = False
        for _ in range(8):
            outputs = cv2.connectedComponentsWithStats(np.zeros((432, 768), dtype=np.uint8), connectivity=8)
            W.require(outputs[0] == 1 and outputs[1].shape == (432, 768),
                      "native pool startup operation returned unexpected dimensions")
            del outputs
            thread_counts.append(len(process_threads(os.getpid())))
            if thread_counts[-1] >= caps["opencv_threads"]:
                pool_ready = True
                break
        end, cpu_end = W.raw(), time.thread_time_ns()
        after = process_threads(os.getpid())
        W.require(scheduling() == original and caps == dict(opencv_threads=cv2.getNumThreads(),
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads()),
            "native pool startup changed scheduling or numerical caps")
        self.native_pool_startup = dict(operation="cv2.connectedComponentsWithStats",
            input="synthetic_all_zero", shape=[432, 768], dtype="uint8", connectivity=8,
            calls=len(thread_counts), max_calls=8, thread_counts_after_calls=thread_counts,
            required_threads=caps["opencv_threads"], pool_ready=pool_ready,
            phase="after_archive_READY_before_control_startup_gc_and_placement",
            numerical_caps=caps, threads_before=before, threads_after=after,
            start_raw_ns=start, end_raw_ns=end, wall_ns=end-start, thread_cpu_ns=cpu_end-cpu,
            retained_frame_input=False, retained_output=False, measured_request=False)
        W.require(pool_ready, "native pool did not become ready within eight synthetic startup calls")

    def close(self):
        policy = self.placement
        if policy is None or policy.closed and self.placement_scope_record is not None:
            result = super().close()
            if result is not None and self.placement_scope_record is not None:
                result["runtime_placement_scope"] = self.placement_scope_record
            return result
        observed_error = native_error = cleanup_error = None
        result = None
        try:
            if not policy.closed:
                policy.before_drain()
        except BaseException as error:
            observed_error = error
        # The retained optimization scope compares exit settings with its
        # original startup settings. Restore before that unchanged close
        # validator runs; archive has retained the original mask throughout.
        try:
            policy.restore()
        except BaseException as error:
            cleanup_error = error
        try:
            result = super().close()
        except BaseException as error:
            native_error = error
        finally:
            try:
                policy.restore()
            except BaseException as error:
                cleanup_error = error
        try:
            scope = policy.snapshot(include_requests=True)
            scope.update(native_close_succeeded=native_error is None,
                native_close_error=None if native_error is None else repr(native_error),
                observation_error=None if observed_error is None else repr(observed_error),
                cleanup_error=None if cleanup_error is None else repr(cleanup_error),
                archive_returncode=self.archiver.process.poll())
            payload = W.canonical(scope)
            path = self.attempt / "PLACEMENT_SCOPE.json"
            T._write_x(path, payload)
            self.placement_scope_record = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(),
                restored=scope["restored"], native_close_succeeded=native_error is None)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        primary = native_error or observed_error or cleanup_error
        if primary is not None:
            other = next((error for error in (observed_error, cleanup_error) if error is not None
                          and error is not primary), None)
            if other is not None:
                raise primary from other
            raise primary
        if result is not None:
            result["runtime_placement_scope"] = self.placement_scope_record
        return result
