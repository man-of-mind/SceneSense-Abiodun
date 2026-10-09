"""Edge: the evaluator submit path can no longer suppress map publication of a completed prediction.

Retained order on the edge process thread (``EarlyGtProcessorV1.process`` then ``process_loop``):
    compute (prediction materialized on CPU) -> evaluator.submit(ticket) -> build map update -> (return)
    -> publisher.publish(update)  [direct map branch]
``submit`` is ``put_nowait``. On ``queue.Full`` the retained branch emits an INFRASTRUCTURE_FAULT feedback
synchronously (``_emit`` -> ``control.sendto`` to the UE) before returning. Normal case: bounded (one small UDP send),
then publication proceeds. Defect: any exception escaping that branch (``sendto`` OSError, encode failure) propagates
out of ``process``; ``process_loop`` counts ``edge_processing_failed`` and SKIPS publication of a prediction that
already completed.

Narrow repair (this class only, installed inside the early-GT edge scope): ``submit`` calls the retained
implementation unchanged; an exception is counted, recorded and NOT re-raised, so the map update is still built and
published. No feedback is sent for that ticket by this path; the UE resolves it through its retained timeout. The
wall time spent inside ``submit`` is recorded per call (max and total), so a slow synchronous fault send is visible.
"""
import time


def build_guarded_evaluator(base):
    class GuardedEarlyGtEvaluatorM3(base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.counters.update(m3_submit_calls=0, m3_submit_guarded_errors=0, m3_submit_max_ns=0,
                                 m3_submit_total_ns=0, m3_overflow_branch_max_ns=0)
            self.m3_guarded_errors = []

        def submit(self, ticket):
            started = time.perf_counter_ns()
            overflow_before = self.counters.get("queue_overflow", 0)
            try:
                return super().submit(ticket)
            except Exception as exc:  # noqa: BLE001 - recorded; publication of the completed prediction continues
                with self._lock:
                    self.counters["m3_submit_guarded_errors"] += 1
                    if len(self.m3_guarded_errors) < 64:
                        self.m3_guarded_errors.append(dict(frame_id=int(ticket.context.frame_id),
                                                           error=f"{type(exc).__name__}: {exc}"[:200],
                                                           wall_ns=time.time_ns()))
                return None
            finally:
                elapsed = time.perf_counter_ns() - started
                with self._lock:
                    self.counters["m3_submit_calls"] += 1
                    self.counters["m3_submit_total_ns"] += elapsed
                    self.counters["m3_submit_max_ns"] = max(self.counters["m3_submit_max_ns"], elapsed)
                    if self.counters.get("queue_overflow", 0) != overflow_before:
                        self.counters["m3_overflow_branch_max_ns"] = max(self.counters["m3_overflow_branch_max_ns"],
                                                                         elapsed)

    return GuardedEarlyGtEvaluatorM3
