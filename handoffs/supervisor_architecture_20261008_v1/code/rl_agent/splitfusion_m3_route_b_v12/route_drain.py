"""M3.2 v12 bounded end-of-route terminal drain (UE child collector, main route thread only).

The V8 capture completed the route but failed G1 ``one_terminal_per_sent_frame`` on its last sent frame (8595).
Cause, in the retained base ``PassiveSplitCollector.finish`` (``ue_route_b_split_cell_adapter_v1.py:2403-2424``): it
waits only on the *pending slot* (``prepared_queue.depth()``; "a frame already handed to a worker is never
interrupted"), then on the ``feedback.pending`` view for 2 s, then forces ``record_expired`` and sets ``stop_event``
-- and only afterwards joins the preparation worker.  A token already executing registers its ACK obligation
(``on_commit`` -> ``register_capture``) *after* that forced expiry and after the feedback worker exited, so no row is
ever written to ``map_feedback.csv``: neither the ACK nor a timeout.  G1 counts exactly those rows
(``phase6_timing_characterization_v2.py:328-332``).  In V8 frame 8595's map ACK arrived at +80 ms, well inside its
own 500-ms horizon, but 110 ms after ``finish()`` had already started.

This layer closes that window prospectively, before ``super().finish()`` and therefore before the base's forced
expiry and ``stop_event``, while the preparation, feedback, result and evaluation threads are all still alive:

1. **Stop new submissions.**  ``_process_token`` records any further token as ``DROPPED_END_OF_ROUTE_DRAIN`` and
   returns without preparing or sending it (so no obligation can be created after the drain decision).  The pending
   slot is NOT closed early: ``take()`` would then return immediately and the worker would spin.
2. **Wait for the in-flight token**, bounded: the one token already inside ``_process_token`` finishes registering,
   sending and accounting its frame.  Claiming the slot and setting the drain flag share one lock, so a token is
   either counted in flight (and waited for) or dropped -- never half of each.
3. **Wait for outstanding terminals**, bounded by the obligations' OWN registered ACK horizons
   (``ack_timeout_at`` = capture + ``ack_timeout_ms``, unchanged) plus a small settle margin, and by an absolute cap.
   The retained feedback worker keeps writing real ACK and real timeout rows throughout; this layer writes none.
4. Anything still open at the cap is closed by the retained ``record_expired`` with the REAL clock, which writes an
   honest ``FEEDBACK_TIMEOUT`` / ``TIMEOUT_NO_ACK`` terminal (``ue_ledger.py:274-307``: ``accepted=False``).

What this does NOT do: it does not extend the 170-ms decision deadline (reward tickets are resolved by the retained
kernel and are untouched), does not reset or re-stamp any timestamp, does not drop or re-order the final *sent*
frame, does not mark a missing terminal successful, and does not change the ACK horizon.  Evidence is written to
``<attempt>/m3_route/END_OF_ROUTE_DRAIN.json``.
"""
from contextlib import contextmanager
from pathlib import Path
import threading
import time

from rl_agent.splitfusion_m3_route_b_v3 import route_completion as RC

from . import contract as C

SCHEMA = "splitfusion.m3_route_b_v12.end_of_route_drain.v1"
DROP_STATUS = "DROPPED_END_OF_ROUTE_DRAIN"


def build_drain_layer(inherited):
    """The v12 drain layer, placed directly above the unchanged v3 ``M3CollectorLayer``."""

    class M3EndOfRouteDrainLayer(inherited):
        def __init__(self, **kwargs):
            self._m3_drain_lock = threading.Lock()
            self._m3_draining = False
            self._m3_in_flight = 0
            self._m3_idle = threading.Event()
            self._m3_idle.set()
            self._m3_dropped_tokens = []
            self._m3_drain_evidence = None
            super().__init__(**kwargs)

        # -- 1/2. new submissions stop; the token already executing is never interrupted
        def _process_token(self, token):
            with self._m3_drain_lock:
                draining = self._m3_draining
                if not draining:
                    self._m3_in_flight += 1
                    self._m3_idle.clear()
                else:
                    self._m3_dropped_tokens.append(int(token["frame_id"]))
            if draining:
                self.dropped += 1
                self.transport_counters.bump("m3_end_of_route_drain_skipped")
                self._append_row({**token, "prepare_status": DROP_STATUS, "processing_late": 1})
                return None
            try:
                return super()._process_token(token)
            finally:
                with self._m3_drain_lock:
                    self._m3_in_flight -= 1
                    if self._m3_in_flight == 0:
                        self._m3_idle.set()

        # -- 3/4. the drain itself
        def _m3_open_obligations(self):
            ledger = self.feedback._ledger
            with ledger.lock:
                return {key: float(row["ack_timeout_at"]) for key, row in ledger.pending.items()}

        def _m3_end_of_route_drain(self):
            started_raw, started_wall = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), time.time_ns()
            evidence = dict(schema=SCHEMA, started_raw_ns=started_raw, started_wall_ns=started_wall,
                            ack_timeout_s=float(self.ack_timeout_s), settle_margin_s=C.DRAIN_SETTLE_MARGIN_S,
                            absolute_cap_s=C.DRAIN_ABSOLUTE_CAP_S, in_flight_cap_s=C.DRAIN_IN_FLIGHT_CAP_S,
                            sent_at_drain_start=int(self.sent),
                            open_at_start=sorted(self._m3_open_obligations()),
                            inbox_at_start=self.feedback._inbox.qsize())
            with self._m3_drain_lock:
                self._m3_draining = True
                evidence["in_flight_at_drain_start"] = int(self._m3_in_flight)
            absolute = time.monotonic() + C.DRAIN_ABSOLUTE_CAP_S
            in_flight_deadline = min(absolute, time.monotonic() + C.DRAIN_IN_FLIGHT_CAP_S)
            while not self._m3_idle.is_set() and time.monotonic() < in_flight_deadline:
                self._m3_idle.wait(timeout=0.05)
            evidence["in_flight_completed"] = bool(self._m3_idle.is_set())
            evidence["sent_after_in_flight"] = int(self.sent)
            evidence["open_after_in_flight"] = sorted(self._m3_open_obligations())
            polls = 0
            while time.monotonic() < absolute:
                open_now = self._m3_open_obligations()
                if not open_now and self.feedback._inbox.empty():
                    break
                if open_now and time.time() >= max(open_now.values()) + C.DRAIN_SETTLE_MARGIN_S:
                    break                       # every open obligation is past its own registered horizon
                polls += 1
                time.sleep(0.05)
            evidence["polls"] = polls
            evidence["waited_s"] = (time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - started_raw) / 1e9
            remaining = self._m3_open_obligations()
            evidence["open_before_expiry"] = sorted(remaining)
            evidence["expired_on_real_clock"] = int(self.feedback.record_expired()) if remaining else 0
            summary = self.feedback.summary()
            with self._m3_drain_lock:
                dropped_tokens = list(self._m3_dropped_tokens)
            evidence.update(open_at_end=sorted(self._m3_open_obligations()),
                            inbox_at_end=self.feedback._inbox.qsize(),
                            ledger_summary=summary, sent_at_end=int(self.sent),
                            tokens_dropped_during_drain=dropped_tokens,
                            closed_equals_sent=int(summary["obligations_closed"]) == int(self.sent),
                            ended_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), ended_wall_ns=time.time_ns(),
                            deadline_extended=False, timestamps_reset=False, final_sent_frame_omitted=False,
                            missing_terminal_declared_successful=False)
            self._m3_drain_evidence = evidence
            return evidence

        def finish(self):
            evidence, error = None, None
            try:
                evidence = self._m3_end_of_route_drain()
            except BaseException as exc:          # the drain never replaces the retained teardown
                error = f"{type(exc).__name__}: {exc}"[:500]
                with self._m3_drain_lock:
                    self._m3_draining = True
            finally:
                RC._write_json(Path(self.attempt_dir) / RC.EVIDENCE_DIR / "END_OF_ROUTE_DRAIN.json",
                               evidence or dict(schema=SCHEMA, error=error, completed=False))
            if error is not None:
                self.failures.append("m3 end-of-route drain: " + error)
            return super().finish()

    return M3EndOfRouteDrainLayer


@contextmanager
def install_collector_layer():
    """v3 ``install_collector_layer`` with the v12 drain layer directly above the unchanged v3 M3 layer.

    A faithful copy of the v3 context manager (same builder seam, same map-journal tee, same restoration); the only
    difference is the extra layer.  It is NOT written as a wrapper that rebinds ``RC.build_layer``: the v3 builder
    resolves that name when the collector is constructed, long after such a wrapper would have restored it.
    """
    from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
    from rl_agent.splitfusion_m3_route_b_v3 import map_journal as MJ
    inner = LG.build_live_gt_collector
    built = []

    def builder(base):
        layer = build_drain_layer(RC.build_layer(inner(base)))
        built.append(layer)
        return layer

    LG.build_live_gt_collector = builder
    MJ.install_tee()
    try:
        yield built
    finally:
        MJ.uninstall_tee()
        LG.build_live_gt_collector = inner


def drain_registration():
    return dict(schema=SCHEMA, layer=__name__ + ".build_drain_layer", seam=__name__ + ".install_collector_layer",
                position="directly above the unchanged splitfusion_m3_route_b_v3.route_completion.M3CollectorLayer, "
                         "below EarlyGtCollector; same builder seam (live_gt.build_live_gt_collector)",
                thread="main route-runner thread (collecting_drive finally), before the retained "
                       "PassiveSplitCollector.finish forced expiry and stop_event",
                steps=["stop new submissions (DROPPED_END_OF_ROUTE_DRAIN, counted, never sent)",
                       "wait for the one in-flight token to finish registering/sending (bounded)",
                       "wait for outstanding terminals within each obligation's own registered ACK horizon",
                       "close anything still open through the retained record_expired on the real clock"],
                bounds=dict(ack_horizon="unchanged per-obligation ack_timeout_at (capture + ack_timeout_ms)",
                            settle_margin_s=C.DRAIN_SETTLE_MARGIN_S, absolute_cap_s=C.DRAIN_ABSOLUTE_CAP_S,
                            in_flight_cap_s=C.DRAIN_IN_FLIGHT_CAP_S),
                unchanged=["170 ms decision deadline and reward resolution", "all timestamps",
                           "the final transmitted frame", "the ACK horizon", "map freshness rules",
                           "every retained finish() step (run unchanged afterwards)"],
                never=["declaring a missing terminal successful (a timeout terminal is written as a timeout)",
                       "resending", "re-stamping", "omitting a sent frame"],
                evidence=f"{RC.EVIDENCE_DIR}/END_OF_ROUTE_DRAIN.json",
                v8_failure="G1 one_terminal_per_sent_frame on the final in-flight frame 8595 (ACK at +80 ms, inside "
                           "its 500 ms horizon, 110 ms after finish() began)")
