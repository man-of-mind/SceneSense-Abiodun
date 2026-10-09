"""Achieved-radio evidence for M3.1 (runner process only; never on the UE/edge path).

The OAI gNB prints periodic MAC statistics into ``<radio_state>/gnb.log`` (e.g. ``ulsch_rounds ... SNR 22.8 (+7.8) dB``:
the gNB-measured UL PUSCH SNR the Phase-14a mapping was calibrated against). The retained teardown deletes the radio
scratch. This observer (a) tails gnb.log at 2 Hz from radio attach to teardown and records every statistics line with
its host observation times (CLOCK_MONOTONIC_RAW and wall; the line itself carries no time, so the observation time is
an upper bound up to one poll period after the print), and (b) copies the complete gnb.log/ue.log and the radio
materialization into the capture immediately before the retained scratch removal. Labelled ACHIEVED_RADIO_MEASUREMENT;
never substituted for the commanded target or the actor's proxy.
"""
import json
import shutil
import threading
import time
from pathlib import Path

KEEP = ("gnb.log", "ue.log", "radio_materialization.json", "ATTACHED_RADIO_STATE.json",
        "effective_gnb_100mhz_4d5u_clean_minus50.conf", "effective_ue_100mhz_clean_minus50.conf")
STATS_TOKENS = ("ulsch_rounds", "dlsch_rounds", "average RSRP", "UL-RI")


class RadioObserver:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.radio_state = None
        self._stop = threading.Event()
        self._thread = None
        self._offset = 0
        self._stream = None
        self.counts = dict(polls=0, lines=0, stats_lines=0, errors=0)
        self.copied = []

    def start(self, radio_state):
        if self._thread is not None:
            raise RuntimeError("radio observer already started")
        self.radio_state = Path(radio_state)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stream = open(self.directory / "gnb_stats_observed.jsonl", "x", buffering=1)
        self._thread = threading.Thread(target=self._loop, name="m3-radio-observer", daemon=True)
        self._thread.start()

    def _poll(self):
        path = self.radio_state / "gnb.log"
        if not path.is_file():
            return
        with path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
        if not chunk:
            return
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        self._offset += cut + 1
        raw, wall = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW), time.time_ns()
        for line in chunk[:cut].decode("utf-8", "replace").splitlines():
            self.counts["lines"] += 1
            if any(token in line for token in STATS_TOKENS):
                self.counts["stats_lines"] += 1
                self._stream.write(json.dumps(dict(observed_raw_ns=raw, observed_wall_ns=wall, line=line)) + "\n")

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.counts["polls"] += 1
                self._poll()
            except Exception:  # noqa: BLE001 - evidence only
                self.counts["errors"] += 1
            self._stop.wait(0.5)

    def finish(self):
        """Stop tailing, take a last poll and copy the complete logs (called before the scratch is removed)."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            try:
                self._poll()
            except Exception:  # noqa: BLE001
                self.counts["errors"] += 1
        if self.radio_state is not None and self.radio_state.is_dir():
            target = self.directory / "radio_state_copy"
            target.mkdir(parents=True, exist_ok=True)
            for name in KEEP:
                source = self.radio_state / name
                if source.is_file():
                    shutil.copyfile(source, target / name)
                    self.copied.append(name)
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        summary = dict(schema="splitfusion.m3_route_b_v1.radio_observer.v1", radio_state=str(self.radio_state),
                       counts=self.counts, copied=self.copied, semantics="ACHIEVED_RADIO_MEASUREMENT (gNB MAC stats); "
                       "observation time is an upper bound (2-Hz poll); not the commanded target or the actor proxy")
        path = self.directory / "RADIO_OBSERVER.json"
        if not path.exists():
            path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        return summary


def install(output_root):
    """Wrap the supervisor radio start and the phase14b scratch removal in this (runner) process."""
    from rl_agent import ue_288_campaign_supervisor as S
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as B
    observer = RadioObserver(Path(output_root) / "m3_radio")
    start, remove = S._start_live_radio, B._remove_owned_radio_scratch

    def start_live_radio(*args, **kwargs):
        namespace, radio_state, attached = start(*args, **kwargs)
        observer.start(radio_state)
        return namespace, radio_state, attached

    def remove_owned_radio_scratch(radio_state, radio_namespace):
        if observer.radio_state is not None and Path(radio_state) == observer.radio_state:
            observer.finish()
        return remove(radio_state, radio_namespace)

    S._start_live_radio, B._remove_owned_radio_scratch = start_live_radio, remove_owned_radio_scratch
    return observer
