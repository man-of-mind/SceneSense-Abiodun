"""M3.1 v9 calibrated radio telemetry: per-transmission PUSCH SNR, scheduler avg_snr and actual MCS (evidence only).

V8's "measured SNR" came from the gNB MAC log (``ulsch_rounds ... SNR x (+d) dB``), which is ``nr_mac_get_snr()`` =
``avg_snr + tpc_in_flight``.  The TPC loop pins that near ``pusch_TargetSNR + 2`` dB whatever the channel, so it
reads ~17 dB at every setting and is NOT the channel SNR (qualification
``experiments/splitfusion_m3_good_all_snr_qual_v1/20261008_run01/REPORT.md``).  The calibrated quantity is the
per-transmission T-tracer value ``GNB_MAC_PUSCH_POWER_CONTROL.snrx10 / 10`` (the Phase-14A anchor), with the
scheduler EMA ``GNB_MAC_UL_MCS_DECISION.avg_snr_x10 / 10`` and the actual ``selected_mcs`` / ``final_mcs``.

This collector reproduces the qualified radio-only measurement path inside the live capture:

    taskset -c <radio cpus> multi -d T_messages.txt -ip 127.0.0.1 -p 2021 -lp 2031      (relay)
    taskset -c <radio cpus> csv  -d T_messages.txt -ip 127.0.0.1 -p 2031 -f -s , -t time <EVENT> <fields...>

* The gNB T server accepts ONE client and closes its listener while connected, so the relay (``multi``) is required;
  the Phase-6 runner's existing tracer attaches to the **UE** port 2023 (relay 2123) and never touches 2021, which
  is free in a live capture (verified in the V8 gnb.log: "waiting for remote tracer on port 2021", never connected).
* Every collector process is placed on the registered **radio** CPUs, so it shares the partition of the gNB tracer
  worker it reads from and never competes with the edge or the UE/GT deadline paths.
* Rows are drained by daemon threads in the **runner parent** process (the process that already hosts the MAC-log
  observer) and written straight to create-only CSVs, each line prefixed with the host ``wall_ns`` and
  ``CLOCK_MONOTONIC_RAW`` of ingest.  Nothing is kept in memory, nothing is parsed, hashed or joined here, and no
  inference, GT, scoring or map thread is involved.
* Row counts are bounded per stream; overflow is counted, never silent.

The MAC-log statistic keeps its own separate, explicitly labelled artefact (the unchanged v3 ``radio_observer``).
Neither quantity reaches the actor: the actor's causal input stays the ACKed commanded target.
"""
import json
from pathlib import Path
import os
import signal
import socket
import subprocess
import threading
import time

from . import contract as C

SCHEMA = "splitfusion.m3_route_b_v9.radio_telemetry.v1"
TRACER_RELATIVE = "OAI/openairinterface5g/common/utils/T/tracer"
MESSAGES_RELATIVE = "OAI/openairinterface5g/common/utils/T/T_messages.txt"
GNB_T_PORT, RELAY_PORT = 2021, 2031
# Exactly the field lists of the two T events (verified against T_messages.txt by contract.radio_telemetry_registration)
PUSCH_EVENT = "GNB_MAC_PUSCH_POWER_CONTROL"
PUSCH_FIELDS = ("time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
                "txpower_calc", "rbSize", "mcs", "rssi")
MCS_EVENT = "GNB_MAC_UL_MCS_DECISION"
MCS_FIELDS = ("time", "rnti", "frame", "slot", "sched_frame", "sched_slot", "avg_snr_x10", "mcs_table",
              "ul_bler_mcs_before", "selected_mcs", "pre_phr_mcs", "post_phr_mcs", "final_mcs",
              "estimated_ul_buffer", "sched_ul_bytes", "B", "min_rb", "available_rb_before", "available_rb_after",
              "ph", "pcmax", "rb_size_final", "tbs_final", "force_ul_mcs")
STREAMS = (("pusch", PUSCH_EVENT, PUSCH_FIELDS, "live_pusch_with_ingest.csv"),
           ("mcs", MCS_EVENT, MCS_FIELDS, "live_mcs_with_ingest.csv"))


def port_listening(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(port)))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def wait_tcp(port, timeout_s):
    deadline = time.monotonic() + float(timeout_s)
    last = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
                return True
        except OSError as error:
            last = error
            time.sleep(0.1)
    raise RuntimeError(f"T-tracer relay port {port} did not become ready: {last}")


class _Stream:
    """One ``csv`` client: its own process, drain thread and create-only output file."""

    def __init__(self, name, argv, path, log, cpus, max_rows):
        self.name, self.path, self.max_rows = name, Path(path), int(max_rows)
        self.rows = self.dropped = self.preamble = 0
        self.log_handle = Path(log).open("xb")
        self.handle = self.path.open("x", encoding="utf-8")
        self.process = subprocess.Popen(_placed(argv, cpus), cwd=str(C.repo_root()), text=True,
                                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=self.log_handle, bufsize=1, start_new_session=True)
        self.thread = threading.Thread(target=self._drain, name=f"m3-radio-{name}", daemon=True)
        self.thread.start()

    def _drain(self):
        for line in self.process.stdout:
            wall, raw = time.time_ns(), time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            if line[:1].isdigit():
                if self.rows >= self.max_rows:
                    self.dropped += 1
                    continue
                self.rows += 1
            else:
                self.preamble += 1
            self.handle.write(f"{wall},{raw},{line.rstrip()}\n")

    def stop(self):
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        self.thread.join(timeout=3)
        self.handle.flush()
        self.handle.close()
        self.log_handle.close()
        return dict(name=self.name, rows=self.rows, dropped=self.dropped, preamble_lines=self.preamble,
                    returncode=self.process.returncode, output=self.path.name, max_rows=self.max_rows)


def _placed(argv, cpus):
    """Pin a collector process to the registered radio CPUs (no sudo: the tracer tools run as this user)."""
    return ["taskset", "-c", ",".join(str(int(cpu)) for cpu in cpus), *argv]


class RadioTelemetry:
    """Qualified per-transmission PUSCH SNR / scheduler avg_snr / MCS, collected for the capture's duration."""

    def __init__(self, directory, cpus):
        self.directory = Path(directory)
        self.cpus = sorted(int(cpu) for cpu in cpus)
        self.relay = None
        self.relay_log = None
        self.streams = []
        self.started = False
        self.error = None
        self.clock_anchor = None

    def start(self):
        root = C.repo_root()
        tracer, messages = root / TRACER_RELATIVE, root / MESSAGES_RELATIVE
        for path in (tracer / "multi", tracer / "csv", messages):
            if not path.is_file():
                raise RuntimeError("T-tracer component missing: " + str(path))
        if not port_listening(GNB_T_PORT):
            raise RuntimeError(f"gNB T port {GNB_T_PORT} is not listening; the radio must be attached first")
        if port_listening(RELAY_PORT):
            raise RuntimeError(f"T-tracer relay port {RELAY_PORT} is occupied")
        if not self.cpus:
            raise RuntimeError("radio CPU set required for the telemetry collector")
        out = self.directory / "ttracer"
        out.mkdir(parents=True, exist_ok=False)
        self.relay_log = (out / "relay.log").open("xb")
        self.relay = subprocess.Popen(
            _placed([str(tracer / "multi"), "-d", str(messages), "-ip", "127.0.0.1", "-p", str(GNB_T_PORT),
                     "-lp", str(RELAY_PORT)], self.cpus),
            cwd=str(root), stdin=subprocess.DEVNULL, stdout=self.relay_log, stderr=subprocess.STDOUT,
            start_new_session=True)
        wait_tcp(RELAY_PORT, 10)
        common = [str(tracer / "csv"), "-d", str(messages), "-ip", "127.0.0.1", "-p", str(RELAY_PORT),
                  "-f", "-s", ",", "-t", "time"]
        for name, event, fields, filename in STREAMS:
            self.streams.append(_Stream(name, [*common, event, *fields], out / filename, out / f"{name}.stderr.log",
                                        self.cpus, C.RADIO_TELEMETRY_MAX_ROWS))
        self.clock_anchor = dict(wall_ns=time.time_ns(), raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                                 monotonic_ns=time.monotonic_ns())
        self.started = True
        return self

    def stop(self):
        """Stop the collectors (before the radio teardown; the relay port must be free for the cold check)."""
        summaries = [stream.stop() for stream in self.streams]
        if self.relay is not None and self.relay.poll() is None:
            try:
                os.killpg(os.getpgid(self.relay.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.relay.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.relay.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.relay.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        if self.relay_log is not None:
            self.relay_log.close()
            self.relay_log = None
        document = dict(
            schema=SCHEMA, started=self.started, error=self.error, cpus=self.cpus,
            relay=dict(port=RELAY_PORT, gnb_port=GNB_T_PORT,
                       returncode=None if self.relay is None else self.relay.returncode),
            streams=summaries, clock_anchor=self.clock_anchor,
            relay_port_free_after_stop=not port_listening(RELAY_PORT),
            quantities=dict(
                pusch_snr_db=f"{PUSCH_EVENT}.snrx10/10 per transmission (Phase-14A calibrated channel SNR)",
                scheduler_avg_snr_db=f"{MCS_EVENT}.avg_snr_x10/10 (EMA; the SINR MCS lookup reads this)",
                mcs=f"{MCS_EVENT}.selected_mcs / final_mcs, and {PUSCH_EVENT}.mcs",
                mac_log_snr="NOT here: kept separately in m3_radio/gnb_stats_observed.jsonl, labelled "
                            "power-control SNR (avg_snr + tpc_in_flight), not channel SNR"),
            ingest="each line prefixed with host wall_ns and CLOCK_MONOTONIC_RAW ns at drain time",
            actor_input_unchanged=True, decision_path_touched=False)
        path = self.directory / "RADIO_TELEMETRY.json"
        if not path.exists():
            path.write_text(json.dumps(document, indent=1, sort_keys=True, default=str) + "\n")
        self.streams = []
        self.relay = None
        return document


def install(output_root, cpus):
    """Wrap the supervisor radio start/stop in this (runner) process.

    Start: after the radio has attached (so port 2021 is listening and the cold check inside ``_start_live_radio``
    has already run; that check refuses a listener on the relay port).  A start failure is raised there, before
    CARLA and the capture exist, so no capture is ever spent without its calibrated channel evidence.
    Stop: before the retained teardown, which requires the relay port to be free again for the post-teardown cold
    proof.  A stop failure never masks the teardown's own outcome.
    """
    from rl_agent import ue_288_campaign_supervisor as S
    telemetry = RadioTelemetry(Path(output_root) / "m3_radio", cpus)
    start, stop = S._start_live_radio, S._stop_live_radio

    def start_live_radio(*args, **kwargs):
        result = start(*args, **kwargs)
        telemetry.start()
        return result

    def stop_live_radio(*args, **kwargs):
        if telemetry.started:
            telemetry.stop()
        return stop(*args, **kwargs)

    S._start_live_radio, S._stop_live_radio = start_live_radio, stop_live_radio
    return telemetry


def registered_arm_file(output_root, levels=2):
    """The prospective's ``arm.json``.

    The runner is given the CAPTURE directory (``<prospective>/capture``, which need not exist yet), while the
    partition arm file is published in the prospective root beside the registration.  Search the directory itself
    and its bounded parents for the pair whose registration is this contract's, so the file is found by identity
    rather than by a fixed relative depth.
    """
    start = Path(output_root).resolve()
    for directory in (start, *list(start.parents)[:levels]):
        arm, registration = directory / "arm.json", directory / "registration.json"
        if arm.is_file() and registration.is_file():
            if json.loads(registration.read_bytes()).get("id") == C.ID:
                return arm
    raise RuntimeError(f"registered arm.json (with a {C.ID} registration) not found at or above {start}")


def radio_cpus(output_root):
    """The registered radio CPU set of this attempt (the partition arm file published with the prospective)."""
    value = json.loads(registered_arm_file(output_root).read_bytes())
    cpus = [int(cpu) for cpu in value["radio_cpus"]]
    if not cpus or sorted(cpus) != cpus:
        raise ValueError("radio CPU set is empty or unordered")
    return cpus
