#!/usr/bin/env python3
"""M3.1 GOOD_ALL SNR qualification: bounded radio-only measurement on an already-attached 100-MHz/4D5U radio.

Reuses the Phase-14A calibration collectors (T-tracer GNB_MAC_PUSCH_POWER_CONTROL + GNB_MAC_UL_MCS_DECISION), the
RFsim telnet actuator, the calibration UDP sink/sender, and tails gnb.log for the MAC-statistics line
(``SNR = avg_snr + tpc_in_flight``). Changes only rfsimu_channel_ue0.noise_power_dB and restores -50 dB. No OAI change.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rl_agent import ue_n2_oai_ul_calibration_smoke as n2  # noqa: E402

TRACER = ROOT / "OAI/openairinterface5g/common/utils/T/tracer"
MESSAGES = ROOT / "OAI/openairinterface5g/common/utils/T/T_messages.txt"
SENDER = ROOT / "oai_layer_latency/carla_shaped_udp_burst_sender.py"
SINK = ROOT / "bursty_traffic/udp_sink.py"
GNB_PORT, RELAY_PORT, TELNET_PORT = 2021, 2031, 9090
MODEL = "rfsimu_channel_ue0"
CLEAN = -50.0
UE_IP, EXT_DN_IP, PORT = "10.0.0.2", "192.168.70.135", 56140
PUSCH_FIELDS = ("time", "rnti", "frame", "slot", "snrx10", "phr", "tpc", "tb_size",
                "txpower_calc", "rbSize", "mcs", "rssi")
MCS_FIELDS = ("time", "rnti", "frame", "slot", "sched_frame", "sched_slot", "avg_snr_x10", "mcs_table",
              "ul_bler_mcs_before", "selected_mcs", "pre_phr_mcs", "post_phr_mcs", "final_mcs",
              "estimated_ul_buffer", "sched_ul_bytes", "B", "min_rb", "available_rb_before", "available_rb_after",
              "ph", "pcmax", "rb_size_final", "tbs_final", "force_ul_mcs")
STATS_TOKENS = ("ulsch_rounds", "UL-RI", "average RSRP")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class LogTail(threading.Thread):
    """Tail gnb.log at 5 Hz; stamp every MAC-statistics line with CLOCK_MONOTONIC (upper bound, <=0.2 s late)."""

    def __init__(self, log: Path, out: Path):
        super().__init__(daemon=True)
        self.log, self.out, self.stop_event = log, out.open("x", buffering=1), threading.Event()
        self.offset = log.stat().st_size

    def poll(self):
        with self.log.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        cut = chunk.rfind(b"\n")
        if cut < 0:
            return
        self.offset += cut + 1
        mono, wall = time.monotonic_ns(), time.time_ns()
        for line in chunk[:cut].decode("utf-8", "replace").splitlines():
            if any(token in line for token in STATS_TOKENS):
                self.out.write(json.dumps(dict(mono_ns=mono, wall_ns=wall, line=line)) + "\n")

    def run(self):
        while not self.stop_event.is_set():
            self.poll()
            self.stop_event.wait(0.2)
        self.poll()
        self.out.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radio-state", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--setting", action="append", required=True, help="nominal_target_db:noise_power_db")
    ap.add_argument("--baseline-s", type=float, default=5.0)
    ap.add_argument("--settle-s", type=float, default=10.0)
    ap.add_argument("--measure-s", type=float, default=45.0)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--frame-bytes", type=int, default=112800)
    ap.add_argument("--chunk-bytes", type=int, default=60000)
    args = ap.parse_args()
    settings = [tuple(float(x) for x in item.split(":")) for item in args.setting]
    for _, noise in settings:
        require(-13.0 <= noise <= -2.0, f"noise {noise} outside the calibrated anchor range")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "ttracer").mkdir()
    (out / "traffic").mkdir()
    gnb_log = args.radio_state.resolve() / "gnb.log"
    require(gnb_log.is_file(), "gnb.log missing")
    require(not n2.port_is_free(GNB_PORT) and not n2.port_is_free(TELNET_PORT), "gNB T/telnet port not listening")
    require(n2.port_is_free(RELAY_PORT), "relay port occupied")
    require(subprocess.run(["pgrep", "-f", "CarlaUnreal"], capture_output=True).returncode != 0, "CARLA running")

    meta = dict(schema="splitfusion.m3_good_all_snr_qual.v1", argv=sys.argv, radio_state=str(args.radio_state),
                settings=[dict(nominal_target_db=t, noise_power_db=n) for t, n in settings],
                traffic=dict(fps=args.fps, frame_bytes=args.frame_bytes, chunk_bytes=args.chunk_bytes,
                             offered_mbps=args.fps * args.frame_bytes * 8 / 1e6),
                timing=dict(baseline_s=args.baseline_s, settle_s=args.settle_s, measure_s=args.measure_s))
    procs, collectors, commands, windows, errors = [], [], [], [], []
    telnet = tail = model_index = None

    def spawn(name, argv, root_owned=False):
        handle = (out / f"{name}.log").open("xb")
        proc = subprocess.Popen(argv, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)
        procs.append(n2.ManagedProcess(name, proc, handle, root_owned=root_owned))

    def modify(value, purpose):
        target = f"{value:.12g}"
        cmd = f"channelmod modify {model_index} noise_power_dB {target}"
        s_mono, s_wall, a_mono, a_wall, response = telnet.command(cmd)
        n2.Runner.validate_modify_response(response, target)
        state = telnet.command("channelmod show current")[-1]
        got = float(n2.parse_channel_models(state)[MODEL]["noise_power_db"])
        require(math.isclose(got, value, abs_tol=1e-6), f"post-state {got} != {value}")
        commands.append(dict(purpose=purpose, command=cmd, noise_power_db=value, post_state_noise_power_db=got,
                             send_monotonic_ns=s_mono, ack_monotonic_ns=a_mono, send_wall_ns=s_wall,
                             ack_wall_ns=a_wall, ack_latency_ms=(a_mono - s_mono) / 1e6))

    try:
        spawn("gnb_relay", [str(TRACER / "multi"), "-d", str(MESSAGES), "-ip", "127.0.0.1", "-p", str(GNB_PORT),
                            "-lp", str(RELAY_PORT)])
        n2.wait_tcp(RELAY_PORT, 10)
        common = [str(TRACER / "csv"), "-d", str(MESSAGES), "-ip", "127.0.0.1", "-p", str(RELAY_PORT),
                  "-f", "-s", ",", "-t", "time"]
        collectors.append(n2.LiveCsv([*common, "GNB_MAC_PUSCH_POWER_CONTROL", *PUSCH_FIELDS],
                                     out / "ttracer/live_pusch_with_ingest.csv"))
        collectors.append(n2.LiveCsv([*common, "GNB_MAC_UL_MCS_DECISION", *MCS_FIELDS],
                                     out / "ttracer/live_mcs_with_ingest.csv"))
        tail = LogTail(gnb_log, out / "gnb_stats_observed.jsonl")
        tail.start()
        telnet = n2.TelnetSession("127.0.0.1", TELNET_PORT, 2.0, 1 << 20)
        before = telnet.command("channelmod show current")[-1]
        (out / "channel_state_before.txt").write_text(before)
        model = n2.parse_channel_models(before)[MODEL]
        require(math.isclose(float(model["noise_power_db"]), CLEAN, abs_tol=1e-6), "radio not at clean -50 dB")
        require(model.get("model_type") == "AWGN", "RFsim model type drift")
        model_index = int(model["model_index"])

        total_s = args.baseline_s + len(settings) * (args.settle_s + args.measure_s) + 10.0
        ext_dn_pid = int(n2.run_checked(["sudo", "-n", "docker", "inspect", "-f", "{{.State.Pid}}",
                                         "oai-ext-dn"]).stdout.strip())
        meta["sink_spawn_wall_ns"], meta["sink_spawn_monotonic_ns"] = time.time_ns(), time.monotonic_ns()
        spawn("sink", ["sudo", "-n", "nsenter", "-t", str(ext_dn_pid), "-n", "/usr/bin/python3", str(SINK),
                       "--bind", EXT_DN_IP, "--port", str(PORT), "--out", str(out / "traffic/sink_packets.csv"),
                       "--timeout-s", "10"], root_owned=True)
        time.sleep(0.5)
        meta["sender_spawn_wall_ns"], meta["sender_spawn_monotonic_ns"] = time.time_ns(), time.monotonic_ns()
        spawn("sender", [sys.executable, str(SENDER), "--bind-host", UE_IP, "--remote-host", EXT_DN_IP,
                         "--remote-port", str(PORT), "--fps", str(args.fps),
                         "--frames", str(int(math.ceil(total_s * args.fps))), "--frame-bytes", str(args.frame_bytes),
                         "--chunk-bytes", str(args.chunk_bytes), "--idle-before-s", "0", "--cooldown-s", "2",
                         "--log-csv", str(out / "traffic/sender.csv")])
        time.sleep(args.baseline_s)
        for target, noise in settings:
            modify(noise, f"SETTING_{target}")
            time.sleep(args.settle_s)
            start = time.monotonic_ns()
            time.sleep(args.measure_s)
            end = time.monotonic_ns()
            for p in procs:
                require(p.process.poll() is None, f"{p.name} exited during measurement")
            windows.append(dict(nominal_target_db=target, noise_power_db=noise, start_monotonic_ns=start,
                                end_monotonic_ns=end, start_wall_ns=time.time_ns() - (time.monotonic_ns() - start)))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if telnet is not None and model_index is not None:
            try:
                modify(CLEAN, "RESTORE_CLEAN")
                (out / "channel_state_restored.txt").write_text(telnet.command("channelmod show current")[-1])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"restore: {type(exc).__name__}: {exc}")
            telnet.close()
        for c in collectors:
            try:
                c.stop()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"collector: {exc}")
        if tail is not None:
            tail.stop_event.set()
            tail.join(timeout=5)
        for p in reversed(procs):
            try:
                p.stop()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{p.name}: {exc}")
        meta.update(commands=commands, windows=windows, errors=errors,
                    restored_clean=any(c["purpose"] == "RESTORE_CLEAN" for c in commands))
        (out / "measurement.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(json.dumps(dict(output=str(out), errors=errors, windows=len(windows)), indent=1))
    return 0 if not errors and len(windows) == len(settings) else 1


if __name__ == "__main__":
    raise SystemExit(main())
