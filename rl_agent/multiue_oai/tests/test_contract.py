from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import yaml

from rl_agent.multiue_oai.analyze import main as analyze_main
from rl_agent.multiue_oai.analyze import pair_effect, service_family
from rl_agent.multiue_oai.analyze_v2 import (
    SimDemand,
    arrival_blueprint,
    main as analyze_v2_main,
    max_min_allocate,
    simulated_trial_metrics,
    validate_model_config,
)
from rl_agent.multiue_oai.endpoint import (
    FRAME_HEADER,
    GrantObserver,
    MAGIC,
    VERSION,
    build_parser as build_endpoint_parser,
    build_frame_blob,
    build_ttracer_csv_command,
    chunks_per_frame,
    frame_onwire_bytes,
    parse_ul_new_data_grant,
    staggered_arrival_credits,
    validate_send_args,
)
from rl_agent.multiue_oai.runner import (
    DEFAULT_CONFIG,
    Runner,
    load_config,
    parse_channel_models,
    parse_interface_ipv4,
    parse_runtime_uicc_imsis,
    parse_uicc_profiles,
    per_ue_radio_summary,
    receiver_identity_report,
    sender_route_report,
    ue_network_contract_report,
)


class EndpointContractTest(unittest.TestCase):
    def test_frame_metadata_and_wire_accounting(self) -> None:
        blob = build_frame_blob(409600, 1, 1234, 987654321)
        self.assertEqual(len(blob), 409600)
        magic, version, ue_id, _flags, frame_id, scheduled, payload_bytes, _crc = FRAME_HEADER.unpack_from(blob)
        self.assertEqual(magic, MAGIC)
        self.assertEqual(version, VERSION)
        self.assertEqual(ue_id, 1)
        self.assertEqual(frame_id, 1234)
        self.assertEqual(scheduled, 987654321)
        self.assertEqual(payload_bytes, 409600)
        self.assertEqual(chunks_per_frame(409600, 60000), 7)
        self.assertEqual(frame_onwire_bytes(409600, 60000), 409600 + 7 * 36)

    def test_demand_seed_controls_a_reproducible_staggered_phase(self) -> None:
        first = staggered_arrival_credits([0, 1], 61301)
        repeated = staggered_arrival_credits([0, 1], 61301)
        changed = staggered_arrival_credits([0, 1], 61302)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertAlmostEqual((first[1] - first[0]) % 1.0, 0.5)

    def test_sender_semantic_validation_rejects_controller_kind_mismatch(self) -> None:
        parser = build_endpoint_parser()
        args = parser.parse_args(
            [
                "send",
                "--remote-host",
                "127.0.0.1",
                "--remote-port",
                "1",
                "--run-dir",
                "/tmp/not-created",
                "--ue",
                "0,127.0.0.1,0.5",
                "--kind",
                "controlled",
                "--mu-hat-mbps",
                "10",
                "--duration-s",
                "1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires a non-open-loop controller"):
            validate_send_args(args)

    def test_ttracer_csv_command_excludes_record_only_flags(self) -> None:
        command = build_ttracer_csv_command(
            "/tmp/csv",
            "/tmp/T_messages.txt",
            2033,
            "NRUE_MAC_DCI_GRANT",
            ("time", "direction", "rnti", "tbs", "ndi", "rv", "round"),
        )
        self.assertNotIn("-OFF", command)
        self.assertNotIn("-on", command)
        self.assertEqual(command[-8:], [
            "NRUE_MAC_DCI_GRANT", "time", "direction", "rnti", "tbs", "ndi", "rv", "round"
        ])

    def test_live_grant_parser_accepts_only_first_transmission_uplink(self) -> None:
        mapping = {23716: 1}
        self.assertEqual(
            parse_ul_new_data_grant(
                "19:45:45.100000,1,23716,2178,1,0,0", mapping
            ),
            (1, 2178),
        )
        self.assertIsNone(
            parse_ul_new_data_grant(
                "19:45:45.100000,0,23716,168,1,0,0", mapping
            )
        )
        self.assertIsNone(
            parse_ul_new_data_grant(
                "19:45:45.100000,1,23716,168,0,2,1", mapping
            )
        )

    def test_grant_observer_health_covers_process_and_reader_thread(self) -> None:
        observer = GrantObserver.__new__(GrantObserver)
        observer.process = Mock(returncode=None)
        observer.process.poll.return_value = None
        observer.thread = Mock()
        observer.thread.is_alive.return_value = True

        self.assertEqual(
            observer.health(),
            {
                "process_alive": True,
                "process_returncode": None,
                "reader_thread_alive": True,
            },
        )

        observer.process.returncode = 139
        observer.process.poll.return_value = 139
        observer.thread.is_alive.return_value = False
        self.assertFalse(observer.health()["process_alive"])
        self.assertFalse(observer.health()["reader_thread_alive"])


class ConfigContractTest(unittest.TestCase):
    def test_stage_is_n2_and_cannot_chain(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config["radio"]["ue_count"], 2)
        self.assertEqual(config["transport"]["payload_bytes"], 409600)
        self.assertEqual(config["c1"]["pessimism_factor"], 0.70)
        self.assertEqual(config["c1"]["estimator_window_s"], 1.0)
        self.assertEqual(config["c1"]["estimator_ewma_alpha"], 0.20)
        self.assertEqual(config["radio"]["expected_receiver_nat_sources"], ["192.168.70.134"])
        self.assertEqual(config["radio"]["expected_dnn"], "oai")
        self.assertEqual(config["radio"]["expected_ue_subnet"], "10.0.0.0/24")
        self.assertEqual(config["instrumentation"]["tunnel_tx_to_sender_ratio_min"], 1.0)
        self.assertEqual(config["instrumentation"]["tunnel_tx_to_sender_ratio_max"], 1.08)
        self.assertGreater(config["instrumentation"]["receiver_finalize_timeout_s"], 0)
        self.assertGreater(
            config["instrumentation"]["network_sampler_finalize_timeout_s"], 0
        )
        self.assertEqual(config["radio"]["attach_strategy"], "clean_then_runtime_strong")
        self.assertEqual(config["radio"]["expected_snr_db"], 6.0)
        self.assertEqual(config["radio"]["snr_tolerance_db"], 2.0)
        self.assertEqual(config["radio"]["expected_mcs"], 8)
        self.assertEqual(config["radio"]["mcs_tolerance"], 2)
        self.assertEqual(
            config["radio"]["strong_rung_registration"]["source_run"],
            "runtime_switch_smoke_fix2_20260813_1709_pdt",
        )
        self.assertIn("DG-B", config["authorization_boundary"]["forbidden"])
        self.assertEqual([row["id"] for row in config["trials"]], [f"A{i}" for i in range(1, 10)])

    def test_receiver_finalize_uses_stop_file_and_requires_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            trial_dir = output / "runs" / "A3"
            trial_dir.mkdir(parents=True)
            for name in (
                "receiver_chunks.csv",
                "receiver_frames.csv",
                "receiver_summary.json",
            ):
                (trial_dir / name).write_text("", encoding="utf-8")
            (trial_dir / "receiver_stdout.log").write_text("clean exit\n", encoding="utf-8")
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            process = Mock(pid=4242, returncode=0)
            process.wait.return_value = 0
            process._scenesense_log_handle = None

            failure = runner._finalize_receiver(process, trial_dir)

            self.assertIsNone(failure)
            self.assertTrue((trial_dir / "receiver_stop.request.json").is_file())
            process.wait.assert_called_once_with(
                timeout=float(runner.config["instrumentation"]["receiver_finalize_timeout_s"])
            )

    def test_receiver_finalize_reports_missing_artifact_instead_of_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            trial_dir = output / "runs" / "A3"
            trial_dir.mkdir(parents=True)
            (trial_dir / "receiver_chunks.csv").write_text("", encoding="utf-8")
            (trial_dir / "receiver_stdout.log").write_text("clean exit\n", encoding="utf-8")
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            process = Mock(pid=4242, returncode=0)
            process.wait.return_value = 0
            process._scenesense_log_handle = None

            failure = runner._finalize_receiver(process, trial_dir)

            self.assertIn("receiver_frames.csv", failure)
            self.assertIn("receiver_summary.json", failure)

    def test_receiver_finalize_timeout_fails_closed_after_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            trial_dir = output / "runs" / "A3"
            trial_dir.mkdir(parents=True)
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            runner._stop_root_process = Mock()
            process = Mock(pid=4242, returncode=None)
            process.wait.side_effect = subprocess.TimeoutExpired("receiver", 10)
            process._scenesense_log_handle = None

            failure = runner._finalize_receiver(process, trial_dir)

            self.assertIn("graceful stop timed out", failure)
            runner._stop_root_process.assert_called_once_with(process)

    def test_network_sampler_finalize_uses_stop_file_and_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            trial_dir = output / "runs" / "A1"
            network_dir = trial_dir / "network"
            network_dir.mkdir(parents=True)
            (trial_dir / "network_sampler_stdout.log").write_text(
                "summary written\n", encoding="utf-8"
            )
            (network_dir / "network_timeseries.csv").write_text(
                "sample_index\n0\n1\n", encoding="utf-8"
            )
            (network_dir / "network_summary.csv").write_text(
                "iface_label,samples\nue0,2\nue1,2\n", encoding="utf-8"
            )
            (network_dir / "network_manifest.json").write_text("{}\n", encoding="utf-8")
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            process = Mock(pid=5252, returncode=0)
            process.wait.return_value = 0

            failure = runner._finalize_network_sampler(process, trial_dir)

            self.assertIsNone(failure)
            self.assertTrue((trial_dir / "network_sampler_stop.request.json").is_file())
            process.wait.assert_called_once_with(
                timeout=float(
                    runner.config["instrumentation"]["network_sampler_finalize_timeout_s"]
                )
            )

    def test_network_sampler_finalize_rejects_signal_exit_even_with_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            trial_dir = output / "runs" / "A1"
            network_dir = trial_dir / "network"
            network_dir.mkdir(parents=True)
            (trial_dir / "network_sampler_stdout.log").write_text(
                "summary written\n", encoding="utf-8"
            )
            for name in (
                "network_timeseries.csv",
                "network_summary.csv",
                "network_manifest.json",
            ):
                (network_dir / name).write_text("complete\n", encoding="utf-8")
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            process = Mock(pid=5252, returncode=-signal.SIGINT)
            process.wait.return_value = -signal.SIGINT

            failure = runner._finalize_network_sampler(process, trial_dir)

            self.assertIn("sampler exited with -2", failure)

    def test_strong_channel_has_explicit_models_for_both_ues(self) -> None:
        runner = Runner(DEFAULT_CONFIG, Path("/tmp/not-created"), dry_run=True)
        channel_path = runner.paths["oai_ran_conf"] / runner.config["radio"]["channel_config"]
        channel = channel_path.read_text(encoding="utf-8")
        for index in range(2):
            self.assertIn(f'model_name     = "rfsimu_channel_enB{index}"', channel)
            self.assertIn(f'model_name     = "rfsimu_channel_ue{index}"', channel)

    def test_attach_smoke_mode_is_explicit_and_cannot_be_negative(self) -> None:
        runner = Runner(
            DEFAULT_CONFIG,
            Path("/tmp/not-created"),
            dry_run=True,
            attach_smoke_repeats=3,
        )
        self.assertEqual(runner.attach_smoke_repeats, 3)
        self.assertEqual(runner.attach_channel_mode, "strong")
        clean = Runner(
            DEFAULT_CONFIG,
            Path("/tmp/not-created"),
            dry_run=True,
            attach_smoke_repeats=1,
            attach_channel_mode="clean",
        )
        self.assertEqual(clean.attach_channel_mode, "clean")
        clean._materialize_ue_config()
        self.assertEqual(
            clean.runtime_ue_config,
            clean.paths["oai_ran_conf"] / clean.config["radio"]["ue_base_config"],
        )
        with self.assertRaises(ValueError):
            Runner(
                DEFAULT_CONFIG,
                Path("/tmp/not-created"),
                dry_run=True,
                attach_smoke_repeats=-1,
            )
        with self.assertRaises(ValueError):
            Runner(
                DEFAULT_CONFIG,
                Path("/tmp/not-created"),
                dry_run=True,
                attach_channel_mode="clean",
            )
        with self.assertRaises(ValueError):
            Runner(
                DEFAULT_CONFIG,
                Path("/tmp/not-created"),
                dry_run=True,
                attach_smoke_repeats=1,
                runtime_switch_smoke=True,
            )

    def test_runtime_switch_materializes_initial_clean_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner = Runner(
                DEFAULT_CONFIG,
                Path(temp) / "switch",
                dry_run=True,
                runtime_switch_smoke=True,
            )
            runner._materialize_ue_config()
            ue_config = runner.runtime_ue_config.read_text(encoding="utf-8")
            gnb_config = runner.runtime_gnb_config.read_text(encoding="utf-8")
            self.assertEqual(ue_config.count("noise_power_dB = -50;"), 4)
            self.assertEqual(gnb_config.count("noise_power_dB = -50;"), 4)
            self.assertNotIn("noise_power_dB = -4;", ue_config)
            self.assertNotIn("@include", gnb_config)
            self.assertIn("Active_gNBs", gnb_config)

    def test_full_dg_a_also_materializes_initial_clean_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runner = Runner(
                DEFAULT_CONFIG,
                Path(temp) / "dg_a",
                dry_run=True,
            )
            self.assertTrue(runner.runtime_channel_control)
            runner._materialize_ue_config()
            ue_config = runner.runtime_ue_config.read_text(encoding="utf-8")
            gnb_config = runner.runtime_gnb_config.read_text(encoding="utf-8")
            self.assertEqual(ue_config.count("noise_power_dB = -50;"), 4)
            self.assertEqual(gnb_config.count("noise_power_dB = -50;"), 4)
            self.assertNotIn("noise_power_dB = -4;", ue_config)

    def test_full_dg_a_enters_strong_rung_before_each_block(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            output.mkdir()
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            runner._wait_for_cold_ran = Mock()
            runner.start_ran = Mock()
            runner._enter_registered_strong_rung = Mock(
                side_effect=[{"block": "A"}, {"block": "B"}]
            )
            runner.d0 = Mock()
            runner.calibrate = Mock(
                side_effect=[
                    {
                        "mu_hat_mbps": 10.0,
                        "rnti_map": {0: 100, 1: 200},
                        "service_conversion": 0.8,
                    },
                    {
                        "mu_hat_mbps": 10.0,
                        "rnti_map": {0: 101, 1: 201},
                        "service_conversion": 0.8,
                    },
                ]
            )
            runner.run_trial = Mock(return_value={})
            runner.stop_ran = Mock()
            runner.command = Mock()

            runner.run_dg_a()

            self.assertEqual(
                runner._enter_registered_strong_rung.call_args_list,
                [call("A"), call("B")],
            )
            runner.d0.assert_called_once_with()
            self.assertEqual(runner.start_ran.call_args_list, [call("A"), call("B")])
            self.assertEqual(runner._wait_for_cold_ran.call_count, 2)
            self.assertEqual(runner.stop_ran.call_count, 1)

    def test_strong_rung_entry_uses_real_traffic_and_radio_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            output.mkdir()
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            runner._attach_stability_evidence = Mock(
                side_effect=[{"phase": "pre"}, {"phase": "post"}]
            )
            runner._switch_both_uplinks = Mock(return_value={"both_uplinks_modified": True})
            runner.run_trial = Mock(
                side_effect=[
                    {
                        "radio_validity": {
                            "per_ue": {
                                "0": {"pass": True, "rnti": 100},
                                "1": {"pass": True, "rnti": 200},
                            }
                        },
                        "sender_route_gate": {"pass": True},
                    },
                    {
                        "radio_validity": {
                            "per_ue": {
                                "0": {"pass": True, "rnti": 100},
                                "1": {"pass": True, "rnti": 200},
                            }
                        },
                        "sender_route_gate": {"pass": True},
                        "observer": {"service_event_count": {"0": 10, "1": 11}},
                    },
                ]
            )

            evidence = runner._enter_registered_strong_rung("A")

            self.assertTrue(evidence["both_uplinks_on_registered_strong_rung"])
            strong_call, control_call = runner.run_trial.call_args_list
            trial = strong_call.args[0]
            self.assertEqual(trial["id"], "STRONG_GATE_A")
            self.assertEqual(trial["kind"], "smoke")
            self.assertEqual(trial["fractions"], [0.2, 0.2])
            self.assertTrue(strong_call.kwargs["enforce_strong_rung"])
            self.assertEqual(control_call.args[0]["id"], "CONTROLLED_PATH_GATE_A")
            self.assertEqual(control_call.args[0]["kind"], "controlled")
            self.assertEqual(control_call.kwargs["rnti_map"], {0: 100, 1: 200})
            self.assertTrue(evidence["controlled_path_gate"]["recorder_and_observer_concurrent"])
            self.assertTrue((output / "blocks" / "A" / "strong_rung_entry_gate.json").exists())

    def test_every_stage_sender_command_passes_the_real_endpoint_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            output.mkdir()
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)

            runner._validate_sender_contracts()

            contract = json.loads(
                (output / "preflight" / "sender_contracts.json").read_text()
            )
            self.assertEqual(contract["validated_count"], 18)
            by_id = {row["trial_id"]: row for row in contract["trials"]}
            self.assertEqual(by_id["STRONG_GATE_A"]["kind"], "smoke")
            self.assertEqual(by_id["CONTROLLED_PATH_GATE_A"]["kind"], "controlled")
            self.assertIn(
                "2033",
                by_id["CONTROLLED_PATH_GATE_A"]["sender_argv"],
            )
            self.assertEqual(by_id["A6"]["controller"], "decentralized_c1")
            for row in contract["trials"]:
                parsed = build_endpoint_parser().parse_args(row["sender_argv"][3:])
                validate_send_args(parsed)

    def test_oai_trace_relay_accepts_real_record_and_csv_clients(self) -> None:
        runner = Runner(DEFAULT_CONFIG, Path("/tmp/not-created"), dry_run=True)
        relay_binary = runner.paths["ttracer_dir"] / "multi"
        record_binary = runner.paths["ttracer_dir"] / "record"
        csv_binary = runner.paths["ttracer_dir"] / "csv"
        self.assertTrue(relay_binary.is_file())
        self.assertTrue(record_binary.is_file())
        self.assertTrue(csv_binary.is_file())

        try:
            upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except PermissionError as exc:
            self.skipTest(f"network namespace forbids loopback relay test: {exc}")
        upstream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        upstream.bind(("127.0.0.1", 0))
        source_port = int(upstream.getsockname()[1])
        upstream.listen(1)
        reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        reservation.bind(("127.0.0.1", 0))
        relay_port = int(reservation.getsockname()[1])
        reservation.close()
        accepted = threading.Event()
        stop = threading.Event()

        def drain_upstream() -> None:
            connection, _address = upstream.accept()
            connection.settimeout(0.2)
            accepted.set()
            try:
                while not stop.is_set():
                    try:
                        if not connection.recv(65536):
                            break
                    except socket.timeout:
                        continue
            finally:
                connection.close()

        thread = threading.Thread(target=drain_upstream, daemon=True)
        thread.start()
        relay = subprocess.Popen(
            [
                str(relay_binary),
                "-d",
                str(runner.paths["t_messages"]),
                "-ip",
                "127.0.0.1",
                "-p",
                str(source_port),
                "-lp",
                str(relay_port),
            ],
            cwd=str(runner.paths["ttracer_dir"]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        clients = []
        client_handles = []
        try:
            self.assertTrue(accepted.wait(5), "relay never connected to fake trace source")
            with tempfile.TemporaryDirectory() as temp:
                clients.append(
                    subprocess.Popen(
                        [
                            str(record_binary),
                            "-d",
                            str(runner.paths["t_messages"]),
                            "-o",
                            str(Path(temp) / "recorder.raw"),
                            "-ip",
                            "127.0.0.1",
                            "-p",
                            str(relay_port),
                            "-OFF",
                            "-on",
                            "NRUE_MAC_DCI_GRANT",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                csv_path = Path(temp) / "observer.csv"
                csv_handle = csv_path.open("wb")
                client_handles.append(csv_handle)
                clients.append(
                    subprocess.Popen(
                        build_ttracer_csv_command(
                            str(csv_binary),
                            str(runner.paths["t_messages"]),
                            relay_port,
                            "NRUE_MAC_DCI_GRANT",
                            ("time", "direction", "rnti", "tbs", "ndi", "rv", "round"),
                        ),
                        stdout=csv_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    established = sum(
                        state == "01" and relay_port in {local, remote}
                        for local, remote, state in Runner._tcp_socket_rows()
                    )
                    if established >= 4:
                        break
                    time.sleep(0.05)
                self.assertGreaterEqual(established, 4)
                self.assertTrue(all(process.poll() is None for process in clients))
                self.assertIsNone(relay.poll())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and not csv_path.read_text():
                    time.sleep(0.05)
                self.assertEqual(
                    csv_path.read_text().strip().splitlines()[-1],
                    "time,direction,rnti,tbs,ndi,rv,round",
                )
        finally:
            for process in clients:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
            for handle in client_handles:
                if not handle.closed:
                    handle.close()
            if relay.poll() is None:
                os.killpg(relay.pid, signal.SIGTERM)
                relay.wait(timeout=3)
            stop.set()
            upstream.close()
            thread.join(timeout=2)

    def test_d0_minimal_trace_does_not_require_absent_rlc_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            root = output / "ttracer" / "D0_TRACE_MIN"
            (root / "gnb" / "csv").mkdir(parents=True)
            (root / "analysis").mkdir(parents=True)
            (root / "gnb" / "csv" / "GNB_MAC_PUSCH_POWER_CONTROL.csv").write_text(
                "rnti,snrx10,mcs\n100,60,8\n", encoding="utf-8"
            )
            (root / "analysis" / "ue_gnb_grant_validation.csv").write_text(
                "rnti,direction,ue_vs_gnb_mac_tbs_ratio\n"
                "0x0064,ul,1.0\n0x00c8,ul,1.0\n",
                encoding="utf-8",
            )
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            runner.command = Mock()

            result = runner._radio_and_grant_validity(
                "D0_TRACE_MIN",
                enforce_strong_rung=False,
                require_per_ue_radio=False,
            )

            self.assertEqual(result["per_ue"], {})
            self.assertIsNone(result["median_pusch_snr_db"])
            self.assertEqual(len(result["ue_gnb_tbs_ratios"]), 2)

    def test_d0_pairs_identical_demands_and_relaxes_only_minimal_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dg_a"
            output.mkdir()
            runner = Runner(DEFAULT_CONFIG, output, dry_run=True)
            common = {
                "demand_trace_sha256": "same",
                "receiver_onwire_mbps": 8.0,
                "latency_p95_ms": 100.0,
            }
            runner.run_trial = Mock(side_effect=[{}, dict(common), dict(common)])

            runner.d0()

            minimum = runner.run_trial.call_args_list[1]
            full = runner.run_trial.call_args_list[2]
            self.assertFalse(minimum.kwargs["enforce_strong_rung"])
            self.assertFalse(minimum.kwargs["require_per_ue_radio"])
            self.assertTrue(full.kwargs["enforce_strong_rung"])
            self.assertTrue(full.kwargs["require_per_ue_radio"])
            verdict = json.loads((output / "D0_instrumentation_verdict.json").read_text())
            self.assertTrue(verdict["demand_hash_match"])
            self.assertTrue(verdict["pass"])

    def test_interface_identity_is_stable_while_ip_is_discovered(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(
            [(row["ue_id"], row["iface"]) for row in config["radio"]["expected_tunnels"]],
            [(0, "oaitun_ue1"), (1, "oaitun_ue2")],
        )
        payload = json.dumps(
            [
                {
                    "ifname": "oaitun_ue1",
                    "addr_info": [{"family": "inet", "local": "10.0.0.3", "prefixlen": 24}],
                }
            ]
        )
        self.assertEqual(parse_interface_ipv4(payload, "oaitun_ue1"), "10.0.0.3")
        self.assertIsNone(parse_interface_ipv4("[]", "oaitun_ue1"))
        self.assertEqual(
            [row["imsi"] for row in config["radio"]["expected_tunnels"]],
            ["001010000000001", "001010000000002"],
        )
        dynamic = {
            0: {"ue_id": 0, "iface": "oaitun_ue1", "ip": "10.0.0.4"},
            1: {"ue_id": 1, "iface": "oaitun_ue2", "ip": "10.0.0.5"},
        }
        self.assertTrue(
            ue_network_contract_report(
                dynamic,
                config["radio"]["expected_tunnels"],
                config["radio"]["expected_ue_subnet"],
            )["pass"]
        )
        outside = {key: dict(value) for key, value in dynamic.items()}
        outside[1]["ip"] = "10.0.1.5"
        self.assertFalse(
            ue_network_contract_report(
                outside,
                config["radio"]["expected_tunnels"],
                config["radio"]["expected_ue_subnet"],
            )["pass"]
        )
        report = receiver_identity_report(
            [
                {"ue_id": 0, "source_ip": "192.168.70.134", "message_id": 1 << 28},
                {"ue_id": 1, "source_ip": "192.168.70.134", "message_id": 2 << 28},
            ],
            {0, 1},
            {"192.168.70.134"},
        )
        self.assertEqual(report["invalid_message_id_count"], 0)
        self.assertEqual(report["unexpected_nat_sources"], [])
        self.assertEqual(report["missing_ues"], [])
        mismatch = receiver_identity_report(
            [{"ue_id": 0, "source_ip": "192.168.70.200", "message_id": 2 << 28}],
            {0, 1},
            {"192.168.70.134"},
        )
        self.assertEqual(mismatch["invalid_message_id_count"], 1)
        self.assertEqual(mismatch["unexpected_nat_sources"], ["192.168.70.200"])

    def test_uicc_identity_contract_is_parsed_from_config_and_runtime_log(self) -> None:
        config_text = """
uicc0 = {
  imsi = "001010000000001";
  pdu_sessions = ({ dnn = "oai"; nssai_sst = 1; });
}
uicc1 = {
  imsi = "001010000000002";
  pdu_sessions = ({ dnn = "oai"; nssai_sst = 1; });
}
"""
        self.assertEqual(
            parse_uicc_profiles(config_text),
            {
                0: {"imsi": "001010000000001", "dnns": ["oai"]},
                1: {"imsi": "001010000000002", "dnns": ["oai"]},
            },
        )
        log = (
            "UICC simulation: IMSI=001010000000001, Ki=x\n"
            "UICC simulation: IMSI=001010000000002, Ki=x\n"
        )
        self.assertEqual(
            parse_runtime_uicc_imsis(log),
            ["001010000000001", "001010000000002"],
        )

    def test_sender_route_uses_bind_and_tunnel_evidence_before_nat(self) -> None:
        sender = {
            "socket_bindings": {
                "0": {"requested_bind_ip": "10.0.0.3", "actual_local_ip": "10.0.0.3", "actual_local_port": 10001},
                "1": {"requested_bind_ip": "10.0.0.2", "actual_local_ip": "10.0.0.2", "actual_local_port": 10002},
            },
            "per_ue": {
                "0": {"sent_onwire_bytes": 3_688_668},
                "1": {"sent_onwire_bytes": 4_098_520},
            },
        }
        network = {
            0: {"ue_id": 0, "iface": "oaitun_ue1", "ip": "10.0.0.3"},
            1: {"ue_id": 1, "iface": "oaitun_ue2", "ip": "10.0.0.2"},
        }
        tunnels = {
            "ue0": {"tx_bytes_delta": 3_737_808},
            "ue1": {"tx_bytes_delta": 4_153_120},
        }
        report = sender_route_report(
            sender, network, tunnels, tx_ratio_min=1.0, tx_ratio_max=1.08
        )
        self.assertTrue(report["pass"])
        sender["socket_bindings"]["0"]["actual_local_ip"] = "10.0.0.2"
        self.assertFalse(
            sender_route_report(
                sender, network, tunnels, tx_ratio_min=1.0, tx_ratio_max=1.08
            )["pass"]
        )

    def test_channel_state_and_per_ue_radio_evidence_are_parsed(self) -> None:
        channel = parse_channel_models(
            "model 2 rfsimu_channel_ue0 type AWGN:\n"
            "max Doppler: 0 path loss: 0.000000  noise: -4.000000 rchannel offset: 0\n"
            "model 3 rfsimu_channel_ue1 type AWGN:\n"
            "max Doppler: 0 path loss: 0.000000  noise: -4.000000 rchannel offset: 0\n"
        )
        self.assertEqual(channel["rfsimu_channel_ue0"]["model_index"], 2)
        self.assertEqual(channel["rfsimu_channel_ue1"]["noise_power_db"], -4.0)
        radio = per_ue_radio_summary(
            [
                {"rnti": "100", "snrx10": "82", "mcs": "9"},
                {"rnti": "100", "snrx10": "84", "mcs": "10"},
                {"rnti": "200", "snrx10": "80", "mcs": "8"},
            ],
            [
                {"ue_id": "0", "rnti": "100"},
                {"ue_id": "0", "rnti": "100"},
                {"ue_id": "1", "rnti": "200"},
            ],
        )
        self.assertEqual(radio["0"]["rnti"], 100)
        self.assertAlmostEqual(radio["0"]["median_pusch_snr_db"], 8.3)
        self.assertEqual(radio["1"]["median_ul_mcs"], 8.0)

    def test_dry_run_writes_completion_without_oai(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "dry"
            status = Runner(DEFAULT_CONFIG, output, dry_run=True).run()
            self.assertEqual(status, 0)
            self.assertTrue((output / "COMPLETED.json").exists())
            self.assertFalse((output / "FAILED.json").exists())


class DecisionContractTest(unittest.TestCase):
    def test_pair_requires_effect_and_goodput(self) -> None:
        config = load_config(DEFAULT_CONFIG)["decision"]
        greedy = {
            "trial_id": "A6",
            "demand_trace_sha256": "same",
            "worst_deadline_0.25s_fraction": 0.40,
            "worst_deadline_0.50s_fraction": 0.50,
            "worst_complete_latency_p95_ms": 500.0,
            "worst_max_starvation_ms": 1000.0,
            "aggregate_complete_goodput_mbps": 7.0,
        }
        central = {
            "trial_id": "A7",
            "demand_trace_sha256": "same",
            "worst_deadline_0.25s_fraction": 0.46,
            "worst_deadline_0.50s_fraction": 0.56,
            "worst_complete_latency_p95_ms": 400.0,
            "worst_max_starvation_ms": 750.0,
            "aggregate_complete_goodput_mbps": 6.9,
        }
        effect = pair_effect(greedy, central, config)
        self.assertTrue(effect["meaningful_gap"])
        central["aggregate_complete_goodput_mbps"] = 6.0
        self.assertFalse(pair_effect(greedy, central, config)["meaningful_gap"])

    def test_service_families_reproduce_n2(self) -> None:
        for family in ("constant_ceiling", "saturating", "power_law"):
            self.assertAlmostEqual(service_family(family, 2, 10.4, 9.8, 36.7), 9.8)

    def test_full_analyzer_contract_accepts_stage_with_entry_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            run_dir.mkdir()
            config = load_config(DEFAULT_CONFIG)
            stage = {
                "schema_version": config["schema_version"],
                "stage": config["stage"],
                "blocks": {
                    block: {
                        "mu_hat_mbps": 10.0,
                        "service_conversion": 0.8,
                        "strong_rung_entry_gate": {"both_uplinks_on_registered_strong_rung": True},
                    }
                    for block in ("A", "B")
                },
                "trials": [dict(row) for row in config["trials"]],
            }
            (run_dir / "stage_manifest.json").write_text(json.dumps(stage), encoding="utf-8")
            for trial in config["trials"]:
                trial_id = str(trial["id"])
                trial_dir = run_dir / "runs" / trial_id
                trial_dir.mkdir(parents=True)
                demand_hash = (
                    "pair-a"
                    if trial_id in {"A6", "A7"}
                    else "pair-b"
                    if trial_id in {"A8", "A9"}
                    else trial_id
                )
                (trial_dir / "sender_summary.json").write_text(
                    json.dumps(
                        {
                            "start_raw_ns": 1_000_000_000,
                            "duration_target_s": float(trial["duration_s"]),
                            "demand_trace_sha256": demand_hash,
                        }
                    ),
                    encoding="utf-8",
                )
                (trial_dir / "sender_demands.csv").write_text(
                    "ue_id,frame_id\n0,1\n0,2\n1,1\n1,2\n",
                    encoding="utf-8",
                )
                (trial_dir / "receiver_frames.csv").write_text(
                    "ue_id,frame_id,complete_raw_ns,complete_latency_ms,onwire_bytes\n"
                    "0,1,1100000000,100,409852\n"
                    "0,2,2100000000,100,409852\n"
                    "1,1,1200000000,200,409852\n"
                    "1,2,2200000000,200,409852\n",
                    encoding="utf-8",
                )
                if trial_id == "A4":
                    (trial_dir / "receiver_chunks.csv").write_text(
                        "recv_raw_ns,ue_id,onwire_bytes\n"
                        "1100000000,0,409852\n1200000000,1,409852\n",
                        encoding="utf-8",
                    )

            with patch(
                "sys.argv",
                [
                    "analyze",
                    "--run-dir",
                    str(run_dir),
                    "--config",
                    str(DEFAULT_CONFIG),
                ],
            ):
                status = analyze_main()

            self.assertEqual(status, 0)
            summary = json.loads((run_dir / "results_summary.json").read_text())
            self.assertIn(summary["decision"], {"STOP_CHEAP_NO", "CANDIDATE_GO_DG_B_HUMAN_REVIEW_REQUIRED"})
            self.assertFalse(summary["next_stage_launched"])
            self.assertTrue((run_dir / "DG_A_DECISION.md").exists())


class CorrectedReanalysisContractTest(unittest.TestCase):
    def test_max_min_serves_cold_demand_before_sharing_hot_residual(self) -> None:
        demand = [0.088] * 10 + [0.0055] * 40
        central = max_min_allocate(demand, 0.70)
        local = [min(value, 0.70 / 50) for value in demand]

        for value in central[:10]:
            self.assertAlmostEqual(value, 0.048)
        for value in central[10:]:
            self.assertAlmostEqual(value, 0.0055)
        self.assertAlmostEqual(sum(central), 0.70)
        self.assertAlmostEqual(min(value / need for value, need in zip(local, demand)), 0.1590909091)
        self.assertAlmostEqual(min(value / need for value, need in zip(central, demand)), 0.5454545455)
        self.assertTrue(all(new + 1e-12 >= old for old, new in zip(local, central)))

    def test_arrival_blueprint_matches_tick_credit_with_multiple_frames(self) -> None:
        blueprint, end_tick = arrival_blueprint(
            [0.8],
            onwire_bytes=1000,
            tick_s=0.05,
            minimum_arrivals_per_ue=8,
            demand_seed=1,
            synchronized=True,
            maximum_demands=100,
        )

        self.assertEqual(end_tick, 1)
        self.assertEqual([tick for tick, _ue, _demand in blueprint], [0] * 5 + [1] * 5)

    def test_deadline_denominator_includes_replaced_and_end_skipped(self) -> None:
        demands = [
            SimDemand(0, 0, 0, status="replaced"),
            SimDemand(1, 0, 0, status="admitted", admitted_tick=0, completion_s=0.10),
            SimDemand(2, 0, 1, status="skipped_end"),
        ]

        metrics = simulated_trial_metrics(
            demands,
            controller="decentralized_c1",
            tick_s=0.05,
            end_tick=1,
            onwire_bytes=1000,
            deadlines=[0.25, 0.50],
        )

        self.assertEqual(metrics["per_ue"]["0"]["demand_frames"], 3)
        self.assertAlmostEqual(metrics["worst_deadline_0.25s_fraction"], 1 / 3)
        self.assertAlmostEqual(metrics["worst_deadline_0.50s_fraction"], 1 / 3)

    def test_v2_config_is_locked_to_source_decision_contract(self) -> None:
        source = load_config(DEFAULT_CONFIG)
        model_path = DEFAULT_CONFIG.with_name("dg_a_reanalysis_v2.yaml")
        model = yaml.safe_load(model_path.read_text(encoding="utf-8"))

        validate_model_config(model, source)
        self.assertEqual(
            [row["name"] for row in model["allocation_envelopes"]],
            ["ideal_max_min", "measured_residual_max_min"],
        )

    def test_v2_analyzer_refuses_output_inside_immutable_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            child = source / "reanalysis"
            with patch(
                "sys.argv",
                [
                    "analyze_v2",
                    "--run-dir",
                    str(source),
                    "--config",
                    str(DEFAULT_CONFIG),
                    "--model-config",
                    str(DEFAULT_CONFIG.with_name("dg_a_reanalysis_v2.yaml")),
                    "--output-dir",
                    str(child),
                ],
            ):
                with self.assertRaisesRegex(SystemExit, "new sibling"):
                    analyze_v2_main()


if __name__ == "__main__":
    unittest.main()
