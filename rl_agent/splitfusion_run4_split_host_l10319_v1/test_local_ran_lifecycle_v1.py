"""Offline tests for the W10275 local-RAN / L10319 remote-CN seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from . import local_ran_lifecycle_v1 as L


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fixture(unittest.TestCase):
    def make_state(self) -> tuple[tempfile.TemporaryDirectory, Path, Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        state = Path(temporary.name).resolve()
        gnb = state / "effective_gnb_100mhz_4d5u_clean_minus50.conf"
        ue = state / "effective_ue_100mhz_clean_minus50.conf"
        gnb.write_text(
            'amf_ip_address = ({ ipv4 = "192.168.70.132"; });\n'
            'NETWORK_INTERFACES : {\n'
            ' GNB_IPV4_ADDRESS_FOR_NG_AMF = "192.168.70.129/24";\n'
            ' GNB_IPV4_ADDRESS_FOR_NGU = "192.168.70.129/24";\n'
            '};\nnoise_power_dB = -50;\n', encoding="utf-8")
        ue.write_text("ue=true;\nnoise_power_dB = -50;\n", encoding="utf-8")
        materialization = {
            "schema": "scenesense.splitfusion_phase14a_radio_materialization.v1",
            "status": "MATERIALIZED_NOT_LAUNCHED",
            "radio_profile_id": L.RADIO_PROFILE_ID,
            "source_gnb_sha256": "1" * 64,
            "selected_gnb_before_channelmod_sha256": "2" * 64,
            "source_ue_sha256": "3" * 64,
            "source_channel_sha256": "4" * 64,
            "effective_gnb_path": str(gnb),
            "effective_gnb_sha256": _sha(gnb),
            "effective_ue_path": str(ue),
            "effective_ue_sha256": _sha(ue),
            "clean_noise_power_db": -50.0,
            "cpu_reconciliation": "PHASE14A_CPU_RECONCILIATION_PASSED",
        }
        path = state / "radio_materialization.json"
        path.write_text(json.dumps(materialization, sort_keys=True), encoding="utf-8")
        return temporary, state, gnb, ue


class SourceAndDerivationTests(Fixture):
    def test_real_inherited_source_pins(self) -> None:
        observed = L.verify_source_pins(pins=L.SOURCE_PINS[:6])
        self.assertEqual(set(observed), {pin.name for pin in L.SOURCE_PINS[:6]})
        self.assertEqual(observed["phase14a_launcher"],
                         "8e02f0913338a187bff1de24bfea09d7eca03607da0e383ef8ad4a005b64ce61")

    def test_runtime_copy_is_exact_and_originals_are_unchanged(self) -> None:
        temporary, state, gnb, ue = self.make_state()
        self.addCleanup(temporary.cleanup)
        before = (_sha(gnb), _sha(ue), _sha(state / "radio_materialization.json"))
        result = L.prepare_runtime_gnb(state, _source_verifier=lambda _root: {pin.name: pin.sha256 for pin in L.SOURCE_PINS})
        runtime = state / L.RUNTIME_GNB_NAME
        self.assertTrue(runtime.is_file())
        text = runtime.read_text()
        self.assertIn('amf_ip_address = ({ ipv4 = "192.168.70.132"; });', text)
        self.assertEqual(text.count('"10.21.16.222/24"'), 2)
        self.assertEqual(before,
                         (_sha(gnb), _sha(ue), _sha(state / "radio_materialization.json")))
        self.assertFalse(result["original_phase14a_attached_same_file_assertion_satisfied"])
        self.assertEqual(result["attached_state_authority"],
                         "SPLIT_HOST_LOCAL_RAN_ATTESTATION_ONLY")
        self.assertFalse(result["cross_host_monotonic_comparison_permitted"])
        loaded = L.load_attestation(
            state, _source_verifier=lambda _root: {
                pin.name: pin.sha256 for pin in L.SOURCE_PINS
            })
        self.assertEqual(loaded["split_host_runtime_gnb_sha256"], _sha(runtime))

    def test_create_only_and_tamper_fail_closed(self) -> None:
        temporary, state, gnb, _ue = self.make_state()
        self.addCleanup(temporary.cleanup)
        L.prepare_runtime_gnb(state, _source_verifier=lambda _root: {pin.name: pin.sha256 for pin in L.SOURCE_PINS})
        with self.assertRaises(L.LocalRanLifecycleError):
            L.prepare_runtime_gnb(state, _source_verifier=lambda _root: {pin.name: pin.sha256 for pin in L.SOURCE_PINS})
        gnb.write_text(gnb.read_text() + "# drift\n", encoding="utf-8")
        with self.assertRaises(L.LocalRanLifecycleError):
            L.load_attestation(
                state, _source_verifier=lambda _root: {
                    pin.name: pin.sha256 for pin in L.SOURCE_PINS
                })

    def test_attestation_consumer_refuses_source_drift(self) -> None:
        temporary, state, _gnb, _ue = self.make_state()
        self.addCleanup(temporary.cleanup)
        pins = {pin.name: pin.sha256 for pin in L.SOURCE_PINS}
        L.prepare_runtime_gnb(state, _source_verifier=lambda _root: pins)
        drifted = {**pins, "phase14a_launcher": "0" * 64}
        with self.assertRaises(L.LocalRanLifecycleError):
            L.load_attestation(
                state, _source_verifier=lambda _root: drifted
            )

    def test_foreign_materialization_fields_and_escaped_paths_refused(self) -> None:
        temporary, state, _gnb, _ue = self.make_state()
        self.addCleanup(temporary.cleanup)
        path = state / "radio_materialization.json"
        document = json.loads(path.read_text())
        document["foreign"] = True
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(L.LocalRanLifecycleError):
            L.prepare_runtime_gnb(state, _source_verifier=lambda _root: {pin.name: pin.sha256 for pin in L.SOURCE_PINS})


class CommandPlanTests(unittest.TestCase):
    def test_plan_preserves_phase14a_radio_and_phase6_tracer_commands(self) -> None:
        state = Path("/tmp/registered-split-host-radio-state")
        plan = L.build_local_ran_plan(state)
        self.assertEqual(plan.gnb.env_set, (("SCENESENSE_MCS_POLICY", "sinr"),))
        self.assertEqual(plan.gnb.env_unset, (
            "SCENESENSE_FORCE_UL_MCS", "SCENESENSE_HOLD_MCS_FEW_SAMPLES",
            "SCENESENSE_AIMD_MAX_DROP"))
        self.assertEqual(plan.gnb.argv[1:4], ("-O", str(state / L.RUNTIME_GNB_NAME),
                                              "--gNBs.[0].min_rxtxtime"))
        self.assertEqual(plan.gnb.argv[-6:],
                         ("--telnetsrv.listenport", "9090", "--T_stdout", "2",
                          "--T_nowait", "--T_port", "2021")[-6:])
        for pair in (("-r", "273"), ("--numerology", "1"), ("--band", "78"),
                     ("-C", "3649260000"), ("--ssb", "516"),
                     ("--T_port", "2023")):
            self.assertIn(pair, tuple(zip(plan.ue.argv, plan.ue.argv[1:])))
        for event in L.UE_EVENTS:
            self.assertIn(event, plan.tracer_record.argv)
        self.assertIn(("-p", "2023"),
                      tuple(zip(plan.tracer_multi.argv, plan.tracer_multi.argv[1:])))
        self.assertIn(("-lp", "2123"),
                      tuple(zip(plan.tracer_multi.argv, plan.tracer_multi.argv[1:])))

    def test_no_local_plan_owns_or_stops_remote_cn(self) -> None:
        plan = L.build_local_ran_plan(Path("/tmp/radio"))
        all_plans = (plan.materialize, plan.derive_runtime, plan.gnb, plan.ue,
                     plan.tracer_multi, plan.tracer_record, plan.attach_probe)
        rendered = "\n".join(" ".join(item.argv).lower() for item in all_plans)
        for forbidden in ("docker", "compose", "cn_start", "cn_stop", "ssh"):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(plan.remote_cn_prerequisite_only)
        self.assertEqual(plan.policy_deadline_clock,
                         "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY")
        self.assertFalse(plan.cross_host_monotonic_comparison_permitted)

    def test_only_amf_and_ext_dn_are_preflighted_through_l10319(self) -> None:
        plan = L.build_local_ran_plan(Path("/tmp/radio"))
        self.assertEqual({p.destination for p in plan.preflight},
                         {"192.168.70.132", "192.168.70.135"})
        for probe in plan.preflight:
            self.assertIn("10.21.16.222", probe.route_argv)
            self.assertEqual(probe.route_argv[:4], ("ip", "-j", "route", "get"))
            self.assertEqual(probe.reachability_argv[:2], ("ping", "-c"))


class EvidenceTests(unittest.TestCase):
    ROUTE = json.dumps([{
        "dst": "192.168.70.132", "gateway": "10.21.16.162",
        "dev": "wlp130s0f0", "prefsrc": "10.21.16.222",
    }])

    def test_routed_reachability_passes_only_exact_gateway_and_source(self) -> None:
        result = L.validate_route_evidence("192.168.70.132", self.ROUTE, 0)
        self.assertEqual(result["reachability"], "PASS")
        for field, value in (("gateway", "10.21.16.1"),
                             ("prefsrc", "10.21.16.9"), ("dev", "docker0")):
            row = json.loads(self.ROUTE)
            row[0][field] = value
            with self.subTest(field=field), self.assertRaises(L.LocalRanLifecycleError):
                L.validate_route_evidence("192.168.70.132", json.dumps(row), 0)
        with self.assertRaises(L.LocalRanLifecycleError):
            L.validate_route_evidence("192.168.70.132", self.ROUTE, 1)

    def test_tunnel_attachment_is_exact(self) -> None:
        payload = json.dumps([{"addr_info": [
            {"family": "inet", "local": "10.0.0.2"}
        ]}])
        self.assertEqual(L.validate_tunnel_attachment(payload, 0)["ipv4"], "10.0.0.2")
        with self.assertRaises(L.LocalRanLifecycleError):
            L.validate_tunnel_attachment(payload.replace("10.0.0.2", "10.0.0.3"), 0)


def _owned(role: str, pid: int, plan: L.ProcessPlan) -> L.OwnedProcess:
    return L.OwnedProcess(
        role=role, pid=pid, pgid=pid + 1000, executable=plan.argv[0],
        argv_sha256=plan.argv_sha256, started_monotonic_raw_ns=1000000 + pid,
    )


class OwnershipTests(Fixture):
    def process_fixture(self):
        plan = L.build_local_ran_plan(Path("/tmp/radio"))
        processes = (
            _owned("gnb", 101, plan.gnb), _owned("ue", 102, plan.ue),
            _owned("tracer_multi", 103, plan.tracer_multi),
            _owned("tracer_record", 104, plan.tracer_record),
        )
        observed = {
            item.pid: {"pid": item.pid, "pgid": item.pgid,
                       "executable": item.executable,
                       "argv_sha256": item.argv_sha256, "host": "W10275"}
            for item in processes
        }
        tunnel = L.TunnelOwnership(
            interface="oaitun_ue1", absent_before_start=True,
            observed_ipv4_after_attach="10.0.0.2", created_by_local_ue=True)
        return processes, observed, tunnel

    def test_cleanup_is_exact_local_reverse_order_and_never_remote_cn(self) -> None:
        processes, observed, tunnel = self.process_fixture()
        cleanup = L.cleanup_plan(processes, tunnel, observed)
        self.assertTrue(cleanup.restore_channel_before_process_stop)
        self.assertEqual(cleanup.restore_noise_power_db, -50.0)
        self.assertFalse(cleanup.remote_cn_cleanup_permitted)
        self.assertEqual(cleanup.policy_deadline_clock,
                         "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY")
        commands = cleanup.process_commands
        self.assertEqual([row.role for row in commands], [
            "stop_tracer_record", "stop_tracer_multi", "stop_ue", "stop_gnb",
            "remove_owned_ue_tunnel_if_present"])
        rendered = "\n".join(" ".join(row.argv).lower() for row in commands)
        self.assertNotIn("docker", rendered)
        self.assertNotIn("compose", rendered)
        self.assertNotIn("ssh", rendered)
        self.assertNotIn("l10319", rendered)
        self.assertIn("ip link delete dev oaitun_ue1", rendered)

    def test_pid_reuse_foreign_host_and_preexisting_tunnel_are_refused(self) -> None:
        processes, observed, tunnel = self.process_fixture()
        observed[101] = {**observed[101], "argv_sha256": "0" * 64}
        with self.assertRaises(L.LocalRanLifecycleError):
            L.cleanup_plan(processes, tunnel, observed)
        processes, observed, tunnel = self.process_fixture()
        foreign = L.OwnedProcess(**{**processes[0].__dict__, "host": "L10319"})
        with self.assertRaises(L.LocalRanLifecycleError):
            L.cleanup_plan((foreign,) + processes[1:], tunnel, observed)
        with self.assertRaises(L.LocalRanLifecycleError):
            L.cleanup_plan(processes, L.TunnelOwnership(
                "oaitun_ue1", False, "10.0.0.2", True), observed)

    def test_attached_claim_is_separate_and_local_clock_only(self) -> None:
        temporary, state, _gnb, _ue = self.make_state()
        self.addCleanup(temporary.cleanup)
        L.prepare_runtime_gnb(state, _source_verifier=lambda _root: {pin.name: pin.sha256 for pin in L.SOURCE_PINS})
        processes, _observed, _tunnel = self.process_fixture()
        routes = [
            {"destination": "192.168.70.132", "reachability": "PASS"},
            {"destination": "192.168.70.135", "reachability": "PASS"},
        ]
        attached = L.attached_attestation(
            state_dir=state, route_evidence=routes,
            tunnel_evidence={"interface": "oaitun_ue1", "ipv4": "10.0.0.2",
                             "ext_dn_reachability": "PASS"},
            processes=processes,
            _source_verifier=lambda _root: {
                pin.name: pin.sha256 for pin in L.SOURCE_PINS
            })
        self.assertEqual(attached["status"], "LOCAL_RAN_ATTACHED_TO_REMOTE_CN")
        self.assertFalse(attached["original_phase14a_attached_same_file_assertion_satisfied"])
        self.assertFalse(attached["remote_cn_lifecycle_owned"])
        self.assertFalse(attached["cross_host_monotonic_comparison_permitted"])
        self.assertFalse(attached["live_run_authorized"])


if __name__ == "__main__":
    unittest.main()
