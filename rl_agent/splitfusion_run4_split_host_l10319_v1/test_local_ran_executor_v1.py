"""Offline fault-injection tests for the attempt-scoped local-RAN executor."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import contract as C
from . import local_ran_executor_v1 as E
from . import local_ran_lifecycle_v1 as L
from . import split_host_phase6_coordinator_v1 as S


ROUTE_PREFLIGHT = json.dumps([{
    "dst": "DEST", "gateway": C.default_topology().remote_lan_ip,
    "dev": "wlp130s0f0", "prefsrc": C.default_topology().local_lan_ip,
}])
TUNNEL = json.dumps([{"addr_info": [{"family": "inet", "local": L.UE_IP}]}])
SOURCE_ROUTE = json.dumps([{
    "dst": C.default_topology().edge_ip, "dev": L.UE_INTERFACE,
    "table": S.POLICY_TABLE, "prefsrc": L.UE_IP,
}])


class _Proc:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


class FakeOps(E.LocalRanOps):
    def __init__(self, *, fail: str = "") -> None:
        self.fail = fail
        self.events: list[str] = []
        self.next_pid = 100
        self.commands: list[tuple[str, ...]] = []
        self.alive: dict[str, bool] = {}
        self.tunnel_present = False
        self.rule_present = False
        self.route_present = False

    def _hit(self, stage: str) -> None:
        self.events.append(stage)
        if self.fail == stage:
            raise RuntimeError(f"injected {stage}")

    def run(self, argv, *, timeout_s):
        key = " ".join(argv)
        self.commands.append(tuple(argv))
        if argv == ("sudo", "-n", "true"):
            self._hit("sudo-preflight")
            return E.CommandResultV1(0)
        if "--materialize-radio-config" in argv:
            self._hit("materialize")
            return E.CommandResultV1(0)
        if "--prepare-runtime" in argv:
            self._hit("derive")
            return E.CommandResultV1(0)
        if argv == S.source_route_probe_argv():
            self._hit("source-route")
            return E.CommandResultV1(0, SOURCE_ROUTE)
        if argv[:4] == ("ip", "-j", "route", "get") and "from" in argv and "iif" not in argv:
            destination = argv[4]
            self._hit(f"preflight-route:{destination}")
            return E.CommandResultV1(0, ROUTE_PREFLIGHT.replace("DEST", destination))
        if argv[:2] == ("ping", "-c"):
            destination = argv[-1]
            self._hit(f"preflight-ping:{destination}")
            return E.CommandResultV1(0)
        if argv == ("ip", "-j", "link", "show", "dev", L.UE_INTERFACE):
            self.events.append("tunnel-before")
            return E.CommandResultV1(0 if self.tunnel_present else 1, "[]")
        if argv == ("ip", "-j", "rule", "show"):
            self._hit("snapshot-rules")
            rows = ([{"priority": E.POLICY_PRIORITY, "from": L.UE_IP,
                      "table": S.POLICY_TABLE}] if self.rule_present else [])
            return E.CommandResultV1(0, json.dumps(rows))
        if argv == ("ip", "-j", "route", "show", "table", str(S.POLICY_TABLE)):
            self._hit("snapshot-table")
            rows = ([{"dst": f"{C.default_topology().edge_ip}/32"}]
                    if self.route_present else [])
            return E.CommandResultV1(0, json.dumps(rows))
        if argv == ("ip", "-j", "-4", "addr", "show", "dev", L.UE_INTERFACE):
            self.events.append("tunnel-address")
            return E.CommandResultV1(0, TUNNEL) if self.tunnel_present else E.CommandResultV1(1)
        if argv[0] == "ping" and "-I" in argv:
            self._hit("attach-ping")
            return E.CommandResultV1(0)
        if argv[:6] == ("sudo", "-n", "ip", "route", "add", f"{C.default_topology().edge_ip}/32"):
            self._hit("add-route")
            self.route_present = True
            return E.CommandResultV1(0)
        if argv[:6] == ("sudo", "-n", "ip", "rule", "add", "priority"):
            self._hit("add-rule")
            self.rule_present = True
            return E.CommandResultV1(0)
        if argv[:6] == ("sudo", "-n", "ip", "rule", "del", "priority"):
            self._hit("remove-rule")
            self.rule_present = False
            return E.CommandResultV1(0)
        if argv[:6] == ("sudo", "-n", "ip", "route", "del", f"{C.default_topology().edge_ip}/32"):
            self._hit("remove-route")
            self.route_present = False
            return E.CommandResultV1(0)
        if argv == ("sudo", "-n", "ip", "link", "delete", "dev", L.UE_INTERFACE):
            self._hit("remove-tunnel")
            self.tunnel_present = False
            return E.CommandResultV1(0)
        raise AssertionError(f"unexpected command: {key}")

    def spawn(self, plan, *, log_path):
        self._hit(f"spawn:{plan.role}")
        self.next_pid += 1
        owned = L.OwnedProcess(
            role=plan.role, pid=self.next_pid, pgid=self.next_pid,
            executable=plan.argv[0], argv_sha256=plan.argv_sha256,
            started_monotonic_raw_ns=self.next_pid,
        )
        self.alive[plan.role] = True
        if plan.role == "ue":
            self.tunnel_present = True
        return E.SpawnedProcessV1(owned, _Proc(self.next_pid), object(), plan)

    def process_alive(self, process):
        return self.alive.get(process.owned.role, False)

    def stop_process(self, process):
        role = process.owned.role
        self._hit(f"stop:{role}")
        self.alive[role] = False
        if role == "ue":
            # Model OAI's normal tunnel removal on exit.
            self.tunnel_present = False

    def wait_tcp(self, host, port, *, timeout_s):
        self._hit("wait-tracer")

    def sleep(self, seconds):
        self.events.append("gnb-lead")

    def restore_channel(self):
        self._hit("restore-channel")
        return {"verified": True, "noise_power_db": -50.0}


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / "radio"
        self.attempt = self.root / "attempt"
        self.plan = L.build_local_ran_plan(self.state)

    def executor(self, ops: FakeOps) -> E.LocalRanExecutorV1:
        return E.LocalRanExecutorV1(
            plan=self.plan, attempt_dir=self.attempt, ops=ops,
            config=E.LocalRanExecutorConfigV1(
                command_timeout_s=1, gnb_lead_s=0.001,
                attach_timeout_s=1, attach_poll_s=0.001,
                tracer_ready_timeout_s=1),
        )

    def patched_start(self, ops: FakeOps):
        attested = {
            "split_host_runtime_gnb_sha256": "a" * 64,
            "phase14a_effective_ue_sha256": "b" * 64,
        }
        with mock.patch.object(L, "load_attestation", return_value=attested), \
             mock.patch.object(L, "attached_attestation", return_value={
                 "schema": L.ATTACHED_SCHEMA, "status": "LOCAL_RAN_ATTACHED_TO_REMOTE_CN"}):
            return self.executor(ops).start()


class SuccessTests(Fixture):
    def test_exact_start_cleanup_order_and_no_remote_operations(self) -> None:
        ops = FakeOps()
        session = self.patched_start(ops)
        self.assertEqual(tuple(session.processes),
                         ("gnb", "ue", "tracer_multi", "tracer_record"))
        self.assertEqual(session.source_route["radio_path"], "PASS")
        self.assertEqual(session.policy_ownership.rules[0].priority, 31001)
        self.assertTrue((self.attempt / E.RULES_BEFORE).is_file())
        self.assertTrue((self.attempt / E.TABLE_BEFORE).is_file())
        report = session.close()
        self.assertTrue(report.ok)
        self.assertEqual(report.stopped_roles,
                         ("tracer_record", "tracer_multi", "ue", "gnb"))
        self.assertTrue(report.tunnel_absent_after_cleanup)
        self.assertEqual(report.policy_removed,
                         ("rule:31001", f"route:{C.default_topology().edge_ip}/32"))
        restore = ops.events.index("restore-channel")
        for stage in ("stop:tracer_record", "stop:tracer_multi", "stop:ue", "stop:gnb"):
            self.assertLess(restore, ops.events.index(stage))
        self.assertLess(ops.events.index("stop:tracer_record"),
                        ops.events.index("stop:tracer_multi"))
        self.assertLess(ops.events.index("stop:tracer_multi"),
                        ops.events.index("remove-rule"))
        self.assertLess(ops.events.index("remove-rule"), ops.events.index("stop:ue"))
        self.assertLess(ops.events.index("remove-rule"),
                        ops.events.index("remove-route"))
        self.assertLess(ops.events.index("remove-route"), ops.events.index("stop:ue"))
        self.assertLess(ops.events.index("stop:ue"), ops.events.index("stop:gnb"))
        rendered = "\n".join(ops.events).lower()
        self.assertNotIn("docker", rendered)
        self.assertNotIn("ssh", rendered)
        for command in ops.commands:
            if command[:1] == ("sudo",):
                self.assertEqual(command[1], "-n", command)

    def test_close_is_idempotent(self) -> None:
        ops = FakeOps()
        session = self.patched_start(ops)
        first = session.close()
        event_count = len(ops.events)
        self.assertIs(session.close(), first)
        self.assertEqual(len(ops.events), event_count)


class RefusalTests(Fixture):
    def test_preexisting_tunnel_refused_before_materialization(self) -> None:
        ops = FakeOps()
        ops.tunnel_present = True
        with self.assertRaises(E.LocalRanStartError):
            self.patched_start(ops)
        self.assertNotIn("materialize", ops.events)

    def test_preexisting_priority_and_route_each_refused(self) -> None:
        for field in ("rule_present", "route_present"):
            ops = FakeOps()
            setattr(ops, field, True)
            with self.subTest(field=field), self.assertRaises(E.LocalRanStartError):
                self.patched_start(ops)
            self.assertNotIn("spawn:gnb", ops.events)
            self.attempt = self.root / f"attempt-{field}"

    def test_source_route_bypass_fails_and_rolls_back(self) -> None:
        class Bypass(FakeOps):
            def run(self, argv, *, timeout_s):
                result = super().run(argv, timeout_s=timeout_s)
                if argv == S.source_route_probe_argv():
                    return E.CommandResultV1(0, json.dumps([{
                        "dst": C.default_topology().edge_ip,
                        "dev": "wlp130s0f0", "gateway": C.default_topology().remote_lan_ip,
                        "prefsrc": L.UE_IP,
                    }]))
                return result
        ops = Bypass()
        with self.assertRaises(E.LocalRanStartError) as caught:
            self.patched_start(ops)
        self.assertIn("remove-rule", ops.events)
        self.assertIn("remove-route", ops.events)
        self.assertTrue(caught.exception.cleanup.channel_restored)


class FaultInjectionTests(Fixture):
    def test_sudo_preflight_failure_precedes_every_mutation(self) -> None:
        ops = FakeOps(fail="sudo-preflight")
        with self.assertRaises(E.LocalRanStartError) as caught:
            self.patched_start(ops)
        self.assertIn("injected sudo-preflight", str(caught.exception.primary))
        self.assertEqual(ops.events, ["sudo-preflight"])
        self.assertFalse(caught.exception.cleanup.channel_restore_attempted)

    def test_every_start_stage_preserves_primary_and_attempts_cleanup(self) -> None:
        stages = (
            "materialize", "derive",
            f"preflight-route:{C.default_topology().amf_ip}",
            f"preflight-ping:{C.default_topology().amf_ip}",
            f"preflight-route:{C.default_topology().ext_dn_ip}",
            f"preflight-ping:{C.default_topology().ext_dn_ip}",
            "snapshot-rules", "snapshot-table", "spawn:gnb", "spawn:ue",
            "sudo-preflight",
            "attach-ping", "add-route", "add-rule", "source-route",
            "spawn:tracer_multi", "wait-tracer", "spawn:tracer_record",
        )
        for index, stage in enumerate(stages):
            with self.subTest(stage=stage):
                self.attempt = self.root / f"fault-{index}"
                ops = FakeOps(fail=stage)
                with self.assertRaises(E.LocalRanStartError) as caught:
                    self.patched_start(ops)
                self.assertIn(f"injected {stage}", str(caught.exception.primary))
                if "spawn:gnb" in ops.events and stage != "spawn:gnb":
                    self.assertIn("restore-channel", ops.events)
                    self.assertIn("stop:gnb", ops.events)
                self.assertNotIn("docker", " ".join(ops.events).lower())

    def test_partial_route_is_removed_when_rule_add_fails(self) -> None:
        ops = FakeOps(fail="add-rule")
        with self.assertRaises(E.LocalRanStartError):
            self.patched_start(ops)
        self.assertIn("remove-route", ops.events)
        self.assertNotIn("remove-rule", ops.events)

    def test_immediate_health_failure_still_stops_spawned_process(self) -> None:
        class Unhealthy(FakeOps):
            def process_alive(self, process):
                if process.owned.role == "gnb":
                    return False
                return super().process_alive(process)

        ops = Unhealthy()
        with self.assertRaises(E.LocalRanStartError) as caught:
            self.patched_start(ops)
        self.assertIn("gnb exited at startup", str(caught.exception.primary))
        self.assertIn("stop:gnb", ops.events)
        self.assertFalse(ops.alive["gnb"])

    def test_cleanup_failures_are_independent_and_primary_survives(self) -> None:
        class ManyFailures(FakeOps):
            def _hit(self, stage):
                self.events.append(stage)
                if stage in {"source-route", "restore-channel", "stop:ue", "remove-route"}:
                    raise RuntimeError(f"injected {stage}")
        ops = ManyFailures()
        with self.assertRaises(E.LocalRanStartError) as caught:
            self.patched_start(ops)
        self.assertIn("injected source-route", str(caught.exception.primary))
        self.assertIn("stop:gnb", ops.events)
        self.assertGreaterEqual(len(caught.exception.cleanup.errors), 3)


class SourceSafetyTests(unittest.TestCase):
    def test_module_contains_no_remote_or_broad_cleanup_command(self) -> None:
        source = Path(E.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ip rule flush", source)
        self.assertNotIn("ip route flush", source)
        self.assertNotIn("docker compose", source)
        self.assertNotIn("ssh ", source)
        self.assertNotIn("remote_cn_cleanup", source)


class _RootOps(E.LocalRanOps):
    root_discovery_timeout_s = 0.0

    def __init__(self, identities):
        self.identities = identities
        self.members = sorted(identities)
        self.launched = ()
        self.signals = []

    def _launch(self, argv, *, cwd, log_handle, new_session):
        self.launched = tuple(argv)
        return _Proc(500)

    def _group_pids(self, pgid, *, root_owned):
        return list(self.members)

    def _process_identity(self, pid, *, root_owned):
        return self.identities[pid]

    def _signal_members(self, members, sig, *, root_owned, pgid):
        self.signals.append((tuple(members), sig, root_owned, pgid))
        self.members.clear()


class RootOwnedProcessTests(unittest.TestCase):
    def make_plan(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        executable = root / "nr-softmodem"
        executable.write_bytes(b"binary")
        executable.chmod(0o755)
        plan = L.ProcessPlan(
            role="gnb", host="W10275", argv=(str(executable), "--flag", "value"),
            env_set=(("SCENESENSE_MCS_POLICY", "sinr"),),
            env_unset=("SCENESENSE_FORCE_UL_MCS",),
            stdout_name="gnb.log",
        ).validate()
        return root, executable.resolve(), plan

    def test_wrapper_is_not_recorded_and_exact_root_child_is(self) -> None:
        root, executable, plan = self.make_plan()
        ops = _RootOps({
            500: ("/usr/bin/sudo", ("sudo", "-n"), 700),
            501: (str(executable), plan.argv, 700),
        })
        ops.root_discovery_timeout_s = 1.0
        with mock.patch.object(E.os, "getpgid", return_value=700):
            spawned = ops.spawn(plan, log_path=root / "gnb.log")
        self.assertEqual(spawned.owned.pid, 501)
        self.assertEqual(spawned.owned.pgid, 700)
        self.assertTrue(spawned.root_owned)
        self.assertEqual(ops.launched[:3], ("sudo", "-n", "env"))
        self.assertIn(("-u", "SCENESENSE_FORCE_UL_MCS"),
                      tuple(zip(ops.launched, ops.launched[1:])))
        self.assertIn("SCENESENSE_MCS_POLICY=sinr", ops.launched)
        self.assertEqual(ops.launched[-len(plan.argv):], plan.argv)
        ops.stop_process(spawned)
        self.assertEqual(ops.signals[0][0], (500, 501))
        self.assertTrue(ops.signals[0][2])

    def test_duplicate_exact_children_are_refused(self) -> None:
        root, executable, plan = self.make_plan()
        ops = _RootOps({
            500: ("/usr/bin/sudo", ("sudo",), 700),
            501: (str(executable), plan.argv, 700),
            502: (str(executable), plan.argv, 700),
        })
        ops.root_discovery_timeout_s = 1.0
        with mock.patch.object(E.os, "getpgid", return_value=700), \
             self.assertRaises(E.LocalRanExecutorError):
            ops.spawn(plan, log_path=root / "ambiguous.log")
        self.assertEqual(ops.signals[0][0], (500, 501, 502))

    def test_matching_executable_with_wrong_argv_is_refused(self) -> None:
        root, executable, plan = self.make_plan()
        ops = _RootOps({
            500: ("/usr/bin/sudo", ("sudo",), 700),
            501: (str(executable), (str(executable), "--wrong"), 700),
        })
        ops.root_discovery_timeout_s = 1.0
        with mock.patch.object(E.os, "getpgid", return_value=700), \
             self.assertRaises(E.LocalRanExecutorError):
            ops.spawn(plan, log_path=root / "mismatch.log")
        self.assertEqual(ops.signals[0][0], (500, 501))

    def test_cleanup_identity_drift_refuses_every_signal(self) -> None:
        root, executable, plan = self.make_plan()
        ops = _RootOps({
            500: ("/usr/bin/sudo", ("sudo", "-n"), 700),
            501: (str(executable), plan.argv, 700),
        })
        ops.root_discovery_timeout_s = 1.0
        with mock.patch.object(E.os, "getpgid", return_value=700):
            spawned = ops.spawn(plan, log_path=root / "identity-drift.log")
        ops.identities[501] = (
            str(executable), (str(executable), "--foreign"), 700)
        with self.assertRaises(L.LocalRanLifecycleError):
            ops.stop_process(spawned)
        self.assertEqual(ops.signals, [])
        self.assertTrue(spawned.log_handle.closed)

    def test_cleanup_refuses_surviving_group_without_attested_pid(self) -> None:
        root, executable, plan = self.make_plan()
        ops = _RootOps({
            500: ("/usr/bin/sudo", ("sudo", "-n"), 700),
            501: (str(executable), plan.argv, 700),
        })
        ops.root_discovery_timeout_s = 1.0
        with mock.patch.object(E.os, "getpgid", return_value=700):
            spawned = ops.spawn(plan, log_path=root / "owner-vanished.log")
        ops.members = [500]
        with self.assertRaises(E.LocalRanExecutorError):
            ops.stop_process(spawned)
        self.assertEqual(ops.signals, [])
        self.assertTrue(spawned.log_handle.closed)

    def test_zero_root_child_is_refused(self) -> None:
        root, _executable, plan = self.make_plan()
        ops = _RootOps({500: ("/usr/bin/sudo", ("sudo",), 700)})
        with mock.patch.object(E.os, "getpgid", return_value=700), \
             self.assertRaises(E.LocalRanExecutorError):
            ops.spawn(plan, log_path=root / "missing.log")
        self.assertEqual(ops.signals[0][0], (500,))


if __name__ == "__main__":
    unittest.main()
