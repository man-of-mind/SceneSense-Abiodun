"""Attempt-scoped executor for the registered split-host local RAN plan.

The executor deliberately owns only W10275 resources: the gNB, UE, UE
T-tracer relay/record processes, the UE tunnel created by that UE, and one
exact policy rule plus one exact table-9999 /32 route.  It never invokes
Docker, SSH, or any remote-CN operation.

All operating-system effects live behind :class:`LocalRanOps`.  This keeps
the lifecycle exhaustively fault-testable without launching OAI or touching
the host network.  The production implementation is used only when an
explicit caller invokes :meth:`LocalRanExecutorV1.start`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from typing import Any, Mapping, Optional, Sequence

from . import contract as C
from . import local_ran_lifecycle_v1 as L
from . import split_host_phase6_coordinator_v1 as S


SCHEMA = "scenesense.run4.split_host_local_ran_executor.v1"
POLICY_PRIORITY = 31001
ATTACHED_RECORD = "LOCAL_RAN_ATTACHED_EXECUTOR_V1.json"
RULES_BEFORE = "policy_rules_before.json"
TABLE_BEFORE = "policy_table_9999_before.json"


class LocalRanExecutorError(RuntimeError):
    """The bounded local-RAN lifecycle failed closed."""


class LocalRanStartError(LocalRanExecutorError):
    """A start stage failed; acquired resources were still cleaned."""

    def __init__(self, primary: BaseException, cleanup: "CleanupReportV1") -> None:
        self.primary = primary
        self.cleanup = cleanup
        suffix = "" if not cleanup.errors else f"; cleanup errors={list(cleanup.errors)!r}"
        super().__init__(f"{type(primary).__name__}: {primary}{suffix}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalRanExecutorError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_create_only(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


@dataclass(frozen=True)
class CommandResultV1:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class SpawnedProcessV1:
    owned: L.OwnedProcess
    process: Any  # the direct process or the sudo wrapper owning the session
    log_handle: Any
    plan: L.ProcessPlan
    root_owned: bool = False


ROOT_SOFTMODEM_ROLES = frozenset({"gnb", "ue"})


class LocalRanOps:
    """Production OS adapter; tests replace it with a deterministic fake."""

    root_discovery_timeout_s = 5.0
    process_attestation_timeout_s = 1.0
    process_attestation_poll_s = 0.025
    process_attestation_consecutive_matches = 2

    def run(self, argv: Sequence[str], *, timeout_s: float) -> CommandResultV1:
        completed = subprocess.run(
            list(argv), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(L.ROOT),
            timeout=timeout_s, check=False,
        )
        return CommandResultV1(completed.returncode, completed.stdout, completed.stderr)

    def spawn(self, plan: L.ProcessPlan, *, log_path: Path) -> SpawnedProcessV1:
        plan.validate()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        root_owned = plan.role in ROOT_SOFTMODEM_ROLES
        cwd = Path(plan.argv[0]).parent
        if root_owned:
            # The softmodems write nr*_stats.log into their working directory
            # (the UE AssertFatal-s if it cannot), and the OAI tree may be a
            # read-only bind mount.  Run them from an attempt-owned directory.
            cwd = log_path.parent / f"{plan.role}_workdir"
            cwd.mkdir(exist_ok=False)
        handle = log_path.open("x", encoding="utf-8")
        launch_argv = self._launch_argv(plan, root_owned=root_owned)
        started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        process: Any = None
        wrapper_pgid: Optional[int] = None
        try:
            process = self._launch(
                launch_argv, cwd=cwd,
                log_handle=handle, new_session=plan.new_session)
            wrapper_pgid = int(os.getpgid(process.pid))
            if root_owned:
                pid, pgid = self._discover_root_softmodem(plan, wrapper_pgid)
            else:
                pid, pgid = int(process.pid), wrapper_pgid
            executable = str(Path(plan.argv[0]).resolve(strict=True))
            owned = L.OwnedProcess(
                role=plan.role, pid=pid, pgid=pgid,
                executable=executable, argv_sha256=plan.argv_sha256,
                started_monotonic_raw_ns=started,
            ).validate()
            spawned = SpawnedProcessV1(
                owned, process, handle, plan, root_owned=root_owned)
            self._attest_stable_spawn_identity(owned)
            return spawned
        except BaseException as primary:
            cleanup_error: Optional[BaseException] = None
            if process is not None and wrapper_pgid is not None:
                try:
                    self._abort_unattested_group(
                        process, wrapper_pgid, root_owned=root_owned)
                except BaseException as exc:
                    cleanup_error = exc
            handle.close()
            if cleanup_error is not None:
                raise LocalRanExecutorError(
                    f"spawn failed ({primary}); unattested group cleanup failed: "
                    f"{cleanup_error}") from primary
            raise

    @staticmethod
    def _launch_argv(plan: L.ProcessPlan, *, root_owned: bool) -> tuple[str, ...]:
        if not root_owned:
            return plan.argv
        if plan.role == "gnb":
            env_args: list[str] = []
            for name in plan.env_unset:
                env_args.extend(("-u", name))
            env_args.extend(f"{name}={value}" for name, value in plan.env_set)
            return ("sudo", "-n", "env", *env_args, *plan.argv)
        _require(not plan.env_set and not plan.env_unset,
                 "root UE plan unexpectedly changes environment")
        return ("sudo", "-n", *plan.argv)

    def _launch(self, argv: Sequence[str], *, cwd: Path,
                log_handle: Any, new_session: bool) -> Any:
        return subprocess.Popen(
            list(argv), cwd=str(cwd), env=os.environ.copy(), stdout=log_handle,
            stderr=subprocess.STDOUT, text=True, start_new_session=new_session,
        )

    def _group_pids(self, pgid: int, *, root_owned: bool) -> list[int]:
        prefix = ["sudo", "-n"] if root_owned else []
        completed = subprocess.run(
            [*prefix, "ps", "-eo", "pid=,pgid=,stat="], text=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        _require(completed.returncode == 0,
                 f"process-group query failed: {completed.stderr[-500:]!r}")
        members: list[int] = []
        for line in completed.stdout.splitlines():
            fields = line.split()
            if len(fields) != 3:
                continue
            try:
                pid, observed_group = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if observed_group == pgid and "Z" not in fields[2]:
                members.append(pid)
        return sorted(set(members))

    def _process_identity(self, pid: int, *, root_owned: bool
                          ) -> tuple[str, tuple[str, ...], int]:
        prefix = ["sudo", "-n"] if root_owned else []
        executable = subprocess.run(
            [*prefix, "readlink", "-f", f"/proc/{pid}/exe"], text=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        _require(executable.returncode == 0 and executable.stdout.strip(),
                 f"cannot resolve process executable for PID {pid}")
        cmdline = subprocess.run(
            [*prefix, "cat", f"/proc/{pid}/cmdline"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        _require(cmdline.returncode == 0, f"cannot read argv for PID {pid}")
        raw = cmdline.stdout.rstrip(b"\0")
        argv = tuple(part.decode("utf-8") for part in raw.split(b"\0") if part)
        group = subprocess.run(
            [*prefix, "ps", "-o", "pgid=", "-p", str(pid)], text=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
        _require(group.returncode == 0 and group.stdout.strip().isdigit(),
                 f"cannot read PGID for PID {pid}")
        return executable.stdout.strip(), argv, int(group.stdout.strip())

    def _root_softmodem_rows(self, wrapper_pgid: int
                             ) -> tuple[Mapping[str, Any], ...]:
        from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as A

        return tuple(
            row for row in A.process_table()
            if int(row["process_group_id"]) == wrapper_pgid
        )

    def _discover_root_softmodem(self, plan: L.ProcessPlan,
                                 wrapper_pgid: int) -> tuple[int, int]:
        from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as A

        expected_executable = Path(plan.argv[0]).resolve(strict=True)
        deadline = time.monotonic() + self.root_discovery_timeout_s
        last_error = "topology was not yet observable"
        while time.monotonic() < deadline:
            try:
                selected = A.select_softmodem_process(
                    self._root_softmodem_rows(wrapper_pgid),
                    command_name=expected_executable.name,
                    expected_executable=expected_executable,
                )
            except A.Phase14AError as exc:
                last_error = str(exc)
                time.sleep(0.05)
                continue
            pid = int(selected["pid"])
            executable, argv, pgid = self._process_identity(
                pid, root_owned=True)
            _require(executable == str(expected_executable),
                     f"root {plan.role} executable drift: {executable}")
            _require(argv == plan.argv,
                     f"root {plan.role} argv drift for PID {pid}")
            _require(pgid == wrapper_pgid,
                     f"root {plan.role} PGID drift for PID {pid}: {pgid}")
            return pid, pgid
        raise LocalRanExecutorError(
            f"root {plan.role} topology was not attested in PGID "
            f"{wrapper_pgid}: {last_error}")

    def _abort_unattested_group(self, process: Any, pgid: int,
                                *, root_owned: bool) -> None:
        """Remove only the fresh session created by a failed spawn attempt."""
        members = self._group_pids(pgid, root_owned=root_owned)
        if members:
            self._signal_members(
                members, signal.SIGKILL, root_owned=root_owned, pgid=pgid)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if not self._group_pids(pgid, root_owned=root_owned):
                break
            time.sleep(0.05)
        _require(not self._group_pids(pgid, root_owned=root_owned),
                 "unattested startup process group survived cleanup")
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass

    def process_alive(self, process: SpawnedProcessV1) -> bool:
        try:
            if process.owned.pid not in self._group_pids(
                    process.owned.pgid, root_owned=process.root_owned):
                return False
            L.validate_observed_process(
                process.owned, self.observe_process(process.owned))
            return True
        except BaseException:
            return False

    def observe_process(self, owned: L.OwnedProcess) -> Mapping[str, Any]:
        executable, argv, pgid = self._process_identity(
            owned.pid, root_owned=owned.role in ROOT_SOFTMODEM_ROLES)
        return {
            "pid": owned.pid,
            "pgid": pgid,
            "executable": executable,
            "argv_sha256": L.sha256_json(list(argv)),
            "host": "W10275",
        }

    def _attest_stable_spawn_identity(self, owned: L.OwnedProcess) -> None:
        """Require two consecutive exact identities during process acquisition."""
        deadline = time.monotonic() + self.process_attestation_timeout_s
        consecutive = 0
        last_error: Optional[BaseException] = None
        while True:
            try:
                L.validate_observed_process(
                    owned, self.observe_process(owned))
                consecutive += 1
                if consecutive >= self.process_attestation_consecutive_matches:
                    return
            except (LocalRanExecutorError, L.LocalRanLifecycleError) as exc:
                consecutive = 0
                last_error = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self.process_attestation_poll_s, remaining))
        raise LocalRanExecutorError(
            f"stable process identity not attested: {owned.role}: {last_error}")

    def _signal_members(self, members: Sequence[int], sig: signal.Signals,
                        *, root_owned: bool, pgid: int) -> None:
        if root_owned:
            completed = subprocess.run(
                ["sudo", "-n", "kill", f"-{sig.name.removeprefix('SIG')}",
                 "--", *map(str, members)], stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True, check=False)
            _require(completed.returncode == 0,
                     f"root group signal failed: {completed.stderr[-500:]!r}")
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return

    def stop_process(self, process: SpawnedProcessV1) -> None:
        try:
            initial_members = self._group_pids(
                process.owned.pgid, root_owned=process.root_owned)
            if not initial_members:
                return
            _require(process.owned.pid in initial_members,
                     f"owned {process.owned.role} PID vanished while its PGID remains")
            L.validate_observed_process(
                process.owned, self.observe_process(process.owned))
            for sig, timeout_s in ((signal.SIGINT, 4.0),
                                   (signal.SIGTERM, 3.0),
                                   (signal.SIGKILL, 3.0)):
                members = self._group_pids(
                    process.owned.pgid, root_owned=process.root_owned)
                if not members:
                    break
                self._signal_members(members, sig, root_owned=process.root_owned,
                                     pgid=process.owned.pgid)
                deadline = time.monotonic() + timeout_s
                while time.monotonic() < deadline:
                    if not self._group_pids(
                            process.owned.pgid,
                            root_owned=process.root_owned):
                        break
                    time.sleep(0.1)
            if self._group_pids(process.owned.pgid,
                                root_owned=process.root_owned):
                raise LocalRanExecutorError(
                    f"owned {process.owned.role} PGID survived cleanup")
            try:
                process.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            process.log_handle.close()

    def wait_tcp(self, host: str, port: int, *, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.25):
                    return
            except OSError:
                time.sleep(0.1)
        raise LocalRanExecutorError(f"TCP endpoint did not become ready: {host}:{port}")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def restore_channel(self) -> Mapping[str, Any]:
        from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as R
        from rl_agent import ue_n2_oai_ul_calibration_smoke as N

        config_path = L.ROOT / "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        actuator = config["actuator"]
        session = N.TelnetSession(
            str(actuator["telnet_host"]), int(actuator["telnet_port"]),
            float(actuator["response_timeout_s"]),
            int(actuator["max_response_bytes"]),
        )
        try:
            before = session.command("channelmod show current")[-1]
        finally:
            session.close()
        try:
            model = N.parse_channel_models(before).get(
                str(actuator["channel_model_name"]))
        except N.SmokeFailure as exc:
            raise LocalRanExecutorError(
                f"registered RFsim state is malformed: {exc}") from exc
        _require(type(model) is dict, "registered RFsim channel is absent")
        _require(model.get("model_type") == actuator["channel_model_type"],
                 f"registered RFsim channel type drift: {model}")
        path_loss = float(model.get("path_loss_db", math.nan))
        noise = float(model.get("noise_power_db", math.nan))
        _require(math.isfinite(path_loss) and math.isfinite(noise),
                 f"registered RFsim channel has non-finite state: {model}")
        _require(math.isclose(path_loss, float(actuator["path_loss_db"]),
                              abs_tol=1e-6),
                 f"registered RFsim path-loss drift: {model}")
        clean = float(actuator["clean_restore_noise_power_db"])
        if math.isclose(noise, clean, abs_tol=1e-6):
            return {
                "noise_power_db": noise,
                "before_sha256": _sha256_text(before),
                "verified": True,
                "status": "ALREADY_CLEAN_NO_MUTATION",
            }

        result = R.restore_interrupted_radio(config)
        _require(type(result) is dict and result.get("verified") is True,
                 "registered RFsim authority did not verify restoration")
        _require(float(result.get("noise_power_db")) == clean,
                 "registered RFsim authority restored the wrong value")
        return result


@dataclass(frozen=True)
class CleanupReportV1:
    schema: str
    channel_restore_attempted: bool
    channel_restored: bool
    stopped_roles: tuple[str, ...]
    policy_removed: tuple[str, ...]
    tunnel_removed: bool
    errors: tuple[str, ...]
    tunnel_absent_after_cleanup: bool
    remote_cn_touched: bool = False

    @property
    def ok(self) -> bool:
        return not self.errors and not self.remote_cn_touched


@dataclass
class LocalRanSessionV1:
    executor: "LocalRanExecutorV1"
    plan: L.LocalRanPlan
    processes: dict[str, SpawnedProcessV1]
    tunnel: Optional[L.TunnelOwnership]
    policy_ownership: Optional[S.PolicyRoutingOwnershipV1]
    route_evidence: tuple[Mapping[str, Any], ...]
    source_route: Optional[Mapping[str, Any]]
    attached: Optional[Mapping[str, Any]]
    cleanup_report: Optional[CleanupReportV1] = None

    def close(self) -> CleanupReportV1:
        if self.cleanup_report is None:
            self.cleanup_report = self.executor._cleanup(self)
        return self.cleanup_report

    def __enter__(self) -> "LocalRanSessionV1":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


@dataclass(frozen=True)
class LocalRanExecutorConfigV1:
    command_timeout_s: float = 30.0
    gnb_lead_s: float = 2.0
    attach_timeout_s: float = 45.0
    attach_poll_s: float = 0.25
    tracer_ready_timeout_s: float = 10.0

    def validate(self) -> "LocalRanExecutorConfigV1":
        for name, value in self.__dict__.items():
            _require(type(value) in (int, float) and float(value) > 0.0,
                     f"invalid executor timeout: {name}")
        return self


class LocalRanExecutorV1:
    def __init__(self, *, plan: L.LocalRanPlan, attempt_dir: Path,
                 ops: Optional[LocalRanOps] = None,
                 config: LocalRanExecutorConfigV1 = LocalRanExecutorConfigV1()) -> None:
        self.plan = plan.validate()
        self.attempt_dir = Path(attempt_dir).resolve()
        self.ops = ops if ops is not None else LocalRanOps()
        self.config = config.validate()

    def _checked(self, argv: Sequence[str], *, label: str) -> CommandResultV1:
        rendered = " ".join(argv).lower()
        _require("docker" not in rendered and "ssh" not in rendered,
                 f"forbidden remote operation in {label}")
        result = self.ops.run(argv, timeout_s=self.config.command_timeout_s)
        _require(type(result.returncode) is int and result.returncode == 0,
                 f"{label} failed: rc={result.returncode} stderr={result.stderr[-500:]!r}")
        return result

    def _capture_json(self, argv: Sequence[str], *, label: str) -> tuple[str, list[Any]]:
        result = self._checked(argv, label=label)
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LocalRanExecutorError(f"{label} did not return JSON: {exc}") from exc
        _require(type(value) is list, f"{label} must return a JSON list")
        return result.stdout, value

    @staticmethod
    def _rule_priority(row: Mapping[str, Any]) -> Optional[int]:
        raw = row.get("priority", row.get("pref"))
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _route_destination(row: Mapping[str, Any]) -> str:
        value = str(row.get("dst") or "")
        return value if "/" in value else (f"{value}/32" if value else "")

    def _snapshot_policy_before(self) -> S.PolicyRoutingOwnershipV1:
        rules_text, rules = self._capture_json(
            ("ip", "-j", "rule", "show"), label="policy-rule snapshot")
        table_text, routes = self._capture_json(
            ("ip", "-j", "route", "show", "table", str(S.POLICY_TABLE)),
            label="table-9999 snapshot")
        _require(not any(type(row) is dict and self._rule_priority(row) == POLICY_PRIORITY
                         for row in rules),
                 f"policy priority {POLICY_PRIORITY} was not absent before start")
        destination = f"{C.default_topology().edge_ip}/32"
        _require(not any(type(row) is dict and self._route_destination(row) == destination
                         for row in routes),
                 f"policy route {destination} was not absent before start")
        _write_create_only(self.attempt_dir / RULES_BEFORE, rules_text)
        _write_create_only(self.attempt_dir / TABLE_BEFORE, table_text)
        # Ownership is populated only after successful mutations.
        return S.PolicyRoutingOwnershipV1(
            schema=S.POLICY_OWNERSHIP_SCHEMA,
            before_rules_sha256=_sha256_text(rules_text),
            before_table_sha256=_sha256_text(table_text),
            rules=(), routes=(),
        )

    def _spawn(self, plan: L.ProcessPlan, processes: dict[str, SpawnedProcessV1]) -> None:
        _require(plan.role not in processes, f"duplicate process role: {plan.role}")
        spawned = self.ops.spawn(plan, log_path=self.attempt_dir / "logs" / plan.stdout_name)
        _require(spawned.owned.role == plan.role, "spawned process role drift")
        processes[plan.role] = spawned
        _require(self.ops.process_alive(spawned), f"{plan.role} exited at startup")

    def _wait_attach(self, ue: SpawnedProcessV1,
                     gnb: SpawnedProcessV1) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.config.attach_timeout_s
        last_error = "UE tunnel did not appear"
        while time.monotonic() < deadline:
            _require(self.ops.process_alive(gnb), "gNB exited before UE attachment")
            _require(self.ops.process_alive(ue), "UE exited before attachment")
            result = self.ops.run(
                ("ip", "-j", "-4", "addr", "show", "dev", L.UE_INTERFACE),
                timeout_s=self.config.command_timeout_s)
            if result.returncode == 0:
                ping = self.ops.run(self.plan.attach_probe.argv,
                                    timeout_s=self.config.command_timeout_s)
                try:
                    return L.validate_tunnel_attachment(result.stdout, ping.returncode)
                except L.LocalRanLifecycleError as exc:
                    last_error = str(exc)
            self.ops.sleep(self.config.attach_poll_s)
        raise LocalRanExecutorError(last_error)

    def _install_policy(self, base: S.PolicyRoutingOwnershipV1
                        ) -> S.PolicyRoutingOwnershipV1:
        destination = f"{C.default_topology().edge_ip}/32"
        route = S.OwnedPolicyRouteV1(destination, True).validate()
        rule = S.OwnedPolicyRuleV1(POLICY_PRIORITY, True).validate()
        added_route = False
        try:
            self._checked(("sudo", "-n", "ip", "route", "add", destination,
                           "dev", S.POLICY_INTERFACE, "src", S.UE_CONTROL_HOST,
                           "table", str(S.POLICY_TABLE)), label="add owned UE route")
            added_route = True
            self._checked(("sudo", "-n", "ip", "rule", "add", "priority",
                           str(POLICY_PRIORITY), "from", S.UE_CONTROL_HOST,
                           "lookup", str(S.POLICY_TABLE)), label="add owned UE rule")
        except BaseException as exc:
            # Convey partial ownership to the caller without broad cleanup.
            partial = S.PolicyRoutingOwnershipV1(
                schema=base.schema, before_rules_sha256=base.before_rules_sha256,
                before_table_sha256=base.before_table_sha256, rules=(),
                routes=(route,) if added_route else (),
            )
            setattr(exc, "split_host_partial_policy_ownership", partial)
            raise
        return S.PolicyRoutingOwnershipV1(
            schema=base.schema, before_rules_sha256=base.before_rules_sha256,
            before_table_sha256=base.before_table_sha256,
            rules=(rule,), routes=(route,),
        ).validate()

    def start(self) -> LocalRanSessionV1:
        _require(not self.attempt_dir.exists(),
                 f"create-only attempt directory exists: {self.attempt_dir}")
        self.attempt_dir.mkdir(parents=True)
        processes: dict[str, SpawnedProcessV1] = {}
        session = LocalRanSessionV1(
            executor=self, plan=self.plan, processes=processes, tunnel=None,
            policy_ownership=None, route_evidence=(), source_route=None,
            attached=None,
        )
        try:
            self._checked(("sudo", "-n", "true"),
                          label="noninteractive privilege preflight")
            # A pre-existing UE interface can never become attempt-owned.
            before_tunnel = self.ops.run(
                ("ip", "-j", "link", "show", "dev", L.UE_INTERFACE),
                timeout_s=self.config.command_timeout_s)
            _require(before_tunnel.returncode != 0,
                     f"pre-existing {L.UE_INTERFACE} cannot become attempt-owned")

            self._checked(self.plan.materialize.argv, label="Phase-14a materialization")
            self._checked(self.plan.derive_runtime.argv, label="split-host derivation")
            L.load_attestation(Path(self.plan.state_dir))

            routes: list[Mapping[str, Any]] = []
            for probe in self.plan.preflight:
                route = self._checked(probe.route_argv, label=f"route to {probe.destination}")
                ping = self.ops.run(probe.reachability_argv,
                                    timeout_s=self.config.command_timeout_s)
                routes.append(L.validate_route_evidence(
                    probe.destination, route.stdout, ping.returncode))
            session.route_evidence = tuple(routes)
            base_policy = self._snapshot_policy_before()

            self._spawn(self.plan.gnb, processes)
            self.ops.sleep(self.config.gnb_lead_s)
            _require(self.ops.process_alive(processes["gnb"]),
                     "gNB exited during startup lead")
            self._spawn(self.plan.ue, processes)
            tunnel_evidence = self._wait_attach(processes["ue"], processes["gnb"])
            session.tunnel = L.TunnelOwnership(
                interface=L.UE_INTERFACE, absent_before_start=True,
                observed_ipv4_after_attach=L.UE_IP, created_by_local_ue=True,
            ).validate()

            try:
                session.policy_ownership = self._install_policy(base_policy)
            except BaseException as exc:
                session.policy_ownership = getattr(
                    exc, "split_host_partial_policy_ownership", None)
                raise
            source = self._checked(S.source_route_probe_argv(),
                                   label="UE source-route proof")
            session.source_route = S.validate_ue_source_route(source.stdout)

            self._spawn(self.plan.tracer_multi, processes)
            self.ops.wait_tcp("127.0.0.1", L.UE_RELAY_PORT,
                              timeout_s=self.config.tracer_ready_timeout_s)
            _require(self.ops.process_alive(processes["tracer_multi"]),
                     "UE tracer relay exited before readiness")
            self._spawn(self.plan.tracer_record, processes)

            owned = [processes[name].owned for name in
                     ("gnb", "ue", "tracer_multi", "tracer_record")]
            attached = L.attached_attestation(
                state_dir=Path(self.plan.state_dir), route_evidence=routes,
                tunnel_evidence=tunnel_evidence, processes=owned)
            _write_create_only(
                self.attempt_dir / ATTACHED_RECORD,
                json.dumps(attached, sort_keys=True, indent=2) + "\n")
            session.attached = attached
            return session
        except BaseException as primary:
            cleanup = self._cleanup(session)
            session.cleanup_report = cleanup
            raise LocalRanStartError(primary, cleanup) from primary

    def _remove_policy(self, ownership: Optional[S.PolicyRoutingOwnershipV1],
                       errors: list[str], removed: list[str]) -> None:
        if ownership is None:
            return
        # Partial ownership is valid during rollback even though the public
        # contract requires at least one entry; never synthesize ownership.
        commands = ownership.cleanup_commands
        labels = ([f"rule:{row.priority}" for row in reversed(ownership.rules)]
                  + [f"route:{row.destination}" for row in reversed(ownership.routes)])
        for argv, label in zip(commands, labels):
            try:
                _require(argv[:1] == ("sudo",),
                         "owned policy cleanup lost its sudo boundary")
                command = ("sudo", "-n", *argv[1:])
                result = self.ops.run(command, timeout_s=self.config.command_timeout_s)
                _require(result.returncode == 0,
                         f"remove {label} failed: rc={result.returncode}")
                removed.append(label)
            except BaseException as exc:
                errors.append(f"policy {label}: {type(exc).__name__}: {exc}")

    def _cleanup(self, session: LocalRanSessionV1) -> CleanupReportV1:
        errors: list[str] = []
        stopped: list[str] = []
        removed: list[str] = []
        restore_attempted = "gnb" in session.processes
        restored = False
        if restore_attempted:
            try:
                result = self.ops.restore_channel()
                _require(result.get("verified") is True
                         and float(result.get("noise_power_db")) == -50.0,
                         "RFsim restoration read-back did not verify -50 dB")
                restored = True
            except BaseException as exc:
                errors.append(f"channel restore: {type(exc).__name__}: {exc}")

        # Relative process order remains record -> relay -> UE -> gNB.  The
        # two policy entries are removed between tracer and UE shutdown while
        # their exact tunnel device still exists.
        for role in ("tracer_record", "tracer_multi"):
            process = session.processes.get(role)
            if process is not None:
                try:
                    self.ops.stop_process(process)
                    stopped.append(role)
                except BaseException as exc:
                    errors.append(f"stop {role}: {type(exc).__name__}: {exc}")
        self._remove_policy(session.policy_ownership, errors, removed)
        for role in ("ue", "gnb"):
            process = session.processes.get(role)
            if process is not None:
                try:
                    self.ops.stop_process(process)
                    stopped.append(role)
                except BaseException as exc:
                    errors.append(f"stop {role}: {type(exc).__name__}: {exc}")

        tunnel_removed = False
        tunnel_absent = session.tunnel is None
        if session.tunnel is not None:
            try:
                observed = self.ops.run(
                    ("ip", "-j", "-4", "addr", "show", "dev", L.UE_INTERFACE),
                    timeout_s=self.config.command_timeout_s)
                if observed.returncode == 0:
                    L.validate_tunnel_attachment(observed.stdout, 0)
                    removed_tunnel = self.ops.run(
                        ("sudo", "-n", "ip", "link", "delete", "dev", L.UE_INTERFACE),
                        timeout_s=self.config.command_timeout_s)
                    _require(removed_tunnel.returncode == 0,
                             "owned UE tunnel deletion failed")
                    tunnel_removed = True
                final_tunnel = self.ops.run(
                    ("ip", "-j", "link", "show", "dev", L.UE_INTERFACE),
                    timeout_s=self.config.command_timeout_s)
                _require(final_tunnel.returncode != 0,
                         "owned UE tunnel remains after cleanup")
                tunnel_absent = True
            except BaseException as exc:
                errors.append(f"tunnel cleanup: {type(exc).__name__}: {exc}")
        return CleanupReportV1(
            schema=SCHEMA, channel_restore_attempted=restore_attempted,
            channel_restored=restored, stopped_roles=tuple(stopped),
            policy_removed=tuple(removed), tunnel_removed=tunnel_removed,
            tunnel_absent_after_cleanup=tunnel_absent,
            errors=tuple(errors), remote_cn_touched=False,
        )
