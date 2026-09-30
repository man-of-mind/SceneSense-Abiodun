"""Local RAN lifecycle contract for the W10275/L10319 split-host run.

This module is additive and import-pure.  It does not start OAI, Docker,
CARLA, CUDA, a tracer, or a network service.  It provides three bounded
pieces for a later reviewed executor:

* derive a create-only, runtime gNB config from the qualified Phase-14a
  materialization while leaving the Phase-14a files byte-identical;
* build exact local gNB/UE/tracer command plans and validate independently
  captured routed-connectivity evidence; and
* represent process/tunnel ownership so teardown can affect only resources
  created on W10275 by this lifecycle.

The remote CN and edge are prerequisites, never children of this lifecycle.
Consequently neither the start nor cleanup plans contain Docker, SSH, CN
compose, or any L10319 process operation.  The policy deadline remains a
single-clock W10275 measurement; remote monotonic timestamps are never
compared with it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from . import contract as split_contract


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "scenesense.run4.split_host_local_ran.v1"
ATTESTATION_SCHEMA = "scenesense.run4.split_host_local_ran_attestation.v1"
ATTACHED_SCHEMA = "scenesense.run4.split_host_local_ran_attached.v1"
RADIO_PROFILE_ID = "OAI_N78_100MHZ_273PRB_4D5U_V1"
RUNTIME_GNB_NAME = "effective_gnb_100mhz_4d5u_split_host.conf"
ATTESTATION_NAME = "SPLIT_HOST_LOCAL_RAN_ATTESTATION.json"
PYTHON = "/usr/bin/python3"
UE_INTERFACE = "oaitun_ue1"
UE_IP = "10.0.0.2"
GNB_T_PORT = 2021
UE_T_PORT = 2023
UE_RELAY_PORT = 2123
UE_EVENTS = (
    "NRUE_MAC_DCI_GRANT",
    "NRUE_MAC_RLC_BUFFER_STATUS",
    "NR_PDCP_TX_SDU",
    "NR_RLC_TX_SDU",
    "NR_RLC_TX_DEQUEUE",
)


class LocalRanLifecycleError(RuntimeError):
    """A split-host local-RAN input or observed fact failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalRanLifecycleError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalRanLifecycleError(f"{label} is unreadable: {exc}") from exc
    _require(type(value) is dict, f"{label} is not a JSON object")
    return value


@dataclass(frozen=True)
class SourcePin:
    name: str
    relative_path: str
    sha256: str


SOURCE_PINS = (
    SourcePin(
        "phase14a_launcher",
        "uplink_only_spatial_map_pipeline/run_splitfusion_oai_100mhz_4d5u_v1.sh",
        "8e02f0913338a187bff1de24bfea09d7eca03607da0e383ef8ad4a005b64ce61",
    ),
    SourcePin(
        "phase14a_runner",
        "rl_agent/splitfusion_phase14a_100mhz_calibration_v1.py",
        "09fb82c0ba644a44bdf5fb5e5ad92269c1daa31ffbeee1fc0864d5b7ce688c93",
    ),
    SourcePin(
        "phase14a_config",
        "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json",
        "ad541f71f5659e1bb08d7d2c48a45086dfbd5bf45e669b32adfb320ddcab5cd9",
    ),
    SourcePin(
        "phase14a_binding",
        "rl_agent/configs/splitfusion_phase14a_campaign_binding_v1.json",
        "103aeda31a37594c89e820daf1794d28af0440f5e4324450cac01266f5540004",
    ),
    SourcePin(
        "split_host_contract_json",
        "rl_agent/splitfusion_run4_split_host_l10319_v1/SPLIT_HOST_CONTRACT_V1.json",
        "8100883e29a89d55fbfdfb43ec619580ed3ed0f777dabd1b5f1b4818a3f4866c",
    ),
    SourcePin(
        "split_host_contract_python",
        "rl_agent/splitfusion_run4_split_host_l10319_v1/contract.py",
        "8ed42f73f7fa2cfc47bb5cb28291e1a7b2674b0ee770044abba2b2404fdce85a",
    ),
    SourcePin(
        "nr_softmodem",
        "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-softmodem",
        "ebcd85f4c96cf6e3d0a1752d0047e8377811014be75cdeeef6334c982fcfab70",
    ),
    SourcePin(
        "nr_uesoftmodem",
        "OAI/openairinterface5g/cmake_targets/ran_build/build/nr-uesoftmodem",
        "60ecc9a1d102e8b66871727a6a23da22977ff907fecc3870a8dc46414080c975",
    ),
    SourcePin(
        "libtelnetsrv",
        "OAI/openairinterface5g/cmake_targets/ran_build/build/libtelnetsrv.so",
        "7c815fbfbd987b256c1992666dff142d683b5a548ba1c6f9a5a39f6fede118f7",
    ),
    SourcePin(
        "tracer_multi",
        "OAI/openairinterface5g/common/utils/T/tracer/multi",
        "455ff3083dfb30f82ffee941dac01f888b39705f9d3ffab2b019ff5430da9542",
    ),
    SourcePin(
        "tracer_record",
        "OAI/openairinterface5g/common/utils/T/tracer/record",
        "517150245a1317c2e117c0ac3adde1ee097a3592658316fe5b40edcbd5235e61",
    ),
    SourcePin(
        "t_messages",
        "OAI/openairinterface5g/common/utils/T/T_messages.txt",
        "2f4945814ad2f47197816756819afc0d087156c499e8c746cf0d9a81653005c5",
    ),
)


def verify_source_pins(root: Path = ROOT, *,
                       pins: Sequence[SourcePin] = SOURCE_PINS) -> dict[str, str]:
    """Hash every inherited authority before any runtime file is written."""
    root = Path(root).resolve(strict=True)
    verified: dict[str, str] = {}
    for pin in pins:
        path = root / pin.relative_path
        _require(path.is_file(), f"source authority is missing: {pin.name}")
        observed = sha256_file(path)
        _require(observed == pin.sha256, f"source authority drift: {pin.name}")
        verified[pin.name] = observed
    return verified


_MATERIALIZATION_FIELDS = {
    "schema", "status", "radio_profile_id", "source_gnb_sha256",
    "selected_gnb_before_channelmod_sha256", "source_ue_sha256",
    "source_channel_sha256", "effective_gnb_path", "effective_gnb_sha256",
    "effective_ue_path", "effective_ue_sha256", "clean_noise_power_db",
    "cpu_reconciliation",
}


def _within(parent: Path, child: Path, *, label: str) -> Path:
    parent = Path(parent).resolve(strict=True)
    child = Path(child).resolve(strict=True)
    try:
        child.relative_to(parent)
    except ValueError as exc:
        raise LocalRanLifecycleError(f"{label} escaped radio state") from exc
    _require(child.parent == parent, f"{label} must be directly inside radio state")
    return child


def _verify_materialization(state_dir: Path) -> tuple[dict[str, Any], Path, Path, str]:
    state = Path(state_dir).resolve(strict=True)
    path = state / "radio_materialization.json"
    document = _load_object(path, label="Phase-14a radio materialization")
    _require(set(document) == _MATERIALIZATION_FIELDS,
             "Phase-14a materialization fields are incomplete or foreign")
    _require(document["schema"] == "scenesense.splitfusion_phase14a_radio_materialization.v1",
             "Phase-14a materialization schema drift")
    _require(document["status"] == "MATERIALIZED_NOT_LAUNCHED",
             "Phase-14a materialization is not pre-launch")
    _require(document["radio_profile_id"] == RADIO_PROFILE_ID,
             "Phase-14a radio profile drift")
    _require(float(document["clean_noise_power_db"]) == -50.0,
             "Phase-14a clean channel drift")
    _require(document["cpu_reconciliation"] == "PHASE14A_CPU_RECONCILIATION_PASSED",
             "Phase-14a CPU reconciliation drift")
    gnb = _within(state, Path(str(document["effective_gnb_path"])), label="effective gNB")
    ue = _within(state, Path(str(document["effective_ue_path"])), label="effective UE")
    _require(sha256_file(gnb) == document["effective_gnb_sha256"],
             "effective Phase-14a gNB hash drift")
    _require(sha256_file(ue) == document["effective_ue_sha256"],
             "effective Phase-14a UE hash drift")
    return document, gnb, ue, sha256_file(path)


def prepare_runtime_gnb(
        state_dir: Path, *, root: Path = ROOT,
        _source_verifier: Callable[[Path], dict[str, str]] = verify_source_pins,
) -> dict[str, Any]:
    """Create and attest the split-host runtime copy of a Phase-14a gNB config.

    The original ``radio_materialization.json`` and both files it names remain
    unchanged.  The new attestation explicitly says the original Phase-14a
    same-file attached-state assertion is *not* being claimed.
    """
    sources = _source_verifier(Path(root))
    state = Path(state_dir).resolve(strict=True)
    materialization, source_gnb, source_ue, materialization_sha = (
        _verify_materialization(state)
    )
    before = {
        "materialization": materialization_sha,
        "gnb": sha256_file(source_gnb),
        "ue": sha256_file(source_ue),
    }
    runtime_path = state / RUNTIME_GNB_NAME
    attestation_path = state / ATTESTATION_NAME
    _require(not runtime_path.exists(), "split-host runtime gNB already exists")
    _require(not attestation_path.exists(), "split-host attestation already exists")

    rewritten = split_contract.rewrite_runtime_gnb_config(
        source_gnb.read_text(encoding="utf-8")
    )
    with runtime_path.open("x", encoding="utf-8") as handle:
        handle.write(rewritten)
    runtime_sha = sha256_file(runtime_path)

    after = {
        "materialization": sha256_file(state / "radio_materialization.json"),
        "gnb": sha256_file(source_gnb),
        "ue": sha256_file(source_ue),
    }
    _require(after == before, "Phase-14a materialization changed during derivation")
    topology = split_contract.default_topology()
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "status": "RUNTIME_GNB_DERIVED_NOT_LAUNCHED",
        "radio_profile_id": RADIO_PROFILE_ID,
        "source_pins": sources,
        "phase14a_materialization_path": str(state / "radio_materialization.json"),
        "phase14a_materialization_sha256": materialization_sha,
        "phase14a_effective_gnb_path": str(source_gnb),
        "phase14a_effective_gnb_sha256": materialization["effective_gnb_sha256"],
        "phase14a_effective_ue_path": str(source_ue),
        "phase14a_effective_ue_sha256": materialization["effective_ue_sha256"],
        "split_host_runtime_gnb_path": str(runtime_path),
        "split_host_runtime_gnb_sha256": runtime_sha,
        "rewrite": {
            "amf_ip_address": topology.amf_ip,
            "GNB_IPV4_ADDRESS_FOR_NG_AMF":
                f"{topology.local_lan_ip}/{topology.local_prefix_length}",
            "GNB_IPV4_ADDRESS_FOR_NGU":
                f"{topology.local_lan_ip}/{topology.local_prefix_length}",
        },
        "phase14a_files_preserved": before == after,
        "original_phase14a_attached_same_file_assertion_satisfied": False,
        "attached_state_authority": "SPLIT_HOST_LOCAL_RAN_ATTESTATION_ONLY",
        "remote_cn_lifecycle_owned": False,
        "policy_deadline_clock": "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
        "cross_host_monotonic_comparison_permitted": False,
        "live_run_authorized": False,
    }
    with attestation_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(attestation, sort_keys=True, indent=2) + "\n")
    return {
        **attestation,
        "attestation_path": str(attestation_path),
        "attestation_sha256": sha256_file(attestation_path),
    }


def load_attestation(
        state_dir: Path, *,
        _source_verifier: Callable[[Path], dict[str, str]] = verify_source_pins,
) -> dict[str, Any]:
    state = Path(state_dir).resolve(strict=True)
    value = _load_object(state / ATTESTATION_NAME, label="split-host local-RAN attestation")
    observed_sources = _source_verifier(ROOT)
    _require(value.get("source_pins") == observed_sources,
             "attested source authorities changed before consumption")
    _require(value.get("schema") == ATTESTATION_SCHEMA, "split-host attestation schema drift")
    _require(value.get("status") == "RUNTIME_GNB_DERIVED_NOT_LAUNCHED",
             "split-host attestation status drift")
    _require(value.get("original_phase14a_attached_same_file_assertion_satisfied") is False,
             "split-host attestation falsely claims Phase-14a same-file attachment")
    _require(value.get("remote_cn_lifecycle_owned") is False,
             "local lifecycle must not own the remote CN")
    _require(value.get("cross_host_monotonic_comparison_permitted") is False,
             "cross-host monotonic comparison was enabled")
    runtime = _within(state, Path(str(value.get("split_host_runtime_gnb_path"))),
                      label="split-host runtime gNB")
    _require(sha256_file(runtime) == value.get("split_host_runtime_gnb_sha256"),
             "split-host runtime gNB hash drift")
    split_contract.assert_runtime_gnb_config(runtime.read_text(encoding="utf-8"))
    materialization, source_gnb, source_ue, materialization_sha = (
        _verify_materialization(state)
    )
    _require(materialization_sha == value.get("phase14a_materialization_sha256"),
             "Phase-14a materialization changed after attestation")
    _require(sha256_file(source_gnb) == value.get("phase14a_effective_gnb_sha256"),
             "Phase-14a gNB changed after attestation")
    _require(sha256_file(source_ue) == value.get("phase14a_effective_ue_sha256"),
             "Phase-14a UE changed after attestation")
    _require(materialization["radio_profile_id"] == RADIO_PROFILE_ID,
             "attested materialization radio drift")
    return value


@dataclass(frozen=True)
class ProcessPlan:
    role: str
    host: str
    argv: tuple[str, ...]
    env_set: tuple[tuple[str, str], ...] = ()
    env_unset: tuple[str, ...] = ()
    new_session: bool = True
    stdout_name: str = ""

    def validate(self) -> "ProcessPlan":
        _require(self.host == "W10275", f"{self.role} is not local")
        _require(bool(self.argv) and all(type(v) is str and v for v in self.argv),
                 f"{self.role} argv is invalid")
        rendered = " ".join(self.argv).lower()
        for forbidden in ("docker", "compose", "cn_start", "cn_stop", "ssh"):
            _require(forbidden not in rendered, f"{self.role} contains forbidden {forbidden}")
        return self

    @property
    def argv_sha256(self) -> str:
        return sha256_json(list(self.argv))


@dataclass(frozen=True)
class ConnectivityProbe:
    destination: str
    route_argv: tuple[str, ...]
    reachability_argv: tuple[str, ...]


@dataclass(frozen=True)
class LocalRanPlan:
    schema: str
    state_dir: str
    materialize: ProcessPlan
    derive_runtime: ProcessPlan
    preflight: tuple[ConnectivityProbe, ...]
    gnb: ProcessPlan
    ue: ProcessPlan
    tracer_multi: ProcessPlan
    tracer_record: ProcessPlan
    attach_probe: ProcessPlan
    remote_cn_prerequisite_only: bool
    policy_deadline_clock: str
    cross_host_monotonic_comparison_permitted: bool

    def validate(self) -> "LocalRanPlan":
        _require(self.schema == SCHEMA, "local-RAN plan schema drift")
        for process in (self.materialize, self.derive_runtime, self.gnb, self.ue,
                        self.tracer_multi, self.tracer_record, self.attach_probe):
            process.validate()
        _require(self.remote_cn_prerequisite_only, "remote CN became locally owned")
        _require(self.policy_deadline_clock == "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
                 "policy deadline left W10275's clock")
        _require(not self.cross_host_monotonic_comparison_permitted,
                 "cross-host monotonic comparison is forbidden")
        _require(len(self.preflight) == 2, "AMF/ext-DN preflight count drift")
        expected = {split_contract.default_topology().amf_ip,
                    split_contract.default_topology().ext_dn_ip}
        _require({probe.destination for probe in self.preflight} == expected,
                 "AMF/ext-DN preflight destinations drift")
        return self


def _module_name() -> str:
    return "rl_agent.splitfusion_run4_split_host_l10319_v1.local_ran_lifecycle_v1"


def build_local_ran_plan(state_dir: Path, *, root: Path = ROOT,
                         raw_path: Optional[Path] = None) -> LocalRanPlan:
    """Build exact command plans; constructing the plan performs no I/O."""
    topology = split_contract.default_topology()
    root = Path(root).resolve()
    state = Path(state_dir).resolve()
    runtime_gnb = state / RUNTIME_GNB_NAME
    effective_ue = state / "effective_ue_100mhz_clean_minus50.conf"
    ran = root / "OAI/openairinterface5g/cmake_targets/ran_build/build"
    tracer = root / "OAI/openairinterface5g/common/utils/T/tracer"
    messages = root / "OAI/openairinterface5g/common/utils/T/T_messages.txt"
    raw = Path(raw_path).resolve() if raw_path else state / "ue.raw"

    materialize = ProcessPlan(
        role="phase14a_materialize", host="W10275",
        argv=(PYTHON, str(root / "rl_agent/splitfusion_phase14a_100mhz_calibration_v1.py"),
              "--config", str(root / "rl_agent/configs/splitfusion_phase14a_100mhz_calibration_v1.json"),
              "--materialize-radio-config", "--output", str(state)),
        stdout_name="radio_materialize.log",
    )
    derive = ProcessPlan(
        role="split_host_runtime_derivation", host="W10275",
        argv=(PYTHON, "-m", _module_name(), "--prepare-runtime", "--state-dir", str(state)),
        stdout_name="split_host_derivation.log",
    )
    probes = tuple(
        ConnectivityProbe(
            destination=destination,
            route_argv=("ip", "-j", "route", "get", destination,
                        "from", topology.local_lan_ip),
            reachability_argv=("ping", "-c", "1", "-W", "2", "-I",
                               topology.local_lan_ip, destination),
        )
        for destination in (topology.amf_ip, topology.ext_dn_ip)
    )
    gnb = ProcessPlan(
        role="gnb", host="W10275",
        argv=(str(ran / "nr-softmodem"), "-O", str(runtime_gnb),
              "--gNBs.[0].min_rxtxtime", "6", "--rfsim",
              "--rfsimulator.[0].options", "chanmod", "--telnetsrv",
              "--telnetsrv.listenaddr", "127.0.0.1", "--telnetsrv.listenport", "9090",
              "--T_stdout", "2", "--T_nowait", "--T_port", str(GNB_T_PORT)),
        env_set=(("SCENESENSE_MCS_POLICY", "sinr"),),
        env_unset=("SCENESENSE_FORCE_UL_MCS", "SCENESENSE_HOLD_MCS_FEW_SAMPLES",
                   "SCENESENSE_AIMD_MAX_DROP"),
        stdout_name="gnb.log",
    )
    ue = ProcessPlan(
        role="ue", host="W10275",
        argv=(str(ran / "nr-uesoftmodem"), "--rfsim",
              "--rfsimulator.[0].serveraddr", "127.0.0.1",
              "--rfsimulator.[0].options", "chanmod", "-r", "273",
              "--numerology", "1", "--band", "78", "-C", "3649260000",
              "--ssb", "516", "-O", str(effective_ue), "--T_stdout", "2",
              "--T_nowait", "--T_port", str(UE_T_PORT)),
        stdout_name="ue.log",
    )
    multi = ProcessPlan(
        role="tracer_multi", host="W10275",
        argv=(str(tracer / "multi"), "-d", str(messages), "-ip", "127.0.0.1",
              "-p", str(UE_T_PORT), "-lp", str(UE_RELAY_PORT)),
        stdout_name="ue_relay.log",
    )
    record_argv = [str(tracer / "record"), "-d", str(messages), "-o", str(raw), "-OFF"]
    for event in UE_EVENTS:
        record_argv += ["-on", event]
    record_argv += ["-ip", "127.0.0.1", "-p", str(UE_RELAY_PORT)]
    record = ProcessPlan(
        role="tracer_record", host="W10275", argv=tuple(record_argv),
        stdout_name="ue_record.log",
    )
    attach = ProcessPlan(
        role="attach_probe", host="W10275",
        argv=("ping", "-I", UE_INTERFACE, "-c", "3", "-W", "2",
              topology.ext_dn_ip), new_session=False,
    )
    return LocalRanPlan(
        schema=SCHEMA, state_dir=str(state), materialize=materialize,
        derive_runtime=derive, preflight=probes, gnb=gnb, ue=ue,
        tracer_multi=multi, tracer_record=record, attach_probe=attach,
        remote_cn_prerequisite_only=True,
        policy_deadline_clock="CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
        cross_host_monotonic_comparison_permitted=False,
    ).validate()


def validate_route_evidence(destination: str, route_json: str,
                            reachability_returncode: int) -> dict[str, Any]:
    """Validate independently captured route and ICMP evidence, fail closed."""
    topology = split_contract.default_topology()
    _require(destination in (topology.amf_ip, topology.ext_dn_ip),
             "unregistered reachability destination")
    try:
        rows = json.loads(route_json)
    except json.JSONDecodeError as exc:
        raise LocalRanLifecycleError(f"route evidence is not JSON: {exc}") from exc
    _require(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict,
             "route evidence must contain exactly one route")
    row = rows[0]
    _require(str(row.get("dst")) == destination, "route destination drift")
    _require(str(row.get("gateway")) == topology.remote_lan_ip,
             "CN route does not traverse L10319")
    preferred = str(row.get("prefsrc") or row.get("src") or "")
    _require(preferred == topology.local_lan_ip, "CN route source is not W10275")
    device = str(row.get("dev") or "")
    _require(bool(device) and device not in {"lo", "docker0", "oai-cn5g"},
             "CN route uses a forbidden local/bridge device")
    _require(type(reachability_returncode) is int and reachability_returncode == 0,
             f"{destination} is not reachable before RAN start")
    return {
        "destination": destination, "gateway": topology.remote_lan_ip,
        "source": topology.local_lan_ip, "device": device,
        "reachability": "PASS",
    }


def validate_tunnel_attachment(address_json: str, ping_returncode: int) -> dict[str, Any]:
    try:
        rows = json.loads(address_json)
    except json.JSONDecodeError as exc:
        raise LocalRanLifecycleError(f"tunnel evidence is not JSON: {exc}") from exc
    _require(type(rows) is list, "tunnel evidence must be a list")
    addresses = [
        str(info.get("local"))
        for row in rows if type(row) is dict
        for info in row.get("addr_info", []) if type(info) is dict
        if info.get("family") == "inet" and info.get("local")
    ]
    _require(addresses == [UE_IP], f"UE tunnel address drift: {addresses}")
    _require(type(ping_returncode) is int and ping_returncode == 0,
             "UE tunnel cannot reach the remote ext-DN")
    return {"interface": UE_INTERFACE, "ipv4": UE_IP,
            "ext_dn_reachability": "PASS"}


@dataclass(frozen=True)
class OwnedProcess:
    role: str
    pid: int
    pgid: int
    executable: str
    argv_sha256: str
    started_monotonic_raw_ns: int
    host: str = "W10275"

    def validate(self) -> "OwnedProcess":
        _require(self.role in {"gnb", "ue", "tracer_multi", "tracer_record"},
                 "unregistered local process role")
        _require(self.host == "W10275", "local lifecycle cannot own remote processes")
        _require(type(self.pid) is int and self.pid > 1, "owned PID is invalid")
        _require(type(self.pgid) is int and self.pgid > 1, "owned PGID is invalid")
        _require(bool(self.executable) and os.path.isabs(self.executable),
                 "owned executable is not absolute")
        _require(bool(re.fullmatch(r"[0-9a-f]{64}", self.argv_sha256)),
                 "owned argv digest is invalid")
        _require(type(self.started_monotonic_raw_ns) is int
                 and self.started_monotonic_raw_ns > 0,
                 "owned start time is invalid")
        return self


@dataclass(frozen=True)
class TunnelOwnership:
    interface: str
    absent_before_start: bool
    observed_ipv4_after_attach: str
    created_by_local_ue: bool

    def validate(self) -> "TunnelOwnership":
        _require(self.interface == UE_INTERFACE, "foreign tunnel interface")
        _require(self.absent_before_start is True, "pre-existing tunnel is not owned")
        _require(self.created_by_local_ue is True, "tunnel creator is not local UE")
        _require(self.observed_ipv4_after_attach == UE_IP, "owned tunnel IP drift")
        return self


def validate_observed_process(owned: OwnedProcess,
                              observed: Mapping[str, Any]) -> None:
    """Prevent PID reuse or a changed command from becoming a teardown target."""
    owned.validate()
    _require(set(observed) == {"pid", "pgid", "executable", "argv_sha256", "host"},
             "observed process fields are incomplete or foreign")
    expected = {
        "pid": owned.pid, "pgid": owned.pgid, "executable": owned.executable,
        "argv_sha256": owned.argv_sha256, "host": "W10275",
    }
    _require(dict(observed) == expected, f"owned process identity drift: {owned.role}")


@dataclass(frozen=True)
class LocalCleanupPlan:
    """Local-only shutdown obligations for a reviewed executor.

    The channel restore is represented as a prerequisite rather than an
    executable telnet command: the existing registered actuator owns the
    protocol.  A live executor must invoke that authority and verify its
    read-back before executing ``process_commands``.
    """

    restore_channel_before_process_stop: bool
    restore_noise_power_db: float
    restore_binding: str
    process_commands: tuple[ProcessPlan, ...]
    tunnel_delete_only_if_present: bool
    remote_cn_cleanup_permitted: bool
    policy_deadline_clock: str

    def validate(self) -> "LocalCleanupPlan":
        _require(self.restore_channel_before_process_stop is True,
                 "RFsim clean-channel restore must precede local process stop")
        _require(self.restore_noise_power_db == -50.0,
                 "RFsim clean-channel restore value drift")
        _require(
            self.restore_binding
            == "splitfusion_phase14b_corrected_four_profile_replay_v1."
               "restore_interrupted_radio",
            "RFsim restore authority drift",
        )
        _require(self.tunnel_delete_only_if_present is True,
                 "owned tunnel deletion must be conditional")
        _require(self.remote_cn_cleanup_permitted is False,
                 "local teardown cannot clean up the remote CN")
        _require(self.policy_deadline_clock
                 == "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
                 "policy deadline left the local W10275 clock")
        _require(bool(self.process_commands), "local cleanup commands are absent")
        for command in self.process_commands:
            command.validate()
        return self


def cleanup_plan(processes: Sequence[OwnedProcess], tunnel: TunnelOwnership,
                 observed: Mapping[int, Mapping[str, Any]]) -> LocalCleanupPlan:
    """Return local-only teardown obligations after ownership revalidation."""
    tunnel.validate()
    by_role: dict[str, OwnedProcess] = {}
    for process in processes:
        process.validate()
        _require(process.role not in by_role, f"duplicate owned role: {process.role}")
        _require(process.pid in observed, f"missing observed process: {process.role}")
        validate_observed_process(process, observed[process.pid])
        by_role[process.role] = process
    _require(set(by_role) == {"gnb", "ue", "tracer_multi", "tracer_record"},
             "local teardown ownership set is incomplete")
    commands: list[ProcessPlan] = []
    for role in ("tracer_record", "tracer_multi", "ue", "gnb"):
        process = by_role[role]
        commands.append(ProcessPlan(
            role=f"stop_{role}", host="W10275",
            argv=("sudo", "kill", "-INT", "--", f"-{process.pgid}"),
            new_session=False,
        ).validate())
    commands.append(ProcessPlan(
        role="remove_owned_ue_tunnel_if_present", host="W10275",
        argv=("sudo", "ip", "link", "delete", "dev", UE_INTERFACE),
        new_session=False,
    ).validate())
    return LocalCleanupPlan(
        restore_channel_before_process_stop=True,
        restore_noise_power_db=-50.0,
        restore_binding=(
            "splitfusion_phase14b_corrected_four_profile_replay_v1."
            "restore_interrupted_radio"
        ),
        process_commands=tuple(commands),
        tunnel_delete_only_if_present=True,
        remote_cn_cleanup_permitted=False,
        policy_deadline_clock="CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
    ).validate()


def attached_attestation(*, state_dir: Path,
                         route_evidence: Sequence[Mapping[str, Any]],
                         tunnel_evidence: Mapping[str, Any],
                         processes: Sequence[OwnedProcess],
                         _source_verifier: Callable[[Path], dict[str, str]]
                         = verify_source_pins) -> dict[str, Any]:
    """Build the split-host attached-state claim without invoking Phase-14a's
    original same-file assertion.
    """
    attestation = load_attestation(
        state_dir, _source_verifier=_source_verifier
    )
    topology = split_contract.default_topology()
    _require({str(row.get("destination")) for row in route_evidence}
             == {topology.amf_ip, topology.ext_dn_ip},
             "attached route evidence is incomplete")
    _require(tunnel_evidence == {"interface": UE_INTERFACE, "ipv4": UE_IP,
                                 "ext_dn_reachability": "PASS"},
             "attached tunnel evidence drift")
    roles: dict[str, OwnedProcess] = {}
    for process in processes:
        process.validate()
        _require(process.role not in roles, "duplicate attached process role")
        roles[process.role] = process
    _require(set(roles) == {"gnb", "ue", "tracer_multi", "tracer_record"},
             "attached process set is incomplete")
    return {
        "schema": ATTACHED_SCHEMA,
        "status": "LOCAL_RAN_ATTACHED_TO_REMOTE_CN",
        "radio_profile_id": RADIO_PROFILE_ID,
        "local_host": topology.local_name,
        "remote_cn_host": topology.remote_name,
        "runtime_gnb_sha256": attestation["split_host_runtime_gnb_sha256"],
        "phase14a_ue_sha256": attestation["phase14a_effective_ue_sha256"],
        "route_evidence": list(route_evidence),
        "tunnel_evidence": dict(tunnel_evidence),
        "processes": {
            role: {
                "pid": process.pid, "pgid": process.pgid,
                "executable": process.executable,
                "argv_sha256": process.argv_sha256,
                "started_monotonic_raw_ns": process.started_monotonic_raw_ns,
            }
            for role, process in sorted(roles.items())
        },
        "original_phase14a_attached_same_file_assertion_satisfied": False,
        "remote_cn_lifecycle_owned": False,
        "cleanup_scope": "LOCAL_GNB_UE_TRACERS_AND_OWNED_TUNNEL_ONLY",
        "policy_deadline_clock": "CLOCK_MONOTONIC_RAW_ON_W10275_ONLY",
        "cross_host_monotonic_comparison_permitted": False,
        "live_run_authorized": False,
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-runtime", action="store_true")
    modes.add_argument("--print-plan", action="store_true")
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prepare_runtime:
        result = prepare_runtime_gnb(args.state_dir)
    else:
        plan = build_local_ran_plan(args.state_dir)
        result = {
            "schema": plan.schema,
            "state_dir": plan.state_dir,
            "preflight": [probe.__dict__ for probe in plan.preflight],
            "processes": {
                item.role: {
                    "argv": list(item.argv), "env_set": list(item.env_set),
                    "env_unset": list(item.env_unset),
                }
                for item in (plan.materialize, plan.derive_runtime, plan.gnb, plan.ue,
                             plan.tracer_multi, plan.tracer_record, plan.attach_probe)
            },
            "remote_cn_prerequisite_only": plan.remote_cn_prerequisite_only,
            "live_run_authorized": False,
        }
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - explicit offline CLI
    raise SystemExit(_main())
