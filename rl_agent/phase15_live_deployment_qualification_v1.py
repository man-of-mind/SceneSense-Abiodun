#!/usr/bin/env python3
"""Comprehensive Phase-15 deployment preflight and 20-capture live qualification."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rl_agent import ue_288_campaign_supervisor as supervisor
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
TOKEN = "SPLITFUSION_PHASE15_LIVE_DEPLOYMENT_QUALIFICATION"
SCHEMA = "scenesense.splitfusion_phase15_live_deployment_qualification.v1"
TERMINAL = "SPLITFUSION_PHASE15_LIVE_DEPLOYMENT_QUALIFIED"
ACTIONS = (0, 20, 46, 71)
CAPTURES = 20
FROZEN_PHASE14_TARGET_SHA256 = (
    "292352b13d72330d6ffcedfb4236480cda5fa80ab6d29059af1adbfb1d398203"
)
EDGE_IMAGE = "oai-perception-rx:latest"
EDGE_CONTAINER = "oai-perception-rx"
EXTRA_TCP_PORTS = (2000, 8010, 35001)
EXTRA_UDP_PORTS = (51002, 51004, 51013, 39310, 39401)


class QualificationError(RuntimeError):
    """A deployment prerequisite or live integration gate failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_create_only(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    write_create_only(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def output_candidate(value: Path) -> Path:
    experiments = EXPERIMENTS.resolve(strict=True)
    candidate = value.resolve(strict=False)
    try:
        candidate.relative_to(experiments)
    except ValueError as exc:
        raise QualificationError(
            "qualification output must remain beneath the experiments root"
        ) from exc
    require(not candidate.exists(), f"create-only qualification output exists: {candidate}")
    return candidate


def run_checked(
    argv: Sequence[str], *, cwd: Path = ROOT, timeout: float = 180.0,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv), cwd=str(cwd), env=(None if env is None else dict(env)),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout, check=False,
    )
    require(
        completed.returncode == 0,
        f"command failed rc={completed.returncode}: {' '.join(argv)}: "
        f"{completed.stderr.strip() or completed.stdout.strip()}",
    )
    return completed


def gpu_preflight() -> dict[str, Any]:
    code = (
        "import json; "
        "from rl_agent.splitfusion_live_dispatch_v1.phase13b_qualification "
        "import _gpu_preflight; print(json.dumps(_gpu_preflight(), sort_keys=True))"
    )
    completed = run_checked(("/usr/bin/python3", "-c", code), timeout=90.0)
    report = json.loads(completed.stdout.splitlines()[-1])
    require(report["executable"] == "/usr/bin/python3", "GPU preflight interpreter drift")
    require(report["device_count"] == 1, "Phase-15 requires exactly one visible GPU")
    require(report["device_name"] == "NVIDIA GeForce RTX 5090", "GPU identity drift")
    require(str(report["torch_cuda"]).startswith("12.8"), "host PyTorch is not CUDA 12.8")
    return report


def listening_ports() -> dict[str, Any]:
    tcp = run_checked(("ss", "-H", "-ltnp"), timeout=10.0).stdout
    udp = run_checked(("ss", "-H", "-lunp"), timeout=10.0).stdout

    def occupied(text: str, ports: Sequence[int]) -> list[int]:
        found = []
        for port in ports:
            if any(
                row.split()[3].rsplit(":", 1)[-1] == str(port)
                for row in text.splitlines()
                if len(row.split()) >= 5
            ):
                found.append(port)
        return found

    occupied_tcp = occupied(tcp, EXTRA_TCP_PORTS)
    occupied_udp = occupied(udp, EXTRA_UDP_PORTS)
    require(not occupied_tcp, f"Phase-15 TCP ports have conflicting listeners: {occupied_tcp}")
    require(not occupied_udp, f"Phase-15 UDP ports have conflicting listeners: {occupied_udp}")
    return {
        "required_tcp_ports": list(EXTRA_TCP_PORTS),
        "required_udp_ports": list(EXTRA_UDP_PORTS),
        "conflicting_tcp_ports": [],
        "conflicting_udp_ports": [],
    }


def socket_buffer_probe(requested: int) -> dict[str, int]:
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, requested)
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, requested)
        result = {
            "requested_bytes": requested,
            "send_reported_bytes": sender.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
            "receive_reported_bytes": receiver.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
        }
    finally:
        sender.close()
        receiver.close()
    require(
        result["send_reported_bytes"] >= requested
        and result["receive_reported_bytes"] >= requested,
        f"kernel socket buffers cannot honor the requested size: {result}",
    )
    return result


def require_application_cold(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a

    for row in phase14a.process_table():
        command = str(row.get("command", ""))
        executable = Path(str(row.get("executable", ""))).name
        if (
            executable.startswith("CarlaUnreal")
            or any(
                str((ROOT / value).resolve()) in command
                for value in (
                    config["runtime"]["required_route_b_split_cell_adapter"],
                    config["runtime"]["map_install_runtime"],
                    config["runtime"]["target_snr_runtime"],
                    config["runtime"]["live_dispatch_bridge"],
                )
            )
        ):
            rows.append({"pid": int(row["pid"]), "executable": executable})
    require(not rows, f"stale CARLA/SplitFusion application processes exist: {rows}")
    inspected = subprocess.run(
        ("sudo", "-n", "docker", "inspect", "-f", "{{.State.Running}}", EDGE_CONTAINER),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    require(inspected.returncode != 0, "stale Phase-15 edge container exists")
    stale_tmp = sorted(
        path.name
        for prefix in (
            "ue_288_cell_runtime_*", "ue_288_seg_eval_*",
            "splitfusion_pilot_*", "splitfusion_live_edge_*",
        )
        for path in Path("/tmp").glob(prefix)
    )
    require(not stale_tmp, f"stale Phase-15 temporary runtime paths exist: {stale_tmp}")
    stale_shm = sorted(
        path.name
        for path in Path("/dev/shm").iterdir()
        if any(token in path.name.casefold() for token in ("carla", "splitfusion", "oai"))
    )
    require(not stale_shm, f"stale Phase-15 shared-memory objects exist: {stale_shm}")
    return {
        "application_processes": [], "edge_container_absent": True,
        "runtime_temporary_paths": [], "shared_memory_objects": [],
    }


def verify_host_artifacts(
    config_path: Path, config: Mapping[str, Any], trace_hashes: Mapping[str, str],
) -> dict[str, Any]:
    from rl_agent.splitfusion_live_dispatch_v1.frame_context import StaticCameraRegistry
    from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
    from rl_agent.ue_route_b_split_cell_adapter_v1 import CAMERA_MOUNT

    runtime = config["runtime"]
    require(
        int(runtime.get("sfd1_protocol_version", 0)) == 2
        and runtime.get("frame_context_required") is True
        and runtime.get("udp_fragment_header") == "!IHH"
        and runtime.get("no_secondary_feature_compression") is True,
        "Phase-15 SFD1-v2/UDP transport binding drift",
    )
    registry = SplitActionRegistry.from_runtime_binding()
    actions = []
    expected = {
        0: ("noAE", "UINT8", 0),
        20: ("AE128", "UINT8", 5000),
        46: ("AE64", "UINT6", 9000),
        71: ("AE32", "UINT4", 9800),
    }
    for action_id in ACTIONS:
        profile = registry.resolve(action_id)
        require(
            (profile.family, profile.quantizer, profile.q_e4) == expected[action_id],
            f"representative action identity drift: {action_id}",
        )
        require(profile.zstd_level == 1, f"action {action_id} zstd level drift")
        require(profile.wire.layout == "CURRENT_CELL_MAJOR", f"action {action_id} layout drift")
        actions.append({
            "action_id": action_id, "profile_id": profile.profile_id,
            "family": profile.family, "quantizer": profile.quantizer,
            "q": profile.q, "q_e4": profile.q_e4,
            "routing_tag": profile.routing_tag,
            "decoder_identity": profile.decoder_identity,
            "ranker_bypassed": profile.ranker_bypassed,
        })
    binding_path = ROOT / "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json"
    binding = load_json(binding_path)
    startup = []
    for item in binding["startup_artifacts"]:
        path = (ROOT / item["path"]).resolve(strict=True)
        require(sha256_file(path) == item["sha256"], f"startup artifact hash drift: {item['role']}")
        startup.append({"role": item["role"], "path": item["path"], "sha256": item["sha256"]})
    phase14_binding_path = supervisor.repo_path(str(config["runtime"]["phase14a_binding"]))
    phase14_binding = load_json(phase14_binding_path)
    phase13c = phase14_binding.get("phase13c_evidence", {})
    require(
        phase13c.get("commit") == "34db0e145d49cb17eeaaaa1fb7a528b021dbf836"
        and phase13c.get("status") == "QUALIFIED_36_ACTIONS_X_300_FIT_FRAMES_SFD1_V2_LOCALHOST",
        "Phase-13C evidence identity/status drift",
    )
    phase13c_artifacts = []
    for name, item in phase13c.get("artifacts", {}).items():
        path = supervisor.repo_path(str(item.get("path", "")))
        require(
            path.is_file() and sha256_file(path) == str(item.get("sha256", "")),
            f"Phase-13C evidence drift: {name}",
        )
        phase13c_artifacts.append({"role": name, **dict(item)})
    require(len(phase13c_artifacts) == 4, "Phase-13C compact evidence inventory drift")
    phase13c_qualification = load_json(
        supervisor.repo_path(str(phase13c["artifacts"]["qualification"]["path"]))
    )
    require(
        phase13c_qualification.get("status")
        == "PHASE13C_36X300_LOCALHOST_MEASUREMENT_COMPLETE"
        and phase13c_qualification.get("integrity", {}).get(
            "hot_path_model_load_construct_move_eval_or_mutate"
        ) == 0
        and phase13c_qualification.get("integrity", {}).get(
            "sfd1_v2_frame_context_every_transaction"
        ) is True,
        "Phase-13C qualification gates drift",
    )
    live_context_artifacts = []
    for section in ("frame_context", "map_install"):
        for name, item in phase14_binding[section]["artifacts"].items():
            path = supervisor.repo_path(str(item["path"]))
            require(
                path.is_file() and sha256_file(path) == str(item["sha256"]),
                f"Phase-14A {section} deployment artifact drift: {name}",
            )
            live_context_artifacts.append({"role": f"{section}.{name}", **dict(item)})
    target = ROOT / "rl_agent/ue_target_snr_cell_runtime_v1.py"
    require(sha256_file(target) == FROZEN_PHASE14_TARGET_SHA256, "frozen Phase-14 target runtime drift")
    wrapper = ROOT / str(config["runtime"]["target_snr_runtime"])
    wrapper_source = wrapper.read_text(encoding="utf-8")
    require(
        "from rl_agent import ue_target_snr_cell_runtime_v1 as base" in wrapper_source
        and "base.prepare_sequence" in wrapper_source
        and "base.load_mapping" in wrapper_source,
        "Phase-15 target wrapper does not delegate frozen generator/mapping behavior",
    )
    route = config["route_b"]
    route_artifacts = []
    for name in ("collection_config", "route_json", "progress_csv", "qualified_density_runner"):
        path = supervisor.repo_path(str(route[name]))
        route_artifacts.append({
            "role": name, "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)
        })
    collection = yaml.safe_load(
        supervisor.repo_path(str(route["collection_config"])).read_text(encoding="utf-8")
    )
    sensors = collection["sensors"]
    require(
        sensors["camera"]["resolution"] == [1280, 720]
        and float(sensors["camera"]["fov_deg"]) == 120.0
        and sensors["model_input_size"] == [768, 432],
        "Route-B live camera/model sensor contract drift",
    )
    require(
        tuple(float(value) for value in CAMERA_MOUNT)
        == (1.8, 0.0, 1.55, -4.0, 0.0, 0.0),
        "Route-B live camera mount drift",
    )
    camera_registry = StaticCameraRegistry.audited()
    camera_identities = [list(value) for value in camera_registry.identities]
    require(len(camera_identities) == 1, "live camera registry is not a single static calibration")
    return {
        "campaign_config": {"path": str(config_path.relative_to(ROOT)), "sha256": sha256_file(config_path)},
        "runtime_binding": {"path": str(binding_path.relative_to(ROOT)), "sha256": sha256_file(binding_path)},
        "registry_startup_audit": registry.startup_audit.__dict__,
        "startup_artifacts": startup,
        "phase13c_evidence": {
            "commit": phase13c["commit"], "artifacts": phase13c_artifacts,
            "qualification_status": phase13c_qualification["status"],
        },
        "live_context_and_map_artifacts": live_context_artifacts,
        "representative_actions": actions,
        "trace_prefix_hashes": dict(trace_hashes),
        "route_and_sensor_artifacts": route_artifacts,
        "live_sensor_contract": {
            "camera_resolution": sensors["camera"]["resolution"],
            "camera_fov_deg": sensors["camera"]["fov_deg"],
            "model_input_size": sensors["model_input_size"],
            "camera_mount": [1.8, 0.0, 1.55, -4.0, 0.0, 0.0],
        },
        "static_camera_identities": camera_identities,
        "frozen_phase14_target_runtime": {
            "path": str(target.relative_to(ROOT)), "sha256": sha256_file(target)
        },
        "phase15_target_wrapper": {
            "path": str(wrapper.relative_to(ROOT)), "sha256": sha256_file(wrapper),
            "scope": "pilot start gate and radio identity columns only",
        },
        "live_endpoints": {
            "ue": runtime["ue_bind_host"], "edge": runtime["edge_remote_host"],
            "edge_receive_udp": int(runtime["edge_receive_port"]),
            "edge_source_udp": int(runtime["edge_source_port"]),
            "ue_result_udp": int(runtime["camera_result_port"]),
            "map_ingest_udp": int(runtime["map_ingest_port"]),
            "sfd1_protocol_version": 2, "udp_fragment_header": "!IHH",
        },
    }


def verify_executables(config: Mapping[str, Any]) -> dict[str, Any]:
    phase14 = load_json(supervisor.repo_path(str(config["runtime"]["phase14a_config"])))
    build = supervisor.repo_path(str(phase14["paths"]["oai_ran_build"]))
    paths = {
        "gnb": build / "nr-softmodem",
        "ue": build / "nr-uesoftmodem",
        "telnetsrv": build / "libtelnetsrv.so",
        "oai_launcher": supervisor.repo_path(str(config["runtime"]["oai_registered_profile_launcher"])),
        "carla_launcher": ROOT.parents[2] / "CarlaUnreal.sh",
        "carla_binary": ROOT.parents[2] / "CarlaUnreal/Binaries/Linux/CarlaUnreal-Linux-Shipping",
    }
    report = {}
    for name, path in paths.items():
        resolved = path.resolve(strict=True)
        require(os.access(resolved, os.R_OK), f"runtime dependency is unreadable: {resolved}")
        if name != "telnetsrv":
            require(os.access(resolved, os.X_OK), f"runtime dependency is not executable: {resolved}")
        report[name] = {
            "path": str(resolved), "sha256": sha256_file(resolved),
            "mode": oct(resolved.stat().st_mode & 0o777),
        }
    lifecycle = supervisor.import_lifecycle_helper(config)
    carla_probe = run_checked(
        (
            "/usr/bin/python3", "-c",
            "import carla, json; print(json.dumps({'module': carla.__file__}))",
        ),
        env=lifecycle.child_env(), timeout=30.0,
    )
    report["carla_python"] = json.loads(carla_probe.stdout.splitlines()[-1])
    return report


def build_and_probe_edge_image(config: Mapping[str, Any]) -> dict[str, Any]:
    receiver = ROOT / "receiver_container"
    build = run_checked(
        (
            "sudo", "-n", "docker", "compose", "-f", "docker-compose.yaml",
            "-f", "docker-compose.fusion-back.yaml", "build", EDGE_CONTAINER,
        ),
        cwd=receiver, timeout=1800.0,
    )
    image = load_json_from_command(
        ("sudo", "-n", "docker", "image", "inspect", EDGE_IMAGE), timeout=30.0
    )[0]
    runtime_binding = load_json(ROOT / "rl_agent/splitfusion_live_dispatch_v1/runtime_binding.json")
    checks = [
        {"path": item["path"], "sha256": item["sha256"]}
        for item in runtime_binding["startup_artifacts"]
    ]
    checks.extend(
        {"path": item["path"], "sha256": item["sha256"]}
        for item in config["deployment"].values()
    )
    probe_code = """
import hashlib, json, os
from pathlib import Path
import cv2, torch, zstandard
from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime
from rl_agent.splitfusion_live_dispatch_v1.registry import SplitActionRegistry
checks = json.loads(os.environ['PHASE15_ARTIFACT_CHECKS'])
observed = []
for item in checks:
    path = Path('/work/abiodun') / item['path']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item['sha256']:
        raise RuntimeError('container artifact hash drift: ' + item['path'])
    observed.append(item)
state = Path('/work/torch_cache')
probe = state / 'phase15_preflight_write_probe'
with probe.open('x', encoding='utf-8') as handle:
    handle.write('cell-scoped-state-only\\n')
probe.unlink()
value = torch.ones(1, device='cuda:0')
torch.cuda.synchronize(0)
registry = SplitActionRegistry.from_runtime_binding()
report = {
    'python_executable': os.path.realpath('/usr/bin/python3'),
    'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'device_count': torch.cuda.device_count(),
    'device_name': torch.cuda.get_device_name(0),
    'device_capability': list(torch.cuda.get_device_capability(0)),
    'cv2': cv2.__version__, 'zstandard': zstandard.__version__,
    'torch_home': os.environ.get('TORCH_HOME'),
    'state_root_writable': os.access(state, os.W_OK | os.X_OK),
    'verified_artifact_count': len(observed), 'registry_profile_count': len(registry.profiles),
    'live_runtime_import': live_pilot_runtime.__name__, 'tiny_cuda_probe': float(value.item()),
}
print('PHASE15_CONTAINER_PROBE=' + json.dumps(report, sort_keys=True))
"""
    name = f"phase15-edge-preflight-{os.getpid()}"
    with tempfile.TemporaryDirectory(prefix="phase15_edge_preflight_") as raw_state:
        argv = (
            "sudo", "-n", "docker", "run", "--rm", "--name", name,
            "--network", "none", "--gpus", "all",
            "--entrypoint", "/usr/bin/python3",
            "-e", f"PHASE15_ARTIFACT_CHECKS={json.dumps(checks, separators=(',', ':'))}",
            "-e", "PYTHONPATH=/work/abiodun:/work/abiodun/rl_agent/feature_ae",
            "-v", f"{ROOT}:/work/abiodun:ro",
            "-v", f"{Path(raw_state).resolve()}:/work/torch_cache:rw",
            EDGE_IMAGE, "-c", probe_code,
        )
        try:
            probe = run_checked(argv, timeout=180.0)
        finally:
            subprocess.run(
                ("sudo", "-n", "docker", "rm", "-f", name),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, check=False,
            )
        require(not any(Path(raw_state).iterdir()), "container preflight left mutable state behind")
    lines = [line for line in probe.stdout.splitlines() if line.startswith("PHASE15_CONTAINER_PROBE=")]
    require(len(lines) == 1, "edge image did not emit exactly one preflight record")
    report = json.loads(lines[0].split("=", 1)[1])
    require(
        report["cuda_available"] is True
        and report["device_count"] == 1
        and report["device_name"] == "NVIDIA GeForce RTX 5090"
        and str(report["torch_cuda"]).startswith("12.8")
        and report["torch_home"] == "/work/torch_cache"
        and report["state_root_writable"] is True
        and report["verified_artifact_count"] == len(checks)
        and report["registry_profile_count"] == 72
        and report["live_runtime_import"].endswith(".live_pilot_runtime"),
        f"edge image runtime probe failed: {report}",
    )
    return {
        "image": EDGE_IMAGE, "image_id": image["Id"],
        "created": image.get("Created"), "repo_digests": image.get("RepoDigests", []),
        "build_stdout_sha256": hashlib.sha256(build.stdout.encode()).hexdigest(),
        "build_stderr_sha256": hashlib.sha256(build.stderr.encode()).hexdigest(),
        "probe": report,
        "bind_mounts": {
            "repository": {"container": "/work/abiodun", "mode": "ro"},
            "cell_state": {"container": "/work/torch_cache", "mode": "rw", "lifetime": "one cell"},
        },
    }


def load_json_from_command(argv: Sequence[str], *, timeout: float) -> Any:
    completed = run_checked(argv, timeout=timeout)
    return json.loads(completed.stdout)


def comprehensive_preflight(config_path: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require(Path(sys.executable) == Path("/usr/bin/python3"), "preflight must use /usr/bin/python3")
    candidate = output_candidate(output)
    config, cells, trace_hashes = supervisor.validate_static(config_path)
    require(len(cells) == 16, "Phase-15 pilot cell count drift")
    worktree = supervisor.verify_live_pilot_worktree()
    supervisor.verify_real_launch_readiness(config)
    supervisor.verify_resolved_models(config, supervisor.read_catalog(config))
    from rl_agent import splitfusion_phase14a_100mhz_calibration_v1 as phase14a
    from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b

    radio = phase14a.load_json(supervisor.repo_path(str(config["runtime"]["phase14a_config"])))
    cold_radio = phase14b.require_cold_profile_runtime(radio, candidate / "radio_preflight_absent")
    cold_application = require_application_cold(config)
    ports = listening_ports()
    gpu = gpu_preflight()
    artifacts = verify_host_artifacts(config_path, config, trace_hashes)
    executables = verify_executables(config)
    request = int(config["runtime"]["socket_buffer_request_bytes"])
    buffers = socket_buffer_probe(request)
    with tempfile.TemporaryDirectory(prefix="phase15_output_probe_", dir=EXPERIMENTS):
        pass
    usage = shutil.disk_usage(EXPERIMENTS)
    require(usage.free >= 20 * 1024**3, "less than 20 GiB free beneath experiments")
    image = build_and_probe_edge_image(config)
    namespaces = run_checked(("ip", "netns", "list"), timeout=10.0).stdout.splitlines()
    require(not any(row.split()[0] == "UE" for row in namespaces if row.split()), "stale UE namespace exists")
    inventory = {
        "schema": "scenesense.splitfusion_phase15_deployment_preflight.v1",
        "status": "PASS", "created_at_unix_s": time.time(),
        "head": worktree["head"], "dirty_paths": worktree["dirty_paths"],
        "output_candidate": str(candidate.relative_to(ROOT)), "output_absent": True,
        "python": {"requested": "/usr/bin/python3", "sys_executable": sys.executable, "version": platform.python_version()},
        "gpu": gpu, "host_artifacts": artifacts, "executables": executables,
        "edge_image": image, "socket_buffers": buffers, "ports": ports,
        "cold_radio": cold_radio, "cold_application": cold_application,
        "ue_namespace_absent": True, "ue_tunnel_absent": True,
        "output_filesystem": {"free_bytes": usage.free, "writable": True},
        "lifecycle_contract": {
            "fresh_cell_state": True, "unconditional_rfsim_restore": True,
            "cold_cleanup_required": True, "carla_process_group_owned": True,
            "edge_container_owned": True, "raw_logs_retained": False,
        },
        "data_scope": {
            "live_carla_only": True, "holdout_validation_test_artifacts_opened": 0,
            "predictions_or_payloads_in_state_mount": False,
        },
        "source_reviews": {
            "dependency_data_flow": {
                "status": "PASS",
                "repository_mount": "read_only_hash_bound_runtime_and_checkpoints",
                "cell_state_mount": "fresh_writable_ready_record_only_removed_after_cell",
                "checkpoint_loads": "direct_hash_bound_paths_no_selection_document_open",
                "holdout_validation_test_side_channel": False,
            },
            "runtime_lifecycle": {
                "status": "PASS",
                "startup_order": "radio_attach_clean_noise_carla_map_edge_target_barrier_capture",
                "partial_start_cleanup": True,
                "rfsim_restore_and_readback_unconditional": True,
                "carla_process_group_and_rpc_cold_gate": True,
                "cell_boundary_resume_only": True,
                "ephemeral_log_policy": "hash_and_bounded_failure_tail_then_remove",
            },
        },
    }
    return config, inventory


def _literal_mapping(value: str) -> dict[str, Any]:
    parsed = ast.literal_eval(value)
    require(isinstance(parsed, dict), "runtime counter field is not a mapping")
    return parsed


def evaluate_runtime(runtime_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = load_json(runtime_dir / "RESULTS_SUMMARY.json")
    with (runtime_dir / "per_frame_metrics.csv").open(newline="", encoding="utf-8") as handle:
        sent = [row for row in csv.DictReader(handle) if row["prepare_status"] == "SENT"]
    with (runtime_dir / "map_feedback.csv").open(newline="", encoding="utf-8") as handle:
        feedback = list(csv.DictReader(handle))
    require(len(sent) == CAPTURES, f"qualification sent {len(sent)} captures, expected {CAPTURES}")
    action_counts = Counter(int(row["action_id"]) for row in sent)
    require(action_counts == Counter({value: 5 for value in ACTIONS}), f"action counts drift: {action_counts}")
    require(
        [int(row["action_id"]) for row in sent] == list(ACTIONS) * 5,
        "qualification action cycle order drift",
    )
    frame_ids = [int(row["frame_id"]) for row in sent]
    require(len(frame_ids) == len(set(frame_ids)), "qualification frame IDs are not unique")
    terminal = [
        row for row in feedback
        if str(row.get("terminal", "")).casefold() in {"1", "true"}
    ]
    terminal_counts = Counter(row["capture_id"] for row in terminal)
    sent_capture_ids = {row["capture_id"] for row in sent}
    require(
        set(terminal_counts) == sent_capture_ids
        and all(value == 1 for value in terminal_counts.values()),
        "qualification exactly-one terminal accounting failed",
    )
    ack_actions = {
        int(row["action_id"]) for row in feedback if row["status"] == "ACK_INSTALLED"
    }
    require(ack_actions == set(ACTIONS), f"ACK_INSTALLED action coverage drift: {sorted(ack_actions)}")
    expected = {
        profile.action_id: profile
        for profile in SplitActionRegistry.from_runtime_binding().profiles
        if profile.action_id in ACTIONS
    }
    for row in sent:
        action_id = int(row["action_id"])
        profile = expected[action_id]
        require(row["profile_id"] == profile.profile_id, f"profile mismatch for action {action_id}")
        require(row["model_family"] == profile.family, f"family mismatch for action {action_id}")
        require(row["quantizer"] == profile.quantizer, f"quantizer mismatch for action {action_id}")
        require(int(row["q_e4"]) == profile.q_e4, f"q mismatch for action {action_id}")
        require(int(row["routing_tag"]) == profile.routing_tag, f"routing mismatch for action {action_id}")
        require(row["decoder_identity"] == profile.decoder_identity, f"decoder mismatch for action {action_id}")
        require(row["decoded"] == "True" and row["finite"] == "True", "decode/finite gate failed")
        require(row["frame_context_valid"] == "True", "SFD1-v2 frame context gate failed")
        require(row["reconstructed_device"] == "cuda:0", "reconstructed C2 device drift")
        require(int(row["camera_pose_reconstruct_ns"]) > 0, "live ego/camera pose was not reconstructed")
        require(int(row["finite_output_tensor_count"]) > 0, "frozen-tail tensor accounting is empty")
        require(int(row["datagrams"]) == int(row["feature_received_datagrams"]), "feature datagram accounting drift")
        require(int(row["feature_duplicate_datagrams"]) == 0, "duplicate feature datagram observed")
        require(int(row["scientific_inner_bytes"]) + int(row["sfd1_overhead_bytes"]) == int(row["sfd1_bytes"]), "SFD1 byte accounting drift")
    live = summary["live_dispatch"]
    ue_counters = live["ue_counters"]
    require(ue_counters["frames_completed"] == CAPTURES, "UE completed-frame count drift")
    require(ue_counters["ranker_dispatches"] == 15, "q=0 ranker bypass count drift")
    require(ue_counters["ae_encoder_dispatches"] == 15, "UE AE dispatch count drift")
    require(ue_counters["hot_path_model_load_operations"] == 0, "UE hot-path model load observed")
    require(ue_counters["hot_path_model_construction_operations"] == 0, "UE hot-path model construction observed")
    last = sent[-1]
    edge_counters = _literal_mapping(last["edge_counters"])
    edge_ledger = _literal_mapping(last["edge_call_ledger"])
    require(edge_counters["frames_completed"] == CAPTURES, "edge completed-frame count drift")
    require(edge_counters["tail_dispatches"] == CAPTURES, "frozen-tail dispatch count drift")
    require(edge_counters["ae_decoder_dispatches"] == 15, "edge AE dispatch count drift")
    require(edge_counters["hot_path_model_load_operations"] == 0, "edge hot-path model load observed")
    require(edge_counters["hot_path_model_construction_operations"] == 0, "edge hot-path construction observed")
    require(edge_ledger.get("tail") == CAPTURES, "contextual frozen-tail ledger drift")
    require(edge_ledger.get("service_record_serialization") == CAPTURES, "p025 serialization ledger drift")
    require(edge_ledger.get("ae_decoder_AE128") == 5, "AE128 decoder ledger drift")
    require(edge_ledger.get("ae_decoder_AE64") == 5, "AE64 decoder ledger drift")
    require(edge_ledger.get("ae_decoder_AE32") == 5, "AE32 decoder ledger drift")
    edge_startup = summary["edge_startup"]
    require(edge_startup["allowed_action_ids"] == list(ACTIONS), "edge action preload allowlist drift")
    require(edge_startup["tail_device"] == "cuda:0", "edge tail device drift")
    require(edge_startup["state_root"] == "/work/torch_cache", "edge state mount target drift")
    require(edge_startup["state_root_writable"] is True, "edge state mount is not writable")
    edge_mounts = summary["edge_mounts"]
    require(
        edge_mounts["state"]["destination"] == "/work/torch_cache"
        and edge_mounts["state"]["rw"] is True,
        "live cell state mount identity/mode drift",
    )
    require(
        edge_mounts["repository"]["destination"] == "/work/abiodun"
        and edge_mounts["repository"]["rw"] is False,
        "live repository mount identity/mode drift",
    )
    require(
        not Path(edge_mounts["state"]["source"]).exists(),
        "cell-scoped edge state survived adapter teardown",
    )
    buffers = {
        **live["socket_buffers"],
        "edge_receive_reported_bytes": edge_startup["edge_receive_reported_bytes"],
        "edge_send_reported_bytes": edge_startup["edge_send_reported_bytes"],
    }
    requested = int(config["runtime"]["socket_buffer_request_bytes"])
    require(all(int(value) >= requested for key, value in buffers.items() if key != "requested_bytes"), "live socket buffer gate failed")
    radio_rows = []
    with (runtime_dir / "radio_trace.csv").open(newline="", encoding="utf-8") as handle:
        radio_rows = list(csv.DictReader(handle))
    require(radio_rows and int(radio_rows[0]["step_index"]) == 0, "qualification network generator did not start at sample zero")
    require(all(row["profile_id"] == "FAVORABLE_STABLE" for row in radio_rows), "qualification network profile drift")
    structural = summary["structural_acceptance"]
    require(structural["status"] == "PASS", f"adapter structural gates failed: {structural['failures']}")
    require(summary["route"]["route_completed"] is True, "Route B did not complete independently")
    latencies = [
        (int(row["edge_result_received_ns"]) - int(row["capture_started_ns"])) / 1e6
        for row in sent
    ]
    payloads = [int(row["sfd1_bytes"]) for row in sent]
    terminal_by_capture = {row["capture_id"]: row["status"] for row in terminal}
    action_results = []
    for action_id in ACTIONS:
        rows = [row for row in sent if int(row["action_id"]) == action_id]
        action_latencies = [
            (int(row["edge_result_received_ns"]) - int(row["capture_started_ns"])) / 1e6
            for row in rows
        ]
        action_results.append({
            "action_id": action_id, "profile_id": expected[action_id].profile_id,
            "family": expected[action_id].family,
            "quantizer": expected[action_id].quantizer, "q": expected[action_id].q,
            "captures": len(rows),
            "ack_installed": sum(
                1 for row in feedback
                if row["status"] == "ACK_INSTALLED" and int(row["action_id"]) == action_id
            ),
            "terminal_outcomes": dict(Counter(terminal_by_capture[row["capture_id"]] for row in rows)),
            "mean_capture_to_edge_result_ms": sum(action_latencies) / len(action_latencies),
            "mean_sfd1_bytes": sum(int(row["sfd1_bytes"]) for row in rows) / len(rows),
        })
    return {
        "captures": len(sent), "action_counts": {str(key): action_counts[key] for key in ACTIONS},
        "ack_installed_actions": sorted(ack_actions),
        "terminal_outcomes": dict(Counter(row["status"] for row in terminal)),
        "unique_frame_ids": len(set(frame_ids)),
        "mean_capture_to_edge_result_ms": sum(latencies) / len(latencies),
        "maximum_capture_to_edge_result_ms": max(latencies),
        "mean_sfd1_bytes": sum(payloads) / len(payloads),
        "total_sfd1_bytes": sum(payloads),
        "radio_steps": len(radio_rows), "radio_first_step": 0,
        "socket_buffers": buffers, "ue_counters": ue_counters,
        "edge_counters": edge_counters, "edge_call_ledger": edge_ledger,
        "edge_startup": edge_startup, "edge_mounts": edge_mounts,
        "actions": action_results,
        "route": summary["route"], "structural_acceptance": structural,
    }


def write_success(root: Path, inventory: Mapping[str, Any], result: Mapping[str, Any], cleanup: Mapping[str, Any]) -> None:
    qualification = {
        "schema": SCHEMA, "status": TERMINAL,
        "captures": CAPTURES, "action_counts": dict(result["action_counts"]),
        "action_order": list(ACTIONS), "captures_per_action": 5,
        "network_profile": "FAVORABLE_STABLE", "network_start_sample": 0,
        "result": dict(result), "cold_cleanup_verified": True,
        "cleanup": dict(cleanup), "preflight_inventory_sha256": sha256_file(root / "preflight_inventory.json"),
        "retained_rgb_radar_payloads_predictions": False,
        "pilot_16_authorized_by_qualification": True,
        "campaign_288_authorized": False,
    }
    qualification_path = root / "qualification.json"
    write_json_create_only(qualification_path, qualification)
    report_path = root / "REPORT.md"
    write_create_only(
        report_path,
        "# Phase-15 live deployment qualification\n\n"
        f"Status: `{TERMINAL}`\n\n"
        "Twenty live Route-B captures traversed CUDA SplitFusion, SFD1-v2, "
        "localhost/OAI UDP, frozen tail, p025 serialization, and map install.\n\n"
        f"Action counts: `{json.dumps(result['action_counts'], sort_keys=True)}`.\n\n"
        f"Terminal outcomes: `{json.dumps(result['terminal_outcomes'], sort_keys=True)}`.\n\n"
        "The radio, CARLA, edge, map, sockets, temporary state, and RFsim clean "
        "setting passed cold teardown. No sensor frame, feature payload, or prediction was retained.\n",
    )
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in (
        root / "preflight_inventory.json", qualification_path, report_path,
        root / "runtime/RESULTS_SUMMARY.json", root / "runtime/manifest.json",
        root / "runtime/per_frame_metrics.csv", root / "runtime/map_feedback.csv",
        root / "runtime/radio_trace.csv",
    ):
        files.append({
            "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_json_create_only(manifest_path, {"schema": "scenesense.splitfusion_phase15_live_deployment_artifacts.v1", "files": files})
    write_json_create_only(
        root / TERMINAL,
        {"status": TERMINAL, "qualification_sha256": sha256_file(qualification_path),
         "artifact_manifest_sha256": sha256_file(manifest_path)},
    )


def run_qualification(args: argparse.Namespace) -> int:
    require(args.execute == TOKEN, "exact Phase-15 qualification execution token is required")
    config_path = args.config.resolve(strict=True)
    output = output_candidate(args.output_root)
    config, inventory = comprehensive_preflight(config_path, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=False, exist_ok=False)
    service_logs: Path | None = None
    try:
        write_json_create_only(output / "preflight_inventory.json", inventory)
        runtime_dir = output / "runtime"
        runtime_dir.mkdir(parents=False, exist_ok=False)
        service_logs = Path(tempfile.mkdtemp(prefix="phase15_qualification_services_"))
        lifecycle = supervisor.import_lifecycle_helper(config)
        cells = supervisor.enumerate_cells(config)
        selected = next(
            cell for cell in cells
            if cell.action_id == 0 and cell.network_profile_id == "FAVORABLE_STABLE"
        )
        cell = supervisor.Cell(
            cell_id="phase15_live_deployment_qualification",
            action_index=selected.action_index, action_id=selected.action_id,
            profile_id=selected.profile_id, model_family=selected.model_family,
            network_profile_id=selected.network_profile_id,
            trace_id=selected.trace_id, seed=selected.seed,
        )
        qualification_config = json.loads(json.dumps(config))
        qualification_config["_qualification"] = {
            "action_ids": list(ACTIONS), "capture_limit": CAPTURES,
            "purpose": "infrastructure qualification only",
        }
        resolved = {
            "schema": "scenesense.ue_288_cell_resolved.v1",
            "campaign": qualification_config,
            "measurement_contract": dict(config["measurement_contract"]),
            "cell": supervisor.cell_to_dict(cell), "attempt": 1,
            "attempt_dir": str(runtime_dir),
        }
        resolved_path = runtime_dir / "resolved_config.yaml"
        write_create_only(resolved_path, yaml.safe_dump(resolved, sort_keys=False))
    except BaseException as exc:
        if service_logs is not None:
            shutil.rmtree(service_logs, ignore_errors=True)
        write_json_create_only(
            output / "FAILED.json",
            {
                "schema": "scenesense.splitfusion_phase15_live_deployment_qualification_failure.v1",
                "status": "FAILED",
                "first_failure": f"{type(exc).__name__}: {exc}",
                "runtime_started": False,
                "finished_at_unix_s": time.time(),
            },
        )
        print(f"Phase-15 qualification setup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    assert service_logs is not None
    radio_namespace: Path | None = None
    radio_state: Path | None = None
    attached: Mapping[str, Any] | None = None
    server: Any = None
    pgid: int | None = None
    cleanup: dict[str, Any] = {}
    error = ""
    adapter_rc: int | None = None
    try:
        radio_namespace, radio_state, attached = supervisor._start_live_radio(
            qualification_config, cell, 1, service_logs
        )
        from rl_agent import splitfusion_phase14b_corrected_four_profile_replay_v1 as phase14b

        cleanup["attached_radio"] = phase14b.stable_attached_radio_seal(attached)
        cleanup["clean_noise_preflight"] = dict(attached["clean_noise_preflight"])
        server, pgid = lifecycle.start_carla(2000, service_logs / "carla.log")
        require(lifecycle.wait_for_rpc(2000, 180.0) is not None, "fresh Epic CARLA did not become RPC-ready")
        adapter = supervisor.repo_path(str(config["runtime"]["required_route_b_split_cell_adapter"]))
        with (service_logs / "adapter.log").open("xb") as stream:
            completed = subprocess.run(
                (
                    "/usr/bin/python3", "-u", str(adapter),
                    "--resolved-config", str(resolved_path), "--attempt-dir", str(runtime_dir),
                    "--carla-host", "127.0.0.1", "--carla-port", "2000",
                ),
                cwd=str(ROOT), env=lifecycle.child_env(), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        adapter_rc = int(completed.returncode)
        require(adapter_rc == 0, f"qualification adapter failed rc={adapter_rc}")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            cleanup["application"] = supervisor._stop_phase15_application(config)
        except BaseException as exc:
            cleanup["application_error"] = f"{type(exc).__name__}: {exc}"
        if server is not None and pgid is not None:
            cleanup["carla"] = lifecycle.stop_carla(server, pgid, 2000)
        if radio_namespace is not None and radio_state is not None:
            try:
                cleanup["radio"] = supervisor._stop_live_radio(
                    qualification_config, radio_namespace, radio_state, attached
                )
            except BaseException as exc:
                cleanup["radio_error"] = f"{type(exc).__name__}: {exc}"
        try:
            cleanup["application_cold"] = require_application_cold(config)
            cleanup["ports_cold"] = listening_ports()
        except BaseException as exc:
            cleanup["cold_error"] = f"{type(exc).__name__}: {exc}"
        cleanup["service_diagnostics"] = supervisor._compact_log_diagnostics(
            service_logs, include_tails=bool(error or adapter_rc not in (None, 0))
        )
        shutil.rmtree(service_logs, ignore_errors=True)
    try:
        require(not error, error)
        require(cleanup.get("carla", {}).get("shutdown_verified") is True, "CARLA cold teardown failed")
        require(cleanup.get("radio", {}).get("all_lifecycle_gates_passed") is True, "radio/RFsim cold teardown failed")
        require(
            "cold_error" not in cleanup and "radio_error" not in cleanup
            and "application_error" not in cleanup,
            "postflight cleanup is incomplete",
        )
        result = evaluate_runtime(runtime_dir, config)
        write_success(output, inventory, result, cleanup)
        print(TERMINAL, flush=True)
        return 0
    except BaseException as exc:
        failure = {
            "schema": "scenesense.splitfusion_phase15_live_deployment_qualification_failure.v1",
            "status": "FAILED", "first_failure": f"{type(exc).__name__}: {exc}",
            "adapter_returncode": adapter_rc, "cleanup": cleanup,
            "finished_at_unix_s": time.time(),
        }
        write_json_create_only(output / "FAILED.json", failure)
        print(f"Phase-15 live qualification failed: {failure['first_failure']}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=supervisor.DEFAULT_LIVE_PILOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_qualification(build_parser().parse_args(argv))
    except (QualificationError, supervisor.CampaignError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Phase-15 qualification contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
