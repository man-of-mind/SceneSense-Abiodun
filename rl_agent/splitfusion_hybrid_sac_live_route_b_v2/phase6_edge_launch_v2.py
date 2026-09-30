"""Phase 6 no-build edge launch bound to one content-addressed image.

Setup-repair addendum 3. This changes only how the already-built edge image is
started and how its provenance is proven; the scientific protocol is unchanged.

The shared, hash-pinned ``scripts/receiver_container_fusion_back_up.sh`` runs
``docker compose ... up -d --build --force-recreate``. With the host build
cache and base image pruned, that rebuilt the image inside the 180-s launch
budget and failed the first Phase-6 attempt. This module reproduces the
launcher exactly -- the same network check, NVIDIA-runtime probe, edge-state
validation, compose files, variable derivation and ``sudo VAR=... docker
compose`` environment filtering -- except that ``up`` runs with
``--no-build --pull never --force-recreate``. It never builds, pulls, retags or
downloads anything.

Image authority is the full image ID :data:`ADMITTED_IMAGE_ID`, not the mutable
tag. The tag is resolved and required to equal the admitted ID before launch,
and the created container's ``.Image`` is required to equal it before the
caller accepts readiness. Both are written as durable evidence, closing the
tag-movement / TOCTOU gap. The mounts are checked at the same point:
repository -> ``/work/abiodun`` read-only, per-attempt state ->
``/work/torch_cache`` read-write, FCOS constructor checkpoint read-only.

:func:`install_adapter_launch_seam` routes only the direct adapter's call to the
legacy launcher through :func:`launch_no_build`; every other subprocess call and
every readiness, CUDA and ready-record check in the adapter is unchanged.

Importing this module performs no I/O and starts nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, IO, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
ADMITTED_IMAGE_ID = "sha256:2be62d533b8077ceecab5455d5377f2f952b6a50d43ff8c04dc89ce18027d6ba"
IMAGE_TAG = "oai-perception-rx:latest"
CONTAINER = "oai-perception-rx"
NETWORK = "oai-cn5g-public-net"
COMPOSE_DIR = ROOT / "receiver_container"
COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.fusion-back.yaml")
CONFIG_ENV = ROOT / "scripts" / "config.env"
LEGACY_LAUNCHER = ROOT / "scripts" / "receiver_container_fusion_back_up.sh"
LEGACY_LAUNCHER_SHA256 = "abf532c88d27fcf101dcecbfcf55f4994fd44cd287d5d1a9245ce0ca3b333610"
REPOSITORY_DESTINATION = "/work/abiodun"
STATE_DESTINATION = "/work/torch_cache"
FCOS_DESTINATION = ("/home/shr_aisvcs/.cache/torch/hub/checkpoints/"
                    "fcos_resnet50_fpn_coco-99b0c9b7.pth")
NVIDIA_PROBE_ATTEMPTS = 5
NVIDIA_PROBE_TIMEOUT_S = 20.0
NVIDIA_RUNTIME = "nvidia"
UP_ARGUMENTS = ("up", "-d", "--no-build", "--pull", "never", "--force-recreate")

# Exactly the variables the legacy launcher passes through ``sudo`` (in order).
COMPOSE_VARIABLES = (
    "FUSION_BACK_BIND_HOST", "FUSION_BACK_REMOTE_HOST", "FUSION_BACK_REMOTE_HOST_1",
    "FUSION_BACK_REMOTE_HOST_2", "FUSION_BACK_DEVICE", "FUSION_BACK_SCRIPT",
    "FUSION_BACK_CHECKPOINT", "FUSION_QUANTIZATION_MODE", "FUSION_ENTROPY_CODER",
    "FUSION_BACK_LOG_EVERY", "FUSION_BACK_DUAL", "FUSION_BACK_EXTRA_ARGS",
    "FUSION_REMOTE_PORT_1", "FUSION_REMOTE_SOURCE_PORT_1", "FUSION_CAMERA_RESULT_PORT_1",
    "FUSION_REMOTE_PORT_2", "FUSION_REMOTE_SOURCE_PORT_2", "FUSION_CAMERA_RESULT_PORT_2",
    "SPLITFUSION_EDGE_STATE_ROOT",
)
# The legacy ``${NAME:-default}`` defaults (empty counts as unset).
_DEFAULTS = {
    "FUSION_BACK_BIND_HOST": "0.0.0.0",
    "FUSION_BACK_DUAL": "1",
    "FUSION_BACK_DEVICE": "cuda",
    "FUSION_BACK_SCRIPT":
        "/work/abiodun/carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py",
    "FUSION_BACK_CHECKPOINT": "/work/abiodun/checkpoints/fusion_object_best.pt",
    "FUSION_QUANTIZATION_MODE": "per_channel_uint8",
    "FUSION_ENTROPY_CODER": "zstd",
    "FUSION_BACK_LOG_EVERY": "30",
    "FUSION_BACK_EXTRA_ARGS": "",
    "FUSION_REMOTE_PORT_1": "51002",
    "FUSION_REMOTE_SOURCE_PORT_1": "51003",
    "FUSION_CAMERA_RESULT_PORT_1": "51004",
    "FUSION_REMOTE_PORT_2": "51102",
    "FUSION_REMOTE_SOURCE_PORT_2": "51103",
    "FUSION_CAMERA_RESULT_PORT_2": "51104",
}
_EXPORT = re.compile(r'^export\s+([A-Z0-9_]+)="([^"]*)"')

Runner = Callable[..., subprocess.CompletedProcess]


class EdgeImageError(RuntimeError):
    """The admitted edge image is missing, drifted, or launched incorrectly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EdgeImageError(message)


def _run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    kwargs.setdefault("check", False)
    return subprocess.run(list(argv), **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


# ---------------------------------------------------------------------------
# Command and environment (pure)
# ---------------------------------------------------------------------------


def config_env(path: Path = CONFIG_ENV) -> dict[str, str]:
    """The ``export NAME="value"`` assignments the legacy launcher sources."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = _EXPORT.match(line.strip())
        if match:
            values[match.group(1)] = re.sub(
                r"\$\{([A-Z0-9_]+)\}", lambda m: values.get(m.group(1), ""), match.group(2))
    return values


def compose_variables(env: Mapping[str, str], *, config: Mapping[str, str],
                      interface_exists: Callable[[str], bool],
                      default_state_root: Optional[Path] = None) -> dict[str, str]:
    """The legacy launcher's variable derivation, reproduced exactly."""

    def get(name: str) -> str:
        value = env.get(name, "")
        return value if value != "" else _DEFAULTS.get(name, "")

    out: dict[str, str] = {}
    out["FUSION_BACK_BIND_HOST"] = get("FUSION_BACK_BIND_HOST")
    out["FUSION_BACK_REMOTE_HOST"] = env.get("FUSION_BACK_REMOTE_HOST") or config["OAI_UE_IP"]
    out["FUSION_BACK_DUAL"] = get("FUSION_BACK_DUAL")
    out["FUSION_BACK_REMOTE_HOST_1"] = (env.get("FUSION_BACK_REMOTE_HOST_1")
                                        or out["FUSION_BACK_REMOTE_HOST"])
    if env.get("FUSION_BACK_REMOTE_HOST_2"):
        out["FUSION_BACK_REMOTE_HOST_2"] = env["FUSION_BACK_REMOTE_HOST_2"]
    elif out["FUSION_BACK_DUAL"] == "1" and interface_exists(config["OAI_UE2_IFACE"]):
        out["FUSION_BACK_REMOTE_HOST_2"] = config["OAI_UE2_IP"]
    else:
        out["FUSION_BACK_REMOTE_HOST_2"] = out["FUSION_BACK_REMOTE_HOST"]
    for name in ("FUSION_BACK_DEVICE", "FUSION_BACK_SCRIPT", "FUSION_BACK_CHECKPOINT",
                 "FUSION_QUANTIZATION_MODE", "FUSION_ENTROPY_CODER", "FUSION_BACK_LOG_EVERY",
                 "FUSION_BACK_EXTRA_ARGS", "FUSION_REMOTE_PORT_1",
                 "FUSION_REMOTE_SOURCE_PORT_1", "FUSION_CAMERA_RESULT_PORT_1",
                 "FUSION_REMOTE_PORT_2", "FUSION_REMOTE_SOURCE_PORT_2",
                 "FUSION_CAMERA_RESULT_PORT_2"):
        out[name] = get(name)
    state = env.get("SPLITFUSION_EDGE_STATE_ROOT", "")
    if state:
        path = Path(state)
        require(path.is_dir() and os.access(path, os.W_OK),
                "supplied edge state root must be an existing writable directory")
        out["SPLITFUSION_EDGE_STATE_ROOT"] = str(path.resolve(strict=True))
    else:
        root = default_state_root or (ROOT / "torch_cache")
        root.mkdir(parents=True, exist_ok=True)
        out["SPLITFUSION_EDGE_STATE_ROOT"] = str(root.resolve(strict=True))
    return out


def up_command(variables: Mapping[str, str]) -> list[str]:
    """``sudo VAR=... docker compose -f ... -f ... up -d --no-build --pull never ...``."""
    return ["sudo", *[f"{name}={variables[name]}" for name in COMPOSE_VARIABLES],
            "docker", "compose", "-f", COMPOSE_FILES[0], "-f", COMPOSE_FILES[1],
            *UP_ARGUMENTS]


def forbidden_operations(argv: Sequence[str]) -> list[str]:
    """Any build/pull operation in a docker command line (must be empty)."""
    words = [str(item) for item in argv if "=" not in str(item) or str(item).startswith("-")]
    found = []
    if "--build" in words or "build" in words:
        found.append("build")
    if "pull" in words or any(w.startswith("--pull") and w != "--pull" for w in words):
        found.append("pull")
    if "--pull" in words:
        index = words.index("--pull")
        if index + 1 >= len(words) or words[index + 1] != "never":
            found.append("pull-policy-not-never")
    if "tag" in words or "push" in words or "load" in words:
        found.append("retag/push/load")
    return found


# ---------------------------------------------------------------------------
# Docker identity checks (injectable runner)
# ---------------------------------------------------------------------------


def resolve_admitted_image(run: Runner = _run) -> dict[str, Any]:
    """Resolve the tag; require its full ID to be the admitted image ID."""
    completed = run(["sudo", "-n", "docker", "image", "inspect", IMAGE_TAG,
                     "--format", "{{json .}}"], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, timeout=30.0)
    require(completed.returncode == 0, f"admitted edge image {IMAGE_TAG} is missing")
    document = json.loads(completed.stdout)
    image_id = str(document.get("Id", ""))
    require(image_id == ADMITTED_IMAGE_ID,
            f"{IMAGE_TAG} resolves to {image_id}, not the admitted {ADMITTED_IMAGE_ID}")
    layers = list((document.get("RootFS") or {}).get("Layers") or ())
    config = document.get("Config") or {}
    return {
        "tag": IMAGE_TAG,
        "id": image_id,
        "admitted_id": ADMITTED_IMAGE_ID,
        "created": document.get("Created"),
        "repo_tags": list(document.get("RepoTags") or ()),
        "repo_digests": list(document.get("RepoDigests") or ()),
        "architecture": document.get("Architecture"),
        "os": document.get("Os"),
        "rootfs_type": (document.get("RootFS") or {}).get("Type"),
        "rootfs_layers": layers,
        "rootfs_layers_sha256": _canonical_sha256(layers),
        "config_entrypoint": config.get("Entrypoint"),
        "config_cmd": config.get("Cmd"),
        "config_sha256": _canonical_sha256(config),
        "resolved_at_unix_s": time.time(),
    }


def inspect_container(run: Runner = _run, *, state_root: Path, fcos_source: Path,
                      fcos_sha256: str) -> dict[str, Any]:
    """The created container must run the admitted image with the exact mounts."""
    completed = run(["sudo", "-n", "docker", "container", "inspect", CONTAINER,
                     "--format", "{{json .}}"], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, timeout=30.0)
    require(completed.returncode == 0, f"edge container {CONTAINER} was not created")
    document = json.loads(completed.stdout)
    image = str(document.get("Image", ""))
    require(image == ADMITTED_IMAGE_ID,
            f"edge container runs {image}, not the admitted {ADMITTED_IMAGE_ID}")
    mounts = list(document.get("Mounts") or ())

    def selected(destination: str) -> dict[str, Any]:
        rows = [row for row in mounts if row.get("Destination") == destination]
        require(len(rows) == 1, f"edge container mount is not unique: {destination}")
        return rows[0]

    repository = selected(REPOSITORY_DESTINATION)
    state = selected(STATE_DESTINATION)
    fcos = selected(FCOS_DESTINATION)
    require(Path(str(repository["Source"])).resolve(strict=True) == ROOT.resolve(strict=True)
            and bool(repository.get("RW")) is False,
            "edge repository mount source/mode drift")
    require(Path(str(state["Source"])).resolve(strict=True)
            == Path(state_root).resolve(strict=True) and bool(state.get("RW")) is True,
            "edge state mount source/mode drift")
    require(Path(str(fcos["Source"])).resolve(strict=True)
            == Path(fcos_source).resolve(strict=True) and bool(fcos.get("RW")) is False,
            "edge FCOS checkpoint mount source/mode drift")
    require(sha256_file(Path(str(fcos["Source"]))) == str(fcos_sha256),
            "edge FCOS checkpoint content drift")
    return {
        "container": CONTAINER,
        "container_id": document.get("Id"),
        "image": image,
        "config_image": (document.get("Config") or {}).get("Image"),
        "admitted_id": ADMITTED_IMAGE_ID,
        "state_running": (document.get("State") or {}).get("Running"),
        "mounts": {
            "repository": {"source": str(repository["Source"]),
                           "destination": REPOSITORY_DESTINATION, "rw": False},
            "state": {"source": str(state["Source"]), "destination": STATE_DESTINATION,
                      "rw": True},
            "fcos": {"source": str(fcos["Source"]), "destination": FCOS_DESTINATION,
                     "rw": False, "sha256": str(fcos_sha256)},
        },
        "inspected_at_unix_s": time.time(),
    }


def network_present(run: Runner = _run) -> bool:
    return run(["sudo", "docker", "network", "inspect", NETWORK], stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL, timeout=30.0).returncode == 0


def nvidia_runtime_probe(run: Runner = _run, *, log: Optional[IO[bytes]] = None,
                         sleep: Callable[[float], None] = time.sleep) -> bool:
    """The legacy structured, retried probe of Docker's runtime map."""
    backoff = 1.0
    for attempt in range(1, NVIDIA_PROBE_ATTEMPTS + 1):
        try:
            completed = run(["sudo", "docker", "info", "--format",
                             "{{range $name, $runtime := .Runtimes}}{{$name}}\n{{end}}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            timeout=NVIDIA_PROBE_TIMEOUT_S)
            output, rc = completed.stdout or "", completed.returncode
        except subprocess.TimeoutExpired:
            output, rc = "", 124
        names = [line.strip() for line in output.splitlines()]
        if log is not None:
            log.write(f"[phase6_edge_launch] nvidia runtime probe attempt {attempt}: "
                      f"rc={rc} runtimes={' '.join(names)}\n".encode())
        if rc == 0 and NVIDIA_RUNTIME in names:
            return True
        if attempt < NVIDIA_PROBE_ATTEMPTS:
            sleep(backoff)
            backoff = min(backoff * 2, 8.0)
    return False


def interface_exists(name: str) -> bool:
    return _run(["ip", "-br", "addr", "show", name], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=10.0).returncode == 0


def _write_create_only(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(document), sort_keys=True, indent=1, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch_no_build(env: Mapping[str, str], *, stdout: IO[bytes], timeout: float,
                    evidence_path: Path, fcos_source: Path, fcos_sha256: str,
                    run: Runner = _run,
                    interface_exists_fn: Callable[[str], bool] = interface_exists,
                    sleep: Callable[[float], None] = time.sleep) -> int:
    """Start the admitted edge image without build/pull; prove its identity.

    Returns the compose return code (0 on success) like the legacy launcher;
    raises :class:`EdgeImageError` on missing/drifted image or mounts.
    """
    deadline = time.monotonic() + float(timeout)
    evidence: dict[str, Any] = {
        "schema": "scenesense.run4_live_v2.phase6_edge_image_launch.v1",
        "addendum": "PHASE6_SETUP_REPAIR_ADDENDUM_3_NO_BUILD_EDGE",
        "admitted_image_id": ADMITTED_IMAGE_ID,
        "legacy_launcher_sha256": sha256_file(LEGACY_LAUNCHER),
        "legacy_launcher_invoked": False,
    }
    try:
        require(evidence["legacy_launcher_sha256"] == LEGACY_LAUNCHER_SHA256,
                "shared legacy launcher hash drift")
        require(network_present(run), f"{NETWORK} not found (core network is not up)")
        require(nvidia_runtime_probe(run, log=stdout, sleep=sleep),
                "Docker does not report the nvidia runtime")
        variables = compose_variables(env, config=config_env(),
                                      interface_exists=interface_exists_fn)
        evidence["pre_launch_tag_resolution"] = resolve_admitted_image(run)
        command = up_command(variables)
        require(not forbidden_operations(command), "launch command contains build/pull")
        evidence["compose_command"] = command
        evidence["compose_dir"] = str(COMPOSE_DIR)
        stdout.write(("[phase6_edge_launch] " + " ".join(
            [*COMPOSE_FILES, *UP_ARGUMENTS]) + "\n").encode())
        stdout.flush()
        completed = run(command, cwd=str(COMPOSE_DIR), stdout=stdout,
                        stderr=subprocess.STDOUT,
                        timeout=max(1.0, deadline - time.monotonic()))
        evidence["compose_returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            return int(completed.returncode)
        evidence["post_create_container"] = inspect_container(
            run, state_root=Path(variables["SPLITFUSION_EDGE_STATE_ROOT"]),
            fcos_source=fcos_source, fcos_sha256=fcos_sha256)
        evidence["verdict"] = "ADMITTED_IMAGE_LAUNCHED"
        return 0
    except BaseException as exc:
        evidence["verdict"] = "REFUSED"
        evidence["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_create_only(Path(evidence_path), evidence)


class _AdapterSubprocess:
    """``subprocess`` stand-in for ``adapter_direct_v1``: reroutes only the launcher."""

    def __init__(self, real: Any, launch: Callable[..., int]) -> None:
        self._real = real
        self._launch = launch

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def run(self, args: Any, *pos: Any, **kwargs: Any) -> Any:
        if isinstance(args, (list, tuple)) and [str(a) for a in args] == [str(LEGACY_LAUNCHER)]:
            code = self._launch(env=kwargs.get("env") or os.environ,
                                stdout=kwargs["stdout"], timeout=float(kwargs.get("timeout", 180.0)))
            return self._real.CompletedProcess(args, int(code))
        return self._real.run(args, *pos, **kwargs)


def install_adapter_launch_seam(adapter: Any, campaign: Mapping[str, Any],
                                evidence_path: Path) -> None:
    """Route ``adapter_direct_v1``'s legacy-launcher call through the no-build path."""
    record = campaign["deployment"]["fcos_constructor_weights"]
    fcos_source = (ROOT / str(record["path"])).resolve(strict=True)

    def launch(*, env: Mapping[str, str], stdout: IO[bytes], timeout: float) -> int:
        return launch_no_build(env, stdout=stdout, timeout=timeout,
                               evidence_path=evidence_path, fcos_source=fcos_source,
                               fcos_sha256=str(record["sha256"]))

    real = adapter.subprocess
    if isinstance(real, _AdapterSubprocess):
        real = real._real
    adapter.subprocess = _AdapterSubprocess(real, launch)


__all__ = [
    "ADMITTED_IMAGE_ID",
    "IMAGE_TAG",
    "COMPOSE_VARIABLES",
    "UP_ARGUMENTS",
    "EdgeImageError",
    "config_env",
    "compose_variables",
    "up_command",
    "forbidden_operations",
    "resolve_admitted_image",
    "inspect_container",
    "nvidia_runtime_probe",
    "launch_no_build",
    "install_adapter_launch_seam",
]
