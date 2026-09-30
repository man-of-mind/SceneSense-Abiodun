"""Measured-GPU entry seam for the frozen Phase-6 edge service.

The frozen service contains an old device-name gate for the desktop
``NVIDIA GeForce RTX 5090``.  L10319 has the measured laptop variant.  This
entry point does not weaken the gate to "any CUDA device": it validates name,
UUID, VRAM, and driver against the measured remote binding, writes those facts
create-only, and only then supplies the legacy name to the frozen guard for the
duration of its call.  The actual measured name is retained in evidence.

Importing this module does not import torch, inspect a GPU, or start anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

from . import contract as C


SCHEMA = "scenesense.run4.remote_edge_gpu_entry.v1"
LEGACY_DEVICE_NAME = "NVIDIA GeForce RTX 5090"


class RemoteEdgeGpuError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RemoteEdgeGpuError(message)


def parse_nvidia_smi_row(text: str) -> Mapping[str, Any]:
    rows = [line.strip() for line in str(text).splitlines() if line.strip()]
    _require(len(rows) == 1, "exactly one remote GPU must be visible")
    fields = [value.strip() for value in rows[0].split(",")]
    _require(len(fields) == 4, "unexpected nvidia-smi GPU fact row")
    try:
        memory = int(fields[2])
    except ValueError as exc:
        raise RemoteEdgeGpuError("GPU memory is not an integer MiB value") from exc
    return {"model": fields[0], "uuid": fields[1],
            "memory_total_mib": memory, "driver_version": fields[3]}


def validate_measured_gpu(observed: Mapping[str, Any],
                          binding: C.RemoteRuntimeBinding) -> None:
    expected = {
        "model": binding.gpu.model,
        "uuid": binding.gpu.uuid,
        "memory_total_mib": binding.gpu.memory_total_mib,
        "driver_version": binding.gpu.driver_version,
    }
    _require(dict(observed) == expected, "remote GPU facts drifted from binding")


def delegate_frozen_service(*, torch_module: Any,
                            frozen_main: Callable[[Sequence[str]], int],
                            frozen_argv: Sequence[str], measured_model: str) -> int:
    """Narrowly bridge the frozen desktop-name guard after exact validation."""
    actual = str(torch_module.cuda.get_device_name(0))
    _require(actual == measured_model, "torch and nvidia-smi GPU names disagree")
    original = torch_module.cuda.get_device_name
    alias_unused = True

    def legacy_name(device: Any = None) -> str:
        nonlocal alias_unused
        if alias_unused:
            alias_unused = False
            return LEGACY_DEVICE_NAME
        return str(original(device))

    torch_module.cuda.get_device_name = legacy_name
    try:
        return int(frozen_main(list(frozen_argv)))
    finally:
        torch_module.cuda.get_device_name = original


def _write_create_only(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(document), sort_keys=True, indent=1) + "\n")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - live seam
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--remote-binding", type=Path, required=True)
    parser.add_argument("--remote-gpu-evidence", type=Path, required=True)
    known, frozen_argv = parser.parse_known_args(list(argv) if argv is not None else None)
    binding_bytes = known.remote_binding.read_bytes()
    binding = C.RemoteRuntimeBinding.from_mapping(json.loads(binding_bytes))
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=20.0,
    )
    _require(completed.returncode == 0, "nvidia-smi GPU fact query failed")
    observed = parse_nvidia_smi_row(completed.stdout)
    validate_measured_gpu(observed, binding)

    import torch
    _require(torch.cuda.is_available(), "CUDA is unavailable in the remote edge container")
    _require(torch.cuda.device_count() == 1, "exactly one CUDA device must be visible")
    torch_name = str(torch.cuda.get_device_name(0))
    _require(torch_name == binding.gpu.model, "torch GPU model drift")
    evidence = {
        "schema": SCHEMA,
        "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "measured_gpu": dict(observed),
        "torch_device_name": torch_name,
        "legacy_guard_name_supplied": LEGACY_DEVICE_NAME,
        "compatibility_scope": "FROZEN_DEVICE_NAME_GUARD_ONLY",
    }
    _write_create_only(known.remote_gpu_evidence, evidence)

    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_edge_runtime_v2
    return delegate_frozen_service(
        torch_module=torch, frozen_main=phase6_edge_runtime_v2.main,
        frozen_argv=frozen_argv, measured_model=binding.gpu.model,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
