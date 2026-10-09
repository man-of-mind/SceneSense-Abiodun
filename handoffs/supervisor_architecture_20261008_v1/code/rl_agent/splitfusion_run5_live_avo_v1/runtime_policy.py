"""Process-local original Run-5 policy and causal-SNR seams for Phase 6.

Install after the Phase-6 runtime factory exists and before profiler seams.
The legacy lifecycle still owns the controller process and its restoration;
this module owns only the UE's short-lived Unix socket and policy overrides.
No imported module starts services, initializes CUDA, or accesses the radio.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
from typing import Any

from . import actor_binding as A
from . import snr_live as S


CONTROLLER_MODULE = "rl_agent.splitfusion_run5_live_avo_v1.snr_live"
LEGACY_CONTROLLER_MODULE = "rl_agent.splitfusion_live_dispatch_v1.live_pilot_target_snr_runtime"
SCHEMA = "scenesense.original_run5_live_policy_seams.v1"


class PolicySeamError(RuntimeError):
    pass


def _write_new(path: Path, body: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


class PolicySeams:
    """One attempt, one physical UE and controller; sessions may roll over.

    The provider receives ACKs before CARLA's runtime is constructed. Its UE
    identity is bound exactly once, to that runtime's actual attached OAI UE,
    before any actor state can be constructed. It never derives a UE identity
    from a guessed IP, RNTI, actor or profile.
    """
    def __init__(self, *, attempt_dir: Path, export_dir: Path,
                 controller_id: str | None = None) -> None:
        from rl_agent import ue_route_b_split_cell_adapter_v1 as pinned
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2 as FA
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import live_state_v2 as LS

        self.attempt_dir = Path(attempt_dir).resolve(strict=True)
        self.export_dir = Path(export_dir).resolve(strict=True)
        self.controller_id = str(uuid.UUID(controller_id)) if controller_id else str(uuid.uuid4())
        self.actor = A.load_source_export(self.export_dir)
        self._closed = False
        self._runtime_bound = False
        self._controller_started = False
        self._state_rows: list[dict[str, Any]] = []
        self._originals = [(FA, "load_registered_actor", FA.load_registered_actor),
                           (LS, "build_live_state", LS.build_live_state),
                           (pinned, "LivePilotCellRuntime", pinned.LivePilotCellRuntime),
                           (pinned, "start_target_snr", pinned.start_target_snr)]
        self.evidence_dir = self.attempt_dir / "run5_policy"
        self.evidence_dir.mkdir(exist_ok=False)
        # AF_UNIX paths are bounded to about 108 bytes: experiment roots are
        # much longer. mkdtemp also prevents a guessed/foreign socket reuse.
        self.socket_dir = Path(tempfile.mkdtemp(prefix="scenesense-r5-", dir="/tmp"))
        try:
            self.provider = S.LiveSnrProvider(self.socket_dir / "snr.sock", self.controller_id,
                                             "NOT_BOUND_TO_RUNTIME")
            self.builder = A.LiveStateBuilder(self.provider)
            _write_new(self.evidence_dir / "POLICY_INSTALL.json", {
                "schema": SCHEMA, "actor_tree_sha256": self.actor.boundary_sha256,
                "weights_sha256": A.WEIGHTS_SHA256, "state_dim": 22,
                "selection_scope": A.SELECTION_SCOPE,
                "feature_schema_sha256": A.FEATURE_SCHEMA_SHA256,
                "controller_uuid": self.controller_id, "socket_path": str(self.provider.path),
                "socket_uid": self.provider.path.stat().st_uid,
                "process_uid": os.getuid(), "clock_domain": S.CLOCK_DOMAIN,
                "lease_max_age_ns": S.LEASE_NS,
                "action_deadline_ns": 170_000_000,
                "quality_in_state": True, "quality_in_reward": True,
                "source_export": str(self.export_dir),
            })
        except BaseException:
            if hasattr(self, "provider"):
                self.provider.close()
            self.socket_dir.rmdir()
            raise
        original_factory = pinned.LivePilotCellRuntime

        def factory(**kwargs):
            if self._closed or self._runtime_bound:
                raise PolicySeamError("exactly one runtime may bind this SNR provider")
            runtime = original_factory(**kwargs)
            label = runtime.provider.ue_label
            if not isinstance(label, str) or not label or runtime.engine.ue_id != label:
                raise PolicySeamError("runtime has no unambiguous attached UE identity")
            if runtime.actor.boundary_sha256 != A.ACTOR_TREE_SHA256:
                raise PolicySeamError("runtime did not load the selected original Run-5 actor")
            self.provider.ue_id = label
            self._runtime_bound = True
            return runtime

        def build_state(**kwargs):
            if self._closed or not self._runtime_bound:
                raise PolicySeamError("SNR provider must bind to runtime before state construction")
            identity, boundary = kwargs["identity"], kwargs["boundary"]
            self.provider.adopt_session(identity.session_uuid)
            observation = self.provider.observe(boundary)
            self.builder.last_observation = observation
            row = {
                "frame_id": int(kwargs["camera_si"].metadata.identity.sample_seq),
                "decision_identity": dataclasses.asdict(identity),
                "boundary": dataclasses.asdict(boundary),
                "snr": dataclasses.asdict(observation), "admitted": False,
            }
            # Bounded by the registered transmitted frame budget in production.
            self._state_rows.append(row)
            try:
                guarded, features = A.build_run5_live_state(snr=observation, **kwargs)
            except Exception as exc:
                row["refusal"] = f"{type(exc).__name__}: {exc}"
                raise
            row["admitted"] = True
            row["snr_scaled"] = features.as_tuple()[21]
            return guarded, features

        def start_target(campaign, *, campaign_path, profile_id, temporary_dir, start_file):
            if self._closed or self._controller_started:
                raise PolicySeamError("exactly one controller may start per provider")
            runtime = pinned.repo_path(str(campaign["runtime"]["target_snr_runtime"]))
            try:
                module = ".".join(runtime.relative_to(pinned.ROOT).with_suffix("").parts)
            except ValueError as exc:
                raise PolicySeamError("target-SNR runtime escaped repository") from exc
            if module != LEGACY_CONTROLLER_MODULE:
                raise PolicySeamError("legacy controller module identity drift")
            if not self.provider.path.is_socket() or self.provider.path.stat().st_uid != os.getuid():
                raise PolicySeamError("causal SNR receiver must exist with the UE process UID")
            output = Path(temporary_dir) / "radio_trace.csv"
            stop_file = Path(temporary_dir) / "stop_target_snr"
            argv = [sys.executable, "-m", CONTROLLER_MODULE, "--controller",
                    "--snr-socket", str(self.provider.path), "--snr-controller-id", self.controller_id,
                    "--snr-audit", str(self.evidence_dir / "CONTROLLER_ACKS.jsonl"),
                    "--campaign", str(campaign_path), "--profile-id", str(profile_id),
                    "--output", str(output), "--stop-file", str(stop_file),
                    "--start-file", str(start_file)]
            _write_new(self.evidence_dir / "CONTROLLER_LAUNCH.json", {
                "schema": SCHEMA, "argv": argv, "cwd": str(pinned.ROOT),
                "launcher_uid": os.getuid(), "provider_uid": self.provider.path.stat().st_uid,
            })
            process = subprocess.Popen(argv, cwd=str(pinned.ROOT), stdin=subprocess.DEVNULL)
            self._controller_started = True
            return process, output, stop_file

        # FA is imported dynamically inside the already-built runtime class;
        # LS is likewise looked up per decision. Profiler wraps these next.
        FA.load_registered_actor = lambda: self.actor
        LS.build_live_state = build_state
        pinned.LivePilotCellRuntime = factory
        pinned.start_target_snr = start_target

    def evidence(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "actor_tree_sha256": self.actor.boundary_sha256,
                "weights_sha256": A.WEIGHTS_SHA256, "state_dim": 22,
                "runtime_bound": self._runtime_bound,
                "controller_started": self._controller_started,
                "provider": self.provider.evidence(), "decision_snr": list(self._state_rows)}

    def close(self) -> None:
        """Call after the legacy lifecycle has stopped/restored its controller."""
        if self._closed:
            return
        self._closed = True
        try:
            self.provider.close()
            _write_new(self.evidence_dir / "SNR_EVIDENCE.json", self.evidence())
        finally:
            for module, name, original in reversed(self._originals):
                setattr(module, name, original)
            # rmdir refuses to remove anything if a foreign file appeared.
            self.socket_dir.rmdir()


def install_policy_seams(*, attempt_dir: Path, export_dir: Path,
                         controller_id: str | None = None) -> PolicySeams:
    return PolicySeams(attempt_dir=attempt_dir, export_dir=export_dir,
                       controller_id=controller_id)
