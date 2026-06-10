#!/usr/bin/env python3

"""Small JSON-file transmission gate for SceneSense split-inference demos."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


ACTIVE_ALL = {"all", "both", "*", ""}


@dataclass(frozen=True)
class TxGateDecision:
    active: bool
    task_name: str
    active_task: str
    control_file: str
    updated_at: float
    reason: str
    profile: str = ""


class TxGate:
    """Read a controller-written JSON file and decide if a task may transmit."""

    def __init__(
        self,
        control_file: str,
        task_name: str,
        *,
        default_active: bool = True,
        stale_timeout_s: float = 0.0,
    ) -> None:
        self.control_file = str(control_file or "").strip()
        self.task_name = str(task_name or "").strip().lower()
        self.default_active = bool(default_active)
        self.stale_timeout_s = max(0.0, float(stale_timeout_s))

    @property
    def enabled(self) -> bool:
        return bool(self.control_file)

    def decide(self) -> TxGateDecision:
        if not self.enabled:
            return TxGateDecision(
                active=True,
                task_name=self.task_name,
                active_task="all",
                control_file="",
                updated_at=0.0,
                reason="gate_disabled",
            )

        path = Path(self.control_file).expanduser()
        if not path.exists():
            return TxGateDecision(
                active=self.default_active,
                task_name=self.task_name,
                active_task="missing",
                control_file=str(path),
                updated_at=0.0,
                reason="missing_control_file",
            )

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return TxGateDecision(
                active=self.default_active,
                task_name=self.task_name,
                active_task="invalid",
                control_file=str(path),
                updated_at=0.0,
                reason=f"invalid_control_file:{exc.__class__.__name__}",
            )

        if not isinstance(payload, dict):
            return TxGateDecision(
                active=self.default_active,
                task_name=self.task_name,
                active_task="invalid",
                control_file=str(path),
                updated_at=0.0,
                reason="invalid_control_payload",
            )

        active_task = str(payload.get("active_task", "all")).strip().lower()
        updated_at = _float(payload.get("updated_at", 0.0))
        profile = str(payload.get("profile", "") or "")
        if (
            self.stale_timeout_s > 0.0
            and updated_at > 0.0
            and time.time() - updated_at > self.stale_timeout_s
        ):
            return TxGateDecision(
                active=self.default_active,
                task_name=self.task_name,
                active_task=active_task,
                control_file=str(path),
                updated_at=updated_at,
                reason="stale_control_file",
                profile=profile,
            )

        active = active_task in ACTIVE_ALL or active_task == self.task_name
        return TxGateDecision(
            active=active,
            task_name=self.task_name,
            active_task=active_task,
            control_file=str(path),
            updated_at=updated_at,
            reason="match" if active else "muted_by_controller",
            profile=profile,
        )


def decision_to_stats(decision: TxGateDecision) -> Dict[str, object]:
    return {
        "tx_active": int(decision.active),
        "tx_task_name": decision.task_name,
        "tx_gate_active_task": decision.active_task,
        "tx_gate_reason": decision.reason,
        "tx_gate_profile": decision.profile,
    }


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

