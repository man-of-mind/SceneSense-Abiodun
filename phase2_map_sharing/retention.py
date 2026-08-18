"""Hard, non-destructive raw-artifact quotas for the paired pilot."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional


class RetentionQuotaExceeded(RuntimeError):
    """Raised before a raw write that would cross a registered quota."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RetentionLimits:
    maximum_window_seconds_per_trajectory: float
    maximum_raw_bytes_per_trajectory: int
    maximum_raw_bytes_pilot_total: int
    minimum_free_bytes_after_reservation: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "RetentionLimits":
        result = cls(
            maximum_window_seconds_per_trajectory=float(
                payload["maximum_window_seconds_per_trajectory"]
            ),
            maximum_raw_bytes_per_trajectory=int(
                payload["maximum_raw_bytes_per_trajectory"]
            ),
            maximum_raw_bytes_pilot_total=int(payload["maximum_raw_bytes_pilot_total"]),
            minimum_free_bytes_after_reservation=int(
                payload["minimum_free_bytes_after_reservation"]
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.maximum_window_seconds_per_trajectory <= 0.0:
            raise ValueError("maximum raw window duration must be positive")
        if min(
            self.maximum_raw_bytes_per_trajectory,
            self.maximum_raw_bytes_pilot_total,
            self.minimum_free_bytes_after_reservation,
        ) <= 0:
            raise ValueError("raw-retention byte quotas and reserve must be positive")
        if self.maximum_raw_bytes_per_trajectory > self.maximum_raw_bytes_pilot_total:
            raise ValueError("per-trajectory raw quota cannot exceed the pilot-total quota")


@dataclass
class _Window:
    started_at_s: float
    attempted_bytes: int = 0
    written_bytes: int = 0
    status: str = "active"
    stop_reason: Optional[str] = None


@dataclass(frozen=True)
class RawWritePermit:
    permit_id: int
    trajectory_id: str
    byte_count: int


class RawRetentionBudget:
    """Authorize every heavy-artifact write before it reaches the filesystem.

    The class never deletes data.  If a quota is reached it stops raw retention
    for that trajectory while allowing the caller's lightweight event logger to
    continue.
    """

    def __init__(
        self,
        output_root: Path,
        limits: RetentionLimits,
        *,
        free_bytes_provider: Optional[Callable[[Path], int]] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.limits = limits
        self.limits.validate()
        self._free_bytes_provider = free_bytes_provider or (
            lambda path: int(shutil.disk_usage(path).free)
        )
        self.windows: Dict[str, _Window] = {}
        self.total_attempted_bytes = 0
        self.total_written_bytes = 0
        self.preflight_complete = False
        self.expected_trajectory_count: Optional[int] = None
        self._next_permit_id = 1
        self._pending: Dict[int, RawWritePermit] = {}

    def preflight(self, trajectory_count: int) -> dict:
        if int(trajectory_count) <= 0:
            raise ValueError("trajectory_count must be positive")
        self.output_root.mkdir(parents=True, exist_ok=True)
        free_bytes = self._free_bytes_provider(self.output_root)
        required_free = (
            self.limits.minimum_free_bytes_after_reservation
            + self.limits.maximum_raw_bytes_pilot_total
        )
        if free_bytes < required_free:
            raise RetentionQuotaExceeded("insufficient_free_space_for_pilot_reservation")
        if (
            int(trajectory_count) * self.limits.maximum_raw_bytes_per_trajectory
            < self.limits.maximum_raw_bytes_pilot_total
        ):
            # This is allowed but must be visible: the global quota is then not
            # the binding allocation.
            allocation_note = "per_trajectory_quotas_bind_before_pilot_total"
        else:
            allocation_note = "pilot_total_may_bind"
        self.preflight_complete = True
        self.expected_trajectory_count = int(trajectory_count)
        return {
            "free_bytes": free_bytes,
            "reserved_raw_bytes": self.limits.maximum_raw_bytes_pilot_total,
            "minimum_free_bytes_after_reservation": (
                self.limits.minimum_free_bytes_after_reservation
            ),
            "allocation_note": allocation_note,
        }

    def start_window(self, trajectory_id: str, started_at_s: float) -> None:
        if not self.preflight_complete:
            raise RuntimeError("raw-retention preflight must pass before starting a window")
        key = str(trajectory_id).strip()
        if not key:
            raise ValueError("trajectory_id is required")
        if key in self.windows:
            raise ValueError("raw window already exists for trajectory")
        if self.expected_trajectory_count is None or len(self.windows) >= self.expected_trajectory_count:
            raise ValueError("raw windows exceed the preflight trajectory allocation")
        self.windows[key] = _Window(float(started_at_s))

    def authorize_write(
        self, trajectory_id: str, byte_count: int, at_s: float
    ) -> RawWritePermit:
        key = str(trajectory_id)
        if key not in self.windows:
            raise ValueError("raw window is not active for trajectory")
        window = self.windows[key]
        if window.status != "active":
            raise RetentionQuotaExceeded(window.stop_reason or "raw_window_not_active")
        amount = int(byte_count)
        if amount < 0:
            raise ValueError("byte_count must be nonnegative")
        window.attempted_bytes += amount
        self.total_attempted_bytes += amount
        pending_for_window = sum(
            permit.byte_count
            for permit in self._pending.values()
            if permit.trajectory_id == key
        )
        pending_total = sum(permit.byte_count for permit in self._pending.values())
        elapsed = float(at_s) - window.started_at_s
        if elapsed < -1e-12:
            self._stop(window, "raw_write_timestamp_precedes_window")
        if elapsed > self.limits.maximum_window_seconds_per_trajectory + 1e-12:
            self._stop(window, "maximum_window_duration_reached")
        if (
            window.written_bytes + pending_for_window + amount
            > self.limits.maximum_raw_bytes_per_trajectory
        ):
            self._stop(window, "maximum_trajectory_raw_bytes_reached")
        if (
            self.total_written_bytes + pending_total + amount
            > self.limits.maximum_raw_bytes_pilot_total
        ):
            self._stop(window, "maximum_pilot_raw_bytes_reached")
        free_bytes = self._free_bytes_provider(self.output_root)
        if (
            free_bytes - pending_total - amount
            < self.limits.minimum_free_bytes_after_reservation
        ):
            self._stop(window, "minimum_free_space_reserve_reached")
        permit = RawWritePermit(self._next_permit_id, key, amount)
        self._next_permit_id += 1
        self._pending[permit.permit_id] = permit
        return permit

    def record_write_complete(self, permit: RawWritePermit) -> None:
        """Account a successful write after a prior ``authorize_write`` call."""

        active = self._pending.get(permit.permit_id)
        if active != permit:
            raise ValueError("unknown, stale, or altered raw-write permit")
        key = permit.trajectory_id
        if key not in self.windows or self.windows[key].status != "active":
            raise RuntimeError("cannot record a write for an inactive raw window")
        amount = permit.byte_count
        window = self.windows[key]
        window.written_bytes += amount
        self.total_written_bytes += amount
        del self._pending[permit.permit_id]

    def cancel_write(self, permit: RawWritePermit) -> None:
        """Release a permit when the underlying write fails before completion."""

        active = self._pending.get(permit.permit_id)
        if active != permit:
            raise ValueError("unknown, stale, or altered raw-write permit")
        del self._pending[permit.permit_id]

    def finish_window(self, trajectory_id: str, reason: str = "completed") -> None:
        key = str(trajectory_id)
        if key not in self.windows:
            raise ValueError("unknown raw window")
        window = self.windows[key]
        if window.status == "active":
            if any(
                permit.trajectory_id == key for permit in self._pending.values()
            ):
                raise RuntimeError("cannot close a raw window with a pending write permit")
            window.status = "closed"
            window.stop_reason = str(reason)

    def _stop(self, window: _Window, reason: str) -> None:
        window.status = "quota_stopped"
        window.stop_reason = reason
        stopped_keys = {
            key for key, candidate in self.windows.items() if candidate is window
        }
        self._pending = {
            permit_id: permit
            for permit_id, permit in self._pending.items()
            if permit.trajectory_id not in stopped_keys
        }
        raise RetentionQuotaExceeded(reason)

    def summary(self) -> dict:
        return {
            "preflight_complete": self.preflight_complete,
            "total_attempted_bytes": self.total_attempted_bytes,
            "total_written_bytes": self.total_written_bytes,
            "pending_authorized_bytes": sum(
                permit.byte_count for permit in self._pending.values()
            ),
            "automatic_deletion_performed": False,
            "windows": {
                key: {
                    "started_at_s": item.started_at_s,
                    "attempted_bytes": item.attempted_bytes,
                    "written_bytes": item.written_bytes,
                    "status": item.status,
                    "stop_reason": item.stop_reason,
                }
                for key, item in sorted(self.windows.items())
            },
        }
