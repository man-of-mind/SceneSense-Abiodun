"""Canonical Track A profile and flattened action catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import REPO_ROOT


RETAINED_PROFILES = (
    "ae32__uint4__roi0.5",
    "ae64__uint4__roi0.5",
    "ae32__uint4__roi0.3",
    "ae64__uint4__roi0.3",
    "ae128__uint4__roi0.5",
    "ae32__uint4__roi0.0",
    "ae128__uint4__roi0.0",
)


@dataclass(frozen=True)
class Action:
    action_id: str
    mode: str
    profile_id: Optional[str] = None
    target_fps: int = 0
    payload_kib: float = 0.0
    roi_q: float = 0.0
    miou: float = 0.0
    pedestrian_recall: float = 0.0
    vehicle_recall: float = 0.0
    object_recall: float = 0.0
    base_loc_m: float = 0.0
    front_ms: float = 0.0
    back_ms: float = 0.0
    core_tier: bool = False

    @property
    def offered_mbps(self) -> float:
        return self.payload_kib * 1024.0 * 8.0 * self.target_fps / 1_000_000.0


def load_profile_catalog(path: str | Path) -> pd.DataFrame:
    catalog_path = Path(path)
    if not catalog_path.is_absolute():
        catalog_path = REPO_ROOT / catalog_path
    frame = pd.read_csv(catalog_path)
    missing = set(RETAINED_PROFILES) - set(frame["profile_id"])
    if missing:
        raise ValueError(f"canonical action catalog is missing profiles: {sorted(missing)}")
    if len(frame) != len(RETAINED_PROFILES):
        raise ValueError(f"expected {len(RETAINED_PROFILES)} retained profiles, found {len(frame)}")
    return frame


def _is_core(profile: Dict[str, float], preferred_core_kib: int) -> bool:
    if float(profile["roi_q"]) != 0.0:
        return False
    if preferred_core_kib == 90:
        return float(profile["payload_kib"]) >= 89.5
    return float(profile["payload_kib"]) >= 128.5


def flatten_actions(
    profiles: pd.DataFrame,
    fps_values: Iterable[int],
    preferred_core_kib: int,
) -> List[Action]:
    actions: List[Action] = [Action(action_id="SKIP", mode="SKIP")]
    rows = profiles.set_index("profile_id").loc[list(RETAINED_PROFILES)].reset_index()
    for row in rows.to_dict(orient="records"):
        for fps in fps_values:
            profile_id = str(row["profile_id"])
            actions.append(
                Action(
                    action_id=f"SPLIT::{profile_id}::{int(fps)}fps",
                    mode="SPLIT",
                    profile_id=profile_id,
                    target_fps=int(fps),
                    payload_kib=float(row["payload_kib"]),
                    roi_q=float(row["roi_q"]),
                    miou=float(row["miou"]),
                    pedestrian_recall=float(row["pedestrian_recall"]),
                    vehicle_recall=float(row["vehicle_recall"]),
                    object_recall=float(row["object_recall"]),
                    base_loc_m=float(row["base_loc_calibrated_m"]),
                    front_ms=float(row["front_ms"]),
                    back_ms=float(row["back_ms"]),
                    core_tier=_is_core(row, preferred_core_kib),
                )
            )
    if len(actions) != 36:
        raise AssertionError(f"Track A must have exactly 36 actions, got {len(actions)}")
    return actions
