from __future__ import annotations

import numpy as np

from .run_quick_ablation import balanced_subset, evenly_spaced, raster_from_points


def test_evenly_spaced_is_deterministic_and_unique() -> None:
    assert evenly_spaced(list(range(10)), 4) == [0, 3, 6, 9]
    assert evenly_spaced(list(range(4)), 4) == [0, 1, 2, 3]


def test_balanced_subset() -> None:
    frames = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(7)]
    episodes = {frame: frame[0] for frame in frames}
    selected = balanced_subset(frames, episodes, 6)
    chosen = [frames[index] for index in selected]
    assert sum(value.startswith("a") for value in chosen) == 3
    assert sum(value.startswith("b") for value in chosen) == 3
    assert selected == sorted(selected)


def test_current_sweep_filter_changes_support() -> None:
    payload = {
        "u": np.asarray([2.0, 6.0], dtype=np.float32),
        "v": np.asarray([2.0, 6.0], dtype=np.float32),
        "camera_depth_m": np.asarray([10.0, 20.0], dtype=np.float32),
        "velocity_mps": np.zeros(2, dtype=np.float32),
        "stationary_age_s": np.zeros(2, dtype=np.float32),
        "valid_projection": np.ones(2, dtype=np.uint8),
        "sweep_offset": np.asarray([0, 1], dtype=np.uint8),
    }
    current = raster_from_points(payload, current_only=True)
    both = raster_from_points(payload, current_only=False)
    assert current.shape == (4, 432, 768)
    assert int(np.count_nonzero(both[0])) > int(np.count_nonzero(current[0]))
