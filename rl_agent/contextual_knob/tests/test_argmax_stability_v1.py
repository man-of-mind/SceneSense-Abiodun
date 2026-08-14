from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rl_agent.contextual_knob.argmax_stability_v1 import (
    add_utilities,
    cluster_bootstrap_mean,
    holm_adjust,
    profile_id,
)


class TaskAContractTests(unittest.TestCase):
    def test_profile_id_is_canonical(self) -> None:
        self.assertEqual(profile_id("ae32", "per_channel_uint4", 0.3), "ae32__uint4__roi0.3")

    def test_absent_class_weights_are_inactive(self) -> None:
        row = {
            "tp_ped": 0,
            "fn_ped": 0,
            "tp_veh": 1,
            "fn_veh": 0,
            **{f"conf_{i}{j}": 0 for i in range(3) for j in range(3)},
        }
        row.update({"conf_00": 10, "conf_11": 10})
        frame = add_utilities(
            pd.DataFrame([row]),
            {
                "weights": {"miou": 0.35, "pedestrian_recall": 0.40, "vehicle_recall": 0.25},
                "references": {"miou": 1.0, "pedestrian_recall": 1.0, "vehicle_recall": 1.0},
            },
        )
        self.assertAlmostEqual(float(frame.iloc[0]["utility_v5_frame"]), 1.0)

    def test_holm_adjust_is_monotone_in_order(self) -> None:
        adjusted = holm_adjust([0.01, 0.03, 0.2])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.2])

    def test_cluster_bootstrap_detects_positive_lift(self) -> None:
        differences = np.array([0.02, 0.03, 0.01, 0.04, 0.025, 0.035])
        groups = np.array(["a", "a", "b", "b", "c", "c"])
        low, high, p_value = cluster_bootstrap_mean(differences, groups, 2000, 7)
        self.assertGreater(low, 0.0)
        self.assertGreater(high, low)
        self.assertLess(p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
