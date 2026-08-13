import argparse
import unittest

from uplink_only_spatial_map_pipeline import (
    carla_fusion_staleness_scenario_uplink_only as collector,
)


class DualClockContractTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {"fps": 10.0, "world_tick_hz": 20.0, "sensor_every_tick": False}
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_twenty_hz_control_and_ten_hz_sensor_resolve_two_ticks(self):
        args = self._args()

        self.assertEqual(collector.resolved_world_tick_hz(args), 20.0)
        self.assertEqual(collector.synchronous_ticks_per_sensor_frame(args), 2)

    def test_non_integral_clock_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integral ratio"):
            collector.resolved_world_tick_hz(self._args(world_tick_hz=15.0))

    def test_every_tick_cannot_claim_a_lower_sensor_rate(self):
        with self.assertRaisesRegex(ValueError, "sensor-every-tick"):
            collector.resolved_world_tick_hz(self._args(sensor_every_tick=True))

    def test_radar_noise_seed_defaults_to_frozen_trajectory_seed(self):
        args = argparse.Namespace(seed=7002, radar_noise_seed=None)
        self.assertEqual(collector.resolved_radar_noise_seed(args), 7002)
        args.radar_noise_seed = 91
        self.assertEqual(collector.resolved_radar_noise_seed(args), 91)


if __name__ == "__main__":
    unittest.main()
