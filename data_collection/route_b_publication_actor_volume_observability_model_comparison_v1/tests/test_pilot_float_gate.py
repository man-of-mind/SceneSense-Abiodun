from __future__ import annotations

import math
import unittest

from ..run_comparison import pilot_float_comparison


class PilotFloatGateTest(unittest.TestCase):
    def test_one_ulp_passes_but_meaningful_geometry_difference_fails(self) -> None:
        registered = 0.18767888844013214
        one_ulp_parser_variant = math.nextafter(registered, math.inf)
        compatible = pilot_float_comparison(one_ulp_parser_variant, registered)
        self.assertTrue(compatible["passed"])
        self.assertGreater(compatible["absolute_difference"], 0.0)

        meaningful_geometry_variant = registered + 1e-6
        incompatible = pilot_float_comparison(meaningful_geometry_variant, registered)
        self.assertFalse(incompatible["passed"])
        self.assertGreater(incompatible["absolute_difference"], 1e-12)
