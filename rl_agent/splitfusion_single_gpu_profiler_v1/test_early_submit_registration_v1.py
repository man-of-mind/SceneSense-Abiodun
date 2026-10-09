"""The sealed early-submit registration binds the working tree (kept out of
``test_early_submit_v1.py`` because that file is itself bound)."""

from __future__ import annotations

import unittest
from pathlib import Path

from . import runner_v1 as RUN
from . import threadcap_v1 as TC

EXPERIMENT_PATH = Path(__file__).resolve().parent / "experiment_early_submit_v1.json"


class RegistrationTest(unittest.TestCase):
    @unittest.skipUnless(EXPERIMENT_PATH.is_file(), "early-submit registration not sealed yet")
    def test_registration_is_sealed_and_bound(self) -> None:
        registration = RUN.load_qualification(EXPERIMENT_PATH)
        self.assertEqual(set(registration["arms"]), {"C2", "E"})
        self.assertEqual(registration["arms"]["C2"]["ordering"], "serial")
        self.assertEqual(registration["arms"]["E"]["ordering"], "early")
        for arm in ("C2", "E"):
            self.assertEqual(registration["arms"][arm]["bundle"], "production")
            self.assertEqual(registration["arms"][arm]["env"], TC.TREATMENT)
            self.assertEqual(RUN.select_arm(registration, arm)["ordering"], TC.ORDERINGS[arm])
        RUN.verify_capture_binding(registration)
        bound = RUN.verify_experiment_binding(registration)       # working tree, no commit fallback
        for name in ("child_entry", "edge_ordering_module", "edge_ordering_entry",
                     "early_submit_tests", "runner_module", "threadcap_module"):
            self.assertIn(name, bound)
        self.assertEqual(registration["child_module"],
                         "rl_agent.splitfusion_single_gpu_profiler_v1.child_v1_threadcap")


if __name__ == "__main__":
    unittest.main()
