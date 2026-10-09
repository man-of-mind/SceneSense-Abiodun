"""The frozen phase table matches the pinned FADE_RECOVERY design states (CPU only)."""
import csv
import json
from pathlib import Path
import unittest

from . import definitions as D

ROOT = Path(__file__).resolve().parents[2]


class Definitions(unittest.TestCase):
    def setUp(self):
        config = json.loads((ROOT / "rl_agent/configs/splitfusion_direct_edge_map_live_validation_v1.json").read_bytes())
        with open(ROOT / config["network"]["traces_csv"], newline="") as stream:
            self.states = [r["state"] for r in csv.DictReader(stream) if r["profile_id"] == "FADE_RECOVERY"]

    def runs(self, a, b):
        return set(self.states[a:b])

    def test_phase_states(self):
        s = self.states
        self.assertEqual(self.runs(361, 460), {"FAVORABLE"})
        self.assertEqual(s[360], "INTERMEDIATE")
        self.assertNotIn("FAVORABLE", self.runs(658, 972))
        self.assertEqual(s[657], "FAVORABLE")
        self.assertEqual(self.runs(972, 1114), {"FAVORABLE"})
        self.assertEqual(s[971], "INTERMEDIATE")
        self.assertEqual(s[1114], "INTERMEDIATE")
        self.assertEqual((s[745], s[746], s[914], s[915]), ("INTERMEDIATE", "ADVERSE", "ADVERSE", "INTERMEDIATE"))
        self.assertEqual(self.runs(746, 788), {"ADVERSE"})
        self.assertEqual(s[1261], "ADVERSE")

    def test_phases_partition_and_transitions_are_state_changes(self):
        for a, b in zip(D.PHASES, D.PHASES[1:]):
            self.assertEqual(a["end"], b["start"])
        changes = {t["step"] for t in D.transition_steps(self.states[:1400])}
        for t in D.PRIMARY_TRANSITIONS:
            self.assertIn(t["step"], changes)


if __name__ == "__main__":
    unittest.main()
