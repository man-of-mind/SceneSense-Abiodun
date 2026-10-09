"""Focused CPU-only checks of the ONE changed configuration (FADE_RECOVERY carrier cell); no OAI/CARLA/GPU.

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m unittest -v \
      rl_agent.splitfusion_run5_channel_response_v1.test_channel_selection
"""
import copy
import csv
import json
from pathlib import Path
import types
import unittest

from . import contract as C
from . import runner

ROOT = Path(__file__).resolve().parents[2]


class ChannelSelection(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / C.LIVE_CONFIG).read_bytes())

    def test_registration_binds_pinned_unchanged_inputs(self):
        reg = C.channel_registration(ROOT)
        self.assertEqual(reg["network_profile"], "FADE_RECOVERY")
        self.assertEqual(reg["trace"], C.TRACE)
        self.assertEqual(reg["sample_period_ms"], 100)
        self.assertEqual(reg["catch_up_policy"], "SKIP_OBSOLETE_NEVER_BURST")
        self.assertFalse(reg["transitions_depend_on_actor"])
        net = self.config["network"]
        for key, digest in (("design_config", "design_config_sha256"), ("traces_csv", "traces_sha256"),
                            ("mapping_csv", "mapping_sha256")):
            self.assertEqual(C.sha(ROOT / net[key]), net[digest])

    def test_pinned_generator_reproduces_the_registered_trace(self):
        from rl_agent import ue_target_snr_cell_runtime_v1 as base
        sequence, prefix, frozen = base.prepare_sequence(campaign=self.config, profile_id=C.NETWORK_PROFILE)
        self.assertEqual((frozen["trace_id"], int(frozen["seed"]), frozen["trace_sha256"]),
                         (C.TRACE["trace_id"], C.TRACE["seed"], C.TRACE["trace_sha256"]))
        with open(ROOT / self.config["network"]["traces_csv"], newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r["profile_id"] == C.NETWORK_PROFILE]
        self.assertEqual(len(prefix), len(rows))
        for (state, target), row in zip(prefix[:1600], rows[:1600]):
            self.assertEqual(int(state), int(row["state_index"]))
            self.assertAlmostEqual(float(target), float(row["target_snr_db"]), delta=5.1e-7)
        lo, hi = 5.5, 24.5    # qualified operational design bounds (network_profile_design_v2)
        self.assertTrue(all(lo - 1e-9 <= float(r["target_snr_db"]) <= hi + 1e-9 for r in rows[:1600]))

    def test_config_resolves_exactly_one_pinned_a71_fade_recovery_cell(self):
        from rl_agent.splitfusion_quality_feedback_probe_v1 import live_probe as LP
        cells = LP._registered_cells(self.config, action_ids=(C.CARRIER_ACTION,), profile_ids=(C.NETWORK_PROFILE,))
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual((int(cell.action_id), cell.network_profile_id, cell.trace_id, int(cell.seed)),
                         (71, "FADE_RECOVERY", C.TRACE["trace_id"], C.TRACE["seed"]))

    def test_runner_seam_substitutes_once_and_checks_the_cell(self):
        sealed = dict(network_profile=C.NETWORK_PROFILE, channel=C.channel_registration(ROOT))
        fake = types.SimpleNamespace(PROFILE="FAVORABLE_STABLE")
        check = runner.select_channel(fake, sealed, ROOT)
        self.assertEqual(fake.PROFILE, "FADE_RECOVERY")
        good = types.SimpleNamespace(action_id=71, network_profile_id="FADE_RECOVERY", trace_id=C.TRACE["trace_id"],
                                     seed=C.TRACE["seed"], cell_id="a71__fade_recovery")
        self.assertEqual(check([good])["cell_id"], "a71__fade_recovery")
        for bad in (dict(network_profile_id="FAVORABLE_STABLE"), dict(action_id=50), dict(seed=1)):
            with self.assertRaises(ValueError):
                check([types.SimpleNamespace(**dict(vars(good), **bad))])
        with self.assertRaises(ValueError):
            check([good, good])
        with self.assertRaises(ValueError):          # seam drift: never substitutes twice / from a foreign value
            runner.select_channel(fake, sealed, ROOT)

    def test_runner_seam_refuses_foreign_channel_registration(self):
        sealed = dict(network_profile=C.NETWORK_PROFILE, channel=C.channel_registration(ROOT))
        for mutate in (lambda s: s.update(network_profile="FAVORABLE_STABLE"),
                       lambda s: s["channel"]["trace"].update(seed=1),
                       lambda s: s.pop("channel")):
            value = copy.deepcopy(sealed)
            mutate(value)
            fake = types.SimpleNamespace(PROFILE="FAVORABLE_STABLE")
            with self.assertRaises(ValueError):
                runner.select_channel(fake, value, ROOT)
            self.assertEqual(fake.PROFILE, "FAVORABLE_STABLE")

    def test_real_runner_constant_is_the_substituted_original(self):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_live_runner_v2 as R
        self.assertEqual(R.PROFILE, C.ORIGINAL_PROFILE)
        self.assertEqual(R.FALLBACK_ACTION, C.CARRIER_ACTION)


if __name__ == "__main__":
    unittest.main()
