"""CPU-only tests: accepted SNR kernel, joint channel causality, duration, resume."""

from __future__ import annotations

import os
import random
import unittest
from pathlib import Path

from rl_agent.splitfusion_hybrid_sac_run4_v1 import mcs_transition_provider as P
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_channel as J
from rl_agent.splitfusion_hybrid_sac_run5_v1 import successor_mcs_snr_audit as SA

EVIDENCE_ROOT = Path(os.environ.get(
    "RUN5_EVIDENCE_ROOT", Path(__file__).resolve().parents[3] / "abiodun"))


class KernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence, cls.base, cls.kernel, cls.rows, _, _ = SA.fit_all(EVIDENCE_ROOT)

    def test_accepted_kernel_reproduces_the_committed_audit(self) -> None:
        kernel = J.load_accepted_kernel(EVIDENCE_ROOT)
        self.assertEqual(kernel.document(), self.kernel.document())
        self.assertGreater(kernel.theta, 0.0)
        self.assertEqual(self.base.binding_sha256, SA.REGISTERED_RUN4_KERNEL_BINDING)

    def test_theta_zero_is_exactly_run4(self) -> None:
        flat = SA.SnrTiltedKernelV1(self.base, 0.0, self.kernel.center_by_mcs,
                                    self.kernel.global_center)
        for mcs in range(P.MCS_MIN, P.MCS_MAX + 1):
            for snr in (5.5, 15.0, 24.5):
                self.assertEqual(flat.probabilities(mcs, snr), self.base.probabilities(mcs))

    def test_successor_mcs_is_stochastically_monotone_in_snr(self) -> None:
        grid = [5.5 + 0.5 * i for i in range(39)]
        for mcs in range(P.MCS_MIN, P.MCS_MAX + 1):
            previous_cdf = None
            previous_mean = None
            for snr in grid:
                probs = self.kernel.probabilities(mcs, snr)
                cdf, total = [], 0.0
                for value in probs:
                    total += value
                    cdf.append(total)
                mean = sum((P.MCS_MIN + i) * p for i, p in enumerate(probs))
                if previous_cdf is not None:
                    for a, b in zip(cdf, previous_cdf):
                        self.assertLessEqual(a, b + 1e-12)
                    self.assertGreaterEqual(mean, previous_mean - 1e-12)
                previous_cdf, previous_mean = cdf, mean
            self.assertGreater(previous_mean,
                               sum((P.MCS_MIN + i) * p for i, p in
                                   enumerate(self.kernel.probabilities(mcs, 5.5))))

    def test_temporally_shuffled_snr_loses_the_predictive_gain(self) -> None:
        validation = [r for r in self.rows if r["partition"] == "INTERNAL_VALIDATION"]
        real = SA.evaluate(self.base, self.kernel, validation)["pooled"]["nll_gain"]
        self.assertGreater(real, 0.3)
        rng = random.Random(17)
        gains = []
        for _ in range(20):
            permuted = SA.shuffle_within(self.rows, rng)
            kernel = SA.fit_tilted(self.base, [r for r in permuted if r["partition"] == "FIT"])
            gains.append(SA.evaluate(
                self.base, kernel,
                [r for r in permuted if r["partition"] == "INTERNAL_VALIDATION"])["pooled"]["nll_gain"])
        self.assertLess(max(gains), 0.1 * real)

    def test_fit_uses_fit_transitions_only(self) -> None:
        fit = [r for r in self.rows if r["partition"] == "FIT"]
        self.assertEqual(len(fit), 416)
        poisoned = [dict(r) for r in self.rows]
        for r in poisoned:
            if r["partition"] == "INTERNAL_VALIDATION":
                r["snr_db"] = 5.5
        refit = SA.fit_tilted(self.base, [r for r in poisoned if r["partition"] == "FIT"])
        self.assertEqual(refit.document(), self.kernel.document())


class ChannelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel = J.load_accepted_kernel(EVIDENCE_ROOT)
        cls.design = J.load_design()

    def channel(self, seed=17):
        return J.JointSnrMcsChannelV1(kernel=self.kernel, design=self.design, seed=seed)

    def test_exact_duration_advancement_in_100ms_ticks(self) -> None:
        channel = self.channel()
        for _ in range(50):
            observation = channel.observe()
            step = channel.advance(2)
            self.assertEqual(step.current_tick, observation.tick)
            self.assertEqual(step.generated_ticks, (observation.tick + 1, observation.tick + 2))
            self.assertEqual(step.successor_tick, observation.tick + 2)
        for bad in (1, 3, 0):
            channel.observe()
            with self.assertRaises(J.ChannelError):
                channel.advance(bad)

    def test_future_is_generated_only_after_observation(self) -> None:
        channel = self.channel()
        with self.assertRaises(J.ChannelError):
            channel.advance(2)            # must observe first
        observation = channel.observe()
        step = channel.advance(2)
        self.assertEqual((step.current_snr_db, step.current_mcs),
                         (observation.snr_db, observation.mcs))
        self.assertTrue(all(t > observation.tick for t in step.generated_ticks))
        # Observing again cannot reveal anything but the new current sample.
        again = channel.observe()
        self.assertEqual((again.snr_db, again.mcs, again.tick),
                         (step.successor_snr_db, step.successor_mcs, step.successor_tick))
        self.assertEqual(channel.future_sample_violations, 0)

    def test_actor_surface_has_no_hidden_profile_or_state(self) -> None:
        fields = set(J.ChannelObservationV1.__dataclass_fields__)
        self.assertEqual(fields, {"snr_db", "mcs", "tick"})
        a, b = self.channel(), self.channel()
        b._profile = 1 - a._profile
        oa, ob = a.observe(), b.observe()
        self.assertEqual((oa.snr_db, oa.mcs), (ob.snr_db, ob.mcs))

    def test_support_and_segment_profile_redraw(self) -> None:
        channel = self.channel()
        profiles = set()
        for _ in range(1500):
            observation = channel.observe()
            self.assertTrue(5.5 <= observation.snr_db <= 24.5)
            self.assertTrue(P.MCS_MIN <= observation.mcs <= P.MCS_MAX)
            profiles.add(channel._profile)
            channel.advance(2)
        self.assertEqual(profiles, {0, 1})

    def test_checkpoint_restore_is_exact(self) -> None:
        a = self.channel()
        for _ in range(123):
            a.observe()
            a.advance(2)
        document = a.checkpoint()
        b = self.channel()
        b.restore(document)
        for _ in range(200):
            oa, ob = a.observe(), b.observe()
            self.assertEqual((oa.snr_db, oa.mcs, oa.tick), (ob.snr_db, ob.mcs, ob.tick))
            a.advance(2)
            b.advance(2)
        with self.assertRaises(J.ChannelError):
            self.channel(seed=18).restore(document)

    def test_snr_changes_only_the_successor_mcs_distribution(self) -> None:
        low = self.kernel.probabilities(18, 8.0)
        high = self.kernel.probabilities(18, 22.0)
        self.assertNotEqual(low, high)
        mean = lambda probs: sum((P.MCS_MIN + i) * p for i, p in enumerate(probs))
        self.assertGreater(mean(high), mean(low))


if __name__ == "__main__":
    unittest.main()
