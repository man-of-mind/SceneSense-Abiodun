from __future__ import annotations

import unittest

import torch

from rl_agent.rl_policy_study_v1.model import (
    SplitOnlyRecurrentActorCritic,
)


class SplitOnlyRecurrentActorCriticTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = SplitOnlyRecurrentActorCritic(
            observation_dim=12,
            encoder_dim=24,
            hidden_dim=16,
            cost_count=3,
        )

    def test_sequence_shapes_and_state(self) -> None:
        observations = torch.randn(2, 5, 12)
        mask = torch.ones(2, 5, 72, dtype=torch.bool)
        output = self.model(observations, action_mask=mask)
        self.assertEqual(tuple(output.logits.shape), (2, 5, 72))
        self.assertEqual(tuple(output.reward_value.shape), (2, 5))
        self.assertEqual(tuple(output.cost_values.shape), (2, 5, 3))
        self.assertEqual(tuple(output.next_snr_mean.shape), (2, 5))
        self.assertEqual(tuple(output.next_snr_std.shape), (2, 5))
        self.assertTrue((output.next_snr_std > 0).all())
        self.assertEqual(tuple(output.state.hidden.shape), (1, 2, 16))
        self.assertEqual(tuple(output.state.cell.shape), (1, 2, 16))

    def test_invalid_actions_have_zero_probability(self) -> None:
        observations = torch.randn(3, 12)
        mask = torch.zeros(3, 72, dtype=torch.bool)
        mask[:, [4, 50, 71]] = True
        output = self.model(observations, action_mask=mask)
        probabilities = self.model.distribution(output).probs[:, 0]
        self.assertTrue(
            torch.equal(
                probabilities[~mask],
                torch.zeros_like(probabilities[~mask]),
            )
        )
        sampled = self.model.distribution(output).sample((100,))
        self.assertTrue(torch.isin(sampled, torch.tensor([4, 50, 71])).all())

    def test_all_invalid_mask_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one action"):
            self.model(
                torch.randn(1, 12),
                action_mask=torch.zeros(1, 72, dtype=torch.bool),
            )

    def test_recurrent_state_changes_next_decision(self) -> None:
        observation = torch.randn(1, 12)
        mask = torch.ones(1, 72, dtype=torch.bool)
        first = self.model(observation, action_mask=mask)
        carried = self.model(observation, action_mask=mask, state=first.state)
        reset = self.model(observation, action_mask=mask)
        self.assertFalse(torch.equal(carried.logits, reset.logits))

    def test_forecast_loss_reaches_lstm(self) -> None:
        observations = torch.randn(2, 4, 12)
        mask = torch.ones(2, 4, 72, dtype=torch.bool)
        output = self.model(observations, action_mask=mask)
        loss = self.model.forecast_loss(output, torch.randn(2, 4))
        loss.backward()
        gradient = self.model.memory.weight_ih_l0.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_forecast_influences_current_policy_without_policy_gradient_leak(self) -> None:
        observation = torch.randn(1, 12)
        mask = torch.ones(1, 72, dtype=torch.bool)
        before = self.model(observation, action_mask=mask)
        with torch.no_grad():
            self.model.next_snr_head.bias[0].add_(2.0)
        after = self.model(observation, action_mask=mask)
        self.assertFalse(torch.equal(before.logits, after.logits))

        self.model.zero_grad(set_to_none=True)
        after.logits.sum().backward()
        gradient = self.model.next_snr_head.weight.grad
        self.assertTrue(gradient is None or torch.equal(gradient, torch.zeros_like(gradient)))


if __name__ == "__main__":
    unittest.main()
