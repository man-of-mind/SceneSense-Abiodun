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
        )

    def test_sequence_shapes_and_state(self) -> None:
        observations = torch.randn(2, 5, 12)
        mask = torch.ones(2, 5, 72, dtype=torch.bool)
        output = self.model(observations, action_mask=mask)
        self.assertEqual(tuple(output.logits.shape), (2, 5, 72))
        self.assertEqual(tuple(output.reward_value.shape), (2, 5))
        self.assertEqual(tuple(output.cost_values.shape), (2, 5, 2))
        self.assertEqual(self.model.cost_names, ("payload_bytes", "compute_ms"))
        self.assertEqual(tuple(output.next_snr_mean.shape), (2, 5))
        self.assertEqual(tuple(output.next_snr_std.shape), (2, 5))
        self.assertEqual(tuple(output.policy_forecast_features.shape), (2, 5, 2))
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

    def test_forecast_loss_masks_unavailable_future_window(self) -> None:
        observations = torch.randn(1, 3, 12)
        mask = torch.ones(1, 3, 72, dtype=torch.bool)
        output = self.model(observations, action_mask=mask)
        targets = torch.tensor([[20.0, float("nan"), 18.0]])
        valid = torch.tensor([[True, False, True]])
        loss = self.model.forecast_loss(output, targets, validity_mask=valid)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.model.forecast_loss(
                output,
                targets,
                validity_mask=torch.zeros_like(valid),
            )

    def test_forecast_mean_and_log_std_influence_current_policy(self) -> None:
        observation = torch.randn(1, 12)
        mask = torch.ones(1, 72, dtype=torch.bool)

        # Isolate the two forecast inputs from the direct LSTM features. Action
        # 0 reads only the forecast mean; action 1 reads only log(std).
        with torch.no_grad():
            self.model.policy_head.weight.zero_()
            self.model.policy_head.bias.zero_()
            self.model.policy_head.weight[0, self.model.hidden_dim] = 1.0
            self.model.policy_head.weight[1, self.model.hidden_dim + 1] = 1.0
            original_forecast_bias = self.model.next_snr_head.bias.clone()

        before = self.model(observation, action_mask=mask)
        torch.testing.assert_close(
            before.logits[..., 0],
            before.next_snr_mean,
        )
        torch.testing.assert_close(
            before.logits[..., 1],
            torch.log(before.next_snr_std),
        )

        with torch.no_grad():
            self.model.next_snr_head.bias[0].add_(2.0)
        changed_mean = self.model(observation, action_mask=mask)
        self.assertFalse(
            torch.equal(before.logits[..., 0], changed_mean.logits[..., 0])
        )
        torch.testing.assert_close(before.logits[..., 1], changed_mean.logits[..., 1])

        with torch.no_grad():
            self.model.next_snr_head.bias.copy_(original_forecast_bias)
            self.model.next_snr_head.bias[1].add_(2.0)
        changed_uncertainty = self.model(observation, action_mask=mask)
        torch.testing.assert_close(
            before.logits[..., 0], changed_uncertainty.logits[..., 0]
        )
        self.assertFalse(
            torch.equal(before.logits[..., 1], changed_uncertainty.logits[..., 1])
        )
        torch.testing.assert_close(
            changed_uncertainty.logits[..., 1],
            torch.log(changed_uncertainty.next_snr_std),
        )

    def test_policy_gradient_stops_at_forecast_head_but_trains_policy_columns(
        self,
    ) -> None:
        observation = torch.randn(1, 12)
        mask = torch.ones(1, 72, dtype=torch.bool)
        with torch.no_grad():
            self.model.next_snr_head.weight.zero_()
            self.model.next_snr_head.bias.copy_(torch.tensor([1.0, 0.0]))

        self.model.zero_grad(set_to_none=True)
        output = self.model(observation, action_mask=mask)
        output.logits.sum().backward()

        for parameter in self.model.next_snr_head.parameters():
            self.assertTrue(
                parameter.grad is None
                or torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
            )
        forecast_column_gradient = self.model.policy_head.weight.grad[
            :, self.model.hidden_dim :
        ]
        for column_gradient in forecast_column_gradient.unbind(dim=1):
            self.assertGreater(float(column_gradient.abs().sum()), 0.0)

    def test_stored_forecast_features_prevent_auxiliary_head_policy_drift(self) -> None:
        observation = torch.randn(1, 12)
        mask = torch.ones(1, 72, dtype=torch.bool)
        before = self.model(observation, action_mask=mask)
        stored = before.policy_forecast_features.clone()
        with torch.no_grad():
            self.model.next_snr_head.bias.add_(torch.tensor([5.0, 3.0]))
        recomputed = self.model(observation, action_mask=mask)
        replayed = self.model(
            observation,
            action_mask=mask,
            policy_forecast_features=stored,
        )
        self.assertFalse(torch.equal(before.logits, recomputed.logits))
        torch.testing.assert_close(before.logits, replayed.logits, rtol=0.0, atol=0.0)

    def test_future_observation_suffix_cannot_change_current_prefix(self) -> None:
        observations = torch.randn(2, 6, 12)
        changed_future = observations.clone()
        changed_future[:, 3:] = 100.0 * torch.randn_like(changed_future[:, 3:])
        mask = torch.ones(2, 6, 72, dtype=torch.bool)

        with torch.no_grad():
            original = self.model(observations, action_mask=mask)
            perturbed = self.model(changed_future, action_mask=mask)

        torch.testing.assert_close(
            original.logits[:, :3],
            perturbed.logits[:, :3],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            original.next_snr_mean[:, :3],
            perturbed.next_snr_mean[:, :3],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            original.next_snr_std[:, :3],
            perturbed.next_snr_std[:, :3],
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
