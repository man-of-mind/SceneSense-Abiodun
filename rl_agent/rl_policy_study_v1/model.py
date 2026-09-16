"""Initial split-only recurrent PPO network.

This module defines architecture only.  It does not authorize training and it
does not read campaign evidence.  The policy has 72 SPLIT actions, LSTM memory,
one reward-value critic, named byte/compute cost critics, and a causal channel
forecast head. The predicted next-100-ms radio-window mean SNR and uncertainty
influence the current action through a stop-gradient policy input; the true
future window is available only later as a forecast target. The stop-gradient
blocks the direct policy-loss path into the forecast-head parameters. The
encoder and LSTM remain shared, so policy and forecast optimization are still
coupled through that common trunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


COST_CRITIC_NAMES_V1 = ("payload_bytes", "compute_ms")


class RecurrentState(NamedTuple):
    hidden: Tensor
    cell: Tensor


@dataclass(frozen=True)
class PolicyOutput:
    logits: Tensor
    reward_value: Tensor
    cost_values: Tensor
    next_snr_mean: Tensor
    next_snr_std: Tensor
    policy_forecast_features: Tensor
    state: RecurrentState

    @property
    def next_snr_prediction(self) -> Tensor:
        """Compatibility alias for the forecast mean."""

        return self.next_snr_mean


class SplitOnlyRecurrentActorCritic(nn.Module):
    """Compact LSTM actor-critic for the registered 72-action SPLIT catalog."""

    def __init__(
        self,
        observation_dim: int,
        *,
        action_count: int = 72,
        encoder_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_count != 72:
            raise ValueError("v1 is bound to exactly 72 SPLIT actions")
        if encoder_dim <= 0 or hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        self.observation_dim = int(observation_dim)
        self.action_count = int(action_count)
        self.hidden_dim = int(hidden_dim)
        self.cost_names = COST_CRITIC_NAMES_V1
        self.cost_count = len(self.cost_names)
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.observation_dim),
            nn.Linear(self.observation_dim, encoder_dim),
            nn.SiLU(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.SiLU(),
        )
        self.memory = nn.LSTM(
            input_size=encoder_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.policy_head = nn.Linear(self.hidden_dim + 2, self.action_count)
        self.reward_value_head = nn.Linear(self.hidden_dim, 1)
        self.cost_value_head = nn.Linear(self.hidden_dim, self.cost_count)
        self.next_snr_head = nn.Linear(self.hidden_dim, 2)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecurrentState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        target_device = parameter.device if device is None else torch.device(device)
        target_dtype = parameter.dtype if dtype is None else dtype
        shape = (1, int(batch_size), self.hidden_dim)
        return RecurrentState(
            torch.zeros(shape, device=target_device, dtype=target_dtype),
            torch.zeros(shape, device=target_device, dtype=target_dtype),
        )

    def forward(
        self,
        observation: Tensor,
        *,
        action_mask: Tensor,
        state: RecurrentState | None = None,
        policy_forecast_features: Tensor | None = None,
    ) -> PolicyOutput:
        if observation.ndim == 2:
            observation = observation.unsqueeze(1)
        if observation.ndim != 3:
            raise ValueError("observation must have shape [B,D] or [B,T,D]")
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("observation feature dimension is invalid")
        if not observation.is_floating_point() or not torch.isfinite(observation).all():
            raise ValueError("observation must be finite floating point")
        batch, steps, _ = observation.shape
        if action_mask.ndim == 2 and steps == 1:
            action_mask = action_mask.unsqueeze(1)
        expected_mask_shape = (batch, steps, self.action_count)
        if tuple(action_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"action_mask must have shape {expected_mask_shape}, "
                f"got {tuple(action_mask.shape)}"
            )
        mask = action_mask.to(device=observation.device, dtype=torch.bool)
        if not mask.any(dim=-1).all():
            raise ValueError("every decision step must keep at least one action valid")
        if state is None:
            state = self.initial_state(
                batch,
                device=observation.device,
                dtype=observation.dtype,
            )
        else:
            expected_state_shape = (1, batch, self.hidden_dim)
            if tuple(state.hidden.shape) != expected_state_shape:
                raise ValueError("hidden-state shape is invalid")
            if tuple(state.cell.shape) != expected_state_shape:
                raise ValueError("cell-state shape is invalid")
        encoded = self.encoder(observation)
        memory, next_state = self.memory(encoded, (state.hidden, state.cell))
        forecast_raw = self.next_snr_head(memory)
        next_snr_mean = forecast_raw[..., 0]
        next_snr_std = nn.functional.softplus(forecast_raw[..., 1]) + 1e-4
        # The prediction influences this decision in the forward pass, while
        # policy gradients do not traverse this branch into ``next_snr_head``.
        # The encoder/LSTM trunk is still shared: PPO updates to that trunk can
        # change later forecasts. The head learns from the next 100-ms radio-window
        # mean SNR observed after this action through ``forecast_loss``.
        predicted_policy_input = torch.stack(
            (next_snr_mean, torch.log(next_snr_std)), dim=-1
        ).detach()
        if policy_forecast_features is None:
            forecast_policy_input = predicted_policy_input
        else:
            expected_forecast_shape = (batch, steps, 2)
            if tuple(policy_forecast_features.shape) != expected_forecast_shape:
                raise ValueError(
                    "policy_forecast_features must have shape "
                    f"{expected_forecast_shape}"
                )
            if not policy_forecast_features.is_floating_point() or not torch.isfinite(
                policy_forecast_features
            ).all():
                raise ValueError("stored policy forecast features must be finite")
            # PPO re-evaluation uses the exact detached forecast features stored
            # with the rollout, so an auxiliary forecast-head update cannot
            # silently alter the sampled policy distribution.
            forecast_policy_input = policy_forecast_features.to(
                device=memory.device,
                dtype=memory.dtype,
            ).detach()
        raw_logits = self.policy_head(
            torch.cat((memory, forecast_policy_input), dim=-1)
        )
        logits = raw_logits.masked_fill(~mask, torch.finfo(raw_logits.dtype).min)
        return PolicyOutput(
            logits=logits,
            reward_value=self.reward_value_head(memory).squeeze(-1),
            cost_values=self.cost_value_head(memory),
            next_snr_mean=next_snr_mean,
            next_snr_std=next_snr_std,
            policy_forecast_features=forecast_policy_input,
            state=RecurrentState(*next_state),
        )

    @staticmethod
    def distribution(output: PolicyOutput) -> Categorical:
        return Categorical(logits=output.logits)

    @staticmethod
    def forecast_loss(
        output: PolicyOutput,
        next_snr_target: Tensor,
        *,
        validity_mask: Tensor | None = None,
    ) -> Tensor:
        """Score valid later-observed next-100-ms-window mean-SNR targets.

        A missing or incomplete future radio window is not a numerical target.
        Callers must mark it invalid; invalid elements contribute no loss.
        """

        if next_snr_target.shape != output.next_snr_mean.shape:
            raise ValueError("next-SNR target shape is invalid")
        if validity_mask is None:
            valid = torch.ones_like(next_snr_target, dtype=torch.bool)
        else:
            if validity_mask.shape != next_snr_target.shape:
                raise ValueError("next-SNR validity mask shape is invalid")
            valid = validity_mask.to(device=next_snr_target.device, dtype=torch.bool)
        if not valid.any():
            raise ValueError("forecast loss requires at least one valid target")
        if not torch.isfinite(next_snr_target[valid]).all():
            raise ValueError("valid next-SNR targets must be finite")
        safe_target = torch.where(
            valid,
            next_snr_target,
            output.next_snr_mean.detach(),
        )
        normalized_residual = (
            safe_target - output.next_snr_mean
        ) / output.next_snr_std
        element_loss = (
            0.5 * normalized_residual.square() + torch.log(output.next_snr_std)
        )
        return element_loss[valid].mean()
