"""Initial split-only recurrent, factorized PPO actor-critic network.

This is an architecture component, not a trainer or deployment authority.
The only mask accepted here is the runtime technical-executability mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


FAMILY_COUNT = 4
QUANTIZER_COUNT = 3
Q_COUNT = 6
ACTION_COUNT = FAMILY_COUNT * QUANTIZER_COUNT * Q_COUNT


@dataclass(frozen=True)
class PolicyAction:
    action_id: Tensor
    family_index: Tensor
    quantizer_index: Tensor
    q_index: Tensor
    log_probability: Tensor
    entropy: Tensor


def action_id_from_factors(family: Tensor, quantizer: Tensor, q_index: Tensor) -> Tensor:
    return family * (QUANTIZER_COUNT * Q_COUNT) + quantizer * Q_COUNT + q_index


def factors_from_action_id(action_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if torch.any(action_id < 0) or torch.any(action_id >= ACTION_COUNT):
        raise ValueError("action_id is outside 0..71")
    family = torch.div(action_id, QUANTIZER_COUNT * Q_COUNT, rounding_mode="floor")
    remainder = action_id % (QUANTIZER_COUNT * Q_COUNT)
    quantizer = torch.div(remainder, Q_COUNT, rounding_mode="floor")
    q_index = remainder % Q_COUNT
    return family, quantizer, q_index


def _masked_categorical(logits: Tensor, mask: Tensor) -> Categorical:
    if logits.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} does not match logits {logits.shape}")
    if mask.dtype != torch.bool:
        raise ValueError("technical action mask must have boolean dtype")
    if torch.any(~mask.any(dim=-1)):
        raise ValueError("every policy row must retain at least one executable option")
    return Categorical(logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))


class RecurrentFactorizedSplitActorCritic(nn.Module):
    """GRU actor-critic over the measured 4 x 3 x 6 split catalog.

    The factorization is autoregressive, so it can represent any distribution
    over the 72 actions while retaining the catalog's engineering structure.
    """

    def __init__(
        self,
        observation_dim: int = 7,
        encoder_dim: int = 64,
        recurrent_dim: int = 128,
        family_embedding_dim: int = 8,
        quantizer_embedding_dim: int = 4,
        cost_value_count: int = 2,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or cost_value_count <= 0:
            raise ValueError("observation_dim and cost_value_count must be positive")
        self.observation_dim = observation_dim
        self.recurrent_dim = recurrent_dim
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, encoder_dim),
            nn.Tanh(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.Tanh(),
        )
        self.memory = nn.GRU(encoder_dim, recurrent_dim, batch_first=True)
        self.family_embedding = nn.Embedding(FAMILY_COUNT, family_embedding_dim)
        self.quantizer_embedding = nn.Embedding(
            QUANTIZER_COUNT, quantizer_embedding_dim
        )
        self.family_head = nn.Linear(recurrent_dim, FAMILY_COUNT)
        self.quantizer_head = nn.Linear(
            recurrent_dim + family_embedding_dim, QUANTIZER_COUNT
        )
        self.q_head = nn.Linear(
            recurrent_dim + family_embedding_dim + quantizer_embedding_dim,
            Q_COUNT,
        )
        self.reward_value_head = nn.Linear(recurrent_dim, 1)
        self.cost_value_head = nn.Linear(recurrent_dim, cost_value_count)

    def encode_sequence(
        self, observations: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                "observations must have shape [batch, time, observation_dim]"
            )
        encoded = self.encoder(observations)
        recurrent, next_hidden = self.memory(encoded, hidden)
        reward_value = self.reward_value_head(recurrent).squeeze(-1)
        cost_values = self.cost_value_head(recurrent)
        return recurrent, next_hidden, reward_value, cost_values

    def _distributions(
        self,
        recurrent_state: Tensor,
        technical_mask: Tensor,
        family: Tensor | None = None,
        quantizer: Tensor | None = None,
    ) -> tuple[Categorical, Categorical | None, Categorical | None]:
        if recurrent_state.ndim != 2:
            raise ValueError("recurrent_state must have shape [batch, hidden]")
        expected = (recurrent_state.shape[0], FAMILY_COUNT, QUANTIZER_COUNT, Q_COUNT)
        if technical_mask.shape != expected:
            raise ValueError(f"technical_mask must have shape {expected}")
        family_mask = technical_mask.any(dim=(2, 3))
        family_dist = _masked_categorical(self.family_head(recurrent_state), family_mask)
        if family is None:
            return family_dist, None, None

        batch = torch.arange(recurrent_state.shape[0], device=recurrent_state.device)
        family_context = self.family_embedding(family)
        quantizer_logits = self.quantizer_head(
            torch.cat((recurrent_state, family_context), dim=-1)
        )
        quantizer_mask = technical_mask[batch, family].any(dim=-1)
        quantizer_dist = _masked_categorical(quantizer_logits, quantizer_mask)
        if quantizer is None:
            return family_dist, quantizer_dist, None

        quantizer_context = self.quantizer_embedding(quantizer)
        q_logits = self.q_head(
            torch.cat((recurrent_state, family_context, quantizer_context), dim=-1)
        )
        q_mask = technical_mask[batch, family, quantizer]
        q_dist = _masked_categorical(q_logits, q_mask)
        return family_dist, quantizer_dist, q_dist

    def choose(
        self,
        recurrent_state: Tensor,
        technical_mask: Tensor,
        deterministic: bool = False,
    ) -> PolicyAction:
        family_dist, _, _ = self._distributions(recurrent_state, technical_mask)
        family = (
            family_dist.logits.argmax(dim=-1)
            if deterministic
            else family_dist.sample()
        )
        _, quantizer_dist, _ = self._distributions(
            recurrent_state, technical_mask, family=family
        )
        assert quantizer_dist is not None
        quantizer = (
            quantizer_dist.logits.argmax(dim=-1)
            if deterministic
            else quantizer_dist.sample()
        )
        _, _, q_dist = self._distributions(
            recurrent_state, technical_mask, family=family, quantizer=quantizer
        )
        assert q_dist is not None
        q_index = q_dist.logits.argmax(dim=-1) if deterministic else q_dist.sample()
        return PolicyAction(
            action_id=action_id_from_factors(family, quantizer, q_index),
            family_index=family,
            quantizer_index=quantizer,
            q_index=q_index,
            log_probability=(
                family_dist.log_prob(family)
                + quantizer_dist.log_prob(quantizer)
                + q_dist.log_prob(q_index)
            ),
            entropy=(
                family_dist.entropy() + quantizer_dist.entropy() + q_dist.entropy()
            ),
        )

    def evaluate_actions(
        self, recurrent_state: Tensor, technical_mask: Tensor, action_id: Tensor
    ) -> tuple[Tensor, Tensor]:
        family, quantizer, q_index = factors_from_action_id(action_id)
        batch = torch.arange(recurrent_state.shape[0], device=recurrent_state.device)
        if torch.any(~technical_mask[batch, family, quantizer, q_index]):
            raise ValueError("an evaluated action is excluded by its technical mask")
        family_dist, quantizer_dist, q_dist = self._distributions(
            recurrent_state,
            technical_mask,
            family=family,
            quantizer=quantizer,
        )
        assert quantizer_dist is not None and q_dist is not None
        log_probability = (
            family_dist.log_prob(family)
            + quantizer_dist.log_prob(quantizer)
            + q_dist.log_prob(q_index)
        )
        entropy = family_dist.entropy() + quantizer_dist.entropy() + q_dist.entropy()
        return log_probability, entropy
