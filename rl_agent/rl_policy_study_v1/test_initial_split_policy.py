"""Small hand-checkable regressions for the initial split-only policy."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from rl_agent.rl_policy_study_v1.initial_split_policy import (
    ACTION_COUNT,
    RecurrentFactorizedSplitActorCritic,
    action_id_from_factors,
    factors_from_action_id,
)


def test_factor_round_trip() -> None:
    action_ids = torch.arange(ACTION_COUNT)
    family, quantizer, q_index = factors_from_action_id(action_ids)
    assert torch.equal(action_id_from_factors(family, quantizer, q_index), action_ids)


def test_single_allowed_action_is_selected() -> None:
    torch.manual_seed(7)
    model = RecurrentFactorizedSplitActorCritic()
    observations = torch.zeros(2, 3, 7)
    recurrent, _, reward_value, cost_values = model.encode_sequence(observations)
    mask = torch.zeros(2, 4, 3, 6, dtype=torch.bool)
    mask[:, 3, 2, 3] = True  # AE32 / UINT4 / q=.70 = action 69.
    selected = model.choose(recurrent[:, -1], mask)
    assert selected.action_id.tolist() == [69, 69]
    assert torch.isfinite(selected.log_probability).all()
    assert reward_value.shape == (2, 3)
    assert cost_values.shape == (2, 3, 2)


def test_evaluation_matches_sample_log_probability() -> None:
    torch.manual_seed(11)
    model = RecurrentFactorizedSplitActorCritic()
    observations = torch.randn(4, 2, 7)
    recurrent, _, _, _ = model.encode_sequence(observations)
    mask = torch.ones(4, 4, 3, 6, dtype=torch.bool)
    chosen = model.choose(recurrent[:, -1], mask)
    evaluated, entropy = model.evaluate_actions(
        recurrent[:, -1], mask, chosen.action_id
    )
    assert torch.allclose(chosen.log_probability, evaluated)
    assert torch.isfinite(entropy).all()


def test_masked_action_cannot_be_evaluated() -> None:
    model = RecurrentFactorizedSplitActorCritic()
    recurrent, _, _, _ = model.encode_sequence(torch.zeros(1, 1, 7))
    mask = torch.ones(1, 4, 3, 6, dtype=torch.bool)
    mask[0, 0, 0, 0] = False
    try:
        model.evaluate_actions(recurrent[:, -1], mask, torch.tensor([0]))
    except ValueError as exc:
        assert "excluded by its technical mask" in str(exc)
    else:
        raise AssertionError("a technically masked action was evaluated")


def test_real_catalog_factor_order() -> None:
    path = Path(
        "experiments/splitfusion_288_offline_rl_dataset_v1/"
        "20260909_offline_consolidation_v1/action_72_summary.csv"
    )
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    family_order = ("noAE", "AE128", "AE64", "AE32")
    quantizer_order = ("UINT8", "UINT6", "UINT4")
    q_order = (0.0, 0.3, 0.5, 0.7, 0.9, 0.98)
    assert len(rows) == ACTION_COUNT
    for row in rows:
        action_id = int(row["action_id"])
        family, quantizer, q_index = factors_from_action_id(torch.tensor([action_id]))
        assert row["family"] == family_order[int(family.item())]
        assert row["quantizer"] == quantizer_order[int(quantizer.item())]
        assert float(row["q"]) == q_order[int(q_index.item())]


def main() -> int:
    test_factor_round_trip()
    test_single_allowed_action_is_selected()
    test_evaluation_matches_sample_log_probability()
    test_masked_action_cannot_be_evaluated()
    test_real_catalog_factor_order()
    print("5 focused initial-policy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
