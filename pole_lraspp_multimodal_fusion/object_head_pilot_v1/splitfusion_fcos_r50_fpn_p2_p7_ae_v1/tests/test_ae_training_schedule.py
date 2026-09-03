"""Two focused CPU checks over the Phase-9C schedule and ranking rule.

No checkpoint, CUDA, dataset, cache, training, inference or evaluation is
touched: both tests are pure arithmetic over the locked configuration and over
synthetic ranking records.
"""

from __future__ import annotations

import random
import unittest

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from .. import ae_contract, ae_holdout_selection, ae_loss
from .. import ae_training_common as common


def synthetic_record(
    epoch: int, q: float, *, gates_passed: int, worst_normalized: float, loss: float
) -> dict:
    """Exactly the fields the ranking rule reads, and nothing else."""
    return {
        "epoch": epoch,
        "q": q,
        "gate_result": {
            "gates_passed": gates_passed,
            "all_passed": gates_passed == ae_holdout_selection.GATE_COUNT,
            "worst_normalized_degradation": worst_normalized,
        },
        "reconstruction": {"mean_total_loss": loss},
    }


def synthetic_checkpoint(
    epoch: int, passes: tuple[int, int, int, int], worst: float, losses: tuple[float, ...]
) -> list[dict]:
    return [
        synthetic_record(
            epoch,
            float(q),
            gates_passed=passes[index],
            worst_normalized=worst,
            loss=losses[index],
        )
        for index, q in enumerate(common.AE_HOLDOUT_Q_VALUES)
    ]


class AeTrainingScheduleChecks(unittest.TestCase):
    def test_stage_schedule_and_balanced_stage_b_q_counts(self) -> None:
        # Stage boundaries and the two learning rates.
        for epoch in range(1, common.AE_STAGE_A_EPOCHS + 1):
            self.assertEqual(common.stage_for_epoch(epoch), common.AE_STAGE_A)
        for epoch in range(common.AE_STAGE_A_EPOCHS + 1, common.AE_TRAINING_EPOCHS + 1):
            self.assertEqual(common.stage_for_epoch(epoch), common.AE_STAGE_B)
        self.assertEqual(common.learning_rate_for_stage(common.AE_STAGE_A), 1e-3)
        self.assertEqual(common.learning_rate_for_stage(common.AE_STAGE_B), 3e-4)
        for outside in (0, common.AE_TRAINING_EPOCHS + 1):
            with self.assertRaises(guards.HybridQConfigError):
                common.stage_for_epoch(outside)

        # Every fit frame exactly once per epoch, final short batch retained.
        self.assertFalse(common.AE_DROP_LAST)
        self.assertEqual(common.AE_BATCH_SIZE, 16)
        self.assertEqual(common.batches_per_epoch(), 847)
        self.assertEqual(
            common.batches_per_epoch() * common.AE_BATCH_SIZE - contract.TRAIN_FIT_FRAMES,
            9,  # the final batch carries 7 of 16 frames
        )
        self.assertEqual(common.stage_b_updates_total(), 6776)

        # Walk the schedule exactly as the trainer does.
        position = 0
        updates = 0
        stage_a_q: set[float] = set()
        stage_b_counts: dict[str, int] = {
            f"{float(q):.2f}": 0 for q in common.AE_STAGE_B_Q_CYCLE
        }
        first_q_of_epoch: dict[int, float] = {}
        for epoch in range(1, common.AE_TRAINING_EPOCHS + 1):
            stage = common.stage_for_epoch(epoch)
            for batch in range(common.batches_per_epoch()):
                if stage == common.AE_STAGE_A:
                    q = float(common.AE_STAGE_A_Q)
                    stage_a_q.add(q)
                else:
                    q = common.stage_b_q_at(position)
                    stage_b_counts[f"{q:.2f}"] += 1
                    position += 1
                if batch == 0:
                    first_q_of_epoch[epoch] = q
                # The trainer admits every scheduled q through this guard.
                ae_loss.require_optimization_q(q)
                updates += 1

        self.assertEqual(updates, common.AE_TRAINING_EPOCHS * 847)
        self.assertEqual(stage_a_q, {0.00})
        self.assertEqual(position, 6776)
        self.assertEqual(sum(stage_b_counts.values()), 6776)
        self.assertEqual(
            stage_b_counts, {"0.00": 1694, "0.30": 1694, "0.50": 1694, "0.70": 1694}
        )
        self.assertEqual(common.require_balanced_stage_b(), stage_b_counts)

        # The cycle carries across epoch boundaries: 847 is not a multiple of
        # four, so a per-epoch restart would put q=0.00 first in every Stage-B
        # epoch and unbalance the totals.
        starts = [first_q_of_epoch[epoch] for epoch in range(5, 13)]
        self.assertNotEqual(set(starts), {0.00})
        self.assertEqual(starts[0], 0.00)
        self.assertEqual(starts[1], common.AE_STAGE_B_Q_CYCLE[847 % 4])

        # The two stress values are never scheduled and are refused outright.
        self.assertEqual(tuple(common.AE_EXCLUDED_Q), (0.90, 0.98))
        for excluded in common.AE_EXCLUDED_Q:
            self.assertNotIn(f"{float(excluded):.2f}", stage_b_counts)
            with self.assertRaises(guards.HybridQConfigError):
                ae_loss.require_optimization_q(float(excluded))
        self.assertEqual(
            tuple(common.AE_STAGE_B_Q_CYCLE), tuple(ae_contract.AE_STAGE_B_Q_CYCLE)
        )

    def test_preregistered_checkpoint_ranking_is_deterministic(self) -> None:
        full = ae_holdout_selection.GATE_COUNT
        losses = (0.5, 0.5, 0.5, 0.5)

        def decide(records: list[dict]) -> dict:
            shuffled = list(records)
            random.Random(20260829).shuffle(shuffled)
            decision = ae_holdout_selection.rank_checkpoints(shuffled)
            # Order of the input must not matter.
            self.assertEqual(
                decision["ranking"],
                ae_holdout_selection.rank_checkpoints(records)["ranking"],
            )
            return decision

        # 1) the worst same-q gate count wins, even against a larger total.
        decision = decide(
            synthetic_checkpoint(4, (10, 10, 10, 10), 0.5, losses)
            + synthetic_checkpoint(8, (full, full, full, 9), 0.1, losses)
        )
        self.assertEqual(decision["selected_epoch"], 4)
        self.assertEqual(decision["decided_at_criterion"], "min_same_q_gates_passed")
        self.assertEqual(decision["selected"]["min_same_q_gates_passed"], 10)
        self.assertEqual(decision["selected"]["total_gates_passed"], 40)

        # 2) equal minima: the larger total wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.1, losses)
            + synthetic_checkpoint(8, (9, 11, 11, 11), 0.9, losses)
        )
        self.assertEqual(decision["selected_epoch"], 8)
        self.assertEqual(decision["decided_at_criterion"], "total_gates_passed")

        # 3) equal minima and totals: the smaller worst normalized degradation wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.90, losses)
            + synthetic_checkpoint(8, (9, 9, 9, 9), 0.25, losses)
        )
        self.assertEqual(decision["selected_epoch"], 8)
        self.assertEqual(
            decision["decided_at_criterion"], "worst_normalized_degradation"
        )

        # 4) equal through three: the smaller mean holdout reconstruction loss wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, (0.9, 0.9, 0.9, 0.9))
            + synthetic_checkpoint(8, (9, 9, 9, 9), 0.25, (0.4, 0.4, 0.4, 0.4))
        )
        self.assertEqual(decision["selected_epoch"], 8)
        self.assertEqual(
            decision["decided_at_criterion"], "mean_holdout_reconstruction_loss"
        )
        self.assertAlmostEqual(
            decision["selected"]["mean_holdout_reconstruction_loss"], 0.4
        )

        # 5) fully tied: the earlier epoch wins.
        decision = decide(
            synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, losses)
            + synthetic_checkpoint(8, (9, 9, 9, 9), 0.25, losses)
            + synthetic_checkpoint(12, (9, 9, 9, 9), 0.25, losses)
        )
        self.assertEqual(decision["selected_epoch"], 4)
        self.assertEqual(decision["decided_at_criterion"], "epoch")
        self.assertEqual([row["epoch"] for row in decision["ranking"]], [4, 8, 12])

        # Selecting a checkpoint is never a service-ready claim.
        self.assertFalse(decision["selection_is_a_service_ready_claim"])
        self.assertFalse(decision["selected"]["all_gates_passed_at_every_q"])
        passing = decide(
            synthetic_checkpoint(4, (full, full, full, full), -0.2, losses)
        )
        self.assertTrue(passing["selected"]["all_gates_passed_at_every_q"])
        self.assertFalse(passing["selection_is_a_service_ready_claim"])

        # An incomplete q sweep for a candidate fails closed.
        incomplete = synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, losses)[:3]
        with self.assertRaises(guards.HybridQConfigError):
            ae_holdout_selection.rank_checkpoints(incomplete)
        duplicated = synthetic_checkpoint(4, (9, 9, 9, 9), 0.25, losses)
        duplicated.append(duplicated[0])
        with self.assertRaises(guards.HybridQConfigError):
            ae_holdout_selection.rank_checkpoints(duplicated)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
