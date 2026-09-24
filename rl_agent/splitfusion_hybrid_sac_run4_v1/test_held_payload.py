from __future__ import annotations

import hashlib
import random
import unittest
from dataclasses import replace
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run4_v1.held_payload import (
    BindingMismatch,
    EVIDENCE_LABEL,
    EXACT_NODE_STATUS,
    INTERPOLATED_STATUS,
    IdentifierConflict,
    HeldPayloadProviderV1,
    HeldSelectionCounterV1,
    LocalRngRequired,
    PayloadNodeV1,
    ScenePayloadCurveV1,
    SourceRejected,
    UnsupportedAction,
    payload_curve_inventory_sha256,
)
from rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.contract import (
    Q_E4_GRID,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


class _StubRng:
    def __init__(self, *draws: float) -> None:
        self.draws = list(draws)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.draws.pop(0)


def _curve(
    scene: int,
    mode_id: int,
    *,
    source_split: str = "train",
    selection_sha: str | None = None,
    database_sha: str | None = None,
    episode_id: str | None = None,
    frame_id: int | None = None,
    scene_sha: str | None = None,
    inclusion_probability: float | None = None,
    sampling_weight: float | None = None,
) -> ScenePayloadCurveV1:
    probability = (0.5, 0.25, 0.2)[scene % 3]
    weight = 1.0 / probability
    base = 1_000_000 + scene * 20_000 + mode_id * 10_000
    nodes = tuple(
        PayloadNodeV1(
            q_e4=q_e4,
            total_transmitted_bytes=base - 50 * q_e4,
            source_row_sha256=_digest(f"row:{scene}:{mode_id}:{q_e4}"),
        )
        for q_e4 in Q_E4_GRID
    )
    return ScenePayloadCurveV1(
        sample_id=f"sample-{scene}",
        episode_id=episode_id or f"episode-{scene}",
        frame_id=frame_id if frame_id is not None else 100 + scene,
        selection_rank_within_train=scene,
        source_split=source_split,
        inclusion_probability=(
            probability if inclusion_probability is None else inclusion_probability
        ),
        sampling_weight=weight if sampling_weight is None else sampling_weight,
        mode_id=mode_id,
        scene_source_sha256=scene_sha or _digest(f"scene:{scene}"),
        source_selection_sha256=selection_sha or _digest("selection"),
        source_database_sha256=database_sha or _digest("database"),
        nodes=nodes,
    )


def _inventory(scene_count: int = 2) -> tuple[ScenePayloadCurveV1, ...]:
    return tuple(
        _curve(scene, mode_id)
        for scene in range(scene_count)
        for mode_id in range(12)
    )


def _provider(curves: tuple[ScenePayloadCurveV1, ...] | None = None) -> HeldPayloadProviderV1:
    material = _inventory() if curves is None else curves
    return HeldPayloadProviderV1(
        material,
        expected_inventory_sha256=payload_curve_inventory_sha256(material),
    )


def _counter(
    *, decision_seq: int = 3, held_ordinal: int = 1, stream: str = "local-stream"
) -> HeldSelectionCounterV1:
    return HeldSelectionCounterV1(
        session_id="session-a",
        decision_seq=decision_seq,
        held_ordinal=held_ordinal,
        rng_stream_id=stream,
    )


class HeldPayloadProviderTests(unittest.TestCase):
    def test_weighted_scene_selection_and_identity_retention(self) -> None:
        provider = _provider()
        # Weights are 2 and 4, so the first scene owns [0, 1/3).
        first = provider.evaluate(
            counter=_counter(held_ordinal=1), rng=_StubRng(0.0), mode_id=2, q_e4=5000
        )
        second = provider.evaluate(
            counter=_counter(held_ordinal=2), rng=_StubRng(1.0 / 3.0), mode_id=2, q_e4=5000
        )
        self.assertEqual(first.selected_sample_id, "sample-0")
        self.assertEqual(second.selected_sample_id, "sample-1")
        self.assertAlmostEqual(first.marginal_selection_probability, 2.0 / 6.0)
        self.assertAlmostEqual(second.marginal_selection_probability, 4.0 / 6.0)
        self.assertEqual(first.evidence_label, EVIDENCE_LABEL)
        self.assertEqual(first.interpolation_status, EXACT_NODE_STATUS)
        self.assertRegex(first.selected_scene_source_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(first.selected_curve_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual((first.mode_id, first.q_e4), (2, 5000))

    def test_same_scene_q_interpolation_matches_empirical_surface_rule(self) -> None:
        provider = _provider()
        result = provider.evaluate(
            counter=_counter(), rng=_StubRng(0.0), mode_id=0, q_e4=2000
        )
        lower = 1_000_000 - 50 * 1500
        upper = 1_000_000 - 50 * 3000
        expected = lower + (2000 - 1500) / (3000 - 1500) * (upper - lower)
        self.assertEqual(result.total_transmitted_bytes, expected)
        self.assertEqual(result.interpolation_status, INTERPOLATED_STATUS)
        self.assertEqual(tuple(item.q_e4 for item in result.endpoints), (1500, 3000))
        self.assertEqual(result.selected_sample_id, "sample-0")

    def test_same_counter_is_idempotent_and_does_not_consume_rng_twice(self) -> None:
        provider = _provider()
        rng = _StubRng(0.1, 0.9)
        first = provider.evaluate(counter=_counter(), rng=rng, mode_id=1, q_e4=1234)
        second = provider.evaluate(counter=_counter(), rng=rng, mode_id=1, q_e4=1234)
        self.assertIs(first, second)
        self.assertEqual(rng.calls, 1)
        self.assertEqual(provider.issued_count, 1)

    def test_counter_reuse_with_action_or_stream_conflict_fails_before_rng(self) -> None:
        provider = _provider()
        provider.evaluate(counter=_counter(), rng=_StubRng(0.1), mode_id=1, q_e4=1234)
        conflict_rng = _StubRng(0.8)
        with self.assertRaises(IdentifierConflict):
            provider.evaluate(
                counter=_counter(), rng=conflict_rng, mode_id=2, q_e4=1234
            )
        self.assertEqual(conflict_rng.calls, 0)
        with self.assertRaises(IdentifierConflict):
            provider.evaluate(
                counter=_counter(stream="another-stream"),
                rng=conflict_rng,
                mode_id=1,
                q_e4=1234,
            )
        self.assertEqual(conflict_rng.calls, 0)

    def test_global_rng_module_is_rejected_and_global_function_is_unused(self) -> None:
        provider = _provider()
        with self.assertRaises(LocalRngRequired):
            provider.evaluate(counter=_counter(), rng=random, mode_id=0, q_e4=0)
        local = random.Random(7)
        with mock.patch("random.random", side_effect=AssertionError("global RNG used")):
            result = provider.evaluate(
                counter=_counter(held_ordinal=2), rng=local, mode_id=0, q_e4=0
            )
        self.assertEqual(result.q_e4, 0)

    def test_bad_rng_draws_fail_closed(self) -> None:
        for draw in (-0.1, 1.0, float("nan"), True):
            with self.subTest(draw=draw):
                with self.assertRaises((LocalRngRequired, SourceRejected, ValueError)):
                    _provider().evaluate(
                        counter=_counter(), rng=_StubRng(draw), mode_id=0, q_e4=0
                    )

    def test_held_validation_and_test_curves_are_rejected(self) -> None:
        for split in ("fit", "held", "held_scene", "validation", "val", "test"):
            with self.subTest(split=split):
                with self.assertRaises(SourceRejected):
                    _curve(0, 0, source_split=split)

    def test_missing_q_endpoint_and_nonpositive_payload_are_rejected(self) -> None:
        good = _curve(0, 0)
        with self.assertRaises(SourceRejected):
            replace(good, nodes=good.nodes[:-1])
        with self.assertRaises(ValueError):
            replace(good.nodes[-1], total_transmitted_bytes=0)

    def test_incomplete_modes_and_unsupported_actions_are_rejected(self) -> None:
        curves = _inventory()[:-1]
        with self.assertRaises(SourceRejected):
            _provider(curves)
        provider = _provider()
        for mode_id, q_e4 in ((12, 0), (-1, 0), (0, 9801), (0, -1), (True, 0)):
            with self.subTest(mode_id=mode_id, q_e4=q_e4):
                with self.assertRaises((UnsupportedAction, ValueError)):
                    provider.evaluate(
                        counter=_counter(held_ordinal=10 + abs(int(mode_id))),
                        rng=_StubRng(0.0),
                        mode_id=mode_id,
                        q_e4=q_e4,
                    )

    def test_mixed_bindings_are_rejected(self) -> None:
        curves = list(_inventory())
        curves[-1] = replace(curves[-1], source_database_sha256=_digest("foreign"))
        material = tuple(curves)
        with self.assertRaises(BindingMismatch):
            _provider(material)

    def test_inventory_digest_is_caller_pinned(self) -> None:
        curves = _inventory()
        with self.assertRaises(BindingMismatch):
            HeldPayloadProviderV1(
                curves, expected_inventory_sha256="0" * 64
            )

    def test_scene_identifier_conflicts_are_rejected(self) -> None:
        curves = list(_inventory())
        curves = [
            replace(curve, episode_id="episode-0", frame_id=100)
            if curve.sample_id == "sample-1"
            else curve
            for curve in curves
        ]
        with self.assertRaises(IdentifierConflict):
            _provider(tuple(curves))

    def test_global_train_selection_rank_conflict_is_rejected(self) -> None:
        curves = tuple(
            replace(curve, selection_rank_within_train=0)
            if curve.sample_id == "sample-1"
            else curve
            for curve in _inventory()
        )
        with self.assertRaises(IdentifierConflict):
            _provider(curves)

    def test_duplicate_endpoint_digest_is_rejected(self) -> None:
        curves = list(_inventory())
        first_digest = curves[0].nodes[0].source_row_sha256
        target = curves[-1]
        changed = list(target.nodes)
        changed[0] = replace(changed[0], source_row_sha256=first_digest)
        curves[-1] = replace(target, nodes=tuple(changed))
        with self.assertRaises(IdentifierConflict):
            _provider(tuple(curves))

    def test_sampling_weight_must_match_inclusion_probability(self) -> None:
        with self.assertRaises(SourceRejected):
            _curve(0, 0, inclusion_probability=0.5, sampling_weight=3.0)

    def test_boolean_counter_and_action_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HeldSelectionCounterV1("s", True, 1, "rng")
        with self.assertRaises(ValueError):
            HeldSelectionCounterV1("s", 0, False, "rng")


if __name__ == "__main__":
    unittest.main()
