"""Exhaustive tests for the Hybrid-SAC action contract adapter.

Two top-level test methods, intentionally:

1. ``test_catalog_reconciles_all_72_profiles`` -- everything about binding the
   frozen catalog to the 12 joint modes and the 72 measured anchors.
2. ``test_continuous_q_boundaries_half_up_and_non_anchor`` -- everything about
   the continuous-q wire conversion, its boundaries, and the strict refusal to
   name a catalog action for an unmeasured quality.

The catalog document is re-read independently of the adapter so that the
declared orders are cross-checked against the adapter's derived orders rather
than against hard-coded expectations.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from itertools import product

from . import action_contract as ac


class ActionContractTest(unittest.TestCase):
    """Phase-1 action-contract reconciliation and continuous-q semantics."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ac.load_contract()
        cls.catalog_bytes = ac.default_catalog_path().read_bytes()
        cls.document = json.loads(cls.catalog_bytes.decode("utf-8"))

    # ------------------------------------------------------------------ #

    def test_catalog_reconciles_all_72_profiles(self) -> None:
        contract = self.contract
        document = self.document

        # --- catalog SHA / schema ------------------------------------- #
        self.assertEqual(
            hashlib.sha256(self.catalog_bytes).hexdigest(), ac.CATALOG_SHA256
        )
        self.assertEqual(contract.catalog_sha256, ac.CATALOG_SHA256)
        self.assertEqual(contract.schema, ac.CATALOG_SCHEMA)
        self.assertEqual(document["schema"], ac.CATALOG_SCHEMA)
        self.assertEqual(
            document["mode_policy_boundary"]["catalog_execution_mode"],
            ac.EXECUTION_MODE,
        )
        self.assertEqual(
            document["fixed_transport_contract"]["spatial_cells"], ac.SPATIAL_CELLS
        )

        # --- declared orders come from the catalog, not from action IDs - #
        declared_families = tuple(document["action_order"]["family"])
        declared_quantizers = tuple(document["action_order"]["quantizer"])
        declared_anchors = tuple(document["action_order"]["q_e4"])
        self.assertEqual(contract.family_order, declared_families)
        self.assertEqual(contract.quantizer_order, declared_quantizers)
        self.assertEqual(contract.q_anchor_order, declared_anchors)
        self.assertEqual(len(declared_families), ac.EXPECTED_FAMILY_COUNT)
        self.assertEqual(len(declared_quantizers), ac.EXPECTED_QUANTIZER_COUNT)
        self.assertEqual(len(declared_anchors), ac.EXPECTED_Q_ANCHOR_COUNT)
        self.assertEqual(
            tuple(document["continuous_q"]["validated_anchor_q_e4"]), declared_anchors
        )

        # --- declared inventory reconciles to 4 x 3 x 6 = 72 ----------- #
        overall = document["summary"]["overall"]
        self.assertEqual(overall["families"], ac.EXPECTED_FAMILY_COUNT)
        self.assertEqual(overall["quantizers"], ac.EXPECTED_QUANTIZER_COUNT)
        self.assertEqual(overall["q_anchors"], ac.EXPECTED_Q_ANCHOR_COUNT)
        self.assertEqual(overall["profiles"], ac.EXPECTED_PROFILE_COUNT)
        self.assertEqual(
            ac.EXPECTED_FAMILY_COUNT
            * ac.EXPECTED_QUANTIZER_COUNT
            * ac.EXPECTED_Q_ANCHOR_COUNT,
            ac.EXPECTED_PROFILE_COUNT,
        )
        rows = document["profiles"]
        self.assertEqual(len(rows), ac.EXPECTED_PROFILE_COUNT)

        # --- 12 stable joint modes in declared Cartesian order --------- #
        self.assertEqual(contract.mode_count, ac.EXPECTED_MODE_COUNT)
        self.assertEqual(contract.mode_count, 12)
        expected_mode_keys = list(product(declared_families, declared_quantizers))
        self.assertEqual([m.key for m in contract.modes], expected_mode_keys)
        self.assertEqual(
            [m.mode_id for m in contract.modes], list(range(ac.EXPECTED_MODE_COUNT))
        )
        self.assertEqual(
            len({m.canonical for m in contract.modes}), ac.EXPECTED_MODE_COUNT
        )
        for mode_id, (family, quantizer) in enumerate(expected_mode_keys):
            mode = contract.mode(mode_id)
            self.assertIs(mode, contract.mode_for(family, quantizer))
            self.assertEqual(mode.canonical, f"SPLIT/{family}/{quantizer}")
            # bit width is supplied by the catalog for every mode and agrees
            # with the declared quantizer label.
            self.assertIsInstance(mode.bit_width, int)
            self.assertEqual(mode.bit_width, int(quantizer.replace("UINT", "")))
            self.assertIn(mode.bit_width, (4, 6, 8))
            self.assertEqual(mode.zstd_level, 1)
            self.assertEqual(mode.wire_layout, "CURRENT_CELL_MAJOR")
            self.assertIsInstance(mode.transported_channels, int)
            self.assertIsInstance(mode.decoder_identity, str)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                mode.mode_id = 99  # type: ignore[misc]

        with self.assertRaises(ac.UnknownJointModeError):
            contract.mode(ac.EXPECTED_MODE_COUNT)
        with self.assertRaises(ac.UnknownJointModeError):
            contract.mode(-1)
        with self.assertRaises(ac.UnknownJointModeError):
            contract.mode_for("noAE", "UINT7")
        with self.assertRaises(ac.UnknownJointModeError):
            contract.mode_for("AE256", "UINT8")

        # --- exactly six declared anchors per joint mode --------------- #
        for mode in contract.modes:
            anchors = contract.anchors_for_mode(mode.mode_id)
            self.assertEqual(len(anchors), ac.EXPECTED_Q_ANCHOR_COUNT)
            self.assertEqual(tuple(a.q_e4 for a in anchors), declared_anchors)
            for anchor in anchors:
                self.assertIs(anchor.mode, mode)
                self.assertEqual(anchor.execution_mode, ac.EXECUTION_MODE)

        # --- complete 72-row exact lookup and unique identities -------- #
        self.assertEqual(contract.anchor_count, ac.EXPECTED_PROFILE_COUNT)
        self.assertEqual(
            len({a.action_id for a in contract.anchors}), ac.EXPECTED_PROFILE_COUNT
        )
        self.assertEqual(
            len({a.profile_id for a in contract.anchors}), ac.EXPECTED_PROFILE_COUNT
        )
        self.assertEqual(
            len({a.key for a in contract.anchors}), ac.EXPECTED_PROFILE_COUNT
        )
        self.assertEqual(
            {a.action_id for a in contract.anchors}, {r["action_id"] for r in rows}
        )
        self.assertEqual(
            {a.profile_id for a in contract.anchors}, {r["profile_id"] for r in rows}
        )

        # no missing or duplicate tuple: the anchor set is exactly the full
        # declared Cartesian product.
        expected_keys = set(
            product(declared_families, declared_quantizers, declared_anchors)
        )
        self.assertEqual(len(expected_keys), ac.EXPECTED_PROFILE_COUNT)
        self.assertEqual({a.key for a in contract.anchors}, expected_keys)
        self.assertEqual(
            {(r["family"], r["quantizer"], r["q_e4"]) for r in rows}, expected_keys
        )

        # --- every row reconciles exactly, including keep/drop --------- #
        for row in rows:
            key = (row["family"], row["quantizer"], row["q_e4"])
            anchor = contract.find_anchor(*key)
            self.assertIsNotNone(anchor, f"no anchor for registered row {key}")
            self.assertEqual(anchor.action_id, row["action_id"])
            self.assertEqual(anchor.profile_id, row["profile_id"])
            self.assertEqual(anchor.q_e4, row["q_e4"])
            self.assertEqual(anchor.q, float(row["q"]))
            self.assertEqual(anchor.q * ac.Q_E4_SCALE, float(row["q_e4"]))
            self.assertEqual(anchor.execution_mode, row["execution_mode"])
            self.assertEqual(anchor.execution_mode, ac.EXECUTION_MODE)
            self.assertEqual(anchor.mode.family, row["family"])
            self.assertEqual(anchor.mode.quantizer, row["quantizer"])
            self.assertEqual(anchor.mode.family_id, row["family_id"])
            self.assertEqual(anchor.mode.bit_width, row["bit_width"])
            self.assertEqual(anchor.mode.latent_width, row["latent_width"])
            self.assertEqual(anchor.mode.routing_tag, row["routing_tag"])

            # exact keep/drop reconciliation against the catalog row
            keep, drop = ac.keep_drop_counts(row["q_e4"])
            self.assertEqual(keep, row["keep_count"], f"keep mismatch for {key}")
            self.assertEqual(drop, row["drop_count"], f"drop mismatch for {key}")
            self.assertEqual(anchor.keep_count, row["keep_count"])
            self.assertEqual(anchor.drop_count, row["drop_count"])
            self.assertEqual(keep + drop, ac.SPATIAL_CELLS)

            # resolving the anchor's own q recovers the same catalog identity
            resolved = contract.resolve(anchor.mode.mode_id, float(row["q"]))
            self.assertTrue(resolved.is_registered_anchor)
            self.assertEqual(resolved.action_id, row["action_id"])
            self.assertEqual(resolved.profile_id, row["profile_id"])
            self.assertEqual(resolved.q_e4, row["q_e4"])
            self.assertEqual(resolved.keep_count, row["keep_count"])
            self.assertEqual(resolved.drop_count, row["drop_count"])
            self.assertEqual(resolved.execution_mode, ac.EXECUTION_MODE)

        # --- the adapter does not import perception/payload fields ----- #
        anchor_fields = {f.name for f in dataclasses.fields(ac.AnchorAction)}
        for leaked in ("perception", "payload", "capabilities", "source_evidence"):
            self.assertNotIn(leaked, anchor_fields)

        # --- catalog bytes untouched by binding ------------------------ #
        self.assertEqual(
            hashlib.sha256(ac.default_catalog_path().read_bytes()).hexdigest(),
            ac.CATALOG_SHA256,
        )

    # ------------------------------------------------------------------ #

    def test_continuous_q_boundaries_half_up_and_non_anchor(self) -> None:
        contract = self.contract
        declared_anchors = contract.q_anchor_order
        rows_by_key = {
            (r["family"], r["quantizer"], r["q_e4"]): r for r in self.document["profiles"]
        }

        # --- q = 0 and q = 0.98, the two registered extremes ----------- #
        low = contract.quality_for(0.0)
        self.assertEqual(low.q_e4, ac.Q_E4_MIN)
        self.assertEqual(low.q_exec, 0.0)
        self.assertEqual(low.drop_count, 0)
        self.assertEqual(low.keep_count, ac.SPATIAL_CELLS)
        self.assertFalse(low.was_clipped)

        high = contract.quality_for(0.98)
        self.assertEqual(high.q_e4, ac.Q_E4_MAX)
        self.assertEqual(high.q_exec, 0.98)
        self.assertFalse(high.was_clipped)
        self.assertEqual(high.keep_count + high.drop_count, ac.SPATIAL_CELLS)
        reference_row = rows_by_key[
            (contract.family_order[0], contract.quantizer_order[0], ac.Q_E4_MAX)
        ]
        self.assertEqual(high.keep_count, reference_row["keep_count"])
        self.assertEqual(high.drop_count, reference_row["drop_count"])
        self.assertEqual(ac.Q_MIN, 0.0)
        self.assertEqual(ac.Q_MAX, 0.98)

        # --- exact half-up ties, not banker's rounding ----------------- #
        # 10000 * 0.12345 is exactly 1234.5 in binary64, so Python's built-in
        # round() returns 1234 (round-half-to-even).  The contract requires 1235.
        self.assertEqual(10000 * 0.12345, 1234.5)
        self.assertEqual(round(10000 * 0.12345), 1234)
        self.assertEqual(ac.round_half_up_q_e4(0.12345), 1235)
        self.assertEqual(contract.quality_for(0.12345).q_e4, 1235)
        self.assertEqual(
            int(Decimal("1234.5").to_integral_value(rounding=ROUND_HALF_EVEN)), 1234
        )
        self.assertEqual(
            int(Decimal("1234.5").to_integral_value(rounding=ROUND_HALF_UP)), 1235
        )
        # a tie that rounds up from zero, where banker's would return 0
        self.assertEqual(ac.round_half_up_q_e4(0.00005), 1)
        self.assertEqual(round(10000 * 0.00005), 0)
        # a tie at an even/odd boundary in the other parity
        self.assertEqual(ac.round_half_up_q_e4(0.12335), 1234)
        # non-tie neighbours are unaffected
        self.assertEqual(ac.round_half_up_q_e4(0.123449), 1234)
        self.assertEqual(ac.round_half_up_q_e4(0.123451), 1235)

        # --- clipping at both mechanical boundaries -------------------- #
        below = contract.quality_for(-0.5)
        self.assertEqual(below.q_e4, ac.Q_E4_MIN)
        self.assertEqual(below.requested_q, -0.5)
        self.assertTrue(below.clipped_below)
        self.assertFalse(below.clipped_above)
        self.assertEqual(below.drop_count, 0)
        self.assertEqual(below.keep_count, ac.SPATIAL_CELLS)

        above = contract.quality_for(1.0)
        self.assertEqual(above.q_e4, ac.Q_E4_MAX)
        self.assertEqual(above.requested_q, 1.0)
        self.assertTrue(above.clipped_above)
        self.assertFalse(above.clipped_below)

        self.assertTrue(contract.quality_for(0.99).clipped_above)
        self.assertEqual(contract.quality_for(0.99).q_e4, ac.Q_E4_MAX)
        self.assertEqual(ac.round_half_up_q_e4(-1e9), ac.Q_E4_MIN)
        self.assertEqual(ac.round_half_up_q_e4(1e9), ac.Q_E4_MAX)
        # rounding happens before clipping: 0.98004 rounds *to* the bound and is
        # therefore not a clipped request.
        at_bound = contract.quality_for(0.98004)
        self.assertEqual(at_bound.q_e4, ac.Q_E4_MAX)
        self.assertFalse(at_bound.was_clipped)
        just_over = contract.quality_for(0.98006)
        self.assertEqual(just_over.q_e4, ac.Q_E4_MAX)
        self.assertTrue(just_over.clipped_above)
        # a tiny negative request rounds to 0 without being flagged as clipped
        self.assertFalse(contract.quality_for(-0.000004).was_clipped)
        self.assertTrue(contract.quality_for(-0.00006).clipped_below)

        # --- NaN / infinity / non-real rejection ----------------------- #
        for bad in (
            float("nan"),
            float("inf"),
            float("-inf"),
            Decimal("NaN"),
        ):
            with self.assertRaises(ac.InvalidQualityError):
                ac.round_half_up_q_e4(bad)
            with self.assertRaises(ac.InvalidQualityError):
                contract.quality_for(bad)
        for bad in ("0.5", None, True, [0.5], 1j):
            with self.assertRaises(ac.InvalidQualityError):
                contract.quality_for(bad)
            with self.assertRaises(ac.InvalidQualityError):
                ac.round_half_up_q_e4(bad)
        with self.assertRaises(ac.InvalidQualityError):
            ac.keep_drop_counts(ac.Q_E4_MAX + 1)
        with self.assertRaises(ac.InvalidQualityError):
            ac.keep_drop_counts(-1)
        with self.assertRaises(ac.InvalidQualityError):
            ac.keep_drop_counts(0.5)

        # --- keep/drop is total and exact across the whole wire range -- #
        previous_keep = ac.SPATIAL_CELLS + 1
        for q_e4 in range(ac.Q_E4_MIN, ac.Q_E4_MAX + 1):
            keep, drop = ac.keep_drop_counts(q_e4)
            self.assertEqual(keep + drop, ac.SPATIAL_CELLS)
            self.assertGreaterEqual(keep, 0)
            self.assertGreaterEqual(drop, 0)
            self.assertLessEqual(keep, previous_keep)
            previous_keep = keep
            # integer half-up form agrees with the declared float rule
            self.assertEqual(
                drop, int(q_e4 * ac.SPATIAL_CELLS / ac.Q_E4_SCALE + 0.5)
            )
            # the wire value round-trips exactly through q_exec
            self.assertEqual(
                ac.round_half_up_q_e4(q_e4 / ac.Q_E4_SCALE),
                q_e4,
                f"q_exec round-trip failed at q_e4={q_e4}",
            )
        # no representable q_e4 lands on a keep/drop rounding tie
        self.assertFalse(
            any(
                (q_e4 * ac.SPATIAL_CELLS) % ac.Q_E4_SCALE == ac.Q_E4_SCALE // 2
                for q_e4 in range(ac.Q_E4_MIN, ac.Q_E4_MAX + 1)
            )
        )

        # --- arbitrary interior non-anchor execution ------------------- #
        non_anchor_q_e4 = [
            q for q in (1, 2999, 3001, 4237, 5001, 6500, 8999, 9001, 9799)
        ]
        for q_e4 in non_anchor_q_e4:
            self.assertNotIn(q_e4, declared_anchors)

        interior = contract.resolve(0, 0.4237)
        self.assertEqual(interior.q_e4, 4237)
        self.assertEqual(interior.quality.requested_q, 0.4237)
        self.assertEqual(interior.quality.q_exec, 0.4237)
        self.assertFalse(interior.quality.was_clipped)
        self.assertFalse(interior.is_registered_anchor)
        self.assertIsNone(interior.anchor)
        self.assertIsNone(interior.action_id)
        self.assertIsNone(interior.profile_id)
        # still fully executable: complete mode identity and exact keep/drop
        self.assertEqual(interior.mode, contract.mode(0))
        self.assertEqual(interior.execution_mode, ac.EXECUTION_MODE)
        expected_keep, expected_drop = ac.keep_drop_counts(4237)
        self.assertEqual(interior.keep_count, expected_keep)
        self.assertEqual(interior.drop_count, expected_drop)
        self.assertEqual(interior.keep_count + interior.drop_count, ac.SPATIAL_CELLS)

        # --- non-anchor lookup returns no action/profile ID, for every mode #
        for mode in contract.modes:
            anchor_q = {a.q_e4 for a in contract.anchors_for_mode(mode.mode_id)}
            self.assertEqual(anchor_q, set(declared_anchors))
            for q_e4 in non_anchor_q_e4:
                self.assertIsNone(
                    contract.find_anchor(mode.family, mode.quantizer, q_e4),
                    f"{mode.canonical} fabricated an anchor for q_e4={q_e4}",
                )
                executed = contract.resolve(mode.mode_id, q_e4 / ac.Q_E4_SCALE)
                self.assertEqual(executed.q_e4, q_e4)
                self.assertIsNone(executed.action_id)
                self.assertIsNone(executed.profile_id)
                self.assertIsNone(executed.anchor)
                self.assertFalse(executed.is_registered_anchor)
            # registered anchors still resolve, so the None above is not blanket
            for q_e4 in declared_anchors:
                found = contract.find_anchor(mode.family, mode.quantizer, q_e4)
                self.assertIsNotNone(found)
                self.assertEqual(found.q_e4, q_e4)

        # --- no nearest-anchor substitution near an anchor ------------- #
        for anchor_q_e4 in declared_anchors:
            for delta in (-1, 1):
                neighbour = anchor_q_e4 + delta
                if not ac.Q_E4_MIN <= neighbour <= ac.Q_E4_MAX:
                    continue
                if neighbour in declared_anchors:
                    continue
                for mode in contract.modes:
                    self.assertIsNone(
                        contract.find_anchor(
                            mode.family, mode.quantizer, neighbour
                        ),
                        f"{mode.canonical} substituted the nearest anchor for "
                        f"q_e4={neighbour}",
                    )
                    executed = contract.resolve(
                        mode.mode_id, neighbour / ac.Q_E4_SCALE
                    )
                    self.assertEqual(executed.q_e4, neighbour)
                    self.assertIsNone(executed.profile_id)
                    self.assertIsNone(executed.action_id)
                    # the executed keep/drop is the exact continuous value, not
                    # the neighbouring anchor's measured value
                    anchor = contract.find_anchor(
                        mode.family, mode.quantizer, anchor_q_e4
                    )
                    expected = ac.keep_drop_counts(neighbour)
                    self.assertEqual(
                        (executed.keep_count, executed.drop_count), expected
                    )
                    if expected[1] != anchor.drop_count:
                        self.assertNotEqual(
                            executed.drop_count, anchor.drop_count
                        )

        # --- lookups on an undeclared mode still fail closed ----------- #
        with self.assertRaises(ac.UnknownJointModeError):
            contract.find_anchor("noAE", "UINT5", 3000)
        with self.assertRaises(ac.UnknownJointModeError):
            contract.resolve(ac.EXPECTED_MODE_COUNT, 0.5)

        # --- value objects are immutable ------------------------------- #
        with self.assertRaises(dataclasses.FrozenInstanceError):
            interior.anchor = None  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            interior.quality.q_e4 = 0  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
