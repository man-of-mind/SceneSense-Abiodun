"""Focused tests for the Hybrid-SAC measured-anchor evidence store.

The store is an evidence binder, so the tests are organized around the things
that make bound evidence trustworthy rather than around methods:

* inventory reconciles exactly (72 / 12x6 / 288 / four-per-action);
* both sources are hash-bound, and rows are individually digested;
* action identity reconciles with the frozen catalog;
* action-level quality is verified to be profile-independent;
* zero-delivery cells survive as explicit measured outcomes;
* an unobserved latency stays missing and never becomes ``0.0``;
* an off-anchor ``q`` is refused with ``UNSUPPORTED_COUNTERFACTUAL``; and
* canonical serialization is deterministic.

Structural-failure cases call :meth:`AnchorEvidenceStore._bind` directly with
mutated copies of the real rows.  Going through :meth:`from_paths` would stop
at the SHA-256 pin, which is itself tested separately; ``_bind`` is where the
inventory and identity gates live.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from . import action_contract as ac
from . import anchor_store as st


def _thawed(rows):
    """Return mutable dict copies of immutable source rows."""
    return [dict(row) for row in rows]


class AnchorStoreTestBase(unittest.TestCase):
    """Shared one-time binding of the real, pinned evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ac.load_contract()
        cls.store = st.load_anchor_store(contract=cls.contract)
        root = st.default_project_root()
        cls.summary_path = root / st.ACTION_SUMMARY_RELATIVE_PATH
        cls.latency_path = root / st.PROFILE_LATENCY_RELATIVE_PATH
        (
            cls.summary_sha,
            cls.summary_header,
            cls.summary_rows,
        ) = st._read_pinned_csv(
            cls.summary_path, st.ACTION_SUMMARY_SHA256, st.SOURCE_ACTION_SUMMARY
        )
        (
            cls.latency_sha,
            cls.latency_header,
            cls.latency_rows,
        ) = st._read_pinned_csv(
            cls.latency_path,
            st.PROFILE_LATENCY_SHA256,
            st.SOURCE_PROFILE_LATENCY,
        )

    def rebind(self, summary_rows=None, latency_rows=None):
        """Re-run ``_bind`` with optionally mutated rows, skipping the pin."""
        return st.AnchorEvidenceStore._bind(
            self.summary_path,
            self.summary_sha,
            self.summary_header,
            self.summary_rows if summary_rows is None else summary_rows,
            self.latency_path,
            self.latency_sha,
            self.latency_header,
            self.latency_rows if latency_rows is None else latency_rows,
            self.contract,
        )


class InventoryTest(AnchorStoreTestBase):
    """The registered inventory must reconcile exactly, or not bind at all."""

    def test_binds_exactly_72_anchors_12_modes_and_288_cells(self) -> None:
        store = self.store
        self.assertEqual(store.anchor_count, st.EXPECTED_PROFILE_COUNT)
        self.assertEqual(store.anchor_count, 72)
        self.assertEqual(store.cell_count, st.EXPECTED_CELL_COUNT)
        self.assertEqual(store.cell_count, 288)

        action_ids = [record.action_id for record in store.records]
        self.assertEqual(sorted(action_ids), list(range(72)))
        self.assertEqual(len(set(action_ids)), 72, "duplicate action anchor")

        keys = [record.key for record in store.records]
        self.assertEqual(len(set(keys)), 72, "duplicate (family, quantizer, q_e4)")

        # 12 family-quantizer modes, each carrying exactly the six registered
        # q anchors -- 12 x 6 reconciles to all 72 catalog actions.
        modes = {(r.quality.family, r.quality.quantizer) for r in store.records}
        self.assertEqual(len(modes), st.EXPECTED_MODE_COUNT)
        self.assertEqual(len(modes), 12)
        covered = set()
        for mode_id in range(store.contract.mode_count):
            mode_records = store.records_for_mode(mode_id)
            self.assertEqual(len(mode_records), st.EXPECTED_Q_ANCHOR_COUNT)
            anchors = [record.quality.q_e4 for record in mode_records]
            self.assertEqual(anchors, sorted(anchors), "anchors not q-ordered")
            self.assertEqual(set(anchors), set(st.REGISTERED_Q_ANCHORS_E4))
            covered.update(record.action_id for record in mode_records)
        self.assertEqual(covered, set(range(72)), "12x6 does not cover all 72")

        # Exactly four network profiles per action, no more and no fewer.
        for record in store.records:
            self.assertEqual(
                sorted(record.profiles), sorted(st.NETWORK_PROFILE_ORDER)
            )
            self.assertEqual(
                len(record.profiles), st.EXPECTED_NETWORK_PROFILE_COUNT
            )
            for profile in st.NETWORK_PROFILE_ORDER:
                outcome = record.outcome(profile)
                self.assertEqual(outcome.network_profile, profile)
                self.assertEqual(outcome.action_id, record.action_id)

    def test_missing_action_row_is_rejected(self) -> None:
        rows = [row for row in self.summary_rows if row["action_id"] != "37"]
        self.assertEqual(len(rows), 71)
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(summary_rows=rows)
        self.assertIn("37", str(caught.exception))

    def test_duplicate_action_row_is_rejected(self) -> None:
        rows = _thawed(self.summary_rows)
        rows.append(dict(rows[5]))
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(summary_rows=rows)
        self.assertIn("duplicate action_id", str(caught.exception))

    def test_foreign_action_row_is_rejected(self) -> None:
        rows = _thawed(self.summary_rows)
        foreign = dict(rows[0])
        foreign["action_id"] = "72"
        rows.append(foreign)
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(summary_rows=rows)
        self.assertIn("72", str(caught.exception))

    def test_missing_cell_is_rejected(self) -> None:
        rows = [
            row
            for row in self.latency_rows
            if not (row["action_id"] == "4" and row["network_profile"] == "MID_VARIABLE")
        ]
        self.assertEqual(len(rows), 287)
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(latency_rows=rows)
        self.assertIn("MID_VARIABLE", str(caught.exception))

    def test_duplicate_cell_is_rejected(self) -> None:
        rows = _thawed(self.latency_rows)
        rows.append(dict(rows[11]))
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(latency_rows=rows)
        self.assertIn("duplicate cell key", str(caught.exception))

    def test_foreign_network_profile_is_rejected(self) -> None:
        rows = _thawed(self.latency_rows)
        rows[0]["network_profile"] = "GLORIOUS_STABLE"
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(latency_rows=rows)
        self.assertIn("foreign network_profile", str(caught.exception))

    def test_absent_registered_column_fails_closed(self) -> None:
        header = tuple(
            name for name in self.latency_header if name != "network_p50_ms"
        )
        with self.assertRaises(st.EvidenceIntegrityError) as caught:
            st.AnchorEvidenceStore._bind(
                self.summary_path,
                self.summary_sha,
                self.summary_header,
                self.summary_rows,
                self.latency_path,
                self.latency_sha,
                header,
                self.latency_rows,
                self.contract,
            )
        self.assertIn("network_p50_ms", str(caught.exception))


class HashBindingTest(AnchorStoreTestBase):
    """Both sources are pinned, and every row carries its own digest."""

    def test_source_pins_match_the_bytes_on_disk(self) -> None:
        for path, pinned in (
            (self.summary_path, st.ACTION_SUMMARY_SHA256),
            (self.latency_path, st.PROFILE_LATENCY_SHA256),
        ):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, pinned, f"pin drifted for {path}")
        self.assertEqual(self.store.action_summary_sha256, st.ACTION_SUMMARY_SHA256)
        self.assertEqual(
            self.store.profile_latency_sha256, st.PROFILE_LATENCY_SHA256
        )

    def test_hash_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            drifted = Path(tmp) / "action_72_summary.csv"
            drifted.write_bytes(self.summary_path.read_bytes() + b"\n")
            with self.assertRaises(st.EvidenceIntegrityError) as caught:
                st.load_anchor_store(
                    action_summary_path=drifted, contract=self.contract
                )
            message = str(caught.exception)
            self.assertIn("SHA-256 mismatch", message)
            self.assertIn(st.ACTION_SUMMARY_SHA256, message)

    def test_unreadable_source_fails_closed(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(st.EvidenceIntegrityError):
                st.load_anchor_store(
                    action_summary_path=Path(tmp) / "absent.csv",
                    contract=self.contract,
                )

    def test_every_record_carries_source_and_row_digests(self) -> None:
        row_digests = set()
        for record in self.store.records:
            quality = record.quality
            self.assertEqual(quality.source_sha256, st.ACTION_SUMMARY_SHA256)
            self.assertEqual(len(quality.source_row_sha256), 64)
            row_digests.add(quality.source_row_sha256)
            for profile in st.NETWORK_PROFILE_ORDER:
                outcome = record.outcome(profile)
                self.assertEqual(
                    dict(outcome.source_sha256),
                    {
                        st.SOURCE_ACTION_SUMMARY: st.ACTION_SUMMARY_SHA256,
                        st.SOURCE_PROFILE_LATENCY: st.PROFILE_LATENCY_SHA256,
                    },
                )
                cell_digest = outcome.source_row_sha256[
                    st.SOURCE_PROFILE_LATENCY
                ]
                self.assertEqual(len(cell_digest), 64)
                row_digests.add(cell_digest)
        # 72 distinct action rows + 288 distinct cell rows.
        self.assertEqual(len(row_digests), 360)

    def test_row_digest_is_deterministic_and_field_sensitive(self) -> None:
        row = dict(self.summary_rows[0])
        header_sha = st.canonical_sha256(list(self.summary_header))
        first = st._row_sha256(st.SOURCE_ACTION_SUMMARY, header_sha, row)
        again = st._row_sha256(
            st.SOURCE_ACTION_SUMMARY, header_sha, dict(reversed(list(row.items())))
        )
        self.assertEqual(first, again, "row digest depends on column order")

        mutated = dict(row)
        mutated["val_vehicle_iou"] = mutated["val_vehicle_iou"] + "0"
        self.assertNotEqual(
            first,
            st._row_sha256(st.SOURCE_ACTION_SUMMARY, header_sha, mutated),
            "row digest ignored a changed character",
        )


class IdentityReconciliationTest(AnchorStoreTestBase):
    """Action identity must reconcile with the frozen 72-action catalog."""

    def test_every_record_reconciles_with_the_frozen_catalog(self) -> None:
        by_id = {anchor.action_id: anchor for anchor in self.contract.anchors}
        self.assertEqual(len(by_id), 72)
        for record in self.store.records:
            anchor = by_id[record.action_id]
            quality = record.quality
            self.assertEqual(quality.profile_id, anchor.profile_id)
            self.assertEqual(quality.family, anchor.mode.family)
            self.assertEqual(quality.quantizer, anchor.mode.quantizer)
            self.assertEqual(quality.mode_id, anchor.mode.mode_id)
            self.assertEqual(quality.q_e4, anchor.q_e4)
            self.assertEqual(quality.q, anchor.q)
            self.assertEqual(record.key, anchor.key)
            payload = quality.payload_identity
            self.assertEqual(payload["keep_count"], anchor.keep_count)
            self.assertEqual(payload["drop_count"], anchor.drop_count)
            self.assertEqual(
                payload["keep_count"] + payload["drop_count"], ac.SPATIAL_CELLS
            )
            self.assertEqual(payload["spatial_cells"], ac.SPATIAL_CELLS)

    def test_action_identity_drift_is_rejected(self) -> None:
        rows = _thawed(self.summary_rows)
        rows[0]["family"] = "AE32"
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(summary_rows=rows)
        self.assertIn("does not reconcile", str(caught.exception))

    def test_cell_identity_drift_is_rejected(self) -> None:
        rows = _thawed(self.latency_rows)
        rows[0]["profile_id"] = "split_ae32_uint4_q9800"
        with self.assertRaises(st.EvidenceInventoryError) as caught:
            self.rebind(latency_rows=rows)
        self.assertIn("does not reconcile", str(caught.exception))

    def test_cell_q_drift_is_rejected(self) -> None:
        rows = _thawed(self.latency_rows)
        rows[0]["q"] = "0.5"
        with self.assertRaises(st.EvidenceIntegrityError) as caught:
            self.rebind(latency_rows=rows)
        self.assertIn("catalog anchor", str(caught.exception))


class QualitySeparationTest(AnchorStoreTestBase):
    """Action-level quality is kept apart from network-profile outcomes."""

    def test_quality_is_action_level_and_payload_identity_preserved(self) -> None:
        record = self.store.lookup("noAE", "UINT8", 0.0)
        quality = record.quality

        # Raw quality fields survive verbatim, including non-numeric evidence
        # and deliberately empty fields.
        self.assertEqual(len(quality.raw_quality), 45)
        self.assertEqual(quality.raw_quality["val_preservation_gates_passed_total"], "12/12")
        self.assertEqual(len(quality.raw_quality["val_checkpoint_sha256"]), 64)
        self.assertEqual(quality.raw_quality["val_mask_accuracy"], "")

        # The typed view parses the numeric subset and leaves empties missing.
        self.assertAlmostEqual(
            quality.quality_metrics["val_vehicle_iou"], 0.898997205787936
        )
        self.assertIsNone(quality.quality_metrics["val_mask_accuracy"])
        self.assertNotIn("val_checkpoint_sha256", quality.quality_metrics)

        # Derived presentation coordinates are kept in a separate block, not
        # mixed into the raw measured fields.
        self.assertIn("combined_quality", quality.derived_presentation_quality)
        self.assertNotIn("combined_quality", quality.raw_quality)

        # Payload identity is action-level; measured bytes are cell-level.
        self.assertEqual(quality.payload_identity["bit_width"], 8)
        self.assertIsNone(quality.payload_identity["latent_width"])
        for profile in st.NETWORK_PROFILE_ORDER:
            outcome = record.outcome(profile)
            self.assertIn(
                "replay_v3__median_payload_bytes", outcome.measured_payload_bytes
            )
            self.assertNotIn("bit_width", outcome.measured_payload_bytes)

    def test_profile_dependent_quality_is_rejected(self) -> None:
        rows = _thawed(self.latency_rows)
        rows[0]["val_vehicle_iou"] = "0.5"
        with self.assertRaises(st.EvidenceIntegrityError) as caught:
            self.rebind(latency_rows=rows)
        message = str(caught.exception)
        self.assertIn("val_vehicle_iou", message)
        self.assertIn("must not depend on", message)

    def test_authored_profile_label_is_not_a_policy_observation(self) -> None:
        outcome = self.store.by_action_id(0).outcome("ADVERSE_STABLE")
        full = outcome.to_canonical_dict()
        self.assertEqual(full["network_profile"], "ADVERSE_STABLE")
        with self.assertRaises(st.PolicyObservationLeakError):
            st.assert_no_forbidden_policy_observation(full)

        safe = outcome.policy_safe_dict()
        self.assertNotIn("network_profile", safe)
        self.assertNotIn("cell_id", safe)
        st.assert_no_forbidden_policy_observation(safe)
        # The measured content itself is untouched by the guard.
        self.assertEqual(safe["counts"], full["counts"])


class ZeroDeliveryTest(AnchorStoreTestBase):
    """Zero-delivery cells are measured outcomes, not gaps in the evidence."""

    def test_zero_delivery_cells_are_explicit_and_cross_consistent(self) -> None:
        live_zero = []
        replay_zero = []
        for record in self.store.records:
            for profile in st.NETWORK_PROFILE_ORDER:
                outcome = record.outcome(profile)
                if outcome.zero_delivery_live_campaign:
                    live_zero.append((record.action_id, profile))
                if outcome.zero_admission_replay_v3:
                    replay_zero.append((record.action_id, profile))

        # Both zero-delivery notions are present and distinct.
        self.assertEqual(len(live_zero), 66)
        self.assertEqual(len(replay_zero), 46)
        self.assertTrue(
            set(replay_zero) < set(live_zero),
            "the permissive v3 replay must be a strict subset of live zeros",
        )

        for action_id, profile in live_zero:
            outcome = self.store.by_action_id(action_id).outcome(profile)
            # A zero-delivery cell is still a fully measured cell: frames were
            # sent, terminals account for every one of them, and the absent
            # latency is absent rather than zero.
            self.assertGreater(outcome.frames_sent, 0)
            self.assertEqual(
                sum(outcome.terminal_counts.values()), outcome.frames_sent
            )
            self.assertEqual(outcome.counts["live__maps_installed"], 0)
            self.assertFalse(outcome.latency_stat("network").observed)
            self.assertIsNone(outcome.latency_stat("network").p50_ms)
            self.assertFalse(outcome.latency_stat("install_aoi_ms").observed)

    def test_delivering_cells_report_positive_support(self) -> None:
        delivering = [
            outcome
            for record in self.store.records
            for outcome in (
                record.outcome(profile) for profile in st.NETWORK_PROFILE_ORDER
            )
            if not outcome.zero_delivery_live_campaign
        ]
        self.assertEqual(len(delivering), 288 - 66)
        for outcome in delivering:
            self.assertGreater(outcome.counts["live__maps_installed"], 0)
            self.assertTrue(outcome.latency_stat("network").observed)
            self.assertIsNotNone(outcome.latency_stat("network").p50_ms)

    def test_inconsistent_zero_delivery_flag_is_rejected(self) -> None:
        rows = _thawed(self.summary_rows)
        rows[0]["zero_delivery__FAVORABLE_STABLE"] = "False"
        with self.assertRaises(st.EvidenceIntegrityError) as caught:
            self.rebind(summary_rows=rows)
        self.assertIn("zero_delivery", str(caught.exception))


class MissingLatencyTest(AnchorStoreTestBase):
    """An unobserved latency stays missing; it never becomes ``0.0``."""

    def test_unobserved_stages_are_none_and_observed_stages_are_complete(
        self,
    ) -> None:
        unobserved = 0
        observed = 0
        for record in self.store.records:
            for profile in st.NETWORK_PROFILE_ORDER:
                outcome = record.outcome(profile)
                self.assertEqual(
                    len(outcome.latency), len(st._B_LATENCY_STAGES) + 1
                )
                for stage, stat in outcome.latency.items():
                    values = list(stat.percentiles_ms.values())
                    if stat.support == 0:
                        unobserved += 1
                        self.assertTrue(
                            all(value is None for value in values),
                            f"{stage} has zero support but carries {values}",
                        )
                    else:
                        observed += 1
                        self.assertTrue(
                            all(value is not None for value in values),
                            f"{stage} has support {stat.support} but is missing "
                            f"a percentile",
                        )
        self.assertGreater(unobserved, 0)
        self.assertGreater(observed, 0)

    def test_absent_scalar_measurement_is_none_not_zero(self) -> None:
        # The campaign recorded no achieved-SNR median in any of the 288 cells.
        for record in self.store.records:
            for profile in st.NETWORK_PROFILE_ORDER:
                value = record.outcome(profile).measured_payload_bytes[
                    "live__radio_achieved_snr_db_median"
                ]
                self.assertIsNone(value)
                self.assertNotEqual(value, 0.0)

    def test_latency_stat_rejects_support_value_disagreement(self) -> None:
        with self.assertRaises(st.EvidenceIntegrityError):
            st.LatencyStat(
                stage="network",
                source=st.SOURCE_PROFILE_LATENCY,
                support=0,
                percentiles_ms={"p50": 0.0},
            )
        with self.assertRaises(st.EvidenceIntegrityError):
            st.LatencyStat(
                stage="network",
                source=st.SOURCE_PROFILE_LATENCY,
                support=12,
                percentiles_ms={"p50": None},
            )
        with self.assertRaises(st.EvidenceIntegrityError):
            st.LatencyStat(
                stage="network",
                source=st.SOURCE_PROFILE_LATENCY,
                support=-1,
                percentiles_ms={},
            )

    def test_latency_support_drift_is_rejected(self) -> None:
        rows = _thawed(self.latency_rows)
        rows[0]["network_count"] = "5"
        with self.assertRaises(st.EvidenceIntegrityError):
            self.rebind(latency_rows=rows)


class ExactAnchorLookupTest(AnchorStoreTestBase):
    """Lookup is exact: no snapping, no interpolation, no extrapolation."""

    def test_all_72_registered_anchors_resolve_exactly(self) -> None:
        seen = set()
        for anchor in self.contract.anchors:
            record = self.store.lookup(
                anchor.mode.family, anchor.mode.quantizer, anchor.q
            )
            self.assertEqual(record.action_id, anchor.action_id)
            by_wire = self.store.lookup_by_q_e4(
                anchor.mode.family, anchor.mode.quantizer, anchor.q_e4
            )
            self.assertIs(record, by_wire)
            seen.add(record.action_id)
        self.assertEqual(len(seen), 72)
        # Equivalent exact spellings of the same anchor all resolve.
        target = self.store.lookup("AE32", "UINT4", 0.3)
        for spelling in (Decimal("0.3"), Decimal("0.3000"), "0.30"):
            self.assertIs(self.store.lookup("AE32", "UINT4", spelling), target)
        self.assertIs(self.store.lookup("AE32", "UINT4", 0), self.store.lookup("AE32", "UINT4", 0.0))

    def test_off_anchor_q_raises_unsupported_counterfactual(self) -> None:
        for q in (0.1, 0.2, 0.35, 0.45, 0.6, 0.75, 0.95, 0.97, 0.5001, 0.001):
            with self.assertRaises(st.UnsupportedCounterfactualError) as caught:
                self.store.lookup("noAE", "UINT8", q)
            message = str(caught.exception)
            self.assertIn(st.UNSUPPORTED_COUNTERFACTUAL, message)
            self.assertEqual(
                caught.exception.code, st.UNSUPPORTED_COUNTERFACTUAL
            )
            self.assertIn("does not snap", message)

    def test_near_anchor_q_is_not_snapped(self) -> None:
        # The execution path rounds half-up to the wire grid, so 0.30001 would
        # *transmit* as the 3000 anchor.  The evidence store must still refuse
        # it: rounding chooses what to send, it does not create a measurement.
        self.assertEqual(self.contract.quality_for(0.30001).q_e4, 3000)
        self.assertIsNone(st.exact_q_e4(0.30001))
        with self.assertRaises(st.UnsupportedCounterfactualError):
            self.store.lookup("noAE", "UINT8", 0.30001)

        # Float error must not be laundered into an anchor hit either.
        self.assertNotEqual(0.1 + 0.2, 0.3)
        self.assertIsNone(st.exact_q_e4(0.1 + 0.2))
        with self.assertRaises(st.UnsupportedCounterfactualError):
            self.store.lookup("noAE", "UINT8", 0.1 + 0.2)

    def test_out_of_range_and_malformed_q_are_refused(self) -> None:
        for q in (-0.1, 1.0, 1.5, 9800):
            with self.assertRaises(st.UnsupportedCounterfactualError):
                self.store.lookup("noAE", "UINT8", q)
        for q in (None, True, float("nan"), float("inf"), "not-a-number"):
            with self.assertRaises(st.UnsupportedCounterfactualError):
                self.store.lookup("noAE", "UINT8", q)

    def test_undeclared_mode_is_a_distinct_contract_violation(self) -> None:
        # An undeclared mode is a contract violation, not a legitimate
        # unmeasured continuous action, so it must not be reported as an
        # unsupported counterfactual.
        for family, quantizer in (
            ("AE16", "UINT8"),
            ("noAE", "UINT2"),
            ("LOCAL", "UINT8"),
        ):
            with self.assertRaises(ac.UnknownJointModeError):
                self.store.lookup(family, quantizer, 0.0)

    def test_unknown_action_id_is_a_key_error(self) -> None:
        self.assertEqual(self.store.by_action_id(71).action_id, 71)
        with self.assertRaises(KeyError):
            self.store.by_action_id(72)


class EvidenceLabellingTest(AnchorStoreTestBase):
    """Every record announces what it is and refuses to be a transition."""

    def test_records_are_labelled_measured_anchor_aggregate(self) -> None:
        for record in self.store.records:
            self.assertEqual(record.evidence_class, "MEASURED_ANCHOR_AGGREGATE")
            self.assertEqual(record.evidence_class, st.EVIDENCE_CLASS)
            self.assertFalse(record.replay_admissible)
            self.assertIn("NOT a per-frame causal transition", record.evidence_use_restriction)
            self.assertIn("ReplayTransitionV1", record.replay_restriction)
            serialized = record.to_canonical_dict()
            self.assertEqual(serialized["evidence_class"], st.EVIDENCE_CLASS)
            self.assertFalse(serialized["replay_admissible"])
            self.assertIn("ReplayTransitionV1", serialized["replay_restriction"])

    def test_replay_insertion_is_refused(self) -> None:
        record = self.store.by_action_id(0)
        with self.assertRaises(st.ReplayInsertionForbiddenError) as caught:
            record.as_replay_transition()
        self.assertIn("ReplayTransitionV1", str(caught.exception))

    def test_store_serialization_declares_its_provenance(self) -> None:
        document = self.store.to_canonical_dict()
        self.assertEqual(document["schema"], st.STORE_SCHEMA_ID)
        self.assertEqual(document["evidence_class"], st.EVIDENCE_CLASS)
        self.assertEqual(
            document["catalog_sha256"], self.contract.catalog_sha256
        )
        self.assertEqual(
            document["sources"][st.SOURCE_ACTION_SUMMARY]["sha256"],
            st.ACTION_SUMMARY_SHA256,
        )
        self.assertEqual(
            document["sources"][st.SOURCE_PROFILE_LATENCY]["sha256"],
            st.PROFILE_LATENCY_SHA256,
        )
        self.assertEqual(document["inventory"]["anchor_count"], 72)
        self.assertEqual(document["inventory"]["cell_count"], 288)
        self.assertEqual(document["inventory"]["mode_count"], 12)
        self.assertEqual(len(document["records"]), 72)


class DeterministicSerializationTest(AnchorStoreTestBase):
    """Canonical serialization is byte-stable across independent binds."""

    def test_independent_binds_serialize_identically(self) -> None:
        other = st.load_anchor_store(contract=ac.load_contract())
        self.assertIsNot(other, self.store)
        self.assertEqual(other.canonical_bytes(), self.store.canonical_bytes())
        self.assertEqual(other.canonical_sha256(), self.store.canonical_sha256())

    def test_repeated_serialization_is_stable(self) -> None:
        first = self.store.canonical_bytes()
        self.assertEqual(first, self.store.canonical_bytes())
        self.assertEqual(
            hashlib.sha256(first).hexdigest(), self.store.canonical_sha256()
        )

    def test_canonical_form_is_sorted_compact_and_nan_free(self) -> None:
        document = self.store.to_canonical_dict()
        raw = self.store.canonical_bytes()
        # Sorted keys, compact separators, ASCII-escaped, NaN/Infinity refused.
        self.assertEqual(
            raw,
            json.dumps(
                document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8"),
        )
        raw.decode("ascii")
        self.assertNotIn(b"NaN", raw)
        self.assertNotIn(b"Infinity", raw)
        self.assertEqual(
            list(document["records"][0]["profiles"]),
            sorted(st.NETWORK_PROFILE_ORDER),
        )
        action_ids = [record["quality"]["action_id"] for record in document["records"]]
        self.assertEqual(action_ids, list(range(72)))

    def test_record_digests_are_stable_and_unique(self) -> None:
        digests = {record.action_id: record.canonical_sha256() for record in self.store.records}
        self.assertEqual(len(set(digests.values())), 72)
        other = st.load_anchor_store(contract=self.contract)
        for record in other.records:
            self.assertEqual(record.canonical_sha256(), digests[record.action_id])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
