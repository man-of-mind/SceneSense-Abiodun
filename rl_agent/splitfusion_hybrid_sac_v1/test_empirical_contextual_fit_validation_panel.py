"""Focused tests for the deterministic 340-entry fit-validation panel."""

from __future__ import annotations

import hashlib
import unittest
from collections import Counter
from dataclasses import fields

from . import empirical_contextual_fit_validation_panel as panel_module
from .anchor_store import NETWORK_PROFILE_ORDER
from .empirical_contextual_fit_partition import (
    FIT_VALIDATION_SPLIT,
    REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
    load_registered_empirical_fit_partition,
)


class EmpiricalFitValidationPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partition = load_registered_empirical_fit_partition()
        cls.panel = panel_module.load_registered_fit_validation_panel()

    def test_exact_cross_and_balanced_round_robin_counts(self) -> None:
        self.assertEqual(len(self.panel.entries), 340)
        self.assertEqual(
            len({row.scene_sample_id for row in self.panel.entries}), 85
        )
        for profile in NETWORK_PROFILE_ORDER:
            rows = [row for row in self.panel.entries if row.network_profile == profile]
            self.assertEqual(len(rows), 85)
            usage = Counter(row.radio_csv_row_number for row in rows)
            self.assertEqual(len(usage), 20)
            self.assertEqual(sorted(usage.values()), [4] * 15 + [5] * 5)

    def test_every_entry_joins_exactly_to_the_registered_partition(self) -> None:
        scenes = {row.sample_id: row for row in self.partition.scene_assignments}
        radios = {row.csv_row_number: row for row in self.partition.radio_assignments}
        for entry in self.panel.entries:
            scene = scenes[entry.scene_sample_id]
            self.assertEqual(
                (entry.scene_episode_id, entry.scene_frame_id, entry.scene_split),
                (scene.episode_id, scene.frame_id, scene.split),
            )
            radio = radios[entry.radio_csv_row_number]
            self.assertEqual(
                (
                    entry.network_profile,
                    entry.radio_trace_id,
                    entry.radio_trace_step_index,
                    entry.radio_row_sha256,
                    entry.radio_split,
                ),
                (
                    radio.network_profile,
                    radio.trace_id,
                    radio.trace_step_index,
                    radio.row_sha256,
                    radio.split,
                ),
            )
            self.assertEqual(entry.scene_split, FIT_VALIDATION_SPLIT)
            self.assertEqual(entry.radio_split, FIT_VALIDATION_SPLIT)
            self.assertEqual(
                entry.fit_partition_sha256,
                REGISTERED_EMPIRICAL_FIT_PARTITION_SHA256,
            )
            self.assertEqual(
                entry.d1_pilot_binding_sha256,
                self.partition.source_bindings["d1_pilot_binding_sha256"],
            )
            self.assertEqual(
                entry.source_bindings_sha256,
                self.panel.source_bindings_sha256,
            )

    def test_assignment_field_inventory_is_identity_only(self) -> None:
        self.assertEqual(
            self.panel.assignment_input_fields,
            panel_module.PANEL_ASSIGNMENT_INPUT_FIELDS,
        )
        inventory = " ".join(self.panel.assignment_input_fields).lower()
        for forbidden in (
            "reward",
            "quality",
            "payload",
            "latency",
            "delivery",
            "action",
            "outcome",
        ):
            self.assertNotIn(forbidden, inventory)
        entry_fields = {item.name for item in fields(panel_module.FitValidationPanelEntryV1)}
        for forbidden in ("reward", "quality", "payload", "latency", "delivery", "action", "outcome"):
            self.assertFalse(any(forbidden in name for name in entry_fields))

    def test_rebuild_is_byte_identical_and_registered_hash_matches(self) -> None:
        rebuilt = panel_module._build_panel(self.partition)
        self.assertEqual(self.panel, rebuilt)
        self.assertEqual(self.panel.canonical_bytes(), rebuilt.canonical_bytes())
        self.assertEqual(
            hashlib.sha256(self.panel.canonical_bytes()).hexdigest(),
            panel_module.REGISTERED_FIT_VALIDATION_PANEL_SHA256,
        )
        self.assertEqual(
            self.panel.canonical_sha256(),
            panel_module.REGISTERED_FIT_VALIDATION_PANEL_SHA256,
        )

    def test_round_robin_is_exact_per_profile(self) -> None:
        for profile in NETWORK_PROFILE_ORDER:
            rows = [row for row in self.panel.entries if row.network_profile == profile]
            identities = sorted(
                {
                    (
                        row.radio_trace_id,
                        row.radio_trace_step_index,
                        row.radio_csv_row_number,
                        row.radio_row_sha256,
                    )
                    for row in rows
                }
            )
            for row in rows:
                self.assertEqual(row.radio_rank, row.scene_rank % 20)
                self.assertEqual(
                    (
                        row.radio_trace_id,
                        row.radio_trace_step_index,
                        row.radio_csv_row_number,
                        row.radio_row_sha256,
                    ),
                    identities[row.scene_rank % 20],
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
