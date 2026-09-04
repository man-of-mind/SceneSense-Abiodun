"""Two focused CPU checks over the Phase-10B AE64/AE32 validation contract.

Nothing is inferred, trained, validated, selected or scored here, neither test
creates a CUDA context (each asserts the process-global flag is exactly where it
found it), and no dataset, teacher-cache shard or CARLA process is touched. The
two selected checkpoints are opened only to read the identity and source-map
fields they recorded -- their tensors are never built into a model and never
moved to a device.

1. family/token/artifact/routing separation: every emitted token, terminal,
   schema, filename, family id, latent width, analytical payload size, range
   byte count and routing tag is derived from `--bottleneck` and names its own
   family only; the two families disagree on all of them; a token/bottleneck
   mismatch and AE128 are both refused; each selected checkpoint and holdout
   decision is the bound artifact for its own family and cannot be read as the
   other's; and the source-map delta admits exactly the registered Phase-10B
   additions while refusing any change or removal.
2. the shared acceptance and durable-resume behaviour, run identically for both
   families: the reused Phase-9D rule applied verbatim with only the family
   terminal relabelled, the secondary prospective classification (including that
   stress q stay EMERGENCY_ONLY, that segmentation installability is separate
   from 12/12 preservation and that 9/9 service readiness is separate again),
   and the interruption window between a durable per-q setting and its cleanup.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.continuous_q import quantize_q
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import sha256_file
from .. import ae_contract, ae_phase10b_uint8_validation as validation
from .. import ae_phase10_common as family
from .. import ae_training_common as common
from .. import ae_uint8_validation as phase9d
from ..ae_gpu_qualification import ae_package_source_hashes


FAMILIES = family.AE_PHASE10_BOTTLENECKS  # (64, 32)
GATES = validation.GATE_COUNT  # 12
SERVICE_GATES = validation.SERVICE_GATE_COUNT  # 9
BASELINE = validation.BASELINE_SERVICE_PASS_COUNT  # 7 of 9


# ---------------------------------------------------------------------------
# 1. Family / token / artifact / routing separation
# ---------------------------------------------------------------------------


class Phase10bFamilySeparationChecks(unittest.TestCase):
    """One runner, two families, and no way for either to borrow the other."""

    def setUp(self) -> None:
        family._reset_process_family_for_tests()

    def tearDown(self) -> None:
        family._reset_process_family_for_tests()

    def test_every_family_quantity_is_derived_and_self_labelled(self) -> None:
        cuda_before = torch.cuda.is_initialized()

        self.assertEqual(FAMILIES, (64, 32))
        self.assertEqual(set(validation.SELECTED_ARTIFACTS), set(FAMILIES))

        # Every emitted string is family-labelled, and the emitters are the only
        # place a family label is produced.
        emitters = (
            validation.execute_token,
            validation.terminal,
            validation.setting_terminal,
            validation.cleanup_terminal,
            validation.accepted_terminal,
            validation.not_accepted_terminal,
            validation.schema,
            validation.setting_schema,
            validation.cleanup_schema,
            validation.result_json_filename,
            validation.result_csv_filename,
            validation.report_filename,
            validation.manifest_filename,
        )
        for emitter in emitters:
            emitted = {size: emitter(size) for size in FAMILIES}
            # Two families, two distinct strings, each naming only itself.
            self.assertEqual(len(set(emitted.values())), len(FAMILIES), emitter)
            for size, text in emitted.items():
                lowered = text.lower()
                self.assertIn(f"ae{size}", lowered, f"{emitter} {text}")
                for other in ae_contract.AE_BOTTLENECKS:
                    if int(other) != size:
                        self.assertNotIn(
                            f"ae{int(other)}", lowered, f"{emitter} {text}"
                        )
                # And it must survive the family-labelling guard it was built
                # with, so no emitter can bypass that check.
                family.require_family_labelled(text, size, what=str(emitter))

        # Latent geometry, payload accounting and routing are all derived.
        for size in FAMILIES:
            self.assertEqual(validation.latent_width(size), size)
            self.assertEqual(family.family_id(size), ae_contract.family_for_bottleneck(size))
            self.assertEqual(validation.range_bytes(size), size * 2 * 4)
            for q in validation.Q_VALUES:
                plan = quantize_q(float(q))
                analytical = validation.analytical_size(size, q)
                self.assertEqual(analytical.bottleneck, size)
                self.assertEqual(analytical.family_id, family.family_id(size))
                self.assertEqual(analytical.range_bytes, validation.range_bytes(size))
                self.assertEqual(analytical.value_bytes, plan.keep_count * size)
                self.assertEqual(
                    analytical.total_bytes,
                    analytical.header_bytes
                    + analytical.mask_bytes
                    + analytical.range_bytes
                    + analytical.value_bytes,
                )

        # AE64 and AE32 disagree on family id, latent width, range bytes, every
        # analytical payload size and the routing tag.
        self.assertNotEqual(family.family_id(64), family.family_id(32))
        self.assertNotEqual(validation.range_bytes(64), validation.range_bytes(32))
        self.assertNotEqual(validation.routing_tag(64), validation.routing_tag(32))
        for q in validation.Q_VALUES:
            self.assertNotEqual(
                validation.analytical_size(64, q).total_bytes,
                validation.analytical_size(32, q).total_bytes,
            )

        # The routing tag is the leading 32 bits of the *full* selected digest,
        # is nonzero, and is explicitly not the checkpoint's identity.
        for size in FAMILIES:
            row = validation.selected(size)
            tag = validation.routing_tag(size)
            self.assertEqual(
                tag,
                ae_contract.routing_tag_from_sha256(row["selected_checkpoint_sha256"]),
            )
            self.assertEqual(tag, int(row["selected_checkpoint_sha256"][:8], 16))
            self.assertNotEqual(tag, ae_contract.AE_UNBOUND_ROUTING_TAG)
            record = validation.routing_record(size)
            self.assertFalse(record["routing_tag_is_checkpoint_identity"])
            self.assertEqual(record["routing_tag_hex"], f"0x{tag:08x}")

        # AE128 is not constructible anywhere in this runner, and every public
        # emitter refuses it by name.
        for emitter in emitters + (
            validation.latent_width,
            validation.range_bytes,
            validation.routing_tag,
            validation.selected,
        ):
            with self.assertRaises(guards.HybridQConfigError):
                emitter(common.AE_TRAINING_BOTTLENECK)  # 128
        self.assertNotIn(128, validation.SELECTED_ARTIFACTS)
        self.assertNotIn(
            common.AE_TRAINING_BOTTLENECK, family.AE_PHASE10_BOTTLENECKS
        )

        # A token/bottleneck mismatch is refused, and so is a Phase-10A token.
        for size in FAMILIES:
            self.assertEqual(
                validation.require_token_agrees_with_bottleneck(
                    validation.execute_token(size), size
                ),
                size,
            )
            other = 32 if size == 64 else 64
            with self.assertRaises(guards.HybridQConfigError):
                validation.require_token_agrees_with_bottleneck(
                    validation.execute_token(size), other
                )
            with self.assertRaises(guards.HybridQConfigError):
                validation.require_token_agrees_with_bottleneck(
                    family.holdout_token(size), size
                )
            with self.assertRaises(guards.HybridQConfigError):
                validation.require_token_agrees_with_bottleneck(
                    family.training_token(size), size
                )
        with self.assertRaises(guards.HybridQConfigError):
            validation.require_token_agrees_with_bottleneck(
                phase9d.EXECUTE_TOKEN, 64
            )

        # The parser exposes exactly the two tokens and the two families, and no
        # bounded/smoke/frame-limiting option.
        parser = validation.build_parser()
        options = {
            action.dest: action for action in parser._actions if action.dest != "help"
        }
        self.assertEqual(
            sorted(options), ["bottleneck", "execute", "output", "resume", "workers"]
        )
        self.assertEqual(tuple(options["execute"].choices), validation.EXECUTE_TOKENS)
        self.assertEqual(tuple(options["bottleneck"].choices), FAMILIES)

        # One family per process: binding one refuses the other.
        family.bind_process_family(64)
        with self.assertRaises(guards.HybridQOwnershipError):
            family.bind_process_family(32)

        self.assertEqual(torch.cuda.is_initialized(), cuda_before)

    def test_each_family_binds_only_its_own_artifacts(self) -> None:
        cuda_before = torch.cuda.is_initialized()
        root = contract.repository_root()

        for size in FAMILIES:
            row = validation.selected(size)
            other = 32 if size == 64 else 64

            # The bound epoch is a registered candidate epoch, and the two
            # families were selected at different epochs of different runs.
            self.assertIn(int(row["selected_epoch"]), common.AE_CANDIDATE_EPOCHS)
            self.assertNotEqual(
                row["selected_checkpoint_relpath"],
                validation.selected(other)["selected_checkpoint_relpath"],
            )
            self.assertNotEqual(
                row["selected_checkpoint_sha256"],
                validation.selected(other)["selected_checkpoint_sha256"],
            )

            # Every artifact path is derived from the family's own registered
            # filename helpers, not spelled out twice.
            self.assertTrue(
                row["selected_checkpoint_relpath"].endswith(
                    f"/checkpoints/"
                    f"{family.candidate_filename(size, int(row['selected_epoch']))}"
                )
            )
            self.assertTrue(
                row["holdout_decision_relpath"].endswith(
                    f"/{family.holdout_selection_dirname(size)}/"
                    f"{family.holdout_report_filename(size)}"
                )
            )

            # Both bound artifacts exist at exactly the bound digest.
            checkpoint = (root / row["selected_checkpoint_relpath"]).resolve(strict=True)
            decision = (root / row["holdout_decision_relpath"]).resolve(strict=True)
            self.assertEqual(
                sha256_file(checkpoint), row["selected_checkpoint_sha256"]
            )
            self.assertEqual(sha256_file(decision), row["holdout_decision_sha256"])

            # The decision's own completion marker records that digest, so the
            # selector signed the document Phase 10B binds.
            marker = (root / row["holdout_decision_terminal_relpath"]).resolve(
                strict=True
            )
            self.assertEqual(
                marker.read_text(encoding="utf-8").strip(),
                row["holdout_decision_sha256"],
            )

            # The decision is this family's own, selected the bound epoch, and
            # recorded the bound checkpoint hash for it -- and is not readable as
            # the other family's decision.
            document = json.loads(decision.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], family.holdout_schema(size))
            self.assertEqual(document["terminal"], family.holdout_terminal(size))
            family.require_family_fields(
                document["scope"], size, what="decision scope"
            )
            with self.assertRaises(guards.HybridQConfigError):
                family.require_family_fields(
                    document["scope"], other, what="decision scope"
                )
            self.assertFalse(document["scope"]["validation_or_test_accessed"])
            self.assertFalse(document["scope"]["deployment_validation_performed_here"])
            self.assertEqual(
                int(document["selection"]["selected_epoch"]),
                int(row["selected_epoch"]),
            )
            self.assertFalse(
                document["selection"]["selection_is_a_service_ready_claim"]
            )
            self.assertEqual(
                document["training_run"]["candidate_checkpoints"][
                    row["selected_checkpoint_name"]
                ],
                row["selected_checkpoint_sha256"],
            )

            # The checkpoint itself declares this family and this epoch. Only
            # its metadata is read: no model is built and no tensor is moved.
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema"], family.candidate_schema(size))
            family.require_family_fields(payload, size, what=checkpoint.name)
            with self.assertRaises(guards.HybridQConfigError):
                family.require_family_fields(payload, other, what=checkpoint.name)
            self.assertEqual(int(payload["epoch"]), int(row["selected_epoch"]))
            self.assertEqual(int(payload["bottleneck"]), validation.latent_width(size))
            self.assertEqual(
                payload["configuration"], family.training_configuration(size)
            )

            # The AE package source delta admits exactly the registered
            # Phase-10B additions and nothing else.
            recorded = dict(payload["ae_package_source_sha256"])
            live = ae_package_source_hashes()
            delta = validation.ae_package_source_delta(recorded, live)
            self.assertEqual(delta["changed"], [])
            self.assertEqual(delta["removed"], [])
            self.assertEqual(delta["changed_files_allowed"], 0)
            self.assertTrue(delta["semantics_modules_unchanged"])
            for entry in delta["added"]:
                self.assertIn(
                    entry["path"], validation.AE_PHASE10B_ADDED_SOURCES
                )
            # This module is one of the registered additions.
            self.assertIn(
                "ae_phase10b_uint8_validation.py",
                validation.AE_PHASE10B_ADDED_SOURCES,
            )

            # A changed recorded file, a removed one, and an unregistered
            # addition are each refused rather than allowlisted.
            semantics = validation.AE_CHECKPOINT_SEMANTICS_SOURCES[0]
            with self.assertRaises(guards.HybridQConfigError):
                validation.ae_package_source_delta(
                    recorded, {**live, semantics: "0" * 64}
                )
            other_recorded = next(
                name
                for name in recorded
                if name
                not in validation.AE_CHECKPOINT_SEMANTICS_SOURCES
                + validation.AE_TRANSPORT_SEMANTICS_SOURCES
            )
            with self.assertRaises(guards.HybridQConfigError):
                validation.ae_package_source_delta(
                    recorded, {**live, other_recorded: "1" * 64}
                )
            with self.assertRaises(guards.HybridQConfigError):
                validation.ae_package_source_delta(
                    recorded,
                    {k: val for k, val in live.items() if k != other_recorded},
                )
            with self.assertRaises(guards.HybridQConfigError):
                validation.ae_package_source_delta(
                    recorded, {**live, "unregistered_new_file.py": "2" * 64}
                )
            del payload

        self.assertEqual(torch.cuda.is_initialized(), cuda_before)


# ---------------------------------------------------------------------------
# 2. Shared acceptance, secondary classification and durable resume
# ---------------------------------------------------------------------------


def acceptance_row(q: float, *, gates: int, service: int, noae_service: int) -> dict:
    """Exactly the fields the reused Phase-9D rule reads, and nothing else."""
    return {
        "q": float(q),
        "q_e4": quantize_q(float(q)).q_e4,
        "preservation_gates_passed": int(gates),
        "preservation_all_passed": int(gates) == GATES,
        "failed_preservation_gates": [] if int(gates) == GATES else ["vehicle_f1"],
        "absolute_service_pass_count": int(service),
        "failed_absolute_service_gates": [],
        "noae_same_q_absolute_service_pass_count": int(noae_service),
    }


# One synthetic ladder that satisfies the rule: q=0 keeps every gate and the
# baseline service count, q=0.30 keeps every gate without reducing the noAE
# service count, and the rest do not.
BASE_LADDER = {
    0.00: {"gates": GATES, "service": BASELINE, "noae_service": BASELINE},
    0.30: {"gates": GATES, "service": BASELINE, "noae_service": BASELINE},
    0.50: {"gates": GATES - 1, "service": 4, "noae_service": 4},
    0.70: {"gates": GATES - 1, "service": 3, "noae_service": 4},
    0.90: {"gates": GATES, "service": 3, "noae_service": 3},
    0.98: {"gates": GATES, "service": 3, "noae_service": 3},
}


def ladder(**overrides: dict) -> list[dict]:
    """The base ladder with `q<e4 slug>=dict(...)` overrides applied."""
    spec = {q: dict(values) for q, values in BASE_LADDER.items()}
    for slug, values in overrides.items():
        q = float(slug[1:]) / 10000.0
        if q not in spec:
            raise AssertionError(f"{slug} is not a registered q of the ladder")
        spec[q].update(values)
    return [acceptance_row(q, **values) for q, values in spec.items()]


DURABLE_Q = 0.30

# The published frozen validation AVO GT per distance bin. GT does not move with
# q, so one eligible-GT vector serves every row.
BIN_ELIGIBLE_GT = {"00_10m": 124, "10_20m": 1008, "20_30m": 1004, "30_40m": 741}
NEAR_BINS = ("00_10m", "10_20m", "20_30m")
GT_20_40M = BIN_ELIGIBLE_GT["20_30m"] + BIN_ELIGIBLE_GT["30_40m"]
GT_0_30M = sum(BIN_ELIGIBLE_GT[name] for name in NEAR_BINS)


def reference_bins(q: float, *, tp_20_30m: int | None = None) -> dict:
    """Per-bin recall slices consistent with the frozen 20-40 m metric at this q.

    The recorded ``person_avo_recall_20_40m`` is ``tp/1745`` for an integer tp, so
    recovering tp and dividing again reproduces the published float exactly --
    which is what the writer's own cross-check requires. ``tp_20_30m`` moves TP
    between the two long-range bins, so a fixture can change the 0-30 m recall
    while leaving the historical 20-40 m recall untouched: that is precisely the
    distinction the corrected gate is built on.
    """
    recorded = float(reference_metrics(q)["person_avo_recall_20_40m"])
    tp_long = round(recorded * GT_20_40M)
    assert tp_long / GT_20_40M == recorded, "fixture cannot reproduce the frozen metric"
    if tp_20_30m is None:
        tp_20_30m = tp_long - min(BIN_ELIGIBLE_GT["30_40m"], tp_long)
    tp_30_40m = tp_long - tp_20_30m
    assert 0 <= tp_20_30m <= BIN_ELIGIBLE_GT["20_30m"]
    assert 0 <= tp_30_40m <= BIN_ELIGIBLE_GT["30_40m"]
    tp = {
        "00_10m": BIN_ELIGIBLE_GT["00_10m"],
        "10_20m": BIN_ELIGIBLE_GT["10_20m"],
        "20_30m": tp_20_30m,
        "30_40m": tp_30_40m,
    }
    return {
        name: {
            "eligible_gt": BIN_ELIGIBLE_GT[name],
            "tp": tp[name],
            "fn": BIN_ELIGIBLE_GT[name] - tp[name],
            "recall": tp[name] / BIN_ELIGIBLE_GT[name],
            "xy_mae_m": 0.5,
        }
        for name in BIN_ELIGIBLE_GT
    }


def passing_primary_tp() -> int:
    """The smallest 20-30 m TP that lifts 0-30 m recall to the 0.70 bar."""
    near = BIN_ELIGIBLE_GT["00_10m"] + BIN_ELIGIBLE_GT["10_20m"]
    return math.ceil(0.70 * GT_0_30M) - near


def reference_metrics(q: float) -> dict:
    """The frozen noAE UINT8+zstd protected metrics at one q.

    Read from the already-published frozen document; nothing is measured here.
    The q=0 row is the only one that passes seven of the eight registered
    absolute object requirements, so it is the fixture a promotion check needs.
    """
    return dict(validation.load_noae_reference()[quantize_q(float(q)).q_e4]["metrics"])


def durable_setting(
    bottleneck: int,
    predictions: Path,
    identity: dict,
    *,
    q: float = DURABLE_Q,
    tp_20_30m: int | None = None,
) -> dict:
    """One structurally complete per-q setting, built by the real writer.

    Produced by ``_setting_document`` from a frozen noAE reference row used as
    both reference and candidate, so it also pins that the writer's own output
    satisfies the resume-path validator for this family.
    """
    size = family.require_phase10_bottleneck(bottleneck)
    plan = quantize_q(float(q))
    reference = validation.load_noae_reference()[plan.q_e4]
    analytical = validation.analytical_size(size, plan.wire_q)
    frames = contract.VALIDATION_FRAMES
    metrics = dict(reference["metrics"])
    canonical = dict(reference["canonical_person_metrics"])
    raw: dict[str, Any] = {
        **validation.family_fields(size),
        "q": plan.wire_q,
        "q_e4": plan.q_e4,
        "frames": frames,
        "prediction_root": str(predictions),
        "retained_cells": plan.keep_count,
        "dropped_cells": plan.drop_count,
        "payload": {
            "family": family.family_label(size),
            "family_id": family.family_id(size),
            "transported_latent_channels": validation.latent_width(size),
            "routing_tag": validation.routing_tag(size),
            "analytical_pre_zstd_bytes": analytical.total_bytes,
            "analytical_breakdown": {
                "header_bytes": analytical.header_bytes,
                "mask_bytes": analytical.mask_bytes,
                "range_bytes": analytical.range_bytes,
                "value_bytes": analytical.value_bytes,
            },
            "pre_zstd_bytes": validation._byte_stats(
                [analytical.total_bytes] * frames
            ),
            "zstd_bytes": validation._byte_stats(
                [int(analytical.total_bytes * 0.6)] * frames
            ),
            "zstd_mandatory": True,
        },
        "integrity": {
            "ranker_invocations": 0 if plan.is_bypass else frames,
            "q0_ranker_bypassed": plan.is_bypass,
            "ae_encoder_bypassed": False,
            "ranked_original_fp32_c2_per_frame": not plan.is_bypass,
            "selection_independent_per_frame": True,
            "batched_or_cross_frame_selection_used": False,
            "ranges_from_complete_latent_before_dropping": True,
            "retained_uint8_cells_equal_selected_indices": True,
            "dropped_cells_scattered_to_exact_zero": True,
            "zstd_decompressions": frames,
            "decoder_selected_from_received_header_bytes": True,
            "received_family_matches_selected_family": True,
            "received_latent_width_matches_bottleneck": True,
            "received_routing_tag_matches_bound_tag": True,
            "received_family_ids": [family.family_id(size)],
            "received_latent_widths": [validation.latent_width(size)],
            "local_packet_metadata_used_for_selection": False,
            "reconstruction_is_identity_at_any_q": False,
            "all_outputs_finite": True,
        },
    }
    scored = {
        "metrics": metrics,
        "canonical_person_metrics": canonical,
        "person_avo_detail": {
            "distance_bins": reference_bins(q, tp_20_30m=tp_20_30m)
        },
        "absolute_service_gates": contract.absolute_service_gates(
            {
                "vehicle_precision": metrics["vehicle_precision"],
                "vehicle_recall": metrics["vehicle_recall"],
                "vehicle_xy_mae_m": metrics["vehicle_xy_mae_m"],
                "vehicle_iou": metrics["vehicle_iou"],
                "person_box_mask_iou": metrics["person_box_mask_iou"],
                "foreground_miou": metrics["foreground_miou"],
                "person_precision": canonical["person_precision"],
                "person_recall": canonical["person_recall"],
                "person_xy_mae_m": canonical["person_xy_mae_m"],
            }
        ),
    }
    return validation._setting_document(
        bottleneck=size,
        raw=raw,
        scored=scored,
        reference=reference,
        identity=identity,
    )


class Phase10bAcceptanceAndResumeChecks(unittest.TestCase):
    """One shared implementation, exercised identically for both families."""

    def setUp(self) -> None:
        family._reset_process_family_for_tests()

    def tearDown(self) -> None:
        family._reset_process_family_for_tests()

    def test_shared_acceptance_rule_and_secondary_classification(self) -> None:
        cuda_before = torch.cuda.is_initialized()

        # The primary scope is the AE128 one, reused: the same six q, the same
        # three primary q, the same two stress q, the same twelve preservation
        # gates, the same nine service gates and the same registered baseline.
        self.assertEqual(validation.Q_VALUES, contract.REGISTERED_Q_VALUES)
        self.assertEqual(validation.ACCEPTANCE_PRIMARY_Q, (0.30, 0.50, 0.70))
        self.assertEqual(validation.STRESS_Q_VALUES, contract.EVALUATION_STRESS_Q_VALUES)
        self.assertEqual(GATES, len(contract.HOLDOUT_PRESERVATION_GATES))
        self.assertEqual(SERVICE_GATES, len(contract.ABSOLUTE_SERVICE_TARGETS))
        self.assertEqual(BASELINE, contract.FROZEN_Q0_SERVICE_PASS_COUNT)
        self.assertIs(validation.acceptance_inputs, phase9d.acceptance_inputs)
        self.assertEqual(
            validation.NOAE_UINT8_VALIDATION_SHA256,
            phase9d.NOAE_UINT8_VALIDATION_SHA256,
        )

        for size in FAMILIES:
            label = family.family_label(size)

            # The rule text is the Phase-9D rule with only the family name
            # substituted -- it cannot have been reworded here.
            rule = validation.acceptance_rule(size)
            self.assertNotIn("AE128", rule)
            self.assertIn(label, rule)
            self.assertEqual(rule.replace(label, "AE128"), phase9d.ACCEPTANCE_RULE)

            # 1. Both conditions met: accepted, family-labelled terminal, and
            #    only q=0.30 qualifies.
            result = validation.evaluate_acceptance(ladder(), size)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["decision"], validation.accepted_terminal(size))
            self.assertEqual(result["qualifying_primary_q"], [0.30])
            self.assertTrue(result["registered_before_measurement"])
            self.assertEqual(result["family"], label)
            self.assertEqual(result["rule_source"], validation.ACCEPTANCE_RULE_SOURCE)
            self.assertTrue(result["failed_acceptance_suppresses_no_measured_q_row"])
            self.assertTrue(
                result["preservation_is_relative_to_frozen_noae_uint8_zstd_same_q"]
            )
            self.assertTrue(result["preservation_is_not_an_absolute_service_claim"])

            # 2. q=0 losing one gate blocks acceptance even though q=0.30 still
            #    qualifies, and the terminal is this family's.
            blocked = validation.evaluate_acceptance(
                ladder(q0=dict(gates=GATES - 1)), size
            )
            self.assertFalse(blocked["accepted"])
            self.assertEqual(
                blocked["decision"], validation.not_accepted_terminal(size)
            )
            self.assertEqual(blocked["qualifying_primary_q"], [0.30])

            # 3. Stress q can neither create nor destroy acceptance.
            perfect_stress = validation.evaluate_acceptance(
                ladder(
                    q3000=dict(gates=GATES - 1),
                    q9000=dict(gates=GATES, service=SERVICE_GATES, noae_service=1),
                    q9800=dict(gates=GATES, service=SERVICE_GATES, noae_service=1),
                ),
                size,
            )
            self.assertFalse(perfect_stress["accepted"])
            broken_stress = validation.evaluate_acceptance(
                ladder(q9000=dict(gates=0), q9800=dict(gates=0)), size
            )
            self.assertTrue(broken_stress["accepted"])
            for entry in broken_stress["stress_q_status"]:
                self.assertFalse(entry["influences_acceptance"])

            # The secondary registration is fixed in source, with the required
            # provenance wording and no independence claim.
            self.assertEqual(
                validation.LOCALIZATION_OBJECT_REQUIREMENTS,
                (
                    ("vehicle_precision", 0.80, "higher"),
                    ("vehicle_recall", 0.85, "higher"),
                    ("vehicle_xy_mae_m", 1.00, "lower"),
                    ("person_avo_precision", 0.70, "higher"),
                    ("person_avo_recall", 0.70, "higher"),
                    ("person_avo_f1", 0.70, "higher"),
                    ("person_avo_xy_mae_m", 1.20, "lower"),
                    ("person_avo_recall_0_30m", 0.70, "higher"),
                ),
            )

            # The frozen pedestrian operating ranges, the provenance wording, and
            # the evaluation-only character of the 30 m boundary.
            self.assertEqual(
                validation.PEDESTRIAN_PRIMARY_RANGE_BINS,
                ("00_10m", "10_20m", "20_30m"),
            )
            self.assertEqual(
                validation.PEDESTRIAN_EXTENDED_RANGE_BINS, ("30_40m",)
            )
            self.assertEqual(
                validation.PEDESTRIAN_PRIMARY_RANGE, "0 <= gt_distance_m < 30"
            )
            self.assertEqual(
                validation.PEDESTRIAN_EXTENDED_DIAGNOSTIC_RANGE,
                "30 <= gt_distance_m <= 40",
            )
            for sentence in (
                "The 0-30 m primary operating range was selected from frozen noAE "
                "range-stratified analysis and literature context before "
                "Phase-10B AE64/AE32 validation.",
                "The 30-40 m results remain reported as extended-range stress.",
                "Independent test-set confirmation has not been performed.",
            ):
                self.assertIn(sentence, validation.PEDESTRIAN_RANGE_PROVENANCE)
            self.assertTrue(
                validation.EVALUATION_ONLY_BOUNDARY_DECLARATIONS[
                    "boundary_is_evaluation_only"
                ]
            )
            self.assertTrue(
                validation.EVALUATION_ONLY_BOUNDARY_DECLARATIONS[
                    "deployment_emits_all_p025_detections_at_every_range"
                ]
            )
            for flag in (
                "boundary_runtime_computable",
                "runtime_detections_filtered_by_range",
                "runtime_detections_suppressed_by_range",
                "runtime_detections_relabelled_by_range",
                "runtime_detections_rescored_by_range",
                "frozen_p025_perception_path_changed",
                "range_aware_runtime_policy_implemented",
                "rejected_feasibility_policies_abc_implemented",
            ):
                self.assertFalse(
                    validation.EVALUATION_ONLY_BOUNDARY_DECLARATIONS[flag], flag
                )
            self.assertIn(
                "does not filter, suppress, relabel, rescore or otherwise change "
                "any runtime detection",
                validation.EVALUATION_ONLY_BOUNDARY_RULE,
            )

            # The other seven object requirements are untouched by the swap.
            self.assertEqual(
                [name for name, _t, _d in validation.LOCALIZATION_OBJECT_REQUIREMENTS
                 if name != "person_avo_recall_0_30m"],
                [
                    "vehicle_precision", "vehicle_recall", "vehicle_xy_mae_m",
                    "person_avo_precision", "person_avo_recall", "person_avo_f1",
                    "person_avo_xy_mae_m",
                ],
            )
            # The superseded 20-40 m requirement is gone from the gate set but
            # survives as a protected metric, so history stays comparable.
            self.assertNotIn(
                "person_avo_recall_20_40m",
                [name for name, _t, _d in validation.LOCALIZATION_OBJECT_REQUIREMENTS],
            )
            self.assertIn("person_avo_recall_20_40m", contract.PROTECTED_METRICS)
            self.assertEqual(len(contract.HOLDOUT_PRESERVATION_GATES), GATES)
            self.assertEqual(
                validation.SEGMENTATION_INSTALL_REQUIREMENTS,
                (
                    ("vehicle_iou", 0.85, "higher"),
                    ("person_box_mask_iou", 0.50, "higher"),
                    ("foreground_miou", 0.675, "higher"),
                ),
            )
            self.assertIn(
                "Holdout-informed thresholds frozen before AE64/AE32 held-out "
                "deployment validation.",
                validation.LOCALIZATION_THRESHOLD_PROVENANCE,
            )
            self.assertIn(
                "The validation frames were not used for AE training or "
                "checkpoint selection.",
                validation.LOCALIZATION_THRESHOLD_PROVENANCE,
            )
            self.assertEqual(validation.TIER_FULL_PRESERVATION, "FULL_PRESERVATION")
            self.assertNotIn(
                "FULL_PERCEPTION",
                [name for name, _definition in validation.CLASSIFICATION_TIERS],
            )
            self.assertEqual(validation.TIER_STATE_INFEASIBLE, "STATE_INFEASIBLE")
            # No segmentation metric may enter the localization classification.
            objects = {name for name, _t, _d in validation.LOCALIZATION_OBJECT_REQUIREMENTS}
            segmentation = {
                name for name, _t, _d in validation.SEGMENTATION_INSTALL_REQUIREMENTS
            }
            self.assertEqual(objects & segmentation, set())

            # Classify six synthetic rows: the four non-stress rows carry the
            # frozen noAE reference metrics, so their absolute requirements
            # behave exactly as the registration-time feasibility report says.
            rows = []
            for q in validation.Q_VALUES:
                with tempfile.TemporaryDirectory() as scratch:
                    row = durable_setting(
                        size, Path(scratch), {"sha256": "a" * 64}, q=q
                    )
                rows.append(row)
            acceptance = validation.evaluate_acceptance(
                [validation.acceptance_inputs(row) for row in rows], size
            )
            secondary = validation.classify_profiles(
                bottleneck=size, rows=rows, acceptance=acceptance
            )
            self.assertEqual(secondary["profiles_classified"], len(rows))
            self.assertTrue(secondary["every_measured_profile_classified_independently"])
            self.assertFalse(secondary["changed_checkpoint_selection"])
            self.assertFalse(secondary["changed_primary_acceptance_terminal"])
            self.assertFalse(secondary["changed_any_threshold_nms_model_or_scorer"])
            self.assertFalse(
                secondary["erased_or_reinterpreted_original_preservation_failures"]
            )
            self.assertFalse(secondary["independent_test_set_confirmation"])
            self.assertFalse(secondary["untouched_test_set_confirmation"])
            self.assertFalse(
                secondary["state_infeasible"]["assignable_by_this_validation"]
            )
            self.assertEqual(secondary["tier_counts"]["STATE_INFEASIBLE"], 0)

            by_q = {entry["q"]: entry for entry in secondary["profiles"]}

            # Stress profiles are EMERGENCY_ONLY regardless of their metrics --
            # here they carry a *12/12-passing* row and still do not promote.
            for q in validation.STRESS_Q_VALUES:
                entry = by_q[float(q)]
                self.assertEqual(entry["tier"], validation.TIER_EMERGENCY_ONLY)
                self.assertTrue(entry["is_registered_stress_profile"])
                self.assertIn("stress profile", entry["tier_reason"])
                self.assertEqual(
                    rows[validation.Q_VALUES.index(q)][validation.PRESERVATION_KEY][
                        "gates_passed"
                    ],
                    GATES,
                )

            # q=0 reproduces the noAE reference exactly, so it passes 12/12 and
            # is FULL_PRESERVATION -- and that alone authorizes neither a
            # segmentation install nor a SERVICE_READY claim.
            zero = by_q[0.0]
            self.assertEqual(zero["tier"], validation.TIER_FULL_PRESERVATION)
            self.assertTrue(zero["full_preservation"]["passed"])
            self.assertFalse(
                zero["full_preservation"]["authorizes_segmentation_install"]
            )
            self.assertFalse(zero["full_preservation"]["implies_service_ready"])
            self.assertFalse(
                zero["segmentation"]["twelve_of_twelve_preservation_authorizes_install"]
            )
            self.assertFalse(
                zero["service_readiness"]["derived_from_relative_preservation"]
            )
            # SERVICE_READY is the separate 9/9 absolute result, and the frozen
            # reference passes only the registered 7/9.
            self.assertEqual(
                zero["service_readiness"]["absolute_service_pass_count"], BASELINE
            )
            self.assertFalse(zero["service_readiness"]["service_ready"])
            self.assertIsNone(zero["service_readiness"]["terminal"])
            # Its segmentation is installable, so the action is to install.
            self.assertTrue(zero["segmentation"]["segmentation_installable"])
            self.assertEqual(
                zero["segmentation"]["action"],
                validation.SEGMENTATION_INSTALL_ACTION_INSTALL,
            )

            # Canonical person metrics are recorded as diagnostics and are not
            # what the classification reads.
            self.assertFalse(
                zero["canonical_person_diagnostics"][
                    "used_for_localization_priority_classification"
                ]
            )
            self.assertEqual(
                set(zero["canonical_person_diagnostics"]["metrics"]),
                set(validation.CANONICAL_PERSON_METRICS),
            )

            # Segmentation installability is decided only by its three
            # requirements, independently of the tier: drop one below its bound
            # and the action flips to retaining the previous layer.
            metrics = reference_metrics(0.0)
            self.assertTrue(
                validation.segmentation_installability(metrics)[
                    "segmentation_installable"
                ]
            )
            reduced = {**metrics, "foreground_miou": 0.60}
            flipped = validation.segmentation_installability(reduced)
            self.assertFalse(flipped["segmentation_installable"])
            self.assertEqual(
                flipped["action"], validation.SEGMENTATION_INSTALL_ACTION_RETAIN
            )
            self.assertEqual(flipped["failed"], ["foreground_miou"])

            # Without a range stratification the primary-range requirement is
            # *not evaluable*, so the result fails closed and records why rather
            # than reporting a fabricated miss. This is the frozen noAE reference
            # document's situation: it publishes no per-bin slices.
            unstratified = validation.localization_requirements(metrics)
            self.assertEqual(unstratified["failed"], [])
            self.assertEqual(
                unstratified["not_evaluable"], ["person_avo_recall_0_30m"]
            )
            self.assertFalse(unstratified["all_registered_requirements_evaluated"])
            self.assertFalse(unstratified["all_passed"])
            self.assertEqual(unstratified["total"], 7)
            self.assertEqual(unstratified["registered_total"], 8)

            # With the stratification the default fixture misses exactly the one
            # corrected requirement.
            stratified = rows[0][validation.RANGE_STRATIFIED_KEY]
            self.assertEqual(
                validation.localization_requirements(
                    metrics, range_stratified=stratified
                )["failed"],
                ["person_avo_recall_0_30m"],
            )

            # A row whose relative rule fails but whose absolute object
            # requirements all pass is LOCALIZATION_PRIORITY, not EMERGENCY_ONLY.
            # The promotion moves TP between the two long-range bins, so the
            # historical 20-40 m recall is byte-identical and only the corrected
            # 0-30 m gate changes -- the whole point of the correction.
            with tempfile.TemporaryDirectory() as scratch:
                promoted = durable_setting(
                    size,
                    Path(scratch),
                    {"sha256": "a" * 64},
                    q=0.0,
                    tp_20_30m=passing_primary_tp(),
                )
            self.assertEqual(
                promoted[validation.RANGE_STRATIFIED_KEY]["historical_20_40m"][
                    "recall"
                ],
                stratified["historical_20_40m"]["recall"],
            )
            self.assertGreaterEqual(
                promoted[validation.RANGE_STRATIFIED_KEY][
                    "person_avo_recall_0_30m"
                ],
                0.70,
            )
            self.assertLess(stratified["person_avo_recall_0_30m"], 0.70)
            localization = validation.classify_profile(
                bottleneck=size,
                row=promoted,
                full_preservation_passed=False,
                full_preservation_basis={"primary_verdict_available": True},
            )
            self.assertEqual(
                localization["tier"], validation.TIER_LOCALIZATION_PRIORITY
            )
            self.assertTrue(localization["localization_priority"]["all_passed"])
            self.assertTrue(
                localization["localization_priority"][
                    "all_registered_requirements_evaluated"
                ]
            )

            # The 0-30 m recall is a plain sum of the frozen 0-10, 10-20 and
            # 20-30 m TP/FN counts -- no new matching or inference.
            bins = stratified["bins"]
            near_tp = sum(int(bins[name]["tp"]) for name in NEAR_BINS)
            near_gt = sum(int(bins[name]["eligible_gt"]) for name in NEAR_BINS)
            self.assertEqual(near_gt, GT_0_30M)
            self.assertEqual(
                stratified["person_avo_recall_0_30m"], near_tp / near_gt
            )
            for name, bucket in bins.items():
                self.assertEqual(
                    int(bucket["tp"]) + int(bucket["fn"]),
                    int(bucket["eligible_gt"]),
                    name,
                )
                self.assertFalse(bucket["precision_available"], name)
                self.assertIsNone(bucket["precision"], name)
                self.assertFalse(bucket["is_tier_gate"], name)

            # 20-30 m is reported on its own so the cumulative result cannot hide
            # the boundary band; 30-40 m stays reported and ungated; the original
            # 20-40 m recall survives for historical comparison and reproduces
            # the protected metric exactly.
            self.assertEqual(stratified["boundary_band"]["band"], "20_30m")
            self.assertEqual(
                stratified["boundary_band"]["recall"], bins["20_30m"]["recall"]
            )
            self.assertFalse(stratified["extended_range_stress"]["is_tier_gate"])
            self.assertEqual(
                stratified["extended_range_stress"]["bins"], ["30_40m"]
            )
            self.assertTrue(stratified["primary_operating_range_detail"]["is_tier_gate"])
            self.assertFalse(stratified["historical_20_40m"]["is_tier_gate"])
            self.assertEqual(
                stratified["historical_20_40m"]["recall"],
                rows[0]["metrics"]["person_avo_recall_20_40m"],
            )
            self.assertFalse(stratified["precision_by_range"]["available"])
            self.assertIn(
                "not derivable from the frozen artifacts",
                stratified["precision_by_range"]["reason"],
            )

            # The evaluation-only declarations travel with the classification.
            for name, value in validation.EVALUATION_ONLY_BOUNDARY_DECLARATIONS.items():
                self.assertEqual(
                    localization["localization_priority"][name], value, name
                )
                self.assertEqual(
                    localization["person_range_stratified"][name], value, name
                )

            # The frozen noAE reference document publishes no per-bin slices, so
            # registration-time feasibility records the corrected requirement as
            # not evaluable instead of counting it as a miss.
            feasibility = validation.reference_feasibility(size)
            self.assertEqual(
                feasibility["not_evaluable_requirements"],
                ["person_avo_recall_0_30m"],
            )
            for entry in feasibility["per_q"]:
                self.assertEqual(
                    entry["not_evaluable_object_requirements"],
                    ["person_avo_recall_0_30m"],
                )
                self.assertEqual(entry["object_requirements_evaluated"], 7)
                self.assertEqual(entry["object_requirements_registered_total"], 8)
                self.assertFalse(entry["object_requirements_all_passed"])
                self.assertNotIn(
                    "person_avo_recall_0_30m", entry["failed_object_requirements"]
                )
            # q=0 now clears all seven evaluable object requirements: the
            # superseded 20-40 m bar was the only one it missed.
            self.assertEqual(feasibility["per_q"][0]["q"], 0.0)
            self.assertEqual(feasibility["per_q"][0]["object_requirements_passed"], 7)
            self.assertEqual(feasibility["per_q"][0]["failed_object_requirements"], [])

            # The same row with an object requirement missed is EMERGENCY_ONLY,
            # and degradation never masks the action.
            demoted = validation.classify_profile(
                bottleneck=size,
                row=dict(rows[1]),
                full_preservation_passed=False,
                full_preservation_basis={"primary_verdict_available": True},
            )
            self.assertEqual(demoted["tier"], validation.TIER_EMERGENCY_ONLY)
            self.assertFalse(demoted["availability"]["masked"])
            self.assertFalse(
                demoted["availability"]["masked_for_perception_degradation"]
            )

            # Only technical invalidity is INVALID: a broken transport
            # declaration, not a quality shortfall.
            invalid_row = dict(rows[1])
            invalid_row["integrity"] = {
                **dict(invalid_row["integrity"]),
                "decoder_selected_from_received_header_bytes": False,
            }
            invalid = validation.classify_profile(
                bottleneck=size,
                row=invalid_row,
                full_preservation_passed=True,
                full_preservation_basis={"primary_verdict_available": True},
            )
            self.assertEqual(invalid["tier"], validation.TIER_INVALID)
            self.assertFalse(invalid["integrity"]["valid"])

        self.assertEqual(torch.cuda.is_initialized(), cuda_before)

    def test_durable_setting_resume_never_remeasures_either_family(self) -> None:
        cuda_before = torch.cuda.is_initialized()

        for size in FAMILIES:
            identity = {"sha256": f"{size:064d}"}
            plan = quantize_q(DURABLE_Q)
            slug = validation._q_slug(DURABLE_Q)
            other = 32 if size == 64 else 64

            with tempfile.TemporaryDirectory() as raw_output:
                output = Path(raw_output)
                predictions = validation.prediction_root(output, size, DURABLE_Q)
                (predictions / "segmentation").mkdir(parents=True)
                (predictions / "detections.csv").write_text("x", encoding="utf-8")
                (predictions / "segmentation" / "a.png").write_bytes(bytes(1))
                path = validation.setting_path(output, size, DURABLE_Q)
                self.assertEqual(path.name, f"{slug}.json")

                # Exactly the interruption window: the setting is durably on
                # disk and nothing after it ran.
                digest = validation._atomic_json(
                    path, durable_setting(size, predictions, identity)
                )
                self.assertTrue(path.is_file())
                self.assertTrue(predictions.is_dir())
                self.assertFalse(
                    validation.cleanup_is_complete(
                        output, size, DURABLE_Q, identity, digest
                    )
                )

                written: list[str] = []
                real_atomic_json = validation._atomic_json

                def recording_atomic_json(target, document):
                    written.append(str(target))
                    return real_atomic_json(target, document)

                with mock.patch.object(
                    validation, "_atomic_json", side_effect=recording_atomic_json
                ), mock.patch.object(
                    validation,
                    "run_validation_pass",
                    side_effect=AssertionError("a durable q was remeasured"),
                ):
                    reused = validation.reuse_or_complete(
                        output=output,
                        bottleneck=size,
                        q=DURABLE_Q,
                        identity=identity,
                    )

                    # The durable record is reused byte-for-byte, not rebuilt.
                    self.assertIsNotNone(reused)
                    self.assertEqual(
                        reused, json.loads(path.read_text(encoding="utf-8"))
                    )
                    self.assertEqual(int(reused["q_e4"]), plan.q_e4)
                    self.assertEqual(reused["family"], family.family_label(size))
                    self.assertEqual(sha256_file(path), digest)

                    # Only the cleanup was completed: the sole write is the
                    # marker, and the scratch predictions are gone.
                    marker_path = validation.cleanup_marker_path(
                        output, size, DURABLE_Q
                    )
                    self.assertEqual(written, [str(marker_path)])
                    self.assertFalse(predictions.exists())
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    self.assertEqual(marker["schema"], validation.cleanup_schema(size))
                    self.assertEqual(
                        marker["terminal"], validation.cleanup_terminal(size)
                    )
                    self.assertEqual(marker["setting_sha256"], digest)
                    self.assertEqual(int(marker["q_e4"]), plan.q_e4)
                    self.assertTrue(
                        marker["prediction_artifacts_removed_after_scoring"]
                    )
                    self.assertTrue(
                        validation.cleanup_is_complete(
                            output, size, DURABLE_Q, identity, digest
                        )
                    )
                    # The marker belongs to this family only.
                    self.assertFalse(
                        validation.cleanup_is_complete(
                            output, other, DURABLE_Q, identity, digest
                        )
                    )

                    # A further resume is a no-op: no write, no measurement.
                    again = validation.reuse_or_complete(
                        output=output,
                        bottleneck=size,
                        q=DURABLE_Q,
                        identity=identity,
                    )
                    self.assertEqual(again, reused)
                    self.assertEqual(written, [str(marker_path)])
                    self.assertEqual(sha256_file(path), digest)

                # The other family cannot read this record at all.
                with self.assertRaises(guards.HybridQConfigError):
                    validation.load_durable_setting(path, other, DURABLE_Q, identity)
                # Nor can a different run identity.
                with self.assertRaises(guards.HybridQConfigError):
                    validation.load_durable_setting(
                        path, size, DURABLE_Q, {"sha256": "b" * 64}
                    )
                # Nor may it stand in for a different q.
                with self.assertRaises(guards.HybridQConfigError):
                    validation.load_durable_setting(path, size, 0.50, identity)

                # An incomplete record is never reused, and never silently
                # remeasured either: it fails closed. Each of a wrong frame
                # count, a wrong payload width and a cleared integrity flag is
                # refused rather than overwritten.
                good = json.loads(path.read_text(encoding="utf-8"))
                for mutate in (
                    lambda d: d.update(frames=contract.VALIDATION_FRAMES - 1),
                    lambda d: d["payload"].update(
                        transported_latent_channels=other
                    ),
                    lambda d: d["integrity"].update(
                        dropped_cells_scattered_to_exact_zero=False
                    ),
                    lambda d: d["integrity"].update(ae_encoder_bypassed=True),
                    lambda d: d.update(selected_ae_state_unchanged=False),
                ):
                    damaged = json.loads(json.dumps(good))
                    mutate(damaged)
                    real_atomic_json(path, damaged)
                    with self.assertRaises(guards.HybridQConfigError):
                        validation.reuse_or_complete(
                            output=output,
                            bottleneck=size,
                            q=DURABLE_Q,
                            identity=identity,
                        )
                    # Refused, not rewritten: the damaged bytes are still there.
                    self.assertEqual(
                        json.loads(path.read_text(encoding="utf-8")), damaged
                    )

                # No durable record at all is the one case that permits a pass.
                self.assertIsNone(
                    validation.reuse_or_complete(
                        output=output, bottleneck=size, q=0.70, identity=identity
                    )
                )

        self.assertEqual(torch.cuda.is_initialized(), cuda_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
