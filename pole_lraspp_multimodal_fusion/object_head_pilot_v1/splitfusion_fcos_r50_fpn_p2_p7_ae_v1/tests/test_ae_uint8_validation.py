"""Two focused CPU checks over the Phase-9D validation contract.

Nothing is inferred, trained, validated or scored here, neither test creates a
CUDA context (each asserts the process-global flag is exactly where it found
it), and no dataset, teacher-cache shard or CARLA process is touched. The
selected AE128 checkpoint is opened only to read the source map and identity
fields it recorded -- its tensors are never built into a model and never moved
to a device.

1. the preregistered acceptance rule, applied verbatim to synthetic per-q rows,
   including that q=0.90 and q=0.98 can neither create nor destroy acceptance;
2. the exact selected-checkpoint and routing binding: the three bound SHA-256
   constants, the chain from the holdout decision to this checkpoint, the
   routing tag derived from the full digest, and the fail-closed AE package
   source delta.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.continuous_q import quantize_q
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import sha256_file
from .. import ae_contract, ae_uint8_validation as validation
from .. import ae_training_common as common
from ..ae_gpu_qualification import ae_package_source_hashes


GATES = validation.GATE_COUNT  # 12
BASELINE = validation.BASELINE_SERVICE_PASS_COUNT  # 7 of 9


def acceptance_row(
    q: float, *, gates: int, service: int, noae_service: int
) -> dict:
    """Exactly the fields `evaluate_acceptance` reads, and nothing else."""
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


class AcceptanceRuleTest(unittest.TestCase):
    """`evaluate_acceptance` is the registered rule and nothing more."""

    def test_preregistered_acceptance_rule_is_applied_verbatim(self) -> None:
        cuda_before = torch.cuda.is_initialized()
        # The rule is registered in source, before any Phase-9D number exists.
        self.assertIn("q=0 passes all 12 same-q preservation gates", validation.ACCEPTANCE_RULE)
        self.assertEqual(GATES, len(contract.HOLDOUT_PRESERVATION_GATES))
        self.assertEqual(BASELINE, contract.FROZEN_Q0_SERVICE_PASS_COUNT)
        self.assertEqual(validation.ACCEPTANCE_PRIMARY_Q, (0.30, 0.50, 0.70))
        self.assertEqual(
            validation.STRESS_Q_VALUES, contract.EVALUATION_STRESS_Q_VALUES
        )

        # 1. Both conditions met: accepted, and only q=0.30 qualifies.
        result = validation.evaluate_acceptance(ladder())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["decision"], validation.ACCEPTED_TERMINAL)
        self.assertTrue(result["q0_condition"]["passed"])
        self.assertTrue(result["q0_condition"]["retains_baseline_absolute_service_gates"])
        self.assertEqual(result["qualifying_primary_q"], [0.30])
        self.assertTrue(result["registered_before_measurement"])
        self.assertFalse(result["setting_tuned_after_observing_validation"])
        self.assertFalse(result["setting_removed_after_observing_validation"])

        # 2. q=0 losing a single preservation gate blocks acceptance outright,
        #    even though q=0.30 still qualifies.
        result = validation.evaluate_acceptance(ladder(q0000={"gates": GATES - 1}))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], validation.NOT_ACCEPTED_TERMINAL)
        self.assertFalse(result["q0_condition"]["preservation_all_passed"])
        self.assertTrue(result["q0_condition"]["retains_baseline_absolute_service_gates"])
        self.assertEqual(result["qualifying_primary_q"], [0.30])

        # 3. q=0 falling below the 7/9 baseline blocks acceptance; matching or
        #    beating the baseline does not.
        self.assertFalse(
            validation.evaluate_acceptance(
                ladder(q0000={"service": BASELINE - 1})
            )["accepted"]
        )
        self.assertTrue(
            validation.evaluate_acceptance(ladder(q0000={"service": 9}))["accepted"]
        )

        # 4. No qualifying primary q means no acceptance: 0.30 and 0.50 lose a
        #    gate, and 0.70 keeps all twelve but reduces the service count.
        result = validation.evaluate_acceptance(
            ladder(
                q3000={"gates": GATES - 1},
                q5000={"gates": GATES - 1},
                q7000={"gates": GATES, "service": 3, "noae_service": 4},
            )
        )
        self.assertFalse(result["accepted"])
        self.assertTrue(result["q0_condition"]["passed"])
        self.assertEqual(result["qualifying_primary_q"], [])
        self.assertFalse(result["primary_condition_satisfied"])
        by_q = {entry["q"]: entry for entry in result["primary_q_conditions"]}
        self.assertTrue(by_q[0.70]["preservation_all_passed"])
        self.assertTrue(by_q[0.70]["reduces_absolute_service_gate_count"])
        self.assertFalse(by_q[0.70]["qualifies"])

        # 5. "Without reducing" is >=, not >: a primary q that improves on the
        #    noAE service count qualifies, and one that merely matches does too.
        self.assertEqual(
            validation.evaluate_acceptance(
                ladder(
                    q3000={"gates": GATES - 1},
                    q5000={"gates": GATES, "service": 6, "noae_service": 4},
                )
            )["qualifying_primary_q"],
            [0.50],
        )

        # 6. The stress rungs cannot rescue a failing ladder and cannot spoil a
        #    passing one, whatever they measure.
        failing = dict(
            q3000={"gates": GATES - 1},
            q5000={"gates": GATES - 1},
            q7000={"gates": GATES - 1},
        )
        perfect_stress = {"gates": GATES, "service": 9, "noae_service": 3}
        rescued = validation.evaluate_acceptance(
            ladder(**failing, q9000=perfect_stress, q9800=perfect_stress)
        )
        self.assertFalse(rescued["accepted"])
        collapsed_stress = {"gates": 0, "service": 0, "noae_service": 3}
        spoiled = validation.evaluate_acceptance(
            ladder(q9000=collapsed_stress, q9800=collapsed_stress)
        )
        self.assertTrue(spoiled["accepted"])
        for entry in spoiled["stress_q_status"]:
            self.assertIn(entry["q"], contract.EVALUATION_STRESS_Q_VALUES)
            self.assertFalse(entry["influences_acceptance"])
            self.assertEqual(
                entry["designation"], "stress/emergency profile regardless of result"
            )

        # 7. The rule is only defined over exactly the six registered q, and
        #    only against a frozen noAE q=0 row that reports the 7/9 baseline.
        rows = ladder()
        with self.assertRaises(guards.HybridQConfigError):
            validation.evaluate_acceptance(rows[:-1])
        with self.assertRaises(guards.HybridQConfigError):
            validation.evaluate_acceptance(rows + [copy.deepcopy(rows[0])])
        with self.assertRaises(guards.HybridQConfigError):
            validation.evaluate_acceptance(
                ladder(q0000={"noae_service": BASELINE - 1})
            )
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)


class SelectedCheckpointBindingTest(unittest.TestCase):
    """The bound artifacts, the decision chain and the routing derivation."""

    def test_selected_checkpoint_and_routing_binding_is_exact(self) -> None:
        cuda_before = torch.cuda.is_initialized()
        root = contract.repository_root()

        # 1. The three bound artifacts are exactly the files on disk.
        for relative, expected in (
            (validation.SELECTED_CHECKPOINT_RELPATH, validation.SELECTED_CHECKPOINT_SHA256),
            (validation.HOLDOUT_DECISION_RELPATH, validation.HOLDOUT_DECISION_SHA256),
            (
                validation.NOAE_UINT8_VALIDATION_RELPATH,
                validation.NOAE_UINT8_VALIDATION_SHA256,
            ),
        ):
            with self.subTest(artifact=relative):
                self.assertEqual(sha256_file((root / relative).resolve(strict=True)), expected)
        self.assertEqual(
            Path(validation.SELECTED_CHECKPOINT_RELPATH).name,
            common.candidate_filename(validation.SELECTED_CHECKPOINT_EPOCH),
        )
        self.assertEqual(
            validation.contract.FRAMED_Q0_PAYLOAD_BYTES, 22020140
        )

        # 2. The bound Phase-9C decision selects this epoch and records exactly
        #    this checkpoint hash, and it is refused if a frozen input moves.
        import json

        decision = json.loads(
            (root / validation.HOLDOUT_DECISION_RELPATH).read_text(encoding="utf-8")
        )
        self.assertEqual(
            int(decision["selection"]["selected_epoch"]),
            validation.SELECTED_CHECKPOINT_EPOCH,
        )
        self.assertEqual(
            decision["training_run"]["candidate_checkpoints"][
                common.candidate_filename(validation.SELECTED_CHECKPOINT_EPOCH)
            ],
            validation.SELECTED_CHECKPOINT_SHA256,
        )
        frozen_binding = {
            name: {"sha256": decision["binding"][name]["sha256"]}
            for name in (
                "frozen_checkpoint",
                "stable_epoch4_ranker",
                "perception_forward_lock",
                "hybrid_q_locked_config",
            )
        }
        loaded = validation.load_holdout_decision(frozen_binding)
        self.assertEqual(loaded["selected_epoch"], validation.SELECTED_CHECKPOINT_EPOCH)
        self.assertEqual(
            loaded["selected_checkpoint_sha256"], validation.SELECTED_CHECKPOINT_SHA256
        )
        self.assertFalse(loaded["selection_is_a_service_ready_claim"])
        for name in frozen_binding:
            drifted = copy.deepcopy(frozen_binding)
            drifted[name]["sha256"] = "0" * 64
            with self.subTest(drifted=name), self.assertRaises(guards.HybridQConfigError):
                validation.load_holdout_decision(drifted)

        # 3. The routing tag is derived from the *full* digest, is nonzero, fits
        #    in 32 bits, and actually discriminates this checkpoint from the two
        #    candidates that were not selected.
        tag = validation.routing_tag()
        self.assertEqual(
            tag, ae_contract.routing_tag_from_sha256(validation.SELECTED_CHECKPOINT_SHA256)
        )
        self.assertNotEqual(tag, ae_contract.AE_UNBOUND_ROUTING_TAG)
        self.assertTrue(0 < tag < 2 ** (8 * ae_contract.AE_ROUTING_TAG_BYTES))
        others = {
            digest
            for name, digest in decision["training_run"]["candidate_checkpoints"].items()
            if digest != validation.SELECTED_CHECKPOINT_SHA256
        }
        self.assertEqual(len(others), 2)
        for digest in others:
            self.assertNotEqual(tag, ae_contract.routing_tag_from_sha256(digest))

        record = validation.routing_record()
        self.assertEqual(record["routing_tag"], tag)
        self.assertEqual(
            record["selected_checkpoint_sha256"], validation.SELECTED_CHECKPOINT_SHA256
        )
        self.assertFalse(record["routing_tag_is_checkpoint_identity"])
        self.assertEqual(record["routing_tag_bytes"], 4)

        # 4. The AE package source delta: every module that defines the saved
        #    tensors or the measured transport is byte-identical to what the
        #    checkpoint recorded, and only the allowlisted files differ.
        payload = torch.load(
            (root / validation.SELECTED_CHECKPOINT_RELPATH).resolve(strict=True),
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(payload["schema"], common.AE_CANDIDATE_SCHEMA)
        self.assertEqual(int(payload["epoch"]), validation.SELECTED_CHECKPOINT_EPOCH)
        self.assertEqual(int(payload["bottleneck"]), validation.AE_BOTTLENECK)
        self.assertEqual(int(payload["family_id"]), validation.AE_FAMILY_ID)
        self.assertEqual(payload["configuration"], common.training_configuration())
        recorded = dict(payload["ae_package_source_sha256"])
        live = ae_package_source_hashes()
        del payload

        delta = validation.ae_package_source_delta(recorded, live)
        self.assertTrue(delta["semantics_modules_unchanged"])
        self.assertEqual(delta["removed"], [])
        changed = [entry["path"] for entry in delta["changed"]]
        self.assertTrue(set(changed) <= set(validation.AE_PHASE9D_MODIFIED_SOURCES))
        added = [entry["path"] for entry in delta["added"]]
        self.assertIn(Path(validation.__file__).name, added)
        for name in (
            *validation.AE_CHECKPOINT_SEMANTICS_SOURCES,
            *validation.AE_TRANSPORT_SEMANTICS_SOURCES,
        ):
            with self.subTest(frozen=name):
                self.assertEqual(live[name], recorded[name])

        # 5. It fails closed on a changed semantics module, on a removal, and on
        #    an unallowlisted change.
        for name in (
            *validation.AE_CHECKPOINT_SEMANTICS_SOURCES,
            *validation.AE_TRANSPORT_SEMANTICS_SOURCES,
        ):
            mutated = dict(live)
            mutated[name] = "1" * 64
            with self.subTest(mutated=name), self.assertRaises(guards.HybridQConfigError):
                validation.ae_package_source_delta(recorded, mutated)
        shrunk = {
            name: digest
            for name, digest in live.items()
            if name != "ae_gpu_qualification.py"
        }
        with self.assertRaises(guards.HybridQConfigError):
            validation.ae_package_source_delta(recorded, shrunk)
        unallowlisted = dict(live)
        unallowlisted["ae_training.py"] = "2" * 64
        with self.assertRaises(guards.HybridQConfigError):
            validation.ae_package_source_delta(recorded, unallowlisted)

        # 6. Every saved binding field is enforced exactly, except the one AE
        #    source map that Phase 9D necessarily changes.
        binding = {
            "stable_epoch4_ranker": {"sha256": contract.VALIDATION_RANKER_SHA256},
            "hybrid_q_locked_config": {"sha256": contract.LOCKED_CONFIG_SHA256},
            "teacher_cache_manifest": {"sha256": contract.TEACHER_CACHE_MANIFEST_SHA256},
            "hybrid_q_source_sha256": {"contract.py": "abc"},
            "ae_package_source_sha256": live,
        }
        saved = dict(common.binding_fields(binding))
        saved["ae_package_source_sha256"] = recorded
        enforced = validation.require_selected_bindings(saved, binding)
        self.assertNotIn("ae_package_source_sha256", enforced["enforced_exactly"])
        self.assertEqual(
            set(enforced["enforced_exactly"]),
            set(common.binding_fields(binding)) - {"ae_package_source_sha256"},
        )
        for name in enforced["enforced_exactly"]:
            drifted = dict(saved)
            drifted[name] = "3" * 64
            with self.subTest(binding=name), self.assertRaises(guards.HybridQConfigError):
                validation.require_selected_bindings(drifted, binding)
            missing = {
                key: value for key, value in saved.items() if key != name
            }
            with self.subTest(missing=name), self.assertRaises(guards.HybridQConfigError):
                validation.require_selected_bindings(missing, binding)

        self.assertEqual(torch.cuda.is_initialized(), cuda_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
