"""Focused CPU checks over the Phase-10A AE64/AE32 family separation.

No real checkpoint, dataset, teacher-cache shard or CUDA context is touched, and
nothing is trained, inferred, selected or evaluated. Three checks:

1. token / bottleneck / family / schema / seed separation, including the refusal
   of the out-of-scope 128-channel family and of any AE128 label;
2. family-specific checkpoint naming and optimizer ownership;
3. recovery and holdout isolation through the shared family-aware path.

The AE128 fixtures are imported from the Phase-9C test file rather than copied,
so both families are checked against the same synthetic partition and binding.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import torch

from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1 import contract, guards
from ...splitfusion_fcos_r50_fpn_p2_p7_hybrid_q_v1.gpu_qualification import sha256_file
from .. import ae_contract, ae_loss, ae_phase10_common as family, ae_phase10_training
from .. import ae_training_common as common
from ..ae_model import ae_parameters
from .test_ae_training_schedule import (
    cpu_rng_state,
    registered_shape_partition,
    synthetic_binding,
)

FAMILIES = family.AE_PHASE10_BOTTLENECKS  # (64, 32)


def write_partial_family_run(
    output: Path,
    bottleneck: int,
    partition: Any,
    dataset: Any,
    binding: dict,
    *,
    through_epoch: int,
) -> tuple[dict[str, str], dict[str, str]]:
    """Write epochs 1..`through_epoch` for one family in the trainer's order.

    Per epoch: the candidate checkpoint, then the family epoch-summary file,
    then the recovery checkpoint last -- exactly what the runner commits.
    """
    recovery_dir = output / "recovery"
    checkpoint_dir = output / "checkpoints"
    recovery_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    autoencoder = family.build_family_ae(bottleneck, torch.device("cpu"))
    optimizer = family.build_family_optimizer(
        autoencoder,
        lr=common.learning_rate_for_stage(common.AE_STAGE_A),
        frozen_modules=(),
    )
    recovery_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    with mock.patch.object(common, "rng_state", cpu_rng_state):
        for epoch in range(1, through_epoch + 1):
            identity = ae_phase10_training.order_identity(partition, epoch, dataset)
            updates = epoch * common.batches_per_epoch()
            position = ae_phase10_training.expected_stage_b_position(epoch)
            summary = {
                **family.family_fields(bottleneck),
                "epoch": epoch,
                "stage": common.stage_for_epoch(epoch),
                "epoch_sample_id_sha256": identity["sample_id_sha256"],
                "global_update_index": updates,
                "stage_b_cycle_position": position,
            }
            if epoch in common.AE_CANDIDATE_EPOCHS:
                candidate = family.candidate_filename(bottleneck, epoch)
                candidate_hashes[candidate] = family.save_candidate(
                    checkpoint_dir / candidate,
                    bottleneck=bottleneck,
                    epoch=epoch,
                    autoencoder=autoencoder,
                    global_update_index=updates,
                    stage_b_position=position,
                    binding=binding,
                )
                summary["candidate_checkpoint"] = candidate
                summary["candidate_checkpoint_sha256"] = candidate_hashes[candidate]
            summaries.append(summary)
            family.write_epoch_summaries(output, bottleneck, summaries)
            name = family.recovery_filename(bottleneck, epoch)
            recovery_hashes[name] = family.save_recovery(
                recovery_dir / name,
                bottleneck=bottleneck,
                epoch=epoch,
                autoencoder=autoencoder,
                optimizer=optimizer,
                global_update_index=updates,
                stage_b_position=position,
                order_identity=identity,
                summary=summary,
                binding=binding,
            )
    return recovery_hashes, candidate_hashes


class Phase10FamilySeparationChecks(unittest.TestCase):
    """Token, bottleneck, family, schema and seed must agree, family by family."""

    def test_tokens_schemas_and_seeds_separate_ae64_from_ae32_and_from_ae128(
        self,
    ) -> None:
        self.assertEqual(FAMILIES, (64, 32))

        # The completed family is not constructible by this phase at all.
        with self.assertRaises(guards.HybridQConfigError):
            family.require_phase10_bottleneck(common.AE_TRAINING_BOTTLENECK)
        for rejected in (0, 16, 256, True, "64"):
            with self.assertRaises(guards.HybridQConfigError):
                family.require_phase10_bottleneck(rejected)  # type: ignore[arg-type]

        emitted: dict[int, list[str]] = {}
        for size in FAMILIES:
            label = family.family_label(size)
            self.assertEqual(label, f"AE{size}")
            self.assertEqual(
                family.family_id(size), ae_contract.family_for_bottleneck(size)
            )
            emitted[size] = [
                family.training_token(size),
                family.holdout_token(size),
                family.training_terminal(size),
                family.holdout_terminal(size),
                family.training_schema(size),
                family.recovery_schema(size),
                family.candidate_schema(size),
                family.holdout_schema(size),
                family.candidate_filename(size, 4),
                family.recovery_filename(size, 4),
                family.holdout_selection_dirname(size),
                family.training_report_filename(size),
                family.epoch_summaries_filename(size),
                family.holdout_report_filename(size),
            ]
            for text in emitted[size]:
                # Names its own family, and no other AE family, ever.
                self.assertIn(f"ae{size}", text.lower())
                self.assertNotIn("ae128", text.lower())
                other = 32 if size == 64 else 64
                self.assertNotIn(f"ae{other}", text.lower())

        # The two families share no emitted name, and neither reuses an AE128
        # schema or the AE128 candidate filename.
        self.assertFalse(set(emitted[64]).intersection(emitted[32]))
        ae128_schemas = {
            common.AE_TRAINING_SCHEMA,
            common.AE_RECOVERY_SCHEMA,
            common.AE_CANDIDATE_SCHEMA,
            common.AE_HOLDOUT_SCHEMA,
            common.candidate_filename(4),
        }
        for size in FAMILIES:
            self.assertFalse(set(emitted[size]).intersection(ae128_schemas))

        # The registered commands, exactly as documented.
        self.assertEqual(family.training_token(64), "SPLITFUSION_AE64_PHASE10_TRAINING")
        self.assertEqual(family.training_token(32), "SPLITFUSION_AE32_PHASE10_TRAINING")
        self.assertEqual(
            family.holdout_token(64), "SPLITFUSION_AE64_PHASE10_HOLDOUT_SELECTION"
        )
        self.assertEqual(
            family.holdout_token(32), "SPLITFUSION_AE32_PHASE10_HOLDOUT_SELECTION"
        )

        # Token and --bottleneck must agree exactly, in both commands.
        for kind, token in (("training", family.training_token), ("holdout", family.holdout_token)):
            for size in FAMILIES:
                self.assertEqual(
                    family.require_token_agrees_with_bottleneck(
                        token(size), size, kind=kind
                    ),
                    size,
                )
                crossed = 32 if size == 64 else 64
                with self.assertRaises(guards.HybridQConfigError):
                    family.require_token_agrees_with_bottleneck(
                        token(size), crossed, kind=kind
                    )
                with self.assertRaises(guards.HybridQConfigError):
                    family.require_token_agrees_with_bottleneck(
                        token(size), common.AE_TRAINING_BOTTLENECK, kind=kind
                    )
        # A selection token is not a training token, and vice versa.
        with self.assertRaises(guards.HybridQConfigError):
            family.require_token_agrees_with_bottleneck(
                family.holdout_token(64), 64, kind="training"
            )
        with self.assertRaises(guards.HybridQConfigError):
            family.require_token_agrees_with_bottleneck(
                family.training_token(32), 32, kind="holdout"
            )
        # No AE128 token can be routed through this phase.
        for foreign in (
            "SPLITFUSION_AE128_PHASE9C_TRAINING",
            "SPLITFUSION_AE128_PHASE9C_HOLDOUT_SELECTION",
        ):
            for kind in ("training", "holdout"):
                with self.assertRaises(guards.HybridQConfigError):
                    family.require_token_agrees_with_bottleneck(foreign, 64, kind=kind)

        # Deterministic initialization is separated per family, and the locked
        # configuration is inherited from AE128 with only the family keys moved.
        seeds = {size: ae_contract.ae_init_seed(size) for size in FAMILIES}
        self.assertEqual(seeds[64], ae_contract.AE_INIT_BASE_SEED + 64)
        self.assertEqual(seeds[32], ae_contract.AE_INIT_BASE_SEED + 32)
        self.assertNotEqual(seeds[64], seeds[32])
        self.assertNotEqual(
            set(seeds.values()),
            {ae_contract.ae_init_seed(common.AE_TRAINING_BOTTLENECK)},
        )
        inherited = common.training_configuration()
        for size in FAMILIES:
            configuration = family.training_configuration(size)
            self.assertEqual(family.process_seed(size), seeds[size])
            self.assertEqual(configuration["init_seed"], seeds[size])
            self.assertEqual(configuration["family"], f"AE{size}")
            self.assertEqual(configuration["bottleneck"], size)
            self.assertEqual(configuration["phase"], family.PHASE)
            delta = family.configuration_delta(size)
            self.assertEqual(
                set(delta["changed"]), set(family.FAMILY_DEPENDENT_KEYS)
            )
            # Every other locked scientific setting is byte-identical to AE128.
            for key, value in inherited.items():
                if key in family.FAMILY_DEPENDENT_KEYS:
                    continue
                self.assertEqual(configuration[key], value, msg=key)
            self.assertEqual(configuration["epochs"], 12)
            self.assertEqual(configuration["stage_a_epochs"], 4)
            self.assertEqual(configuration["stage_b_epochs"], 8)
            self.assertEqual(configuration["stage_a_learning_rate"], 1e-3)
            self.assertEqual(configuration["stage_b_learning_rate"], 3e-4)
            self.assertEqual(configuration["stage_b_q_cycle"], [0.0, 0.30, 0.50, 0.70])
            self.assertTrue(configuration["stage_b_cycle_carries_across_epochs"])
            self.assertEqual(
                configuration["stage_b_q_update_counts"],
                {"0.00": 1694, "0.30": 1694, "0.50": 1694, "0.70": 1694},
            )
            self.assertEqual(configuration["weight_decay"], 1e-4)
            self.assertEqual(configuration["grad_clip_global_norm"], 5.0)
            self.assertEqual(configuration["batch_size"], 16)
            self.assertFalse(configuration["drop_last"])
            self.assertFalse(configuration["augmentation"])
            self.assertFalse(configuration["fake_quantization_in_training"])
            self.assertFalse(configuration["zstd_in_training"])
            self.assertEqual(configuration["candidate_epochs"], [4, 8, 12])
            self.assertEqual(
                configuration["optimization_frames"], contract.TRAIN_FIT_FRAMES
            )
            self.assertFalse(configuration["validation_or_test_accessed"])
        self.assertNotEqual(
            family.training_configuration(64), family.training_configuration(32)
        )
        for size in FAMILIES:
            self.assertNotEqual(family.training_configuration(size), inherited)

        # One family per process: the second family is refused.
        with mock.patch.object(family, "_BOUND_FAMILY", None):
            self.assertEqual(family.bind_process_family(64), 64)
            self.assertEqual(family.bind_process_family(64), 64)
            with self.assertRaises(guards.HybridQOwnershipError):
                family.bind_process_family(32)


class Phase10CheckpointAndOptimizerChecks(unittest.TestCase):
    """Family-specific checkpoint naming and eight-tensor optimizer ownership."""

    def test_checkpoint_names_and_optimizer_ownership_are_family_specific(self) -> None:
        binding = synthetic_binding()
        autoencoders = {
            size: family.build_family_ae(size, torch.device("cpu")) for size in FAMILIES
        }
        for size, autoencoder in autoencoders.items():
            self.assertEqual(autoencoder.bottleneck, size)
            self.assertEqual(autoencoder.family_name, f"AE{size}")
            self.assertEqual(autoencoder.init_seed, ae_contract.ae_init_seed(size))
            # Exactly the eight trainable tensors, and the optimizer owns them
            # and nothing else -- not even the other family's tensors.
            tensors = ae_parameters(autoencoder)
            self.assertEqual(len(tensors), family.AE_TRAINABLE_TENSOR_COUNT)
            other = autoencoders[32 if size == 64 else 64]
            common.freeze(other)
            optimizer = family.build_family_optimizer(
                autoencoder,
                lr=common.learning_rate_for_stage(common.AE_STAGE_A),
                frozen_modules=(other,),
            )
            owned = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            self.assertEqual(len(owned), family.AE_TRAINABLE_TENSOR_COUNT)
            self.assertEqual(
                {id(parameter) for parameter in owned},
                {id(parameter) for parameter in tensors},
            )
            self.assertFalse(
                {id(parameter) for parameter in owned}.intersection(
                    id(parameter) for parameter in other.parameters()
                )
            )
            # The same optimizer is not an AE-only optimizer for the other family.
            with self.assertRaises(guards.HybridQOwnershipError):
                ae_loss.require_ae_only_optimizer(optimizer, other)
            # Restore the frozen companion for its own turn.
            for parameter in other.parameters():
                parameter.requires_grad_(True)
            other.train()

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for size in FAMILIES:
                autoencoder = autoencoders[size]
                for parameter in autoencoder.parameters():
                    parameter.requires_grad_(True)
                directory = root / f"ae{size}"
                directory.mkdir()
                name = family.candidate_filename(size, 8)
                self.assertEqual(name, f"ae{size}_epoch_08.pt")
                digest = family.save_candidate(
                    directory / name,
                    bottleneck=size,
                    epoch=8,
                    autoencoder=autoencoder,
                    global_update_index=8 * common.batches_per_epoch(),
                    stage_b_position=ae_phase10_training.expected_stage_b_position(8),
                    binding=binding,
                )
                self.assertEqual(digest, sha256_file(directory / name))
                payload = torch.load(
                    directory / name, map_location="cpu", weights_only=False
                )
                self.assertEqual(payload["schema"], family.candidate_schema(size))
                self.assertEqual(payload["family"], f"AE{size}")
                self.assertEqual(payload["bottleneck"], size)
                self.assertEqual(payload["init_seed"], ae_contract.ae_init_seed(size))
                self.assertNotIn("ae128", payload["schema"])

                # A candidate cannot be saved under another family's filename.
                with self.assertRaises(guards.HybridQConfigError):
                    family.save_candidate(
                        directory / common.candidate_filename(8),
                        bottleneck=size,
                        epoch=8,
                        autoencoder=autoencoder,
                        global_update_index=8 * common.batches_per_epoch(),
                        stage_b_position=0,
                        binding=binding,
                    )
                # Nor loaded as the other family.
                crossed = 32 if size == 64 else 64
                with self.assertRaises(guards.HybridQConfigError):
                    family.load_candidate(
                        directory / name, crossed, 8, torch.device("cpu"), binding
                    )
                restored, metadata = family.load_candidate(
                    directory / name, size, 8, torch.device("cpu"), binding
                )
                self.assertEqual(restored.bottleneck, size)
                self.assertEqual(metadata["family"], f"AE{size}")
                self.assertFalse(
                    any(parameter.requires_grad for parameter in restored.parameters())
                )
                # A source-binding edit since the checkpoint is drift, not detail.
                with self.assertRaises(guards.HybridQConfigError):
                    family.load_candidate(
                        directory / name,
                        size,
                        8,
                        torch.device("cpu"),
                        synthetic_binding(ae_source="f" * 64),
                    )
                # Epoch 5 is not a candidate epoch in the locked schedule.
                with self.assertRaises(guards.HybridQConfigError):
                    family.save_candidate(
                        directory / family.candidate_filename(size, 4),
                        bottleneck=size,
                        epoch=5,
                        autoencoder=autoencoder,
                        global_update_index=0,
                        stage_b_position=0,
                        binding=binding,
                    )


class Phase10RecoveryAndHoldoutIsolationChecks(unittest.TestCase):
    """Resume rebuilds one family's record; selection stays a separate command."""

    def test_recovery_is_family_scoped_and_selection_stays_separate(self) -> None:
        partition, dataset = registered_shape_partition()
        binding = synthetic_binding()
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "ae64_run"
            written_recovery, written_candidates = write_partial_family_run(
                output, 64, partition, dataset, binding, through_epoch=8
            )
            self.assertEqual(len(written_recovery), 8)
            self.assertEqual(
                set(written_candidates), {"ae64_epoch_04.pt", "ae64_epoch_08.pt"}
            )
            self.assertEqual(
                sorted(path.name for path in (output / "recovery").iterdir()),
                [f"ae64_recovery_epoch_{epoch:02d}.pt" for epoch in range(1, 9)],
            )
            self.assertTrue((output / "ae64_epoch_summaries.json").is_file())

            autoencoder = family.build_family_ae(64, torch.device("cpu"))
            optimizer = family.build_family_optimizer(
                autoencoder,
                lr=common.learning_rate_for_stage(common.AE_STAGE_A),
                frozen_modules=(),
            )
            state = ae_phase10_training.TrainingState()
            (
                completed,
                summaries,
                recovery_hashes,
                candidate_hashes,
            ) = ae_phase10_training.restore_completed_epochs(
                bottleneck=64,
                output=output,
                recovery_dir=output / "recovery",
                checkpoint_dir=output / "checkpoints",
                autoencoder=autoencoder,
                optimizer=optimizer,
                state=state,
                binding=binding,
                partition=partition,
                dataset=dataset,
                device=torch.device("cpu"),
            )
            # Counters continue rather than replay, and every completed file is
            # back in the bookkeeping.
            self.assertEqual(completed, 8)
            self.assertEqual(state.global_update_index, 8 * common.batches_per_epoch())
            self.assertEqual(state.stage_b_position, 4 * common.batches_per_epoch())
            self.assertEqual([row["epoch"] for row in summaries], list(range(1, 9)))
            self.assertEqual(recovery_hashes, written_recovery)
            self.assertEqual(candidate_hashes, written_candidates)
            self.assertTrue(
                all(row["family"] == "AE64" for row in summaries)
            )

            # The other family cannot resume from this run.
            with self.assertRaises(guards.HybridQOwnershipError):
                ae_phase10_training.completed_recovery_epochs(output / "recovery", 32)
            with self.assertRaises(guards.HybridQOwnershipError):
                ae_phase10_training.restore_completed_epochs(
                    bottleneck=32,
                    output=output,
                    recovery_dir=output / "recovery",
                    checkpoint_dir=output / "checkpoints",
                    autoencoder=family.build_family_ae(32, torch.device("cpu")),
                    optimizer=optimizer,
                    state=ae_phase10_training.TrainingState(),
                    binding=binding,
                    partition=partition,
                    dataset=dataset,
                    device=torch.device("cpu"),
                )

            # Training refuses to run once this run holds a selection output.
            self.assertEqual(
                ae_phase10_training.require_no_selection_output(output, 64), []
            )
            (output / family.holdout_selection_dirname(64)).mkdir()
            for size in FAMILIES:
                with self.assertRaises(guards.HybridQOwnershipError):
                    ae_phase10_training.require_no_selection_output(output, size)

        # The selection runners are a separate command: merely importing one
        # stops training, and the trainer's own source imports neither. This
        # test process has the AE128 selection module loaded (the shared
        # fixtures import it), so the guard is observed firing for real here.
        with self.assertRaises(guards.HybridQOwnershipError):
            ae_phase10_training.require_holdout_unopened()
        for module_name in ae_phase10_training.HOLDOUT_MODULES:
            with mock.patch.dict(sys.modules, {module_name: mock.MagicMock()}):
                with self.assertRaises(guards.HybridQOwnershipError):
                    ae_phase10_training.require_holdout_unopened()
        # ... and passes when no selection runner is present.
        with mock.patch.object(
            ae_phase10_training,
            "HOLDOUT_MODULES",
            ("splitfusion_ae_selection_module_that_is_not_imported",),
        ):
            self.assertIsNone(ae_phase10_training.require_holdout_unopened())
        self.assertEqual(
            ae_phase10_training.HOLDOUT_MODULES,
            tuple(
                "pole_lraspp_multimodal_fusion.object_head_pilot_v1."
                f"splitfusion_fcos_r50_fpn_p2_p7_ae_v1.{name}"
                for name in ("ae_holdout_selection", "ae_phase10_holdout_selection")
            ),
        )
        source = Path(ae_phase10_training.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            self.assertFalse(
                [name for name in names if "holdout_selection" in name],
                msg="the trainer must not import a selection runner",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
