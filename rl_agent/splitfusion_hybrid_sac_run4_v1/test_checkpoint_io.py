"""CPU-only adversarial tests for durable Run-4 checkpoint material."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import canonical_sha256

from . import checkpoint_io as src
from . import test_persistent_runner as fixture


class DurableCheckpointIoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture.PersistentRunnerTest.setUpClass()
        runner = fixture.PersistentRunnerTest.factory()
        runner.start(session_uuid=fixture.SESSION, ue_id=fixture.UE_ID)
        runner.step()
        cls.checkpoint = runner.checkpoint()
        cls.binding = cls.checkpoint.runner_binding_sha256

    @classmethod
    def tearDownClass(cls) -> None:
        fixture.PersistentRunnerTest.tearDownClass()

    def _write(self, root: Path):
        return src.write_checkpoint(root / "checkpoint", self.checkpoint)

    @staticmethod
    def _train_mechanics_to(runner, update_count: int) -> None:
        while runner.trainer.update_count < update_count:
            batch = runner.replay_buffer.sample(2, runner._replay_generator)
            runner.trainer.update_once(batch)

    def test_weights_only_round_trip_preserves_safe_material(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            material = src.read_checkpoint_material(
                artifact.directory,
                expected_runner_binding_sha256=self.binding,
                expected_manifest_sha256=artifact.manifest_sha256,
            )
        self.assertEqual(
            material.source_checkpoint_sha256,
            self.checkpoint.checkpoint_sha256,
        )
        self.assertEqual(len(material.journal), 1)
        source = self.checkpoint.journal[0]
        loaded = material.journal[0]
        self.assertEqual(loaded.decision.canonical_sha256, source.decision.canonical_sha256)
        self.assertEqual(loaded.prediction.canonical_sha256, source.prediction.canonical_sha256)
        self.assertEqual(
            loaded.successor_staged.canonical_sha256,
            source.successor_staged.canonical_sha256,
        )
        self.assertEqual(loaded.transition_sha256, source.transition.canonical_sha256())
        loaded.decision.action.require_reconciled()

    def test_safe_material_loads_in_fresh_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            program = """
import json, sys, torch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import checkpoint_io
cuda_before = torch.cuda.is_initialized()
m = checkpoint_io.read_checkpoint_material(
    sys.argv[1],
    expected_runner_binding_sha256=sys.argv[2],
    expected_manifest_sha256=sys.argv[3],
)
print(json.dumps({
    'checkpoint': m.source_checkpoint_sha256,
    'journal': len(m.journal),
    'action_reconciled': m.journal[0].decision.action.is_catalog_reconciled,
    'cuda_unchanged': torch.cuda.is_initialized() == cuda_before,
}, sort_keys=True))
"""
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    artifact.directory,
                    self.binding,
                    artifact.manifest_sha256,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        observed = json.loads(completed.stdout)
        self.assertEqual(observed["checkpoint"], self.checkpoint.checkpoint_sha256)
        self.assertEqual(observed["journal"], 1)
        self.assertTrue(observed["action_reconciled"])
        self.assertTrue(observed["cuda_unchanged"])

    def test_naive_full_object_pickle_loses_private_attestations(self) -> None:
        """Discriminating proof that opaque torch.save is not a valid design."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.pt"
            torch.save(self.checkpoint, path)
            program = """
import sys, torch
c = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
try:
    c.journal[0].transition.require_attested()
except Exception as exc:
    print(type(exc).__name__)
    raise SystemExit(0)
raise SystemExit(19)
"""
            completed = subprocess.run(
                [sys.executable, "-c", program, str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                completed.stdout.strip(),
                {"UnreconciledActionIdentityError", "TransitionError"},
            )

            weights_only = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys,torch\n"
                        "try:\n"
                        " torch.load(sys.argv[1],map_location='cpu',weights_only=True)\n"
                        "except Exception:\n"
                        " raise SystemExit(0)\n"
                        "raise SystemExit(23)\n"
                    ),
                    str(path),
                ],
                check=False,
            )
            self.assertEqual(weights_only.returncode, 0)

    def test_loader_always_requests_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            original = torch.load
            calls = []

            def observed(*args, **kwargs):
                calls.append(dict(kwargs))
                return original(*args, **kwargs)

            with mock.patch.object(src.torch, "load", side_effect=observed):
                src.read_checkpoint_material(
                    artifact.directory,
                    expected_runner_binding_sha256=self.binding,
                )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["weights_only"], True)
        self.assertEqual(calls[0]["map_location"], "cpu")

    def test_create_only_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = self._write(root)
            before = {
                name: (Path(artifact.directory) / name).read_bytes()
                for name in (src.PAYLOAD_FILENAME, src.MANIFEST_FILENAME)
            }
            with self.assertRaisesRegex(src.CheckpointWriteError, "already exists"):
                self._write(root)
            after = {
                name: (Path(artifact.directory) / name).read_bytes()
                for name in (src.PAYLOAD_FILENAME, src.MANIFEST_FILENAME)
            }
        self.assertEqual(before, after)

    def test_failed_atomic_publication_leaves_no_visible_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint"
            with mock.patch.object(src.os, "rename", side_effect=OSError("injected")):
                with self.assertRaisesRegex(OSError, "injected"):
                    src.write_checkpoint(target, self.checkpoint)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_payload_truncation_is_rejected_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            payload = Path(artifact.directory) / src.PAYLOAD_FILENAME
            data = payload.read_bytes()
            payload.write_bytes(data[: len(data) // 2])
            with self.assertRaisesRegex(src.CheckpointTamperError, "payload bytes"):
                src.read_checkpoint_material(
                    artifact.directory,
                    expected_runner_binding_sha256=self.binding,
                )

    def test_payload_bit_tamper_is_rejected_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            payload = Path(artifact.directory) / src.PAYLOAD_FILENAME
            data = bytearray(payload.read_bytes())
            data[len(data) // 2] ^= 1
            payload.write_bytes(data)
            with self.assertRaisesRegex(src.CheckpointTamperError, "payload bytes"):
                src.read_checkpoint_material(
                    artifact.directory,
                    expected_runner_binding_sha256=self.binding,
                )

    def test_manifest_tamper_is_rejected_by_external_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            manifest = Path(artifact.directory) / src.MANIFEST_FILENAME
            document = json.loads(manifest.read_bytes())
            document["source_checkpoint_sha256"] = canonical_sha256({"foreign": 1})
            manifest.write_bytes(src._canonical_json_bytes(document))
            with self.assertRaisesRegex(src.CheckpointTamperError, "manifest SHA"):
                src.read_checkpoint_material(
                    artifact.directory,
                    expected_runner_binding_sha256=self.binding,
                    expected_manifest_sha256=artifact.manifest_sha256,
                )

    def test_foreign_schema_is_rejected_without_relying_on_external_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            manifest = Path(artifact.directory) / src.MANIFEST_FILENAME
            document = json.loads(manifest.read_bytes())
            document["schema_id"] = "foreign.schema"
            manifest.write_bytes(src._canonical_json_bytes(document))
            with self.assertRaisesRegex(src.CheckpointReadError, "foreign.*schema"):
                src.read_checkpoint_material(
                    artifact.directory,
                    expected_runner_binding_sha256=self.binding,
                )

    def test_foreign_runner_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            with self.assertRaisesRegex(src.CheckpointReadError, "foreign runner"):
                src.read_checkpoint_material(
                    artifact.directory,
                    expected_runner_binding_sha256=canonical_sha256({"other": 1}),
                )

    def test_extra_directory_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            (Path(artifact.directory) / "unexpected").write_bytes(b"x")
            with self.assertRaisesRegex(src.CheckpointReadError, "member set"):
                src.read_checkpoint_material(
                    artifact.directory,
                    expected_runner_binding_sha256=self.binding,
                )

    def test_restore_reissues_attestations_and_continues_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            material = src.read_checkpoint_material(
                artifact.directory,
                expected_runner_binding_sha256=self.binding,
            )
        restored = src.restore_runner(
            material, fresh_factory=fixture.PersistentRunnerTest.factory
        )
        self.assertEqual(
            restored.checkpoint().checkpoint_sha256,
            self.checkpoint.checkpoint_sha256,
        )
        for row in restored.checkpoint().journal:
            row.transition.require_attested()

        expected = fixture.src._TestOnlyPersistentRunnerV1.restore(
            self.checkpoint,
            fresh_factory=fixture.PersistentRunnerTest.factory,
        )
        expected_next = expected.step()
        restored_next = restored.step()
        self.assertEqual(restored_next.canonical_sha256, expected_next.canonical_sha256)
        self.assertEqual(
            restored.checkpoint().checkpoint_sha256,
            expected.checkpoint().checkpoint_sha256,
        )

    def test_fresh_process_can_restore_and_continue_without_cuda(self) -> None:
        expected = fixture.src._TestOnlyPersistentRunnerV1.restore(
            self.checkpoint,
            fresh_factory=fixture.PersistentRunnerTest.factory,
        )
        expected_next = expected.step().canonical_sha256
        expected_checkpoint = expected.checkpoint().checkpoint_sha256
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._write(Path(directory))
            program = """
import json, sys, torch
from rl_agent.splitfusion_hybrid_sac_run4_v1 import checkpoint_io
from rl_agent.splitfusion_hybrid_sac_run4_v1 import test_persistent_runner as fixture
fixture.PersistentRunnerTest.setUpClass()
cuda_before = torch.cuda.is_initialized()
material = checkpoint_io.read_checkpoint_material(
    sys.argv[1],
    expected_runner_binding_sha256=sys.argv[2],
    expected_manifest_sha256=sys.argv[3],
)
runner = checkpoint_io.restore_runner(
    material, fresh_factory=fixture.PersistentRunnerTest.factory
)
checkpoint_before = runner.checkpoint().checkpoint_sha256
next_row = runner.step()
print(json.dumps({
    'checkpoint_before': checkpoint_before,
    'checkpoint_after': runner.checkpoint().checkpoint_sha256,
    'next_row': next_row.canonical_sha256,
    'cuda_unchanged': torch.cuda.is_initialized() == cuda_before,
}, sort_keys=True))
fixture.PersistentRunnerTest.tearDownClass()
"""
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    program,
                    artifact.directory,
                    self.binding,
                    artifact.manifest_sha256,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        observed = json.loads(completed.stdout)
        self.assertEqual(
            observed["checkpoint_before"], self.checkpoint.checkpoint_sha256
        )
        self.assertEqual(observed["next_row"], expected_next)
        self.assertEqual(observed["checkpoint_after"], expected_checkpoint)
        self.assertTrue(observed["cuda_unchanged"])

    def test_durable_resume_update_250_to_500_is_bit_identical(self) -> None:
        cuda_before = torch.cuda.is_initialized()
        uninterrupted = fixture.PersistentRunnerTest.factory()
        uninterrupted.start(session_uuid=fixture.SESSION, ue_id=fixture.UE_ID)
        for _ in range(len(uninterrupted.warmup_schedule)):
            uninterrupted.step()
        self._train_mechanics_to(uninterrupted, 250)
        checkpoint_250 = uninterrupted.checkpoint()

        with tempfile.TemporaryDirectory() as directory:
            artifact = src.write_checkpoint(
                Path(directory) / "update-250", checkpoint_250
            )
            material = src.read_checkpoint_material(
                artifact.directory,
                expected_runner_binding_sha256=(
                    checkpoint_250.runner_binding_sha256
                ),
                expected_manifest_sha256=artifact.manifest_sha256,
            )
            resumed = src.restore_runner(
                material, fresh_factory=fixture.PersistentRunnerTest.factory
            )

        self.assertEqual(resumed.trainer.update_count, 250)
        self.assertEqual(
            resumed.checkpoint().checkpoint_sha256,
            checkpoint_250.checkpoint_sha256,
        )
        while uninterrupted.trainer.update_count < 500:
            expected_batch = uninterrupted.replay_buffer.sample(
                2, uninterrupted._replay_generator
            )
            observed_batch = resumed.replay_buffer.sample(
                2, resumed._replay_generator
            )
            self.assertTrue(torch.equal(expected_batch.state, observed_batch.state))
            self.assertTrue(torch.equal(expected_batch.reward, observed_batch.reward))
            self.assertTrue(
                torch.equal(expected_batch.mode_id, observed_batch.mode_id)
            )
            self.assertTrue(torch.equal(expected_batch.q_e4, observed_batch.q_e4))
            expected_metrics = uninterrupted.trainer.update_once(expected_batch)
            observed_metrics = resumed.trainer.update_once(observed_batch)
            self.assertEqual(expected_metrics, observed_metrics)

        self.assertEqual(uninterrupted.trainer.update_count, 500)
        self.assertEqual(resumed.trainer.update_count, 500)
        self.assertEqual(
            uninterrupted.checkpoint().checkpoint_sha256,
            resumed.checkpoint().checkpoint_sha256,
        )
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)

    def test_material_rejects_tampered_journal_digest(self) -> None:
        material = src.DurableCheckpointMaterialV1.from_checkpoint(self.checkpoint)
        row = material.journal[0]
        with self.assertRaisesRegex(src.CheckpointIoError, "fingerprint mismatch"):
            src.PortableJournalEntryV1(
                decision=row.decision,
                prediction=row.prediction,
                successor_staged=row.successor_staged,
                transition_sha256=row.transition_sha256,
                environment_transition_sha256=row.environment_transition_sha256,
                journal_sha256=canonical_sha256({"tampered": True}),
            )


if __name__ == "__main__":
    unittest.main()
