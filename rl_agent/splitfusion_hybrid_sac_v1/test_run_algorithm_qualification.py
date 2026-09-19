"""Tests for the bounded algorithm-qualification CLI and evidence renderer."""

from __future__ import annotations

import csv
import json
import platform
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from .hybrid_sac_training_runner import (
    PHASE_LABEL,
    HybridSacAlgorithmQualificationRunnerV1,
)
from .run_algorithm_qualification import (
    DEFAULT_SEEDS,
    LEARNING_CURVE_FIELDS,
    MANIFEST_SCHEMA,
    QualificationRenderConfigV1,
    RenderError,
    SUMMARY_SCHEMA,
    build_argument_parser,
    render_algorithm_qualification,
    verify_artifact_directory,
)


def _tiny_config(output: Path) -> QualificationRenderConfigV1:
    return QualificationRenderConfigV1(
        output=output,
        seeds=(3, 5, 7),
        updates=2,
        evaluation_interval=1,
        evaluation_steps=24,
        replay_capacity=24,
        batch_size=4,
        warmup_transitions=4,
        collect_per_update=1,
        episode_horizon=4,
    )


class ConfigurationTests(unittest.TestCase):
    def test_cli_defaults_are_three_fixed_seeds_and_500_updates(self) -> None:
        parser = build_argument_parser()
        args = parser.parse_args(["--output", "/tmp/not-executed"])
        self.assertEqual(args.seeds, DEFAULT_SEEDS)
        self.assertEqual(args.seeds, (17, 29, 43))
        self.assertEqual(args.updates, 500)
        self.assertEqual(args.evaluation_interval, 25)
        self.assertEqual(args.evaluation_steps, 240)
        self.assertEqual(args.torch_threads, 1)

    def test_invalid_seed_and_interval_contracts_fail_closed(self) -> None:
        with self.assertRaises(RenderError):
            QualificationRenderConfigV1(
                output=Path("unused"), seeds=(1, 1, 2)
            )
        with self.assertRaises(RenderError):
            QualificationRenderConfigV1(
                output=Path("unused"), updates=2, evaluation_interval=3
            )
        with self.assertRaises(RenderError):
            QualificationRenderConfigV1(
                output=Path("unused"), torch_threads=0
            )


class RenderTests(unittest.TestCase):
    def test_tiny_render_is_complete_hash_bound_and_labelled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            config = _tiny_config(output)
            manifest = render_algorithm_qualification(config)
            self.assertEqual(manifest["schema"], MANIFEST_SCHEMA)
            self.assertEqual(manifest["phase_label"], PHASE_LABEL)
            verified = verify_artifact_directory(output)
            self.assertEqual(manifest, verified)

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
            self.assertEqual(summary["phase_label"], PHASE_LABEL)
            self.assertEqual(len(summary["seeds"]), 3)
            self.assertEqual(summary["parameters"]["updates"], 2)
            self.assertEqual(summary["parameters"]["torch_threads"], 1)
            self.assertIsNone(summary["parameters"]["output"])
            self.assertEqual(manifest["torch_threads"], 1)
            self.assertEqual(torch.get_num_threads(), 1)
            for seed in summary["seeds"]:
                self.assertEqual(seed["phase_label"], PHASE_LABEL)
                checkpoint = output / seed["checkpoint"]
                restored = HybridSacAlgorithmQualificationRunnerV1.load_checkpoint(
                    checkpoint,
                    expected_sha256=seed["checkpoint_sha256"],
                )
                self.assertEqual(restored.update_count, 2)
                restored.reset_fixed_evaluation_stream()
                reproduced = restored.evaluate(
                    config.evaluation_steps,
                    fixed_mode=config.fixed_mode,
                    fixed_q_e4=config.fixed_q_e4,
                    thresholds=config.thresholds(),
                )
                self.assertEqual(
                    reproduced.as_dict(), seed["post_evaluation"]
                )

            with (output / "learning_curve.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(rows[0]), LEARNING_CURVE_FIELDS)
            self.assertEqual(len(rows), 9)  # pre + updates 1 and 2, for 3 seeds
            self.assertTrue(all(row["phase_label"] == PHASE_LABEL for row in rows))
            report = (output / "REPORT.md").read_text()
            self.assertIn(PHASE_LABEL, report)
            self.assertIn("not SplitFusion evidence", report)
            self.assertIn("PyTorch intra-op threads: 1", report)
            self.assertIn("directly encodes its target mode", report)
            self.assertIn("does not measure generalization", report)
            deprecated_evaluation_term = "held" + "-out"
            self.assertNotIn(deprecated_evaluation_term, report.lower())

            # Provenance is names/hashes only; no diff content is present.
            self.assertIsInstance(manifest["git_dirty_paths"], list)
            self.assertTrue(manifest["git_head"])
            self.assertEqual(
                set(manifest["source_sha256"]),
                {
                    "rl_agent/splitfusion_hybrid_sac_v1/__init__.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/action_contract.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_models.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_trainer.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/hybrid_sac_training_runner.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/replay_buffer.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/reward_ticket_controller.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/run_algorithm_qualification.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/scene_descriptors.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/state_reward_transition_contract.py",
                    "rl_agent/splitfusion_hybrid_sac_v1/transaction_identity.py",
                },
            )
            self.assertEqual(
                manifest["runtime_versions"],
                summary["runtime_versions"],
            )
            self.assertEqual(
                manifest["runtime_versions"]["torch"], str(torch.__version__)
            )
            self.assertEqual(
                manifest["runtime_versions"]["python"],
                platform.python_version(),
            )

    def test_source_and_runtime_provenance_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            render_algorithm_qualification(_tiny_config(output))
            with mock.patch(
                "rl_agent.splitfusion_hybrid_sac_v1."
                "run_algorithm_qualification._source_bindings",
                return_value={},
            ):
                with self.assertRaisesRegex(RenderError, "source hashes"):
                    verify_artifact_directory(output, verify_sources=True)

            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["runtime_versions"]["python"] = "0.0.0"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RenderError, "runtime versions"):
                verify_artifact_directory(output, verify_sources=True)

    def test_two_tiny_runs_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            render_algorithm_qualification(_tiny_config(first))
            render_algorithm_qualification(_tiny_config(second))
            first_files = sorted(
                path.relative_to(first) for path in first.rglob("*") if path.is_file()
            )
            second_files = sorted(
                path.relative_to(second) for path in second.rglob("*") if path.is_file()
            )
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual(
                    (first / relative).read_bytes(),
                    (second / relative).read_bytes(),
                    str(relative),
                )

    def test_existing_output_is_refused_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(RenderError):
                render_algorithm_qualification(_tiny_config(empty))
            self.assertEqual(list(empty.iterdir()), [])

            nonempty = root / "nonempty"
            nonempty.mkdir()
            sentinel = nonempty / "sentinel.txt"
            sentinel.write_text("owned")
            with self.assertRaises(RenderError):
                render_algorithm_qualification(_tiny_config(nonempty))
            self.assertEqual(sentinel.read_text(), "owned")

    def test_artifact_tamper_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            render_algorithm_qualification(_tiny_config(output))
            summary = output / "summary.json"
            summary.write_bytes(summary.read_bytes() + b" ")
            with self.assertRaises(RenderError):
                verify_artifact_directory(output, verify_sources=False)


if __name__ == "__main__":
    unittest.main()
