"""Durability, isolation, and exact-resume tests for the Run-2-v2 CLI."""

from __future__ import annotations

import csv
import json
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from .empirical_contextual_exact_p95_run2_runner_v2 import (
    PHASE_LABEL,
    REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
    ExactP95Run2RunnerConfigV2,
)
from .empirical_contextual_smoke_runner import (
    EmpiricalSmokeCheckpointV1,
    EmpiricalSmokeConfigV1,
)
from .run_empirical_contextual_exact_p95_run2_v2 import (
    BINDINGS_SCHEMA,
    CHECKPOINT_SELECTION,
    CONFIG_SCHEMA,
    FROZEN_COMPARATOR_COMMIT,
    FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256,
    FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256,
    OUTPUT_SCHEMA,
    REPORT_SCHEMA,
    Run2V2ArtifactError,
    load_exact_p95_run2_checkpoint_v2,
    run_seed_to_directory,
)


def _tree_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_directory(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


class CleanImportIsolationTest(unittest.TestCase):
    def test_clean_process_runner_and_cli_import_no_validation_stack(self) -> None:
        script = """
import json, sys
from rl_agent.splitfusion_hybrid_sac_v1 import empirical_contextual_exact_p95_run2_runner_v2
from rl_agent.splitfusion_hybrid_sac_v1 import run_empirical_contextual_exact_p95_run2_v2
forbidden = (
    'empirical_contextual_fit_validation_panel',
    'empirical_contextual_fit_validation_evaluator',
    'empirical_contextual_split_oracle',
    'empirical_contextual_exact_p95_deadline_penalty_v2',
)
print(json.dumps(sorted(name for name in sys.modules if any(item in name for item in forbidden))))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_cli_source_has_no_validation_import(self) -> None:
        source = Path(__file__).with_name(
            "run_empirical_contextual_exact_p95_run2_v2.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "from .empirical_contextual_fit_validation_panel import",
            "from .empirical_contextual_fit_validation_evaluator import",
            "from .empirical_contextual_split_oracle import",
        ):
            self.assertNotIn(forbidden, source)

    def test_comparator_drift_fails_before_output_creation(self) -> None:
        from . import run_empirical_contextual_exact_p95_run2_v2 as module

        config = ExactP95Run2RunnerConfigV2(
            seeds=(17,),
            warmup_transitions=4,
            batch_size=4,
            collect_per_update=1,
            update_count=1,
            replay_capacity=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must_not_exist"
            with mock.patch.dict(
                module._COMPARATOR_FILE_HASHES,
                {"REPORT.md": "0" * 64},
            ):
                with self.assertRaises(Run2V2ArtifactError):
                    run_seed_to_directory(
                        output_directory=output,
                        seed=17,
                        config=config,
                        checkpoint_interval_updates=1,
                        stop_after_updates=0,
                    )
            self.assertFalse(output.exists())


class CampaignIntegrityTest(unittest.TestCase):
    def test_interrupted_inventory_resume_and_subset_preserves_all_seeds(self) -> None:
        from . import run_empirical_contextual_exact_p95_run2_v2 as module

        comparator = module._fixed_comparator_binding(None)

        def fake_seed_runner(*, output_directory: Path, seed: int):
            if (output_directory / "report.json").is_file():
                return json.loads(
                    (output_directory / "report.json").read_text(
                        encoding="utf-8"
                    )
                )
            output_directory.mkdir(parents=True, exist_ok=True)
            module._atomic_json(
                output_directory / "config.json",
                module._config_document(
                    REGISTERED_EXACT_P95_RUN2_CONFIG_V2,
                    seed,
                    module.EXACT_P95_RUN2_CHECKPOINT_INTERVAL_UPDATES_V2,
                    comparator,
                ),
            )
            report = module._content_attested(
                {
                    "completed_updates": 5000,
                    "configured_updates": 5000,
                    "fixed_comparator": comparator,
                    "output_schema": OUTPUT_SCHEMA,
                    "phase_label": PHASE_LABEL,
                    "record": REPORT_SCHEMA,
                    "runner_binding_sha256": f"{seed:064x}",
                    "seed": seed,
                    "status": "COMPLETE",
                },
                field_name="report_content_sha256",
            )
            module._atomic_json(output_directory / "report.json", report)
            return report

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "campaign"
            # Simulate process death after seed 17 is durable but before a
            # campaign report is written.
            seed17 = fake_seed_runner(
                output_directory=output / "seed_17", seed=17
            )
            self.assertFalse((output / "campaign_report.json").exists())
            with mock.patch.object(
                module, "_preflight_existing_output", return_value=(None, None)
            ):
                campaign = module._run_campaign_to_directory(
                    output=output,
                    seeds=(29, 43),
                    comparator_binding=comparator,
                    seed_runner=fake_seed_runner,
                )
            self.assertEqual(campaign["seeds"], [17, 29, 43])
            self.assertEqual(
                [report["seed"] for report in campaign["reports"]],
                [17, 29, 43],
            )
            self.assertEqual(campaign["reports"][0], seed17)
            before_seed_reports = {
                seed: (output / f"seed_{seed}" / "report.json").read_bytes()
                for seed in (17, 29, 43)
            }

            # A subset rerun must retain completed unrequested seed reports
            # and the full campaign seed inventory.
            with mock.patch.object(
                module, "_preflight_existing_output", return_value=(None, None)
            ):
                subset = module._run_campaign_to_directory(
                    output=output,
                    seeds=(17,),
                    comparator_binding=comparator,
                    seed_runner=fake_seed_runner,
                )
            self.assertEqual(subset["seeds"], [17, 29, 43])
            self.assertEqual(
                before_seed_reports,
                {
                    seed: (output / f"seed_{seed}" / "report.json").read_bytes()
                    for seed in (17, 29, 43)
                },
            )

            # Even a correctly re-attested but internally inconsistent seed
            # list is rejected before a seed runner can be invoked.
            malformed = dict(subset)
            malformed.pop("campaign_report_content_sha256")
            malformed["seeds"] = [17]
            malformed = module._content_attested(
                malformed, field_name="campaign_report_content_sha256"
            )
            module._atomic_json(output / "campaign_report.json", malformed)
            before = _tree_bytes(output)
            calls = []

            def forbidden_runner(**kwargs):
                calls.append(kwargs)
                raise AssertionError("seed runner must not be invoked")

            with self.assertRaises(Run2V2ArtifactError):
                with mock.patch.object(
                    module,
                    "_preflight_existing_output",
                    return_value=(None, None),
                ):
                    module._run_campaign_to_directory(
                        output=output,
                        seeds=(17,),
                        comparator_binding=comparator,
                        seed_runner=forbidden_runner,
                    )
            self.assertEqual(calls, [])
            self.assertEqual(before, _tree_bytes(output))


class DurableRun2V2OutputTest(unittest.TestCase):
    def test_direct_interrupted_resume_and_fail_closed_outputs(self) -> None:
        config = ExactP95Run2RunnerConfigV2(
            seeds=(17, 29),
            warmup_transitions=4,
            batch_size=4,
            collect_per_update=2,
            update_count=2,
            replay_capacity=16,
        )
        python_before = random.getstate()
        torch_before = torch.get_rng_state().clone()
        threads_before = torch.get_num_threads()
        cuda_before = torch.cuda.is_initialized()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct_directory = root / "direct"
            resumed_directory = root / "resumed"
            direct = run_seed_to_directory(
                output_directory=direct_directory,
                seed=17,
                config=config,
                checkpoint_interval_updates=1,
            )
            partial = run_seed_to_directory(
                output_directory=resumed_directory,
                seed=17,
                config=config,
                checkpoint_interval_updates=1,
                stop_after_updates=1,
            )
            immutable_zero = (
                resumed_directory / "checkpoint_000000.pt"
            ).read_bytes()
            resumed = run_seed_to_directory(
                output_directory=resumed_directory,
                seed=17,
                config=config,
                checkpoint_interval_updates=1,
            )

            self.assertEqual(partial["status"], "IN_PROGRESS")
            self.assertEqual(direct["status"], "COMPLETE")
            self.assertEqual(resumed["status"], "COMPLETE")
            self.assertEqual(direct, resumed)
            self.assertEqual(_tree_bytes(direct_directory), _tree_bytes(resumed_directory))
            self.assertEqual(
                immutable_zero,
                (resumed_directory / "checkpoint_000000.pt").read_bytes(),
            )
            self.assertEqual(
                {path.name for path in resumed_directory.iterdir()},
                {
                    "bindings.json",
                    "checkpoint_000000.pt",
                    "checkpoint_latest.pt",
                    "checkpoints",
                    "config.json",
                    "metrics.csv",
                    "report.json",
                    "transition_reward_audit.csv",
                },
            )
            self.assertEqual(
                sorted(path.name for path in (resumed_directory / "checkpoints").iterdir()),
                ["checkpoint_000001.pt", "checkpoint_000002.pt"],
            )

            zero = load_exact_p95_run2_checkpoint_v2(
                resumed_directory / "checkpoint_000000.pt"
            )
            latest = load_exact_p95_run2_checkpoint_v2(
                resumed_directory / "checkpoint_latest.pt"
            )
            tail = load_exact_p95_run2_checkpoint_v2(
                resumed_directory / "checkpoints" / "checkpoint_000002.pt"
            )
            self.assertEqual(zero.update_count, 0)
            self.assertEqual(zero.collection_seq, 0)
            self.assertEqual(zero.environment_state.d1_state.reset_count, 0)
            self.assertFalse(zero.trainer_initialized)
            self.assertIsNone(zero.replay_binding)
            self.assertEqual(latest.checkpoint_sha256, tail.checkpoint_sha256)
            self.assertEqual(latest.update_count, 2)
            self.assertEqual(latest.collection_seq, 8)
            self.assertEqual(latest.replay_evicted_count, 0)

            config_json = json.loads(
                (resumed_directory / "config.json").read_text(encoding="utf-8")
            )
            bindings = json.loads(
                (resumed_directory / "bindings.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (resumed_directory / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config_json["record"], CONFIG_SCHEMA)
            self.assertEqual(bindings["record"], BINDINGS_SCHEMA)
            self.assertEqual(report["record"], REPORT_SCHEMA)
            self.assertEqual(report["output_schema"], OUTPUT_SCHEMA)
            self.assertEqual(report["phase_label"], PHASE_LABEL)
            self.assertEqual(report["checkpoint_selection"], CHECKPOINT_SELECTION)
            self.assertEqual(report["sampling_split"], "train")
            self.assertEqual(
                report["fixed_comparator"]["frozen_commit"],
                FROZEN_COMPARATOR_COMMIT,
            )
            self.assertEqual(
                report["fixed_comparator"]["summary_content_sha256"],
                FROZEN_COMPARATOR_SUMMARY_CONTENT_SHA256,
            )
            self.assertEqual(
                report["fixed_comparator"]["manifest_content_sha256"],
                FROZEN_COMPARATOR_MANIFEST_CONTENT_SHA256,
            )
            self.assertTrue(
                report["fixed_comparator"]["frozen_before_run2_outcomes"]
            )
            self.assertEqual(
                report["fixed_comparator"]["fixed_action"]["mode_id"], 11
            )
            self.assertEqual(
                report["fixed_comparator"]["fixed_action"]["q_e4"], 6000
            )
            self.assertEqual(report["completed_updates"], 2)
            self.assertEqual(report["summary"]["transition_count"], 8)
            self.assertNotIn("selected_checkpoint", report)
            self.assertEqual(
                report["artifact_file_sha256"]["checkpoint_000000.pt"],
                __import__("hashlib").sha256(immutable_zero).hexdigest(),
            )

            with (resumed_directory / "metrics.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                metric_rows = list(csv.DictReader(stream))
            with (resumed_directory / "transition_reward_audit.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                audit_rows = list(csv.DictReader(stream))
            self.assertEqual(len(metric_rows), 2)
            self.assertEqual(len(audit_rows), 8)
            self.assertEqual(
                [row["update_index"] for row in metric_rows], ["1", "2"]
            )
            self.assertTrue(
                all(
                    row["target_reward_signed_bit_mismatch_count"] == "0"
                    for row in metric_rows
                )
            )
            self.assertEqual(
                [int(row["collection_seq"]) for row in audit_rows], list(range(8))
            )
            self.assertTrue(
                all(row["collection_session_uuid"] == zero.collection_session_uuid for row in audit_rows)
            )
            self.assertTrue(
                any(
                    float(row["source_d1_reward64"])
                    != float(row["emitted_reward_float32"])
                    for row in audit_rows
                )
            )
            for row in audit_rows:
                reward32 = float(row["emitted_reward_float32"])
                bits = struct.unpack(">I", struct.pack(">f", reward32))[0]
                self.assertEqual(
                    row["emitted_reward_float32_bits_hex"], f"0x{bits:08x}"
                )
                self.assertRegex(row["source_d1_transition_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(row["run2_v2_transition_sha256"], r"^[0-9a-f]{64}$")

            # A subset campaign must validate every existing seed's physical
            # artifacts before invoking a requested seed or writing a campaign
            # report.  Seed 17 is deliberately unrequested here.
            from . import run_empirical_contextual_exact_p95_run2_v2 as module

            campaign_output = root / "corrupt_unrequested_campaign"
            _copy_directory(
                resumed_directory, campaign_output / "seed_17"
            )
            with (campaign_output / "seed_17" / "metrics.csv").open(
                "ab"
            ) as stream:
                stream.write(b"physical-corruption")
            campaign_before = _tree_bytes(campaign_output)
            campaign_runner_calls = []

            def forbidden_campaign_runner(**kwargs):
                campaign_runner_calls.append(kwargs)
                raise AssertionError("seed runner must not be invoked")

            with self.assertRaises(Run2V2ArtifactError):
                module._run_campaign_to_directory(
                    output=campaign_output,
                    seeds=(29,),
                    comparator_binding=report["fixed_comparator"],
                    seed_runner=forbidden_campaign_runner,
                    config=config,
                    checkpoint_interval_updates=1,
                )
            self.assertEqual(campaign_runner_calls, [])
            self.assertEqual(campaign_before, _tree_bytes(campaign_output))
            self.assertFalse(
                (campaign_output / "campaign_report.json").exists()
            )

            # Every malformed or foreign reusable directory is rejected
            # before another byte in that directory changes.
            cases = {}
            foreign_config = _copy_directory(
                resumed_directory, root / "foreign_config"
            )
            (foreign_config / "config.json").write_text(
                json.dumps({"record": "splitfusion.preliminary_baseline_config.v1"}),
                encoding="utf-8",
            )
            cases["foreign_config"] = foreign_config

            foreign_binding = _copy_directory(
                resumed_directory, root / "foreign_binding"
            )
            (foreign_binding / "bindings.json").write_text(
                json.dumps({"record": "splitfusion.preliminary_baseline_bindings.v1"}),
                encoding="utf-8",
            )
            cases["foreign_binding"] = foreign_binding

            foreign_checkpoint = _copy_directory(
                resumed_directory, root / "foreign_checkpoint"
            )
            v1_config = EmpiricalSmokeConfigV1(
                seeds=(17,),
                warmup_transitions=4,
                batch_size=4,
                collect_per_update=2,
                update_count=2,
                replay_capacity=16,
            )
            local_generator = torch.Generator(device="cpu")
            local_generator.manual_seed(1)
            v1_checkpoint = EmpiricalSmokeCheckpointV1(
                config=v1_config,
                seed=17,
                runner_binding_sha256="0" * 64,
                collection_session_uuid=zero.collection_session_uuid,
                actor_state={},
                critics_state={},
                actor_optimizer_state={},
                critic_optimizer_state={},
                init_rng_state=random.Random(1).getstate(),
                collection_rng_state=local_generator.get_state().clone(),
                replay_rng_state=local_generator.get_state().clone(),
                actor_update_rng_state=local_generator.get_state().clone(),
                environment_state=zero.environment_state.d1_state,
                transition_history=(),
                metrics=(),
                collection_seq=0,
                warmup_collected=0,
                post_warmup_collected=0,
                update_count=0,
                support_violations=0,
                checkpoint_sha256="0" * 64,
            )
            torch.save(
                v1_checkpoint,
                foreign_checkpoint / "checkpoint_latest.pt",
            )
            cases["foreign_checkpoint"] = foreign_checkpoint

            malformed_checkpoint = _copy_directory(
                resumed_directory, root / "malformed_checkpoint"
            )
            (malformed_checkpoint / "checkpoint_latest.pt").write_bytes(b"broken")
            cases["malformed_checkpoint"] = malformed_checkpoint

            foreign_output = _copy_directory(
                resumed_directory, root / "foreign_output"
            )
            (foreign_output / "RUN1_OUTPUT").write_bytes(b"foreign")
            cases["foreign_output"] = foreign_output

            for label, directory in cases.items():
                before = _tree_bytes(directory)
                with self.assertRaises(Run2V2ArtifactError, msg=label):
                    run_seed_to_directory(
                        output_directory=directory,
                        seed=17,
                        config=config,
                        checkpoint_interval_updates=1,
                    )
                self.assertEqual(before, _tree_bytes(directory), label)

        self.assertEqual(random.getstate(), python_before)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        self.assertEqual(torch.get_num_threads(), threads_before)
        if not cuda_before:
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
