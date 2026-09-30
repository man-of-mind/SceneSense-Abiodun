"""Separate-process recovery tests for the Run-5 campaign runner (CPU only, ~5 min).

A  uninterrupted 0 -> 500                  (one process)
B  0 -> 250, process exits                 (stop hook at 250)
B' --resume 250 -> 500                     (new process, no gradient replay)
D  0 -> ~120, SIGTERM, emergency bundle    (new process)
D' --resume -> 500                         (new process)

Final bundles, ledgers, actor, critics, targets, optimizers, generators,
replay digest and channel state must be bit-identical across A, B' and D'.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_bundle as B
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_campaign as CAMPAIGN
from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_preregistration as PR

WORKTREE = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = Path(os.environ.get("RUN5_EVIDENCE_ROOT", WORKTREE.parent / "abiodun"))


def launch(directory: Path, *extra: str) -> subprocess.Popen:
    command = [sys.executable, "-m", "rl_agent.splitfusion_hybrid_sac_run5_v1.run5_campaign",
               "--mode", "smoke", "--campaign-dir", str(directory), "--seed", "17",
               "--evidence-root", str(EVIDENCE_ROOT), "--skip-host-check-for-tests", *extra]
    return subprocess.Popen(command, cwd=WORKTREE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def finish(process: subprocess.Popen, timeout: float = 1800) -> tuple[int, str, str]:
    out, err = process.communicate(timeout=timeout)
    return process.returncode, out, err


def fingerprint(seed_dir: Path) -> dict:
    bundle = B.verify_bundle(seed_dir / "checkpoints" / "checkpoint_000500")
    state = B.torch_from_bytes(bundle.payload("training_state.pt"))
    from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_training as RT
    return {
        "boundary": json.loads(bundle.payload("event.json"))["boundary"],
        "actor": B.sha256_bytes(bundle.payload("actor_state_dict.pt")),
        "actor_tree": RT._tree_sha256(B.torch_from_bytes(bundle.payload("actor_state_dict.pt"))),
        "online_critics": RT._tree_sha256(state["online_critics"]),
        "target_critics": RT._tree_sha256(state["target_critics"]),
        "actor_optimizer": RT._tree_sha256(state["actor_optimizer"]),
        "critic_optimizer": RT._tree_sha256(state["critic_optimizer"]),
        "generators": RT._tree_sha256(state["generators"]),
        "channel": B.sha256_bytes(bundle.payload("channel_state.json")),
        "metrics": B.sha256_bytes((seed_dir / "metrics.jsonl").read_bytes()),
        "decisions": B.sha256_bytes((seed_dir / "decisions.jsonl").read_bytes()),
        "final_actor": json.loads((seed_dir / "SMOKE_SEED_COMPLETE.json").read_text())
        ["final_actor"]["tree_sha256"],
    }


class SeparateProcessResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="run5_campaign_test_")
        root = Path(cls.tmp.name)
        cls.a, cls.b, cls.d = root / "a", root / "b", root / "d"
        a = launch(cls.a)
        b = launch(cls.b, "--stop-after-update", "250")
        d = launch(cls.d)
        metrics = cls.d / "seed_17" / "metrics.jsonl"
        deadline = time.time() + 900
        while time.time() < deadline:
            if metrics.exists() and len(metrics.read_bytes().splitlines()) >= 120:
                break
            time.sleep(0.2)
        d.send_signal(signal.SIGTERM)
        cls.results = {"a": finish(a), "b": finish(b), "d": finish(d)}
        cls.results["b_resume"] = finish(launch(cls.b, "--resume"))
        cls.results["d_resume"] = finish(launch(cls.d, "--resume"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def completion_code_ok(self, key: str) -> None:
        code, out, err = self.results[key]
        # 2 = completed, only the (deliberately skipped) cold-host gate is false.
        self.assertIn(code, (0, 2), err[-2000:])
        seed_dir = {"a": self.a, "b_resume": self.b, "d_resume": self.d}[key] / "seed_17"
        report = json.loads((seed_dir / "RUN5_SMOKE_REPORT.json").read_text())
        failed = [g for g, ok in report["gates"].items() if not ok]
        self.assertEqual(failed, ["host_confirmed_cold"])

    def test_uninterrupted_run_completes(self) -> None:
        self.completion_code_ok("a")

    def test_exit_at_250_then_resume_is_bit_identical(self) -> None:
        self.assertEqual(self.results["b"][0], CAMPAIGN.EXIT_STOPPED, self.results["b"][2][-2000:])
        self.completion_code_ok("b_resume")
        self.assertEqual(fingerprint(self.b / "seed_17"), fingerprint(self.a / "seed_17"))
        runs = [json.loads(line) for line in (self.b / "seed_17" / "runs.jsonl").read_text()
                .splitlines()]
        resume = next(r for r in runs if r["event"] == "RESUME")
        self.assertEqual((resume["resumed_from"], resume["update_count"]),
                         ("checkpoint_000250", 250))

    def test_sigterm_writes_emergency_bundle_and_resume_is_bit_identical(self) -> None:
        self.assertEqual(self.results["d"][0], CAMPAIGN.EXIT_STOPPED, self.results["d"][2][-2000:])
        emergencies = sorted((self.d / "seed_17" / "checkpoints").glob("emergency_*"))
        self.assertEqual(len(emergencies), 1)
        bundle = B.verify_bundle(emergencies[0])
        self.assertEqual(bundle.manifest["kind"], "EMERGENCY")
        self.assertFalse(bundle.manifest["selection_candidate"])
        self.completion_code_ok("d_resume")
        self.assertEqual(fingerprint(self.d / "seed_17"), fingerprint(self.a / "seed_17"))
        runs = [json.loads(line) for line in (self.d / "seed_17" / "runs.jsonl").read_text()
                .splitlines()]
        self.assertEqual([r["event"] for r in runs], ["START", "STOPPED", "RESUME", "COMPLETE"])
        self.assertEqual(runs[1]["reason"], "SIGTERM")

    def test_final_actor_loads_directly_in_a_fresh_process(self) -> None:
        actor = self.a / "seed_17" / "final_actor_000500"
        code = subprocess.run(
            [sys.executable, "-m", "rl_agent.splitfusion_hybrid_sac_run5_v1.run5_campaign",
             "--verify-actor", str(actor), "--seed", "17", "--expect-update", "500"],
            cwd=WORKTREE, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        self.assertEqual(code.returncode, 0, code.stderr)
        result = json.loads(code.stdout.strip().splitlines()[-1])
        self.assertTrue(result["verified"])
        self.assertEqual(result["loader"], "torch.load(weights_only=True)")

    def test_resume_refuses_corrupt_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            import shutil
            copy = Path(tmp) / "c"
            shutil.copytree(self.b, copy)
            target = copy / "seed_17" / "checkpoints" / "checkpoint_000500" / "training_state.pt"
            data = bytearray(target.read_bytes())
            data[100] ^= 1
            target.write_bytes(bytes(data))
            code, _, err = finish(launch(copy, "--resume"))
            self.assertEqual(code, CAMPAIGN.EXIT_REFUSED, err[-2000:])
            self.assertIn("RUN5_REFUSED", err)
            self.assertIn("corrupt", err)


class PreflightTest(unittest.TestCase):
    def test_deep_mode_is_refused_without_authorization(self) -> None:
        self.assertFalse(CAMPAIGN.AUTHORIZATION.exists())
        code = CAMPAIGN.main(["--mode", "deep", "--campaign-dir", "/nonexistent", "--seed", "17",
                              "--evidence-root", str(EVIDENCE_ROOT)])
        self.assertEqual(code, CAMPAIGN.EXIT_REFUSED)

    def test_disk_preflight_estimates_the_three_seed_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = CAMPAIGN.disk_preflight(Path(tmp), seeds=PR.CONFIG.seed_order,
                                             target=PR.CONFIG.deep_target_update,
                                             checkpoints=PR.CONFIG.deep_checkpoints)
            self.assertEqual(report["estimate_bytes"], report["deep_three_seed_estimate_bytes"])
            self.assertGreater(report["reserve_bytes"], 0)
            self.assertEqual(report["required_bytes"],
                             report["estimate_bytes"] + report["reserve_bytes"])
            with mock.patch.object(CAMPAIGN.shutil, "disk_usage",
                                   return_value=type("U", (), {"free": 10})()):
                self.assertFalse(CAMPAIGN.disk_preflight(
                    Path(tmp), seeds=[17], target=500,
                    checkpoints=PR.CONFIG.smoke_checkpoints)["passed"])

    def test_sealed_preregistration_detects_source_drift(self) -> None:
        PR.load_sealed()
        with mock.patch.object(PR, "source_hashes",
                               return_value={**PR.source_hashes(), "x": "0" * 64}):
            with self.assertRaises(PR.PreregistrationError):
                PR.load_sealed()


if __name__ == "__main__":
    unittest.main()
