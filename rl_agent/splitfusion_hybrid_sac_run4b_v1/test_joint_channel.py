"""RUN4B_JOINT_CHANNEL_COMPARATOR_V1 tests (CPU-only).

Run with ``env -u PYTHONPATH CUDA_VISIBLE_DEVICES=``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

torch.set_num_threads(4)

from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as R4O
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import contract as C
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import environment as E
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import joint_channel as JC
from rl_agent.splitfusion_hybrid_sac_run4b_v1 import runner as R

PACKAGE = Path(R.__file__).resolve().parent
CROSSCHECK = json.loads((PACKAGE / "evidence_joint_channel_v1" /
                         "RUN5B_CHANNEL_TAPE_CROSSCHECK.json").read_text())
PILOT_SMOKE_MANIFEST = R.ROOT / (
    "rl_agent/experiments/splitfusion_hybrid_sac_run4b_v1/20261001_a589250/"
    "smoke/seed_17/checkpoints/update_000500/manifest.json")


def _bits(values) -> bytes:
    return struct.pack(f"<{len(values)}d", *values)


class _ReplayMcs:
    """SNR-free stand-in that replays an exact MCS sequence."""

    def __init__(self, sequence):
        self._sequence = list(sequence)

    def reset(self):
        return self._sequence.pop(0)

    def step(self):
        return type("Step", (), {"successor_mcs": self._sequence.pop(0)})()


class JointChannelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sources = JC.JointSharedSourcesV1.load()
        cls.pilot_sources = E.SharedSourcesV1.load()
        cls.schedule = R4O.build_frozen_warmup_schedule(17)

    def _actions(self, n, reverse=False):
        actions = [(self.schedule.action_at(i % 288).mode_id,
                    self.schedule.action_at(i % 288).q_e4) for i in range(n)]
        return actions[::-1] if reverse else actions

    def _tape(self, seed, n, reverse=False):
        env = JC.Run4BJointChannelEnvironmentV1(self.sources, seed=seed)
        tape = []
        for mode_id, q_e4 in self._actions(n, reverse):
            tape.append(env.channel_tape_entry())
            env.step(mode_id, q_e4)
        return tape

    def test_imported_run5_files_are_byte_identical(self) -> None:
        JC.verify_imported_files()
        has_ref = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet",
             JC.RUN5B_SOURCE_COMMIT + "^{commit}"], cwd=R.ROOT,
            capture_output=True).returncode == 0
        if not has_ref:
            self.skipTest("Run-5B source commit not present in this clone")
        for relpath in JC.RUN5_IMPORTED_FILES:
            blob = subprocess.run(
                ["git", "rev-parse", f"{JC.RUN5B_SOURCE_COMMIT}:{relpath}"],
                cwd=R.ROOT, capture_output=True, text=True, check=True).stdout
            local = subprocess.run(["git", "hash-object", relpath], cwd=R.ROOT,
                                   capture_output=True, text=True,
                                   check=True).stdout
            self.assertEqual(blob.strip(), local.strip(), relpath)

    def test_channel_binding_equals_run5b(self) -> None:
        self.assertEqual(self.sources.channel_binding_sha256,
                         JC.REGISTERED_CHANNEL_BINDING_SHA256)
        for seed, entry in CROSSCHECK["seeds"].items():
            self.assertEqual(entry["channel_binding_sha256"],
                             JC.REGISTERED_CHANNEL_BINDING_SHA256)
            self.assertEqual(entry["channel_seed"], JC.J.derive_seed(
                int(seed), JC.CHANNEL_SEED_LABEL))
        self.assertEqual(CROSSCHECK["run5b_source_commit"],
                         JC.RUN5B_SOURCE_COMMIT)

    def test_tapes_equal_run5b_collector_and_are_action_independent(self):
        n = CROSSCHECK["decisions"]
        for seed, entry in CROSSCHECK["seeds"].items():
            tape = self._tape(int(seed), n)
            self.assertEqual(hashlib.sha256(json.dumps(tape).encode())
                             .hexdigest(), entry["tape_sha256"], seed)
            self.assertEqual(tape[:5], entry["first5"])
        self.assertEqual(self._tape(29, 300), self._tape(29, 300, reverse=True))

    def test_snr_never_enters_state_or_context(self) -> None:
        self.assertNotIn("snr", " ".join(C.FEATURE_ORDER).lower())
        self.assertEqual({f.name for f in dataclasses.fields(C.ObservationV1)},
                         {"camera_si", "radar_p40", "prior_ul_mcs",
                          "pre_action_rlc_backlog_bytes"})
        env = JC.Run4BJointChannelEnvironmentV1(self.sources, seed=17)
        for mode_id, q_e4 in self._actions(20):
            self.assertFalse(any("snr" in key for key in env.context))
            env.step(mode_id, q_e4)

    def test_state_depends_on_snr_only_through_mcs(self) -> None:
        """Replaying the joint MCS sequence in the SNR-free environment
        reproduces every state, reward and transition digest bit-for-bit."""
        joint = JC.Run4BJointChannelEnvironmentV1(self.sources, seed=43)
        sequence = [joint._mcs_current]
        transitions = []
        for mode_id, q_e4 in self._actions(288):
            transitions.append(joint.step(mode_id, q_e4))
            sequence.append(joint._mcs_current)
        replay = E.Run4BEnvironmentV1(self.pilot_sources, seed=43)
        replay._mcs = _ReplayMcs(sequence)
        replay._mcs_current = replay._mcs.reset()
        replay.context["prior_ul_mcs"] = replay._mcs_current
        snr_values = set()
        for (mode_id, q_e4), expected in zip(self._actions(288), transitions):
            actual = replay.step(mode_id, q_e4)
            self.assertEqual(_bits(actual.state), _bits(expected.state))
            self.assertEqual(_bits(actual.next_state), _bits(expected.next_state))
            self.assertEqual(actual.reward, expected.reward)
            self.assertEqual(actual.digest(), expected.digest())
            snr_values.add(expected.diagnostics["channel_snr_db_at_decision"])
        self.assertGreater(len(snr_values), 100)  # SNR did vary

    def test_restore_reproduces_channel_and_every_stream(self) -> None:
        env = JC.Run4BJointChannelEnvironmentV1(self.sources, seed=29)
        actions = self._actions(288)
        for mode_id, q_e4 in actions[:120]:
            env.step(mode_id, q_e4)
        snapshot = json.loads(json.dumps(env.state_dict()))
        expected = [env.step(m, q) for m, q in actions[120:240]]
        restored = JC.Run4BJointChannelEnvironmentV1(self.sources, seed=29)
        restored.load_state_dict(snapshot)
        actual = [restored.step(m, q) for m, q in actions[120:240]]
        self.assertEqual([t.diagnostics for t in actual],
                         [t.diagnostics for t in expected])
        self.assertEqual(env.state_dict(), restored.state_dict())

    def test_pilot_binding_unchanged_and_variants_refuse_each_other(self):
        pilot = R.Run4BRunnerV1(self.pilot_sources, 17, R.PILOT)
        if PILOT_SMOKE_MANIFEST.exists():
            preserved = json.loads(PILOT_SMOKE_MANIFEST.read_text())
            self.assertEqual(pilot.binding_sha256,
                             preserved["runner_binding_sha256"])
        joint = R.Run4BRunnerV1(self.sources, 17, R.JOINT)
        self.assertNotEqual(joint.binding_sha256, pilot.binding_sha256)
        self.assertEqual(joint.binding["variant"], JC.LABEL)
        with self.assertRaises(R.RunnerError):
            R.Run4BRunnerV1(self.pilot_sources, 17, R.JOINT)
        with self.assertRaises(R.RunnerError):
            R.Run4BRunnerV1(self.sources, 17, R.PILOT)
        tmp = Path(tempfile.mkdtemp(prefix="run4b_joint_"))
        try:
            joint.run_preflight(lambda row: None)
            self.assertTrue(joint.preflight["passed"])
            for _ in range(2):
                for _ in range(4):
                    joint.collect_one()
                joint.train_once()
            bundle = R.write_bundle(joint, tmp / "checkpoints")
            restored = R.restore_runner(bundle, self.sources, 17, R.JOINT)
            self.assertEqual(restored.fingerprint(), joint.fingerprint())
            with self.assertRaisesRegex(R.CheckpointRefused,
                                        "foreign checkpoint schema"):
                R.restore_runner(bundle, self.pilot_sources, 17, R.PILOT)
            if PILOT_SMOKE_MANIFEST.exists():
                with self.assertRaisesRegex(R.CheckpointRefused,
                                            "foreign checkpoint schema"):
                    R.restore_runner(PILOT_SMOKE_MANIFEST.parent, self.sources,
                                     17, R.JOINT)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
