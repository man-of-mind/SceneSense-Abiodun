"""CPU-only acceptance tests for the materialized checkpoint sidecar."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from itertools import chain
from pathlib import Path
from types import MappingProxyType

import torch

from rl_agent.splitfusion_hybrid_sac_checkpoint_sidecar_v1 import sidecar as sc
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import frozen_actor_v2
from rl_agent.splitfusion_hybrid_sac_run4_v1 import checkpoint_io
from rl_agent.splitfusion_hybrid_sac_run4_v1 import models as run4_models
from rl_agent.splitfusion_hybrid_sac_run4_v1 import modeled_smoke_orchestrator as orch
from rl_agent.splitfusion_hybrid_sac_run4_v1.modeled_smoke_orchestrator import (
    _tensor_sha256,
    _tree_sha256,
)

REPO = Path(__file__).resolve().parents[2]
HISTORICAL_SEED43_CHECKPOINTS = (
    REPO / "rl_agent/experiments/splitfusion_hybrid_sac_run4_v2_campaign/"
    "20260928_2a92201_three_seed_10000_v1"
)


def _handle(actor_seed: int, critic_seed: int, rng_base: int, *, lr: float = 3e-4,
            steps: int = 3) -> sc.TrainingStateHandleV1:
    bundle = run4_models.build_run4_models(actor_seed=actor_seed, critic_seed=critic_seed)
    actor_opt = torch.optim.Adam(bundle.actor.parameters(), lr=lr)
    critic_opt = torch.optim.Adam(
        chain(bundle.critics.critic_1.parameters(), bundle.critics.critic_2.parameters()),
        lr=lr,
    )
    generators = {}
    for offset, name in enumerate(sc.GENERATOR_NAMES):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng_base + offset)
        generators[name] = generator
    handle = sc.TrainingStateHandleV1(
        actor=bundle.actor, critics=bundle.critics, actor_optimizer=actor_opt,
        critic_optimizer=critic_opt, generators=MappingProxyType(generators))
    for _ in range(steps):
        _train_step(handle)
    return handle


def _train_step(handle: sc.TrainingStateHandleV1) -> None:
    """Deterministic surrogate update that populates both Adam states."""
    state = torch.rand((4, 21), generator=handle.generators["replay"])
    heads = handle.actor(state)
    loss = heads.logits.square().mean() + heads.mean.square().mean()
    handle.actor_optimizer.zero_grad()
    loss.backward()
    handle.actor_optimizer.step()
    critic_loss = sum(p.square().sum() for p in chain(
        handle.critics.critic_1.parameters(), handle.critics.critic_2.parameters()))
    handle.critic_optimizer.zero_grad()
    critic_loss.backward()
    handle.critic_optimizer.step()
    torch.rand(3, generator=handle.generators["trainer_actor"])


def _binding(handle: sc.TrainingStateHandleV1, *, seed: int = 43,
             update: int = 100) -> sc.EventBindingV1:
    boundary = dict(sc._boundary_view(sc.capture_material(handle)))
    boundary.update(update_count=update, decision_count=688)
    return sc.EventBindingV1(
        seed=seed, seed_plan=MappingProxyType({"master_seed": seed}),
        seed_plan_sha256="1" * 64, update_count=update, decision_count=688,
        event_checkpoint_sha256="2" * 64, boundary=MappingProxyType(boundary))


def _fingerprint(handle: sc.TrainingStateHandleV1) -> dict:
    view = dict(sc._boundary_view(sc.capture_material(handle)))
    view["generators"] = {n: _tensor_sha256(handle.generators[n].get_state())
                          for n in sc.GENERATOR_NAMES}
    return view


class SyntheticSidecarTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = _handle(11, 12, 100)
        self.binding = _binding(self.source)
        self.artifact = sc.write_sidecar(self.root / "update_000100.sidecar",
                                         self.source, self.binding)
        self.directory = Path(self.artifact.directory)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def read(self, **overrides):
        kwargs = dict(expected_seed=43, expected_update_count=100,
                      expected_manifest_sha256=self.artifact.manifest_sha256)
        kwargs.update(overrides)
        return sc.read_sidecar(self.directory, **kwargs)

    def rewrite_manifest(self, mutate) -> str:
        path = self.directory / sc.MANIFEST_FILENAME
        manifest = json.loads(path.read_bytes())
        mutate(manifest)
        data = checkpoint_io._canonical_json_bytes(manifest)
        path.write_bytes(data)
        return checkpoint_io._sha256_bytes(data)

    def test_layout_is_complete_and_manifest_canonical(self) -> None:
        names = sorted(p.name for p in self.directory.iterdir())
        self.assertEqual(names, sorted([sc.MANIFEST_FILENAME,
                                        *sc.ARTIFACT_FILENAMES.values()]))
        manifest = json.loads((self.directory / sc.MANIFEST_FILENAME).read_bytes())
        for name in sc.ARTIFACT_NAMES:
            entry = manifest["artifacts"][name]
            digest, size = checkpoint_io._sha256_file(self.directory / entry["filename"])
            self.assertEqual((digest, size), (entry["sha256"], entry["size_bytes"]))
        self.assertEqual(manifest["identity"]["event_checkpoint_sha256"], "2" * 64)
        self.assertEqual(manifest["schema_identity"], sc.schema_identity())
        online = manifest["artifacts"]["online_critics"]["tensors"]
        target = manifest["artifacts"]["target_critics"]["tensors"]
        self.assertTrue(all(t["path"].startswith(("/critic_1.", "/critic_2."))
                            for t in online))
        self.assertTrue(all(t["path"].startswith(("/target_1.", "/target_2."))
                            for t in target))
        self.assertTrue(manifest["artifacts"]["actor_optimizer"]["tensors"])

    def test_actor_cold_loads_directly_and_matches_source_on_fixtures(self) -> None:
        actor = sc.load_actor_from_sidecar(
            self.directory, expected_seed=43, expected_update_count=100,
            expected_manifest_sha256=self.artifact.manifest_sha256)
        self.source.actor.eval()
        self.assertEqual(frozen_actor_v2.fixture_outputs(actor),
                         frozen_actor_v2.fixture_outputs(self.source.actor))
        for name, value in self.source.actor.state_dict().items():
            self.assertTrue(torch.equal(actor.state_dict()[name], value), name)
        states = torch.tensor(frozen_actor_v2.registered_fixture_states(),
                              dtype=torch.float32)
        with torch.inference_mode():
            got, want = actor(states), self.source.actor(states)
        for field in ("logits", "mean", "log_std"):
            self.assertTrue(torch.equal(getattr(got, field), getattr(want, field)))

    def test_training_state_restores_all_boundary_hashes_and_continues_exactly(self) -> None:
        target = _handle(91, 92, 900, steps=1)
        report = sc.apply_training_state(self.read(), target)
        self.assertEqual(_fingerprint(target), _fingerprint(self.source))
        for field, value in report["boundary"].items():
            self.assertEqual(value, self.binding.boundary[field])
        for _ in range(2):
            _train_step(self.source)
            _train_step(target)
        self.assertEqual(_fingerprint(target), _fingerprint(self.source))

    def test_create_only(self) -> None:
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        with self.assertRaises(sc.SidecarWriteError):
            sc.write_sidecar(self.directory, self.source, self.binding)
        self.assertEqual({p.name: p.read_bytes() for p in self.directory.iterdir()},
                         before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["update_000100.sidecar"])

    def test_write_refuses_state_not_at_boundary(self) -> None:
        _train_step(self.source)
        with self.assertRaises(sc.SidecarWriteError):
            sc.write_sidecar(self.root / "moved.sidecar", self.source, self.binding)
        self.assertFalse((self.root / "moved.sidecar").exists())
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith(".")],
                         [])

    def test_bit_flip_is_tamper(self) -> None:
        path = self.directory / "actor.pt"
        data = bytearray(path.read_bytes())
        data[-40] ^= 0x01
        path.write_bytes(bytes(data))
        with self.assertRaises(sc.SidecarTamperError):
            self.read()

    def test_consistent_forgery_is_refused_by_anchor(self) -> None:
        forged = _handle(31, 32, 300)
        other = self.root / "forged.sidecar"
        sc.write_sidecar(other, forged, _binding(forged))
        for name in os.listdir(self.directory):
            (self.directory / name).write_bytes((other / name).read_bytes())
        with self.assertRaises(sc.SidecarTamperError):
            self.read()

    def test_external_anchor_required(self) -> None:
        with self.assertRaises(sc.SidecarIdentityError):
            sc.read_sidecar(self.directory, expected_seed=43, expected_update_count=100)

    def test_missing_actor_artifact_is_incomplete(self) -> None:
        (self.directory / "actor.pt").unlink()
        with self.assertRaisesRegex(sc.SidecarIncompleteError, "actor-weight"):
            self.read()
        with self.assertRaisesRegex(sc.SidecarIncompleteError, "actor-weight"):
            sc.load_actor_from_sidecar(
                self.directory, expected_seed=43, expected_update_count=100,
                expected_manifest_sha256=self.artifact.manifest_sha256)

    def test_extra_and_symlinked_files_refused(self) -> None:
        (self.directory / "extra.bin").write_bytes(b"x")
        with self.assertRaises(sc.SidecarTamperError):
            self.read()
        (self.directory / "extra.bin").unlink()
        real = self.root / "actor_real.pt"
        (self.directory / "actor.pt").rename(real)
        (self.directory / "actor.pt").symlink_to(real)
        with self.assertRaises(sc.SidecarReadError):
            self.read()

    def test_foreign_seed_update_and_schema_refused(self) -> None:
        with self.assertRaises(sc.SidecarIdentityError):
            self.read(expected_seed=44)
        with self.assertRaises(sc.SidecarIdentityError):
            self.read(expected_update_count=250)
        digest = self.rewrite_manifest(
            lambda m: m["schema_identity"].__setitem__("feature_schema_sha256", "0" * 64))
        with self.assertRaises(sc.SidecarIdentityError):
            self.read(expected_manifest_sha256=digest)
        digest = self.rewrite_manifest(lambda m: m.__setitem__("schema_version", 2))
        with self.assertRaises(sc.SidecarIdentityError):
            self.read(expected_manifest_sha256=digest)

    def test_shape_mismatch_refused_without_mutation(self) -> None:
        loaded = self.read()
        material = dict(loaded.material)
        actor_state = dict(material["actor"])
        name = sorted(actor_state)[0]
        actor_state[name] = torch.zeros(tuple(actor_state[name].shape) + (1,))
        material["actor"] = actor_state
        bad = replace(loaded, material=MappingProxyType(material))
        target = _handle(91, 92, 900, steps=1)
        before = _fingerprint(target)
        with self.assertRaises(sc.SidecarReadError):
            sc.apply_training_state(bad, target)
        self.assertEqual(_fingerprint(target), before)

    def test_foreign_optimizer_hyperparameters_refused_without_mutation(self) -> None:
        target = _handle(91, 92, 900, lr=1e-3, steps=1)
        before = _fingerprint(target)
        with self.assertRaisesRegex(sc.SidecarReadError, "hyperparameters"):
            sc.apply_training_state(self.read(), target)
        self.assertEqual(_fingerprint(target), before)

    def test_failed_post_load_verification_rolls_back(self) -> None:
        loaded = self.read()
        manifest = json.loads(json.dumps(dict(loaded.manifest)))
        manifest["identity"]["boundary"]["critic_optimizer_sha256"] = "3" * 64
        bad = replace(loaded, manifest=MappingProxyType(manifest))
        target = _handle(91, 92, 900, steps=1)
        before = _fingerprint(target)
        with self.assertRaises(sc.SidecarTamperError):
            sc.apply_training_state(bad, target)
        self.assertEqual(_fingerprint(target), before)

    def test_registered_checkpoint_without_actor_artifact_fails(self) -> None:
        checkpoints = self.root / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "update_000100.checkpoint.json").write_bytes(b"{}")
        with self.assertRaisesRegex(sc.SidecarIncompleteError, r"actor-weight.*\[100\]"):
            sc.require_materialized_checkpoints(checkpoints, self.root / "none",
                                                expected_seed=43)

    def test_historical_run4_checkpoints_are_event_sourced_only(self) -> None:
        """Run-4 checkpoints carry hashes, not weights: the gate must fail.

        Only directory listings are read; nothing is parsed or replayed.
        """
        seed_dirs = sorted(HISTORICAL_SEED43_CHECKPOINTS.glob("*43*"))
        if not seed_dirs:
            self.skipTest("historical campaign evidence not present")
        checkpoints = seed_dirs[0] / "checkpoints"
        self.assertTrue(any(checkpoints.glob("update_*.checkpoint.json")))
        with self.assertRaisesRegex(sc.SidecarIncompleteError, "actor-weight"):
            sc.require_materialized_checkpoints(checkpoints, checkpoints,
                                                expected_seed=43)

    def test_import_performs_no_io(self) -> None:
        code = (
            "import sys\n"
            "import rl_agent.splitfusion_hybrid_sac_live_route_b_v2.frozen_actor_v2\n"
            "import rl_agent.splitfusion_hybrid_sac_run4_v1.checkpoint_io\n"
            "events = []\n"
            "def hook(event, args):\n"
            "    if event in ('open', 'os.listdir', 'os.scandir', 'subprocess.Popen',\n"
            "                 'socket.connect', 'os.mkdir', 'os.rename'):\n"
            "        path = str(args[0]) if args else ''\n"
            "        if not path.endswith(('.py', '.pyc', '.so')) and '__pycache__' not in path\\\n"
            "                and 'site-packages' not in path and 'dist-packages' not in path\\\n"
            "                and not path.endswith(('rl_agent', 'splitfusion_hybrid_sac_checkpoint_sidecar_v1')):\n"
            "            events.append((event, path))\n"
            "sys.addaudithook(hook)\n"
            "import rl_agent.splitfusion_hybrid_sac_checkpoint_sidecar_v1.sidecar\n"
            "print(repr(events))\n"
        )
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["CUDA_VISIBLE_DEVICES"] = ""
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "[]", out.stdout + out.stderr)


class OrchestratorSidecarIntegrationTest(unittest.TestCase):
    """Real Run-4 orchestrator + callback on the synthetic fake collector.

    Runs 0 -> 100 (~1 min) and one event-sourced restore of that synthetic
    run; no historical checkpoint is replayed.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from rl_agent.splitfusion_hybrid_sac_run4_v1 import (
            test_modeled_smoke_orchestrator as fixture_module,
        )
        fixture = fixture_module.ModeledSmokeOrchestratorTest
        fixture.setUpClass()
        cls.fixture_class = fixture
        cls.fixture = fixture("test_resume_from_250_is_bit_identical_at_500")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        (cls.root / "checkpoints").mkdir()
        (cls.root / "sidecars").mkdir()
        cls.source = cls.fixture.orchestrator()
        cls.events = {}

        def event_writer(event):
            cls.events[event.update] = event.checkpoint
            orch.write_checkpoint(
                cls.root / "checkpoints" / f"update_{event.update:06d}.checkpoint.json",
                event.checkpoint)

        cls.callback = sc.SidecarCheckpointCallbackV1(
            cls.source, cls.root / "sidecars", chain=event_writer)
        cls.source.run_to_registered_update(100, checkpoint_callback=cls.callback,
                                            emit_current_checkpoint=True)
        cls.seed = cls.fixture.factory.seed_plan.master_seed

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()
        cls.fixture_class.tearDownClass()

    def sidecar(self, update: int) -> Path:
        return self.root / "sidecars" / sc.sidecar_directory_name(update)

    def test_callback_materialized_every_emitted_boundary(self) -> None:
        self.assertEqual(sorted(self.callback.artifacts), [0, 100])
        self.assertEqual(sorted(self.events), [0, 100])
        for update in (0, 100):
            self.assertEqual(self.callback.artifacts[update].event_checkpoint_sha256,
                             self.events[update].canonical_sha256)

    def test_actor_cold_load_equals_live_actor(self) -> None:
        actor = sc.load_actor_from_sidecar(
            self.sidecar(100), expected_seed=self.seed, expected_update_count=100,
            event_checkpoint=self.events[100])
        live = self.source.runner.model_bundle.actor
        self.assertEqual(_tree_sha256(actor.state_dict()),
                         self.events[100].boundary.actor_sha256)
        was_training = live.training
        live.eval()
        try:
            self.assertEqual(frozen_actor_v2.fixture_outputs(actor),
                             frozen_actor_v2.fixture_outputs(live))
        finally:
            live.train(was_training)

    def test_training_state_matches_boundary_and_event_restore(self) -> None:
        fresh = self.fixture.orchestrator()
        handle = sc.training_state_from_orchestrator(fresh)
        loaded = sc.read_sidecar(self.sidecar(100), expected_seed=self.seed,
                                 expected_update_count=100,
                                 event_checkpoint=self.events[100])
        report = sc.apply_training_state(loaded, handle)
        boundary = self.events[100].boundary.to_dict()
        for field, value in report["boundary"].items():
            self.assertEqual(value, boundary[field], field)
        restored = orch.ModeledSmokeOrchestratorV1.restore(
            self.events[100], runner_factory=self.fixture.factory,
            collector_factory=self.fixture.collector_factory,
            preflight_variation_contract=self.fixture.variation_contract)
        replayed = sc.training_state_from_orchestrator(restored)
        self.assertEqual(_fingerprint(handle), _fingerprint(replayed))
        live = sc.training_state_from_orchestrator(self.source)
        self.assertEqual(_fingerprint(handle), _fingerprint(live))

    def test_registered_gate_passes_then_fails_without_actor(self) -> None:
        verified = sc.require_materialized_checkpoints(
            self.root / "checkpoints", self.root / "sidecars", expected_seed=self.seed)
        self.assertEqual(sorted(verified), [0, 100])
        copy_root = self.root / "sidecars_missing_actor"
        copy_root.mkdir()
        for update in (0, 100):
            target = copy_root / sc.sidecar_directory_name(update)
            target.mkdir()
            for item in self.sidecar(update).iterdir():
                if update == 100 and item.name == "actor.pt":
                    continue
                (target / item.name).write_bytes(item.read_bytes())
        with self.assertRaisesRegex(sc.SidecarIncompleteError, r"actor-weight.*\[100\]"):
            sc.require_materialized_checkpoints(
                self.root / "checkpoints", copy_root, expected_seed=self.seed)

    def test_foreign_seed_and_event_checkpoint_refused(self) -> None:
        with self.assertRaises(sc.SidecarIdentityError):
            sc.read_sidecar(self.sidecar(100), expected_seed=self.seed + 1,
                            expected_update_count=100, event_checkpoint=self.events[100])
        with self.assertRaises(sc.SidecarIdentityError):
            sc.read_sidecar(self.sidecar(100), expected_seed=self.seed,
                            expected_update_count=100, event_checkpoint=self.events[0])


if __name__ == "__main__":
    unittest.main()
