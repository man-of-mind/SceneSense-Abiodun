"""CPU-only actor provenance, batch-one parity, and quality-aware state tests."""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import ue_telemetry_provider_v2 as T
from rl_agent.splitfusion_hybrid_sac_live_route_b_v2.test_ue_telemetry_provider_v2 import Harness, _valid_tick
from rl_agent.splitfusion_hybrid_sac_v1.hybrid_sac_models import build_actor
from rl_agent.splitfusion_hybrid_sac_v1.transaction_identity import ExecutedActionIdentity
from rl_agent.splitfusion_run5_live_avo_v1 import actor_binding as A


ROOT = Path(__file__).resolve().parents[2]
EXPORT = Path(os.environ.get("RUN5_LIVE_ACTOR_EXPORT", str(
    ROOT.parent / "abiodun_run5_wt/rl_agent/splitfusion_hybrid_sac_run5_v1/campaign_runs/"
    "run5_three_seed_10000_v1/seed_43/final_actor_010000")))


def state_fixture():
    harness = Harness()
    harness.dci(mcs=24)
    _valid_tick(harness, 3, 1, values=(0, 0, 0))
    _valid_tick(harness, 3, 2, values=(0, 0, 0))
    identity = harness.identity()
    snapshot, boundary, _ = T.open_decision(harness.provider, identity,
                                           clock=harness._ticking_clock())
    evidence = T.assemble_radio_evidence(snapshot, identity=identity, boundary=boundary,
        payload_enqueue_timestamp_ns=boundary.action_open_timestamp_ns + 1)
    camera, radar = A.LS.scene_observations(
        identity, sample_seq=7, camera_si=112.0, radar_p40=0.4, source="unit-test-scene",
        source_timestamp_ns=boundary.state_commit_timestamp_ns - 20_000_000,
        available_timestamp_ns=boundary.state_commit_timestamp_ns - 10_000_000,
        clock_domain=T.CLOCK_DOMAIN)
    kwargs = dict(identity=identity, boundary=boundary, camera_si=camera, radar_p40=radar,
        prior_ul_mcs=evidence.prior_ul_mcs, pre_action_rlc_backlog=evidence.pre_action_rlc_backlog,
        previous=None)
    snr = A.S5.UlSnrLeaseObservationV1(
        identity=A.R4.SampleIdentityV1(identity.session_uuid, identity.ue_id, 0),
        kind=A.C5.SnrProxyKind.SIMULATOR_EFFECTIVE_UL_SNR_PROXY_DB, provider_id="test",
        controller_session_uuid=identity.session_uuid, value_db=15.0,
        active_command_id="c1", effective_since_ns=boundary.state_commit_timestamp_ns - 50_000_000,
        heartbeat_command_id="c1", heartbeat_ns=boundary.state_commit_timestamp_ns - 10_000_000,
        clock_domain=T.CLOCK_DOMAIN, valid=True, missing_reason=None)
    return kwargs, snr


class StateBindingTests(unittest.TestCase):
    def test_frozen_dependencies_and_native_schema_match(self):
        self.assertEqual(len(A.verify_frozen_sources()), 2)
        self.assertEqual(A.canonical_sha256(A.native_model_binding()), A.NATIVE_MODEL_BINDING_SHA256)
        self.assertEqual(A.training_lease_policy().max_heartbeat_age_ns, 200_000_000)

    def test_exact_prefix_and_original_boundary(self):
        kwargs, snr = state_fixture()
        old, old_features = A._RUN4_BUILD_LIVE_STATE(**kwargs)
        guarded, features = A.build_run5_live_state(snr=snr, **kwargs)
        self.assertEqual(guarded, old)
        self.assertEqual(guarded.boundary, kwargs["boundary"])
        self.assertEqual(features.as_tuple()[:21], old_features.as_tuple())
        self.assertEqual(features.as_tuple()[21], 0.5)
        self.assertEqual(len(features.as_tuple()), 22)

    def test_missing_stale_future_foreign_and_out_of_support_refuse(self):
        kwargs, snr = state_fixture()
        boundary = kwargs["boundary"]
        cases = [None, dataclasses.replace(snr, value_db=24.6),
                 dataclasses.replace(snr, heartbeat_command_id="foreign"),
                 dataclasses.replace(snr, clock_domain="CLOCK_MONOTONIC"),
                 dataclasses.replace(snr, controller_session_uuid="00000000-0000-0000-0000-000000000001"),
                 dataclasses.replace(snr, heartbeat_ns=boundary.state_commit_timestamp_ns + 1),
                 dataclasses.replace(snr,
                    effective_since_ns=boundary.action_open_timestamp_ns - 400_000_000,
                    heartbeat_ns=boundary.action_open_timestamp_ns - 200_000_001)]
        for invalid in cases:
            with self.subTest(invalid=invalid):
                with self.assertRaises(A.R4.ExternalFallbackRequired):
                    A.build_run5_live_state(snr=invalid, **kwargs)

    def test_inclusive_lease_limit(self):
        kwargs, snr = state_fixture()
        at = kwargs["boundary"].action_open_timestamp_ns - 200_000_000
        observation = dataclasses.replace(snr, effective_since_ns=at - 1, heartbeat_ns=at)
        A.build_run5_live_state(snr=observation, **kwargs)

    def test_previous_quality_is_retained_in_state_and_failures_keep_original_encoding(self):
        kwargs, snr = state_fixture()
        old_identity = kwargs["identity"]
        identity = dataclasses.replace(old_identity, decision_seq=1)
        kwargs["identity"] = identity
        kwargs["boundary"] = dataclasses.replace(kwargs["boundary"], identity=identity)
        catalog = A.AC.load_contract()
        action = ExecutedActionIdentity.from_executable_action(catalog.resolve(11, 0.3), catalog)
        vectors = []
        for terminal, quality, latency in ((A.R4.RewardTerminal.SUCCESS, 0.2, 150.0),
                                            (A.R4.RewardTerminal.SUCCESS, 0.8, 150.0),
                                            (A.R4.RewardTerminal.TIMEOUT, None, None)):
            previous = A.R4.PreviousOutcomeV1(old_identity, action, terminal, quality, latency,
                kwargs["boundary"].state_commit_timestamp_ns - 1_000_000,
                T.CLOCK_DOMAIN, "a" * 64)
            kwargs["previous"] = previous
            _, expected = A._RUN4_BUILD_LIVE_STATE(**kwargs)
            _, actual = A.build_run5_live_state(snr=snr, **kwargs)
            self.assertEqual(actual.as_tuple()[:21], expected.as_tuple())
            vectors.append(actual.as_tuple())
        self.assertEqual(vectors[0][17], 0.2)
        self.assertEqual(vectors[1][17], 0.8)
        self.assertEqual([i for i, (a, b) in enumerate(zip(*vectors[:2])) if a != b], [17])
        self.assertEqual(vectors[2][17:21], (0.0, 0.0, 1.0, 0.0))

    def test_raw_support_refusal_is_not_hidden_by_backlog_scaling(self):
        kwargs, snr = state_fixture()
        old = kwargs["pre_action_rlc_backlog"]
        kwargs["pre_action_rlc_backlog"] = dataclasses.replace(old, value=50_000_000)
        with self.assertRaises(A.R4.ExternalFallbackRequired):
            A.build_run5_live_state(snr=snr, **kwargs)

    def test_live_builder_explicitly_adopts_then_observes(self):
        kwargs, snr = state_fixture()
        calls = []
        class Provider:
            def adopt_session(self, session):
                calls.append(("adopt", session))
            def observe(self, boundary):
                calls.append(("observe", boundary))
                return snr
        builder = A.LiveStateBuilder(Provider())
        _guard, vector = builder.build(**kwargs)
        self.assertEqual(calls, [("adopt", kwargs["identity"].session_uuid),
                                 ("observe", kwargs["boundary"])])
        self.assertIs(builder.last_observation, snr)
        self.assertEqual(len(vector.as_tuple()), 22)

    def test_state_seam_cannot_recurse_when_phase6_function_is_rebound(self):
        kwargs, snr = state_fixture()
        with patch.object(A.LS, "build_live_state", side_effect=AssertionError("rebound seam")):
            _guard, vector = A.build_run5_live_state(snr=snr, **kwargs)
        self.assertEqual(len(vector.as_tuple()), 22)


@unittest.skipUnless(EXPORT.is_dir(), "explicit original Run-5 actor export required")
class ActorBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actor = A.load_source_export(EXPORT)

    def test_loader_is_cpu_only_eval_weights_only_and_rng_neutral(self):
        before = A.F4.rng_fingerprint()
        cuda_before = torch.cuda.is_initialized()
        loaded = A.load_source_export(EXPORT)
        self.assertEqual(before, A.F4.rng_fingerprint())
        self.assertEqual(cuda_before, torch.cuda.is_initialized())
        self.assertFalse(loaded.module.training)
        self.assertTrue(all(not p.requires_grad and p.device.type == "cpu" for p in loaded.module.parameters()))
        self.assertEqual(loaded.boundary_sha256, A.ACTOR_TREE_SHA256)

    def test_batch_one_parity_with_native_network(self):
        reference = build_actor(A.model_config(), seed=99)
        reference.load_state_dict(torch.load(EXPORT / "actor_state_dict.pt", map_location="cpu", weights_only=True))
        reference.eval()
        for row in A.registered_fixture_states():
            actual = self.actor.act_on_vector(row)
            with torch.inference_mode():
                output = reference.deterministic_execution(torch.tensor((row,), dtype=torch.float32))
            self.assertEqual((actual.mode_id, actual.q_e4),
                             (int(output.mode_index[0]), int(output.q_e4[0])))

    def test_actor_accepts_only_attested_run5_type(self):
        kwargs, snr = state_fixture()
        _, vector = A.build_run5_live_state(snr=snr, **kwargs)
        decision = self.actor.act(vector)
        self.assertEqual(decision, self.actor.act_on_vector(vector.as_tuple()))
        _, old_vector = A._RUN4_BUILD_LIVE_STATE(**kwargs)
        with self.assertRaises(A.Run5LiveBindingError):
            self.actor.act(old_vector)
        forged = A.S5.Run5FeatureVectorV2(vector.values, vector.run4_prefix_sha256, vector.guarded_state_sha256)
        with self.assertRaises(A.R4.ScalingError):
            self.actor.act(forged)

    def test_vectors_refuse_old_width_nonfinite_and_invalid_snr(self):
        row = A.registered_fixture_states()[0]
        for values in (row[:21], list(row), row[:21] + (float("nan"),),
                       row[:21] + (1.0001,), row[:21] + (-0.1,), row[:21] + (True,)):
            with self.assertRaises(A.Run5LiveBindingError):
                self.actor.act_on_vector(values)

    def test_binding_is_repeatable_and_explicitly_new_live_choice(self):
        binding = A.binding_document(self.actor)
        self.assertEqual(binding, A.binding_document(self.actor))
        self.assertEqual(len(binding["batch1_fixtures"]), 20)
        self.assertEqual(binding["model"]["policy_feature_order"][17], "prev_quality_qperc")
        self.assertIn("NEW_LIVE_EVALUATION_CHOICE", binding["selection_scope"])

    def test_manifest_or_weights_tampering_cannot_relabel_foreign_actor(self):
        with tempfile.TemporaryDirectory(prefix="run5_actor_binding_") as tmp:
            copied = Path(tmp)
            for filename in ("manifest.json", "actor_state_dict.pt"):
                shutil.copyfile(EXPORT / filename, copied / filename)
            manifest = json.loads((copied / "manifest.json").read_text())
            manifest["feature_schema_sha256"] = A.R4.FEATURE_SCHEMA_SHA256
            (copied / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(A.Run5LiveBindingError):
                A.load_source_export(copied)
            shutil.copyfile(EXPORT / "manifest.json", copied / "manifest.json")
            state = torch.load(copied / "actor_state_dict.pt", weights_only=True)
            state["encoder.0.weight"][0, 0] += 1
            torch.save(state, copied / "actor_state_dict.pt")
            with self.assertRaises(A.Run5LiveBindingError):
                A.load_source_export(copied)


if __name__ == "__main__":
    unittest.main()
