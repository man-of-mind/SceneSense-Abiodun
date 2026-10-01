from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rl_agent.splitfusion_run4b5b_live_isolation_v1 import live_adapters_v1 as L
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_validation_runner_v1 as R


H = "a" * 64


class FakeLifecycle:
    def __init__(self, *, fail=False):
        self.events = []
        self.fail = fail

    def preflight(self, config, manifest):
        self.events.append(("preflight", config.variant, manifest.variant))

    def start(self, config, manifest):
        self.events.append(("start", config.transmitted_budget))

    def execute(self, config, manifest):
        self.events.append(("execute", config.ack_semantics))
        if self.fail:
            raise RuntimeError("synthetic execute failure")
        return R.LifecycleExecutionV1(
            300, "COMPLETE", hashlib.sha256(b"fake result").hexdigest())

    def stop(self, config):
        self.events.append(("stop", config.run_id))


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.manifest_path = root / "actor.json"
        order = L.RUN4B_FEATURE_ORDER
        manifest = L.BActorManifestV1(
            variant=L.ActorVariant.RUN4B, feature_order=order,
            feature_count=len(order),
            feature_schema_sha256=L.feature_schema_sha256(
                L.ActorVariant.RUN4B, order),
            actor_boundary_sha256="b" * 64,
            weights_file_sha256="c" * 64,
            selected_seed=43, selected_update=10_000,
        )
        self.manifest_path.write_text(
            json.dumps({"schema": L.ACTOR_MANIFEST_SCHEMA,
                        **manifest.as_dict()}), encoding="utf-8")
        # as_dict already carries schema; the duplicate assignment above has
        # the same value and makes the test fixture's authority explicit.
        digest = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        self.raw = {
            "schema": R.CONFIG_SCHEMA,
            "run_id": "run4b_validation_300",
            "variant": L.ActorVariant.RUN4B.value,
            "actor_manifest_path": str(self.manifest_path),
            "actor_manifest_sha256": digest,
            "output_root": str(root / "output"),
            "evidence_root": str(root / "evidence"),
            "transmitted_budget": 300,
            "split_host": {
                "carla_host": "W10275.idcc.lab",
                "ue_host": "W10275.idcc.lab",
                "cn_host": "L10319.idcc.lab",
                "edge_host": "L10319.idcc.lab",
                "ext_dn_host": "L10319.idcc.lab",
                "ack_receiver_host": "W10275.idcc.lab",
                "ack_receiver_port": 41070,
            },
            "runner_semantics": R.RUNNER_SEMANTICS,
            "ack_semantics": R.ACK_SEMANTICS,
            "postrun_semantics": R.POSTRUN_SEMANTICS,
            "clock_domain": R.CLOCK_DOMAIN,
            "deadline_ns": R.DEADLINE_NS,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def config(self):
        return R.BValidationConfigV1.from_mapping(self.raw)

    def test_composed_offline_run_uses_selected_variant_budget_and_stops(self):
        fake = FakeLifecycle()
        result = R.run_with_lifecycle(self.config(), fake)
        self.assertEqual(result.transmitted_frames, 300)
        self.assertEqual(result.terminal_status, "COMPLETE")
        self.assertEqual([row[0] for row in fake.events],
                         ["preflight", "start", "execute", "stop"])
        self.assertEqual(fake.events[0][1:],
                         (L.ActorVariant.RUN4B, L.ActorVariant.RUN4B))

    def test_stop_runs_after_execute_failure(self):
        fake = FakeLifecycle(fail=True)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            R.run_with_lifecycle(self.config(), fake)
        self.assertEqual([row[0] for row in fake.events][-1], "stop")

    def test_seal_round_trip_and_tamper_refusal(self):
        path = Path(self.tmp.name) / "sealed.json"
        sealed = R.seal_mapping(self.config())
        path.write_text(json.dumps(sealed), encoding="utf-8")
        self.assertEqual(R.load_sealed_config(path), self.config())
        sealed["config"]["transmitted_budget"] = 299
        path.write_text(json.dumps(sealed), encoding="utf-8")
        with self.assertRaises(R.ValidationConfigError):
            R.load_sealed_config(path)

    def test_old_phase6_gt_quality_semantics_are_refused(self):
        for key, value in (
            ("runner_semantics", "PHASE6_LIVE_QUALITY_RUNNER"),
            ("ack_semantics", "QUALITY_ACK_AFTER_GT_FEEDBACK"),
            ("postrun_semantics", "LIVE_QPERC_REWARD_TICKET"),
        ):
            with self.subTest(key=key):
                raw = dict(self.raw)
                raw[key] = value
                with self.assertRaises(R.ValidationConfigError):
                    R.BValidationConfigV1.from_mapping(raw)

    def test_host_roles_must_be_isolated(self):
        raw = dict(self.raw)
        raw["split_host"] = dict(self.raw["split_host"])
        raw["split_host"]["edge_host"] = "W10275.idcc.lab"
        with self.assertRaises(R.ValidationConfigError):
            R.BValidationConfigV1.from_mapping(raw)

    def test_manifest_variant_and_digest_are_enforced(self):
        raw = dict(self.raw)
        raw["variant"] = L.ActorVariant.RUN5B.value
        config = R.BValidationConfigV1.from_mapping(raw)
        with self.assertRaises(R.ValidationConfigError):
            R.load_actor_manifest(config)
        raw = dict(self.raw)
        raw["actor_manifest_sha256"] = H
        config = R.BValidationConfigV1.from_mapping(raw)
        with self.assertRaises(R.ValidationConfigError):
            R.load_actor_manifest(config)

    def test_budget_deadline_and_foreign_fields_refused(self):
        for key, value in (("transmitted_budget", 301),
                           ("deadline_ns", R.DEADLINE_NS + 1)):
            with self.subTest(key=key):
                raw = dict(self.raw)
                raw[key] = value
                with self.assertRaises(R.ValidationConfigError):
                    R.BValidationConfigV1.from_mapping(raw)
        raw = dict(self.raw)
        raw["quality_ack_endpoint"] = "forbidden"
        with self.assertRaises(R.ValidationConfigError):
            R.BValidationConfigV1.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
