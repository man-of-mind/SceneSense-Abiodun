"""CPU-only tests for the prospective two-host contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import unittest

from . import contract as C


class TopologyTests(unittest.TestCase):
    def test_registered_topology(self) -> None:
        t = C.default_topology()
        self.assertEqual(t.local_lan_ip, "10.21.16.222")
        self.assertEqual(t.remote_lan_ip, "10.21.16.162")
        self.assertEqual(t.cn_subnet, "192.168.70.128/26")
        self.assertEqual(
            (t.amf_ip, t.upf_ip, t.ext_dn_ip, t.edge_ip),
            ("192.168.70.132", "192.168.70.134", "192.168.70.135", "192.168.70.140"),
        )
        self.assertEqual((t.map_ip, t.map_port), ("10.21.16.222", 39320))

    def test_overlap_and_role_alias_are_refused(self) -> None:
        with self.assertRaises(C.SplitHostContractError):
            replace(C.default_topology(), remote_lan_ip="192.168.70.162").validate()
        with self.assertRaises(C.SplitHostContractError):
            replace(C.default_topology(), edge_ip="192.168.70.135").validate()


class NetworkPlanTests(unittest.TestCase):
    def test_plan_is_scoped_tagged_and_has_no_nat_or_flush(self) -> None:
        plan = C.network_plan(previous_ip_forward=0, previous_route_argv=None)
        rendered = "\n".join(" ".join(command.argv) for command in (
            plan.probes + (plan.local_apply, plan.local_rollback,
                           plan.forwarding_apply, plan.forwarding_rollback)
            + tuple(rule.check() for rule in plan.rules)
            + tuple(rule.add() for rule in plan.rules)
            + tuple(rule.remove() for rule in plan.rules)
        ))
        self.assertIn("192.168.70.128/26 via 10.21.16.162", rendered)
        self.assertEqual(rendered.count(C.RULE_TAG), 9)
        self.assertNotIn(" -F", rendered)
        self.assertNotIn("--flush", rendered)
        self.assertNotIn(" -P ", rendered)
        self.assertNotIn(" nat ", rendered.lower())
        self.assertNotIn("MASQUERADE", rendered)
        self.assertNotIn("allow-direct-routing", rendered.lower())
        self.assertNotIn("nat-unprotected", rendered.lower())
        self.assertEqual([(rule.table, rule.chain) for rule in plan.rules], [
            ("raw", "PREROUTING"),
            ("filter", "DOCKER-USER"),
            ("filter", "DOCKER-USER"),
        ])
        raw_add = plan.rules[0].add().argv
        self.assertEqual(raw_add[:8], (
            "sudo", "iptables", "-t", "raw", "-I", "PREROUTING", "1", "-s",
        ))
        self.assertIn(("-s", "10.21.16.222/32"), tuple(zip(raw_add, raw_add[1:])))
        self.assertIn(("-d", "192.168.70.128/26"), tuple(zip(raw_add, raw_add[1:])))
        for rule in plan.rules[1:]:
            self.assertEqual(rule.add().argv[:7], (
                "sudo", "iptables", "-t", "filter", "-I", "DOCKER-USER", "1",
            ))
        for rule in plan.rules:
            self.assertIn("10.21.16.222/32", rule.rule_args)

    def test_rules_reject_unregistered_table_or_missing_tag(self) -> None:
        with self.assertRaises(C.SplitHostContractError):
            C.TaggedRule("bad", "nat", "PREROUTING", ("-j", "ACCEPT"))
        with self.assertRaises(C.SplitHostContractError):
            C.TaggedRule("untagged", "raw", "PREROUTING", ("-j", "ACCEPT"))

    def test_forwarding_and_route_restore_measured_prior_state(self) -> None:
        prior = ("ip", "route", "replace", "192.168.70.128/26", "via", "10.21.16.9")
        plan = C.network_plan(previous_ip_forward=1, previous_route_argv=prior)
        self.assertEqual(plan.forwarding_rollback.argv[-1], "net.ipv4.ip_forward=1")
        self.assertEqual(plan.local_rollback.argv, ("sudo",) + prior)

    def test_unmeasured_forwarding_state_is_refused(self) -> None:
        with self.assertRaises(C.SplitHostContractError):
            C.network_plan(previous_ip_forward=2, previous_route_argv=None)


class GnbRewriteTests(unittest.TestCase):
    SOURCE = """
    amf_ip_address = ({ ipv4 = "192.168.70.1"; });
    NETWORK_INTERFACES : {
      GNB_IPV4_ADDRESS_FOR_NG_AMF = "192.168.70.129/24";
      GNB_IPV4_ADDRESS_FOR_NGU = "192.168.70.129/24";
    };
    """

    def test_runtime_rewrite_is_exact_and_idempotent(self) -> None:
        result = C.rewrite_runtime_gnb_config(self.SOURCE)
        self.assertIn('ipv4 = "192.168.70.132"', result)
        self.assertEqual(result.count('"10.21.16.222/24"'), 2)
        self.assertEqual(C.rewrite_runtime_gnb_config(result), result)

    def test_missing_or_duplicate_field_is_refused(self) -> None:
        with self.assertRaises(C.SplitHostContractError):
            C.rewrite_runtime_gnb_config(self.SOURCE.replace(
                'GNB_IPV4_ADDRESS_FOR_NGU = "192.168.70.129/24";', ""))
        with self.assertRaises(C.SplitHostContractError):
            C.rewrite_runtime_gnb_config(self.SOURCE + self.SOURCE)


def valid_binding_dict() -> dict:
    return {
        "schema": "scenesense.run4.remote_runtime_binding.v1",
        "hostname": "L10319",
        "host_ipv4": "10.21.16.162",
        "gpu": {
            "model": "AUDITED_REMOTE_MODEL",
            "uuid": "GPU-audited-uuid",
            "memory_total_mib": 24576,
            "driver_version": "AUDITED_DRIVER",
        },
        "image_tag": C.EDGE_IMAGE_TAG,
        "image_manifest_digest": C.EDGE_IMAGE_MANIFEST_DIGEST,
        "image_config_digest": C.EDGE_IMAGE_CONFIG_DIGEST,
        "remote_image_id": C.REMOTE_IMAGE_ID,
        "remote_container_image_id": C.REMOTE_CONTAINER_IMAGE_ID,
        "canonical_inspect_fields_sha256": C.EDGE_IMAGE_CANONICAL_INSPECT_SHA256,
        "artifacts": [
            {"name": item.name, "relative_path": item.relative_path,
             "sha256": item.sha256}
            for item in C.ARTIFACTS
        ],
    }


class RemoteBindingTests(unittest.TestCase):
    def test_measured_remote_facts_and_exact_runtime_closure_are_accepted(self) -> None:
        binding = C.RemoteRuntimeBinding.from_mapping(valid_binding_dict())
        self.assertEqual(binding.gpu.memory_total_mib, 24576)
        self.assertEqual(len(binding.artifacts), 10)
        self.assertNotIn(C.LOCAL_ACTOR_ARTIFACT, binding.artifacts)
        self.assertEqual(binding.artifacts[-1].name, "compose_fusion_checkpoint")
        self.assertEqual(
            {item.name for item in binding.artifacts[-4:-1]},
            {"person_p025_train_qualification",
             "perception_train_only_priors", "run4_reward_spec"})

    def test_no_gpu_defaults_or_foreign_fields(self) -> None:
        raw = valid_binding_dict()
        del raw["gpu"]["driver_version"]
        with self.assertRaises(C.SplitHostContractError):
            C.RemoteRuntimeBinding.from_mapping(raw)
        raw = valid_binding_dict()
        raw["gpu"]["cuda_guess"] = "12.8"
        with self.assertRaises(C.SplitHostContractError):
            C.RemoteRuntimeBinding.from_mapping(raw)

    def test_manifest_config_and_remote_ids_each_refuse_drift(self) -> None:
        for field in (
            "image_manifest_digest", "image_config_digest", "remote_image_id",
            "remote_container_image_id", "canonical_inspect_fields_sha256",
        ):
            with self.subTest(field=field):
                raw = valid_binding_dict()
                raw[field] = ("0" * 64 if field == "canonical_inspect_fields_sha256"
                              else "sha256:" + "0" * 64)
                with self.assertRaises(C.SplitHostContractError):
                    C.RemoteRuntimeBinding.from_mapping(raw)

    def test_missing_checkpoint_is_refused(self) -> None:
        raw = valid_binding_dict()
        raw["artifacts"].pop()
        with self.assertRaises(C.SplitHostContractError):
            C.RemoteRuntimeBinding.from_mapping(raw)

    def test_schema_is_strict_and_requires_remote_gpu_facts(self) -> None:
        schema_path = Path(__file__).with_name("REMOTE_RUNTIME_BINDING_SCHEMA_V1.json")
        schema = json.loads(schema_path.read_text())
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["hostname"]["const"], "L10319")
        self.assertEqual(set(schema["properties"]["gpu"]["required"]),
                         {"model", "uuid", "memory_total_mib", "driver_version"})
        self.assertEqual(schema["properties"]["artifacts"]["minItems"], 10)
        self.assertEqual(schema["properties"]["artifacts"]["maxItems"], 10)
        self.assertEqual(schema["properties"]["image_manifest_digest"]["const"],
                         C.EDGE_IMAGE_MANIFEST_DIGEST)
        self.assertEqual(schema["properties"]["image_config_digest"]["const"],
                         C.EDGE_IMAGE_CONFIG_DIGEST)
        self.assertEqual(schema["properties"]["remote_image_id"]["const"],
                         C.REMOTE_IMAGE_ID)
        self.assertEqual(schema["properties"]["remote_container_image_id"]["const"],
                         C.REMOTE_CONTAINER_IMAGE_ID)
        self.assertEqual(schema["properties"]["canonical_inspect_fields_sha256"]["const"],
                         C.EDGE_IMAGE_CANONICAL_INSPECT_SHA256)


class ReadinessTests(unittest.TestCase):
    def test_initial_status_stays_blocked(self) -> None:
        report = C.readiness_report(
            remote_binding=None, remote_connectivity_qualified=False,
            gt_lifecycle_seam_implemented=False,
        )
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["live_run_authorized"])
        self.assertEqual(len(report["blockers"]), 3)

    def test_even_structural_ready_does_not_authorize_live_run(self) -> None:
        binding = C.RemoteRuntimeBinding.from_mapping(valid_binding_dict())
        report = C.readiness_report(
            remote_binding=binding, remote_connectivity_qualified=True,
            gt_lifecycle_seam_implemented=True,
        )
        self.assertEqual(report["status"], "READY")
        self.assertFalse(report["live_run_authorized"])


if __name__ == "__main__":
    unittest.main()
