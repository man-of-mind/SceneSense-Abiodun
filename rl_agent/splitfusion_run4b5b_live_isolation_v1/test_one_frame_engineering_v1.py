"""CPU-only tests for the isolated one-frame engineering gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import b_validation_runner_v1 as Q
from . import final_actor_gate_v2 as F
from . import one_frame_engineering_v1 as O


class _Identity:
    actor_state_dict_sha256 = "a" * 64


class _Actor:
    identity = _Identity()


class _FakeLifecycle:
    def __init__(self, *, fail: str | None = None,
                 late: bool = False) -> None:
        self.events: list[str] = []
        self.fail, self.late = fail, late

    def preflight(self, config, actor):
        self.events.append("preflight")
        if self.fail == "preflight": raise RuntimeError("preflight")

    def start(self, config, actor):
        self.events.append("start")
        config.local_attempt_root.mkdir(parents=True)
        if self.fail == "start": raise RuntimeError("start")

    def execute(self, config, actor):
        self.events.append("execute")
        if self.fail == "execute": raise RuntimeError("execute")
        return O.OneFrameExecutionV1(
            run_id=config.run_id, variant=config.variant,
            transmitted_frames=1, policy_decisions=1,
            operational_successes=1, operational_timeouts=0,
            observed_latency_ns=(170_000_001 if self.late else 170_000_000),
            ack_before_map_offer=True, prediction_evidence_written=True,
            live_qperc_computed=False, exact_identity_sha256="b" * 64,
            result_sha256="c" * 64)

    def stop(self, config): self.events.append("stop")


class OneFrameEngineeringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manifest = root / "RUN4B_JOINT_FINAL_ACTOR_MANIFEST_V2.json"
        self.manifest.write_text("{}", encoding="ascii")
        self.weights = root / "actor_state_dict.pt"
        self.weights.write_bytes(b"weights")
        self.config = O.OneFrameConfigV1(
            run_id="engineering_1", cell_id="route_b_one_frame",
            variant=F.RUN4B_VARIANT,
            actor_manifest_path=self.manifest,
            actor_manifest_sha256=hashlib.sha256(b"{}").hexdigest(),
            actor_weights_path=self.weights,
            actor_weights_sha256=hashlib.sha256(b"weights").hexdigest(),
            actor_evidence_root=root / "evidence",
            local_repository=root / "local_repo",
            remote_repository=Path("/srv/abiodun_remote"),
            local_attempt_root=root / "attempt",
            remote_attempt_root=Path("/srv/attempts/engineering_1"),
            edge_campaign_config=Path("/srv/config/campaign.json"),
            route_config=root / "route.json",
            network=O.NetworkBindingV1(
                local_host=O.LOCAL_HOST, remote_host=O.REMOTE_HOST,
                remote_ssh=O.REMOTE_SSH, local_lan_ip=O.LOCAL_LAN_IP,
                remote_lan_ip=O.REMOTE_LAN_IP, cn_subnet=O.CN_SUBNET,
                edge_ip=O.EDGE_IP, ext_dn_ip=O.EXT_DN_IP,
                ue_tunnel_ip=O.UE_TUNNEL_IP,
                ue_tunnel_interface=O.UE_TUNNEL_INTERFACE,
                ue_policy_table=O.UE_POLICY_TABLE, edge_route=O.EDGE_ROUTE,
                edge_feature_port=O.EDGE_FEATURE_PORT, ack_port=O.ACK_PORT,
                direct_map_host=O.LOCAL_LAN_IP,
                direct_map_port=O.DIRECT_MAP_PORT,
                carla_rpc_host="127.0.0.1", carla_rpc_port=O.CARLA_RPC_PORT),
            transmitted_budget=1, policy_decision_budget=1,
            deadline_ns=O.DEADLINE_NS, clock_domain=O.CLOCK_DOMAIN,
            ack_semantics=O.ACK_SEMANTICS,
            postrun_semantics=O.POSTRUN_SEMANTICS,
            purpose=O.PURPOSE, policy_performance_claim=False,
            factory_module=O.FACTORY_MODULE)

    def tearDown(self) -> None: self.temp.cleanup()

    def test_seal_roundtrip_and_exact_explicit_network(self) -> None:
        path = Path(self.temp.name) / "sealed.json"
        path.write_text(json.dumps(O.seal(self.config)), encoding="ascii")
        loaded = O.load_config(path)
        self.assertEqual(loaded.binding_sha256(), self.config.binding_sha256())
        self.assertEqual(loaded.network.ue_tunnel_ip, "10.0.0.2")
        self.assertEqual(loaded.network.edge_ip, "192.168.70.140")
        self.assertEqual(loaded.network.ack_port, 51014)

    def test_one_frame_schema_cannot_weaken_300_frame_qualification(self) -> None:
        self.assertEqual(self.config.transmitted_budget, 1)
        with self.assertRaises(Q.ValidationConfigError):
            Q.BValidationConfigV1(
                run_id="x", variant=Q.ActorVariant.RUN4B,
                actor_manifest_path=str(self.manifest),
                actor_manifest_sha256="a" * 64,
                output_root=str(Path(self.temp.name) / "o"),
                evidence_root=str(Path(self.temp.name) / "e"),
                transmitted_budget=1,
                split_host=Q.SplitHostBindingV1(
                    carla_host=Q.__name__, ue_host=Q.__name__,
                    cn_host="r", edge_host="r", ext_dn_host="r",
                    ack_receiver_host=Q.__name__, ack_receiver_port=51014),
                runner_semantics=Q.RUNNER_SEMANTICS,
                ack_semantics=Q.ACK_SEMANTICS,
                postrun_semantics=Q.POSTRUN_SEMANTICS,
                clock_domain=Q.CLOCK_DOMAIN, deadline_ns=Q.DEADLINE_NS)
        self.assertEqual(Q.TRANSMITTED_BUDGET, 300)

    def test_lifecycle_order_and_inclusive_deadline(self) -> None:
        lifecycle = _FakeLifecycle()
        with mock.patch.object(O, "_load_selected_actor", return_value=_Actor()):
            value = O.run(self.config, lifecycle)
        self.assertEqual(value.observed_latency_ns, O.DEADLINE_NS)
        self.assertEqual(lifecycle.events,
                         ["preflight", "start", "execute", "stop"])

    def test_failure_after_start_always_stops(self) -> None:
        lifecycle = _FakeLifecycle(fail="execute")
        with mock.patch.object(O, "_load_selected_actor", return_value=_Actor()):
            with self.assertRaisesRegex(RuntimeError, "execute"):
                O.run(self.config, lifecycle)
        self.assertEqual(lifecycle.events,
                         ["preflight", "start", "execute", "stop"])

    def test_late_ack_and_live_qperc_are_refused(self) -> None:
        lifecycle = _FakeLifecycle(late=True)
        with mock.patch.object(O, "_load_selected_actor", return_value=_Actor()):
            with self.assertRaisesRegex(O.OneFrameEngineeringError,
                                       "exceeded"):
                O.run(self.config, lifecycle)
        self.assertEqual(lifecycle.events[-1], "stop")
        record = lifecycle.execute(self.config, _Actor())
        changed = O.OneFrameExecutionV1(
            **{**{name: getattr(record, name)
                  for name in record.__dataclass_fields__},
               "observed_latency_ns": 1, "live_qperc_computed": True})
        with self.assertRaisesRegex(O.OneFrameEngineeringError, "Qperc"):
            changed.validate(self.config)

    def test_existing_attempt_refused_before_actor_or_lifecycle(self) -> None:
        self.config.local_attempt_root.mkdir()
        lifecycle = _FakeLifecycle()
        with mock.patch.object(O, "_load_selected_actor") as actor:
            with self.assertRaisesRegex(O.OneFrameEngineeringError,
                                       "create-only"):
                O.preflight(self.config, lifecycle)
        actor.assert_not_called()
        self.assertEqual(lifecycle.events, [])

    def test_factory_absence_is_explicit(self) -> None:
        with mock.patch.object(O.importlib, "import_module",
                               side_effect=ImportError("absent")):
            with self.assertRaises(O.ProductionFactoryUnavailable):
                O.load_production_lifecycle(self.config)


if __name__ == "__main__": unittest.main()
