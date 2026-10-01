"""CPU-only tests for the production one-frame B factory."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import (
    continuous_execution_v2 as X,
)

from . import b_edge_process_v1 as EP
from . import b_one_frame_execution_v1 as ONE
from . import b_opportunity_processor_v1 as OPP
from . import b_one_frame_processor_v2 as PROC
from . import b_one_frame_production_factory_v1 as P
from . import b_route_bridge_v4 as BRIDGE
from . import b_ue_process_v1 as UE
from . import final_actor_gate_v2 as F
from . import live_adapters_v1 as L
from . import one_frame_engineering_v1 as O


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def actor(variant: str) -> F.LoadedFinalActorV2:
    live = (L.ActorVariant.RUN4B if variant == F.RUN4B_VARIANT
            else L.ActorVariant.RUN5B)
    order = L.expected_feature_order(live)
    identity = F.FinalActorIdentityV2(
        variant=variant, actor_state_dict_sha256=sha("state"),
        actor_tree_sha256=sha("actor-boundary"),
        feature_schema_id="test",
        feature_schema_sha256=L.feature_schema_sha256(live, order),
        feature_order_sha256=sha("order"), feature_order=order,
        model_binding_sha256=sha("model"),
        operational_latency_provider_sha256=sha("latency"),
        scientific_channel_sha256=sha("channel"),
        run5b_only_authority_sha256=(None if live is L.ActorVariant.RUN4B
                                     else sha("run5")),
        selected_seed=43, selected_update=10000)
    return F.LoadedFinalActorV2(identity, object())


def config(root: Path, variant: str) -> O.OneFrameConfigV1:
    manifest = root / ("RUN4B_JOINT_FINAL_ACTOR_MANIFEST_V2.json"
                       if variant == F.RUN4B_VARIANT
                       else "RUN5B_JOINT_FINAL_ACTOR_MANIFEST_V2.json")
    manifest.write_text("{}", encoding="ascii")
    weights = root / "actor_state_dict.pt"
    weights.write_bytes(b"weights")
    remote_repo = Path("/srv/abiodun_remote")
    return O.OneFrameConfigV1(
        run_id="oneframe_01", cell_id="a71__favorable_stable",
        variant=variant, actor_manifest_path=manifest,
        actor_manifest_sha256=sha("manifest-placeholder"),
        actor_weights_path=weights,
        actor_weights_sha256=hashlib.sha256(b"weights").hexdigest(),
        actor_evidence_root=root / "evidence",
        local_repository=root / "local_repo",
        remote_repository=remote_repo,
        local_attempt_root=root / "attempt",
        remote_attempt_root=Path("/srv/attempts/oneframe_01"),
        edge_campaign_config=remote_repo / "rl_agent/configs/campaign.json",
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


class _DeterministicActor:
    def deterministic_execution(self, state):
        return SimpleNamespace(mode_index=torch.tensor([6]),
                               q_e4=torch.tensor([4321]))


class _Contract:
    def resolve_q_e4(self, mode, q):
        return SimpleNamespace(
            mode_id=mode, q_e4=q, keep_count=5479, action_id=None,
            profile_id=None, execution_bundle_sha256=sha("bundle"))


class _Continuous:
    def prepare(self, profile, _input, frame):
        envelope = X.ExecutionEnvelopeV3(
            mode_id=profile.mode_id, q_e4=profile.q_e4,
            keep_count=profile.keep_count, anchor_action_id=None,
            reward_requested=True, session_uuid=frame.session_uuid,
            controller_lineage_sha256=frame.controller_lineage_sha256,
            decision_seq=frame.decision_seq, ticket_seq=frame.ticket_seq,
            frame_id=frame.frame_id, tensor_seq=frame.tensor_seq,
            capture_timestamp_ns=frame.capture_timestamp_ns,
            execution_bundle_sha256=profile.execution_bundle_sha256,
            inner_payload_sha256=hashlib.sha256(b"inner").hexdigest(),
            inner_payload=b"inner")
        return SimpleNamespace(envelope=envelope)


class _Sender:
    def sendto(self, _packet, _remote):
        return 1


class _Processor(PROC.OneFrameOpportunityProcessorV2):
    def _features(self, opportunity, previous):
        return (0.0,) * 20


class _FakeOps:
    def __init__(self):
        self.events = []
        self.state = SimpleNamespace()

    def preflight(self, config, actor): self.events.append("preflight")
    def start(self, config, actor):
        self.events.append("start")
        return self.state
    def execute(self, state):
        self.events.append("execute")
        return O.OneFrameExecutionV1(
            run_id="oneframe_01", variant=F.RUN4B_VARIANT,
            transmitted_frames=1, policy_decisions=1,
            operational_successes=1, operational_timeouts=0,
            observed_latency_ns=10, ack_before_map_offer=True,
            prediction_evidence_written=True, live_qperc_computed=False,
            exact_identity_sha256=sha("identity"),
            result_sha256=sha("result"))
    def stop(self, state): self.events.append("stop")


class ProductionFactoryTest(unittest.TestCase):
    def test_exact_request_translation_for_both_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for variant in (F.RUN4B_VARIANT, F.RUN5B_VARIANT):
                root = Path(directory) / variant
                root.mkdir()
                cfg, selected = config(root, variant), actor(variant)
                with mock.patch.object(
                        UE.BUEProcessRequestV1, "validate",
                        side_effect=AssertionError("300 gate invoked")):
                    request = P.build_ue_request(cfg, selected)
                    encoded, edge = P.build_edge_request(cfg, selected)
                self.assertEqual(request.transmitted_budget, 1)
                self.assertEqual(edge["transmitted_budget"], 1)
                self.assertEqual(edge["split_host"]["ack_receiver_host"],
                                 "10.0.0.2")
                self.assertEqual(edge["ack_semantics"], EP.ACK_SEMANTICS)
                self.assertNotEqual(edge["ack_semantics"], cfg.ack_semantics)
                self.assertNotIn("quality_ack", encoded.lower())

    def test_remote_plan_is_no_build_no_pull_and_container_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, selected = config(root, F.RUN4B_VARIANT), actor(F.RUN4B_VARIANT)
            plan = P.build_remote_edge_plan(cfg, selected)
            service = plan.compose["services"][P.EDGE_SERVICE]
            self.assertNotIn("build", service)
            self.assertEqual(service["pull_policy"], "never")
            self.assertEqual(service["container_name"], "oai-perception-rx")
            self.assertEqual(list(plan.compose["services"]), [P.EDGE_SERVICE])
            command = service["command"]
            self.assertIn("b_edge_runtime_v2", " ".join(command))
            self.assertNotIn("quality", " ".join(command).lower())

    def test_authoritative_lineage_is_distinct_and_emitted(self) -> None:
        boundary, lineage = sha("actor"), sha("controller")
        request = SimpleNamespace(
            variant=L.ActorVariant.RUN4B,
            feature_schema_sha256=sha("schema"), run_id="run",
            actor_boundary_sha256=boundary)
        identity = F.FinalActorIdentityV2(
            variant=F.RUN4B_VARIANT,
            actor_state_dict_sha256=sha("weights"),
            actor_tree_sha256=boundary, feature_schema_id="r4b",
            feature_schema_sha256=sha("schema"),
            feature_order_sha256=sha("order"),
            feature_order=tuple(f"f{i}" for i in range(20)),
            model_binding_sha256=sha("model"),
            operational_latency_provider_sha256=sha("latency"),
            scientific_channel_sha256=sha("channel"),
            run5b_only_authority_sha256=None,
            selected_seed=43, selected_update=10000)
        binding = OPP.LoadedActorBindingV1(
            loaded=F.LoadedFinalActorV2(identity, _DeterministicActor()),
            manifest_path=Path("/authoritative/actor_manifest.json"),
            weights_path=Path("/authoritative/actor_state_dict.pt"))
        with self.assertRaisesRegex(PROC.ControllerLineageError, "substituted"):
            _Processor(
                controller_lineage_sha256=boundary, request=request,
                actor=binding, telemetry=SimpleNamespace(),
                dynamic_contract=_Contract(), continuous_ue=_Continuous(),
                sender=_Sender(), remote=("127.0.0.1", 5000),
                input_builder=lambda *_: None, cell_id="cell",
                chunk_bytes=1200, snr_provider=None)
        processor = _Processor(
            controller_lineage_sha256=lineage, request=request,
            actor=binding,
            telemetry=SimpleNamespace(
                session_uuid="00000000-0000-4000-8000-000000000001"),
            dynamic_contract=_Contract(), continuous_ue=_Continuous(),
            sender=_Sender(), remote=("127.0.0.1", 5000),
            input_builder=lambda frame, radar: (frame, radar),
            cell_id="cell", chunk_bytes=1200, snr_provider=None)
        opportunity = BRIDGE.RouteOpportunityV4(
            sequence=0, frame_id=9, capture_timestamp_ns=10,
            action_open_monotonic_raw_ns=11,
            submit_kwargs={"frame_bgr": "f", "radar_tensor": "r",
                           "ego_pose": (1, 2, 3, 4, 5, 6),
                           "stream_id": "ego"})
        sent = processor(opportunity, None)
        self.assertEqual(sent.identity.controller_lineage_sha256, lineage)
        self.assertNotEqual(sent.identity.controller_lineage_sha256, boundary)

    def test_lifecycle_delegates_and_stop_is_idempotent(self) -> None:
        settings = mock.Mock()
        ops = _FakeOps()
        lifecycle = P.ProductionOneFrameLifecycleV1(settings, ops=ops)
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory), F.RUN4B_VARIANT)
            selected = actor(F.RUN4B_VARIANT)
            lifecycle.preflight(cfg, selected)
            lifecycle.start(cfg, selected)
            value = lifecycle.execute(cfg, selected)
            lifecycle.stop(cfg)
            lifecycle.stop(cfg)
        self.assertEqual(value.transmitted_frames, 1)
        self.assertEqual(ops.events,
                         ["preflight", "start", "execute", "stop"])

    def test_run5_lease_pump_uses_only_observed_ack_rows(self) -> None:
        class Adapter:
            def __init__(self): self.rows = []
            def record_command_ack(self, **kwargs):
                self.rows.append(("ack", kwargs))
            def record_heartbeat(self, **kwargs):
                self.rows.append(("heartbeat", kwargs))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "radio.csv"
            adapter = Adapter()
            pump = P.TargetSnrLeasePumpV1(path, adapter)
            pump.start()
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "step_index", "target_snr_db", "command_timing_status"))
                writer.writeheader()
                writer.writerow({"step_index": 0})
            time.sleep(0.05)
            self.assertEqual(adapter.rows, [])
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "step_index", "target_snr_db", "command_timing_status"))
                writer.writeheader()
                writer.writerow({"step_index": 0, "target_snr_db": 18.5,
                                 "command_timing_status": "ACK_ON_TIME"})
            pump.wait_ready(2.0)
            pump.close()
        self.assertEqual([kind for kind, _ in adapter.rows],
                         ["ack", "heartbeat"])
        self.assertEqual(adapter.rows[0][1]["target_snr_db"], 18.5)

    def test_remote_attempt_creation_is_atomic_and_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory), F.RUN4B_VARIANT)
            selected = actor(F.RUN4B_VARIANT)
            plan = P.build_remote_edge_plan(cfg, selected)
            state = P.StartedOneFrameV1(
                config=cfg, actor=selected,
                ue_request=P.build_ue_request(cfg, selected), edge_plan=plan)
            ops = P.RealProductionOpsV1.__new__(P.RealProductionOpsV1)
            with mock.patch.object(ops, "_checked_ssh") as checked, \
                    mock.patch.object(
                        ops, "_upload_create_only",
                        side_effect=RuntimeError("stop after directory creation")):
                with self.assertRaisesRegex(RuntimeError, "directory creation"):
                    ops._start_remote_edge(state)
            self.assertEqual(
                [call.args[:2] for call in checked.call_args_list],
                [(("mkdir", "--", str(cfg.remote_attempt_root)),
                  "attempt root creation"),
                 (("mkdir", "--", str(plan.state_root)),
                  "attempt state creation")])

    def test_teardown_stops_lease_before_dependencies(self) -> None:
        events = []
        state = SimpleNamespace(
            stopped=False,
            lease_pump=SimpleNamespace(close=lambda: events.append("lease")),
            dependencies=SimpleNamespace(
                close=lambda: events.append("dependencies")),
            target_process=None, target_output=None, target_stop=None,
            map_process=None, carla=None, ran=None, edge_plan=None,
            config=SimpleNamespace(local_attempt_root=Path("/unused")))
        ops = P.RealProductionOpsV1.__new__(P.RealProductionOpsV1)
        with mock.patch.dict(sys.modules, {
                "rl_agent.ue_route_b_split_cell_adapter_v1":
                    SimpleNamespace()}):
            ops.stop(state)
        self.assertEqual(events, ["lease", "dependencies"])

    def test_execution_uses_budget_one_entry_point(self) -> None:
        identity = SimpleNamespace(exact_sha256=lambda: sha("exact"))
        outcome = SimpleNamespace(
            success=True, observed_latency_ns=10, identity=identity,
            tail_output_sha256=sha("tail"))
        store = SimpleNamespace(
            verify_all=lambda **_kwargs: SimpleNamespace(outcomes=[outcome]))
        request = SimpleNamespace(evidence_root=Path("/evidence"))
        state = SimpleNamespace(
            pipeline=object(), dependencies=object(), ue_request=request,
            config=SimpleNamespace(run_id="oneframe_01",
                                   variant=F.RUN4B_VARIANT))
        ops = P.RealProductionOpsV1.__new__(P.RealProductionOpsV1)
        with mock.patch.object(UE, "UdpOperationalAckReceiverV1",
                               return_value=object()), \
                mock.patch.object(ONE, "execute_one", return_value={
                    "transmitted_frames": 1, "result_sha256": sha("result")
                }) as execute_one, \
                mock.patch.object(
                    UE, "execute_300",
                    side_effect=AssertionError("legacy 300-frame entry called")), \
                mock.patch.object(P.ACK.OperationalEvidenceStoreV1,
                                  "open_existing", return_value=store), \
                mock.patch.object(ops, "_download_prediction"):
            result = ops.execute(state)
        execute_one.assert_called_once_with(request, state.pipeline, mock.ANY)
        self.assertEqual(result.transmitted_frames, 1)


if __name__ == "__main__":
    unittest.main()
