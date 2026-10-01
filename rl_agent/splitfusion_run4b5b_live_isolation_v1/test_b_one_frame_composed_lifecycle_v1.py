"""Composed one-frame lifecycle: real factory start -> execute_one -> stop.

Real: RealProductionOpsV1.start/execute/stop, the attempt-path ownership map,
build_production_dependencies_v1 with the real telemetry AuditWriterV2 (and,
for Run-5B, the real RFsim SNR lease adapter + lease pump), the real
one-frame route bridge and raw GT spool, execute_one, the operational ACK
ledger/evidence stores, the edge prediction store and the prediction download.
Faked: only external services (CARLA, OAI RAN/CN, remote edge host, map
process, target-SNR actuator, tracer readers, UDP sockets, GPU).
"""

from __future__ import annotations

import contextlib
import dataclasses
import csv
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import rl_agent

from . import b_one_frame_execution_v1 as ONE
from . import b_one_frame_pipeline_v1 as PV1
from . import b_one_frame_production_factory_v1 as P
from . import b_production_dependencies_v1 as D
from . import b_route_bridge_v4 as V4
from . import b_ue_process_v1 as UE
from . import branch_evidence_v1 as BE
from . import final_actor_gate_v2 as F
from . import operational_ack_v1 as A
from . import test_b_one_frame_production_factory_v1 as FX

LINEAGE = "c" * 64


def _now() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


class _Telemetry:
    session_uuid = str(uuid.UUID(int=7))
    ue_label = "oai-nrue-1"

    def __init__(self, *args, **kwargs):
        self.unbound_ue_candidates = {(0x4601, 0)}

    def snapshot(self):
        return SimpleNamespace(all_readers_alive=True)

    def bind_ue(self, *, rnti, oai_ue_id):
        self.bound = (rnti, oai_ue_id)

    def on_dci(self, *_): pass
    def on_rlc(self, *_): pass
    def on_pdcp(self, *_): pass


class _Reader:
    def __init__(self, event, handler, telemetry):
        import queue
        self.event, self.audit = event, queue.Queue()
        self.audit.put(f"{event},row")
        self.started = self.stopped = False

    def start(self, argv, cwd):
        self.started = True

    def stop(self):
        self.stopped = True


class _Socket:
    def __init__(self, *_): self.closed = False
    def setsockopt(self, *_): pass
    def bind(self, *_): pass
    def close(self): self.closed = True


class _System:
    """Fake split-host system ops that create exactly what the real ones do."""

    def __init__(self, root: Path, tracer: Path):
        self.root, self.tracer, self.events = root, tracer, []

    def prepare(self, plan):
        self.events.append("prepare")
        out = Path(plan.output_root)
        out.mkdir(parents=True, exist_ok=False)
        (out / "attempt/phase6_artifacts").mkdir(parents=True, exist_ok=False)
        (out / "attempt/ttracer/ue").mkdir(parents=True, exist_ok=False)
        (out / "service").mkdir(exist_ok=False)
        files = {
            "campaign": {"runtime": {"map_install_runtime": "unused",
                                     "udp_chunk_bytes": 1200,
                                     "socket_buffer_request_bytes": 1 << 20},
                         "measurement_contract": {
                             "installed_frame_history_size": 8}},
            "cell": {"network_profile_id": "FAVORABLE_STABLE",
                     "action_id": 71},
            "bindings": {"controller_lineage_sha256": LINEAGE,
                         "tracer_dir": str(self.tracer),
                         "t_messages": str(self.tracer / "T_messages.txt"),
                         "ue_relay_port": 4044},
        }
        for name, value in files.items():
            with (out / "service" / f"{name}.json").open("x") as handle:
                json.dump(value, handle)
        return SimpleNamespace(child_args=SimpleNamespace(
            campaign_json=str(out / "service/campaign.json"),
            cell_json=str(out / "service/cell.json"),
            bindings_json=str(out / "service/bindings.json")))

    def restart_remote_core(self, plan):
        self.events.append("core")
        (Path(plan.output_root) / "REMOTE_CORE_RESET_EVIDENCE.json").write_text("{}")

    def start_ran(self, prepared, plan):
        self.events.append("ran")
        (Path(plan.output_root) / "local_ran_executor").mkdir(exist_ok=False)
        return SimpleNamespace(close=lambda: self.events.append("ran-stop"))

    def start_carla(self, prepared, plan):
        self.events.append("carla")
        (Path(plan.output_root) / "carla_server.log").write_text("ok\n")
        return "carla-handle"

    def stop_carla(self, handle):
        self.events.append("carla-stop")


class _Pinned(SimpleNamespace):
    """Fake target-SNR actuator / process helpers of the pinned adapter."""

    @staticmethod
    def start_target_snr(campaign, *, campaign_path, profile_id,
                         temporary_dir, start_file):
        output = Path(temporary_dir) / "radio_trace.csv"
        with output.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "step_index", "target_snr_db", "command_timing_status"))
            writer.writeheader()
            writer.writerow({"step_index": 0, "target_snr_db": 18.5,
                             "command_timing_status": "ACK_ON_TIME"})
        return (SimpleNamespace(poll=lambda: None), output,
                Path(temporary_dir) / "stop_target_snr")

    @staticmethod
    def stop_target_snr(process, output, stop_file, destination):
        Path(stop_file).write_text("stop\n")
        shutil.move(str(output), str(destination))
        Path(str(output) + ".summary.json").write_text("{}")
        return True

    @staticmethod
    def stop_process(process):
        return True

    @staticmethod
    def action_row(campaign, action_id):
        return {"action_id": action_id}


class _Edge:
    """Fake remote GT-free edge: publish -> ACK -> prediction evidence."""

    def __init__(self, local_remote: Path):
        self.coordinator = BE.TailOutputBranchCoordinatorV1()
        self.local_remote = local_remote
        self.store = None
        self.acks = 0

    def bind(self, runtime_root: Path):
        self.runtime_root = runtime_root
        # The real edge runtime creates its attempt runtime root first.
        self.local_path(runtime_root).mkdir(parents=True, exist_ok=False)
        self.store = BE.PredictionEvidenceStoreV1.create(
            self.local_path(runtime_root / "prediction"))

    def local_path(self, remote: Path) -> Path:
        return self.local_remote / Path(remote).relative_to("/")

    def ack(self, identity):
        tail = b"tail-output:" + identity.exact_sha256().encode()
        publication = self.coordinator.publish(identity, tail)
        self.acks += 1
        self.store.write(publication, tail, _now())
        return A.encode_ack(publication.ack)


def _processor_for(request, calls):
    def processor(opportunity, previous):
        calls.append(opportunity.frame_id)
        identity = A.FrameActionIdentityV1(
            run_id=request.run_id, cell_id="a71__favorable_stable",
            stream_id="ue0_route_b", session_uuid=_Telemetry.session_uuid,
            controller_lineage_sha256=LINEAGE, decision_seq=0, ticket_seq=0,
            frame_id=opportunity.frame_id, tensor_seq=0,
            capture_timestamp_ns=opportunity.capture_timestamp_ns,
            mode_id=11, q_e4=3000, keep_count=7000, anchor_action_id=None,
            profile_id=None, execution_bundle_sha256="b" * 64)
        return UE.BTransmissionV1(
            identity=identity,
            action_open_monotonic_raw_ns=opportunity.action_open_monotonic_raw_ns,
            payload_bytes=177_000, decision_frame=True)
    return processor


def _route_driver(bridge):
    opened = _now()
    try:
        bridge.offer_prepared(V4.RouteOpportunityV4(
            sequence=0, frame_id=7, capture_timestamp_ns=opened - 1_000_000,
            action_open_monotonic_raw_ns=opened, submit_kwargs={}))
    except V4.RouteStopped:
        return
    bridge.stop_requested.wait(10.0)


def _declared(paths: P.AttemptPathsV1) -> set[Path]:
    return {paths.root / row[1] for row in P.ATTEMPT_PATH_OWNERSHIP}


def _present_top(paths: P.AttemptPathsV1) -> set[Path]:
    top = {item for item in paths.root.iterdir()}
    service = paths.root / "service"
    if service.is_dir():
        top |= {item for item in service.iterdir()}
    return top


class ComposedLifecycleTest(unittest.TestCase):
    def run_variant(self, variant: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            cfg = FX.config(base, variant)
            registered = P.OLD.REMOTE_REPOSITORY
            cfg = dataclasses.replace(
                cfg, remote_repository=registered,
                edge_campaign_config=registered / "rl_agent/configs/campaign.json")
            selected = FX.actor(variant)
            paths = P.attempt_paths(cfg)
            tracer = base / "tracer"; tracer.mkdir()
            (tracer / "T_messages.txt").write_text("T")
            remote = base / "remote_host"
            edge = _Edge(remote)
            system = _System(cfg.local_attempt_root, tracer)
            calls: list[int] = []
            ops = P.RealProductionOpsV1.__new__(P.RealProductionOpsV1)
            ops.settings, ops.system = None, system

            def start_remote_edge(state):
                edge.bind(state.edge_plan.runtime_root)

            def ssh(argv, timeout_s=60.0, data=None):
                code = 0
                if tuple(argv[:2]) == ("test", "-f"):
                    code = 0 if edge.local_path(Path(argv[2])).is_file() else 1
                return SimpleNamespace(returncode=code, stdout=b"", stderr=b"")

            ops.remote = SimpleNamespace(download=lambda src, dst: shutil.copyfile(
                edge.local_path(Path(src)), dst))
            built = {}

            def build_pipeline(**kwargs):
                built.update(kwargs)
                request = kwargs["request"]
                return PV1.EngineeringOneFrameBridgeV1(
                    variant=request.variant,
                    feature_schema_sha256=request.feature_schema_sha256,
                    actor_boundary_sha256=request.actor_boundary_sha256,
                    processor=_processor_for(request, calls),
                    route_driver=_route_driver,
                    raw_spool_root=kwargs["raw_spool_root"])

            class Receiver:
                def __init__(self, host, port):
                    self.closed = False
                def receive_until(self, identity, deadline):
                    return edge.ack(identity), _now()
                def close(self):
                    self.closed = True

            import torch
            from rl_agent.splitfusion_live_dispatch_v1 import live_pilot_runtime as BASE
            ue_stub = SimpleNamespace(_front=object(), _ranker=object(),
                                      _ae_encoders={}, _codec=object())
            pinned = _Pinned()
            patches = [
                mock.patch.dict(sys.modules, { "rl_agent.ue_route_b_split_cell_adapter_v1": pinned}),
                mock.patch.object(rl_agent, "ue_route_b_split_cell_adapter_v1", pinned, create=True),
                mock.patch.object(P.CO, "validate_campaign_binding"),
                mock.patch.object(ops, "_start_remote_edge", start_remote_edge),
                mock.patch.object(ops, "_ssh", ssh),
                mock.patch.object(ops, "_checked_ssh", lambda argv, label, timeout_s=60.0: b"edge log\n"),
                mock.patch.object(ops, "_start_map", lambda campaign, root, port: ( (Path(root) / "map").mkdir(exist_ok=False) or SimpleNamespace(poll=lambda: None))),
                mock.patch.object(P.PIPE, "build_one_frame_pipeline_v2", build_pipeline),
                mock.patch.object(UE, "UdpOperationalAckReceiverV1", Receiver),
                mock.patch.object(torch.cuda, "is_available", return_value=True),
                mock.patch.object(BASE, "_preload_ue", return_value=(ue_stub, None, [])),
                mock.patch.object(D.DEC, "load_dynamic_execution_contract", return_value=object()),
                mock.patch.object(D.X, "ContinuousUERuntimeV2", lambda *a, **k: object()),
                mock.patch.object(D.P6, "_SqueezedFront", lambda *a: object()),
                mock.patch.object(D.T, "UeTelemetryProviderV2", _Telemetry),
                mock.patch.object(D.T, "CausalClockBridgeV2", lambda: None),
                mock.patch.object(D.T, "LiveEventReaderV2", _Reader),
                mock.patch.object(D.T, "csv_reader_argv", lambda *a: ["fake"]),
                mock.patch.object(D.socket, "socket", _Socket),
            ]
            with contextlib.ExitStack() as stack:
                for patcher in patches:
                    stack.enter_context(patcher)
                lifecycle = P.ProductionOneFrameLifecycleV1(None, ops=ops)
                lifecycle.start(cfg, selected)
                state = lifecycle.state

                # --- after startup -------------------------------------
                telemetry = paths.get("ue_telemetry") / "telemetry_live"
                self.assertEqual(sorted(p.name for p in telemetry.iterdir()), [
                    "NRUE_MAC_DCI_GRANT_live.csv",
                    "NRUE_MAC_RLC_BUFFER_STATUS_live.csv",
                    "NR_PDCP_TX_SDU_live.csv"])
                for owned in paths.phase("execute"):
                    self.assertFalse(owned.exists(), owned)
                self.assertLessEqual(_present_top(paths), _declared(paths))
                self.assertEqual(built["raw_spool_root"], paths.get("raw_gt_spool"))
                self.assertTrue(paths.get("raw_gt_spool").is_dir())
                if variant == F.RUN5B_VARIANT:
                    self.assertIsNotNone(state.lease_pump)
                    self.assertTrue(state.lease_pump.first_ack.is_set())
                    self.assertIsNone(state.lease_pump.error)
                    self.assertIsNotNone(state.dependencies.snr_controller_adapter)
                else:
                    self.assertIsNone(state.lease_pump)
                    self.assertIsNone(state.dependencies.snr_controller_adapter)

                # --- the real execute_one ------------------------------
                execution = lifecycle.execute(cfg, selected)
                execution.validate(cfg)
                self.assertEqual(calls, [7])
                self.assertEqual(edge.acks, 1)
                self.assertEqual(sorted(p.name for p in paths.get("ue_evidence").iterdir()),
                                 ["operational_evidence", "operational_trace"])
                self.assertEqual(sorted(p.name for p in paths.get("ue_output").iterdir()),
                                 [ONE.REPORT_NAME, ONE.RESULT_NAME])
                records = list((paths.get("remote_prediction") / "records").iterdir())
                self.assertEqual(len(records), 1)
                snapshot = A.OperationalEvidenceStoreV1.open_existing(
                    paths.get("ue_evidence") / "operational_evidence"
                ).verify_all(require_all_resolved=True)
                self.assertEqual(len(snapshot.outcomes), 1)
                self.assertTrue(snapshot.outcomes[0].success)

                # --- no reuse of execution-owned roots ------------------
                with self.assertRaisesRegex(ONE.OneFrameExecutionError,
                                            "create-only"):
                    ONE.execute_one(state.ue_request, state.pipeline, Receiver("", 0))

                # --- teardown -------------------------------------------
                lifecycle.stop(cfg)
                self.assertTrue(state.stopped)
                self.assertTrue(all(r.stopped for r in
                                    state.dependencies.telemetry_readers))
                self.assertTrue(state.dependencies.sender.closed)
                self.assertTrue(paths.get("remote_edge_log").is_file())
                self.assertTrue(paths.get("radio_restoration_trace").is_file())
                self.assertEqual(system.events, [
                    "prepare", "core", "ran", "carla", "carla-stop", "ran-stop"])
                self.assertLessEqual(_present_top(paths), _declared(paths))
                self.assertFalse(paths.get("postrun_remote_prediction").exists())

    def test_run4b_composed_lifecycle(self) -> None:
        self.run_variant(F.RUN4B_VARIANT)

    def test_run5b_composed_lifecycle_with_snr_lease(self) -> None:
        self.run_variant(F.RUN5B_VARIANT)


class OwnershipModelTest(unittest.TestCase):
    def test_every_path_has_one_owner_and_roots_are_disjoint(self) -> None:
        paths = P.AttemptPathsV1(Path("/attempt"))
        names = [row[0] for row in P.ATTEMPT_PATH_OWNERSHIP]
        self.assertEqual(len(names), len(set(names)))
        self.assertNotEqual(paths.get("ue_telemetry"), paths.get("ue_evidence"))
        self.assertNotIn(paths.get("ue_evidence"),
                         paths.get("ue_telemetry").parents)
        self.assertNotIn(paths.get("ue_telemetry"),
                         paths.get("ue_evidence").parents)

    def test_dependency_builder_refuses_existing_telemetry_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ue_telemetry"; root.mkdir()
            tracer = Path(directory)
            with self.assertRaisesRegex(D.ProductionDependencyError,
                                        "create-only"):
                D.build_production_dependencies_v1(
                    variant=UE.L.ActorVariant.RUN4B if hasattr(UE, "L") else
                    __import__("rl_agent.splitfusion_run4b5b_live_isolation_v1.live_adapters_v1",
                               fromlist=["ActorVariant"]).ActorVariant.RUN4B,
                    tracer_dir=tracer, t_messages=tracer, ue_relay_port=4044,
                    telemetry_root=root, ue_bind_host="10.0.0.2",
                    edge_remote_host="192.168.70.140", edge_receive_port=51002,
                    udp_chunk_bytes=1200, socket_buffer_request_bytes=1 << 20)

    def test_300_frame_construction_uses_the_same_ownership(self) -> None:
        """execute_300 owns ue_output/ue_evidence exactly like execute_one;
        startup-owned telemetry in ue_telemetry does not collide, while the
        old wiring (telemetry inside ue_evidence) is refused."""
        with tempfile.TemporaryDirectory() as directory:
            cfg = FX.config(Path(directory), F.RUN4B_VARIANT)
            request = P.build_ue_request(cfg, FX.actor(F.RUN4B_VARIANT))
            object.__setattr__(request, "transmitted_budget", 300)
            paths = P.attempt_paths(cfg)
            self.assertEqual(request.output_root, paths.get("ue_output"))
            self.assertEqual(request.evidence_root, paths.get("ue_evidence"))
            (paths.get("ue_telemetry") / "telemetry_live").mkdir(parents=True)
            pipeline = UE._OfflineFakePipeline(request)
            result = UE.execute_300(request, pipeline,
                                    UE._OfflineFakeReceiver(pipeline))
            self.assertEqual(result["transmitted_frames"], 300)
        with tempfile.TemporaryDirectory() as directory:
            cfg = FX.config(Path(directory), F.RUN4B_VARIANT)
            request = P.build_ue_request(cfg, FX.actor(F.RUN4B_VARIANT))
            (request.evidence_root / "telemetry_live").mkdir(parents=True)
            pipeline = UE._OfflineFakePipeline(request)
            with self.assertRaisesRegex(UE.BUEProcessError, "create-only"):
                UE.execute_300(request, pipeline,
                               UE._OfflineFakeReceiver(pipeline))


if __name__ == "__main__":
    unittest.main()
