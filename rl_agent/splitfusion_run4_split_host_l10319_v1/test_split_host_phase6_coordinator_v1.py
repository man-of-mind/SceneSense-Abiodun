"""Offline tests for the additive split-host Phase-6 coordinator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from . import contract as C
from . import remote_edge_gt_entry_v1 as RGT
from . import remote_edge_lifecycle_v1 as RE
from . import split_host_phase6_coordinator_v1 as S


def policy_ownership() -> S.PolicyRoutingOwnershipV1:
    return S.PolicyRoutingOwnershipV1(
        schema=S.POLICY_OWNERSHIP_SCHEMA,
        before_rules_sha256="1" * 64,
        before_table_sha256="2" * 64,
        rules=(S.OwnedPolicyRuleV1(priority=31001, before_absent=True),),
        routes=(S.OwnedPolicyRouteV1(
            destination="192.168.70.140/32", before_absent=True),),
    ).validate()


def source_route() -> dict:
    return dict(S.validate_ue_source_route(json.dumps([{
        "dst": "192.168.70.140", "dev": "oaitun_ue1", "table": 9999,
        "prefsrc": "10.0.0.2", "flags": [],
    }])))


def radio_observations():
    common = dict(
        session_uuid="session-1", frame_id=7, tensor_seq=3,
        payload_sha256="3" * 64, datagram_count=5, payload_bytes=50000,
        destination="192.168.70.140", destination_port=51002,
    )
    capture = S.RadioTensorObservationV1(
        observer="W10275_OAITUN_CAPTURE", interface="oaitun_ue1", **common)
    receipt = S.RadioTensorObservationV1(
        observer="L10319_EDGE_RECEIPT", interface="REMOTE_EDGE_RECEIVER", **common)
    return capture, receipt


def prerequisites() -> S.SplitHostCoordinatorPrerequisitesV1:
    capture, receipt = radio_observations()
    return S.SplitHostCoordinatorPrerequisitesV1(
        remote=S.PrestartedRemoteEdgeV1(
            plan_sha256="4" * 64, attempt_id="attempt-1",
            project_name="run4-edge-l10319-attempt-1", container_id="container-1",
            ready_sha256="5" * 64, gt_ready_sha256="6" * 64,
            feedback_route_sha256="7" * 64,
        ),
        source_route=source_route(),
        radio_tensor_path=S.validate_radio_tensor_path(capture, receipt),
        policy_ownership=policy_ownership(),
    ).validate()


def pre_run() -> S.SplitHostCoordinatorPreRunV1:
    full = prerequisites()
    return S.SplitHostCoordinatorPreRunV1(
        remote=full.remote, source_route=full.source_route,
        policy_ownership=full.policy_ownership,
    ).validate()


def retrieval(root: Path) -> S.RemoteEvidenceRetrievalPlanV1:
    return S.RemoteEvidenceRetrievalPlanV1(
        schema=S.REMOTE_RETRIEVAL_SCHEMA,
        remote_attempt_root=str(root / "remote-attempt"),
        local_destination=str(root / "local-evidence"),
        required_relative_paths=(
            "state/ready.json", "state/remote_gt_listener_ready.json",
            "state/remote_gt_listener_final.json",
            "evidence/run4_phase6_edge_report.json", "OUTPUT_MANIFEST.json",
        ),
    ).validate()


class RouteAndOwnershipTests(unittest.TestCase):
    def test_probe_is_source_bound_and_requires_tunnel_table(self) -> None:
        self.assertEqual(S.source_route_probe_argv(), (
            "ip", "-j", "route", "get", "192.168.70.140",
            "from", "10.0.0.2", "iif", "lo",
        ))
        self.assertEqual(source_route()["radio_path"], "PASS")
        base = {"dst": "192.168.70.140", "dev": "oaitun_ue1",
                "table": 9999, "prefsrc": "10.0.0.2"}
        for field, value in (
            ("dev", "wlp130s0f0"), ("table", 254),
            ("prefsrc", "10.21.16.222"), ("gateway", "10.21.16.162"),
        ):
            with self.subTest(field=field):
                row = dict(base, **{field: value})
                with self.assertRaises(S.SplitHostCoordinatorError):
                    S.validate_ue_source_route(json.dumps([row]))

    def test_policy_cleanup_is_exact_and_never_flushes(self) -> None:
        owned = policy_ownership()
        commands = owned.cleanup_commands
        rendered = "\n".join(" ".join(row) for row in commands)
        self.assertIn("priority 31001 from 10.0.0.2 lookup 9999", rendered)
        self.assertIn("192.168.70.140/32 dev oaitun_ue1", rendered)
        self.assertNotIn("flush", rendered)
        self.assertNotIn("wlp130s0f0", rendered)
        with self.assertRaises(S.SplitHostCoordinatorError):
            S.PolicyRoutingOwnershipV1(
                schema=S.POLICY_OWNERSHIP_SCHEMA,
                before_rules_sha256="1" * 64,
                before_table_sha256="2" * 64,
                rules=(S.OwnedPolicyRuleV1(31001, False),), routes=(),
            ).validate()

    def test_oaitun_capture_must_match_remote_receipt_exactly(self) -> None:
        capture, receipt = radio_observations()
        proof = S.validate_radio_tensor_path(capture, receipt)
        self.assertTrue(proof["remote_receipt"])
        self.assertFalse(proof["cross_host_latency_computed"])
        changed = S.RadioTensorObservationV1(
            **{**receipt.__dict__, "payload_sha256": "9" * 64})
        with self.assertRaises(S.SplitHostCoordinatorError):
            S.validate_radio_tensor_path(capture, changed)

    def test_gt_is_lan_sideband_not_radio_payload(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = S.build_coordinator_plan(
                state_dir=root / "radio", root=root,
                prerequisites=prerequisites())
        self.assertEqual(plan.gt_sideband_source, "10.21.16.222")
        self.assertEqual(plan.gt_sideband_endpoint, "192.168.70.140:51015")
        self.assertEqual(plan.tensor_endpoint, "192.168.70.140:51002")
        self.assertEqual(plan.feedback_endpoint, "10.0.0.2:51014")
        self.assertFalse(plan.local_cn_operations)
        self.assertFalse(plan.local_edge_operations)
        self.assertFalse(plan.remote_cn_teardown_permitted)
        self.assertFalse(plan.live_run_authorized)


class RemoteReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        binding_path = Path(__file__).with_name("REMOTE_RUNTIME_BINDING_L10319_V1.json")
        self.binding = C.RemoteRuntimeBinding.from_mapping(
            json.loads(binding_path.read_text(encoding="utf-8")))
        fcos = next(item for item in C.ARTIFACTS
                    if item.name == "torchvision_fcos")
        self.paths = RE.RemoteEdgePaths(
            repository_root=root / "repo",
            attempt_root=root / "remote-attempt",
            state_root=root / "remote-attempt" / "state",
            evidence_root=root / "remote-attempt" / "evidence",
            compose_path=root / "remote-attempt" / "remote_edge.compose.json",
            fcos_weight_path=root / "repo" / fcos.relative_path,
            campaign_config_relative="rl_agent/configs/campaign.json",
        )
        self.invocation = RE.RemoteEdgeInvocation(
            attempt_id="attempt-001", run_id="run-1", cell_id="cell-1",
            action_id=71, allowed_action_ids=(71,), edge_receive_port=51002,
            ue_control_host="10.0.0.2", ue_control_port=51014,
        )
        self.plan = RE.build_plan(binding=self.binding, paths=self.paths,
                                  invocation=self.invocation)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def records(self):
        service = self.plan.compose_document["services"][RE.SERVICE]
        mounts = {row["target"]: (row["source"], not row["read_only"])
                  for row in service["volumes"]}
        container = {
            "container_id": "abc", "image_id": C.REMOTE_CONTAINER_IMAGE_ID,
            "project": self.invocation.project_name,
            "labels": dict(service["labels"]), "mounts": mounts,
        }
        ready = {
            "schema": RE.READY_SCHEMA, "architecture": RE.READY_ARCHITECTURE,
            "run4_edge": True, "action_id": 71, "tail_device": "cuda:0",
            "direct_map_host": "10.21.16.222", "direct_map_port": 39320,
            "ue_control_host": "10.0.0.2", "ue_control_port": 51014,
            "quality_spec_sha256": RE.QUALITY_SPEC_SHA256,
            "dense_label_map_on_radio": False, "object_records_on_radio": False,
            "evaluation_evidence_dir": RE.EVIDENCE_DESTINATION,
        }
        gt_ready = {
            "schema": RGT.SCHEMA, "status": "LISTENING",
            "run_id": "run-1", "cell_id": "cell-1",
            "bind_host": "192.168.70.140",
            "advertised_endpoint": "192.168.70.140:51015",
            "max_tickets": RGT.MAX_GT_TICKETS,
            "socket_timeout_s": RE.GT_SOCKET_TIMEOUT_S,
            "expectation_timeout_s": RE.GT_EXPECTATION_TIMEOUT_S,
            "cross_host_clock_subtraction": False,
            "policy_deadline_clock_owner": "W10275",
        }
        route = {
            "schema": S.FEEDBACK_ROUTE_SCHEMA, "observed_host": "L10319",
            "container_ip": "192.168.70.140", "destination": "10.0.0.2",
            "via": "192.168.70.134", "device": "eth0", "returncode": 0,
            "route_text": "10.0.0.2 via 192.168.70.134 dev eth0",
        }
        return container, ready, gt_ready, route

    def test_prestarted_edge_binds_all_remote_facts(self) -> None:
        remote = S.validate_prestarted_remote_edge(
            plan=self.plan,
            container_observation=self.records()[0], ready_record=self.records()[1],
            gt_ready_record=self.records()[2], feedback_route=self.records()[3])
        self.assertEqual(remote.container_id, "abc")
        self.assertEqual(remote.project_name, self.invocation.project_name)

    def test_feedback_route_must_be_remote_docker_exec_via_upf(self) -> None:
        route = self.records()[3]
        self.assertEqual(S.validate_remote_feedback_route(route)["via"],
                         "192.168.70.134")
        for field, value in (("observed_host", "W10275"),
                             ("via", "10.21.16.162"),
                             ("returncode", 1)):
            with self.subTest(field=field):
                bad = dict(route, **{field: value})
                with self.assertRaises(S.SplitHostCoordinatorError):
                    S.validate_remote_feedback_route(bad)

    def test_retrieval_is_remote_owned_and_ordered_after_local_close(self) -> None:
        plan = retrieval(Path(self.temp.name))
        self.assertFalse(plan.local_may_stop_remote_cn)
        self.assertTrue(plan.plan_only)
        self.assertFalse(plan.operationally_consumed_by_local_coordinator)
        self.assertIn("AFTER_LOCAL_SENDER_CLOSE", plan.transfer_owner)


class FakeSender:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.events = []
        self.connected = False
        self.closed = False

    def wrap_after_recorder(self, objects, semantic):
        self.events.append(("wrapped", objects, semantic))

        def wrapped_objects(*args, **kwargs):
            self.events.append("objects")
            return objects(*args, **kwargs)

        def wrapped_semantic(*args, **kwargs):
            self.events.append("semantic")
            return semantic(*args, **kwargs)

        return wrapped_objects, wrapped_semantic

    def connect(self):
        self.connected = True
        self.events.append("connect")

    def close(self):
        self.connected = False
        self.closed = True
        self.events.append("close")
        return {"schema": "fake.gt.sender.v1", "closed": True,
                "cross_host_clock_subtraction": False}


class ContextAdapterTests(unittest.TestCase):
    def make_modules(self):
        events = []

        def original_objects(*args, **kwargs):
            del args, kwargs
            events.append("original_objects")
            return Path("objects.json")

        def original_semantic(*args, **kwargs):
            del args, kwargs
            events.append("original_semantic")
            return Path("semantic.npy"), Path("semantic.json")

        def original_factory(**kwargs):
            del kwargs
            return SimpleNamespace(_run4_identity={})

        def forbidden_edge(*args, **kwargs):
            del args, kwargs
            raise AssertionError("local edge lifecycle was reached")

        pinned = SimpleNamespace(
            start_map_process=object(), start_live_edge=forbidden_edge,
            stop_live_edge=forbidden_edge, stop_tail=forbidden_edge,
            LivePilotCellRuntime=original_factory, SceneSnapshotSource=object(),
            PassiveSplitCollector=object(), seed_cell_edge_state=object(),
        )
        direct = SimpleNamespace(
            resolve_direct_map_endpoint=lambda **kwargs: (_ for _ in ()).throw(
                AssertionError(f"local Docker resolver reached: {kwargs}")),
            DIRECT_EDGE_MODULE="old", DIRECT_MAP_SERVER="old-map",
            subprocess=object(), _ENDPOINT={},
        )
        quality = SimpleNamespace(
            write_object_ground_truth=original_objects,
            write_semantic_ground_truth=original_semantic,
        )
        feedback = SimpleNamespace(InstallFeedbackLedger=object(), FIELDS=("old",))
        child = SimpleNamespace(install_run4_seams=object(),
                                verify_feedback_path=object(), run=lambda args: 0)

        def base_installer(campaign, *, attempt_dir, **kwargs):
            del campaign, attempt_dir, kwargs
            endpoint = direct.resolve_direct_map_endpoint(port=39320)
            direct._ENDPOINT["endpoint"] = endpoint
            # Simulate the unchanged direct installer and GtWriteRecorderV2.
            pinned.start_live_edge = forbidden_edge
            pinned.stop_live_edge = forbidden_edge
            pinned.stop_tail = forbidden_edge
            pinned.LivePilotCellRuntime = original_factory

            objects, semantic = quality.write_object_ground_truth, quality.write_semantic_ground_truth

            def recorded_objects(*args, **kwargs):
                events.append("recorder_objects")
                return objects(*args, **kwargs)

            def recorded_semantic(*args, **kwargs):
                events.append("recorder_semantic")
                return semantic(*args, **kwargs)

            quality.write_object_ground_truth = recorded_objects
            quality.write_semantic_ground_truth = recorded_semantic
            return {"endpoint": endpoint.as_dict()}

        nobuild = SimpleNamespace(
            install_run4_seams_nobuild=base_installer,
            _DECISION_CAP={"value": 73},
        )
        return (SimpleNamespace(child=child, nobuild=nobuild, pinned=pinned,
                                direct=direct, quality=quality, feedback=feedback),
                events, forbidden_edge, original_objects, original_semantic)

    @staticmethod
    def campaign():
        return {"runtime": {
            "architecture": "DIRECT_EDGE_TO_MAP_V1",
            "edge_remote_host": "192.168.70.140", "edge_receive_port": 51002,
            "direct_map_ingest_port": 39320,
            "ue_bind_host": "10.0.0.2", "ue_control_port": 51014,
            "object_records_on_radio": False,
        }}

    def test_adapter_uses_local_map_remote_proxy_and_restores_all_globals(self) -> None:
        modules, events, forbidden, original_objects, original_semantic = self.make_modules()
        original_install = modules.child.install_run4_seams
        original_verify = modules.child.verify_feedback_path
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            made = []

            def sender_factory(**kwargs):
                sender = FakeSender(**kwargs)
                made.append(sender)
                return sender

            context = S.SplitHostPhase6ChildContextV1(
                prerequisites=prerequisites(), retrieval=retrieval(root),
                modules=modules, sender_factory=sender_factory)
            with context:
                self.assertIs(modules.child.install_run4_seams.__self__, context)
                self.assertIs(modules.child.install_run4_seams.__func__,
                              context.install.__func__)
                seams = context.install(self.campaign(), attempt_dir=root / "attempt")
                self.assertEqual(seams["endpoint"]["host"], "10.21.16.222")
                self.assertEqual(seams["endpoint"]["port"], 39320)
                scratch = modules.pinned.start_live_edge({}, {}, root / "tmp")
                self.assertEqual(scratch,
                                 root / "attempt" / S.LOCAL_PROXY_ROOT)
                self.assertTrue((scratch / S.EDGE_EVIDENCE_LEAF).is_dir())
                self.assertTrue(made[0].connected)
                self.assertTrue(context.sender_ever_connected)
                with self.assertRaises(AttributeError):
                    context.sender_ever_connected = False
                with self.assertRaises(S.SplitHostCoordinatorError):
                    context.remote_teardown_release()
                with self.assertRaises(S.SplitHostCoordinatorError):
                    context.remote_abort_release()
                route = modules.child.verify_feedback_path(
                    map_host="10.21.16.222", map_port=39320,
                    ue_host="10.0.0.2", ue_port=51014)
                self.assertFalse(route["local_docker_exec_used"])
                self.assertTrue(modules.pinned.stop_live_edge(scratch))
                self.assertTrue(made[0].closed)
                release = context.remote_teardown_release()
                self.assertEqual(release, context.remote_teardown_release())
                self.assertEqual(release["schema"], S.REMOTE_TEARDOWN_RELEASE_SCHEMA)
                self.assertTrue(release["local_gt_sender_closed"])
                self.assertEqual(release["remote_attempt_id"], "attempt-1")
                self.assertEqual(release["remote_project_name"],
                                 "run4-edge-l10319-attempt-1")
                self.assertEqual(release["remote_plan_sha256"], "4" * 64)
                expected_radio = hashlib.sha256(json.dumps(
                    prerequisites().radio_tensor_path, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")).hexdigest()
                self.assertEqual(release["radio_tensor_path_sha256"], expected_radio)
                self.assertEqual(release["remote_teardown_owner"],
                                 "REMOTE_LIFECYCLE_OWNER_ONLY")
                self.assertFalse(
                    release["remote_cn_teardown_permitted_to_local_coordinator"])
                final_path = root / "attempt" / S.LOCAL_GT_FINAL
                self.assertTrue(final_path.is_file())
                abort = context.remote_abort_release()
                self.assertEqual(abort["schema"], S.REMOTE_ABORT_RELEASE_SCHEMA)
                self.assertTrue(abort["local_gt_sender_was_connected"])
                self.assertTrue(abort["local_gt_sender_closed"])
                self.assertFalse(abort["scientific_pass"])
                self.assertNotIn("radio_tensor_path_sha256", abort)
                self.assertEqual(abort["local_gt_sender_final_sha256"],
                                 hashlib.sha256(final_path.read_bytes()).hexdigest())
                self.assertTrue(modules.pinned.stop_tail())
            self.assertIs(modules.child.install_run4_seams, original_install)
            self.assertIs(modules.child.verify_feedback_path, original_verify)
            self.assertIs(modules.pinned.start_live_edge, forbidden)
            self.assertIs(modules.quality.write_object_ground_truth, original_objects)
            self.assertIs(modules.quality.write_semantic_ground_truth, original_semantic)
            self.assertEqual(modules.direct._ENDPOINT, {})
            self.assertEqual(made[0].events.count("close"), 1)
        self.assertNotIn("original_objects", events)

    def test_direct_context_restores_decision_cap_on_exception(self) -> None:
        modules, _events, _forbidden, _objects, _semantic = self.make_modules()
        self.assertEqual(modules.nobuild._DECISION_CAP["value"], 73)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with S.SplitHostPhase6ChildContextV1(
                        prerequisites=pre_run(), retrieval=retrieval(root),
                        modules=modules, sender_factory=FakeSender,
                        decision_cap=1):
                    self.assertEqual(modules.nobuild._DECISION_CAP["value"], 1)
                    raise RuntimeError("intentional")
        self.assertEqual(modules.nobuild._DECISION_CAP["value"], 73)

    def test_generic_child_helper_refuses_pre_run_prerequisites(self) -> None:
        modules, _events, _forbidden, _objects, _semantic = self.make_modules()
        calls = []
        modules.child.run = lambda args: calls.append(args) or 0
        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                    S.SplitHostCoordinatorError, "full radio-proof"):
                S.run_unchanged_child(
                    object(), prerequisites=pre_run(),
                    retrieval=retrieval(Path(temporary)), modules=modules,
                    sender_factory=FakeSender)
        self.assertEqual(calls, [])
        self.assertEqual(modules.nobuild._DECISION_CAP["value"], 73)

    def test_abort_release_never_connected_is_none_and_connected_is_bound(self) -> None:
        remote = prerequisites().remote
        self.assertIsNone(S.build_remote_abort_release(
            remote=remote, sender_ever_connected=False,
            local_gt_sender_final_sha256=None))
        with self.assertRaises(S.SplitHostCoordinatorError):
            S.build_remote_abort_release(
                remote=remote, sender_ever_connected=False,
                local_gt_sender_final_sha256="8" * 64)
        with self.assertRaises(S.SplitHostCoordinatorError):
            S.build_remote_abort_release(
                remote=remote, sender_ever_connected=True,
                local_gt_sender_final_sha256=None)
        release = S.build_remote_abort_release(
            remote=remote, sender_ever_connected=True,
            local_gt_sender_final_sha256="8" * 64)
        self.assertEqual(set(release), {
            "schema", "local_gt_sender_was_connected", "local_gt_sender_closed",
            "local_gt_sender_final_sha256", "remote_attempt_id",
            "remote_project_name", "remote_plan_sha256", "scientific_pass",
            "remote_cn_teardown_permitted_to_local_coordinator",
        })
        self.assertEqual(release["remote_attempt_id"], remote.attempt_id)
        self.assertEqual(release["remote_project_name"], remote.project_name)
        self.assertEqual(release["remote_plan_sha256"], remote.plan_sha256)
        self.assertFalse(release["scientific_pass"])
        self.assertNotIn("radio_tensor_path_sha256", release)

    def test_context_never_connected_emits_no_abort_release(self) -> None:
        modules, _events, _forbidden, _objects, _semantic = self.make_modules()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with S.SplitHostPhase6ChildContextV1(
                    prerequisites=pre_run(), retrieval=retrieval(root),
                    modules=modules, sender_factory=FakeSender,
                    decision_cap=1) as context:
                context.install(self.campaign(), attempt_dir=root / "attempt")
                self.assertFalse(context.sender_ever_connected)
                self.assertIsNone(context.remote_abort_release())
                with self.assertRaises(S.SplitHostCoordinatorError):
                    context.remote_teardown_release()

    def test_pre_run_full_release_remains_radio_proof_gated(self) -> None:
        modules, _events, _forbidden, _objects, _semantic = self.make_modules()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with S.SplitHostPhase6ChildContextV1(
                    prerequisites=pre_run(), retrieval=retrieval(root),
                    modules=modules, sender_factory=FakeSender,
                    decision_cap=1) as context:
                context.install(self.campaign(), attempt_dir=root / "attempt")
                scratch = modules.pinned.start_live_edge({}, {}, root / "tmp")
                self.assertTrue(modules.pinned.stop_live_edge(scratch))
                with self.assertRaisesRegex(
                        S.SplitHostCoordinatorError, "radio tensor-path proof"):
                    context.remote_teardown_release()
                capture, receipt = radio_observations()
                final = context.finalize_radio_tensor_path(capture, receipt)
                release = context.remote_teardown_release()
                self.assertEqual(release["remote_attempt_id"],
                                 final.remote.attempt_id)
                self.assertEqual(release["remote_plan_sha256"],
                                 final.remote.plan_sha256)
                with self.assertRaisesRegex(
                        S.SplitHostCoordinatorError, "already finalized"):
                    context.finalize_radio_tensor_path(capture, receipt)

    def test_sender_wrap_is_installed_after_recorder(self) -> None:
        modules, events, _forbidden, _objects, _semantic = self.make_modules()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            made = []

            def factory(**kwargs):
                value = FakeSender(**kwargs)
                made.append(value)
                return value

            with S.SplitHostPhase6ChildContextV1(
                    prerequisites=prerequisites(), retrieval=retrieval(root),
                    modules=modules, sender_factory=factory) as context:
                context.install(self.campaign(), attempt_dir=root / "attempt")
                wrapped_inputs = made[0].events[0]
                self.assertEqual(wrapped_inputs[0], "wrapped")
                self.assertEqual(wrapped_inputs[1].__name__, "recorded_objects")
                self.assertEqual(wrapped_inputs[2].__name__, "recorded_semantic")

    def test_campaign_drift_fails_before_base_installer(self) -> None:
        modules, _events, _forbidden, _objects, _semantic = self.make_modules()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with S.SplitHostPhase6ChildContextV1(
                    prerequisites=prerequisites(), retrieval=retrieval(root),
                    modules=modules, sender_factory=FakeSender) as context:
                campaign = self.campaign()
                campaign["runtime"]["edge_remote_host"] = "127.0.0.1"
                with self.assertRaises(S.SplitHostCoordinatorError):
                    context.install(campaign, attempt_dir=root / "attempt")


class ImportPurityTests(unittest.TestCase):
    def test_module_has_no_execute_cli_or_broad_cleanup(self) -> None:
        source = Path(S.__file__).read_text(encoding="utf-8")
        self.assertNotIn("if __name__ ==", source)
        self.assertNotIn("docker compose", source)
        self.assertNotIn("ip route flush", source)
        self.assertNotIn("ip rule flush", source)


if __name__ == "__main__":
    unittest.main()
