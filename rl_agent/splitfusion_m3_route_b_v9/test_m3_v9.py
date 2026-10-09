"""M3.1 v9 binding, GT-accounting, GOOD_ALL, radio-telemetry, drain and offline-P3 tests (CPU only, no live service).

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m unittest -v rl_agent.splitfusion_m3_route_b_v9.test_m3_v9
"""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from rl_agent.splitfusion_m3_route_b_v3 import capacity as CAP
from rl_agent.splitfusion_m3_route_b_v3 import pose_tolerance as PT
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_m3_route_b_v8 import contract as V8
from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as REPAIR
from rl_agent.splitfusion_gt_storage_accounting_v1 import accounting as A, switch as ASW
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN
from rl_agent.splitfusion_numeric_gt_integration_v2 import ready as READY
from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA
from rl_agent.splitfusion_avo_tilt_v1 import quality as Q, switch as SW

from . import accounting_timing as AT
from . import contract as C
from . import gc_seams as GS
from . import good_all_controller as GC
from . import host as H
from . import postflight_p3 as P3
from . import radio_telemetry as RT
from . import retention as RET
from . import route_drain as RD
from . import service_entry as SE
from . import worker_entry as WE

ROOT = Path(__file__).resolve().parents[2]
V8_MAIN = "rl_agent.splitfusion_m3_route_b_v8.worker_entry.main"
V4_MAIN = "rl_agent.splitfusion_m3_route_b_v4.worker_entry.main"


class Binding(unittest.TestCase):
    def test_retained_role_constants_identical(self):
        audit = C.role_constant_audit()
        self.assertTrue(audit)
        self.assertEqual([k for k, ok in audit.items() if not ok], [])

    def test_v9_rebinds_no_unregistered_global_of_an_older_m3_or_retained_module(self):
        from . import binding_regression as BR
        self.assertEqual(BR.unregistered_rebinding(ROOT / C.PACKAGE), [])
        found = {h.split(":", 2)[0] + ":" + h.split(":", 2)[2] for h in BR.foreign_rebinding(ROOT / C.PACKAGE)}
        self.assertEqual(found, set(BR.REGISTERED_REBINDINGS))

    def test_the_detector_still_flags_the_v4_rebinding_pattern(self):
        from . import binding_regression as BR
        self.assertTrue(BR.unregistered_rebinding(ROOT / "rl_agent/splitfusion_m3_route_b_v4"))

    def test_every_v9_module_reads_the_v9_contract(self):
        from . import entries, runner
        for module in (GS, RET, H, SE, WE, GC, RD, RT, P3, entries, runner):
            self.assertIs(module.C, C)

    def test_identity_is_new_but_scientific_settings_are_v8(self):
        self.assertNotEqual(C.ID, V8.ID)
        self.assertEqual(C.SCIENTIFIC_RUNTIME_BASIS, V8.SCIENTIFIC_RUNTIME_BASIS)
        self.assertTrue(C.token("candidate").startswith("M3_1_ROUTE_B_GOOD_ALL_V9_CANDIDATE_V2_C3900"))
        for name in ("FRAME_CEILING", "BUDGETS", "GC_POLICY", "SAFETY", "POSE", "MAP_CLASS",
                     "CAPACITY_SUBSTITUTIONS", "POSE_SUBSTITUTIONS", "RETENTION_PLAN_ARGS", "RETENTION_PLANNED_FRAMES",
                     "CHILD_TIMEOUT_S", "SAFETY_TIMEOUT_S", "HOST_ROUTE_LIFECYCLE_S", "CARRIER_TRACE", "ABORT_EXPORT",
                     "RETENTION_PLAN_ID", "TRACE_DIR", "WORKER_MODULE", "SERVICE_MODULE", "PRODUCER_CLASS",
                     "NETWORK_PROFILE", "ORIGINAL_PROFILE", "CARRIER_ACTION", "PREP_LIFETIME_QUALIFIED",
                     "AVO_TILT_QUALIFIED", "AVO_GEOMETRY_VERSION", "FAILURE_SNAPSHOT_DIR", "PREP_BINDING_RECORD",
                     "AVO_BINDING_RECORD"):
            self.assertEqual(getattr(C, name), getattr(V8, name), name)
        self.assertEqual(C.limits(C.FRAME_CEILING), V8.limits(V8.FRAME_CEILING))
        self.assertEqual(C.metrics(), V8.metrics())
        self.assertEqual(C.evidence_registration(), V8.evidence_registration())
        self.assertEqual(C.pose_registration(), V8.pose_registration())
        self.assertEqual(C.capacity_registration(), V8.capacity_registration())
        self.assertEqual(C.map_class_registration(), V8.map_class_registration())
        self.assertEqual(C.avo_registration(), V8.avo_registration())
        self.assertEqual(C.retention_plan(C.FRAME_CEILING), V8.retention_plan(V8.FRAME_CEILING))
        self.assertEqual(C.route_registration(ROOT), V8.route_registration(ROOT))

    def test_launcher_binding_names_existing_modules(self):
        binding = C.launcher_binding()
        workers = binding["gt_workers"]
        modules = {binding["contract_module"], binding["partition"], *binding["host"].values(),
                   binding["runner"]["module"], binding["runner"]["radio_telemetry"],
                   binding["runner"]["mac_log_observer"], binding["gt_service"]["trampoline"],
                   binding["gt_service"]["retained_service"], workers["trampoline"], workers["v8_trampoline"],
                   workers["v7_trampoline"], workers["retained_trampoline"], workers["implementation"],
                   workers["prep_avo_switch"], workers["prep_accounting_switch"], binding["child"]["entries"],
                   binding["child"]["controller"], binding["child"]["collector_layer"],
                   binding["edge"]["entries"], binding["postflight"]["p3"],
                   workers["prep_core"].rsplit(".", 1)[0], workers["prep_avo"].split(":")[0],
                   workers["prep_accounting"].split(":")[0]}
        for name in modules:
            self.assertIsNotNone(importlib.util.find_spec(name), name)
        self.assertIs(getattr(A, workers["prep_accounting"].split(":")[1]), A.payload_storage_bytes)

    def test_v8_difference_whitelist_excludes_scientific_fields(self):
        for key in ("exact_gt_limits", "route", "capacity", "pose_repair", "map_class", "lifecycle",
                    "live_metric_contract", "network_profile", "evidence_additions", "transmitted_budget",
                    "support_sha256"):
            self.assertNotIn(key, H.V8_DIFFERENCES_ALLOWED)

    def test_importing_v9_installs_nothing(self):
        self.assertIs(NA.numeric_avo, SW.ORIGINAL_NUMERIC_AVO)
        self.assertIs(OWN.payload_storage_bytes, ASW.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        self.assertFalse(SW.installed() or ASW.installed())
        self.assertIs(PC.PrepCore, REPAIR.ORIGINAL_PREP_CORE)
        self.assertFalse(hasattr(READY, "_M3_AGREE_RECONSTRUCTED"))


class StorageAccounting(unittest.TestCase):
    def test_registered_sources_are_the_offline_qualified_ones(self):
        for relative, digest in C.ACCOUNTING_QUALIFIED.items():
            self.assertEqual(C.sha(ROOT / relative), digest, relative)
        self.assertEqual(C.ACCOUNTING_VERSION, A.VERSION)
        self.assertEqual(C.ACCOUNTING_SUBSTITUTION, ASW.SUBSTITUTION)
        self.assertEqual(C.ACCOUNTING_SWITCH, ASW.__name__)

    def test_gc_registration_differs_from_v8_only_in_gt_entries_and_accounting(self):
        mine, v8 = C.gc_registration("candidate", 3900), V8.gc_registration("candidate", 3900)
        self.assertEqual(sorted(k for k in set(mine) | set(v8) if mine.get(k) != v8.get(k)),
                         ["implementations", "seam", "storage_accounting", "trampolines"])
        self.assertEqual(mine["trampolines"], dict(service=C.SERVICE_ENTRY, worker=C.WORKER_ENTRY))
        self.assertEqual(mine["avo_geometry"], v8["avo_geometry"])
        self.assertEqual(mine["prep_freeze_lifetime"], v8["prep_freeze_lifetime"])
        accounting = mine["storage_accounting"]
        self.assertEqual(accounting["roles"], ["prep"])
        self.assertIsNone(accounting["cache"])
        self.assertIs(accounting["reservation_bytes_unchanged"], True)
        self.assertIs(accounting["capacity_rule_unchanged"], True)
        self.assertIs(accounting["timing"]["writes_on_request_path"], False)

    def test_sources_and_required_composition_cover_the_candidate(self):
        from . import entries
        sources = C.exact_sources(ROOT)
        for relative in (*C.ACCOUNTING_QUALIFIED, f"{C.PACKAGE}/worker_entry.py", f"{C.PACKAGE}/accounting_timing.py",
                         f"{C.PACKAGE}/route_drain.py", f"{C.PACKAGE}/radio_telemetry.py",
                         f"{C.PACKAGE}/good_all_controller.py", f"{V8.PACKAGE}/worker_entry.py"):
            self.assertIn(relative, sources)
            self.assertIn(relative, entries.REQUIRED_SOURCES)


class Timer(unittest.TestCase):
    def test_returns_the_wrapped_value_unchanged_and_records_one_row_per_call(self):
        timer = AT.AccountingTimer(A.payload_storage_bytes, 10)
        payload = {"a": 1, "b": "two", "c": None}
        self.assertEqual(timer(payload), A.payload_storage_bytes(payload))
        self.assertEqual(timer(payload, [1, 2, 3]), A.payload_storage_bytes(payload, [1, 2, 3]))
        document = timer.document()
        self.assertEqual((document["calls"], document["recorded"], document["dropped"], document["errors"]), (2, 2, 0, 0))
        self.assertEqual(len(document["cpu_ns"]), 2)
        self.assertTrue(all(b > 0 for b in document["result_bytes"]))
        self.assertTrue(all(e >= b for b, e in zip(document["begin_raw_ns"], document["end_raw_ns"])))
        self.assertEqual(document["wrapped"], f"{A.__name__}.payload_storage_bytes")

    def test_exceptions_propagate_unchanged_and_are_counted(self):
        def boom(*values):
            raise ValueError("original accounting refusal")
        timer = AT.AccountingTimer(boom, 4)
        with self.assertRaisesRegex(ValueError, "original accounting refusal"):
            timer({})
        self.assertEqual((timer.errors, timer.calls, len(timer.cpu_ns)), (1, 0, 0))

    def test_capacity_is_bounded_and_overflow_is_counted(self):
        timer = AT.AccountingTimer(A.payload_storage_bytes, 2)
        for _ in range(5):
            timer({"x": 1})
        document = timer.document()
        self.assertEqual((document["calls"], document["recorded"], document["dropped"]), (5, 2, 3))

    def test_caller_attribution_names_the_account_caller(self):
        timer = AT.AccountingTimer(A.payload_storage_bytes, 8)

        def _account(row):
            return timer(row)

        def stage_products():
            return _account({"x": 1})

        stage_products()
        document = timer.document()
        self.assertEqual(document["caller_names"], ["stage_products"])
        self.assertEqual(document["foreign_frames"], 0)

    def test_a_caller_that_is_not_account_is_counted_as_foreign(self):
        timer = AT.AccountingTimer(A.payload_storage_bytes, 8)

        def not_account(row):
            return timer(row)

        not_account({"x": 1})
        self.assertEqual(timer.foreign_frames, 1)


class WorkerEntry(unittest.TestCase):
    def args(self, role, trace):
        return ["--fd", "3", "--parent", "1", "--role", role,
                "--gc-policy", "candidate" if role == "prep" else "control",
                "--trace-dir", str(trace), "--max-frames", "3900", "--producer", "v2" if role == "prep" else "v1"]

    def assert_nothing_installed(self):
        self.assertIs(OWN.payload_storage_bytes, ASW.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        self.assertFalse(ASW.installed())
        self.assertIs(NA.numeric_avo, SW.ORIGINAL_NUMERIC_AVO)
        self.assertFalse(SW.installed())
        self.assertIs(PC.PrepCore, REPAIR.ORIGINAL_PREP_CORE)

    def test_prep_installs_accounting_timer_avo_and_repair_together_and_restores_all(self):
        seen = {}

        def retained(argv):                      # the v4 trampoline, through the REAL v9 -> v8 -> v7 chain
            seen.update(accounting=OWN.payload_storage_bytes, avo=NA.numeric_avo, core=PC.PrepCore, argv=list(argv))
            seen["value"] = OWN.payload_storage_bytes({"a": 1})
            return 0
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args("prep", temporary)
            with patch(V4_MAIN, side_effect=retained):
                self.assertEqual(WE.main(args), 0)
            binding = json.loads((Path(temporary) / C.ACCOUNTING_BINDING_RECORD).read_text())
            timing = json.loads((Path(temporary) / C.ACCOUNTING_TIMING_RECORD).read_text())
            avo = json.loads((Path(temporary) / C.AVO_BINDING_RECORD).read_text())
            life = json.loads((Path(temporary) / C.PREP_BINDING_RECORD).read_text())
        self.assertIsInstance(seen["accounting"], AT.AccountingTimer)
        self.assertIs(seen["accounting"].__wrapped__, A.payload_storage_bytes)
        self.assertEqual(seen["value"], A.payload_storage_bytes({"a": 1}))     # exact value through the timer
        self.assertIs(seen["avo"], Q.numeric_avo)
        self.assertIs(seen["core"], REPAIR.PrepCoreLifetime)
        self.assertEqual(seen["argv"], args)                                   # retained parser gets every argument
        self.assert_nothing_installed()
        self.assertEqual((binding["installed"], binding["timer_installed"], binding["timer_removed"],
                          binding["restored"], binding["exit"]), (True, True, True, True, None))
        self.assertEqual((binding["entry_module"], binding["v8_worker_entry"]), (C.WORKER_ENTRY, C.V8_WORKER_ENTRY))
        self.assertEqual(binding["version"], C.ACCOUNTING_VERSION)
        self.assertEqual(binding["substitution"], C.ACCOUNTING_SUBSTITUTION)
        self.assertEqual(binding["sources"], {n: C.ACCOUNTING_QUALIFIED[f"{C.ACCOUNTING_PACKAGE}/{n}.py"]
                                              for n in ("accounting", "switch")})
        self.assertEqual(timing["schema"], AT.SCHEMA)
        self.assertEqual((timing["calls"], timing["recorded"], timing["dropped"], timing["errors"]), (1, 1, 0, 0))
        self.assertEqual(timing["capacity"], C.ACCOUNTING_TIMING_CAPACITY)
        self.assertEqual((avo["installed"], avo["restored"]), (True, True))
        self.assertEqual((life["installed"], life["restored"]), (True, True))
        self.assertEqual(avo["pid"], life["pid"])
        self.assertEqual(binding["pid"], avo["pid"])

    def test_score_runs_the_v8_worker_with_nothing_installed(self):
        seen = {}
        with tempfile.TemporaryDirectory() as temporary:
            with patch(V4_MAIN, side_effect=lambda argv: seen.update(accounting=OWN.payload_storage_bytes,
                                                                     avo=NA.numeric_avo, core=PC.PrepCore)):
                WE.main(self.args("score", temporary))
            for name in (C.ACCOUNTING_BINDING_RECORD, C.ACCOUNTING_TIMING_RECORD, C.AVO_BINDING_RECORD):
                self.assertFalse((Path(temporary) / name).exists(), name)
        self.assertIs(seen["accounting"], ASW.ORIGINAL_PAYLOAD_STORAGE_BYTES)
        self.assertIs(seen["avo"], SW.ORIGINAL_NUMERIC_AVO)
        self.assertIs(seen["core"], REPAIR.ORIGINAL_PREP_CORE)

    def test_original_error_survives_and_reaches_the_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(V4_MAIN, side_effect=RuntimeError("original producer refusal")):
                with self.assertRaisesRegex(RuntimeError, "original producer refusal"):
                    WE.main(self.args("prep", temporary))
            binding = json.loads((Path(temporary) / C.ACCOUNTING_BINDING_RECORD).read_text())
        self.assertEqual(binding["exit"], "RuntimeError: original producer refusal")
        self.assertEqual((binding["restored"], binding["timer_removed"]), (True, True))
        self.assert_nothing_installed()

    def test_record_write_failure_never_masks_the_worker_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / C.ACCOUNTING_BINDING_RECORD).mkdir()
            with patch(V8_MAIN, side_effect=SystemExit(1)):
                with self.assertRaises(SystemExit) as caught:
                    WE.main(self.args("prep", temporary))
        self.assertEqual(caught.exception.code, 1)
        self.assert_nothing_installed()


class GoodAll(unittest.TestCase):
    def test_schedule_is_the_qualified_23p5_db_anchor(self):
        self.assertEqual(C.SCHEDULE["target_snr_db"], 23.5)
        self.assertEqual(C.SCHEDULE["expected_rfsim_command_db"], -12.0)
        self.assertEqual(C.SCHEDULE["mapping_point"], "PHASE14A_100MHZ_ANCHOR_01")
        self.assertEqual(C.SCHEDULE["schedule_id"], "GOOD_ALL")
        self.assertEqual(C.SCHEDULE["kind"], "CONSTANT_TARGET_EVERY_STEP")
        self.assertEqual(C.SCHEDULE["state_index"], V8.SCHEDULE["state_index"])
        self.assertNotEqual(C.SCHEDULE["trace_id"], V8.SCHEDULE["trace_id"])
        self.assertEqual(sorted(k for k in set(C.SCHEDULE) | set(V8.SCHEDULE)
                                if C.SCHEDULE.get(k) != V8.SCHEDULE.get(k)), sorted(C.SCHEDULE_FIELDS_CHANGED))

    def test_actor_input_is_the_commanded_target_inside_the_frozen_support(self):
        from rl_agent.splitfusion_hybrid_sac_run5_v1 import run5_snr_v2 as R5
        self.assertEqual(R5.scale_snr_db(C.SCHEDULE["target_snr_db"]), C.ACTOR_SNR_FEATURE)
        self.assertEqual(C.ACTOR_SNR_FEATURE, (23.5 - 5.5) / 19.0)
        actor = C.channel_registration(ROOT)["actor_input"]
        self.assertEqual(actor["value"], C.ACTOR_SNR_FEATURE)
        self.assertIs(actor["normalisation_changed"], False)
        self.assertIs(actor["gnb_measurement_to_actor"], False)

    def test_channel_registration_binds_the_measured_mapping_point_and_constant_digest(self):
        channel = C.channel_registration(ROOT)
        self.assertEqual(channel["controller_module"], C.CONTROLLER_MODULE)
        self.assertEqual(channel["carrier_cell"], V8.channel_registration(ROOT)["carrier_cell"])
        prefix, digest = C.schedule_prefix(channel["schedule"]["prefix_samples"])
        self.assertEqual(channel["schedule"]["trace_sha256"], digest)
        self.assertEqual(set(prefix), {(C.SCHEDULE["state_index"], 23.5)})
        self.assertNotEqual(digest, V8.channel_registration(ROOT)["schedule"]["trace_sha256"])

    def test_controller_rewrites_the_carrier_profile_and_refuses_anything_else(self):
        self.assertEqual(GC.rewrite_profile(["--profile-id", C.ORIGINAL_PROFILE])[1], "GOOD_ALL")
        with self.assertRaisesRegex(ValueError, "exactly one --profile-id"):
            GC.rewrite_profile(["--profile-id", "BAD_ALL"])

    def test_controller_prepare_returns_the_constant_sequence_after_the_retained_checks(self):
        channel = C.channel_registration(ROOT)
        seen = {}
        prepare = GC.build_prepare(lambda **kwargs: seen.update(kwargs), channel)
        campaign = dict(network=dict(prefix_samples=channel["schedule"]["prefix_samples"]))
        sequence, prefix, frozen = prepare(campaign=campaign, profile_id="GOOD_ALL")
        self.assertEqual(seen["profile_id"], C.ORIGINAL_PROFILE)      # the retained pin checks still run
        self.assertEqual(sequence.next_sample(), (C.SCHEDULE["state_index"], 23.5))
        self.assertEqual(frozen["trace_sha256"], channel["schedule"]["trace_sha256"])
        self.assertEqual(len(prefix), channel["schedule"]["prefix_samples"])
        with self.assertRaisesRegex(ValueError, "foreign profile id"):
            prepare(campaign=campaign, profile_id=C.ORIGINAL_PROFILE)


class RadioTelemetry(unittest.TestCase):
    def test_field_lists_equal_the_tracer_database(self):
        registration = C.radio_telemetry_registration(ROOT)
        self.assertEqual(registration["events"][RT.PUSCH_EVENT], list(RT.PUSCH_FIELDS))
        self.assertEqual(registration["events"][RT.MCS_EVENT], list(RT.MCS_FIELDS))
        self.assertIn("snrx10", RT.PUSCH_FIELDS)
        self.assertIn("avg_snr_x10", RT.MCS_FIELDS)
        self.assertIn("selected_mcs", RT.MCS_FIELDS)
        self.assertIn("final_mcs", RT.MCS_FIELDS)
        self.assertEqual(registration["database_sha256"], C.sha(ROOT / C.T_MESSAGES))

    def test_registration_refuses_a_field_list_that_drifts_from_the_database(self):
        with patch.object(RT, "PUSCH_FIELDS", RT.PUSCH_FIELDS[:-1]):
            with self.assertRaisesRegex(ValueError, "differs from the tracer database"):
                C.radio_telemetry_registration(ROOT)

    def test_mac_log_statistic_is_kept_but_separately_labelled(self):
        registration = C.radio_telemetry_registration(ROOT)
        mac = registration["quantities"]["mac_log_snr"]
        self.assertIs(mac["kept"], True)
        self.assertEqual(mac["observer"], C.MAC_LOG_OBSERVER)
        self.assertIn("NOT channel SNR", mac["label"])
        self.assertIn("snrx10", registration["quantities"]["pusch_snr_db"])
        self.assertIn("avg_snr_x10", registration["quantities"]["scheduler_avg_snr_db"])
        self.assertNotIn("gnb_stats_observed", registration["quantities"]["pusch_snr_db"])

    def test_collectors_are_placed_on_the_registered_radio_cpus(self):
        self.assertEqual(RT._placed(["csv", "-d", "x"], [0, 1, 8]), ["taskset", "-c", "0,1,8", "csv", "-d", "x"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm.json").write_text(json.dumps(dict(radio_cpus=[0, 1, 2, 3, 8, 9])))
            (root / "registration.json").write_text(json.dumps(dict(id=C.ID)))
            self.assertEqual(RT.radio_cpus(root), [0, 1, 2, 3, 8, 9])

    def test_radio_cpus_resolves_from_the_capture_directory_the_runner_is_given(self):
        """The runner passes --output-root <prospective>/capture, which need not exist when install_parent runs."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm.json").write_text(json.dumps(dict(radio_cpus=[0, 1, 2, 3, 8, 9])))
            (root / "registration.json").write_text(json.dumps(dict(id=C.ID)))
            self.assertEqual(RT.radio_cpus(root / "capture"), [0, 1, 2, 3, 8, 9])      # not created yet
            (root / "capture").mkdir()
            self.assertEqual(RT.radio_cpus(root / "capture"), [0, 1, 2, 3, 8, 9])
            self.assertEqual(RT.registered_arm_file(root / "capture"), root / "arm.json")

    def test_radio_cpus_refuses_a_foreign_or_missing_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm.json").write_text(json.dumps(dict(radio_cpus=[0, 1])))
            with self.assertRaisesRegex(RuntimeError, "not found at or above"):
                RT.radio_cpus(root / "capture")                                        # no registration beside it
            (root / "registration.json").write_text(json.dumps(dict(id="M3_1_ROUTE_B_GOOD_ALL_V8")))
            with self.assertRaisesRegex(RuntimeError, "not found at or above"):
                RT.radio_cpus(root / "capture")                                        # foreign registration

    def test_the_runner_installs_the_collectors_only_for_an_executing_launch(self):
        source = (ROOT / C.PACKAGE / "runner.py").read_text()
        self.assertIn("RT.install(output_root, RT.radio_cpus(output_root))", source)
        self.assertIn('and "--execute" in args else None', source)

    def test_start_refuses_when_the_gnb_tracer_port_is_not_listening(self):
        with tempfile.TemporaryDirectory() as temporary:
            telemetry = RT.RadioTelemetry(Path(temporary) / "m3_radio", [0])
            with patch.object(RT, "port_listening", side_effect=lambda port: False):
                with self.assertRaisesRegex(RuntimeError, "is not listening"):
                    telemetry.start()
            self.assertFalse((Path(temporary) / "m3_radio" / "ttracer").exists())

    def test_start_refuses_when_the_relay_port_is_occupied(self):
        with tempfile.TemporaryDirectory() as temporary:
            telemetry = RT.RadioTelemetry(Path(temporary) / "m3_radio", [0])
            with patch.object(RT, "port_listening", side_effect=lambda port: True):
                with self.assertRaisesRegex(RuntimeError, "relay port .* occupied"):
                    telemetry.start()

    def test_install_starts_after_radio_attach_and_stops_before_teardown(self):
        from rl_agent import ue_288_campaign_supervisor as S
        order, start, stop = [], S._start_live_radio, S._stop_live_radio
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "arm.json").write_text(json.dumps(dict(radio_cpus=[0, 1])))
            (Path(temporary) / "registration.json").write_text(json.dumps(dict(id=C.ID)))
            try:
                S._start_live_radio = lambda *a, **k: order.append("radio_start") or ("ns", "state", {})
                S._stop_live_radio = lambda *a, **k: order.append("radio_stop") or {"ok": True}
                telemetry = RT.install(temporary, RT.radio_cpus(temporary))
                telemetry.start = lambda: order.append("telemetry_start") or setattr(telemetry, "started", True)
                telemetry.stop = lambda: order.append("telemetry_stop")
                S._start_live_radio()
                S._stop_live_radio()
            finally:
                S._start_live_radio, S._stop_live_radio = start, stop
        self.assertEqual(order, ["radio_start", "telemetry_start", "telemetry_stop", "radio_stop"])


class _FakeLedger:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}
        self.closed = {}

    def summary(self):
        return dict(obligations_closed=len(self.closed), obligations_open=len(self.pending))


class _FakeFeedback:
    def __init__(self):
        self._ledger = _FakeLedger()
        self._inbox = types.SimpleNamespace(empty=lambda: True, qsize=lambda: 0)

    def record_expired(self, now=None):
        observed = time.time() if now is None else float(now)
        count = 0
        with self._ledger.lock:
            for key, row in list(self._ledger.pending.items()):
                if observed >= float(row["ack_timeout_at"]):
                    self._ledger.pending.pop(key)
                    self._ledger.closed[key] = "FEEDBACK_TIMEOUT"
                    count += 1
        return count

    def summary(self):
        return self._ledger.summary()


class _Base:
    """Minimal stand-in for the retained collector below the drain layer."""

    def __init__(self, **kwargs):
        self.attempt_dir = kwargs["attempt_dir"]
        self.ack_timeout_s = 0.5
        self.sent = 0
        self.dropped = 0
        self.failures = []
        self.rows = []
        self.finished = []
        self.feedback = _FakeFeedback()
        self.transport_counters = types.SimpleNamespace(bump=lambda name: self.rows.append(("bump", name)))

    def _append_row(self, row):
        self.rows.append(row)

    def _process_token(self, token):
        self.sent += 1
        key = f"c:{token['frame_id']}"
        with self.feedback._ledger.lock:
            self.feedback._ledger.pending[key] = dict(ack_timeout_at=time.time() + self.ack_timeout_s)
        return "sent"

    def finish(self):
        self.finished.append(dict(open=len(self.feedback._ledger.pending)))
        return True


class Drain(unittest.TestCase):
    def layer(self, directory):
        return RD.build_drain_layer(_Base)(attempt_dir=directory)

    def evidence(self, directory):
        return json.loads((Path(directory) / "m3_route" / "END_OF_ROUTE_DRAIN.json").read_text())

    def test_in_flight_token_is_waited_for_and_its_terminal_arrives_before_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            started, release = threading.Event(), threading.Event()

            retained = _Base._process_token

            def slow(self, token):                    # patched on the class: the drain layer's super() call
                started.set()
                release.wait(5)
                return retained(self, token)
            with patch.object(_Base, "_process_token", slow):
                worker = threading.Thread(target=collector._process_token, args=({"frame_id": 8595},))
                worker.start()
                self.assertTrue(started.wait(5))

                def ack():                      # the real ACK arrives while the drain is waiting
                    time.sleep(0.15)
                    with collector.feedback._ledger.lock:
                        for key in list(collector.feedback._ledger.pending):
                            collector.feedback._ledger.pending.pop(key)
                            collector.feedback._ledger.closed[key] = "RESULT_INSTALLED"
                acker = threading.Thread(target=ack)
                release.set()
                worker.join(5)
                acker.start()
                self.assertTrue(collector.finish())
                acker.join(5)
            evidence = self.evidence(temporary)
        self.assertIs(evidence["in_flight_completed"], True)
        self.assertEqual(evidence["sent_at_end"], 1)
        self.assertEqual(evidence["open_at_end"], [])
        self.assertEqual(evidence["expired_on_real_clock"], 0)                 # a real ACK, not a timeout
        self.assertEqual(evidence["ledger_summary"]["obligations_closed"], 1)
        self.assertIs(evidence["closed_equals_sent"], True)
        self.assertEqual(collector.finished, [dict(open=0)])                   # the retained finish still ran

    def test_tokens_offered_during_the_drain_are_dropped_and_never_create_an_obligation(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            collector.finish()
            self.assertIsNone(collector._process_token({"frame_id": 9001}))
            self.assertEqual(collector.sent, 0)
            self.assertEqual(collector.dropped, 1)
            self.assertEqual(collector.feedback._ledger.pending, {})
            evidence = self.evidence(temporary)
        self.assertEqual([r for r in collector.rows if isinstance(r, dict)][0]["prepare_status"], RD.DROP_STATUS)
        self.assertEqual(evidence["tokens_dropped_during_drain"], [])          # dropped after the evidence was written

    def test_an_obligation_that_never_acks_becomes_an_honest_timeout_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            collector._process_token({"frame_id": 4242})
            self.assertTrue(collector.finish())
            evidence = self.evidence(temporary)
        self.assertEqual(evidence["expired_on_real_clock"], 1)
        self.assertEqual(evidence["open_at_end"], [])
        self.assertEqual(collector.feedback._ledger.closed, {"c:4242": "FEEDBACK_TIMEOUT"})
        self.assertIs(evidence["missing_terminal_declared_successful"], False)
        self.assertGreaterEqual(evidence["waited_s"], collector.ack_timeout_s)

    def test_the_drain_never_waits_past_the_absolute_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            with collector.feedback._ledger.lock:                              # an obligation with a far horizon
                collector.feedback._ledger.pending["c:1"] = dict(ack_timeout_at=time.time() + 10_000)
            with patch.object(C, "DRAIN_ABSOLUTE_CAP_S", 0.4):
                began = time.monotonic()
                self.assertTrue(collector.finish())
                elapsed = time.monotonic() - began
            evidence = self.evidence(temporary)
        self.assertLess(elapsed, 5.0)
        self.assertEqual(evidence["open_before_expiry"], ["c:1"])
        self.assertEqual(evidence["expired_on_real_clock"], 0)                 # still inside its own horizon
        self.assertEqual(evidence["open_at_end"], ["c:1"])                     # reported, never silently closed

    def test_a_drain_failure_is_recorded_and_never_replaces_the_retained_teardown(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector = self.layer(temporary)
            with patch.object(collector, "_m3_open_obligations", side_effect=RuntimeError("ledger unavailable")):
                self.assertTrue(collector.finish())
            evidence = self.evidence(temporary)
        self.assertIs(evidence["completed"], False)
        self.assertIn("ledger unavailable", evidence["error"])
        self.assertEqual(collector.finished, [dict(open=0)])
        self.assertTrue(any("end-of-route drain" in f for f in collector.failures))
        self.assertTrue(collector._m3_draining)

    def test_the_layer_sits_directly_above_the_unchanged_v3_layer(self):
        from rl_agent.splitfusion_m3_route_b_v3 import route_completion as RC
        from rl_agent.splitfusion_run5_live_avo_v1 import live_gt as LG
        original = LG.build_live_gt_collector
        with RD.install_collector_layer() as built:
            self.assertIsNot(LG.build_live_gt_collector, original)
            layer = LG.build_live_gt_collector(_Base)
            self.assertEqual(built, [layer])
        self.assertIs(LG.build_live_gt_collector, original)
        names = [cls.__name__ for cls in layer.__mro__]
        # the drain layer wraps the UNCHANGED v3 M3 layer, which wraps whatever the retained builder produced
        self.assertEqual(names[:2], ["M3EndOfRouteDrainLayer", "M3CollectorLayer"])
        self.assertEqual(names[2], original(_Base).__name__)
        self.assertIn(_Base, layer.__mro__)
        self.assertEqual(RC.build_layer(_Base).__name__, "M3CollectorLayer")

    def test_registration_states_what_is_unchanged(self):
        registration = RD.drain_registration()
        self.assertEqual(registration["bounds"]["settle_margin_s"], C.DRAIN_SETTLE_MARGIN_S)
        self.assertIn("170 ms decision deadline and reward resolution", registration["unchanged"])
        self.assertTrue(any("missing terminal successful" in item for item in registration["never"]))


class OfflineP3(unittest.TestCase):
    def test_install_matches_the_live_worker_steps_and_restores(self):
        self.assertFalse(hasattr(READY, "_M3_AGREE_RECONSTRUCTED"))
        original = READY._frame_geometry
        with P3.install_p3() as installed:
            self.assertIs(READY._M3_AGREE_RECONSTRUCTED, PT.agree_reconstructed)
            self.assertIsNot(READY._frame_geometry, original)
            self.assertEqual([row["id"] for row in installed.document()["installed"]], list(CAP.POSE_GT_WORKER))
        self.assertIs(READY._frame_geometry, original)
        self.assertFalse(hasattr(READY, "_M3_AGREE_RECONSTRUCTED"))

    def test_double_install_is_refused(self):
        with P3.install_p3():
            with self.assertRaisesRegex(RuntimeError, "already installed"):
                with P3.install_p3():
                    pass
        self.assertFalse(hasattr(READY, "_M3_AGREE_RECONSTRUCTED"))

    def test_the_registered_rule_is_the_live_rule(self):
        registration = P3.postflight_registration()
        self.assertEqual(registration["rule"], C.POSE)
        self.assertEqual(registration["substitutions"], list(CAP.POSE_GT_WORKER))
        self.assertEqual(registration["rule"]["coordinate_range_m"], 512.0)
        self.assertEqual(len(registration["covers"]), 2)
        self.assertEqual(C.bookkeeping_registration()["offline_p3"], registration)

    def test_the_v3_worker_entry_installs_exactly_these_two_steps(self):
        source = (ROOT / "rl_agent/splitfusion_m3_route_b_v3/worker_entry.py").read_text()
        self.assertIn("READY._M3_AGREE_RECONSTRUCTED = PT.agree_reconstructed", source)
        self.assertIn("CAP.POSE_GT_WORKER", source)


class Seam(unittest.TestCase):
    def registration(self, **gt):
        return dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=dict(C.gc_registration("candidate", 3900), **gt))

    def test_seam_starts_the_v9_service_and_refuses_v8(self):
        self.assertEqual(GS.seam_parameters(self.registration())["service_entry"], C.SERVICE_ENTRY)
        with self.assertRaisesRegex(ValueError, "sealed registration"):
            GS.seam_parameters(dict(self.registration(), id=V8.ID))
        with self.assertRaisesRegex(ValueError, "trampolines differ from the v9 contract"):
            GS.seam_parameters(self.registration(trampolines=V8.gc_registration("candidate", 3900)["trampolines"]))

    def test_service_rewrites_both_roles_to_the_v9_worker_entry(self):
        substitution = SE.WorkerSubstitution(arm="candidate", trace_dir="/t", max_frames=3900, producer="v2")
        argv = ["/usr/bin/python3", "-B", "-u", "-m", C.WORKER_MODULE, "--fd", "3", "--parent", "9"]
        for role, policy, producer in (("prep", "candidate", "v2"), ("score", "control", "v1")):
            substitution._role.value = role
            out = substitution.rewrite(argv)
            self.assertEqual(out[4], C.WORKER_ENTRY)
            self.assertEqual(out[-10:], ["--role", role, "--gc-policy", policy, "--trace-dir", "/t", "--max-frames",
                                         "3900", "--producer", producer])
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            substitution.rewrite(["python3", "-m", "other"])


class Retention(unittest.TestCase):
    def test_v9_loader_accepts_the_unchanged_plan_and_refuses_the_v8_section(self):
        from rl_agent.splitfusion_paired_retention_v2 import plan as RP
        plan = C.retention_plan(C.FRAME_CEILING)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            digest = RP.write_plan(output / C.RETENTION_PLAN_FILE, plan)
            reg = dict(transmitted_budget=C.FRAME_CEILING, paired_retention=C.retention_registration(digest))
            self.assertEqual(RET.load_observer(output, reg).plan["plan_id"], C.RETENTION_PLAN_ID)
            with self.assertRaisesRegex(ValueError, "retention section differs"):
                RET.load_observer(output, dict(reg, paired_retention=V8.retention_registration(digest)))
        self.assertEqual({k: v for k, v in C.retention_registration("a" * 64).items() if k != "loader"},
                         {k: v for k, v in V8.retention_registration("a" * 64).items() if k != "loader"})


if __name__ == "__main__":
    unittest.main()
