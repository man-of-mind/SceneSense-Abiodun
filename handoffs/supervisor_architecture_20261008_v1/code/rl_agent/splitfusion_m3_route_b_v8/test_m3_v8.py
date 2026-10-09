"""M3.1 v8 binding and tilt-aware AVO worker tests (CPU only, no live service).  From the deployment worktree root:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m unittest -v rl_agent.splitfusion_m3_route_b_v8.test_m3_v8
"""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_collection.route_b_publication_actor_volume_visibility_v1 import core
from rl_agent.splitfusion_m3_route_b_v7 import contract as V7
from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as REPAIR
from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA
from rl_agent.splitfusion_avo_tilt_v1 import geometry as G, quality as Q, switch as SW

from . import contract as C
from . import binding_regression as BR
from . import gc_seams as GS
from . import host as H
from . import retention as RT
from . import service_entry as SE
from . import worker_entry as WE

ROOT = Path(__file__).resolve().parents[2]
V7_MAIN = "rl_agent.splitfusion_m3_route_b_v7.worker_entry.main"
V4_MAIN = "rl_agent.splitfusion_m3_route_b_v4.worker_entry.main"


def _c_names(path, scopes=None):
    tree = ast.parse(Path(path).read_text())
    nodes = [tree] if scopes is None else [n for n in tree.body if getattr(n, "name", None) in scopes]
    return {n.attr for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "C"}


class Binding(unittest.TestCase):
    def test_retained_role_constants_identical(self):
        audit = C.role_constant_audit()
        self.assertTrue(audit)
        self.assertEqual([k for k, ok in audit.items() if not ok], [])

    def test_role_constant_lists_cover_every_contract_read(self):
        scopes = {"rl_agent.splitfusion_m3_route_b_v3.entries": set(C.V3_ENTRY_HELPERS),
                  "rl_agent.splitfusion_m3_route_b_v3.retention": {"finalize_after_child", "AttemptRetentionObserver"}}
        for module, _bound, names in C.ROLE_CONSTANTS:
            used = _c_names(ROOT / (module.replace(".", "/") + ".py"), scopes.get(module))
            self.assertTrue(used <= set(names), (module, sorted(used - set(names))))

    def test_v8_rebinds_no_global_of_an_older_m3_or_candidate_module(self):
        self.assertEqual(BR.foreign_rebinding(ROOT / C.PACKAGE), [])

    def test_detector_flags_the_v4_pattern(self):
        self.assertTrue(BR.foreign_rebinding(ROOT / "rl_agent/splitfusion_m3_route_b_v4"))

    def test_child_and_edge_production_scopes_identical_to_v7_v6_v3(self):
        parity = BR.composition_parity()
        self.assertTrue(parity["child"] and parity["edge"])
        self.assertEqual(len(parity["child_items"][0]), 14)

    def test_service_entry_is_the_v7_copy(self):
        self.assertEqual(BR.service_entry_parity(), dict(identical_class=True, identical_main=True))

    def test_worker_entry_is_the_qualified_candidate_trampoline(self):
        self.assertEqual(BR.worker_entry_parity(), dict(candidate_trampoline_is_qualified=True,
                                                        main_differs_only_in_import=True))

    def test_every_v8_module_reads_the_v8_contract(self):
        from . import entries, runner
        for module in (GS, RT, H, SE, WE, entries, runner):
            self.assertIs(module.C, C)

    def test_launcher_binding_names_existing_modules(self):
        binding = C.launcher_binding()
        workers = binding["gt_workers"]
        modules = {binding["contract_module"], binding["partition"], *binding["host"].values(),
                   *binding["runner"].values(), binding["gt_service"]["trampoline"],
                   binding["gt_service"]["retained_service"], workers["trampoline"], workers["v7_trampoline"],
                   workers["retained_trampoline"], workers["implementation"], workers["prep_avo_switch"],
                   binding["child"]["entries"], binding["edge"]["entries"], workers["prep_core"].rsplit(".", 1)[0],
                   workers["prep_avo"].split(":")[0]}
        for name in modules:
            self.assertIsNotNone(importlib.util.find_spec(name), name)
        self.assertIs(getattr(Q, workers["prep_avo"].split(":")[1]), Q.numeric_avo)
        if (ROOT / C.SHIMS[next(iter(C.SHIMS))]).exists():         # deployment worktree only
            for name in (C.CHILD, C.EDGE):
                self.assertIsNotNone(importlib.util.find_spec(name), name)

    def test_launcher_binding_differs_from_v7_only_in_identity_and_gt_roles(self):
        mine, v7 = C.launcher_binding(), V7.launcher_binding()
        self.assertEqual(sorted(k for k in mine if mine[k] != v7[k]),
                         ["child", "contract_module", "edge", "gt_service", "gt_workers", "host", "launcher_revision",
                          "partition", "runner", "runtime_revision"])
        rename = lambda value: json.loads(json.dumps(value).replace("_v8", "_v7"))
        for key in ("child", "edge", "host", "runner", "gt_service"):
            self.assertEqual(rename(mine[key]), v7[key], key)
        added = {k: v for k, v in mine["gt_workers"].items() if k not in v7["gt_workers"]}
        self.assertEqual(sorted(added), ["prep_avo", "prep_avo_switch", "v7_trampoline"])
        self.assertEqual({k: v for k, v in mine["gt_workers"].items() if k in v7["gt_workers"] and k != "trampoline"},
                         {k: v for k, v in v7["gt_workers"].items() if k != "trampoline"})

    def test_identity_is_new_but_scientific_settings_are_v7(self):
        self.assertNotEqual(C.ID, V7.ID)
        self.assertEqual(C.SCIENTIFIC_RUNTIME_BASIS, V7.SCIENTIFIC_RUNTIME_BASIS)
        self.assertTrue(C.token("candidate").startswith("M3_1_ROUTE_B_GOOD_ALL_V8_CANDIDATE_V2_C3900"))
        for name in ("FRAME_CEILING", "BUDGETS", "GC_POLICY", "SAFETY", "POSE", "MAP_CLASS", "SCHEDULE",
                     "CAPACITY_SUBSTITUTIONS", "POSE_SUBSTITUTIONS", "RETENTION_PLAN_ARGS", "RETENTION_PLANNED_FRAMES",
                     "CHILD_TIMEOUT_S", "SAFETY_TIMEOUT_S", "HOST_ROUTE_LIFECYCLE_S", "CARRIER_TRACE", "ABORT_EXPORT",
                     "RETENTION_PLAN_ID", "TRACE_DIR", "WORKER_MODULE", "SERVICE_MODULE", "PRODUCER_CLASS",
                     "NETWORK_PROFILE", "ORIGINAL_PROFILE", "CARRIER_ACTION", "PREP_LIFETIME_QUALIFIED",
                     "PREP_LIFETIME_MODULE", "FAILURE_SNAPSHOT_DIR", "PREP_BINDING_RECORD"):
            self.assertEqual(getattr(C, name), getattr(V7, name), name)
        self.assertEqual(C.limits(C.FRAME_CEILING), V7.limits(V7.FRAME_CEILING))
        self.assertEqual(C.metrics(), V7.metrics())
        self.assertEqual(C.evidence_registration(), V7.evidence_registration())
        self.assertEqual(C.pose_registration(), V7.pose_registration())
        self.assertEqual(C.capacity_registration(), V7.capacity_registration())
        self.assertEqual(C.map_class_registration(), V7.map_class_registration())
        self.assertEqual(C.retention_plan(C.FRAME_CEILING), V7.retention_plan(V7.FRAME_CEILING))
        self.assertEqual(C.channel_registration(ROOT), V7.channel_registration(ROOT))
        self.assertEqual(C.route_registration(ROOT), V7.route_registration(ROOT))

    def test_gc_registration_differs_from_v7_only_in_gt_entries_and_avo(self):
        mine, v7 = C.gc_registration("candidate", 3900), V7.gc_registration("candidate", 3900)
        self.assertEqual(sorted(k for k in set(mine) | set(v7) if mine.get(k) != v7.get(k)),
                         ["avo_geometry", "implementations", "seam", "trampolines"])
        self.assertEqual(mine["trampolines"], dict(service=C.SERVICE_ENTRY, worker=C.WORKER_ENTRY))
        self.assertEqual(mine["prep_freeze_lifetime"], v7["prep_freeze_lifetime"])
        self.assertEqual(mine["avo_geometry"]["roles"], ["prep"])
        self.assertEqual(mine["avo_geometry"]["sources"], C.AVO_TILT_QUALIFIED)
        self.assertIs(mine["avo_geometry"]["silently_skipped"], False)
        self.assertIs(mine["avo_geometry"]["thresholds_changed"], False)

    def test_avo_constants_are_the_candidate_and_frozen_values(self):
        self.assertEqual(C.AVO_GEOMETRY_VERSION, G.GEOMETRY_VERSION)
        self.assertEqual(C.AVO_UPRIGHT_BOUNDARY_DEG, G.UPRIGHT_TOLERANCE_DEG)
        self.assertEqual(C.AVO_GROUND_RULE_ENVELOPE_M, core.GROUND_REJECT_MARGIN_M)
        self.assertEqual(C.AVO_SWITCH, SW.__name__)
        self.assertEqual(C.AVO_SUBSTITUTION["target"], f"{NA.__name__}:numeric_avo")

    def test_avo_and_repair_sources_are_the_qualified_ones_and_registered(self):
        for relative, digest in {**C.PREP_LIFETIME_QUALIFIED, **C.AVO_TILT_QUALIFIED}.items():
            self.assertEqual(C.sha(ROOT / relative), digest, relative)
        sources = C.exact_sources(ROOT)
        from . import entries
        for relative in (*C.PREP_LIFETIME_QUALIFIED, *C.AVO_TILT_QUALIFIED, f"{C.PACKAGE}/worker_entry.py",
                         f"{C.PACKAGE}/service_entry.py", f"{V7.PACKAGE}/worker_entry.py", f"{V7.PACKAGE}/contract.py"):
            self.assertIn(relative, sources)
        for relative in (*C.PREP_LIFETIME_QUALIFIED, *C.AVO_TILT_QUALIFIED, f"{V7.PACKAGE}/worker_entry.py"):
            self.assertIn(relative, entries.REQUIRED_SOURCES)

    def test_v7_difference_whitelist_excludes_scientific_fields(self):
        for key in ("exact_gt_limits", "channel", "route", "capacity", "pose_repair", "map_class", "lifecycle",
                    "live_metric_contract", "network_profile", "evidence_additions", "transmitted_budget"):
            self.assertNotIn(key, H.V7_DIFFERENCES_ALLOWED)

    def test_importing_v8_installs_nothing(self):
        self.assertIs(NA.numeric_avo, SW.ORIGINAL_NUMERIC_AVO)
        self.assertFalse(SW.installed())
        self.assertIs(PC.PrepCore, REPAIR.ORIGINAL_PREP_CORE)


class Seam(unittest.TestCase):
    def registration(self, **gt):
        return dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=dict(C.gc_registration("candidate", 3900), **gt))

    def test_seam_starts_the_v8_service_and_refuses_v7(self):
        self.assertEqual(GS.seam_parameters(self.registration())["service_entry"], C.SERVICE_ENTRY)
        with self.assertRaisesRegex(ValueError, "sealed registration"):
            GS.seam_parameters(dict(self.registration(), id=V7.ID))
        with self.assertRaisesRegex(ValueError, "trampolines differ from the v8 contract"):
            GS.seam_parameters(self.registration(trampolines=V7.gc_registration("candidate", 3900)["trampolines"]))
        argv = ["/usr/bin/python3", "-B", "-u", "-m", C.SERVICE_MODULE, "--fd", "5", "--parent", "7"]
        out = GS.rewrite_service_argv(argv, arm="candidate", trace_dir="/t", max_frames=3900, producer="v2")
        self.assertEqual(out[4], C.SERVICE_ENTRY)

    def test_service_rewrites_both_roles_to_the_v8_worker_entry_and_refuses_other_shapes(self):
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
        substitution._role.value = None
        with self.assertRaisesRegex(RuntimeError, "REFUSED"):
            substitution.rewrite(argv)


class WorkerEntry(unittest.TestCase):
    def args(self, role, trace):
        return ["--fd", "3", "--parent", "1", "--role", role, "--gc-policy", "candidate" if role == "prep" else "control",
                "--trace-dir", str(trace), "--max-frames", "3900", "--producer", "v2" if role == "prep" else "v1"]

    def assert_nothing_installed(self):
        self.assertIs(NA.numeric_avo, SW.ORIGINAL_NUMERIC_AVO)
        self.assertFalse(SW.installed())
        self.assertIs(PC.PrepCore, REPAIR.ORIGINAL_PREP_CORE)
        self.assertIsNone(REPAIR.PrepCoreLifetime.failure_directory)

    def test_prep_installs_avo_and_the_v7_repair_only_while_the_retained_worker_runs(self):
        seen = {}

        def retained(argv):                       # the v4 trampoline, reached through the REAL v7 trampoline
            seen.update(avo=NA.numeric_avo, core=PC.PrepCore, directory=REPAIR.PrepCoreLifetime.failure_directory,
                        argv=list(argv))
            return 0
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args("prep", temporary)
            with patch(V4_MAIN, side_effect=retained):
                self.assertEqual(WE.main(args), 0)
            avo = json.loads((Path(temporary) / C.AVO_BINDING_RECORD).read_text())
            life = json.loads((Path(temporary) / C.PREP_BINDING_RECORD).read_text())
        self.assertIs(seen["avo"], Q.numeric_avo)
        self.assertIs(seen["core"], REPAIR.PrepCoreLifetime)
        self.assertEqual(seen["directory"], temporary)
        self.assertEqual(seen["argv"], args)                       # retained parser receives every argument
        self.assert_nothing_installed()
        self.assertEqual((avo["installed"], avo["restored"], avo["exit"]), (True, True, None))
        self.assertEqual((avo["entry_module"], avo["v7_worker_entry"]), (C.WORKER_ENTRY, C.V7_WORKER_ENTRY))
        self.assertEqual(avo["geometry_version"], C.AVO_GEOMETRY_VERSION)
        self.assertEqual(avo["sources"], {n: C.AVO_TILT_QUALIFIED[f"{C.AVO_TILT_PACKAGE}/{n}.py"]
                                          for n in ("geometry", "quality", "switch")})
        self.assertEqual((life["installed"], life["restored"], life["exit"], life["entry_module"]),
                         (True, True, None, C.V7_WORKER_ENTRY))

    def test_score_runs_the_v7_worker_with_nothing_installed(self):
        seen = {}
        with tempfile.TemporaryDirectory() as temporary:
            with patch(V4_MAIN, side_effect=lambda argv: seen.update(avo=NA.numeric_avo, core=PC.PrepCore)):
                WE.main(self.args("score", temporary))
            self.assertFalse((Path(temporary) / C.AVO_BINDING_RECORD).exists())
            self.assertFalse((Path(temporary) / C.PREP_BINDING_RECORD).exists())
        self.assertIs(seen["avo"], SW.ORIGINAL_NUMERIC_AVO)
        self.assertIs(seen["core"], REPAIR.ORIGINAL_PREP_CORE)

    def test_worker_exception_survives_and_record_write_failure_never_masks_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / C.AVO_BINDING_RECORD).mkdir()          # the record cannot be created
            with patch(V7_MAIN, side_effect=SystemExit(1)):
                with self.assertRaises(SystemExit) as caught:
                    WE.main(self.args("prep", temporary))
        self.assertEqual(caught.exception.code, 1)
        self.assert_nothing_installed()

    def test_original_error_text_reaches_the_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(V4_MAIN, side_effect=RuntimeError("original producer refusal")):
                with self.assertRaisesRegex(RuntimeError, "original producer refusal"):
                    WE.main(self.args("prep", temporary))
            avo = json.loads((Path(temporary) / C.AVO_BINDING_RECORD).read_text())
        self.assertEqual(avo["exit"], "RuntimeError: original producer refusal")
        self.assertIs(avo["restored"], True)
        self.assert_nothing_installed()


class Retention(unittest.TestCase):
    def test_v8_loader_accepts_the_unchanged_plan_and_refuses_the_v7_section(self):
        from rl_agent.splitfusion_paired_retention_v2 import plan as RP
        plan = C.retention_plan(C.FRAME_CEILING)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            digest = RP.write_plan(output / C.RETENTION_PLAN_FILE, plan)
            reg = dict(transmitted_budget=C.FRAME_CEILING, paired_retention=C.retention_registration(digest))
            self.assertEqual(RT.load_observer(output, reg).plan["plan_id"], C.RETENTION_PLAN_ID)
            with self.assertRaisesRegex(ValueError, "retention section differs"):
                RT.load_observer(output, dict(reg, paired_retention=V7.retention_registration(digest)))
        self.assertEqual({k: v for k, v in C.retention_registration("a" * 64).items() if k != "loader"},
                         {k: v for k, v in V7.retention_registration("a" * 64).items() if k != "loader"})


if __name__ == "__main__":
    unittest.main()
