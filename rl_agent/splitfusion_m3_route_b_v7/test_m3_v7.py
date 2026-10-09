"""M3.1 v7 binding and prep FREEZE lifetime tests (CPU only, no live service).  From the deployment worktree root:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m unittest -v rl_agent.splitfusion_m3_route_b_v7.test_m3_v7
"""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rl_agent.splitfusion_m3_route_b_v6 import contract as V6
from rl_agent.splitfusion_exact_gt_service_proto_v2 import prep_core as PC
from rl_agent.splitfusion_gt_freeze_lifetime_v1 import repair as REPAIR

from . import contract as C
from . import binding_regression as BR
from . import gc_seams as GS
from . import host as H
from . import retention as RT
from . import service_entry as SE
from . import worker_entry as WE

ROOT = Path(__file__).resolve().parents[2]


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

    def test_v7_rebinds_no_global_of_an_older_m3_module(self):
        self.assertEqual(BR.foreign_rebinding(ROOT / C.PACKAGE), [])

    def test_detector_flags_the_v4_pattern(self):
        self.assertTrue(BR.foreign_rebinding(ROOT / "rl_agent/splitfusion_m3_route_b_v4"))

    def test_child_and_edge_production_scopes_identical_to_v6_and_v3(self):
        parity = BR.composition_parity()
        self.assertTrue(parity["child"] and parity["edge"])
        self.assertEqual(len(parity["child_items"][0]), 14)

    def test_service_entry_copy_differs_from_v3_only_as_registered(self):
        self.assertEqual(BR.service_entry_parity(), dict(identical_class=True, main_differs_only_as_registered=True))

    def test_every_v7_module_reads_the_v7_contract(self):
        from . import entries, runner
        for module in (GS, RT, H, SE, WE, entries, runner):
            self.assertIs(module.C, C)

    def test_launcher_binding_names_existing_modules(self):
        binding = C.launcher_binding()
        modules = {binding["contract_module"], binding["partition"], *binding["host"].values(),
                   *binding["runner"].values(), binding["gt_service"]["trampoline"],
                   binding["gt_service"]["retained_service"], binding["gt_workers"]["trampoline"],
                   binding["gt_workers"]["retained_trampoline"], binding["gt_workers"]["implementation"],
                   binding["child"]["entries"], binding["edge"]["entries"],
                   binding["gt_workers"]["prep_core"].rsplit(".", 1)[0]}
        for name in modules:
            self.assertIsNotNone(importlib.util.find_spec(name), name)
        if (ROOT / C.SHIMS[next(iter(C.SHIMS))]).exists():         # deployment worktree only
            for name in (C.CHILD, C.EDGE):
                self.assertIsNotNone(importlib.util.find_spec(name), name)

    def test_launcher_binding_differs_from_v6_only_in_identity_and_gt_roles(self):
        mine, v6 = C.launcher_binding(), V6.launcher_binding()
        self.assertEqual(sorted(k for k in mine if mine[k] != v6[k]),
                         ["child", "contract_module", "edge", "gt_service", "gt_workers", "host", "launcher_revision",
                          "partition", "runner", "runtime_revision"])
        rename = lambda value: json.loads(json.dumps(value).replace("_v7", "_v6"))
        for key in ("child", "edge", "host", "runner"):
            self.assertEqual(rename(mine[key]), v6[key], key)

    def test_identity_is_new_but_scientific_settings_are_v6(self):
        self.assertNotEqual(C.ID, V6.ID)
        self.assertEqual(C.SCIENTIFIC_RUNTIME_BASIS, V6.SCIENTIFIC_RUNTIME_BASIS)
        self.assertTrue(C.token("candidate").startswith("M3_1_ROUTE_B_GOOD_ALL_V7_CANDIDATE_V2_C3900"))
        for name in ("FRAME_CEILING", "BUDGETS", "GC_POLICY", "SAFETY", "POSE", "MAP_CLASS", "SCHEDULE",
                     "CAPACITY_SUBSTITUTIONS", "POSE_SUBSTITUTIONS", "RETENTION_PLAN_ARGS", "RETENTION_PLANNED_FRAMES",
                     "CHILD_TIMEOUT_S", "SAFETY_TIMEOUT_S", "HOST_ROUTE_LIFECYCLE_S", "CARRIER_TRACE", "ABORT_EXPORT",
                     "RETENTION_PLAN_ID", "TRACE_DIR", "WORKER_MODULE", "SERVICE_MODULE", "PRODUCER_CLASS"):
            self.assertEqual(getattr(C, name), getattr(V6, name), name)
        self.assertEqual(C.limits(C.FRAME_CEILING), V6.limits(V6.FRAME_CEILING))
        self.assertEqual(C.evidence_registration(), V6.evidence_registration())
        self.assertEqual(C.pose_registration(), V6.pose_registration())
        self.assertEqual(C.capacity_registration(), V6.capacity_registration())
        self.assertEqual(C.retention_plan(C.FRAME_CEILING), V6.retention_plan(V6.FRAME_CEILING))

    def test_gc_registration_differs_from_v6_only_in_gt_entries_and_repair(self):
        mine, v6 = C.gc_registration("candidate", 3900), V6.gc_registration("candidate", 3900)
        self.assertEqual(sorted(k for k in set(mine) | set(v6) if mine.get(k) != v6.get(k)),
                         ["implementations", "prep_freeze_lifetime", "seam", "trampolines"])
        self.assertEqual(mine["trampolines"], dict(service=C.SERVICE_ENTRY, worker=C.WORKER_ENTRY))
        self.assertEqual(mine["prep_freeze_lifetime"]["roles"], ["prep"])

    def test_repair_sources_are_the_qualified_ones_and_registered(self):
        for relative, digest in C.PREP_LIFETIME_QUALIFIED.items():
            self.assertEqual(C.sha(ROOT / relative), digest, relative)
        sources = C.exact_sources(ROOT)
        for relative in (*C.PREP_LIFETIME_QUALIFIED, f"{C.PACKAGE}/worker_entry.py", f"{C.PACKAGE}/service_entry.py"):
            self.assertIn(relative, sources)
        from . import entries
        for relative in C.PREP_LIFETIME_QUALIFIED:
            self.assertIn(relative, entries.REQUIRED_SOURCES)

    def test_v6_difference_whitelist_excludes_scientific_fields(self):
        for key in ("exact_gt_limits", "channel", "route", "capacity", "pose_repair", "map_class", "lifecycle",
                    "live_metric_contract", "network_profile", "evidence_additions", "transmitted_budget"):
            self.assertNotIn(key, H.V6_DIFFERENCES_ALLOWED)


class Seam(unittest.TestCase):
    def registration(self, **gt):
        return dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=dict(C.gc_registration("candidate", 3900), **gt))

    def test_seam_starts_the_v7_service_and_refuses_v6(self):
        self.assertEqual(GS.seam_parameters(self.registration())["service_entry"], C.SERVICE_ENTRY)
        with self.assertRaisesRegex(ValueError, "sealed registration"):
            GS.seam_parameters(dict(self.registration(), id=V6.ID))
        with self.assertRaisesRegex(ValueError, "trampolines differ from the v7 contract"):
            GS.seam_parameters(self.registration(trampolines=V6.gc_registration("candidate", 3900)["trampolines"]))
        argv = ["/usr/bin/python3", "-B", "-u", "-m", C.SERVICE_MODULE, "--fd", "5", "--parent", "7"]
        out = GS.rewrite_service_argv(argv, arm="candidate", trace_dir="/t", max_frames=3900, producer="v2")
        self.assertEqual(out[4], C.SERVICE_ENTRY)

    def test_service_rewrites_both_roles_to_the_v7_worker_entry_and_refuses_other_shapes(self):
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

    def test_prep_installs_the_repair_only_while_the_retained_worker_runs(self):
        seen = {}

        def retained(argv):
            seen.update(core=PC.PrepCore, directory=REPAIR.PrepCoreLifetime.failure_directory, argv=list(argv))
            return 0
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args("prep", temporary)
            with patch("rl_agent.splitfusion_m3_route_b_v4.worker_entry.main", side_effect=retained):
                self.assertEqual(WE.main(args), 0)
            record = json.loads((Path(temporary) / C.PREP_BINDING_RECORD).read_text())
        self.assertIs(seen["core"], REPAIR.PrepCoreLifetime)
        self.assertEqual(seen["directory"], temporary)
        self.assertEqual(seen["argv"], args)                       # retained parser receives every argument
        self.assertIs(PC.PrepCore, REPAIR.ORIGINAL_PREP_CORE)
        self.assertIsNone(REPAIR.PrepCoreLifetime.failure_directory)
        self.assertEqual((record["installed"], record["restored"], record["exit"]), (True, True, None))
        self.assertEqual(record["repair_sha256"], C.PREP_LIFETIME_QUALIFIED[f"{C.PREP_LIFETIME_PACKAGE}/repair.py"])

    def test_score_runs_the_retained_worker_with_nothing_installed(self):
        seen = {}
        with tempfile.TemporaryDirectory() as temporary:
            with patch("rl_agent.splitfusion_m3_route_b_v4.worker_entry.main",
                       side_effect=lambda argv: seen.update(core=PC.PrepCore)):
                WE.main(self.args("score", temporary))
            self.assertFalse((Path(temporary) / C.PREP_BINDING_RECORD).exists())
        self.assertIs(seen["core"], REPAIR.ORIGINAL_PREP_CORE)

    def test_worker_exception_survives_and_record_write_failure_never_masks_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / C.PREP_BINDING_RECORD).mkdir()        # the record cannot be created
            with patch("rl_agent.splitfusion_m3_route_b_v4.worker_entry.main",
                       side_effect=SystemExit(1)):
                with self.assertRaises(SystemExit) as caught:
                    WE.main(self.args("prep", temporary))
        self.assertEqual(caught.exception.code, 1)
        self.assertIs(PC.PrepCore, REPAIR.ORIGINAL_PREP_CORE)

    def test_role_parse_reads_only_role_and_trace(self):
        self.assertEqual(WE.role_and_trace(self.args("score", "/x")), ("score", Path("/x")))
        with self.assertRaises(SystemExit):
            WE.role_and_trace(["--role", "edge", "--trace-dir", "/x"])


class Retention(unittest.TestCase):
    def test_v7_loader_accepts_the_unchanged_plan_and_refuses_the_v6_section(self):
        from rl_agent.splitfusion_paired_retention_v2 import plan as RP
        plan = C.retention_plan(C.FRAME_CEILING)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            digest = RP.write_plan(output / C.RETENTION_PLAN_FILE, plan)
            reg = dict(transmitted_budget=C.FRAME_CEILING, paired_retention=C.retention_registration(digest))
            self.assertEqual(RT.load_observer(output, reg).plan["plan_id"], C.RETENTION_PLAN_ID)
            with self.assertRaisesRegex(ValueError, "retention section differs"):
                RT.load_observer(output, dict(reg, paired_retention=V6.retention_registration(digest)))
        self.assertEqual({k: v for k, v in C.retention_registration("a" * 64).items() if k != "loader"},
                         {k: v for k, v in V6.retention_registration("a" * 64).items() if k != "loader"})


if __name__ == "__main__":
    unittest.main()
