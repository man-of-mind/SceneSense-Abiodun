"""M3.1 v6 binding tests (CPU only, no live service).  From the deployment worktree root:

  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= /usr/bin/python3 -B -m unittest -v rl_agent.splitfusion_m3_route_b_v6.test_m3_v6
"""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_m3_route_b_v3 import contract as V3
from rl_agent.splitfusion_m3_route_b_v4 import contract as V4

from . import contract as C
from . import binding_regression as BR
from . import gc_seams as GS
from . import host as H
from . import retention as RT

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
            path = ROOT / (module.replace(".", "/") + ".py")
            used = _c_names(path, scopes.get(module))
            self.assertTrue(used <= set(names), (module, sorted(used - set(names))))

    def test_v6_rebinds_no_global_of_an_older_m3_module(self):
        self.assertEqual(BR.foreign_rebinding(ROOT / C.PACKAGE), [])

    def test_detector_flags_the_v4_pattern(self):
        self.assertTrue(BR.foreign_rebinding(ROOT / "rl_agent/splitfusion_m3_route_b_v4"))

    def test_child_and_edge_production_scopes_identical_to_v3(self):
        parity = BR.composition_parity()
        self.assertTrue(parity["child"] and parity["edge"])
        self.assertEqual(len(parity["child_items"][0]), 14)

    def test_every_v6_module_reads_the_v6_contract(self):
        for module in (GS, RT, H):
            self.assertIs(module.C, C)
        from . import entries, runner
        self.assertIs(entries.C, C)
        self.assertIs(runner.C, C)

    def test_module_identity_reports_the_import_name_not_main(self):
        import sys
        import types
        self.assertEqual(C.module_identity(H.verify_prospective), C.HOST)
        fake = types.ModuleType("__main__")
        fake.__spec__ = types.SimpleNamespace(name=C.RUNNER)
        saved = sys.modules["__main__"]
        sys.modules["__main__"] = fake
        try:
            function = types.FunctionType(compile("def f(): pass", "<t>", "exec").co_consts[0], {"__name__": "__main__"})
            self.assertEqual(function.__module__, "__main__")
            self.assertEqual(C.module_identity(function), C.RUNNER)
        finally:
            sys.modules["__main__"] = saved

    def test_launcher_binding_names_existing_modules(self):
        binding = C.launcher_binding()
        modules = {binding["contract_module"], binding["partition"], *binding["host"].values(),
                   *(v for v in binding["runner"].values()), binding["gt_service"]["trampoline"],
                   binding["gt_service"]["implementation"], binding["gt_workers"]["trampoline"],
                   binding["gt_workers"]["implementation"], binding["child"]["entries"], binding["edge"]["entries"]}
        for name in modules:
            self.assertIsNotNone(importlib.util.find_spec(name), name)
        if (ROOT / C.SHIMS[next(iter(C.SHIMS))]).exists():         # deployment worktree only
            for name in (C.CHILD, C.EDGE):
                self.assertIsNotNone(importlib.util.find_spec(name), name)

    def test_identity_is_new_but_scientific_basis_is_v4(self):
        self.assertNotEqual(C.ID, V4.ID)
        self.assertEqual(C.SCIENTIFIC_RUNTIME_BASIS, V4.ID)
        self.assertNotEqual(C.token("candidate"), V4.token("candidate"))
        self.assertTrue(C.token("candidate").startswith("M3_1_ROUTE_B_GOOD_ALL_V6_CANDIDATE_V2_C3900"))
        for name in ("FRAME_CEILING", "BUDGETS", "GC_POLICY", "SAFETY", "POSE", "MAP_CLASS", "SCHEDULE",
                     "CAPACITY_SUBSTITUTIONS", "POSE_SUBSTITUTIONS", "RETENTION_PLAN_ARGS", "RETENTION_PLANNED_FRAMES",
                     "CHILD_TIMEOUT_S", "SAFETY_TIMEOUT_S", "HOST_ROUTE_LIFECYCLE_S", "CARRIER_TRACE", "ABORT_EXPORT"):
            self.assertEqual(getattr(C, name), getattr(V4, name), name)
        self.assertEqual(C.limits(C.FRAME_CEILING), V4.limits(V4.FRAME_CEILING))
        self.assertEqual(C.evidence_registration(), V4.evidence_registration())
        self.assertEqual(C.pose_registration(), V4.pose_registration())
        self.assertEqual(C.capacity_registration(), V4.capacity_registration())


class Seam(unittest.TestCase):
    def registration(self, **gt):
        return dict(id=C.ID, transmitted_budget=C.FRAME_CEILING, gt_gc=dict(C.gc_registration("candidate", 3900), **gt))

    def test_registered_trampolines_are_what_runs(self):
        reg = C.gc_registration("candidate", 3900)
        self.assertEqual(reg["trampolines"], dict(service=V4.SERVICE_ENTRY, worker=V4.WORKER_ENTRY))
        self.assertEqual({k: v for k, v in reg.items() if k not in ("trampolines", "implementations", "seam")},
                         {k: v for k, v in V3.gc_registration("candidate", 3900).items() if k != "trampolines"})

    def test_seam_parameters_accept_and_reject(self):
        self.assertEqual(GS.seam_parameters(self.registration())["service_entry"], V4.SERVICE_ENTRY)
        with self.assertRaisesRegex(ValueError, "sealed registration"):
            GS.seam_parameters(dict(self.registration(), id=V4.ID))
        with self.assertRaisesRegex(ValueError, "trampolines differ"):
            GS.seam_parameters(self.registration(trampolines=V3.gc_registration("candidate", 3900)["trampolines"]))
        with self.assertRaisesRegex(ValueError, "sealed registration"):
            GS.seam_parameters(self.registration(arm="other"))

    def test_rewrite_starts_the_v4_service_trampoline(self):
        argv = ["/usr/bin/python3", "-B", "-u", "-m", C.SERVICE_MODULE, "--fd", "5", "--parent", "7"]
        out = GS.rewrite_service_argv(argv, arm="candidate", trace_dir="/t", max_frames=3900, producer="v2")
        self.assertEqual(out[4], V4.SERVICE_ENTRY)
        self.assertEqual(out[-8:], ["--gc-policy", "candidate", "--trace-dir", "/t", "--max-frames", "3900",
                                    "--producer", "v2"])
        self.assertIsNone(GS.rewrite_service_argv(["python3", "-m", "other"], arm="candidate", trace_dir="/t",
                                                  max_frames=3900, producer="v2"))


class Retention(unittest.TestCase):
    def test_v6_loader_accepts_the_unchanged_v4_plan_v3_loader_refuses_it(self):
        from rl_agent.splitfusion_paired_retention_v2 import plan as RP
        from rl_agent.splitfusion_m3_route_b_v3 import retention as V3R
        plan = C.retention_plan(C.FRAME_CEILING)
        self.assertEqual(plan, V4.retention_plan(V4.FRAME_CEILING))
        self.assertEqual(plan["plan_id"], C.RETENTION_PLAN_ID)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            digest = RP.write_plan(output / C.RETENTION_PLAN_FILE, plan)
            reg = dict(transmitted_budget=C.FRAME_CEILING, paired_retention=C.retention_registration(digest))
            observer = RT.load_observer(output, reg)
            self.assertEqual(observer.plan["plan_id"], C.RETENTION_PLAN_ID)
            self.assertIsInstance(observer, V3R.AttemptRetentionObserver)
            legacy = dict(reg, paired_retention=V3.retention_registration(digest))
            with self.assertRaisesRegex(ValueError, "not the registered rule"):
                V3R.load_observer(output, legacy)
            with self.assertRaisesRegex(ValueError, "retention section differs"):
                RT.load_observer(output, legacy)

    def test_registered_plan_id_is_the_plan_files_id(self):
        value = C.retention_registration("a" * 64)
        self.assertEqual(value["plan_id"], C.retention_plan(C.FRAME_CEILING)["plan_id"])
        self.assertEqual({k: v for k, v in value.items() if k not in ("plan_id", "loader")},
                         {k: v for k, v in V3.retention_registration("a" * 64).items() if k != "plan_id"})


class EdgeMount(unittest.TestCase):
    def test_compose_text_is_the_v3_publisher_format(self):
        import inspect
        from rl_agent.splitfusion_m3_route_b_v3 import host as V3H
        source = inspect.getsource(V3H.prepare)
        self.assertIn('f"      - {output}:/partition_output:rw\\n      - {retained}:/cpu_partition_code:ro\\n"', source)
        self.assertEqual(H.compose_text("/o", "/r"), "services:\n  oai-perception-rx:\n    volumes:\n"
                         "      - /o:/partition_output:rw\n      - /r:/cpu_partition_code:ro\n")

    def test_v5_root_mounted_the_v4_root(self):
        if not BR.V5_ROOT.is_dir():
            self.skipTest("V5 evidence lives in the deployment worktree")
        text = (BR.V5_ROOT / "compose_partition.yaml").read_text()
        record = json.loads((BR.V5_ROOT / "PROSPECTIVE.json").read_bytes())
        self.assertIn(f"{BR.V4_ROOT}:/partition_output:rw", text)
        self.assertNotEqual(text, H.compose_text(BR.V5_ROOT, record["retained_code"]))


if __name__ == "__main__":
    unittest.main()
