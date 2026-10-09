"""Independent V7 source-proof, retained-semantics and add-only staging gates."""
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import contract as C, deploy as D
from rl_agent.splitfusion_live_parity_runtime_v6 import contract as V6
from rl_agent.splitfusion_live_gt_optimization_v4 import worker as W

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / C.PACKAGE / 'worker_binding.json'


@contextmanager
def mirrored_worker():
    with tempfile.TemporaryDirectory(prefix='v7-worker-binding-') as temporary:
        root = Path(temporary)
        binding = json.loads(BINDING.read_bytes())
        for name in binding['worker_sources']:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        target = root / C.PACKAGE / 'worker_binding.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(BINDING, target)
        yield root, binding, target


class ContractTests(unittest.TestCase):
    def test_actual_worker_sources_match_sealed_binding_and_engine_inventory(self):
        binding = json.loads(BINDING.read_bytes())
        self.assertEqual(binding['schema'], 'splitfusion.cpu_gt_worker.proof_binding.v7')
        self.assertEqual(binding['exact_replays'], 288)
        self.assertIs(binding['live_qualified'], False)
        self.assertEqual(C.worker_inventory(ROOT), binding['worker_sources'])
        self.assertEqual(C.worker_inventory(ROOT), W.sources())
        for package in C.WORKER_PACKAGES:
            self.assertIn(package + '/__init__.py', binding['worker_sources'])
        for name in ('runtime.py', 'worker.py', 'snapshot.py', 'memfd_transport.py'):
            self.assertIn('rl_agent/splitfusion_live_gt_optimization_v4/' + name, binding['worker_sources'])

    def test_retained_metrics_optimization_pins_actor_family_and_budget_unchanged(self):
        self.assertEqual(C.BUDGET, 100)
        self.assertEqual(C.WORKER_TIMEOUT_S, 30.)
        self.assertEqual(C.metrics(), V6.metrics())
        self.assertEqual(C.OPTIMIZATION_PACKAGE, V6.OPTIMIZATION_PACKAGE)
        self.assertEqual(C.OPTIMIZATION_SOURCES, V6.OPTIMIZATION_SOURCES)
        self.assertEqual(C.optimization_inventory(ROOT), V6.optimization_inventory(ROOT))
        self.assertEqual(C.INPUT_SPEC, V6.INPUT_SPEC)
        self.assertEqual(C.IMAGE_ID, V6.IMAGE_ID)
        self.assertEqual(C.ID, 'RUN5_CPU_GT_WORKER_100_ENGINEERING_V7')
        self.assertTrue(C.CHILD.endswith('child_v1_run5_parity_v7'))
        self.assertTrue(C.EDGE.endswith('edge_profiled_v1_run5_parity_v7'))
        self.assertNotEqual(C.TOKEN, V6.TOKEN)

    def test_changed_missing_extra_and_symbolic_worker_sources_are_refused(self):
        name = 'rl_agent/splitfusion_live_gt_optimization_v4/worker.py'
        for mutation in ('changed', 'missing', 'extra', 'symlink'):
            with self.subTest(mutation=mutation), mirrored_worker() as (root, _binding, _path):
                target = root / name
                if mutation == 'changed':
                    target.write_bytes(target.read_bytes() + b'\n# altered in isolated test fixture\n')
                elif mutation == 'missing':
                    target.unlink()
                elif mutation == 'extra':
                    (target.parent / 'unexpected_source.py').write_text('EXTRA = True\n')
                else:
                    target.unlink()
                    target.symlink_to(ROOT / name)
                with self.assertRaises(ValueError):
                    C.worker_inventory(root)

    def test_changed_binding_schema_claim_or_proof_digest_is_refused(self):
        mutations = (
            ('schema', 'foreign.proof'), ('exact_replays', 287), ('exact_replays', 288.0),
            ('live_qualified', True), ('offline_output_manifest_sha256', '0' * 64),
            ('offline_report_sha256', '0' * 64), ('worker_sources', {}))
        for key, value in mutations:
            with self.subTest(key=key, value=value), mirrored_worker() as (root, binding, path):
                binding[key] = value
                path.write_text(json.dumps(binding, sort_keys=True))
                with self.assertRaises(ValueError):
                    C.worker_inventory(root)

    def test_deploy_dry_covers_worker_binding_tests_and_both_shims_without_staging(self):
        with tempfile.TemporaryDirectory(prefix='v7-deploy-dry-') as temporary:
            root = Path(temporary)
            worktree = root / 'worktree'; worktree.mkdir()
            retained = worktree / 'retained.py'; retained.write_text('RETAINED = True\n')
            previous, replay, report = (root / name for name in ('previous.json', 'replay.json', 'report.json'))
            previous.write_text(json.dumps(dict(schema='splitfusion.parity_runtime.prospective.v6',
                budget=100, worktree=str(worktree), runtime_sources={'retained.py': C.sha(retained)})))
            replay.write_text(json.dumps({'sources': {}}))
            before = retained.read_bytes()
            with patch.object(D, 'PREVIOUS', previous), patch.object(D, 'REPLAY_REPORT', replay):
                result = D.deploy(worktree, report, execute=False)
            self.assertIs(result['executed'], False)
            self.assertEqual(result['prior_source_files_unchanged'], 1)
            self.assertEqual(retained.read_bytes(), before)
            self.assertEqual([item.name for item in worktree.iterdir()], ['retained.py'])
            expected = [C.PACKAGE + '/worker_binding.json',
                'rl_agent/splitfusion_live_gt_optimization_v3/test_worker.py',
                'rl_agent/splitfusion_live_gt_optimization_v4/test_worker.py',
                'rl_agent/splitfusion_live_gt_optimization_v4/runtime.py',
                'rl_agent/splitfusion_live_gt_optimization_v4/memfd_transport.py',
                *(module.replace('.', '/') + '.py' for module in (C.CHILD, C.EDGE))]
            for name in expected:
                self.assertIn(name, result['files'])
            for module in (C.CHILD, C.EDGE):
                shim = ROOT / C.PACKAGE / 'runtime_shims' / (module.rsplit('.', 1)[1] + '.py')
                self.assertEqual(result['files'][module.replace('.', '/') + '.py'], C.sha(shim))
            self.assertEqual(json.loads(report.read_bytes()), result)


if __name__ == '__main__':
    unittest.main()
