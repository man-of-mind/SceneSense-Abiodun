"""Source binding, retained science and add-only staging checks."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from . import contract as C, deploy as D
from rl_agent.splitfusion_live_parity_runtime_v7 import contract as V7
from rl_agent.splitfusion_live_gt_handoff_v1 import contract as H

ROOT=Path(__file__).resolve().parents[2]


class ContractTests(unittest.TestCase):
    def test_final_offline_handoff_sources_and_binding_exact(self):
        self.assertEqual(C.bootstrap_inventory(ROOT),H.sources())
        self.assertEqual(C.handoff_binding(ROOT),H.binding())
        self.assertEqual(C.worker_inventory(ROOT),V7.worker_inventory(ROOT))

    def test_retained_science_and_outer_optimization_exact(self):
        self.assertEqual(C.BUDGET,100)
        self.assertEqual(C.metrics(),V7.metrics())
        self.assertEqual(C.INPUT_SPEC,V7.INPUT_SPEC)
        self.assertEqual(C.IMAGE_ID,V7.IMAGE_ID)
        self.assertEqual(C.optimization_inventory(ROOT),V7.optimization_inventory(ROOT))
        self.assertEqual(C.WORKER_TIMEOUT_S,30.)
        self.assertNotEqual(C.ID,V7.ID)
        self.assertNotEqual(C.TOKEN,V7.TOKEN)
        self.assertTrue(C.CHILD.endswith('_v8'))
        self.assertTrue(C.EDGE.endswith('_v8'))

    def test_changed_missing_extra_and_symlink_bootstrap_refused(self):
        sources=C.bootstrap_inventory(ROOT)
        for mutation in ('change','missing','extra','symlink'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                for name in sources:
                    target=root/name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(ROOT/name,target)
                target=root/'rl_agent/splitfusion_live_gt_handoff_v1/broker.py'
                if mutation=='change':
                    target.write_bytes(target.read_bytes()+b'\n# test mutation\n')
                elif mutation=='missing':
                    target.unlink()
                elif mutation=='extra':
                    (target.parent/'foreign.py').write_text('EXTRA=True\n')
                else:
                    target.unlink()
                    target.symlink_to(ROOT/'rl_agent/splitfusion_live_gt_handoff_v1/broker.py')
                with self.assertRaises(ValueError):
                    C.bootstrap_inventory(root)

    def test_add_only_dry_deployment_preserves_previous_and_does_not_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            worktree=root/'runtime'; worktree.mkdir()
            retained=worktree/'retained.py'; retained.write_text('RETAINED=True\n')
            prior=root/'prior.json'
            prior.write_text(json.dumps(dict(schema='splitfusion.parity_runtime.prospective.v7',
                budget=100,worktree=str(worktree),runtime_sources={'retained.py':C.sha(retained)})))
            with patch.object(D,'PREVIOUS',prior):
                result=D.deploy(worktree,root/'report.json',execute=False)
                self.assertFalse(result['executed'])
                self.assertEqual(result['prior_source_files_unchanged'],1)
                self.assertEqual(list(worktree.iterdir()),[retained])
                self.assertIn(C.PACKAGE+'/entries.py',result['files'])
                self.assertIn('rl_agent/splitfusion_live_gt_handoff_v1/edge.py',result['files'])
                for module in (C.CHILD,C.EDGE):
                    self.assertIn(module.replace('.','/')+'.py',result['files'])
                retained.write_text('DRIFT=True\n')
                with self.assertRaises(ValueError):
                    D.deploy(worktree,root/'refused.json',execute=True)
                self.assertFalse((root/'refused.json').exists())

    def test_validation_suite_set_is_bound_by_host_gate(self):
        from .validate import GROUPS
        from .host_capture import OFFLINE_SUITES
        self.assertEqual(set(GROUPS),OFFLINE_SUITES)


if __name__=='__main__':
    unittest.main()
