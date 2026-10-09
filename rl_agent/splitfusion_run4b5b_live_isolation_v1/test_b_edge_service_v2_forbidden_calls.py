"""Decision-A adversarial proof: imported quality code is never called."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ForbiddenCallTest(unittest.TestCase):
    def test_offline_b_processing_never_calls_evaluator_gt_or_qperc(self):
        root = Path(__file__).resolve().parents[2]
        script = r'''import importlib, sys, tempfile, types
from pathlib import Path
from rl_agent.splitfusion_run4b5b_live_isolation_v1 import b_edge_service_v2 as S
from rl_agent.splitfusion_run4b5b_live_isolation_v1.test_b_edge_process_v1 import request
symbols=S.load_qualified_symbols()
phase6=importlib.import_module(S.PROVEN_EDGE_MODULE)
scoring=importlib.import_module('rl_agent.splitfusion_quality_feedback_probe_v1.scoring')
quality=importlib.import_module(
    'rl_agent.splitfusion_hybrid_sac_v1.offline_quality_grid.quality')
calls=[]
def boom(*args, **kwargs):
    calls.append((args, kwargs))
    raise AssertionError('forbidden evaluator/GT/Qperc call')
phase6.Run4EvaluatorV2=boom
phase6.W.load_run4_quality_spec=boom
phase6.W.quality_feedback=boom
for name in ('segmentation_quality_columns','score_segmentation',
             'score_localization','score_serial','score_concurrent'):
    setattr(scoring,name,boom)
for name in ('load_reward_spec','evaluate_exact_quality'):
    setattr(quality,name,boom)
gt_name='rl_agent.splitfusion_quality_feedback_probe_v1.gt_evidence'
sys.modules[gt_name]=types.SimpleNamespace(read_ground_truth=boom)
with tempfile.TemporaryDirectory() as tmp:
    result=S.offline_fake(request()[0],root=Path(tmp)/'fake')
assert result['events'][0]=='ACK'
assert result['gt_qperc_reward_used'] is False
assert calls==[]
assert symbols.processor_type is not None and symbols.compute is not None
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root)
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=root, env=env,
            capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
