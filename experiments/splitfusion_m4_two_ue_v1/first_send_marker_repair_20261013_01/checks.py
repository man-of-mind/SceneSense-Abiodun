"""Focused checks for the M4 first-send marker ownership repair (no CARLA/OAI/GPU).

argv: <output RESULT json> <repository root under test>
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))
from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
bind_legacy_namespace()
from rl_agent.splitfusion_m4_two_ue_v1 import composed_checks as CC, host as H
from rl_agent.splitfusion_m4_two_ue_v1.radio import resolve


def main():
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        CC.OUTPUT = Path(tmp)/'composed'; CC.OUTPUT.mkdir()
        names = ('test_real_first_send_marker_per_source', 'test_real_collector_chain_one_pose_per_source_frame')
        outcome = unittest.TextTestRunner(stream=open('/dev/null', 'w')).run(
            unittest.TestSuite([CC.ComposedChecks(n) for n in names]))
        assert outcome.wasSuccessful(), outcome.failures + outcome.errors
        results['real_first_send_and_tick_chain_tests'] = list(names)
        # Failing-before: the previous host wiring (both sources -> pre-written TARGET_STARTED.json).
        old = Path(tmp)/'old_wiring'; old.mkdir()
        with patch.object(H, 'first_send_marker', lambda output, source: Path(output)/'TARGET_STARTED.json'):
            r = CC.real_first_send_pair(resolve(**CC.fixture_binding()), old)
        errors = {f'{k[0]}:{k[1]}': repr(v) for k, v in r['errors'].items() if v is not None}
        assert errors and all('FileExistsError' in v for v in errors.values()), errors
        assert not any(r['submits'].values())
        results['previous_shared_wiring_reproduces_live_error'] = errors
    with Path(sys.argv[1]).open('x') as f:
        json.dump(dict(status='PASS', root=str(ROOT), checks=results), f, indent=2, sort_keys=True)
    print('PASS', results)


if __name__ == '__main__':
    main()
