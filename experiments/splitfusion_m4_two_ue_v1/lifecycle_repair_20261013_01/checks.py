"""Focused checks for the two M4 lifecycle repairs (no CARLA/OAI/GPU).

argv: <output RESULT json> <repository root under test> [<pre-repair world.py for failing-before>]
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

OUT = Path(__file__).resolve().parent
ROOT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(ROOT))
from rl_agent.splitfusion_live_parity_repair_v1.gt_bridge import bind_legacy_namespace
bind_legacy_namespace()
from rl_agent.splitfusion_m4_two_ue_v1 import composed_checks as CC, supervisor as SV
from rl_agent.splitfusion_m4_two_ue_v1.radio import resolve
from data_collection.run_route_b_collection_supervised_v1 import process_group_members


def tick_chain(results):
    suite = unittest.TestSuite([CC.ComposedChecks(n) for n in (
        'test_real_collector_chain_one_pose_per_source_frame', 'test_one_tick_owner_independent_completion_and_drain',
        'test_route_failure_preserves_partial_results_and_both_drains')])
    outcome = unittest.TextTestRunner(stream=open('/dev/null', 'w')).run(suite)
    assert outcome.wasSuccessful(), outcome.failures + outcome.errors
    results['real_collector_chain_and_tick_owner_tests'] = outcome.testsRun
    if len(sys.argv) > 3:  # failing-before: the same real chain under the pre-repair TickOwner
        spec = importlib.util.spec_from_file_location('rl_agent.splitfusion_m4_two_ue_v1._pre_repair_world', sys.argv[3])
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        assert 'wait_for_sensor_frame' in Path(sys.argv[3]).read_text()
        try:
            CC.real_chain_pair(resolve(**CC.fixture_binding()), owner_class=old.TickOwner)
        except ValueError as error:
            assert 'duplicate owned pose' in str(error), error
            results['pre_repair_tick_owner_reproduces_live_error'] = str(error)
        else:
            raise AssertionError('pre-repair TickOwner did not reproduce the duplicate pose')


def supervised(results, scenario, expect_rc, expect_verdict):
    # recover() only acts inside the repository under test (scoped capture rule).
    base = ROOT/'experiments/splitfusion_m4_two_ue_v1'/('lifecycle_probes_' + Path(sys.argv[1]).stem)
    base.mkdir(exist_ok=True)
    capture = base/('supervisor_probe_' + scenario)
    prep = base/('supervisor_probe_' + scenario + '_PREPARATION.json')
    prep.write_text(json.dumps(dict(config=dict(carla_port=2999))))
    started = time.monotonic()
    proc = subprocess.run([sys.executable, '-B', '-c',
        'import sys;sys.path.insert(0,%r);from rl_agent.splitfusion_m4_two_ue_v1 import supervisor as S;'
        'raise SystemExit(S.run(%r,%r,smoke=True,token="unused",child_argv=[sys.executable,"-B",%r,%r,%r,%r]))'
        % (str(ROOT), str(prep), str(capture), str(OUT/'fake_child.py'), str(ROOT), str(capture), scenario)],
        capture_output=True, text=True, timeout=180)
    value = json.loads((capture/'SUPERVISOR.json').read_text())
    carla = json.loads((capture/'CARLA_STARTED.json').read_text())
    assert proc.returncode == expect_rc, (proc.returncode, proc.stderr[-2000:])
    assert value['verdict'] == expect_verdict and value['cleanup']['verified'], value['cleanup']
    assert not process_group_members(carla['pgid'])
    assert not any(r['still_owned'] for r in value['cleanup']['owned_after'])
    assert value['parent_imports_carla'] is False and value['parent_imports_torch'] is False
    results['supervisor_' + scenario] = dict(parent_rc=proc.returncode, child_exit=value['child_exit'],
        verdict=value['verdict'], reasons=value['reasons'], first_failure=(value['first_failure'] or {}).get('phase'),
        entered_not_closed=value['phases_entered_not_closed'], carla=value['cleanup']['carla'],
        owned_signals=[(r['name'], r.get('recovery_signal')) for r in value['cleanup']['recover']['owned_processes']],
        seconds=round(time.monotonic() - started, 1))
    return value


def main():
    results = {}
    tick_chain(results)
    v = supervised(results, 'abort', 134, 'FAIL')
    assert v['child_exit']['signal_name'] == 'SIGABRT' and v['first_failure']['phase'] == 'run.world_and_routes'
    assert 'teardown.processes' in v['phases_entered_not_closed'] and not v['child_records']['RUN_RESULT.json']
    assert ('fake_radio', 'SIGTERM_SENT') in results['supervisor_abort']['owned_signals']
    v = supervised(results, 'clean', 0, 'PASS')
    assert v['child_exit']['returncode'] == 0 and v['cleanup']['carla']['members_after'] == []
    v = supervised(results, 'ignore_term', 0, 'PASS')
    assert v['cleanup']['carla']['sigkill'] == 'sent'
    with Path(sys.argv[1]).open('x') as f:
        json.dump(dict(status='PASS', root=str(ROOT), checks=results), f, indent=2, sort_keys=True)
    print('PASS', sorted(results))


if __name__ == '__main__':
    main()
