"""Fresh CPU subprocess proofs; no worker benchmark, simulator or GPU work."""
import gc
import json
import os
from pathlib import Path
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}


class GCPolicyTests(unittest.TestCase):
    def child(self, body, *, environment=None):
        before = (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count(),
                  tuple(id(value) for value in gc.callbacks))
        env = {**os.environ, **ENVIRONMENT, **(environment or {})}
        env.pop("PYTHONPATH", None)
        prelude = """
import gc, json, os, weakref
from unittest import mock
from rl_agent.splitfusion_live_gt_optimization_v8 import gc_policy as G
assert gc.get_freeze_count() == 0
assert gc.isenabled() and gc.get_threshold() == G.THRESHOLDS
"""
        completed = subprocess.run(["/usr/bin/python3", "-c", textwrap.dedent(prelude)
            + textwrap.dedent(body)], cwd=ROOT, env=env, text=True,
            capture_output=True, timeout=30.)
        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
        self.assertEqual(completed.stderr, "")
        after = (gc.isenabled(), gc.get_threshold(), gc.get_freeze_count(),
                 tuple(id(value) for value in gc.callbacks))
        self.assertEqual(after, before, "child policy changed the parent GC configuration")
        return json.loads(completed.stdout)

    def test_both_arms_collect_once_only_treatment_freezes_and_close_is_idempotent(self):
        for mode in ("control", "freeze"):
            with self.subTest(mode=mode):
                result = self.child(f"""
mode = {mode!r}
policy = G.GCPolicy(mode)
initial = policy.snapshot(include_events=True)
assert initial['processed_at_freeze'] == 0 and initial['startup_freeze_count'] == 0
assert initial['startup_collection']['generation'] == 2
assert initial['startup_collection']['stats_delta'][2]['collections'] == 1
assert initial['startup_collection']['wall_ns'] > 0
assert initial['startup_collection']['thread_cpu_ns'] >= 0
assert initial['freeze_calls'] == int(mode == 'freeze')
assert (initial['freeze_count'] > 0) == (mode == 'freeze')
assert initial['enabled'] and initial['thresholds'] == [700, 10, 10]
assert initial['worker_pid'] == os.getpid()
assert initial['startup_rss_bytes'] > 0 and initial['rss_bytes'] > 0
assert len(initial['events']) >= 1
assert any(e['phase'] == 'startup' and e['generation'] == 2 for e in initial['events'])
policy.set_phase('processing', 0)
policy.set_phase('idle', 1)
assert 'events' not in policy.snapshot()
final = policy.close()
again = policy.close()
assert final['closed'] and final['settings_restored']
assert final['unfreeze_calls'] == int(mode == 'freeze')
assert again['unfreeze_calls'] == final['unfreeze_calls']
assert final['freeze_count'] == 0 and not final['callback_installed']
assert final['errors'] == [] and final['dropped'] == 0 and final['callback_errors'] == 0
assert final['processed'] == 1 and not final['per_request_manual_collection']
assert final['original_generation_membership_restored'] is False
print(json.dumps(final))
""")
                self.assertEqual(result["mode"], mode)

    def test_new_cycles_collect_in_both_arms_old_graph_only_waits_in_treatment(self):
        for mode in ("control", "freeze"):
            with self.subTest(mode=mode):
                self.child(f"""
class Node:
    pass
old = Node(); old.peer = old; old_ref = weakref.ref(old)
policy = G.GCPolicy({mode!r})
policy.set_phase('request', 0)
new = Node(); new.peer = new; new_ref = weakref.ref(new)
del old, new
gc.collect()
assert new_ref() is None
assert (old_ref() is not None) == ({mode!r} == 'freeze')
assert policy.snapshot()['enabled']
policy.close()
gc.collect()
assert old_ref() is None and gc.get_freeze_count() == 0
print(json.dumps(policy.snapshot(include_events=True)))
""")

    def test_preexisting_frozen_generation_rejected_without_stealing_it(self):
        self.child("""
gc.collect(); gc.freeze(); count = gc.get_freeze_count()
assert count > 0
try:
    G.GCPolicy('freeze')
except G.GCPolicyError as error:
    assert 'pre-existing frozen' in str(error)
else:
    raise AssertionError('foreign frozen graph accepted')
assert gc.get_freeze_count() > 0
gc.unfreeze()
print(json.dumps({'rejected': True}))
""")

    def test_automatic_collector_still_collects_new_request_cycles(self):
        self.child("""
class Node:
    pass
policy = G.GCPolicy('freeze')
before = gc.get_stats()
references = []
policy.set_phase('request', 0)
for _ in range(5000):
    item = Node(); item.peer = item
    references.append(weakref.ref(item))
del item
after = gc.get_stats()
assert after[0]['collections'] > before[0]['collections']
assert sum(reference() is None for reference in references) > 4000
assert sum(row['collected'] for row in after) > sum(row['collected'] for row in before)
assert policy.snapshot()['enabled'] and gc.get_threshold() == G.THRESHOLDS
policy.close()
print(json.dumps(policy.snapshot(include_events=True)))
""")

    def test_disabled_or_retuned_gc_rejected_without_mutating_foreign_configuration(self):
        self.child("""
gc.disable()
try:
    G.GCPolicy('control')
except G.GCPolicyError:
    pass
else:
    raise AssertionError('disabled collector accepted')
assert not gc.isenabled()
gc.enable(); gc.set_threshold(701, 10, 10)
try:
    G.GCPolicy('freeze')
except G.GCPolicyError:
    pass
else:
    raise AssertionError('foreign thresholds accepted')
assert gc.get_threshold() == (701, 10, 10)
gc.set_threshold(*G.THRESHOLDS)
print(json.dumps({'rejected': True}))
""")

    def test_cpu_environment_required_before_any_collection(self):
        for field, value in (("OMP_NUM_THREADS", "24"), ("MKL_NUM_THREADS", "4"),
                             ("OPENBLAS_NUM_THREADS", "2"), ("CUDA_VISIBLE_DEVICES", "0")):
            with self.subTest(field=field):
                self.child("""
with mock.patch.object(gc, 'collect', side_effect=AssertionError('must reject before collection')):
    try:
        G.GCPolicy('freeze')
    except G.GCPolicyError as error:
        assert 'environment' in str(error)
    else:
        raise AssertionError('wrong environment accepted')
assert gc.get_freeze_count() == 0
print(json.dumps({'rejected': True}))
""", environment={field: value})

    def test_partial_startup_failure_restores_owned_freeze_and_callback(self):
        self.child("""
before = tuple(gc.callbacks)
real_freeze = gc.freeze
def fail_after_freezing():
    real_freeze()
    raise RuntimeError('injected after freeze')
with mock.patch.object(gc, 'freeze', side_effect=fail_after_freezing):
    try:
        G.GCPolicy('freeze')
    except RuntimeError as error:
        assert 'injected after freeze' in str(error)
    else:
        raise AssertionError('injected initialization failure accepted')
assert tuple(gc.callbacks) == before
assert gc.get_freeze_count() == 0 and gc.isenabled()
print(json.dumps({'restored': True}))
""")

    def test_failure_cleanup_and_external_callbacks_preserved(self):
        self.child("""
observed = []
def existing(phase, info):
    if phase == 'stop': observed.append(info['generation'])
gc.callbacks.append(existing)
before = tuple(gc.callbacks)
policy = G.GCPolicy('freeze')
try:
    policy.set_phase('request', 0)
    raise RuntimeError('simulated native/archive drain failure')
except RuntimeError as error:
    assert str(error) == 'simulated native/archive drain failure'
finally:
    result = policy.close()
assert result['settings_restored'] and result['unfreeze_calls'] == 1
assert tuple(gc.callbacks) == before and observed
gc.callbacks.remove(existing)
print(json.dumps(result))
""")

    def test_event_bound_is_reported_and_cleanup_still_occurs(self):
        self.child("""
G.MAX_EVENTS = 2
policy = G.GCPolicy('freeze')
for _ in range(5): gc.collect(0)
snapshot = policy.snapshot(include_events=True)
assert len(snapshot['events']) == 2 and snapshot['dropped'] >= 4
try:
    policy.set_phase('request', 0)
except G.GCPolicyError:
    pass
else:
    raise AssertionError('dropped callback evidence accepted')
try:
    policy.close()
except G.GCPolicyError as error:
    assert 'observation' in str(error)
else:
    raise AssertionError('incomplete GC observation accepted')
assert gc.get_freeze_count() == 0 and policy.closed
assert policy.snapshot()['settings_restored']
print(json.dumps(policy.snapshot()))
""")

    def test_drift_fails_but_restores_registered_enabled_thresholds(self):
        self.child("""
policy = G.GCPolicy('freeze')
gc.disable(); gc.set_threshold(800, 20, 20)
try:
    policy.close()
except G.GCPolicyError as error:
    assert 'settings drifted' in str(error)
else:
    raise AssertionError('settings drift accepted')
assert gc.isenabled() and gc.get_threshold() == G.THRESHOLDS
assert gc.get_freeze_count() == 0
print(json.dumps(policy.snapshot()))
""")

    def test_phase_counter_and_postclose_calls_are_bounded(self):
        self.child("""
policy = G.GCPolicy('control')
policy.set_phase('idle', 2)
for phase, processed in (('', 2), ('x'*81, 2), ('request', 1), ('request', True), ('request', 1001)):
    try:
        policy.set_phase(phase, processed)
    except G.GCPolicyError:
        pass
    else:
        raise AssertionError('bad phase/counter accepted')
policy.close()
try:
    policy.set_phase('request', 2)
except G.GCPolicyError:
    pass
else:
    raise AssertionError('closed policy accepted a request')
print(json.dumps(policy.snapshot()))
""")

    def test_callback_fault_is_not_raised_into_the_collector_but_fails_boundary(self):
        self.child("""
policy = G.GCPolicy('freeze')
policy._observe_collection('stop', {'generation': 0, 'collected': 0, 'uncollectable': 0})
assert policy.snapshot()['callback_errors'] == 1
try:
    policy.set_phase('request', 0)
except G.GCPolicyError:
    pass
else:
    raise AssertionError('callback fault accepted')
try:
    policy.close()
except G.GCPolicyError:
    pass
else:
    raise AssertionError('callback fault not reported at close')
assert gc.get_freeze_count() == 0
print(json.dumps(policy.snapshot()))
""")


if __name__ == "__main__":
    unittest.main()
