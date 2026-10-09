"""Scope integrity/failure regressions without services, CUDA or a live run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import postflight_checks as P


def fixture():
    sources={"source_geometry.py":"a"*64,"target_builder.py":"b"*64}
    settings={"environment":{k:None for k in P.ENV_FIELDS},"calling_thread_affinity":[0,1],
              "torch_threads":1,"torch_interop_threads":24,"opencv_threads":24}
    settings['environment'].update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    state={"schema":"splitfusion.cpu_gt_optimization.scope.v1","sources":sources,
        "retained_settings_unchanged":True,"source_integrity_error":None,
        "new_python_threads":0,"new_queues":0,"new_gpu_work":False,
        "forensic_writes_deferred":False,"live_qualified":False,
        "settings_before":settings,"settings_after":copy.deepcopy(settings),
        "cuda_lifecycle_before":False,"cuda_lifecycle_after":True,
        "cuda_initialized_by_retained_path":True,"prewarm_wall_ns":100,
        "prewarm":{"source":{"source_pins":{"source":{"sha256":"d"*64}},"fixed_grid_proved":True,
                              "state_or_history_created":False},
                   "target":{"sources":{"builder":{"sha256":"e"*64}},"frame_history_advanced":False,
                             "frame_inputs_consumed":0}}}
    scope={"schema":P.SCHEMA,"status":"COMPLETE","returncode":0,"error":None,
        "optimization_sources":sources,"prospective_record_sha256":"c"*64,
        "worker_drained":True,"ownership_manifest_complete":True,"optimization":state}
    ownership={"schema":"splitfusion.parity_repair.exact_source_ownership.v2","complete":True,
        "errors":[],"worker_drained":True,"worker_alive":False,"pending_at_close":0,"owned_bytes_at_close":0}
    return scope,ownership,sources


class PostflightTests(unittest.TestCase):
    def check(self,change=None,*,pass_expected=False):
        scope,ownership,sources=fixture()
        if change: change(scope,ownership)
        with tempfile.TemporaryDirectory() as name:
            cell=Path(name);(cell/'source_ownership').mkdir()
            (cell/'GT_OPTIMIZATION_SCOPE.json').write_text(json.dumps(scope))
            (cell/'source_ownership/MANIFEST.json').write_text(json.dumps(ownership))
            if pass_expected:
                result=P.verify_optimization_scope(cell,expected_sources=sources,expected_prospective_sha256='c'*64)
                self.assertTrue(result['passed']);self.assertFalse(result['latency_qualification_claim'])
            else:
                with self.assertRaises(ValueError):
                    P.verify_optimization_scope(cell,expected_sources=sources,expected_prospective_sha256='c'*64)

    def test_normal_cuda_lifecycle_not_config_change(self): self.check(pass_expected=True)

    def test_no_live_initialization_is_descriptive_not_timing_gate(self):
        self.check(lambda s,o:s['optimization'].update(cuda_lifecycle_after=False,
            cuda_initialized_by_retained_path=False),pass_expected=True)

    def test_binding_and_exit_failures(self):
        for change in (lambda s,o:s.update(prospective_record_sha256='d'*64),
                       lambda s,o:s.update(status='FAILED'),lambda s,o:s.update(returncode=1),
                       lambda s,o:s.update(error='GT failed'),lambda s,o:s.update(worker_drained=False),
                       lambda s,o:s['optimization'].update(sources={}),
                       lambda s,o:s['optimization'].update(source_integrity_error='changed')):
            self.check(change)

    def test_scope_claim_cannot_hide_retained_setting_change(self):
        for key,value in (('torch_threads',2),('opencv_threads',1),('calling_thread_affinity',[2,3])):
            self.check(lambda s,o,k=key,v=value:s['optimization']['settings_after'].update({k:v}))
        self.check(lambda s,o:s['optimization']['settings_before']['environment'].update(OMP_NUM_THREADS='24'))

    def test_no_concurrency_gpu_deferred_archival_widening(self):
        for key,value in (('new_python_threads',1),('new_queues',1),('new_gpu_work',True),
                          ('forensic_writes_deferred',True),('live_qualified',True)):
            self.check(lambda s,o,k=key,v=value:s['optimization'].update({k:v}))

    def test_actual_ownership_drain_required(self):
        for key,value in (('complete',False),('worker_drained',False),('worker_alive',True),
                          ('pending_at_close',1),('owned_bytes_at_close',100),('errors',['lost'])):
            self.check(lambda s,o,k=key,v=value:o.update({k:v}))

    def test_missing_invariant_or_invalid_cuda_telemetry_rejected(self):
        self.check(lambda s,o:s['optimization']['settings_after'].pop('opencv_threads'))
        self.check(lambda s,o:s['optimization'].pop('cuda_lifecycle_before'))
        self.check(lambda s,o:s['optimization'].update(cuda_lifecycle_before=True,cuda_lifecycle_after=False))

    def test_prewarm_may_not_consume_history(self):
        self.check(lambda s,o:s['optimization']['prewarm']['target'].update(frame_inputs_consumed=1))
        self.check(lambda s,o:s['optimization']['prewarm']['source'].update(fixed_grid_proved=False))

    def test_symbolic_scope_refused(self):
        scope,ownership,sources=fixture()
        with tempfile.TemporaryDirectory() as name:
            cell=Path(name);target=cell/'real.json';target.write_text(json.dumps(scope))
            (cell/'GT_OPTIMIZATION_SCOPE.json').symlink_to(target)
            with self.assertRaises(ValueError):
                P.verify_optimization_scope(cell,expected_sources=sources,expected_prospective_sha256='c'*64)


if __name__=='__main__': unittest.main()
