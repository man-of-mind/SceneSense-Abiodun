"""Exact-output and fail-closed CPU tests; never launches a simulator."""
import collections
import copy
import json
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_parity_repair_v1.test_source_geometry import SourceGeometryTests
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A
from . import source_geometry as O


class OptimizationTests(unittest.TestCase):
    def test_recursive_canonical_values_exact(self):
        rng=np.random.default_rng(3)
        values=[None, True, 3, -0., 'x', np.int64(2), np.float64(2.),
                MappingProxyType({'a':(1,2)}), collections.OrderedDict([('b',3),('a',2)]),
                {1:'str-key', 'array':[{(1,2),(3,4)}, frozenset(('a','b'))]}]
        for _ in range(100):
            values.append({'a':rng.integers(0,10,size=10).tolist(),
                           'b':[{str(i):[float(rng.normal()),None,True]} for i in range(3)]})
        for value in values:
            original, optimized=A._json_value(value), O.json_value(value)
            if not isinstance(value,np.integer):
                self.assertEqual(repr(original),repr(optimized))
        for value in values:
            try:
                expected=A.canonical(A._json_value(value))
            except TypeError:
                with self.assertRaises(TypeError): A.canonical(O.json_value(value))
            else:
                self.assertEqual(expected,A.canonical(O.json_value(value)))

    def test_fixed_grid_mapping_and_sparse_dense_noncontiguous_masks(self):
        self.assertTrue(O._fixed_grid_proof())
        original=S._original()[3]._nearest_model_count
        rng=np.random.default_rng(19)
        for density in (0.,.0001,.05,.5,1.):
            mask=rng.random((720,1280))<density
            for value in (mask,mask[:,::-1],mask[::-1,:],mask.astype(np.uint8)*255):
                self.assertEqual(original(value,768,432),O._nearest_model_count(value,768,432))

    def test_other_geometries_and_dtype_use_original(self):
        original=S._original()[3]._nearest_model_count
        for shape,output in (((5,9),(3,7)),((720,1280),(640,360)),((2,2),(3,3))):
            value=np.arange(np.prod(shape)).reshape(shape).astype(np.float64)
            self.assertEqual(original(value,*output),O._nearest_model_count(value,*output))

    def test_scope_restores_and_rejects_foreign_seam(self):
        original=S.LiveCompleteTargetProducer
        with O.install():
            self.assertIs(S.LiveCompleteTargetProducer,O.OptimizedSourceProducer)
        self.assertIs(S.LiveCompleteTargetProducer,original)
        with patch.object(S,'LiveCompleteTargetProducer',object):
            with self.assertRaises(RuntimeError):
                with O.install(): pass

    def test_original_namespaces_unchanged(self):
        objects,semantic,static,visibility,paint=S._original()
        snapshot={name:id(value) for name,value in static.items()}
        original_nearest=visibility._nearest_model_count
        first=O._original_private();second=O._original_private()
        self.assertIsNot(first[2],second[2])
        self.assertIsNot(first[3],second[3])
        first[2]['cv2']='private mutation'
        self.assertEqual(snapshot,{name:id(value) for name,value in static.items()})
        self.assertIs(visibility._nearest_model_count,original_nearest)
        self.assertNotEqual(second[2]['cv2'],'private mutation')

    def _producers(self,directory):
        args=dict(static_catalogue_dir=directory/'catalogue',experiment_id='prospective_cpu_test',
                  ego_id=999,dynamic_population_ids=[1,2],population_evidence_sha256='d'*64,max_frames=1000)
        return O.BASE_PRODUCER(**args),O.OptimizedSourceProducer(**args)

    def test_geometry_evidence_and_target_exact_across_pose_variants(self):
        fixture=SourceGeometryTests()
        for pitch in (0.,8.,-11.):
            with fixture.fixture(pitch=pitch) as (directory,kwargs):
                baseline,optimized=self._producers(directory)
                with patch('rl_agent.splitfusion_run5_live_avo_v1.gt_source_sync.time.clock_gettime_ns',return_value=100):
                    a=baseline.build(**kwargs); b=optimized.build(**kwargs)
                self.assertEqual(A.input_content_sha256(a),A.input_content_sha256(b))
                self.assertEqual(A.canonical(A._json_value(baseline.last_evidence)),
                                 A.canonical(A._json_value(optimized.last_evidence)))
                aa,bb=E.prepare_target(a),E.prepare_target(b)
                self.assertEqual(aa.content_sha256,bb.content_sha256)
                for key in ('segmentation_png','object_ignore_png'):
                    self.assertEqual(aa.target[key],bb.target[key])

    def test_source_validators_not_bypassed(self):
        fixture=SourceGeometryTests()
        with fixture.fixture() as (directory,kwargs):
            for invalid in ('unknown_population','source_timestamp','source_pose','camera_inverse'):
                a,b=self._producers(directory)
                bad=dict(kwargs)
                if invalid=='unknown_population':
                    bad['dynamic_snapshot_evidence']={**kwargs['dynamic_snapshot_evidence'],'active_population_ids':[1,2,3]}
                elif invalid=='source_timestamp':
                    bad['rgb']={**kwargs['rgb'],'timestamp':9.}
                elif invalid=='source_pose':
                    bad['rgb']={**kwargs['rgb'],'camera_matrix':(np.eye(4)*2).tolist()}
                else: bad['camera_inverse']=np.eye(4)*2
                for producer in (a,b):
                    with self.assertRaises(ValueError): producer.build(**bad)
                    self.assertEqual(producer.processed_frames,0)

    def test_frame_order_and_budget_unchanged(self):
        fixture=SourceGeometryTests()
        with fixture.fixture() as (directory,kwargs):
            a,b=self._producers(directory)
            for producer in (a,b):
                producer.build(**kwargs)
                with self.assertRaises(ValueError): producer.build(**kwargs)
                producer.processed_frames=producer.max_frames
                with self.assertRaises(ValueError): producer.build(**kwargs)

    def test_source_pin_verification_not_cached_away(self):
        with patch.object(S,'verify_sources',side_effect=S.SourceGeometryError('tampered')):
            with self.assertRaises(S.SourceGeometryError): O.prewarm()


if __name__=='__main__': unittest.main()
