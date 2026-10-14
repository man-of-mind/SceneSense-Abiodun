"""Offline tests against the actual baseline normalizer and locked installation."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from spatial_map_coop.replay_ui_v1.heading_ingest_v2 import CONTRACT, heading, install_under_state_lock_v2
from spatial_map_coop.replay_ui_v1.recording import Recording, sha_file
from spatial_map_coop.replay_ui_v1.upstream_v2 import join_predictions
from spatial_map_coop.replay_ui_v1.static_geometry_v2 import validated_geometry

class HeadingRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rl_agent.splitfusion_direct_edge_map_v1 import spatial_map_direct_server_v1 as direct
        cls.direct=direct;cls.b=direct.baseline;cls.recording=Recording()

    def setUp(self):
        self.config=patch.object(self.b,'CONFIG',SimpleNamespace(min_object_score=0.,object_yaw_map_offset_deg=90.,installed_frame_history_size=32))
        self.config.start();self.addCleanup(self.config.stop)
        # Isolated test-process globals; not a running server.
        self.b.latest_streams.clear();self.b.installed_frame_history.clear()

    def document(self,records):
        return dict(stream_id='offline_test',frame_id=7,capture_timestamp_ns=1_000_000_000,action_id=71,records=records)

    def install(self,records,contract=CONTRACT):
        doc=self.document(records);before=copy.deepcopy(doc)
        with patch('time.time',return_value=2.):
            install_under_state_lock_v2(doc,1.5,contract=contract,baseline=self.b)
        self.assertEqual(doc,before)
        actual=self.b.installed_frame_history[('offline_test',7)]
        self.assertIs(actual,self.b.latest_streams['offline_test'])
        return copy.deepcopy(actual)

    def test_cardinals_wrap_and_nonunit_through_real_install(self):
        for angle in (0,90,180,270,-180,359,-361,720):
            r=dict(yaw_sin=3*math.sin(math.radians(angle)),yaw_cos=3*math.cos(math.radians(angle)),type='Vehicle',world_x=5,score=.7,size_x=4,size_y=2)
            obj=self.install([r])['objects'][0]
            self.assertAlmostEqual(obj['yaw_deg'],(angle+180)%360-180)
            self.assertEqual(obj['map_yaw_offset_deg'],0)
            self.assertTrue(obj['heading']['valid'])
            self.assertEqual(obj['dimensions']['length'],4)

    def test_invalid_conflict_does_not_invent_heading(self):
        for r in ({'yaw_sin':0,'yaw_cos':0},{'yaw_sin':float('nan'),'yaw_cos':1},{'yaw_sin':1},{'yaw_sin':0,'yaw_cos':1,'yaw_deg':90},{'yaw_sin':0,'yaw_cos':1,'model_yaw_deg':0}):
            obj=self.install([dict(r,score=1)])['objects'][0]
            self.assertIsNone(obj['yaw_deg']);self.assertFalse(obj['heading']['valid'])
        self.assertTrue(heading(dict(yaw_sin=0,yaw_cos=1,yaw_deg=360))['valid'])
        self.assertFalse(heading(dict(yaw_sin=0,yaw_cos=1), 'unqualified')['valid'])

    def test_legacy_install_exactly_preserved(self):
        records=[dict(world_x=2,type='Vehicle',score=.4,yaw_deg=23),dict(score=.5,model_yaw_deg=-100),dict(score=.1)]
        with patch('time.time',return_value=2.):self.direct.install_under_state_lock(self.document(records),1.5)
        expected=copy.deepcopy(self.b.latest_streams['offline_test'])
        self.assertEqual(self.install(records,contract=None),expected)

    def test_real_retained_examples_only_heading_changes(self):
        for frame in (1480,2020,2748):
            path=self.recording.cell/f'numeric_gt/service/archive/frame_{frame}/scorer/numerical_score.json'
            predictions=json.loads(path.read_text())['prediction_records']
            records=[dict(p,type={'vehicle':'Vehicle','person':'Pedestrian'}[p['class_name']]) for p in predictions]
            legacy=self.install(records,contract=None);new=self.install(records)
            self.assertEqual(new['object_count'],len(records))
            for a,b,p in zip(legacy['objects'],new['objects'],predictions):
                for key in ('yaw_deg','model_yaw_deg','map_yaw_offset_deg'):a.pop(key);b.pop(key)
                h=b.pop('heading');self.assertEqual(a,b)
                self.assertAlmostEqual(h['yaw_deg'],math.degrees(math.atan2(p['yaw_sin'],p['yaw_cos'])))

    def test_filtered_records_preserve_indices(self):
        self.b.CONFIG.min_object_score=.5
        rows=self.install([dict(score=.1,yaw_sin=1,yaw_cos=0),dict(score=.8,yaw_sin=0,yaw_cos=1)])['objects']
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['id'],'offline_test:7:1');self.assertEqual(rows[0]['yaw_deg'],0)

    def test_exact_join_rejects_mismatch_and_missing(self):
        e=next(e for e in self.recording.clip('departure')['events'] if e['frame_id']==1480)
        record=dict(frame_id=e['frame_id'],stream_id=e['source'],objects=[o['raw'] for o in e['objects']])
        self.assertEqual(len(join_predictions(self.recording,record,e['pose'])[0]),38)
        for mutation in ('score','identity','dimension'):
            bad=copy.deepcopy(record)
            if mutation=='score':bad['objects'][0]['score']+=.0001
            if mutation=='identity':bad['objects'][0]['id']='wrong'
            if mutation=='dimension':bad['objects'][0]['dimensions']['width']+=.1
            self.assertEqual(join_predictions(self.recording,bad,e['pose'])[0],{})
        self.assertEqual(join_predictions(self.recording,record,None)[1],'NO_BOUND_POSE')

    def test_geometry_identity_and_world_coordinates(self):
        v=validated_geometry(self.recording)['validation'];self.assertTrue(v['valid']);self.assertLess(v['pose_to_lane_p95_m'],2.)
        self.assertIn('bounding-box',v['meaning'])

    def test_missing_geometry_degrades_without_heading_or_raw_changes(self):
        with patch('spatial_map_coop.replay_ui_v1.static_geometry_v2.STATIC',Path('/tmp/missing-spatial-map-fixture.json')):
            self.assertFalse(validated_geometry(self.recording)['validation']['valid'])

    def test_raw_history_and_frozen_sources_unchanged(self):
        files=[self.recording.path,Path(self.b.__file__),Path(self.direct.__file__)]
        hashes=[sha_file(p) for p in files]
        for name in ('departure','turn','stop'):
            clip=self.recording.clip(name)
            for e in clip['events']:
                for o in e['objects']:
                    self.assertEqual(o['raw']['yaw_deg'],90.)
                    if o['upstream']:self.assertEqual(o['upstream']['provenance']['binding'],e['pose']['binding'])
        self.assertEqual(hashes,[sha_file(p) for p in files])

if __name__=='__main__':unittest.main()
