"""Focused CPU / WSGI replay tests. Never imports or starts a live runtime."""
import copy
import json
import unittest
from pathlib import Path

from spatial_map_coop.replay_ui_v1.recording import Recording, bind_pose, digest, MAX_EVENTS, sha_file
from spatial_map_coop.replay_ui_v1.server import create_app
from spatial_map_coop.multi_ue_v1.association import ConservativeTrackStore, associate_snapshot
from spatial_map_coop.multi_ue_v1.buffer import MultiUESnapshot, SourceFrame
from spatial_map_coop.multi_ue_v1.contracts import ObservationBatch, _normalise_record
from spatial_map_coop.replay_ui_v1.recording import POLICY


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = Recording()
        r = cls.recording
        with r.path.open() as f:
            cls.record = json.loads(f.readline())['map_document']['record']
        cls.key = (cls.record['stream_id'], cls.record['frame_id'])
        cls.wire = r.wires[cls.key]
        cls.owned = r.owned[cls.wire['source_sha256']]
        cls.binding = r.install_bindings[cls.key]
        cls.client = create_app(r).test_client()

    def bind(self, r=None, w=None, o=None, binding=None):
        return bind_pose(r or self.record, w or self.wire, o or self.owned,
                         self.recording.calib, binding or self.binding)

    def test_exact_pose_join_mount_and_units(self):
        p, error = self.bind()
        self.assertIsNone(error)
        self.assertAlmostEqual(p['fov_deg'],120)
        self.assertEqual(p['yaw'],self.wire['context']['ego_world'][4])
        self.assertNotEqual(p['x'],p['sensor']['x'])
        self.assertEqual(p['capture_ns'],str(self.wire['context']['capture_timestamp_ns']))

    def test_run_cell_stream_frame_and_capture_mismatch(self):
        for field in ['run_id','cell_id','stream_id']:
            wire=copy.deepcopy(self.wire);wire['gt_identity'][field]='wrong'
            self.assertEqual(self.bind(w=wire)[1],'INSTALL_IDENTITY_MISMATCH')
        wire=copy.deepcopy(self.wire);wire['context']['frame_id']+=1
        self.assertEqual(self.bind(w=wire)[1],'POSE_IDENTITY_MISMATCH')
        wire=copy.deepcopy(self.wire);wire['context']['capture_timestamp_ns']+=1000
        self.assertEqual(self.bind(w=wire)[1],'POSE_TIMESTAMP_MISMATCH')

    def test_missing_pose_or_install_binding(self):
        self.assertEqual(bind_pose(self.record,None,None,self.recording.calib)[1],'MISSING_FRAME_POSE')
        self.assertEqual(bind_pose(self.record,self.wire,self.owned,self.recording.calib)[1],'MISSING_INSTALL_IDENTITY')

    def test_malformed_sidecar_does_not_abort_replay(self):
        wire=copy.deepcopy(self.wire);del wire['context']['ego_world']
        self.assertEqual(self.bind(w=wire)[1],'INVALID_POSE_SIDECAR')

    def test_calibration_drift_is_not_replaced_by_fallback_angle(self):
        wire=copy.deepcopy(self.wire);wire['context']['camera_model_sha256']='wrong'
        self.assertEqual(self.bind(w=wire)[1],'UNREGISTERED_CALIBRATION')

    def test_sensor_matrix_corruption_refused_even_if_rehashed(self):
        owned=copy.deepcopy(self.owned);owned['source']['camera_matrix'][0][3]+=1
        owned['source_sha256']=digest(owned['source'])
        wire=copy.deepcopy(self.wire);wire['source_sha256']=owned['source_sha256']
        self.assertEqual(self.bind(w=wire,o=owned)[1],'SENSOR_MOUNT_MISMATCH')

    def test_pose_rotation_mismatch_refused_even_if_rehashed(self):
        owned=copy.deepcopy(self.owned);owned['source']['ego_world'][4]+=90
        owned['source_sha256']=digest(owned['source'])
        wire=copy.deepcopy(self.wire);wire['context']['ego_world']=owned['source']['ego_world']
        wire['source_sha256']=owned['source_sha256']
        self.assertEqual(self.bind(w=wire,o=owned)[1],'EGO_ROTATION_MISMATCH')

    def test_clock_inversion_install_binding(self):
        record=copy.deepcopy(self.record);record['install_timestamp']-=1
        self.assertEqual(self.bind(r=record)[1],'INSTALL_TIMESTAMP_MISMATCH')

    def test_bound_inputs_and_exact_raw_records(self):
        clip=self.recording.clip('departure')
        self.assertLessEqual(len(clip['events']),MAX_EVENTS)
        first=clip['events'][0]
        self.assertEqual([o['raw'] for o in first['objects']],self.record['objects'])
        self.assertGreater(first['arrival_ms']-first['capture_ms'],150)
        self.assertEqual(clip['errors'],{})

    def test_replay_and_camera_do_not_write_scientific_state(self):
        r=self.recording
        paths=[r.path,r.cell/'pose_archive/pose_evidence.jsonl',r.cell/'direct_edge_map/direct_map_ingest.csv']
        before={p:sha_file(p) for p in paths}
        self.assertEqual(self.client.get('/api/replay/clip/departure').status_code,200)
        self.assertEqual(self.client.get('/api/replay/clip/departure').status_code,200)
        response=self.client.get('/api/replay/camera/1480.png')
        self.assertEqual(response.status_code,200);self.assertTrue(response.data.startswith(b'\x89PNG'))
        self.assertEqual(before,{p:sha_file(p) for p in paths})

    def test_no_live_or_write_endpoints(self):
        for url in ['/api/spatial_map/latest','/api/replay/clip/bad','/assets/../../CLAUDE.md']:
            self.assertEqual(self.client.get(url).status_code,404)
        self.assertEqual(self.client.post('/api/replay/meta',json={}).status_code,405)

    def test_reused_association_identity_continuity_one_to_one(self):
        store=ConservativeTrackStore(POLICY)
        def update(frame,shift):
            records=[]
            for i,x in enumerate([10.,20.]):
                records.append(dict(id=f'{frame}:{i}',type='Vehicle',score=.02,
                    location=dict(x=x+shift,y=0,z=0),dimensions=dict(length=4,width=2,height=1.5),yaw_deg=0))
            batch=ObservationBatch('ego','a','session','unix_wall_ns',frame,frame*100_000_000,
                    'unix_wall_ns',frame*100_000_000+200_000_000,None,'fixture',digest(records),
                    tuple(_normalise_record(o,i) for i,o in enumerate(records)))
            snap=MultiUESnapshot('unix_wall_ns',batch.capture_timestamp_ns,(SourceFrame(batch,0),),0,100_000_000,True)
            return store.update(associate_snapshot(snap,POLICY))
        a=update(1,0);b=update(2,1)
        self.assertEqual([t.track_id for t in a],[t.track_id for t in b])
        self.assertEqual(len(b),2);self.assertEqual(sorted(t.world_xyz[0] for t in b),[11.,21.])
        self.assertEqual(b[0].score,.02)


if __name__=='__main__':
    unittest.main()
