"""Pure reporting tests; no capture, inference or live process is launched."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from rl_agent.splitfusion_run5_live_avo_v1.test_render_results import joined_fixture
from . import render_story as S


def fixture():
    values=joined_fixture()
    values["frames"][0]["capture_wall_ns"]="2000000000"
    decision=values["ue"]["decisions"][0]
    decision.update(capture={"domain":S.WALL,"ns":2000000000},mcs_age_ns=10000000,backlog_age_ns=2000000,
        stages={"rgb_receipt_raw_ns":900000000,"si_p40_start_raw_ns":990000000,"si_p40_end_raw_ns":995000000})
    values["snr"]["decision_snr"][0]["snr"].update(clock_domain=S.RAW,effective_since_ns=900000000,heartbeat_ns=990000000)
    records=S.canonical.join_records(**values)
    values.update(records=records,
        map_rows=[dict(frame_id="10",stream_id="stream",outcome="RESULT_INSTALLED",accepted="1",
            capture_timestamp="2.0",map_install_at="2.2",map_age_at_install_ms="200.0",record_count="2")],
        map_identities=[dict(frame_id="10",outcome="RESULT_INSTALLED",
            run4_identity_json=json.dumps(values["ue"]["transmitted_identities"][0]["run4_identity"]))])
    return values


class StoryTests(unittest.TestCase):
    def test_exact_join_scaling_and_clock_separation(self):
        decisions,events,aoi=S.join_story(**fixture())
        row=decisions[0]
        self.assertEqual(row["camera_si"],S.SI_CENTER)
        self.assertEqual(row["ul_mcs"],0)
        self.assertEqual(row["rlc_backlog_bytes"],0)
        self.assertEqual(row["si_p40_compute_ms"],5.)
        self.assertAlmostEqual(row["scene_rgb_receipt_age_ms"],99.999)
        self.assertAlmostEqual(row["map_age_at_install_ms"],200.)
        self.assertEqual(row["classification_confidence"],None)
        self.assertEqual(events[0]["clock_domain"],S.WALL)
        self.assertEqual(len(aoi),1)

    def test_duplicate_map_or_decision_identity_refused(self):
        for key in ("map_rows","map_identities"):
            values=fixture(); values[key].append(copy.deepcopy(values[key][0]))
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,"duplicate"):
                S.join_story(**values)
        values=fixture(); values["ue"]["decisions"].append(copy.deepcopy(values["ue"]["decisions"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            S.join_story(**values)

    def test_missing_map_identity_refused(self):
        values=fixture(); values["map_identities"]=[]
        with self.assertRaisesRegex(ValueError,"Missing/foreign"):
            S.join_story(**values)

    def test_full_map_identity_refused(self):
        values=fixture(); identity=json.loads(values["map_identities"][0]["run4_identity_json"])
        identity["session_uuid"]="foreign"; values["map_identities"][0]["run4_identity_json"]=json.dumps(identity)
        with self.assertRaisesRegex(ValueError,"full identity"):
            S.join_story(**values)

    def test_canonical_row_mutation_refused(self):
        values=fixture(); values["records"][0]["q_e4"]+=1
        with self.assertRaisesRegex(ValueError,"Canonical policy result"):
            S.join_story(**values)

    def test_wrong_capture_and_snr_clocks_refused(self):
        values=fixture(); values["ue"]["decisions"][0]["capture"]["domain"]=S.RAW
        with self.assertRaisesRegex(ValueError,"capture clock"):
            S.join_story(**values)
        values=fixture(); values["snr"]["decision_snr"][0]["snr"]["clock_domain"]="CLOCK_MONOTONIC"
        with self.assertRaisesRegex(ValueError,"SNR clock"):
            S.join_story(**values)

    def test_future_snr_and_reversed_scene_interval_refused(self):
        values=fixture(); values["snr"]["decision_snr"][0]["snr"]["effective_since_ns"]=2000000000
        with self.assertRaisesRegex(ValueError,"Future SNR"):
            S.join_story(**values)
        values=fixture(); values["ue"]["decisions"][0]["stages"]["si_p40_end_raw_ns"]=989000000
        with self.assertRaisesRegex(ValueError,"ordering"):
            S.join_story(**values)

    def test_map_capture_and_age_arithmetic_refused(self):
        values=fixture(); values["map_rows"][0]["capture_timestamp"]="2.001"
        with self.assertRaisesRegex(ValueError,"capture timestamp mismatch"):
            S.join_story(**values)
        values=fixture(); values["map_rows"][0]["map_age_at_install_ms"]="201.0"
        with self.assertRaisesRegex(ValueError,"age arithmetic"):
            S.join_story(**values)

    def test_aoi_triangle_is_not_install_age_hold(self):
        events=[dict(frame_id=1,capture_wall_s=1.,map_install_wall_s=1.2,clock_domain=S.WALL),
                dict(frame_id=2,capture_wall_s=1.1,map_install_wall_s=1.35,clock_domain=S.WALL)]
        points=S.frame_aoi(events)
        self.assertEqual([p["phase"] for p in points],["AFTER_INSTALL","BEFORE_INSTALL","AFTER_INSTALL"])
        for actual,expected in zip([p["age_ms"] for p in points],[200.,350.,250.]):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual([p["latest_frame_id"] for p in points],[1,1,2])
        self.assertEqual(points[-1]["wall_s"],1.35)

    def test_aoi_nonfinite_and_regression_fail(self):
        for change in ({"capture_wall_s":1.3},{"clock_domain":S.RAW},{"map_install_wall_s":float("nan")}):
            event=dict(frame_id=1,capture_wall_s=1.,map_install_wall_s=1.2,clock_domain=S.WALL,**{})
            event.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError): S.frame_aoi([event])
        events=[dict(frame_id=1,capture_wall_s=1.,map_install_wall_s=1.2,clock_domain=S.WALL),
                dict(frame_id=2,capture_wall_s=.9,map_install_wall_s=1.3,clock_domain=S.WALL)]
        with self.assertRaisesRegex(ValueError,"Regressing map capture"):
            S.frame_aoi(events)

    def test_retained_input_hash_guard_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); p=root/"input.json"; p.write_text("{}\n"); digest=S.sha(p); inputs={}
            S.verify_bound_file(root,"input.json",digest,inputs)
            S.verify_inputs_unchanged(inputs)
            p.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                S.verify_bound_file(root,"input.json",digest,{})
            with self.assertRaisesRegex(ValueError,"source changed"):
                S.verify_inputs_unchanged(inputs)

    def test_manifest_traversal_and_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); p=root/"input"; p.write_text("x"); (root/"link").symlink_to(p)
            for name in ("../input",str(p),"link"):
                with self.subTest(name=name),self.assertRaises(ValueError):
                    S.verify_bound_file(root,name,S.sha(p),{})

    def test_statistics_keep_missing_denominator_and_separate_undefined(self):
        decisions,events,_=S.join_story(**fixture())
        defined=decisions[0]
        undefined=dict(defined,frame_id=11,q_perc=None,latency_ms=180.,timely=False,terminal="TIMEOUT")
        missing=dict(defined,frame_id=12,q_perc=.4,latency_ms=None,timely=False,terminal="TIMEOUT",feedback_available=False)
        data=S.statistical_series([defined,undefined,missing],events)
        self.assertEqual(data["all_decisions"],3)
        self.assertEqual(data["received"],2)
        self.assertEqual(data["missing"],1)
        self.assertEqual(data["defined_feedback_ms"],[150.])
        self.assertEqual(data["undefined_feedback_ms"],[180.])
        self.assertEqual(data["feedback_ms"],[150.,180.])
        self.assertEqual(sum(data["selected_mode_counts"]),3)
        self.assertEqual(len(data["installed_frame_age_ms"]),1)

    def test_statistics_pool_matched_objects_not_frame_medians(self):
        decisions,events,_=S.join_story(**fixture())
        decisions[0]["vehicle"]["errors_m"]=[.1,.4,.9]
        decisions[0]["vehicle"]["xy_median_m"]=.4
        data=S.statistical_series(decisions,events)
        self.assertEqual(data["xy_errors_m"]["vehicle"],[.1,.4,.9])
        self.assertEqual(data["xy_errors_m"]["person"],[.2])

    def test_statistics_reject_nonfinite_measurement(self):
        decisions,events,_=S.join_story(**fixture())
        decisions[0]["latency_ms"]=float("nan")
        with self.assertRaisesRegex(ValueError,"Invalid empirical"):
            S.statistical_series(decisions,events)


if __name__=="__main__": unittest.main()
