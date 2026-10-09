"""CPU-only regression tests; no live producer qualification is implied."""
import copy
import unittest

import numpy as np

from data_collection.route_b_publication_actor_volume_visibility_v1 import core, scoring
from . import quality_adapter as q


def scene():
    k = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
    centre, extent = (10., 0., 0.), (.25, .25, .9)
    boxes = core.projected_boxes(core.oriented_box_corners(centre, extent, 0), np.eye(4), k, width=200, height=200)
    y0, y1, x0, x1 = core.roi_pixel_bounds(boxes, width=200, height=200)
    depth = np.full((200, 200), 900.)
    depth[y0:y1, x0:x1] = 9.8
    row = dict(label="person", gt_actor_id="42", frame_id=10, timestamp=1.,
               object_world_x=10., object_world_y=0., object_world_z=0.,
               gt_extent_x_m=.25, gt_extent_y_m=.25, gt_extent_z_m=.9,
               object_yaw_deg=0., object_pitch_deg=0., object_roll_deg=0.,
               gt_distance_m=10., gt_center_x=100., gt_center_y=100.,
               object_sensor_x=10., gt_bbox_area_px=boxes["clipped_projected_area_px"])
    return dict(frame_id=10, timestamp=1., depth_frame_id=10, depth_timestamp=1.,
                depth_m=depth, camera_matrix=np.eye(4), camera_inverse=np.eye(4),
                intrinsics=k, raw_actor_rows=[row], width=200, height=200)


class QualityAdapterTests(unittest.TestCase):
    def test_raw_semantics_regression(self):
        tags = np.array([[1, 2, 4, 6, 7, 12, 14, 15, 16, 19]], dtype=np.uint8)
        np.testing.assert_array_equal(q.corrected_raw_semantic_3class(tags), [[0, 0, 0, 0, 0, 2, 1, 1, 1, 0]])
        np.testing.assert_array_equal(tags, [[1, 2, 4, 6, 7, 12, 14, 15, 16, 19]])

    def test_raw_lut_not_live_contract_and_no_legacy_table(self):
        with self.assertRaises(q.LiveGtContractError):
            q.require_live_gt_contract({"mask_contract": "CORRECTED_RAW_SEMANTIC"})
        binding = {"mask_contract": "LIVE_CORRECTED_RAW_CARLA_3CLASS_NOT_OFFLINE_V010",
                   "person_localization_contract": "ORIGINAL_UNNORMALIZED_AVO_GE_0_65",
                   "source": "SYNCHRONIZED_LIVE_FRAME", "producer_evidence_sha256": "a" * 64,
                   **{key: True for key in q.REQUIRED_LIVE_INPUTS}}
        q.require_live_gt_contract(binding)
        for key in q.REQUIRED_LIVE_INPUTS:
            with self.subTest(key=key), self.assertRaises(q.LiveGtContractError):
                q.require_live_gt_contract({**binding, key: False})
        with self.assertRaises(q.LiveGtContractError):
            q.require_live_gt_contract({**binding, "source": "OFFLINE_RETAINED_TABLE"})

    def test_exact_original_scorer_parity(self):
        args = scene()
        expected = scoring.score_actor_frame(depth_m=args["depth_m"], camera_matrix=np.eye(4),
            camera_inverse=np.eye(4), intrinsics=args["intrinsics"], width=200, height=200,
            target_key="42", target_centre=(10, 0, 0), target_extent=(.25, .25, .9),
            target_yaw_deg=0., pedestrian_boxes=[dict(key="42", centre=(10, 0, 0), extent=(.25, .25, .9), yaw_deg=0.)])
        result = q.score_live_avo_people(**args)
        self.assertEqual(result["qualified"][0]["avo"], expected["visibility"])
        self.assertGreater(result["qualified"][0]["avo"], .95)

    def test_no_depth_future_foreign_duplicate_refused(self):
        for change in ({"depth_m": None}, {"depth_frame_id": 11}, {"depth_timestamp": 1.1},
                       {"camera_inverse": np.ones((4, 4))}):
            with self.subTest(change=list(change)), self.assertRaises(q.LiveGtContractError):
                q.score_live_avo_people(**{**scene(), **change})
        args = scene()
        args["raw_actor_rows"] *= 2
        with self.assertRaises(q.LiveGtContractError):
            q.score_live_avo_people(**args)
        args = scene()
        args["raw_actor_rows"][0]["frame_id"] = 11
        with self.assertRaises(q.LiveGtContractError):
            q.score_live_avo_people(**args)

    def test_zero_support_is_real_not_missing_depth(self):
        args = scene()
        args["depth_m"][:] = 2.
        result = q.score_live_avo_people(**args)
        self.assertEqual(result["qualified"][0]["avo"], 0.)
        self.assertTrue(result["qualified"][0]["no_support"])

    def test_zero_and_far_plane_are_ignored_like_original_depth_scorer(self):
        for value in (0., 1000.):
            args = scene()
            args["depth_m"][:] = value
            result = q.score_live_avo_people(**args)
            expected = scoring.score_actor_frame(depth_m=args["depth_m"], camera_matrix=np.eye(4),
                camera_inverse=np.eye(4), intrinsics=args["intrinsics"], width=200, height=200,
                target_key="42", target_centre=(10, 0, 0), target_extent=(.25, .25, .9),
                target_yaw_deg=0., pedestrian_boxes=[dict(key="42", centre=(10, 0, 0), extent=(.25, .25, .9), yaw_deg=0.)])
            self.assertEqual(result["qualified"][0]["avo"], expected["visibility"])
            self.assertEqual(result["qualified"][0]["no_support"], expected["no_support"])
        for value in (-.01, 1000.01, float("nan")):
            args = scene()
            args["depth_m"][0, 0] = value
            with self.assertRaises(q.LiveGtContractError):
                q.score_live_avo_people(**args)

    def test_tilt_and_missing_tilt_evidence_refused(self):
        args = scene()
        args["raw_actor_rows"][0]["object_pitch_deg"] = 1.
        with self.assertRaises(q.LiveGtContractError):
            q.score_live_avo_people(**args)
        del args["raw_actor_rows"][0]["object_pitch_deg"]
        with self.assertRaises(q.LiveGtContractError):
            q.score_live_avo_people(**args)

    def test_far_behind_tilted_ignore_preserves_original_target_exactly(self):
        args = scene()
        distant = dict(args["raw_actor_rows"][0], gt_actor_id="103", object_world_x=-187.,
                       object_sensor_x=-187., gt_distance_m=187., gt_center_x=-1., gt_center_y=-1.,
                       gt_bbox_area_px=0.)
        args["raw_actor_rows"].append(distant)
        original = q.score_live_avo_people(**args)
        self.assertNotIn("geometry_relevance", original)
        distant["object_pitch_deg"], distant["object_roll_deg"] = 30., -10.
        result = q.score_live_avo_people(**args)
        proof = result.pop("geometry_relevance")
        self.assertEqual(result, original)
        omitted = proof["omitted_tilted_structural_actors"]
        self.assertEqual([row["gt_actor_id"] for row in omitted], ["103"])
        self.assertGreater(omitted[0]["target_separation"][0]["separation_margin_m"], 190.)
        # It stays available for the original structural-ignore matching stage.
        measured = q.person_avo_measurement([dict(class_name="person", score=.9, world_x=-187., world_y=0.)], result)
        self.assertEqual(measured["loc_person_structural_ignored_predictions"], 1)

    def test_overlapping_and_touching_tilted_competitors_refused(self):
        radius = float(np.linalg.norm(np.array([.25, .25, .9]) + core.ACTOR_VOLUME_TOLERANCE_M))
        for distance in (.8, 2*radius):
            with self.subTest(distance=distance):
                args = scene()
                args["raw_actor_rows"][0].update(object_world_x=39.5, object_sensor_x=39.5, gt_distance_m=39.5)
                competing = dict(args["raw_actor_rows"][0], gt_actor_id="other", object_world_x=39.5+distance,
                    object_sensor_x=39.5+distance, gt_distance_m=39.5+distance, object_pitch_deg=1.)
                args["raw_actor_rows"].append(competing)
                with self.assertRaisesRegex(q.LiveGtContractError, "TILTED_POTENTIALLY_OVERLAPPING") as error:
                    q.score_live_avo_people(**args)
                self.assertEqual(error.exception.geometry_diagnostics["offender"]["gt_actor_id"], "other")

    def test_projection_only_tilted_ignore_is_not_assumed_irrelevant(self):
        args = scene()
        args["raw_actor_rows"][0].update(gt_bbox_area_px=0., gt_center_x=-1., object_pitch_deg=20.)
        with self.assertRaisesRegex(q.LiveGtContractError, "TILTED_PROJECTION_ONLY"):
            q.score_live_avo_people(**args)

    def test_no_target_scene_far_tilted_actor_retains_ignore_identity(self):
        args = scene()
        args["raw_actor_rows"][0].update(object_world_x=-187., object_sensor_x=-187., gt_distance_m=187.,
                                       gt_bbox_area_px=0., gt_center_x=-1., object_pitch_deg=20.)
        result = q.score_live_avo_people(**args)
        self.assertEqual(result["qualified"], [])
        self.assertEqual(result["structural_ignored"][0]["gt_actor_id"], "42")
        self.assertEqual(result["geometry_relevance"]["omitted_tilted_structural_actors"][0]["target_separation"], [])

    def test_tilted_target_is_refused_with_precise_orientation(self):
        args = scene()
        args["raw_actor_rows"][0].update(object_pitch_deg=3.5, object_roll_deg=-2.)
        with self.assertRaisesRegex(q.LiveGtContractError, "TILTED_SCOREABLE_TARGET") as error:
            q.score_live_avo_people(**args)
        offender = error.exception.geometry_diagnostics["offender"]
        self.assertEqual((offender["gt_actor_id"], offender["pitch_deg"], offender["roll_deg"]), ("42", 3.5, -2.))

    def test_structural_gate_preserves_ignore_geometry(self):
        args = scene()
        args["raw_actor_rows"][0]["gt_distance_m"] = 41.
        result = q.score_live_avo_people(**args)
        self.assertEqual(result["qualified"], [])
        self.assertEqual(result["structural_ignored"][0]["gt_actor_id"], "42")

    def test_avo_ignore_cascade_and_threshold(self):
        gt = {"qualified": [dict(world_x=0, world_y=0, avo=.65), dict(world_x=10, world_y=0, avo=.64)],
              "structural_ignored": [dict(world_x=20, world_y=0)]}
        preds = [dict(class_name="person", score=.25, world_x=x, world_y=0) for x in (0, 10, 20, 40)]
        before = copy.deepcopy((gt, preds))
        result = q.person_avo_measurement(preds, gt)
        self.assertEqual((result["loc_person_tp"], result["loc_person_fp"], result["loc_person_fn"],
                          result["loc_person_ignored_predictions"]), (1, 1, 0, 2))
        self.assertEqual((gt, preds), before)

    def test_person_matching_nonfinite_or_invalid_avo_refused(self):
        for score in (float("nan"), 1.1, -.1):
            gt = {"qualified": [dict(world_x=0, world_y=0, avo=score)], "structural_ignored": []}
            with self.assertRaises(q.LiveGtContractError):
                q.person_avo_measurement([], gt)

    def test_person_stats_equal_frozen_fcos_avo_scorer(self):
        from data_collection.route_b_publication_actor_volume_observability_model_comparison_v1 import run_comparison as old
        rng = np.random.default_rng(5)
        for _ in range(20):
            qualified = [dict(world_x=float(rng.uniform(0, 30)), world_y=0.,
                              avo=float(rng.uniform()), distance_bin="10_20m") for __ in range(5)]
            structural = [dict(world_x=40., world_y=0.)]
            preds = [dict(class_name="person", score=float(rng.uniform()),
                          world_x=float(rng.uniform(0, 45)), world_y=0.) for __ in range(12)]
            ours = q.person_avo_measurement(preds, {"qualified": qualified, "structural_ignored": structural})
            reference = old.score_person_view(frame_ids=["x"], predictions={"x": preds},
                qualified_gt={"x": qualified}, structural_ignored_gt={"x": structural},
                episode_by_sample={"x": old.EPISODES[0]}, avo_threshold=.65,
                detection_threshold=.25)["overall"]
            for suffix, old_key in (("eligible_gt", "observable_gt"), ("tp", "tp"),
                                    ("fp", "fp"), ("fn", "fn"), ("ignored_predictions", "ignored_predictions")):
                self.assertEqual(ours["loc_person_" + suffix], reference[old_key])
            errors = ours["loc_person_matched_xy_errors_m"]
            self.assertAlmostEqual(sum(errors) / len(errors) if errors else 0., reference["xy_mae_m"] or 0.)

    def test_segmentation_ignore_never_counted_as_false_positive(self):
        truth = np.array([[0, 1, 2, 255]], np.uint8)
        pred = np.array([[0, 1, 2, 2]], np.uint8)
        result = q.segmentation_sufficient_statistics(pred, truth)
        self.assertEqual(result["seg_ignored_pixels"], 1)
        self.assertEqual(result["seg_person_pred_pixels"], 1)
        self.assertEqual(result["seg_person_union_pixels"], 1)
        self.assertEqual(result["seg_confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_segmentation_exact_nearest_target_grid(self):
        pred = np.array([[1]], np.uint8)
        truth = np.array([[1, 255], [1, 0]], np.uint8)
        result = q.segmentation_sufficient_statistics(pred, truth)
        self.assertEqual(result["seg_vehicle_intersection_pixels"], 2)
        self.assertEqual(result["seg_vehicle_union_pixels"], 3)
        with self.assertRaises(q.LiveGtContractError):
            q.segmentation_sufficient_statistics(np.array([[255]], np.uint8), truth)


if __name__ == "__main__":
    unittest.main()
