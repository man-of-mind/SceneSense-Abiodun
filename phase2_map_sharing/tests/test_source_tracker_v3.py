from __future__ import annotations

import unittest

from phase2_map_sharing.source_tracker_v3 import (
    TRACKER_V3_VERSION,
    SourceLocalCausalTrackerV3,
)


def detection(
    x: float,
    *,
    y: float = 0.0,
    class_name: str = "person",
    score: float = 0.9,
) -> dict:
    return {
        "class_name": class_name,
        "score": score,
        "world_x": x,
        "world_y": y,
        "world_z": 0.0,
    }


class SourceLocalCausalTrackerV3Tests(unittest.TestCase):
    def test_default_gate_confirms_on_two_consecutive_hits(self) -> None:
        tracker = SourceLocalCausalTrackerV3("helper")
        first, _ = tracker.update(
            frame_id=1,
            timestamp_s=0.0,
            detections=[detection(0.0)],
        )
        second, associations = tracker.update(
            frame_id=2,
            timestamp_s=0.1,
            detections=[detection(0.1)],
        )

        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertEqual(associations[0]["association"], "confirmed")

    def test_requires_confirmation_before_a_track_is_published(self) -> None:
        tracker = SourceLocalCausalTrackerV3(
            "helper",
            minimum_confirmation_hits=3,
        )

        first, first_associations = tracker.update(
            frame_id=1,
            timestamp_s=0.0,
            detections=[detection(0.0)],
        )
        second, second_associations = tracker.update(
            frame_id=2,
            timestamp_s=0.1,
            detections=[detection(0.1)],
        )
        third, third_associations = tracker.update(
            frame_id=3,
            timestamp_s=0.2,
            detections=[detection(0.2)],
        )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(first_associations[0]["association"], "birth_tentative")
        self.assertEqual(second_associations[0]["association"], "matched_tentative")
        self.assertEqual(third_associations[0]["association"], "confirmed")
        self.assertEqual(len(third), 1)
        self.assertEqual(third[0]["tracker_version"], TRACKER_V3_VERSION)
        self.assertEqual(third[0]["confirmation_hits"], 3)
        self.assertEqual(third[0]["confirmed_at_frame_id"], 3)

    def test_a_single_frame_tentative_track_does_not_survive_a_miss(self) -> None:
        tracker = SourceLocalCausalTrackerV3(
            "recipient",
            minimum_confirmation_hits=3,
            maximum_missed_frames=3,
        )
        tracker.update(frame_id=10, timestamp_s=1.0, detections=[detection(2.0)])

        tracks, associations = tracker.update(
            frame_id=11,
            timestamp_s=1.1,
            detections=[],
        )
        later, later_associations = tracker.update(
            frame_id=12,
            timestamp_s=1.2,
            detections=[detection(2.0)],
        )

        self.assertEqual(tracks, [])
        self.assertEqual(associations[0]["association"], "death_tentative")
        self.assertEqual(later, [])
        self.assertEqual(later_associations[0]["association"], "birth_tentative")
        self.assertTrue(later_associations[0]["source_track_id"].endswith("000002"))

    def test_same_class_world_duplicates_keep_only_highest_score(self) -> None:
        tracker = SourceLocalCausalTrackerV3(
            "helper",
            minimum_confirmation_hits=1,
            duplicate_suppression_radius_m=0.75,
        )

        tracks, associations = tracker.update(
            frame_id=1,
            timestamp_s=0.0,
            detections=[
                detection(0.2, score=0.4),
                detection(0.0, score=0.9),
                detection(0.1, class_name="vehicle", score=0.8),
            ],
        )

        self.assertEqual(len(tracks), 2)
        self.assertEqual(
            {(track["class_name"], track["world_x"]) for track in tracks},
            {("person", 0.0), ("vehicle", 0.1)},
        )
        suppressed = [
            row for row in associations if row["association"] == "duplicate_suppressed"
        ]
        self.assertEqual(len(suppressed), 1)
        self.assertEqual(suppressed[0]["detection_index"], 0)
        self.assertEqual(suppressed[0]["duplicate_of_detection_index"], 1)

    def test_velocity_is_causally_exponentially_smoothed(self) -> None:
        tracker = SourceLocalCausalTrackerV3(
            "helper",
            minimum_confirmation_hits=1,
            velocity_smoothing_alpha=0.5,
            speed_plausibility_slack_m=0.0,
        )
        tracker.update(
            frame_id=1,
            timestamp_s=0.0,
            detections=[detection(0.0, class_name="vehicle")],
        )
        second, _ = tracker.update(
            frame_id=2,
            timestamp_s=0.1,
            detections=[detection(1.0, class_name="vehicle")],
        )
        third, _ = tracker.update(
            frame_id=3,
            timestamp_s=0.2,
            detections=[detection(2.0, class_name="vehicle")],
        )

        self.assertAlmostEqual(second[0]["velocity_x"], 5.0)
        self.assertAlmostEqual(third[0]["velocity_x"], 7.5)
        self.assertAlmostEqual(third[0]["velocity_y"], 0.0)

    def test_implausible_person_displacement_cannot_hijack_a_track(self) -> None:
        tracker = SourceLocalCausalTrackerV3(
            "recipient",
            minimum_confirmation_hits=1,
            maximum_speed_mps_by_class={"person": 10.0},
            speed_plausibility_slack_m=0.0,
        )
        first, _ = tracker.update(
            frame_id=1,
            timestamp_s=0.0,
            detections=[detection(0.0)],
        )
        second, associations = tracker.update(
            frame_id=2,
            timestamp_s=0.1,
            detections=[detection(4.0)],
        )

        self.assertEqual(len(second), 2)
        births = [row for row in associations if row["association"] == "birth_confirmed"]
        self.assertEqual(len(births), 1)
        self.assertNotEqual(first[0]["source_track_id"], births[0]["source_track_id"])
        old = next(
            track
            for track in second
            if track["source_track_id"] == first[0]["source_track_id"]
        )
        self.assertEqual(old["missed_frames"], 1)

    def test_allowed_slack_is_still_speed_limited_before_smoothing(self) -> None:
        tracker = SourceLocalCausalTrackerV3(
            "helper",
            minimum_confirmation_hits=1,
            velocity_smoothing_alpha=1.0,
            maximum_speed_mps_by_class={"person": 10.0},
            speed_plausibility_slack_m=1.0,
        )
        tracker.update(frame_id=1, timestamp_s=0.0, detections=[detection(0.0)])
        tracks, associations = tracker.update(
            frame_id=2,
            timestamp_s=0.1,
            detections=[detection(1.5)],
        )

        self.assertEqual(len(tracks), 1)
        self.assertAlmostEqual(tracks[0]["velocity_x"], 10.0)
        self.assertTrue(tracks[0]["velocity_limited"])
        self.assertTrue(associations[0]["velocity_limited"])

    def test_confirmed_track_persists_only_for_the_configured_misses(self) -> None:
        tracker = SourceLocalCausalTrackerV3(
            "recipient",
            minimum_confirmation_hits=1,
            maximum_missed_frames=2,
        )
        tracker.update(frame_id=1, timestamp_s=0.0, detections=[detection(0.0)])
        first_miss, _ = tracker.update(frame_id=2, timestamp_s=0.1, detections=[])
        second_miss, _ = tracker.update(frame_id=3, timestamp_s=0.2, detections=[])
        dead, associations = tracker.update(frame_id=4, timestamp_s=0.3, detections=[])

        self.assertEqual(first_miss[0]["missed_frames"], 1)
        self.assertEqual(second_miss[0]["missed_frames"], 2)
        self.assertEqual(dead, [])
        self.assertEqual(associations[0]["association"], "death_confirmed")

    def test_noncausal_update_order_is_rejected(self) -> None:
        tracker = SourceLocalCausalTrackerV3("helper")
        tracker.update(frame_id=2, timestamp_s=0.2, detections=[])

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            tracker.update(frame_id=2, timestamp_s=0.3, detections=[])
        with self.assertRaisesRegex(ValueError, "cannot move backwards"):
            tracker.update(frame_id=3, timestamp_s=0.1, detections=[])


if __name__ == "__main__":
    unittest.main()
