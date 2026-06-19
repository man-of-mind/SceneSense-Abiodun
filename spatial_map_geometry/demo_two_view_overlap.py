#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_map_geometry.association import associate_objects
from spatial_map_geometry.geometry import (
    bounds,
    convex_polygon_intersection,
    object_footprint_polygon,
    polygon_area,
    sensor_fov_polygon,
)
from spatial_map_geometry.occlusion_reasoner import infer_overlap_disagreements
from spatial_map_geometry.schemas import LocalSensorMap, SensorPose2D, SpatialObject


def build_demo_maps() -> list[LocalSensorMap]:
    pose_a = SensorPose2D(x=0.0, y=0.0, yaw_deg=28.0)
    pose_b = SensorPose2D(x=30.0, y=1.0, yaw_deg=152.0)
    fov_a = sensor_fov_polygon(pose_a, fov_deg=75.0, range_m=45.0)
    fov_b = sensor_fov_polygon(pose_b, fov_deg=75.0, range_m=45.0)

    shared_a = SpatialObject(
        object_id="a_vehicle_shared",
        class_name="vehicle",
        x=15.0,
        y=7.5,
        length=4.6,
        width=1.9,
        yaw_deg=5.0,
        confidence=0.88,
        source_stream_id="car_A",
    )
    shared_b = SpatialObject(
        object_id="b_vehicle_shared",
        class_name="vehicle",
        x=15.7,
        y=7.2,
        length=4.5,
        width=1.9,
        yaw_deg=7.0,
        confidence=0.83,
        source_stream_id="car_B",
    )
    only_a = SpatialObject(
        object_id="a_ped_possible_occluded",
        class_name="person",
        x=18.0,
        y=-1.5,
        length=0.6,
        width=0.6,
        height=1.7,
        yaw_deg=0.0,
        confidence=0.76,
        source_stream_id="car_A",
    )
    only_b = SpatialObject(
        object_id="b_vehicle_possible_occluded",
        class_name="vehicle",
        x=12.0,
        y=12.0,
        length=4.3,
        width=1.8,
        yaw_deg=40.0,
        confidence=0.72,
        source_stream_id="car_B",
    )

    return [
        LocalSensorMap(
            stream_id="car_A",
            pose=pose_a,
            fov_polygon=fov_a,
            objects=[shared_a, only_a],
            sensor_type="rgb_radar",
            fov_deg=75.0,
            range_m=45.0,
            provenance={"demo": True},
        ),
        LocalSensorMap(
            stream_id="car_B",
            pose=pose_b,
            fov_polygon=fov_b,
            objects=[shared_b, only_b],
            sensor_type="rgb_radar",
            fov_deg=75.0,
            range_m=45.0,
            provenance={"demo": True},
        ),
    ]


def make_summary(local_maps: list[LocalSensorMap]) -> dict[str, object]:
    associations = associate_objects(local_maps, distance_threshold_m=3.0)
    hypotheses = infer_overlap_disagreements(
        local_maps,
        distance_threshold_m=3.0,
        min_overlap_area_m2=10.0,
    )
    overlap_poly = convex_polygon_intersection(local_maps[0].fov_polygon, local_maps[1].fov_polygon)
    return {
        "local_maps": [item.to_dict() for item in local_maps],
        "overlap_area_m2": polygon_area(overlap_poly),
        "associations": [item.to_dict() for item in associations],
        "occlusion_hypotheses": [item.to_dict() for item in hypotheses],
    }


def plot_demo(local_maps: list[LocalSensorMap], output_path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    overlap_poly = convex_polygon_intersection(local_maps[0].fov_polygon, local_maps[1].fov_polygon)
    all_points = []
    for local_map in local_maps:
        all_points.extend(local_map.fov_polygon)
        all_points.extend(obj.xy for obj in local_map.objects)
    min_x, min_y, max_x, max_y = bounds(all_points)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"car_A": "#1f77b4", "car_B": "#ff7f0e"}
    for local_map in local_maps:
        color = colors.get(local_map.stream_id, "#999999")
        ax.add_patch(
            Polygon(
                local_map.fov_polygon,
                closed=True,
                fill=True,
                alpha=0.13,
                edgecolor=color,
                facecolor=color,
                linewidth=2,
                label=f"{local_map.stream_id} FoV",
            )
        )
        ax.scatter([local_map.pose.x], [local_map.pose.y], color=color, marker="^", s=120)
        ax.text(local_map.pose.x, local_map.pose.y - 1.2, local_map.stream_id, color=color, ha="center")
        for obj in local_map.objects:
            marker = "s" if obj.class_name == "vehicle" else "o"
            ax.scatter([obj.x], [obj.y], color=color, marker=marker, s=80)
            ax.add_patch(
                Polygon(
                    object_footprint_polygon(obj),
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                    linestyle="--",
                )
            )
            ax.text(obj.x, obj.y + 0.8, f"{obj.class_name}\n{obj.object_id}", color=color, fontsize=8)

    if len(overlap_poly) >= 3:
        ax.add_patch(
            Polygon(
                overlap_poly,
                closed=True,
                fill=True,
                alpha=0.25,
                edgecolor="#2ca02c",
                facecolor="#2ca02c",
                linewidth=1.5,
                label="FoV overlap",
            )
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_x - 5.0, max_x + 5.0)
    ax.set_ylim(min_y - 5.0, max_y + 5.0)
    ax.grid(True, alpha=0.25)
    ax.set_title("SceneSense Spatial Map Geometry Demo")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.legend(loc="upper right")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a two-view spatial-map geometry demo.")
    parser.add_argument(
        "--output-json",
        default="analysis_outputs/spatial_map_geometry/two_view_overlap_demo.json",
    )
    parser.add_argument(
        "--output-plot",
        default="analysis_outputs/spatial_map_geometry/two_view_overlap_demo.png",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    local_maps = build_demo_maps()
    summary = make_summary(local_maps)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.no_plot:
        plot_demo(local_maps, args.output_plot)

    print(json.dumps({
        "output_json": args.output_json,
        "output_plot": None if args.no_plot else args.output_plot,
        "overlap_area_m2": summary["overlap_area_m2"],
        "associations": len(summary["associations"]),
        "occlusion_hypotheses": len(summary["occlusion_hypotheses"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
