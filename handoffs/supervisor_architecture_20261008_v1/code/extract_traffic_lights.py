#!/usr/bin/env python3

"""
CARLA Traffic Light Extractor and Visualizer

This script connects to the CARLA server, retrieves all traffic light actors,
saves their IDs and coordinates to a JSON file, and generates a top-down
2D map visualization of the city showing their exact locations.

The static map rendering mirrors the visualization used by
``real_time_spatial_map_server_v4.py`` by plotting:
- dense driving-lane centerlines instead of coarse topology edges
- smoothed road polylines for cleaner curves
- filtered building footprints near the road network
"""

import glob
import json
import os
import sys
import tempfile
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def _bootstrap_carla_module():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pythonapi_dir = os.path.dirname(current_dir)
    version_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    wheel_paths = sorted(
        glob.glob(
            os.path.join(
                pythonapi_dir,
                "carla",
                "dist",
                f"carla-*-{version_tag}-*.whl",
            )
        )
    )
    if not wheel_paths:
        wheel_paths = sorted(
            glob.glob(
                os.path.join(
                    pythonapi_dir,
                    "carla",
                    "dist",
                    "carla-*.whl",
                )
            )
        )

    bootstrap_paths = []
    if wheel_paths:
        bootstrap_root = os.path.join(
            tempfile.gettempdir(),
            "carla_python_bootstrap",
        )
        os.makedirs(bootstrap_root, exist_ok=True)

        for wheel_path in wheel_paths:
            extract_dir = os.path.join(
                bootstrap_root,
                os.path.splitext(os.path.basename(wheel_path))[0],
            )
            os.makedirs(extract_dir, exist_ok=True)

            try:
                with zipfile.ZipFile(wheel_path) as wheel_zip:
                    members = [
                        name for name in wheel_zip.namelist()
                        if name.startswith("carla") and name.endswith((".so", ".pyd"))
                    ]
                    for member in members:
                        extracted_path = os.path.join(extract_dir, member)
                        if not os.path.exists(extracted_path):
                            wheel_zip.extract(member, extract_dir)
            except zipfile.BadZipFile:
                continue

            bootstrap_paths.append(extract_dir)
    else:
        bootstrap_paths.append(pythonapi_dir)

    for path in reversed(bootstrap_paths):
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_carla_module()

import carla

CARLA_SERVER_HOST = '127.0.0.1'
CARLA_SERVER_PORT = 2000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

ROAD_PROXIMITY_THRESHOLD = 20.0
ROAD_ADJACENCY_SAMPLE_SPACING = 5.0
ROAD_CENTERLINE_SAMPLE_STEP = 1.0
ROAD_CENTERLINE_SMOOTHING_PASSES = 2
MIN_BUILDING_HEIGHT = 2.0
MIN_BUILDING_FOOTPRINT_AREA = 20.0
MIN_BUILDING_VOLUME = 80.0


def _sample_polygon_perimeter(points_xy, spacing):
    if len(points_xy) < 2:
        return np.array(points_xy, dtype=np.float64)

    sampled_points = []
    num_points = len(points_xy)
    for i in range(num_points):
        p1 = np.asarray(points_xy[i], dtype=np.float64)
        p2 = np.asarray(points_xy[(i + 1) % num_points], dtype=np.float64)
        edge = p2 - p1
        edge_length = float(np.linalg.norm(edge))

        if i == 0:
            sampled_points.append(p1)

        if edge_length <= 1e-9:
            continue

        num_segments = max(1, int(np.ceil(edge_length / spacing)))
        for step in range(1, num_segments + 1):
            t = step / num_segments
            sampled_points.append(p1 + t * edge)

    return np.vstack(sampled_points)


def _polygon_area(points_xy):
    if len(points_xy) < 3:
        return 0.0

    pts = np.asarray(points_xy, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _environment_object_bbox_footprint(bb):
    bbox_transform = carla.Transform(bb.location, bb.rotation)
    local_corners = [
        carla.Location(x=bb.extent.x, y=bb.extent.y, z=-bb.extent.z),
        carla.Location(x=-bb.extent.x, y=bb.extent.y, z=-bb.extent.z),
        carla.Location(x=-bb.extent.x, y=-bb.extent.y, z=-bb.extent.z),
        carla.Location(x=bb.extent.x, y=-bb.extent.y, z=-bb.extent.z),
    ]

    world_corners = []
    for local_corner in local_corners:
        world_corner = bbox_transform.transform(local_corner)
        world_corners.append((float(world_corner.x), float(world_corner.y)))

    return world_corners


def _waypoint_xy(waypoint):
    location = waypoint.transform.location
    return (float(location.x), float(location.y))


def _dedupe_polyline_points(points_xy, min_distance=0.1):
    if not points_xy:
        return []

    deduped = [points_xy[0]]
    min_distance = float(max(min_distance, 1e-6))
    for point in points_xy[1:]:
        if np.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) >= min_distance:
            deduped.append(point)
    return deduped


def _build_lane_centerlines(carla_map, sample_step):
    lane_samples = {}
    for waypoint in carla_map.generate_waypoints(sample_step):
        if waypoint.lane_type != carla.LaneType.Driving:
            continue

        lane_key = (waypoint.road_id, waypoint.section_id, waypoint.lane_id)
        lane_samples.setdefault(lane_key, []).append((float(waypoint.s), _waypoint_xy(waypoint)))

    polylines = []
    for samples in lane_samples.values():
        samples.sort(key=lambda item: item[0])
        polyline = _dedupe_polyline_points(
            [point for _, point in samples],
            min_distance=sample_step * 0.25,
        )
        if len(polyline) >= 2:
            polylines.append(polyline)

    return polylines


def _polyline_segments(polylines):
    segments = []
    for polyline in polylines:
        for p1, p2 in zip(polyline, polyline[1:]):
            segments.append([p1[0], p1[1], p2[0], p2[1]])
    return np.asarray(segments, dtype=np.float64)


def _smooth_polyline_for_plot(points_xy, passes):
    if passes <= 0 or len(points_xy) < 3:
        return points_xy

    pts = [np.asarray(point, dtype=np.float64) for point in points_xy]
    for _ in range(passes):
        smoothed = [pts[0]]
        for p1, p2 in zip(pts, pts[1:]):
            smoothed.append(0.75 * p1 + 0.25 * p2)
            smoothed.append(0.25 * p1 + 0.75 * p2)
        smoothed.append(pts[-1])
        pts = smoothed

    return [(float(point[0]), float(point[1])) for point in pts]


def _build_precise_static_map(world, carla_map):
    topology_polylines = _build_lane_centerlines(
        carla_map,
        ROAD_CENTERLINE_SAMPLE_STEP,
    )

    segments = _polyline_segments(topology_polylines)
    if len(segments) > 0:
        x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy
        seg_l2_safe = np.where(seg_l2 == 0, 1e-8, seg_l2)

    buildings = []
    all_buildings = world.get_environment_objects(carla.CityObjectLabel.Buildings)
    for building in all_buildings:
        bb = building.bounding_box
        corners = _environment_object_bbox_footprint(bb)

        building_length = float(bb.extent.x * 2.0)
        building_width = float(bb.extent.y * 2.0)
        building_height = float(bb.extent.z * 2.0)
        footprint_area = float(_polygon_area(corners))
        building_volume = float(footprint_area * building_height)

        if (
            building_height < MIN_BUILDING_HEIGHT or
            footprint_area < MIN_BUILDING_FOOTPRINT_AREA or
            building_volume < MIN_BUILDING_VOLUME
        ):
            continue

        is_close_to_road = False
        if len(segments) > 0:
            sampled_points = _sample_polygon_perimeter(
                corners,
                ROAD_ADJACENCY_SAMPLE_SPACING,
            )

            for px, py in sampled_points:
                t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_l2_safe
                t = np.clip(t, 0, 1)

                closest_x = x1 + t * seg_dx
                closest_y = y1 + t * seg_dy
                dists = np.hypot(px - closest_x, py - closest_y)

                if np.min(dists) <= ROAD_PROXIMITY_THRESHOLD:
                    is_close_to_road = True
                    break

        if is_close_to_road or len(segments) == 0:
            buildings.append({
                "id": int(building.id),
                "footprint": [
                    {"x": float(x), "y": float(y)}
                    for x, y in corners
                ],
            })

    return topology_polylines, buildings


def main():
    print("Connecting to CARLA server...")
    try:
        client = carla.Client(CARLA_SERVER_HOST, CARLA_SERVER_PORT)
        client.set_timeout(10.0)
        world = client.get_world()
        carla_map = world.get_map()
    except Exception as e:
        print(f"Error connecting to CARLA: {e}")
        return

    print("Building precise road and building map layers...")
    try:
        topology_polylines, buildings = _build_precise_static_map(world, carla_map)
    except Exception as e:
        print(f"Error building static map layers: {e}")
        return

    print("Fetching traffic light actors...")
    traffic_lights = world.get_actors().filter('traffic.traffic_light')

    data_to_save = []
    x_coords = []
    y_coords = []
    ids = []

    for tl in traffic_lights:
        loc = tl.get_location()
        tl_id = tl.id

        data_to_save.append({
            "id": tl_id,
            "location": {
                "x": float(loc.x),
                "y": float(loc.y),
                "z": float(loc.z),
            },
        })

        x_coords.append(float(loc.x))
        y_coords.append(float(loc.y))
        ids.append(tl_id)

    output_file = os.path.join(OUTPUT_DIR, "traffic_lights_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)
    print(f"Saved {len(data_to_save)} traffic lights to '{output_file}'")

    print("Generating map visualization (this may take a few seconds)...")
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(15, 15))

        for building in buildings:
            poly = Polygon(
                [(p["x"], p["y"]) for p in building["footprint"]],
                closed=True,
                facecolor='#2a2a2a',
                edgecolor='#404040',
                alpha=0.9,
                zorder=2,
            )
            ax.add_patch(poly)

        for polyline in topology_polylines:
            if len(polyline) < 2:
                continue
            draw_polyline = _smooth_polyline_for_plot(
                polyline,
                ROAD_CENTERLINE_SMOOTHING_PASSES,
            )
            xs = [point[0] for point in draw_polyline]
            ys = [point[1] for point in draw_polyline]
            ax.plot(
                xs,
                ys,
                color='#555555',
                linewidth=1.5,
                alpha=0.8,
                zorder=3,
                solid_joinstyle='round',
                solid_capstyle='round',
            )

        ax.scatter(
            x_coords,
            y_coords,
            c='#ff595e',
            s=80,
            label='Traffic Lights',
            edgecolors='white',
            linewidths=0.7,
            zorder=5,
        )

        for i, tl_id in enumerate(ids):
            ax.annotate(
                f"TL {tl_id}",
                (x_coords[i], y_coords[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='#ff9aa2',
                zorder=6,
            )

        map_label = carla_map.name or "CARLA"
        ax.set_title(f"{map_label} - Traffic Light Locations", fontsize=18, pad=20)
        ax.set_xlabel("X Coordinate (meters)", fontsize=14)
        ax.set_ylabel("Y Coordinate (meters)", fontsize=14)
        ax.invert_yaxis()
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.2)
        if ids:
            ax.legend(loc='upper right', fontsize=12)

        img_output = os.path.join(OUTPUT_DIR, "traffic_lights_map.png")
        fig.savefig(img_output, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Rendered {len(topology_polylines)} road polylines and {len(buildings)} buildings.")
    print(f"Visualization saved successfully to '{img_output}'")
    
if __name__ == '__main__':
    main()
