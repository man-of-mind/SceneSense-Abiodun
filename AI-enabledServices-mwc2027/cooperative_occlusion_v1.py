#!/usr/bin/env python3
"""Pure 2-D field-of-view and cooperative-occlusion geometry.

This module deliberately has no CARLA, OpenCV, or pygame dependency.  It is
used by ``manual_control_ar_v13.py`` to reason over display-only virtual sensor
metadata; it never creates, starts, or subscribes to a simulator sensor.

The model is a deterministic 2-D baseline: camera/radar horizontal fields of
view are intersected with opaque actor/building footprints.  Target visibility
is evaluated over deterministic footprint probes instead of only the actor
centre: any in-FoV probe with a clear line of sight makes the target visible to
that sensor.  Either the ego pair or a peer pair may therefore detect a target;
the target's own virtual pair is never allowed to detect itself.
"""

from __future__ import division

import math


GEOMETRY_EPSILON = 1.0e-7
TARGET_FOOTPRINT_EDGE_SUBDIVISIONS = 4


def normalize_angle_degrees(angle_degrees):
    """Return a finite angle in the half-open interval [-180, 180)."""
    angle = float(angle_degrees)
    if not math.isfinite(angle):
        raise ValueError('angle must be finite')
    return (angle + 180.0) % 360.0 - 180.0


def point_in_sensor_fov_xy(
        sensor_xy,
        yaw_degrees,
        target_xy,
        horizontal_fov_degrees,
        range_m,
        epsilon=GEOMETRY_EPSILON):
    """Return whether a target lies inside one bounded horizontal FoV."""
    sensor_x, sensor_y = (float(value) for value in sensor_xy)
    target_x, target_y = (float(value) for value in target_xy)
    fov = float(horizontal_fov_degrees)
    maximum_range = float(range_m)
    values = (
        sensor_x, sensor_y, target_x, target_y,
        float(yaw_degrees), fov, maximum_range)
    if not all(math.isfinite(value) for value in values):
        return False
    if fov <= 0.0 or fov > 180.0 or maximum_range <= 0.0:
        return False
    delta_x = target_x - sensor_x
    delta_y = target_y - sensor_y
    distance = math.hypot(delta_x, delta_y)
    if distance > maximum_range + float(epsilon):
        return False
    if distance <= float(epsilon):
        return True
    bearing = math.degrees(math.atan2(delta_y, delta_x))
    offset = normalize_angle_degrees(bearing - float(yaw_degrees))
    return abs(offset) <= (0.5 * fov) + float(epsilon)


def sensor_fov_polygon_xy(
        sensor_xy,
        yaw_degrees,
        horizontal_fov_degrees,
        range_m,
        arc_segments=24):
    """Return a fan polygon suitable for a top-down FoV overlay."""
    sensor_x, sensor_y = (float(value) for value in sensor_xy)
    yaw = float(yaw_degrees)
    fov = float(horizontal_fov_degrees)
    maximum_range = float(range_m)
    if not all(math.isfinite(value) for value in (
            sensor_x, sensor_y, yaw, fov, maximum_range)):
        raise ValueError('field-of-view inputs must be finite')
    if fov <= 0.0 or fov > 180.0:
        raise ValueError('horizontal field of view must be in (0, 180]')
    if maximum_range <= 0.0:
        raise ValueError('sensor range must be greater than zero')
    segments = max(2, int(arc_segments))
    points = [(sensor_x, sensor_y)]
    start_angle = yaw - 0.5 * fov
    for index in range(segments + 1):
        fraction = float(index) / float(segments)
        angle = math.radians(start_angle + fov * fraction)
        points.append((
            sensor_x + maximum_range * math.cos(angle),
            sensor_y + maximum_range * math.sin(angle),
        ))
    return tuple(points)


def _point_on_segment(point, start, end, epsilon=GEOMETRY_EPSILON):
    point_x, point_y = point
    start_x, start_y = start
    end_x, end_y = end
    cross = (
        (point_x - start_x) * (end_y - start_y)
        - (point_y - start_y) * (end_x - start_x))
    if abs(cross) > epsilon:
        return False
    dot = (
        (point_x - start_x) * (end_x - start_x)
        + (point_y - start_y) * (end_y - start_y))
    if dot < -epsilon:
        return False
    squared_length = (
        (end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    return dot <= squared_length + epsilon


def _point_segment_distance(point, start, end):
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    squared_length = delta_x * delta_x + delta_y * delta_y
    if squared_length <= GEOMETRY_EPSILON:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y) / squared_length
    fraction = max(0.0, min(1.0, fraction))
    closest_x = start[0] + fraction * delta_x
    closest_y = start[1] + fraction * delta_y
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def point_in_polygon_xy(point_xy, polygon, epsilon=GEOMETRY_EPSILON):
    """Return True for points inside or on the boundary of a polygon."""
    point = tuple(float(value) for value in point_xy)
    vertices = [tuple(float(value) for value in vertex[:2])
                for vertex in polygon]
    if len(vertices) < 3:
        return False
    inside = False
    point_x, point_y = point
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        if _point_on_segment(point, start, end, epsilon):
            return True
        start_x, start_y = start
        end_x, end_y = end
        crosses = ((start_y > point_y) != (end_y > point_y))
        if not crosses:
            continue
        crossing_x = (
            start_x
            + (point_y - start_y) * (end_x - start_x)
            / (end_y - start_y))
        if point_x < crossing_x:
            inside = not inside
    return inside


def _segment_edge_intersection_fraction(
        ray_start,
        ray_end,
        edge_start,
        edge_end,
        epsilon=GEOMETRY_EPSILON):
    """Return ray-segment fraction t for an inclusive edge hit, else None."""
    px, py = ray_start
    rx = ray_end[0] - px
    ry = ray_end[1] - py
    qx, qy = edge_start
    sx = edge_end[0] - qx
    sy = edge_end[1] - qy
    denominator = rx * sy - ry * sx
    q_minus_p_x = qx - px
    q_minus_p_y = qy - py
    if abs(denominator) <= epsilon:
        # Collinear overlap is an opaque/tangent hit.  Project both edge
        # endpoints onto the query segment and return the earliest overlap.
        if abs(q_minus_p_x * ry - q_minus_p_y * rx) > epsilon:
            return None
        ray_length_squared = rx * rx + ry * ry
        if ray_length_squared <= epsilon:
            return None
        fractions = (
            (q_minus_p_x * rx + q_minus_p_y * ry) / ray_length_squared,
            ((edge_end[0] - px) * rx +
             (edge_end[1] - py) * ry) / ray_length_squared,
        )
        overlap_start = max(0.0, min(fractions))
        overlap_end = min(1.0, max(fractions))
        if overlap_start <= overlap_end + epsilon:
            return overlap_start
        return None
    t_value = (q_minus_p_x * sy - q_minus_p_y * sx) / denominator
    u_value = (q_minus_p_x * ry - q_minus_p_y * rx) / denominator
    if (
            -epsilon <= t_value <= 1.0 + epsilon
            and -epsilon <= u_value <= 1.0 + epsilon):
        return max(0.0, min(1.0, t_value))
    return None


def segment_blocked_before_target(
        sensor_xy,
        target_xy,
        occluders,
        excluded_actor_ids=(),
        epsilon=GEOMETRY_EPSILON):
    """Return True if an opaque footprint intersects before the target.

    ``occluders`` is an iterable of dictionaries containing ``polygon`` and an
    optional ``actor_id``.  Actor IDs in ``excluded_actor_ids`` are ignored so
    neither the sensor host nor the target can occlude its own sight line.
    """
    sensor = tuple(float(value) for value in sensor_xy)
    target = tuple(float(value) for value in target_xy)
    segment_bounds = (
        min(sensor[0], target[0]),
        min(sensor[1], target[1]),
        max(sensor[0], target[0]),
        max(sensor[1], target[1]),
    )
    excluded = set(int(value) for value in excluded_actor_ids
                   if value is not None)
    for occluder in occluders:
        actor_id = occluder.get('actor_id')
        try:
            if actor_id is not None and int(actor_id) in excluded:
                continue
        except (TypeError, ValueError):
            pass
        polygon = [tuple(float(value) for value in vertex[:2])
                   for vertex in occluder.get('polygon', ())]
        if len(polygon) < 3:
            continue
        try:
            bounds = tuple(float(value) for value in occluder['bounds'])
        except (KeyError, TypeError, ValueError):
            bounds = (
                min(vertex[0] for vertex in polygon),
                min(vertex[1] for vertex in polygon),
                max(vertex[0] for vertex in polygon),
                max(vertex[1] for vertex in polygon),
            )
        if (
                bounds[2] < segment_bounds[0] - epsilon
                or bounds[0] > segment_bounds[2] + epsilon
                or bounds[3] < segment_bounds[1] - epsilon
                or bounds[1] > segment_bounds[3] + epsilon):
            continue
        if point_in_polygon_xy(sensor, polygon, epsilon):
            return True
        if point_in_polygon_xy(target, polygon, epsilon):
            return True
        for edge_start, edge_end in zip(
                polygon, polygon[1:] + polygon[:1]):
            fraction = _segment_edge_intersection_fraction(
                sensor, target, edge_start, edge_end, epsilon)
            if fraction is None:
                continue
            if epsilon < fraction < 1.0 - epsilon:
                return True
    return False


def occlusion_shadow_polygon_xy(
        sensor_xy,
        yaw_degrees,
        horizontal_fov_degrees,
        range_m,
        occluder_polygon):
    """Approximate the in-FoV shadow cast by one opaque 2-D footprint."""
    sensor_x, sensor_y = (float(value) for value in sensor_xy)
    yaw = float(yaw_degrees)
    fov = float(horizontal_fov_degrees)
    maximum_range = float(range_m)
    vertices = [tuple(float(value) for value in vertex[:2])
                for vertex in occluder_polygon]
    if len(vertices) < 3 or fov <= 0.0 or fov > 180.0 or maximum_range <= 0.0:
        return ()
    deltas = []
    for vertex_x, vertex_y in vertices:
        delta_x = vertex_x - sensor_x
        delta_y = vertex_y - sensor_y
        deltas.append(normalize_angle_degrees(
            math.degrees(math.atan2(delta_y, delta_x)) - yaw))
    sensor_point = (sensor_x, sensor_y)
    minimum_distance = min(
        _point_segment_distance(sensor_point, start, end)
        for start, end in zip(vertices, vertices[1:] + vertices[:1]))
    if minimum_distance >= maximum_range - GEOMETRY_EPSILON:
        return ()
    half_fov = 0.5 * fov
    inside = [delta for delta in deltas
              if abs(delta) <= half_fov + GEOMETRY_EPSILON]
    centroid_x = sum(vertex[0] for vertex in vertices) / len(vertices)
    centroid_y = sum(vertex[1] for vertex in vertices) / len(vertices)
    centroid_delta = normalize_angle_degrees(
        math.degrees(math.atan2(
            centroid_y - sensor_y, centroid_x - sensor_x)) - yaw)
    angular_minimum = min(deltas)
    angular_maximum = max(deltas)
    angular_span = angular_maximum - angular_minimum
    if point_in_polygon_xy((sensor_x, sensor_y), vertices):
        left_delta = -half_fov
        right_delta = half_fov
    elif angular_span <= 180.0 + GEOMETRY_EPSILON:
        # Intersect the complete polygon bearing interval with the FoV. This
        # preserves the shadow of an occluder that crosses a cone boundary
        # even when only one of its vertices is inside the cone.
        left_delta = max(-half_fov, angular_minimum)
        right_delta = min(half_fov, angular_maximum)
        if right_delta < left_delta:
            return ()
    elif inside:
        # The polygon wraps across the +/-180 discontinuity. Only retain any
        # portion explicitly inside this forward-facing cone.
        left_delta = min(inside)
        right_delta = max(inside)
    elif abs(centroid_delta) <= half_fov:
        left_delta = -half_fov
        right_delta = half_fov
    else:
        return ()
    if right_delta < left_delta:
        return ()
    near_distance = max(0.05, minimum_distance)

    def point_at(relative_angle, distance):
        angle = math.radians(yaw + relative_angle)
        return (
            sensor_x + distance * math.cos(angle),
            sensor_y + distance * math.sin(angle),
        )

    return (
        point_at(left_delta, near_distance),
        point_at(left_delta, maximum_range),
        point_at(right_delta, maximum_range),
        point_at(right_delta, near_distance),
    )


def modality_limits(marker, modality_config):
    """Resolve and validate range/FoV for one marker modality."""
    modality = str(marker.get('modality', ''))
    config = modality_config.get(modality)
    if config is None:
        return None
    try:
        maximum_range = float(config['range_m'])
        horizontal_fov = float(config['horizontal_fov_degrees'])
    except (KeyError, TypeError, ValueError):
        return None
    if (
            not math.isfinite(maximum_range)
            or not math.isfinite(horizontal_fov)
            or maximum_range <= 0.0
            or horizontal_fov <= 0.0
            or horizontal_fov > 180.0):
        return None
    return horizontal_fov, maximum_range


def target_visibility_points_xy(
        target,
        edge_subdivisions=TARGET_FOOTPRINT_EDGE_SUBDIVISIONS,
        epsilon=GEOMETRY_EPSILON):
    """Return deterministic XY probes covering a target's ground footprint.

    The actor centre remains the first probe for the common fast path.  A valid
    ``polygon`` or ``footprint`` adds its centroid plus evenly spaced boundary
    and centre-to-boundary probes.  The latter make partial exposure observable
    when an occluder hides the centre but not the pedestrian's side.  Invalid or
    missing footprints safely retain the previous centre-point behaviour.
    """
    points = []

    def append_unique(point):
        try:
            candidate = tuple(float(value) for value in point[:2])
        except (TypeError, ValueError):
            return
        if len(candidate) != 2 or not all(
                math.isfinite(value) for value in candidate):
            return
        tolerance_squared = float(epsilon) ** 2
        if any(
                (candidate[0] - existing[0]) ** 2
                + (candidate[1] - existing[1]) ** 2
                <= tolerance_squared
                for existing in points):
            return
        points.append(candidate)

    try:
        centre = (float(target['x']), float(target['y']))
        if not all(math.isfinite(value) for value in centre):
            centre = None
    except (KeyError, TypeError, ValueError):
        centre = None
    if centre is not None:
        append_unique(centre)

    raw_polygon = target.get('footprint', target.get('polygon', ()))
    vertices = []
    try:
        for vertex in raw_polygon:
            candidate = tuple(float(value) for value in vertex[:2])
            if len(candidate) != 2 or not all(
                    math.isfinite(value) for value in candidate):
                vertices = []
                break
            vertices.append(candidate)
    except (TypeError, ValueError):
        vertices = []
    if len(vertices) < 3:
        return tuple(points)

    centroid = (
        sum(vertex[0] for vertex in vertices) / float(len(vertices)),
        sum(vertex[1] for vertex in vertices) / float(len(vertices)),
    )
    if centre is None:
        centre = centroid
        append_unique(centre)
    else:
        append_unique(centroid)

    try:
        subdivisions = max(1, int(edge_subdivisions))
    except (TypeError, ValueError):
        subdivisions = TARGET_FOOTPRINT_EDGE_SUBDIVISIONS
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        for index in range(subdivisions):
            fraction = float(index) / float(subdivisions)
            boundary_point = (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
            append_unique(boundary_point)
            append_unique((
                0.5 * (centre[0] + boundary_point[0]),
                0.5 * (centre[1] + boundary_point[1]),
            ))
    return tuple(points)


def evaluate_cooperative_target(
        active_markers,
        target,
        occluders,
        ego_parent_id,
        modality_config):
    """Evaluate one target against selected virtual camera/radar markers."""
    target_id = int(target['actor_id'])
    target_points = target_visibility_points_xy(target)
    ego_parent_id = int(ego_parent_id)
    observations = []
    for marker in active_markers:
        if not bool(marker.get('active', False)):
            continue
        try:
            parent_id = int(marker['parent_id'])
            marker_id = int(marker['actor_id'])
            sensor_xy = (float(marker['x']), float(marker['y']))
            yaw = float(marker.get('yaw', 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        # A wearable/mounted pair cannot be evidence for its own host object.
        if parent_id == target_id:
            continue
        limits = modality_limits(marker, modality_config)
        if limits is None:
            continue
        horizontal_fov, maximum_range = limits
        inside_fov_points = []
        visible_point = None
        blocked_point_count = 0
        evaluated_point_count = 0
        for target_point in target_points:
            evaluated_point_count += 1
            if not point_in_sensor_fov_xy(
                    sensor_xy,
                    yaw,
                    target_point,
                    horizontal_fov,
                    maximum_range):
                continue
            inside_fov_points.append(target_point)
            point_blocked = segment_blocked_before_target(
                sensor_xy,
                target_point,
                occluders,
                excluded_actor_ids=(parent_id, target_id))
            if point_blocked:
                blocked_point_count += 1
            else:
                visible_point = target_point
                # Visibility is existential. Avoid tracing the remaining
                # footprint probes in the 10 Hz display path once one clear
                # portion has been established.
                break
        inside_fov = bool(inside_fov_points)
        visible = visible_point is not None
        blocked = bool(inside_fov and not visible)
        observations.append({
            'marker_id': marker_id,
            'parent_id': parent_id,
            'site_key': tuple(marker.get('site_key', (parent_id,))),
            'modality': str(marker.get('modality', 'unknown')),
            'inside_fov': bool(inside_fov),
            'blocked': bool(blocked),
            'visible': visible,
            'target_point_count': len(target_points),
            'evaluated_point_count': evaluated_point_count,
            'tested_in_fov_point_count': len(inside_fov_points),
            'blocked_point_count': blocked_point_count,
            'visible_point': visible_point,
        })

    ego_observations = [
        observation for observation in observations
        if observation['parent_id'] == ego_parent_id]
    peer_observations = [
        observation for observation in observations
        if observation['parent_id'] != ego_parent_id]
    inside_ego_fov = any(
        observation['inside_fov'] for observation in ego_observations)
    visible_to_ego = any(
        observation['visible'] for observation in ego_observations)
    visible_to_peer = any(
        observation['visible'] for observation in peer_observations)
    blocked_from_ego = bool(inside_ego_fov and not visible_to_ego)
    # A peer-only observation is a cooperative reveal even when the target is
    # outside the ego FoV rather than geometrically shadowed inside it.
    cooperatively_detected = bool(visible_to_peer and not visible_to_ego)
    seen_by = tuple(sorted(
        (observation['parent_id'], observation['modality'])
        for observation in observations
        if observation['visible']))
    peer_seen_by = tuple(sorted(
        (observation['parent_id'], observation['modality'])
        for observation in peer_observations
        if observation['visible']))
    return {
        'actor_id': target_id,
        'inside_ego_fov': inside_ego_fov,
        'visible_to_ego': visible_to_ego,
        'blocked_from_ego': blocked_from_ego,
        'visible_to_peer': visible_to_peer,
        'network_detected': bool(visible_to_ego or visible_to_peer),
        'cooperatively_detected': cooperatively_detected,
        'seen_by': seen_by,
        'peer_seen_by': peer_seen_by,
        'observations': tuple(observations),
    }


def evaluate_cooperative_targets(
        markers,
        targets,
        occluders,
        ego_parent_id,
        modality_config):
    """Return a deterministic actor-ID keyed cooperative visibility snapshot."""
    active_markers = [marker for marker in markers
                      if bool(marker.get('active', False))]
    results = {}
    for target in sorted(targets, key=lambda item: int(item['actor_id'])):
        result = evaluate_cooperative_target(
            active_markers,
            target,
            occluders,
            ego_parent_id,
            modality_config)
        results[result['actor_id']] = result
    return results
