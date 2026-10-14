"""Opt-in offline installation revision; frozen production modules are untouched.

p025_world_yaw.v1: sin/cos of CARLA world bbox/actor yaw, radians;
size_x is length along local +X, size_y width along local +Y. No camera
rotation or map offset applies. Legacy packets deliberately use the old path.
This module does not start a server or connect to any runtime.
"""
import math
import time

CONTRACT = 'p025_world_yaw.v1'

def wrap(degrees):
    return (degrees + 180.) % 360. - 180.

def heading(record, contract=CONTRACT):
    result = dict(schema='spatial_map_heading.v2', contract=contract, valid=False,
                  yaw_deg=None, space='CARLA_WORLD_XY', axis='length +X; +yaw toward +Y')
    if contract != CONTRACT:
        return dict(result, reason='UNSUPPORTED_HEADING_CONTRACT')
    try:
        s, c = float(record['yaw_sin']), float(record['yaw_cos'])
        if not all(map(math.isfinite, (s, c))) or math.hypot(s, c) < 1e-6:
            return dict(result, reason='INVALID_OR_DEGENERATE_YAW_PAIR')
        yaw = wrap(math.degrees(math.atan2(s, c)))
        # Within this explicit contract yaw_deg is world yaw. A legacy model
        # representation, if supplied too, must declare its offset explicitly.
        alternatives = []
        if 'yaw_deg' in record:
            alternatives.append(float(record['yaw_deg']))
        if 'model_yaw_deg' in record:
            if 'map_yaw_offset_deg' not in record:
                return dict(result, reason='AMBIGUOUS_MODEL_YAW')
            alternatives.append(float(record['model_yaw_deg']) + float(record['map_yaw_offset_deg']))
        if any(not math.isfinite(v) or abs(wrap(v-yaw)) > 1e-4 for v in alternatives):
            return dict(result, reason='CONFLICTING_HEADING_REPRESENTATIONS')
        return dict(result, valid=True, yaw_deg=yaw, reason='PRODUCER_WORLD_YAW')
    except (KeyError, ValueError, TypeError, OverflowError):
        return dict(result, reason='MISSING_OR_INVALID_YAW_PAIR')

def normalize_packet_v2(baseline, payload, received_at, *, contract=None):
    """Reuse the real baseline normalization, then revise only heading fields.

    No module monkeypatch. The caller must explicitly opt into the producer
    contract; legacy packets retain the baseline's entire output unchanged.
    """
    normalized = baseline._normalize_packet(payload, received_at)
    if contract is None:
        return normalized
    inputs = payload.get('objects', [])
    # Match the baseline's own score/class normalization and original indices,
    # including rejected records; never zip a filtered output to raw inputs.
    accepted = []
    for i, raw in enumerate(inputs):
        if isinstance(raw, dict):
            check = baseline._normalize_object(raw, stream_id=normalized['stream_id'],
                                              frame_id=normalized['frame_id'], index=i)
            if check is not None:
                accepted.append(raw)
    if len(accepted) != len(normalized['objects']):
        raise ValueError('normalization inventory changed')
    for obj, raw in zip(normalized['objects'], accepted):
        h = heading(raw, contract)
        obj.update(yaw_deg=h['yaw_deg'], model_yaw_deg=h['yaw_deg'],
                   map_yaw_offset_deg=0., heading=h)
    return normalized

def install_under_state_lock_v2(document, ingest_at, *, contract=None, baseline=None):
    """Same normalization and locked insertion seam as direct-map v1.

    Opt-in revision for independent integration review; tests use the actual
    imported baseline state in a dedicated offline process, never a live map.
    """
    if baseline is None:
        from rl_agent.splitfusion_direct_edge_map_v1.spatial_map_direct_server_v1 import baseline
    payload = dict(schema=baseline.SPATIAL_STREAM_SCHEMA, stream_id=str(document['stream_id']),
        frame_id=int(document['frame_id']), capture_id=f"{document['stream_id']}:{int(document['frame_id'])}",
        capture_timestamp=int(document['capture_timestamp_ns'])/1e9, action_id=str(document['action_id']),
        carla_timestamp=float(document.get('carla_timestamp') or 0.), objects=list(document.get('records') or ()),
        segmentation=dict(document.get('segmentation') or {}), timing=dict(document.get('edge_timing') or {}),
        source_script='splitfusion_direct_edge_map_v1')
    normalized = normalize_packet_v2(baseline, payload, ingest_at, contract=contract)
    association_end_at = time.time()
    normalized['install_timestamp'] = association_end_at
    key = normalized['stream_id'], normalized['frame_id']
    lock_request_at = time.time()
    with baseline.state_lock:
        lock_acquired_at = time.time()
        baseline.latest_streams[key[0]] = normalized
        baseline.installed_frame_history[key] = normalized
        baseline.installed_frame_history.move_to_end(key)
        limit = max(1, int(baseline._config().installed_frame_history_size))
        while len(baseline.installed_frame_history) > limit:
            baseline.installed_frame_history.popitem(last=False)
    return dict(install_timestamp=association_end_at, object_count=normalized['object_count'],
                association_end_at=association_end_at, map_lock_request_at=lock_request_at,
                map_lock_acquired_at=lock_acquired_at, map_lock_released_at=time.time())
