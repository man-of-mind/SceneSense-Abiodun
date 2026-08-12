#!/usr/bin/env python3
"""Read and write the shared CARLA ego-route configuration format.

The module intentionally has no dependency on :mod:`carla`, so both the
scenario-controller UI and a manual-control client can exchange routes without
requiring an active simulator connection.

Schema version 1 is a JSON object with the following core fields::

    {
      "schema_version": 1,
      "type": "carla_ego_route",
      "name": "Town10HD demo route",
      "map": "Carla/Maps/Town10HD_Opt",
      "coordinate_system": "carla_world_left_handed_meters",
      "route_sampling_resolution_m": 2.0,
      "start": {
        "location": {"x": 73.63, "y": 66.36, "z": 0.2},
        "rotation": {"pitch": 0.0, "yaw": 90.0, "roll": 0.0}
      },
      "intermediate_waypoints": [
        {"x": 65.0, "y": 55.0, "z": 0.2}
      ],
      "end": {
        "location": {"x": 40.0, "y": 35.0, "z": 0.2},
        "rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0}
      },
      "planned_path": [
        {"x": 73.63, "y": 66.36, "z": 0.2}
      ],
      "ui_selection": {}
    }

``planned_path`` and ``ui_selection`` are optional.  Unknown JSON-compatible
fields are retained so producers may attach harmless metadata without losing
it in a load/save round trip.  Required routing fields remain strict: booleans,
NaN, infinity, missing coordinates, and values of the wrong type are rejected.
"""

from __future__ import print_function

import json
import math
import os
import tempfile
from collections.abc import Mapping


ROUTE_SCHEMA_VERSION = 1
ROUTE_CONFIG_TYPE = "carla_ego_route"
ROUTE_COORDINATE_SYSTEM = "carla_world_left_handed_meters"


class RouteConfigError(ValueError):
    """Raised when an ego-route configuration cannot be read or validated."""


def _field_error(field_path, message):
    raise RouteConfigError("{}: {}".format(field_path, message))


def _json_value(value, field_path):
    """Return a detached, JSON-compatible copy of *value*.

    This is used for extension metadata.  It deliberately rejects non-string
    object keys and non-finite floats instead of relying on ``json.dump``'s
    permissive NaN handling.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _field_error(field_path, "numeric values must be finite")
        return value
    if isinstance(value, list):
        return [
            _json_value(item, "{}[{}]".format(field_path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _field_error(field_path, "object keys must be strings")
            result[key] = _json_value(item, "{}.{}".format(field_path, key))
        return result
    _field_error(
        field_path,
        "value of type {} is not JSON-compatible".format(type(value).__name__),
    )


def _object(value, field_path):
    if not isinstance(value, Mapping):
        _field_error(field_path, "must be a JSON object")
    return value


def _required_string(value, field_path):
    if not isinstance(value, str):
        _field_error(field_path, "must be a string")
    normalized = value.strip()
    if not normalized:
        _field_error(field_path, "must not be empty")
    return normalized


def _finite_number(value, field_path):
    # bool is a subclass of int and must be checked first.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _field_error(field_path, "must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        _field_error(field_path, "must be a finite number")
    if not math.isfinite(number):
        _field_error(field_path, "must be a finite number")
    return number


def _normalized_object_with_extras(value, field_path):
    source = _object(value, field_path)
    return _json_value(source, field_path)


def _location(value, field_path):
    result = _normalized_object_with_extras(value, field_path)
    for axis in ("x", "y", "z"):
        if axis not in value:
            _field_error("{}.{}".format(field_path, axis), "is required")
        result[axis] = _finite_number(
            value[axis], "{}.{}".format(field_path, axis)
        )
    return result


def _rotation(value, field_path):
    result = _normalized_object_with_extras(value, field_path)
    for axis in ("pitch", "yaw", "roll"):
        if axis not in value:
            _field_error("{}.{}".format(field_path, axis), "is required")
        result[axis] = _finite_number(
            value[axis], "{}.{}".format(field_path, axis)
        )
    return result


def _transform(value, field_path):
    source = _object(value, field_path)
    result = _normalized_object_with_extras(source, field_path)
    if "location" not in source:
        _field_error("{}.location".format(field_path), "is required")
    if "rotation" not in source:
        _field_error("{}.rotation".format(field_path), "is required")
    result["location"] = _location(
        source["location"], "{}.location".format(field_path)
    )
    result["rotation"] = _rotation(
        source["rotation"], "{}.rotation".format(field_path)
    )
    return result


def _location_list(value, field_path):
    if not isinstance(value, list):
        _field_error(field_path, "must be a JSON array")
    return [
        _location(item, "{}[{}]".format(field_path, index))
        for index, item in enumerate(value)
    ]


def validate_route_config(data):
    """Validate *data* and return a normalized, detached dictionary.

    All routing numbers are returned as ``float`` values.  Unknown fields are
    preserved when they are valid JSON data.  The input object is never mutated.
    """

    source = _object(data, "route_config")
    result = _normalized_object_with_extras(source, "route_config")

    if "schema_version" not in source:
        _field_error("schema_version", "is required")
    schema_version = source["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        _field_error("schema_version", "must be integer 1")
    if schema_version != ROUTE_SCHEMA_VERSION:
        _field_error(
            "schema_version",
            "unsupported value {}; expected {}".format(
                schema_version, ROUTE_SCHEMA_VERSION
            ),
        )
    result["schema_version"] = ROUTE_SCHEMA_VERSION

    if source.get("type") != ROUTE_CONFIG_TYPE:
        _field_error("type", "must be {!r}".format(ROUTE_CONFIG_TYPE))
    result["type"] = ROUTE_CONFIG_TYPE

    if "name" not in source:
        _field_error("name", "is required")
    result["name"] = _required_string(source["name"], "name")

    if "map" not in source:
        _field_error("map", "is required")
    result["map"] = _required_string(source["map"], "map")

    if source.get("coordinate_system") != ROUTE_COORDINATE_SYSTEM:
        _field_error(
            "coordinate_system",
            "must be {!r}".format(ROUTE_COORDINATE_SYSTEM),
        )
    result["coordinate_system"] = ROUTE_COORDINATE_SYSTEM

    if "route_sampling_resolution_m" not in source:
        _field_error("route_sampling_resolution_m", "is required")
    sampling_resolution = _finite_number(
        source["route_sampling_resolution_m"], "route_sampling_resolution_m"
    )
    if sampling_resolution <= 0.0:
        _field_error("route_sampling_resolution_m", "must be greater than zero")
    result["route_sampling_resolution_m"] = sampling_resolution

    if "start" not in source:
        _field_error("start", "is required")
    result["start"] = _transform(source["start"], "start")

    if "intermediate_waypoints" not in source:
        _field_error("intermediate_waypoints", "is required")
    result["intermediate_waypoints"] = _location_list(
        source["intermediate_waypoints"], "intermediate_waypoints"
    )

    if "end" not in source:
        _field_error("end", "is required")
    result["end"] = _transform(source["end"], "end")

    if "planned_path" in source:
        result["planned_path"] = _location_list(
            source["planned_path"], "planned_path"
        )

    if "ui_selection" in source:
        result["ui_selection"] = _normalized_object_with_extras(
            source["ui_selection"], "ui_selection"
        )

    return result


def normalize_map_name(map_name):
    """Return a case-normalized CARLA map basename for comparison.

    Only path separators, leading/trailing whitespace, trailing separators, and
    character case are normalized.  Suffixes such as ``_Opt`` remain meaningful,
    so ``Town10HD`` does not silently match ``Town10HD_Opt``.
    """

    value = _required_string(map_name, "map")
    value = value.replace("\\", "/").rstrip("/")
    if not value:
        _field_error("map", "must contain a map basename")
    basename = value.rsplit("/", 1)[-1]
    if not basename:
        _field_error("map", "must contain a map basename")
    return basename.casefold()


def maps_match(left_map, right_map):
    """Return whether two CARLA map names refer to the same map basename."""

    return normalize_map_name(left_map) == normalize_map_name(right_map)


def load_route_config(path):
    """Load *path* as UTF-8 JSON and return a validated normalized route."""

    route_path = os.fspath(path)
    try:
        with open(route_path, "r", encoding="utf-8") as route_file:
            data = json.load(route_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RouteConfigError(
            "could not load route config {!r}: {}".format(route_path, exc)
        ) from exc

    try:
        return validate_route_config(data)
    except RouteConfigError as exc:
        raise RouteConfigError(
            "invalid route config {!r}: {}".format(route_path, exc)
        ) from exc


def save_route_config(path, data):
    """Validate and atomically save *data* as UTF-8 JSON at *path*.

    The temporary file is created in the destination directory, flushed and
    synced, and then committed with :func:`os.replace`, preventing readers from
    observing a partially written route file.
    """

    route_path = os.path.abspath(os.fspath(path))
    destination_dir = os.path.dirname(route_path)
    if not os.path.isdir(destination_dir):
        raise RouteConfigError(
            "route config directory does not exist: {!r}".format(destination_dir)
        )

    normalized = validate_route_config(data)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination_dir,
            prefix=".{}.".format(os.path.basename(route_path)),
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(
                normalized,
                temporary_file,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, route_path)
        temporary_path = None
    except (OSError, TypeError, ValueError) as exc:
        raise RouteConfigError(
            "could not save route config {!r}: {}".format(route_path, exc)
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    return normalized


__all__ = [
    "ROUTE_SCHEMA_VERSION",
    "ROUTE_CONFIG_TYPE",
    "ROUTE_COORDINATE_SYSTEM",
    "RouteConfigError",
    "load_route_config",
    "maps_match",
    "normalize_map_name",
    "save_route_config",
    "validate_route_config",
]
