#!/usr/bin/env python3

"""CARLA-world metadata for the Physical AI network-degradation model.

``spawn_blocker_v5.py`` publishes a small transactional profile with invisible,
parentless ``sensor.other.gnss`` actors.  Zone actors are created first and a
single manifest actor is created last as the commit marker.  Display clients
accept a profile only when its version, session token, zone count, zone
indices, coordinates, and radii are all complete and unambiguous.

The metadata sensors are never subscribed with ``listen()``.  They do not
alter world settings, advance the simulation clock, render geometry, collide,
or contribute camera/radar markers to the spatial map.
"""

import math
import re
import secrets
from dataclasses import dataclass
from typing import Optional, Tuple


NETWORK_PROFILE_VERSION = 1
NETWORK_PROFILE_BLUEPRINT_ID = "sensor.other.gnss"
NETWORK_PROFILE_ROLE_PREFIX = "sb5_ndp_v1_"
NETWORK_PROFILE_MANIFEST_PREFIX = NETWORK_PROFILE_ROLE_PREFIX + "m_"
NETWORK_PROFILE_ZONE_PREFIX = NETWORK_PROFILE_ROLE_PREFIX + "z_"
NETWORK_PROFILE_TOKEN_HEX_LENGTH = 8
NETWORK_PROFILE_MAX_ZONES = 2
NETWORK_PROFILE_MAX_RADIUS_M = 10000.0
NETWORK_PROFILE_SENSOR_TICK_SECONDS = 999999.0

NETWORK_PROFILE_VEHICLE_ZONE_INDEX = 1
NETWORK_PROFILE_PEDESTRIAN_ZONE_INDEX = 2
NETWORK_PROFILE_ZONE_LABELS = {
    NETWORK_PROFILE_VEHICLE_ZONE_INDEX: "ego_vehicle",
    NETWORK_PROFILE_PEDESTRIAN_ZONE_INDEX: "ego_pedestrian",
}

_TOKEN_PATTERN = r"[0-9a-f]{%d}" % NETWORK_PROFILE_TOKEN_HEX_LENGTH
_MANIFEST_PATTERN = re.compile(
    r"^{}(?P<token>{})_n(?P<count>[0-2])_s(?P<stream>[01])$".format(
        re.escape(NETWORK_PROFILE_MANIFEST_PREFIX),
        _TOKEN_PATTERN,
    )
)
_ZONE_PATTERN = re.compile(
    r"^{}(?P<token>{})_i(?P<index>[12])_r"
    r"(?P<radius>(?:0|[1-9][0-9]*)(?:\.[0-9]+)?)$".format(
        re.escape(NETWORK_PROFILE_ZONE_PREFIX),
        _TOKEN_PATTERN,
    )
)


class NetworkProfileError(ValueError):
    """Raised when CARLA contains an incomplete or ambiguous profile."""


@dataclass(frozen=True)
class NetworkDegradationZone:
    """One enabled circular network-degradation zone in CARLA XY metres."""

    index: int
    x: float
    y: float
    radius: float

    @property
    def label(self) -> str:
        return NETWORK_PROFILE_ZONE_LABELS[self.index]

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.radius


@dataclass(frozen=True)
class NetworkProfileManifest:
    session_token: str
    expected_zone_count: int
    start_active_sensors: bool


@dataclass(frozen=True)
class NetworkProfileZoneMetadata:
    session_token: str
    index: int
    radius: float


@dataclass(frozen=True)
class NetworkDegradationProfile:
    """A fully validated profile discovered from one CARLA world."""

    session_token: str
    zones: Tuple[NetworkDegradationZone, ...]
    start_active_sensors: bool
    manifest_actor_id: int
    zone_actor_ids: Tuple[int, ...]

    @property
    def zone_tuples(self) -> Tuple[Tuple[float, float, float], ...]:
        return tuple(zone.as_tuple() for zone in self.zones)


def _finite_float(value, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NetworkProfileError("{} must be a number".format(field_name)) from exc
    if not math.isfinite(parsed):
        raise NetworkProfileError("{} must be finite".format(field_name))
    return parsed


def _exact_integer(value, field_name: str) -> int:
    """Parse an integer without silently truncating fractional values."""
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NetworkProfileError("{} must be an integer".format(field_name)) from exc
    if not math.isfinite(numeric) or numeric != float(parsed):
        raise NetworkProfileError("{} must be an integer".format(field_name))
    return parsed


def normalize_zone(index, x, y, radius) -> NetworkDegradationZone:
    """Validate one configured zone and return its canonical representation."""
    index = _exact_integer(index, "zone index")
    if index not in NETWORK_PROFILE_ZONE_LABELS:
        raise NetworkProfileError(
            "zone index must be one of {}".format(
                ", ".join(str(value) for value in sorted(NETWORK_PROFILE_ZONE_LABELS))
            )
        )
    x = _finite_float(x, "zone x")
    y = _finite_float(y, "zone y")
    radius = _finite_float(radius, "zone radius")
    if radius <= 0.0 or radius > NETWORK_PROFILE_MAX_RADIUS_M:
        raise NetworkProfileError(
            "zone radius must be greater than zero and at most {:.1f} m".format(
                NETWORK_PROFILE_MAX_RADIUS_M
            )
        )
    return NetworkDegradationZone(index=index, x=x, y=y, radius=radius)


def new_session_token() -> str:
    """Return the short lowercase token carried by every actor in one profile."""
    return secrets.token_hex(NETWORK_PROFILE_TOKEN_HEX_LENGTH // 2)


def _validate_session_token(session_token) -> str:
    token = str(session_token)
    if re.fullmatch(_TOKEN_PATTERN, token) is None:
        raise NetworkProfileError(
            "session token must contain exactly {} lowercase hexadecimal "
            "characters".format(NETWORK_PROFILE_TOKEN_HEX_LENGTH)
        )
    return token


def build_manifest_role(
    session_token,
    expected_zone_count,
    start_active_sensors,
) -> str:
    token = _validate_session_token(session_token)
    expected_zone_count = _exact_integer(expected_zone_count, "zone count")
    if not 0 <= expected_zone_count <= NETWORK_PROFILE_MAX_ZONES:
        raise NetworkProfileError(
            "zone count must be between zero and {}".format(
                NETWORK_PROFILE_MAX_ZONES
            )
        )
    return "{}{}_n{}_s{}".format(
        NETWORK_PROFILE_MANIFEST_PREFIX,
        token,
        expected_zone_count,
        1 if bool(start_active_sensors) else 0,
    )


def build_zone_role(session_token, index, radius) -> str:
    token = _validate_session_token(session_token)
    zone = normalize_zone(index, 0.0, 0.0, radius)
    return "{}{}_i{}_r{:.6f}".format(
        NETWORK_PROFILE_ZONE_PREFIX,
        token,
        zone.index,
        zone.radius,
    )


def parse_manifest_role(role_name) -> Optional[NetworkProfileManifest]:
    """Parse a manifest role, returning ``None`` for unrelated actors."""
    role_name = str(role_name)
    if not role_name.startswith(NETWORK_PROFILE_ROLE_PREFIX):
        return None
    match = _MANIFEST_PATTERN.fullmatch(role_name)
    if match is None:
        if role_name.startswith(NETWORK_PROFILE_MANIFEST_PREFIX):
            raise NetworkProfileError(
                "malformed network-profile manifest role {!r}".format(role_name)
            )
        return None
    return NetworkProfileManifest(
        session_token=match.group("token"),
        expected_zone_count=int(match.group("count")),
        start_active_sensors=match.group("stream") == "1",
    )


def parse_zone_role(role_name) -> Optional[NetworkProfileZoneMetadata]:
    """Parse a zone role, returning ``None`` for unrelated actors."""
    role_name = str(role_name)
    if not role_name.startswith(NETWORK_PROFILE_ROLE_PREFIX):
        return None
    match = _ZONE_PATTERN.fullmatch(role_name)
    if match is None:
        if role_name.startswith(NETWORK_PROFILE_ZONE_PREFIX):
            raise NetworkProfileError(
                "malformed network-profile zone role {!r}".format(role_name)
            )
        return None
    radius = _finite_float(match.group("radius"), "zone radius")
    if radius <= 0.0 or radius > NETWORK_PROFILE_MAX_RADIUS_M:
        raise NetworkProfileError(
            "zone radius in role {!r} is outside 0..{:.1f} m".format(
                role_name,
                NETWORK_PROFILE_MAX_RADIUS_M,
            )
        )
    return NetworkProfileZoneMetadata(
        session_token=match.group("token"),
        index=int(match.group("index")),
        radius=radius,
    )


def _profile_actor_records(world):
    try:
        actors = world.get_actors().filter(NETWORK_PROFILE_BLUEPRINT_ID)
    except (AttributeError, RuntimeError) as exc:
        raise NetworkProfileError(
            "unable to enumerate CARLA network-profile actors: {}".format(exc)
        ) from exc
    records = []
    for actor in actors:
        try:
            if not actor.is_alive:
                continue
            role_name = str(actor.attributes.get("role_name", ""))
            if not role_name.startswith(NETWORK_PROFILE_ROLE_PREFIX):
                continue
            records.append((int(actor.id), role_name, actor))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return tuple(sorted(records, key=lambda entry: entry[0]))


def find_profile_actor_conflicts(world) -> Tuple[Tuple[int, str], ...]:
    """Return every pre-existing actor claiming the shared profile prefix."""
    return tuple(
        (actor_id, role_name)
        for actor_id, role_name, _actor in _profile_actor_records(world)
    )


def discover_network_degradation_profile(
    world,
    strict: bool = True,
) -> Optional[NetworkDegradationProfile]:
    """Discover one complete committed profile from ``world``.

    ``None`` means no publisher is present. In strict mode, any malformed,
    partial, duplicate, or mixed-session actor set raises ``NetworkProfileError``.
    Non-strict mode treats those states as no published profile.
    """
    try:
        records = _profile_actor_records(world)
        if not records:
            return None

        manifests = []
        zones = []
        for actor_id, role_name, actor in records:
            manifest = parse_manifest_role(role_name)
            if manifest is not None:
                manifests.append((actor_id, manifest, actor))
                continue
            zone_metadata = parse_zone_role(role_name)
            if zone_metadata is not None:
                zones.append((actor_id, zone_metadata, actor))
                continue
            raise NetworkProfileError(
                "unrecognized actor role using reserved profile prefix: {!r}".format(
                    role_name
                )
            )

        if len(manifests) != 1:
            raise NetworkProfileError(
                "expected exactly one network-profile manifest, found {}".format(
                    len(manifests)
                )
            )
        manifest_actor_id, manifest, _manifest_actor = manifests[0]
        if len(zones) != manifest.expected_zone_count:
            raise NetworkProfileError(
                "manifest expects {} zone actor(s), found {}".format(
                    manifest.expected_zone_count,
                    len(zones),
                )
            )

        discovered_zones = []
        zone_actor_ids = []
        seen_indices = set()
        for actor_id, metadata, actor in zones:
            if metadata.session_token != manifest.session_token:
                raise NetworkProfileError(
                    "zone actor {} uses session token {}, expected {}".format(
                        actor_id,
                        metadata.session_token,
                        manifest.session_token,
                    )
                )
            if metadata.index in seen_indices:
                raise NetworkProfileError(
                    "duplicate network-profile zone index {}".format(metadata.index)
                )
            seen_indices.add(metadata.index)
            try:
                transform = actor.get_transform()
                location = transform.location
                zone = normalize_zone(
                    metadata.index,
                    location.x,
                    location.y,
                    metadata.radius,
                )
            except (AttributeError, RuntimeError) as exc:
                raise NetworkProfileError(
                    "unable to read zone actor {} transform: {}".format(
                        actor_id,
                        exc,
                    )
                ) from exc
            discovered_zones.append(zone)
            zone_actor_ids.append(actor_id)

        discovered_zones.sort(key=lambda zone: zone.index)
        return NetworkDegradationProfile(
            session_token=manifest.session_token,
            zones=tuple(discovered_zones),
            start_active_sensors=manifest.start_active_sensors,
            manifest_actor_id=manifest_actor_id,
            zone_actor_ids=tuple(sorted(zone_actor_ids)),
        )
    except NetworkProfileError:
        if strict:
            raise
        return None
