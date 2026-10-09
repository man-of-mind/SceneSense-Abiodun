"""Explicit route contract for the two-UE cooperative demonstration.

Why this exists
---------------
The existing owned demo drives spawn indices ``80,85,91,94,99,80``. That is the
older **Stage-2 loop**. It is not Route B, it has never been qualified as Route
B, and describing it as Route B would misattribute every result collected on it.

Two modes are therefore registered and they are not interchangeable:

``stage2``
    The historical configurable-spawn-index loop. ``is_route_b`` is ``False`` and
    ``presentation_label`` is fixed to a Stage-2 string. There is no argument
    combination that makes this mode claim Route B.

``route-b-passive``
    Binds the frozen canonical Town10HD_Opt Route-B loop JSON, verifies its
    SHA-256 against a pinned digest, optionally binds the route-progress
    contract, and requires **two explicitly registered ego/stream identities**.
    It is "passive" because it consumes a canonical route definition; it does not
    itself plan or drive one.

No Route-B run is authorized by this module. It contains loaders, hash binding,
validation and a presentation overlay only. A live Route-B qualification must be
driven by the canonical route runner, and a *complete two-UE Route-B
demonstration* cannot be claimed from configurable spawn indices, from this
binding, or from anything short of that run.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .normalized import CoopInputError, _require, nonempty_str


ROUTE_MODE_STAGE2 = "stage2"
ROUTE_MODE_ROUTE_B_PASSIVE = "route-b-passive"
ROUTE_MODES = (ROUTE_MODE_STAGE2, ROUTE_MODE_ROUTE_B_PASSIVE)

# The canonical frozen Route-B loop, repository-relative.
CANONICAL_ROUTE_B_RELPATH = "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json"
CANONICAL_ROUTE_B_PROGRESS_RELPATH = (
    "data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.progress.csv"
)

# Pinned digests observed 2026-09-21. The branch copy and the master copy of the
# route JSON are byte-identical at these values, which is what makes this binding
# meaningful rather than decorative.
CANONICAL_ROUTE_B_SHA256 = (
    "fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5"
)
CANONICAL_ROUTE_B_PROGRESS_SHA256 = (
    "974593859368f24ee2bc4ac31b82118bf2e932d0de1c96858b8771e2dd4d90c0"
)

EXPECTED_ROUTE_MAP = "Carla/Maps/Town10HD_Opt"
EXPECTED_ROUTE_TYPE = "carla_ego_route"
EXPECTED_ROUTE_SCHEMA_VERSION = 1
EXPECTED_ROUTE_NAME = "Town10HD_Opt Route B full-map loop v1"

MINIMUM_REGISTERED_STREAMS = 2

STAGE2_LABEL = "Stage-2 configurable spawn-index loop (NOT Route B)"
ROUTE_B_LABEL = "Town10HD_Opt canonical Route-B full-map loop v1 (bound, not yet run)"

# Guards against the exact mislabelling this module exists to prevent. A
# non-Route-B label may mention Route B only inside an explicit negation, so
# "(NOT Route B)" is allowed while "Route B loop" is refused.
STAGE2_FORBIDDEN_TERMS = ("route b", "route-b", "routeb")
NEGATED_MENTIONS = ("not route b", "not route-b", "not routeb")


def mentions_route_b_as_a_claim(label: str) -> bool:
    """True when ``label`` mentions Route B other than to deny it."""

    lowered = str(label).lower()
    for negation in NEGATED_MENTIONS:
        lowered = lowered.replace(negation, "")
    return any(term in lowered for term in STAGE2_FORBIDDEN_TERMS)

QUALIFICATION_STATUS_STAGE2 = "STAGE2_LOOP_NOT_ROUTE_B"
QUALIFICATION_STATUS_ROUTE_B = "ROUTE_B_CONTRACT_BOUND_LIVE_RUN_UNAUTHORIZED"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_root() -> Path:
    # .../abiodun/spatial_map_coop/coop_input_v1/route_contract_v1.py -> .../abiodun
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RouteWaypoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class CanonicalRoute:
    """A hash-verified canonical route definition."""

    name: str
    map_name: str
    route_type: str
    loop: bool
    coordinate_system: str
    schema_version: int
    path: tuple[RouteWaypoint, ...]
    intermediate_waypoints: tuple[RouteWaypoint, ...]
    sampling_resolution_m: float
    source_path: str
    sha256: str
    progress_path: str | None = None
    progress_sha256: str | None = None
    progress_samples: int = 0

    @property
    def waypoint_count(self) -> int:
        return len(self.path)

    def overlay_polyline(self, *, stride: int = 1) -> tuple[tuple[float, float], ...]:
        """Subsampled XY polyline for the optional background overlay.

        Presentation only: the overlay is a drawn reference path and never
        influences a measured coordinate or an association.
        """

        step = max(1, int(stride))
        points = tuple((item.x, item.y) for item in self.path[::step])
        if self.loop and points and points[0] != points[-1]:
            points = points + (points[0],)
        return points

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "map": self.map_name,
            "type": self.route_type,
            "loop": self.loop,
            "coordinate_system": self.coordinate_system,
            "schema_version": self.schema_version,
            "waypoint_count": self.waypoint_count,
            "intermediate_waypoint_count": len(self.intermediate_waypoints),
            "sampling_resolution_m": self.sampling_resolution_m,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "progress_path": self.progress_path,
            "progress_sha256": self.progress_sha256,
            "progress_samples": self.progress_samples,
        }


def _waypoints(values: Any, label: str) -> tuple[RouteWaypoint, ...]:
    _require(isinstance(values, list) and values, f"{label} must be a non-empty list")
    result = []
    for index, item in enumerate(values):
        _require(isinstance(item, Mapping), f"{label}[{index}] must be an object")
        coords = []
        for axis in ("x", "y", "z"):
            _require(axis in item, f"{label}[{index}] lacks {axis}")
            value = item[axis]
            _require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"{label}[{index}].{axis} must be a number",
            )
            numeric = float(value)
            _require(math.isfinite(numeric), f"{label}[{index}].{axis} must be finite")
            coords.append(numeric)
        result.append(RouteWaypoint(*coords))
    return tuple(result)


def load_canonical_route_b(
    *,
    root: Path | str | None = None,
    route_path: Path | str | None = None,
    expected_sha256: str | None = None,
    progress_path: Path | str | None = None,
    expected_progress_sha256: str | None = None,
    require_progress: bool = False,
) -> CanonicalRoute:
    """Load and hash-verify the canonical Route-B JSON.

    Raises when the file is absent or its digest does not match the pin. There is
    no "continue anyway" path: an unverified route definition cannot support a
    Route-B claim.
    """

    # Resolved at call time, not captured as a default, so repinning
    # CANONICAL_ROUTE_B_SHA256 in this module takes effect everywhere.
    if expected_sha256 is None:
        expected_sha256 = CANONICAL_ROUTE_B_SHA256
    if expected_progress_sha256 is None:
        expected_progress_sha256 = CANONICAL_ROUTE_B_PROGRESS_SHA256
    base = Path(root) if root is not None else repository_root()
    path = Path(route_path) if route_path is not None else base / CANONICAL_ROUTE_B_RELPATH
    if not path.is_file():
        raise CoopInputError(
            f"canonical Route-B definition is absent at {path}; "
            "route-b-passive mode cannot be bound without it"
        )
    digest = _sha256_file(path)
    if digest != str(expected_sha256):
        raise CoopInputError(
            f"canonical Route-B definition hash mismatch at {path}: "
            f"observed {digest}, pinned {expected_sha256}"
        )

    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CoopInputError(f"canonical Route-B definition is not JSON: {exc}") from exc
    _require(isinstance(document, Mapping), "canonical route must be a JSON object")

    _require(
        str(document.get("map")) == EXPECTED_ROUTE_MAP,
        f"canonical route map drift: {document.get('map')!r}",
    )
    _require(
        str(document.get("type")) == EXPECTED_ROUTE_TYPE,
        f"canonical route type drift: {document.get('type')!r}",
    )
    _require(
        int(document.get("schema_version", 0)) == EXPECTED_ROUTE_SCHEMA_VERSION,
        "canonical route schema version drift",
    )
    _require(
        str(document.get("name")) == EXPECTED_ROUTE_NAME,
        f"canonical route name drift: {document.get('name')!r}",
    )

    resolution = document.get("route_sampling_resolution_m")
    _require(
        isinstance(resolution, (int, float)) and float(resolution) > 0.0,
        "canonical route sampling resolution must be positive",
    )

    progress_file: Path | None = None
    progress_digest: str | None = None
    progress_samples = 0
    candidate = (
        Path(progress_path)
        if progress_path is not None
        else base / CANONICAL_ROUTE_B_PROGRESS_RELPATH
    )
    if candidate.is_file():
        progress_digest = _sha256_file(candidate)
        if expected_progress_sha256 is not None and progress_digest != str(
            expected_progress_sha256
        ):
            raise CoopInputError(
                f"Route-B progress contract hash mismatch at {candidate}: "
                f"observed {progress_digest}, pinned {expected_progress_sha256}"
            )
        with candidate.open(newline="") as handle:
            reader = csv.DictReader(handle)
            _require(
                reader.fieldnames is not None
                and {"ego_x", "ego_y", "ego_z"} <= set(reader.fieldnames),
                "Route-B progress contract lacks ego_x/ego_y/ego_z columns",
            )
            progress_samples = sum(1 for _row in reader)
        _require(progress_samples > 0, "Route-B progress contract is empty")
        progress_file = candidate
    elif require_progress:
        raise CoopInputError(
            f"Route-B progress contract is absent at {candidate} and was required"
        )

    return CanonicalRoute(
        name=str(document["name"]),
        map_name=str(document["map"]),
        route_type=str(document["type"]),
        loop=bool(document.get("loop")),
        coordinate_system=str(document.get("coordinate_system") or ""),
        schema_version=int(document["schema_version"]),
        path=_waypoints(document.get("planned_path"), "planned_path"),
        intermediate_waypoints=_waypoints(
            document.get("intermediate_waypoints"), "intermediate_waypoints"
        ),
        sampling_resolution_m=float(resolution),
        source_path=str(path),
        sha256=digest,
        progress_path=None if progress_file is None else str(progress_file),
        progress_sha256=progress_digest,
        progress_samples=progress_samples,
    )


@dataclass(frozen=True)
class RouteContract:
    """The bound route contract for one demonstration configuration."""

    mode: str
    presentation_label: str
    qualification_status: str
    is_route_b: bool
    registered_stream_ids: tuple[str, ...]
    registered_ue_ids: tuple[str, ...]
    stage2_spawn_indices: tuple[int, ...] = ()
    route: CanonicalRoute | None = None

    def __post_init__(self) -> None:
        _require(self.mode in ROUTE_MODES, f"unregistered route mode {self.mode!r}")
        if not self.is_route_b:
            _require(
                not mentions_route_b_as_a_claim(self.presentation_label),
                "a non-Route-B mode must not claim Route B in its "
                f"presentation label: {self.presentation_label!r}",
            )

    def overlay_polyline(self, *, stride: int = 4) -> tuple[tuple[float, float], ...]:
        """Route overlay points, empty unless a verified Route-B route is bound."""

        if self.route is None:
            return ()
        return self.route.overlay_polyline(stride=stride)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "presentation_label": self.presentation_label,
            "qualification_status": self.qualification_status,
            "is_route_b": self.is_route_b,
            "registered_stream_ids": list(self.registered_stream_ids),
            "registered_ue_ids": list(self.registered_ue_ids),
            "stage2_spawn_indices": list(self.stage2_spawn_indices),
            "route": None if self.route is None else self.route.as_dict(),
            "route_overlay_available": self.route is not None,
            "live_route_b_run_authorized": False,
        }


def _registered(stream_ids: Iterable[str], ue_ids: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    streams = tuple(nonempty_str(value, "stream_id") for value in stream_ids)
    ues = tuple(nonempty_str(value, "ue_id") for value in ue_ids)
    _require(len(set(streams)) == len(streams), "registered stream identities must be unique")
    _require(len(set(ues)) == len(ues), "registered UE identities must be unique")
    return streams, ues


def build_route_contract(
    *,
    mode: str,
    stream_ids: Sequence[str],
    ue_ids: Sequence[str],
    stage2_spawn_indices: Sequence[int] = (),
    root: Path | str | None = None,
    route_path: Path | str | None = None,
    expected_sha256: str | None = None,
    progress_path: Path | str | None = None,
    expected_progress_sha256: str | None = None,
    require_progress: bool = False,
) -> RouteContract:
    """Bind and validate a route contract for the requested mode."""

    _require(mode in ROUTE_MODES, f"unregistered route mode {mode!r}")
    streams, ues = _registered(stream_ids, ue_ids)

    if mode == ROUTE_MODE_STAGE2:
        indices = tuple(int(value) for value in stage2_spawn_indices)
        return RouteContract(
            mode=ROUTE_MODE_STAGE2,
            presentation_label=STAGE2_LABEL,
            qualification_status=QUALIFICATION_STATUS_STAGE2,
            is_route_b=False,
            registered_stream_ids=streams,
            registered_ue_ids=ues,
            stage2_spawn_indices=indices,
            route=None,
        )

    # route-b-passive
    if len(streams) < MINIMUM_REGISTERED_STREAMS or len(ues) < MINIMUM_REGISTERED_STREAMS:
        raise CoopInputError(
            "route-b-passive requires at least "
            f"{MINIMUM_REGISTERED_STREAMS} explicitly registered ego/stream "
            f"identities; got {len(streams)} stream(s) and {len(ues)} UE(s)"
        )
    route = load_canonical_route_b(
        root=root,
        route_path=route_path,
        expected_sha256=expected_sha256,
        progress_path=progress_path,
        expected_progress_sha256=expected_progress_sha256,
        require_progress=require_progress,
    )
    return RouteContract(
        mode=ROUTE_MODE_ROUTE_B_PASSIVE,
        presentation_label=ROUTE_B_LABEL,
        qualification_status=QUALIFICATION_STATUS_ROUTE_B,
        is_route_b=True,
        registered_stream_ids=streams,
        registered_ue_ids=ues,
        stage2_spawn_indices=(),
        route=route,
    )
