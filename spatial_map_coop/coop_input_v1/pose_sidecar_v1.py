"""Pose sidecar contract for the production direct-map path (objective C).

The problem
-----------
``splitfusion_direct_object_map_update.v1`` carries object records and timing but
**no ego/sensor pose and no sensor FoV**. The cooperative viewer needs both for
its 40 m ego-follow view and its two measured FoV wedges. Two things are
therefore forbidden:

* inventing a pose (a drawn FoV wedge would then be a fiction), and
* querying CARLA ground-truth actors for it, which would put simulator truth
  behind the perception path and quietly turn a perception demonstration into a
  ground-truth one.

The registered future choice
----------------------------
Two options were considered; **option 2 is the recommendation** and option 1 is
the compatibility path implemented here.

1. ``scenesense.coop_pose_sidecar.v1`` — a separate, versioned pose document
   keyed by ``(run_id, cell_id, stream_id, frame_id, capture_timestamp_ns)``,
   emitted by the same edge frame-context owner that already holds
   ``FrameContextV1.ego_world`` and the static camera calibration. It requires no
   change to the frozen v1 object protocol, which is why it is implementable
   offline in this branch. Its weakness is that pose and objects are two
   messages: they can be lost independently, so the viewer must tolerate
   "objects present, pose absent".

2. ``splitfusion_direct_object_map_update.v2`` — carry the SFD1 frame-context
   ``ego_world`` pose and the camera horizontal FoV **atomically** inside the
   object update. This is strictly better (one message, one identity, no join,
   no partial-arrival state) and the data already exists upstream:
   ``rl_agent/splitfusion_live_dispatch_v1/frame_context.py`` defines
   ``FrameContextV1.ego_world: Pose6D`` and ``StaticCameraCalibration`` with
   intrinsics. It is not implemented here because **the frozen production v1
   protocol in the main worktree must not be modified from this branch**.

This module implements only option 1, as a pure offline contract. Nothing here
opens a socket or writes to the production tree.

Identity is strict: a sidecar that does not match the batch on run, cell, stream,
frame **and** capture timestamp is rejected. The object batch remains
independently valid; only the presentation loses its pose, and it must say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .normalized import (
    POSE_AVAILABLE,
    POSE_UNAVAILABLE_NO_SIDECAR,
    CoopInputError,
    NormalizedCoopBatch,
    NormalizedPose,
    _require,
    nonempty_str,
    strict_finite,
    strict_int,
)


POSE_SIDECAR_SCHEMA = "scenesense.coop_pose_sidecar.v1"
POSE_SIDECAR_VERSION = 1
POSE_SOURCE = "COOP_POSE_SIDECAR_V1"

SIDECAR_REQUIRED_FIELDS = (
    "schema",
    "sidecar_version",
    "run_id",
    "cell_id",
    "stream_id",
    "frame_id",
    "capture_timestamp_ns",
    "ego_world",
    "horizontal_fov_deg",
)

SIDECAR_ALLOWED_FIELDS = SIDECAR_REQUIRED_FIELDS + ("pose_provenance",)

# Pose provenance values the sidecar may declare. A sidecar that claims a CARLA
# ground-truth actor read is refused: the cooperative map must not consume
# simulator truth behind the perception path.
PROVENANCE_SFD1_FRAME_CONTEXT = "SFD1_FRAME_CONTEXT_V1_EGO_WORLD"
PROVENANCE_ALLOWED = (PROVENANCE_SFD1_FRAME_CONTEXT,)
PROVENANCE_FORBIDDEN = (
    "CARLA_ACTOR_GROUND_TRUTH",
    "CARLA_WORLD_SNAPSHOT",
    "SIMULATOR_TRUTH",
)

SidecarKey = tuple[str, str, str, int, int]


@dataclass(frozen=True)
class PoseSidecarRecord:
    run_id: str
    cell_id: str
    stream_id: str
    frame_id: int
    capture_timestamp_ns: int
    pose: NormalizedPose

    @property
    def key(self) -> SidecarKey:
        return (
            self.run_id,
            self.cell_id,
            self.stream_id,
            self.frame_id,
            self.capture_timestamp_ns,
        )


def parse_pose_sidecar(document: Mapping[str, Any]) -> PoseSidecarRecord:
    """Validate one sidecar document with strict schema and unknown-field rejection."""

    _require(isinstance(document, Mapping), "pose sidecar must be an object")
    _require(
        document.get("schema") == POSE_SIDECAR_SCHEMA,
        f"pose sidecar schema drift: {document.get('schema')!r}",
    )
    _require(
        strict_int(document.get("sidecar_version"), "sidecar_version")
        == POSE_SIDECAR_VERSION,
        "pose sidecar version drift",
    )
    missing = [name for name in SIDECAR_REQUIRED_FIELDS if name not in document]
    _require(not missing, f"pose sidecar lacks {missing}")
    unknown = sorted(set(document) - set(SIDECAR_ALLOWED_FIELDS))
    _require(not unknown, f"pose sidecar has unknown fields {unknown}")

    provenance = str(document.get("pose_provenance") or PROVENANCE_SFD1_FRAME_CONTEXT)
    _require(
        provenance not in PROVENANCE_FORBIDDEN,
        f"pose sidecar declares forbidden ground-truth provenance {provenance!r}",
    )
    _require(
        provenance in PROVENANCE_ALLOWED,
        f"pose sidecar declares unregistered provenance {provenance!r}",
    )

    run_id = nonempty_str(document["run_id"], "run_id")
    cell_id = nonempty_str(document["cell_id"], "cell_id")
    stream_id = nonempty_str(document["stream_id"], "stream_id")
    frame_id = strict_int(document["frame_id"], "frame_id", minimum=0)
    capture_ns = strict_int(document["capture_timestamp_ns"], "capture_timestamp_ns", minimum=1)

    ego = document["ego_world"]
    _require(isinstance(ego, Mapping), "ego_world must be an object")
    ego_unknown = sorted(set(ego) - {"x", "y", "z", "pitch", "yaw", "roll"})
    _require(not ego_unknown, f"ego_world has unknown fields {ego_unknown}")
    for axis in ("x", "y", "z", "pitch", "yaw", "roll"):
        _require(axis in ego, f"ego_world lacks {axis}")

    fov = strict_finite(document["horizontal_fov_deg"], "horizontal_fov_deg")
    _require(0.0 < fov <= 360.0, "horizontal_fov_deg must lie in (0, 360]")

    pose = NormalizedPose(
        x=strict_finite(ego["x"], "ego_world.x"),
        y=strict_finite(ego["y"], "ego_world.y"),
        z=strict_finite(ego["z"], "ego_world.z"),
        pitch=strict_finite(ego["pitch"], "ego_world.pitch"),
        yaw=strict_finite(ego["yaw"], "ego_world.yaw"),
        roll=strict_finite(ego["roll"], "ego_world.roll"),
        horizontal_fov_deg=fov,
        source=POSE_SOURCE,
        binding=(run_id, cell_id, stream_id, frame_id, capture_ns),
    )
    return PoseSidecarRecord(run_id, cell_id, stream_id, frame_id, capture_ns, pose)


def attach_pose(
    batch: NormalizedCoopBatch, document: Mapping[str, Any]
) -> NormalizedCoopBatch:
    """Bind one sidecar to one batch, refusing any identity mismatch.

    Returns a *new* batch; the input batch is never mutated.
    """

    record = parse_pose_sidecar(document)
    _require(
        batch.run_id is not None and batch.cell_id is not None,
        "pose sidecars require a batch with run and cell identity",
    )
    expected: SidecarKey = (
        str(batch.run_id),
        str(batch.cell_id),
        batch.stream_id,
        batch.frame_id,
        batch.capture.timestamp_ns,
    )
    if record.key != expected:
        mismatched = [
            name
            for name, left, right in zip(
                ("run_id", "cell_id", "stream_id", "frame_id", "capture_timestamp_ns"),
                record.key,
                expected,
            )
            if left != right
        ]
        raise CoopInputError(
            f"pose sidecar identity mismatch on {mismatched}: "
            f"sidecar {record.key!r} != batch {expected!r}"
        )
    return batch.with_pose(record.pose, POSE_AVAILABLE)


class PoseSidecarIndex:
    """Bounded lookup of validated sidecars by full identity key."""

    def __init__(self, documents: Iterable[Mapping[str, Any]] = ()) -> None:
        self._records: dict[SidecarKey, PoseSidecarRecord] = {}
        for document in documents:
            self.add(document)

    def add(self, document: Mapping[str, Any]) -> PoseSidecarRecord:
        record = parse_pose_sidecar(document)
        existing = self._records.get(record.key)
        _require(
            existing is None or existing.pose == record.pose,
            f"conflicting pose sidecars for identity {record.key!r}",
        )
        self._records[record.key] = record
        return record

    def resolve(self, batch: NormalizedCoopBatch) -> NormalizedCoopBatch:
        """Attach a matching pose, or mark the batch pose-unavailable.

        Absence is a normal, reportable state -- not an error and never a
        fabricated pose. The batch's scientific content is unchanged either way.
        """

        if batch.run_id is None or batch.cell_id is None:
            return batch.with_pose(None, POSE_UNAVAILABLE_NO_SIDECAR)
        key: SidecarKey = (
            str(batch.run_id),
            str(batch.cell_id),
            batch.stream_id,
            batch.frame_id,
            batch.capture.timestamp_ns,
        )
        record = self._records.get(key)
        if record is None:
            return batch.with_pose(None, POSE_UNAVAILABLE_NO_SIDECAR)
        return batch.with_pose(record.pose, POSE_AVAILABLE)

    def __len__(self) -> int:
        return len(self._records)


# ---- presentation-facing view ------------------------------------------------

# Exact text the viewer must show when no pose is available. It is a badge, not
# a silent fallback, and no FoV wedge may be drawn alongside it.
POSE_UNAVAILABLE_LABEL = "pose unavailable — ego-follow and FoV disabled"


def follow_view_contract(
    batch: NormalizedCoopBatch, *, radius_m: float = 40.0, padding_m: float = 0.0
) -> dict[str, Any]:
    """Normalized ego-follow/FoV presentation contract for one batch.

    When the pose is unavailable this returns ``available=False`` with no
    bounds and no FoV, so a caller cannot accidentally render a wedge.
    """

    if batch.pose is None:
        return {
            "available": False,
            "pose_state": batch.pose_state,
            "label": POSE_UNAVAILABLE_LABEL,
            "stream_id": batch.stream_id,
            "radius_m": float(radius_m),
            "ego_pose": None,
            "fov_deg": None,
            "bounds": None,
        }
    pose = batch.pose
    radius = max(1.0, float(radius_m))
    padding = max(0.0, float(padding_m))
    return {
        "available": True,
        "pose_state": batch.pose_state,
        "label": "measured ego pose",
        "stream_id": batch.stream_id,
        "radius_m": radius,
        "ego_pose": {"x": pose.x, "y": pose.y, "yaw_deg": pose.yaw},
        "fov_deg": pose.horizontal_fov_deg,
        "pose_source": pose.source,
        "bounds": {
            "x_min": pose.x - radius - padding,
            "x_max": pose.x + radius + padding,
            "y_min": pose.y - radius - padding,
            "y_max": pose.y + radius + padding,
        },
    }
