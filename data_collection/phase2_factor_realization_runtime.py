"""Fail-closed runtime adapter for the Phase-2 v2 factor contract.

The authored onset and requested speeds in the v2 manifest are scenario-control
metadata.  They are deliberately never policy inputs.  This module admits a
positive trajectory only from the first *realized* onset sample, before any
recipient yield/intervention, and computes the registered instantaneous
relative-motion diagnostics from that sample.

This module starts no process and owns no CARLA actors.  Keeping the arithmetic
pure makes the scientific gate testable without launching CARLA or OAI.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from phase2_map_sharing.adjudicate_future_hazards import oriented_box_clearance_m


REQUESTED_FACTOR_STRING_FIELDS = (
    "factor_realization_status",
    "time_to_hazard_label_status",
    "hazard_actor_role",
    "onset_driver_role",
    "geometry_measurement_basis",
    "closing_speed_measurement_basis",
    "proximity_horizon_measurement_basis",
)
REQUESTED_FACTOR_FLOAT_FIELDS = (
    "requested_helper_speed_mps",
    "requested_recipient_speed_mps",
    "requested_hazard_actor_speed_mps",
    "requested_onset_driver_speed_mps",
    "requested_hazard_onset_s",
    "requested_closing_speed_target_mps",
    "requested_closing_speed_band_min_mps",
    "requested_closing_speed_band_max_mps",
    "requested_proximity_horizon_target_s",
    "requested_proximity_horizon_band_min_s",
    "requested_proximity_horizon_band_max_s",
    "minimum_onset_driver_speed_mps",
)

NON_TREATMENT_MANIFEST_FIELDS = (
    "schema",
    "design_id",
    "suite_id",
    "split",
    "group_id",
    "matched_pair_id",
    "geometry_or_route_id",
    "geometry_or_route_status",
    "traffic_density",
    "traffic_density_status",
    "ambient_population_mode",
    "ambient_population_process_required",
    "weather",
    "renderer_quality_level",
    "renderer_server_launch_flag",
    "carla_seed",
    "traffic_seed",
    "sensor_seed",
    "raw_retention_tier",
    "raw_window_duration_s",
    "raw_window_anchor",
    "pair_contract_id",
    "route_start_anchor_id",
    "recipient_start_index",
    "helper_start_index",
    "recipient_route_sha256",
    "helper_route_sha256",
    "requested_helper_speed_mps",
    "requested_recipient_speed_mps",
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_scalar(value: object) -> object:
    """Normalize pandas/numpy scalars without importing either dependency."""

    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@dataclass(frozen=True)
class FactorRuntimeContract:
    trajectory_id: str
    trajectory_row_sha256: str
    group_id: str
    geometry_or_route_id: str
    hazard_class: str
    scenario_role: str
    controlled_hazard_present: bool
    requested: Mapping[str, object]
    maximum_surface_clearance_m: float

    @classmethod
    def from_plan_row(
        cls,
        row: Mapping[str, object],
        *,
        maximum_surface_clearance_m: float,
    ) -> "FactorRuntimeContract":
        requested = row.get("requested_factor_contract")
        if not isinstance(requested, Mapping):
            raise ValueError("factor-smoke plan row lacks requested_factor_contract")
        expected_keys = {
            "closing_speed_band",
            "time_to_hazard_band",
            *REQUESTED_FACTOR_STRING_FIELDS,
            *REQUESTED_FACTOR_FLOAT_FIELDS,
        }
        if set(requested) != expected_keys:
            raise ValueError(
                "requested factor keys differ: "
                f"missing={sorted(expected_keys - set(requested))} "
                f"extra={sorted(set(requested) - expected_keys)}"
            )
        normalized: dict[str, object] = {
            "closing_speed_band": str(requested["closing_speed_band"]),
            "time_to_hazard_band": str(requested["time_to_hazard_band"]),
        }
        for field in REQUESTED_FACTOR_STRING_FIELDS:
            value = str(requested[field]).strip()
            if not value:
                raise ValueError(f"requested factor {field} is empty")
            normalized[field] = value
        for field in REQUESTED_FACTOR_FLOAT_FIELDS:
            value = _finite(requested[field], field)
            if value < 0.0:
                raise ValueError(f"requested factor {field} must be nonnegative")
            normalized[field] = value
        closing_bounds = (
            float(normalized["requested_closing_speed_band_min_mps"]),
            float(normalized["requested_closing_speed_target_mps"]),
            float(normalized["requested_closing_speed_band_max_mps"]),
        )
        horizon_bounds = (
            float(normalized["requested_proximity_horizon_band_min_s"]),
            float(normalized["requested_proximity_horizon_target_s"]),
            float(normalized["requested_proximity_horizon_band_max_s"]),
        )
        if not closing_bounds[0] <= closing_bounds[1] <= closing_bounds[2]:
            raise ValueError("requested closing-speed target is outside its band")
        if not horizon_bounds[0] <= horizon_bounds[1] <= horizon_bounds[2]:
            raise ValueError("requested proximity-horizon target is outside its band")
        if (
            float(normalized["requested_onset_driver_speed_mps"])
            < float(normalized["minimum_onset_driver_speed_mps"])
        ):
            raise ValueError("requested onset-driver speed is below its measurement floor")
        maximum_clearance = _finite(
            maximum_surface_clearance_m, "maximum_surface_clearance_m"
        )
        if maximum_clearance <= 0.0:
            raise ValueError("maximum surface clearance must be positive")
        marker = row.get("controlled_hazard_present")
        if not isinstance(marker, bool):
            raise ValueError(
                "factor plan controlled_hazard_present must be a JSON boolean"
            )
        positive = marker
        role = str(row.get("scenario_role", ""))
        if positive != (role == "controlled_positive_occlusion"):
            raise ValueError("factor plan treatment marker and scenario role disagree")
        return cls(
            trajectory_id=str(row["trajectory_id"]),
            trajectory_row_sha256=str(row["trajectory_row_sha256"]),
            group_id=str(row["group_id"]),
            geometry_or_route_id=str(row["geometry_or_route_id"]),
            hazard_class=str(row["hazard_class"]),
            scenario_role=role,
            controlled_hazard_present=positive,
            requested=normalized,
            maximum_surface_clearance_m=maximum_clearance,
        )


def nontreatment_plan_record(
    manifest_row: Mapping[str, object],
    *,
    scenario_owned_signature: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return the exact pair-shared plan, excluding the hazard treatment.

    The allow-list intentionally excludes trajectory/scenario-role identifiers,
    hazard presence, hazard/onset controls, and all realized/evaluation fields.
    Live membership proves that the required helper/recipient/occluder roles
    exist, but floating realized poses are deliberately excluded from this
    *plan* hash.  Their bounded equality is enforced by the separate matched-
    pair realization gate.  Hashing settled floating poses here would turn
    harmless CARLA physics jitter into an impossible byte-equality contract.
    """

    missing = [field for field in NON_TREATMENT_MANIFEST_FIELDS if field not in manifest_row]
    if missing:
        raise ValueError(f"manifest lacks non-treatment fields: {missing}")
    values = {
        field: _json_scalar(manifest_row[field])
        for field in NON_TREATMENT_MANIFEST_FIELDS
    }
    membership = [
        {
            "type_id": str(item.get("type_id", "")),
            "role_name": str(item.get("role_name", "")),
            "motion_mode": str(item.get("motion_mode", "")),
        }
        for item in scenario_owned_signature
    ]
    if not membership or any(
        not item["type_id"] or not item["role_name"] for item in membership
    ):
        raise ValueError("scenario-owned non-treatment membership is incomplete")
    if len({item["role_name"] for item in membership}) != len(membership):
        raise ValueError("scenario-owned non-treatment role names are not unique")
    return {
        "schema": "scenesense.phase2_nontreatment_plan.v1",
        "manifest_fields": values,
        "scenario_owned_actor_membership": sorted(
            membership, key=lambda item: (item["role_name"], item["type_id"])
        ),
        "realized_pose_comparison": (
            "separate_matched_pair_owned_nontreatment_realization_gate"
        ),
        "excluded_treatment_fields": [
            "trajectory_id",
            "scenario_role",
            "controlled_hazard_present",
            "hazard_actor_role",
            "onset_driver_role",
            "requested_hazard_actor_speed_mps",
            "requested_onset_driver_speed_mps",
            "requested_hazard_onset_s",
        ],
    }


def _xy(actor: object) -> tuple[float, float]:
    location = actor.get_location()
    return _finite(location.x, "actor.location.x"), _finite(
        location.y, "actor.location.y"
    )


def _velocity_xy(actor: object) -> tuple[float, float]:
    velocity = actor.get_velocity()
    return _finite(velocity.x, "actor.velocity.x"), _finite(
        velocity.y, "actor.velocity.y"
    )


def speed_mps(actor: object) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(
        _finite(velocity.x, "actor.velocity.x") ** 2
        + _finite(velocity.y, "actor.velocity.y") ** 2
        + _finite(velocity.z, "actor.velocity.z") ** 2
    )


def instantaneous_relative_motion(
    recipient: object, hazard: object
) -> dict[str, float]:
    """Return radial closing speed and center closest-approach horizon.

    The horizon is a constant-velocity center-proximity diagnostic.  It is not
    collision TTC, braking TTC, or a safety guarantee.
    """

    rx, ry = _xy(recipient)
    hx, hy = _xy(hazard)
    rvx, rvy = _velocity_xy(recipient)
    hvx, hvy = _velocity_xy(hazard)
    dx, dy = hx - rx, hy - ry
    dvx, dvy = hvx - rvx, hvy - rvy
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        raise ValueError("recipient and hazard centers overlap at onset")
    dot = dx * dvx + dy * dvy
    relative_speed_sq = dvx * dvx + dvy * dvy
    if relative_speed_sq <= 1e-12:
        raise ValueError("relative motion is zero at realized onset")
    return {
        "center_distance_m": distance,
        "radial_closing_speed_mps": max(0.0, -dot / distance),
        "center_proximity_horizon_s": max(0.0, -dot / relative_speed_sq),
        "recipient_x_m": rx,
        "recipient_y_m": ry,
        "recipient_vx_mps": rvx,
        "recipient_vy_mps": rvy,
        "hazard_x_m": hx,
        "hazard_y_m": hy,
        "hazard_vx_mps": hvx,
        "hazard_vy_mps": hvy,
    }


def _actor_box_at(actor: object, horizon_s: float) -> tuple[float, float, float, float, float]:
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    bounding_box = getattr(actor, "bounding_box", None)
    if bounding_box is None:
        raise ValueError("actor lacks a bounding box for surface-clearance measurement")
    extent = bounding_box.extent
    local_center = bounding_box.location
    actor_yaw = _finite(transform.rotation.yaw, "actor.rotation.yaw")
    yaw_rad = math.radians(actor_yaw)
    center_x = (
        _finite(transform.location.x, "actor.transform.x")
        + math.cos(yaw_rad) * _finite(local_center.x, "bbox.location.x")
        - math.sin(yaw_rad) * _finite(local_center.y, "bbox.location.y")
        + _finite(velocity.x, "actor.velocity.x") * float(horizon_s)
    )
    center_y = (
        _finite(transform.location.y, "actor.transform.y")
        + math.sin(yaw_rad) * _finite(local_center.x, "bbox.location.x")
        + math.cos(yaw_rad) * _finite(local_center.y, "bbox.location.y")
        + _finite(velocity.y, "actor.velocity.y") * float(horizon_s)
    )
    bbox_rotation = getattr(bounding_box, "rotation", None)
    bbox_yaw = 0.0 if bbox_rotation is None else _finite(
        bbox_rotation.yaw, "bbox.rotation.yaw"
    )
    length = 2.0 * _finite(extent.x, "bbox.extent.x")
    width = 2.0 * _finite(extent.y, "bbox.extent.y")
    if length <= 0.0 or width <= 0.0:
        raise ValueError("actor bounding-box extents must be positive")
    return center_x, center_y, actor_yaw + bbox_yaw, length, width


def predicted_surface_clearance_m(
    recipient: object, hazard: object, proximity_horizon_s: float
) -> float:
    """OBB clearance at the center closest-approach horizon.

    Positions use the onset sample's constant velocity and fixed orientation.
    This is the surface counterpart to the registered center-proximity horizon,
    not a claim about the later actuated trajectory.
    """

    horizon = _finite(proximity_horizon_s, "proximity_horizon_s")
    if horizon < 0.0:
        raise ValueError("proximity horizon cannot be negative")
    return float(
        oriented_box_clearance_m(
            _actor_box_at(recipient, horizon),
            _actor_box_at(hazard, horizon),
        )
    )


class FactorRealizationMonitor:
    """Capture exactly one first-realized-onset sample and gate it atomically."""

    def __init__(self, contract: FactorRuntimeContract, *, cadence_s: float) -> None:
        cadence = _finite(cadence_s, "cadence_s")
        if cadence <= 0.0:
            raise ValueError("factor monitor cadence must be positive")
        self.contract = contract
        self.cadence_s = cadence
        self.realized: Optional[dict[str, object]] = None
        self.failure: Optional[str] = None

    def observe(
        self,
        *,
        frame_id: int,
        elapsed_s: float,
        helper: object,
        recipient: object,
        hazard: Optional[object],
        onset_driver: Optional[object],
        recipient_intervened: bool,
    ) -> None:
        if (
            not self.contract.controlled_hazard_present
            or self.realized is not None
            or self.failure is not None
        ):
            return
        if hazard is None or onset_driver is None:
            self.failure = "positive row lacks its typed hazard/onset-driver actor"
            return
        try:
            onset_speed = speed_mps(onset_driver)
            minimum_speed = float(
                self.contract.requested["minimum_onset_driver_speed_mps"]
            )
            if onset_speed + 1e-12 < minimum_speed:
                return
            elapsed = _finite(elapsed_s, "elapsed_s")
            requested_onset = float(
                self.contract.requested["requested_hazard_onset_s"]
            )
            if elapsed + self.cadence_s + 1e-12 < requested_onset:
                self.failure = "onset driver moved before the authored onset window"
                return
            if recipient_intervened:
                self.failure = (
                    "first realized onset sample occurred after recipient intervention"
                )
                return
            motion = instantaneous_relative_motion(recipient, hazard)
            horizon = motion["center_proximity_horizon_s"]
            clearance = predicted_surface_clearance_m(recipient, hazard, horizon)
            self.realized = {
                "realized_hazard_onset_s": elapsed,
                "realized_helper_speed_mps": speed_mps(helper),
                "realized_recipient_speed_mps": speed_mps(recipient),
                "realized_hazard_actor_speed_mps": speed_mps(hazard),
                "realized_onset_driver_speed_mps": onset_speed,
                "pre_intervention_radial_closing_speed_mps": motion[
                    "radial_closing_speed_mps"
                ],
                "pre_intervention_hazard_proximity_horizon_s": horizon,
                "pre_intervention_minimum_surface_clearance_m": clearance,
                "geometry_measurement_basis": self.contract.requested[
                    "geometry_measurement_basis"
                ],
                "closing_speed_measurement_basis": self.contract.requested[
                    "closing_speed_measurement_basis"
                ],
                "proximity_horizon_measurement_basis": self.contract.requested[
                    "proximity_horizon_measurement_basis"
                ],
                "realized_onset_frame_id_evaluation_only": int(frame_id),
                "authored_onset_error_s_evaluation_only": elapsed - requested_onset,
                "surface_clearance_prediction_basis": (
                    "constant_velocity_fixed_orientation_obb_at_center_closest_approach"
                ),
                "onset_sample_kinematics_evaluation_only": motion,
            }
        except Exception as exc:
            # A degenerate actor sample is a scientific realization failure,
            # not an excuse to bypass the create-only forensic artifact.  The
            # caller persists diagnostic() and only then invokes finalize().
            self.failure = (
                "factor measurement failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def diagnostic(self) -> dict[str, object]:
        """Return a serializable gate record without hiding a failed corner.

        Geometry preflight must retain the realized sample even when a
        provisional authored control misses its assigned band.  The live
        exact-16 path calls :meth:`finalize`, which converts the same gate into
        a hard failure.  Keeping both paths on this one implementation avoids
        a permissive review-only approximation.
        """

        if not self.contract.controlled_hazard_present:
            return {
                "registered_target_absent": True,
                "realized_factors_status": (
                    "not_applicable_matched_benign_registered_target_absent"
                ),
                "factor_reference_trajectory_id": (
                    self.contract.trajectory_id.removesuffix("_ben") + "_pos"
                ),
            }
        if self.failure is not None or self.realized is None:
            reason = self.failure or "no realized onset sample"
            return {
                "realized_factors": (
                    {} if self.realized is None else dict(self.realized)
                ),
                "factor_realization_gate": {
                    "schema": "scenesense.phase2_factor_realization_gate.v1",
                    "pass": False,
                    "failures": [str(reason)],
                    "measurement_status": (
                        "no_admissible_first_realized_onset_sample"
                        if self.realized is None
                        else "realized_sample_rejected"
                    ),
                },
            }
        realized = self.realized
        requested = self.contract.requested
        closing = float(realized["pre_intervention_radial_closing_speed_mps"])
        horizon = float(realized["pre_intervention_hazard_proximity_horizon_s"])
        clearance = float(realized["pre_intervention_minimum_surface_clearance_m"])
        failures = []
        if not (
            float(requested["requested_closing_speed_band_min_mps"])
            <= closing
            <= float(requested["requested_closing_speed_band_max_mps"])
        ):
            failures.append("radial_closing_speed_out_of_band")
        if not (
            float(requested["requested_proximity_horizon_band_min_s"])
            <= horizon
            <= float(requested["requested_proximity_horizon_band_max_s"])
        ):
            failures.append("proximity_horizon_out_of_band")
        if not 0.0 <= clearance <= self.contract.maximum_surface_clearance_m:
            failures.append("predicted_surface_clearance_out_of_gate")
        return {
            "realized_factors": dict(realized),
            "factor_realization_gate": {
                "schema": "scenesense.phase2_factor_realization_gate.v1",
                "pass": not failures,
                "failures": failures,
                "measurement_status": "realized_pre_intervention_onset_sample",
                "closing_speed_band_mps": [
                    float(requested["requested_closing_speed_band_min_mps"]),
                    float(requested["requested_closing_speed_band_max_mps"]),
                ],
                "proximity_horizon_band_s": [
                    float(requested["requested_proximity_horizon_band_min_s"]),
                    float(requested["requested_proximity_horizon_band_max_s"]),
                ],
                "maximum_surface_clearance_m": self.contract.maximum_surface_clearance_m,
                "recipient_intervention_before_onset": False,
            },
        }

    def finalize(self) -> dict[str, object]:
        result = self.diagnostic()
        gate = result.get("factor_realization_gate")
        if isinstance(gate, Mapping) and gate.get("pass") is not True:
            realized = result.get("realized_factors", {})
            raise RuntimeError(
                "factor realization failed: "
                f"{gate.get('failures')}; realized={realized}"
            )
        return result
