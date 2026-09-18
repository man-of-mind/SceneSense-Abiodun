"""Read-only adapter from the frozen 72-profile SplitFusion action catalog to the
conditional Hybrid-SAC action representation.

A Hybrid-SAC action in this phase is the pair::

    (joint mode m, continuous quality q)

where ``m = (feature family, quantizer)`` is one of 12 stable discrete modes and
``q in [0, 0.98]`` is the continuous spatial-drop fraction carried on the wire at
1e-4 resolution.

Scope
-----
This module is deliberately mechanical.  It performs only:

* catalog binding and integrity verification (exact SHA-256 + schema);
* enumeration of the 12 joint modes in the catalog-declared Cartesian order;
* the already-registered continuous-q wire conversion;
* the already-registered spatial keep/drop derivation;
* exact lookup of the 72 measured anchors.

It makes no reward, radar/camera-state, payload, accuracy, latency, training,
environment or deployment decision.  It never interpolates between the six
measured q anchors, never substitutes a nearest anchor, and never fabricates a
``profile_id``.  ``SPLIT`` is the only execution mode this catalog represents;
``LOCAL_*`` and ``SKIP`` are separate future top-level modes and are absent here.

Importing this module performs no filesystem access, mutation or other runtime
side effect.  The catalog is read only when :func:`load_contract` (or
:func:`default_contract`) is called explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

__all__ = [
    "ActionContractError",
    "CatalogIntegrityError",
    "InvalidQualityError",
    "UnknownJointModeError",
    "CATALOG_RELATIVE_PATH",
    "CATALOG_SHA256",
    "CATALOG_SCHEMA",
    "EXECUTION_MODE",
    "SPATIAL_CELLS",
    "Q_E4_SCALE",
    "Q_E4_MIN",
    "Q_E4_MAX",
    "Q_MIN",
    "Q_MAX",
    "EXPECTED_FAMILY_COUNT",
    "EXPECTED_QUANTIZER_COUNT",
    "EXPECTED_Q_ANCHOR_COUNT",
    "EXPECTED_PROFILE_COUNT",
    "EXPECTED_MODE_COUNT",
    "JointMode",
    "QualityWireValue",
    "AnchorAction",
    "ExecutableAction",
    "SplitActionContract",
    "default_catalog_path",
    "load_contract",
    "default_contract",
    "round_half_up_q_e4",
    "keep_drop_counts",
]

# --------------------------------------------------------------------------- #
# Frozen registered constants
# --------------------------------------------------------------------------- #

#: Catalog location, relative to the ``abiodun/`` project root.
CATALOG_RELATIVE_PATH = (
    "rl_agent/splitfusion_action_catalog_v1/splitfusion_72_action_catalog.json"
)

#: Required exact SHA-256 of the locked catalog bytes.
CATALOG_SHA256 = "07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3"

#: Required catalog schema identifier.
CATALOG_SCHEMA = "splitfusion_72_action_catalog_v1"

#: The single execution mode this catalog represents.
EXECUTION_MODE = "SPLIT"

#: Registered spatial grid cell count (cross-checked against the catalog).
SPATIAL_CELLS = 21504

#: Wire resolution of ``q``: ``q_e4`` counts units of 1e-4.
Q_E4_SCALE = 10000

#: Mechanical clipping bounds of the wire quality field.
Q_E4_MIN = 0
Q_E4_MAX = 9800

#: The same bounds expressed as a real-valued quality.
Q_MIN = Q_E4_MIN / Q_E4_SCALE
Q_MAX = Q_E4_MAX / Q_E4_SCALE

#: Declared inventory that must reconcile exactly to 4 x 3 x 6 = 72.
EXPECTED_FAMILY_COUNT = 4
EXPECTED_QUANTIZER_COUNT = 3
EXPECTED_Q_ANCHOR_COUNT = 6
EXPECTED_PROFILE_COUNT = 72
EXPECTED_MODE_COUNT = EXPECTED_FAMILY_COUNT * EXPECTED_QUANTIZER_COUNT

#: Row fields retained as authoritative action identity.  Perception, payload,
#: capability and evidence blocks are intentionally NOT copied into this
#: contract; the catalog remains their single source of truth.
_MODE_INVARIANT_FIELDS = (
    "family_id",
    "bit_width",
    "latent_width",
    "transported_channels",
    "decoder_identity",
    "routing_tag",
    "zstd_level",
)


# --------------------------------------------------------------------------- #
# Exceptions: fail closed, never normalize
# --------------------------------------------------------------------------- #


class ActionContractError(Exception):
    """Base class for every Hybrid-SAC action-contract failure."""


class CatalogIntegrityError(ActionContractError):
    """The catalog is missing, altered, or internally contradictory."""


class InvalidQualityError(ActionContractError):
    """A requested continuous quality is not a usable finite real number."""


class UnknownJointModeError(ActionContractError):
    """A joint mode was addressed by an identity the catalog does not declare."""


# --------------------------------------------------------------------------- #
# Mechanical conversions
# --------------------------------------------------------------------------- #


def round_half_up_q_e4(q: Any) -> int:
    """Convert a requested quality to the clipped wire integer ``q_e4``.

    Implements the registered contract::

        q_e4 = clip(round_half_up(10000 * q), 0, 9800)

    Rounding is applied *before* clipping, and uses decimal ``ROUND_HALF_UP``
    rather than Python's built-in banker's rounding: ``q = 0.12345`` yields
    ``1235``, whereas ``round(10000 * 0.12345)`` yields ``1234``.

    The float is interpreted through its shortest round-trip decimal
    representation (``Decimal(str(float(q)))``).  This makes decimal ties such
    as ``0.12345`` exactly representable and therefore reachable, which a raw
    binary expansion would not be.  The conversion round-trips exactly for
    every one of the 9,801 representable ``q_e4`` values.

    Raises:
        InvalidQualityError: if ``q`` is not a real number, or is NaN or
            infinite.  Booleans are rejected as a quality is not a flag.
    """
    if isinstance(q, bool) or not isinstance(q, numbers.Real):
        raise InvalidQualityError(
            f"quality must be a finite real number, got {type(q).__name__}: {q!r}"
        )
    value = float(q)
    if math.isnan(value):
        raise InvalidQualityError("quality is NaN; refusing to convert to q_e4")
    if math.isinf(value):
        raise InvalidQualityError(
            f"quality is infinite ({value!r}); refusing to convert to q_e4"
        )
    scaled = Decimal(str(value)) * Q_E4_SCALE
    raw = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
    return min(max(raw, Q_E4_MIN), Q_E4_MAX)


def keep_drop_counts(q_e4: int) -> Tuple[int, int]:
    """Return ``(keep_count, drop_count)`` for a wire quality integer.

    Implements the registered rule ``drop = floor(q * N + 0.5)``,
    ``keep = N - drop`` with ``N = 21504``, using exact integer arithmetic::

        drop = (q_e4 * N + 5000) // 10000

    For non-negative ``q_e4`` this is exactly equivalent to half-up rounding of
    ``q_e4 * N / 10000`` and involves no floating point.  (No representable
    ``q_e4`` in ``[0, 9800]`` actually lands on a ``.5`` tie, because
    ``gcd(21504, 10000) = 16`` does not divide 5000; the integer form is exact
    either way.)

    Raises:
        InvalidQualityError: if ``q_e4`` is not an integer within the
            mechanical bounds.
    """
    if isinstance(q_e4, bool) or not isinstance(q_e4, numbers.Integral):
        raise InvalidQualityError(
            f"q_e4 must be an integer, got {type(q_e4).__name__}: {q_e4!r}"
        )
    q_e4 = int(q_e4)
    if not Q_E4_MIN <= q_e4 <= Q_E4_MAX:
        raise InvalidQualityError(
            f"q_e4 {q_e4} is outside the mechanical range "
            f"[{Q_E4_MIN}, {Q_E4_MAX}]"
        )
    drop = (q_e4 * SPATIAL_CELLS + Q_E4_SCALE // 2) // Q_E4_SCALE
    return SPATIAL_CELLS - drop, drop


# --------------------------------------------------------------------------- #
# Immutable value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class JointMode:
    """One of the 12 stable discrete Hybrid-SAC modes: (family, quantizer).

    ``mode_id`` is the position in the catalog-declared Cartesian order
    ``product(action_order.family, action_order.quantizer)`` and is stable for
    as long as the catalog hash holds.  All fields are mode-invariant: every one
    of the mode's six anchors agrees on them.
    """

    mode_id: int
    family: str
    quantizer: str
    family_id: int
    #: Quantizer bit width, where supplied by the catalog (supplied for all 12).
    bit_width: Optional[int]
    #: Autoencoder latent width; ``None`` for the non-autoencoded family.
    latent_width: Optional[int]
    transported_channels: int
    decoder_identity: str
    routing_tag: int
    zstd_level: int
    wire_layout: str
    wire_codec_id: int
    wire_version: int

    @property
    def canonical(self) -> str:
        """Canonical mode string, e.g. ``'SPLIT/noAE/UINT8'``."""
        return f"{EXECUTION_MODE}/{self.family}/{self.quantizer}"

    @property
    def key(self) -> Tuple[str, str]:
        """The ``(family, quantizer)`` identity tuple."""
        return self.family, self.quantizer

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.canonical


@dataclass(frozen=True, slots=True)
class QualityWireValue:
    """The executable result of the continuous-q wire conversion.

    Both the requested and the executed quality are preserved: ``requested_q``
    is what the policy asked for, ``q_e4``/``q_exec`` is what the wire will
    actually carry.
    """

    requested_q: float
    q_e4: int
    q_exec: float
    keep_count: int
    drop_count: int
    #: True when ``round_half_up(10000*q)`` fell below ``Q_E4_MIN``.
    clipped_below: bool
    #: True when ``round_half_up(10000*q)`` rose above ``Q_E4_MAX``.
    clipped_above: bool

    @property
    def was_clipped(self) -> bool:
        """True when the requested quality was outside the mechanical range."""
        return self.clipped_below or self.clipped_above


@dataclass(frozen=True, slots=True)
class AnchorAction:
    """A measured catalog row: one of the 72 registered anchor actions."""

    mode: JointMode
    action_id: int
    profile_id: str
    q_e4: int
    q: float
    keep_count: int
    drop_count: int
    execution_mode: str

    @property
    def key(self) -> Tuple[str, str, int]:
        """The ``(family, quantizer, q_e4)`` exact-lookup tuple."""
        return self.mode.family, self.mode.quantizer, self.q_e4


@dataclass(frozen=True, slots=True)
class ExecutableAction:
    """A fully described executable Hybrid-SAC action.

    ``anchor`` is populated only when the executed ``q_e4`` is one of the six
    measured anchors of this mode.  For any other ``q_e4`` the action remains
    fully executable but carries no catalog identity: ``action_id`` and
    ``profile_id`` are ``None``.  No nearest anchor is substituted and no
    quality, payload or latency value is interpolated.
    """

    mode: JointMode
    quality: QualityWireValue
    anchor: Optional[AnchorAction]
    execution_mode: str = EXECUTION_MODE

    @property
    def is_registered_anchor(self) -> bool:
        """True when this action coincides exactly with a measured catalog row."""
        return self.anchor is not None

    @property
    def action_id(self) -> Optional[int]:
        """Catalog ``action_id``, or ``None`` for a non-anchor quality."""
        return None if self.anchor is None else self.anchor.action_id

    @property
    def profile_id(self) -> Optional[str]:
        """Catalog ``profile_id``, or ``None`` for a non-anchor quality."""
        return None if self.anchor is None else self.anchor.profile_id

    @property
    def q_e4(self) -> int:
        """Executed wire quality integer."""
        return self.quality.q_e4

    @property
    def keep_count(self) -> int:
        """Spatial cells retained."""
        return self.quality.keep_count

    @property
    def drop_count(self) -> int:
        """Spatial cells dropped."""
        return self.quality.drop_count


# --------------------------------------------------------------------------- #
# Catalog binding helpers
# --------------------------------------------------------------------------- #


def default_catalog_path() -> Path:
    """Resolve the locked catalog path relative to this module's project root."""
    return Path(__file__).resolve().parents[2] / CATALOG_RELATIVE_PATH


def _require(condition: bool, message: str) -> None:
    """Fail closed with a specific catalog-integrity error."""
    if not condition:
        raise CatalogIntegrityError(message)


def _declared_order(document: Mapping[str, Any], key: str, expected: int) -> Tuple:
    """Read one declared order list from ``action_order``, strictly."""
    action_order = document.get("action_order")
    _require(
        isinstance(action_order, Mapping),
        "catalog is missing a mapping at document['action_order']",
    )
    values = action_order.get(key)
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
        f"action_order['{key}'] must be a declared sequence",
    )
    order = tuple(values)
    _require(
        len(order) == expected,
        f"action_order['{key}'] declares {len(order)} entries, expected {expected}",
    )
    _require(
        len(set(order)) == len(order),
        f"action_order['{key}'] contains duplicate entries: {order!r}",
    )
    return order


# --------------------------------------------------------------------------- #
# The adapter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SplitActionContract:
    """Strict, read-only view of the frozen catalog as a Hybrid-SAC action space.

    Construct with :meth:`from_path` / :func:`load_contract`; the constructor is
    not intended to be called directly.
    """

    catalog_path: Path
    catalog_sha256: str
    schema: str
    family_order: Tuple[str, ...]
    quantizer_order: Tuple[str, ...]
    q_anchor_order: Tuple[int, ...]
    modes: Tuple[JointMode, ...]
    anchors: Tuple[AnchorAction, ...]
    _mode_by_key: Mapping[Tuple[str, str], JointMode]
    _anchor_by_key: Mapping[Tuple[str, str, int], AnchorAction]
    _anchors_by_mode: Mapping[int, Tuple[AnchorAction, ...]]

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_path(cls, path: Optional[Path] = None) -> "SplitActionContract":
        """Read, verify and bind the locked catalog without modifying it."""
        catalog_path = Path(path) if path is not None else default_catalog_path()
        try:
            raw = catalog_path.read_bytes()
        except OSError as exc:
            raise CatalogIntegrityError(
                f"cannot read the locked catalog at {catalog_path}: {exc}"
            ) from exc

        digest = hashlib.sha256(raw).hexdigest()
        _require(
            digest == CATALOG_SHA256,
            f"catalog SHA-256 mismatch at {catalog_path}: "
            f"expected {CATALOG_SHA256}, got {digest}",
        )
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogIntegrityError(
                f"locked catalog at {catalog_path} is not decodable JSON: {exc}"
            ) from exc
        _require(
            isinstance(document, Mapping),
            "catalog root must be a JSON object",
        )
        _require(
            document.get("schema") == CATALOG_SCHEMA,
            f"catalog schema mismatch: expected {CATALOG_SCHEMA!r}, "
            f"got {document.get('schema')!r}",
        )
        return cls._bind(catalog_path, digest, document)

    @classmethod
    def _bind(
        cls,
        catalog_path: Path,
        digest: str,
        document: Mapping[str, Any],
    ) -> "SplitActionContract":
        """Reconcile the declared inventory and build the immutable indexes."""
        family_order = _declared_order(document, "family", EXPECTED_FAMILY_COUNT)
        quantizer_order = _declared_order(
            document, "quantizer", EXPECTED_QUANTIZER_COUNT
        )
        q_anchor_order = _declared_order(document, "q_e4", EXPECTED_Q_ANCHOR_COUNT)
        for anchor in q_anchor_order:
            _require(
                isinstance(anchor, int)
                and not isinstance(anchor, bool)
                and Q_E4_MIN <= anchor <= Q_E4_MAX,
                f"declared q anchor {anchor!r} is not an integer within "
                f"[{Q_E4_MIN}, {Q_E4_MAX}]",
            )

        cls._verify_declared_inventory(document, q_anchor_order)
        cls._verify_transport_contract(document)

        rows = document.get("profiles")
        _require(
            isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
            "catalog is missing a sequence at document['profiles']",
        )
        _require(
            len(rows) == EXPECTED_PROFILE_COUNT,
            f"catalog declares {len(rows)} profile rows, "
            f"expected exactly {EXPECTED_PROFILE_COUNT}",
        )

        rows_by_key = cls._index_rows(rows, family_order, quantizer_order, q_anchor_order)
        modes, anchors_by_mode = cls._build_modes(
            rows_by_key, family_order, quantizer_order, q_anchor_order
        )

        anchors = tuple(a for mode in modes for a in anchors_by_mode[mode.mode_id])
        _require(
            len(anchors) == EXPECTED_PROFILE_COUNT,
            f"built {len(anchors)} anchors, expected {EXPECTED_PROFILE_COUNT}",
        )
        return cls(
            catalog_path=catalog_path,
            catalog_sha256=digest,
            schema=str(document["schema"]),
            family_order=family_order,
            quantizer_order=quantizer_order,
            q_anchor_order=q_anchor_order,
            modes=modes,
            anchors=anchors,
            _mode_by_key=MappingProxyType({m.key: m for m in modes}),
            _anchor_by_key=MappingProxyType({a.key: a for a in anchors}),
            _anchors_by_mode=MappingProxyType(dict(anchors_by_mode)),
        )

    # -- validation -------------------------------------------------------- #

    @staticmethod
    def _verify_declared_inventory(
        document: Mapping[str, Any],
        q_anchor_order: Tuple[int, ...],
    ) -> None:
        """Require the catalog's own summary to reconcile to 4 x 3 x 6 = 72."""
        summary = document.get("summary")
        _require(
            isinstance(summary, Mapping),
            "catalog is missing a mapping at document['summary']",
        )
        overall = summary.get("overall")
        _require(
            isinstance(overall, Mapping),
            "catalog is missing a mapping at document['summary']['overall']",
        )
        declared = {
            "families": EXPECTED_FAMILY_COUNT,
            "quantizers": EXPECTED_QUANTIZER_COUNT,
            "q_anchors": EXPECTED_Q_ANCHOR_COUNT,
            "profiles": EXPECTED_PROFILE_COUNT,
        }
        for key, expected in declared.items():
            _require(
                overall.get(key) == expected,
                f"summary.overall['{key}'] is {overall.get(key)!r}, "
                f"expected {expected}",
            )
        product_of_dimensions = (
            EXPECTED_FAMILY_COUNT * EXPECTED_QUANTIZER_COUNT * EXPECTED_Q_ANCHOR_COUNT
        )
        _require(
            product_of_dimensions == EXPECTED_PROFILE_COUNT,
            f"declared inventory does not reconcile: "
            f"{EXPECTED_FAMILY_COUNT} x {EXPECTED_QUANTIZER_COUNT} x "
            f"{EXPECTED_Q_ANCHOR_COUNT} = {product_of_dimensions}, "
            f"expected {EXPECTED_PROFILE_COUNT}",
        )

        continuous_q = document.get("continuous_q")
        _require(
            isinstance(continuous_q, Mapping),
            "catalog is missing a mapping at document['continuous_q']",
        )
        validated = continuous_q.get("validated_anchor_q_e4")
        _require(
            isinstance(validated, Sequence)
            and tuple(validated) == q_anchor_order,
            "continuous_q.validated_anchor_q_e4 contradicts "
            f"action_order.q_e4: {validated!r} vs {list(q_anchor_order)!r}",
        )

        boundary = document.get("mode_policy_boundary")
        _require(
            isinstance(boundary, Mapping)
            and boundary.get("catalog_execution_mode") == EXECUTION_MODE,
            "catalog does not declare "
            f"mode_policy_boundary.catalog_execution_mode == {EXECUTION_MODE!r}",
        )

    @staticmethod
    def _verify_transport_contract(document: Mapping[str, Any]) -> None:
        """Cross-check the registered spatial grid and wire resolution."""
        transport = document.get("fixed_transport_contract")
        _require(
            isinstance(transport, Mapping),
            "catalog is missing a mapping at document['fixed_transport_contract']",
        )
        _require(
            transport.get("spatial_cells") == SPATIAL_CELLS,
            f"catalog spatial_cells is {transport.get('spatial_cells')!r}, "
            f"expected the registered {SPATIAL_CELLS}",
        )
        _require(
            transport.get("q_wire_resolution") == "1e-4",
            "catalog q_wire_resolution is "
            f"{transport.get('q_wire_resolution')!r}, expected '1e-4'",
        )
        _require(
            transport.get("keep_drop_rule") == "drop=floor(q*N+0.5); keep=N-drop",
            "catalog keep_drop_rule is "
            f"{transport.get('keep_drop_rule')!r}, which this adapter does not "
            "implement",
        )

    @staticmethod
    def _index_rows(
        rows: Sequence[Any],
        family_order: Tuple[str, ...],
        quantizer_order: Tuple[str, ...],
        q_anchor_order: Tuple[int, ...],
    ) -> Mapping[Tuple[str, str, int], Mapping[str, Any]]:
        """Validate every row and index it by ``(family, quantizer, q_e4)``."""
        families = set(family_order)
        quantizers = set(quantizer_order)
        anchors = set(q_anchor_order)

        indexed: dict = {}
        seen_action_ids: dict = {}
        seen_profile_ids: dict = {}

        for position, row in enumerate(rows):
            _require(
                isinstance(row, Mapping),
                f"profiles[{position}] is not a JSON object",
            )
            where = f"profiles[{position}]"

            mode_str = row.get("execution_mode")
            _require(
                mode_str == EXECUTION_MODE,
                f"{where} declares execution_mode {mode_str!r}; this catalog "
                f"represents only {EXECUTION_MODE!r}",
            )

            family = row.get("family")
            quantizer = row.get("quantizer")
            q_e4 = row.get("q_e4")
            _require(
                family in families,
                f"{where} declares foreign family {family!r}; declared order is "
                f"{list(family_order)!r}",
            )
            _require(
                quantizer in quantizers,
                f"{where} declares foreign quantizer {quantizer!r}; declared "
                f"order is {list(quantizer_order)!r}",
            )
            _require(
                isinstance(q_e4, int) and not isinstance(q_e4, bool)
                and q_e4 in anchors,
                f"{where} declares foreign q_e4 {q_e4!r}; declared anchors are "
                f"{list(q_anchor_order)!r}",
            )

            q = row.get("q")
            _require(
                isinstance(q, numbers.Real) and not isinstance(q, bool),
                f"{where} q is not a real number: {q!r}",
            )
            _require(
                float(q) * Q_E4_SCALE == float(q_e4),
                f"{where} q {q!r} contradicts q_e4 {q_e4!r}",
            )

            action_id = row.get("action_id")
            profile_id = row.get("profile_id")
            _require(
                isinstance(action_id, int) and not isinstance(action_id, bool),
                f"{where} action_id is not an integer: {action_id!r}",
            )
            _require(
                isinstance(profile_id, str) and profile_id != "",
                f"{where} profile_id is not a non-empty string: {profile_id!r}",
            )
            _require(
                action_id not in seen_action_ids,
                f"{where} repeats action_id {action_id} first seen at "
                f"{seen_action_ids.get(action_id)}",
            )
            _require(
                profile_id not in seen_profile_ids,
                f"{where} repeats profile_id {profile_id!r} first seen at "
                f"{seen_profile_ids.get(profile_id)}",
            )
            seen_action_ids[action_id] = where
            seen_profile_ids[profile_id] = where

            keep_count = row.get("keep_count")
            drop_count = row.get("drop_count")
            expected_keep, expected_drop = keep_drop_counts(q_e4)
            _require(
                keep_count == expected_keep and drop_count == expected_drop,
                f"{where} keep/drop counts ({keep_count!r}, {drop_count!r}) do "
                f"not match the registered rule for q_e4={q_e4} "
                f"({expected_keep}, {expected_drop})",
            )

            wire = row.get("wire")
            _require(
                isinstance(wire, Mapping),
                f"{where} is missing a mapping at 'wire'",
            )
            for field in ("layout", "codec_id", "version"):
                _require(
                    field in wire,
                    f"{where} wire is missing required field {field!r}",
                )
            for field in _MODE_INVARIANT_FIELDS:
                _require(
                    field in row,
                    f"{where} is missing required identity field {field!r}",
                )

            key = (family, quantizer, q_e4)
            _require(
                key not in indexed,
                f"{where} duplicates the action identity {key!r}",
            )
            indexed[key] = row

        expected_keys = set(product(family_order, quantizer_order, q_anchor_order))
        missing = expected_keys - set(indexed)
        foreign = set(indexed) - expected_keys
        _require(
            not missing,
            f"catalog is missing {len(missing)} declared action identities, "
            f"e.g. {sorted(missing)[:3]!r}",
        )
        _require(
            not foreign,
            f"catalog contains {len(foreign)} undeclared action identities, "
            f"e.g. {sorted(foreign)[:3]!r}",
        )
        return indexed

    @staticmethod
    def _build_modes(
        rows_by_key: Mapping[Tuple[str, str, int], Mapping[str, Any]],
        family_order: Tuple[str, ...],
        quantizer_order: Tuple[str, ...],
        q_anchor_order: Tuple[int, ...],
    ) -> Tuple[Tuple[JointMode, ...], Mapping[int, Tuple[AnchorAction, ...]]]:
        """Build the 12 joint modes in declared Cartesian order with 6 anchors each."""
        modes = []
        anchors_by_mode: dict = {}

        for mode_id, (family, quantizer) in enumerate(
            product(family_order, quantizer_order)
        ):
            member_rows = [
                rows_by_key[(family, quantizer, anchor)] for anchor in q_anchor_order
            ]
            _require(
                len(member_rows) == EXPECTED_Q_ANCHOR_COUNT,
                f"joint mode ({family}, {quantizer}) has {len(member_rows)} "
                f"anchors, expected {EXPECTED_Q_ANCHOR_COUNT}",
            )

            reference = member_rows[0]
            for field in _MODE_INVARIANT_FIELDS:
                values = {row[field] for row in member_rows}
                _require(
                    len(values) == 1,
                    f"joint mode ({family}, {quantizer}) has contradictory "
                    f"{field} across its anchors: {sorted(map(repr, values))}",
                )
            wire_values = {
                (row["wire"]["layout"], row["wire"]["codec_id"], row["wire"]["version"])
                for row in member_rows
            }
            _require(
                len(wire_values) == 1,
                f"joint mode ({family}, {quantizer}) has contradictory wire "
                f"identities across its anchors: {sorted(map(repr, wire_values))}",
            )
            layout, codec_id, version = next(iter(wire_values))

            mode = JointMode(
                mode_id=mode_id,
                family=family,
                quantizer=quantizer,
                family_id=reference["family_id"],
                bit_width=reference["bit_width"],
                latent_width=reference["latent_width"],
                transported_channels=reference["transported_channels"],
                decoder_identity=reference["decoder_identity"],
                routing_tag=reference["routing_tag"],
                zstd_level=reference["zstd_level"],
                wire_layout=layout,
                wire_codec_id=codec_id,
                wire_version=version,
            )
            modes.append(mode)

            mode_anchors = []
            for row in member_rows:
                mode_anchors.append(
                    AnchorAction(
                        mode=mode,
                        action_id=row["action_id"],
                        profile_id=row["profile_id"],
                        q_e4=row["q_e4"],
                        q=float(row["q"]),
                        keep_count=row["keep_count"],
                        drop_count=row["drop_count"],
                        execution_mode=row["execution_mode"],
                    )
                )
            anchors_by_mode[mode_id] = tuple(mode_anchors)

        _require(
            len(modes) == EXPECTED_MODE_COUNT,
            f"built {len(modes)} joint modes, expected {EXPECTED_MODE_COUNT}",
        )
        return tuple(modes), anchors_by_mode

    # -- mode access ------------------------------------------------------- #

    @property
    def mode_count(self) -> int:
        """Number of stable joint modes (12)."""
        return len(self.modes)

    @property
    def anchor_count(self) -> int:
        """Number of registered anchor actions (72)."""
        return len(self.anchors)

    def mode(self, mode_id: int) -> JointMode:
        """Return the joint mode with the given stable ``mode_id``.

        Raises:
            UnknownJointModeError: if ``mode_id`` is out of range.
        """
        if isinstance(mode_id, bool) or not isinstance(mode_id, numbers.Integral):
            raise UnknownJointModeError(
                f"mode_id must be an integer, got {type(mode_id).__name__}: "
                f"{mode_id!r}"
            )
        index = int(mode_id)
        if not 0 <= index < len(self.modes):
            raise UnknownJointModeError(
                f"mode_id {index} is outside [0, {len(self.modes) - 1}]"
            )
        return self.modes[index]

    def mode_for(self, family: str, quantizer: str) -> JointMode:
        """Return the joint mode for a declared ``(family, quantizer)`` pair.

        Raises:
            UnknownJointModeError: if the pair is not declared by the catalog.
        """
        try:
            return self._mode_by_key[(family, quantizer)]
        except (KeyError, TypeError) as exc:
            raise UnknownJointModeError(
                f"({family!r}, {quantizer!r}) is not a declared joint mode; "
                f"declared modes are {[m.canonical for m in self.modes]}"
            ) from exc

    def anchors_for_mode(self, mode_id: int) -> Tuple[AnchorAction, ...]:
        """Return a mode's six registered anchors, in declared anchor order."""
        return self._anchors_by_mode[self.mode(mode_id).mode_id]

    # -- quality conversion ------------------------------------------------ #

    def quality_for(self, q: Any) -> QualityWireValue:
        """Convert a requested continuous quality into its executable wire value.

        Boundary behaviour is explicit: rounding happens first, then clipping to
        ``[0, 9800]``.  A request of ``0.98004`` executes as ``9800`` *without*
        being flagged as clipped (it rounds to the bound); ``0.99`` executes as
        ``9800`` *and* is flagged ``clipped_above``.
        """
        if isinstance(q, bool) or not isinstance(q, numbers.Real):
            raise InvalidQualityError(
                f"quality must be a finite real number, got "
                f"{type(q).__name__}: {q!r}"
            )
        requested = float(q)
        q_e4 = round_half_up_q_e4(requested)
        scaled = Decimal(str(requested)) * Q_E4_SCALE
        raw = int(scaled.to_integral_value(rounding=ROUND_HALF_UP))
        keep_count, drop_count = keep_drop_counts(q_e4)
        return QualityWireValue(
            requested_q=requested,
            q_e4=q_e4,
            q_exec=q_e4 / Q_E4_SCALE,
            keep_count=keep_count,
            drop_count=drop_count,
            clipped_below=raw < Q_E4_MIN,
            clipped_above=raw > Q_E4_MAX,
        )

    # -- exact anchor lookup ----------------------------------------------- #

    def find_anchor(
        self,
        family: str,
        quantizer: str,
        q_e4: int,
    ) -> Optional[AnchorAction]:
        """Exact lookup by ``(family, quantizer, q_e4)``.

        Returns the measured :class:`AnchorAction` for a registered anchor, or
        ``None`` for any other ``q_e4``.  No nearest anchor is ever selected and
        no quality, payload or latency value is ever interpolated.

        Raises:
            UnknownJointModeError: if ``(family, quantizer)`` is not declared.
                An undeclared mode is a contract violation, whereas an
                unmeasured ``q_e4`` is a legitimate continuous action.
        """
        self.mode_for(family, quantizer)
        if isinstance(q_e4, bool) or not isinstance(q_e4, numbers.Integral):
            return None
        return self._anchor_by_key.get((family, quantizer, int(q_e4)))

    def resolve(self, mode_id: int, q: Any) -> ExecutableAction:
        """Resolve ``(mode_id, q)`` into a fully described executable action.

        The returned action always carries the complete mode identity and the
        executable quality.  It carries a catalog ``action_id``/``profile_id``
        only when the executed ``q_e4`` is exactly one of the mode's six
        measured anchors.
        """
        mode = self.mode(mode_id)
        quality = self.quality_for(q)
        anchor = self._anchor_by_key.get((mode.family, mode.quantizer, quality.q_e4))
        return ExecutableAction(
            mode=mode,
            quality=quality,
            anchor=anchor,
            execution_mode=EXECUTION_MODE,
        )


# --------------------------------------------------------------------------- #
# Module-level entry points (no work happens at import time)
# --------------------------------------------------------------------------- #


def load_contract(path: Optional[Path] = None) -> SplitActionContract:
    """Read and verify the locked catalog, returning a fresh bound contract."""
    return SplitActionContract.from_path(path)


@lru_cache(maxsize=1)
def default_contract() -> SplitActionContract:
    """Return a process-cached contract bound to the default catalog path.

    The catalog is read on the first explicit call, never at import time.
    """
    return SplitActionContract.from_path(None)
