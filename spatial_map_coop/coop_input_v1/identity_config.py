"""Explicit, configured identity binding for cooperative-map sources.

A UE identity is a configuration fact, never an inference. This module refuses
every shortcut that would let a stream become a vehicle by accident:

* no substring/prefix matching on ``stream_id`` (``two-ue/ue-a`` does not become
  ``ue-a``);
* no ordinal assignment by arrival order or by sorted order;
* no implicit fallback session;
* no wildcard that silently absorbs an unregistered stream.

``(run_id, cell_id, stream_id)`` is the key. ``run_id``/``cell_id`` may be bound
as ``None`` for the HTTP demo transport, which has no run/cell concept, but that
must be *declared* in the configuration rather than assumed at lookup time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .normalized import CoopInputError, _require, nonempty_str


IDENTITY_CONFIG_SCHEMA = "scenesense.coop_identity_binding.v1"

BINDING_SOURCE = "EXPLICIT_CONFIGURED_IDENTITY_BINDING_V1"


@dataclass(frozen=True)
class SourceIdentity:
    ue_id: str
    session_id: str

    def __post_init__(self) -> None:
        nonempty_str(self.ue_id, "ue_id")
        nonempty_str(self.session_id, "session_id")


@dataclass(frozen=True)
class IdentityBinding:
    """One configured (run, cell, stream) -> (session, UE) row."""

    run_id: str | None
    cell_id: str | None
    stream_id: str
    ue_id: str
    session_id: str

    def __post_init__(self) -> None:
        nonempty_str(self.stream_id, "stream_id")
        nonempty_str(self.ue_id, "ue_id")
        nonempty_str(self.session_id, "session_id")
        if self.run_id is not None:
            nonempty_str(self.run_id, "run_id")
        if self.cell_id is not None:
            nonempty_str(self.cell_id, "cell_id")

    @property
    def key(self) -> tuple[str | None, str | None, str]:
        return self.run_id, self.cell_id, self.stream_id


class CoopIdentityConfig:
    """Mandatory lookup table from transport identity to UE/session identity."""

    def __init__(self, bindings: Iterable[IdentityBinding]) -> None:
        table: dict[tuple[str | None, str | None, str], SourceIdentity] = {}
        ue_owner: dict[str, tuple[str | None, str | None, str]] = {}
        for binding in bindings:
            _require(
                binding.key not in table,
                f"duplicate identity binding for {binding.key!r}",
            )
            # One physical UE must not be bound to two transport identities in
            # the same configuration: that would double-count one vehicle as two
            # cooperating sources.
            previous = ue_owner.get(binding.ue_id)
            _require(
                previous is None,
                f"UE {binding.ue_id!r} is already bound to {previous!r}",
            )
            ue_owner[binding.ue_id] = binding.key
            table[binding.key] = SourceIdentity(binding.ue_id, binding.session_id)
        _require(bool(table), "at least one identity binding is required")
        self._table = table
        self._bindings = tuple(bindings)

    @classmethod
    def from_documents(cls, value: Any) -> "CoopIdentityConfig":
        """Build from a plain JSON document (list of rows or a schema envelope)."""

        rows: Any = value
        if isinstance(value, Mapping):
            _require(
                value.get("schema") == IDENTITY_CONFIG_SCHEMA,
                f"identity configuration schema drift: {value.get('schema')!r}",
            )
            rows = value.get("bindings")
        _require(isinstance(rows, (list, tuple)), "identity bindings must be a list")
        bindings = []
        for index, row in enumerate(rows):
            _require(isinstance(row, Mapping), f"bindings[{index}] must be an object")
            unknown = set(row) - {"run_id", "cell_id", "stream_id", "ue_id", "session_id"}
            _require(not unknown, f"bindings[{index}] has unknown fields {sorted(unknown)}")
            bindings.append(
                IdentityBinding(
                    run_id=None if row.get("run_id") is None else str(row["run_id"]),
                    cell_id=None if row.get("cell_id") is None else str(row["cell_id"]),
                    stream_id=str(row.get("stream_id") or ""),
                    ue_id=str(row.get("ue_id") or ""),
                    session_id=str(row.get("session_id") or ""),
                )
            )
        return cls(bindings)

    @classmethod
    def from_cli_pairs(
        cls,
        pairs: Iterable[str],
        *,
        session_id: str,
        run_id: str | None = None,
        cli_cell_id: str | None = None,
    ) -> "CoopIdentityConfig":
        """Parse explicit ``STREAM_ID=UE_ID`` operator arguments.

        The session (and optional run/cell) must be supplied separately; they are
        never derived from the stream string.
        """

        nonempty_str(session_id, "session_id")
        bindings = []
        for pair in pairs:
            text = str(pair)
            _require("=" in text, f"identity binding {text!r} must be STREAM_ID=UE_ID")
            stream_id, _, ue_id = text.partition("=")
            bindings.append(
                IdentityBinding(
                    run_id=run_id,
                    cell_id=cli_cell_id,
                    stream_id=stream_id.strip(),
                    ue_id=ue_id.strip(),
                    session_id=session_id,
                )
            )
        return cls(bindings)

    def resolve(
        self, *, run_id: str | None, cell_id: str | None, stream_id: str
    ) -> SourceIdentity:
        """Return the configured identity or raise. There is no fallback path."""

        key = (run_id, cell_id, nonempty_str(stream_id, "stream_id"))
        identity = self._table.get(key)
        if identity is None:
            raise CoopInputError(
                "no configured cooperative-map identity for "
                f"(run_id={run_id!r}, cell_id={cell_id!r}, stream_id={stream_id!r}); "
                "UE identity must be configured explicitly and is never inferred "
                "from stream naming or arrival order"
            )
        return identity

    @property
    def stream_count(self) -> int:
        return len(self._table)

    @property
    def ue_ids(self) -> tuple[str, ...]:
        return tuple(sorted({identity.ue_id for identity in self._table.values()}))

    @property
    def bindings(self) -> tuple[IdentityBinding, ...]:
        return self._bindings

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": IDENTITY_CONFIG_SCHEMA,
            "bindings": [
                {
                    "run_id": binding.run_id,
                    "cell_id": binding.cell_id,
                    "stream_id": binding.stream_id,
                    "ue_id": binding.ue_id,
                    "session_id": binding.session_id,
                }
                for binding in self._bindings
            ],
        }
