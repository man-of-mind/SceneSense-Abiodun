"""Append-only SQLite row store with fail-closed resume semantics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .contract import OfflineGridContractError, canonical_json_bytes
from .manifest import validate_run_manifest
from .schema import validate_row

DDL = """
CREATE TABLE metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    run_manifest_json BLOB NOT NULL,
    run_manifest_sha256 TEXT NOT NULL,
    run_binding_sha256 TEXT NOT NULL,
    selection_manifest_sha256 TEXT NOT NULL,
    reward_spec_sha256 TEXT NOT NULL
);
CREATE TABLE quality_rows (
    row_key_sha256 TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    frame_id INTEGER NOT NULL,
    grid_split TEXT NOT NULL,
    mode_id INTEGER NOT NULL,
    family TEXT NOT NULL,
    quantizer TEXT NOT NULL,
    q_e4 INTEGER NOT NULL,
    row_sha256 TEXT NOT NULL UNIQUE,
    row_json BLOB NOT NULL,
    UNIQUE (episode_id, sample_id, frame_id, grid_split, mode_id, family, quantizer, q_e4)
);
CREATE INDEX quality_rows_progress ON quality_rows (episode_id, sample_id, mode_id, q_e4);
"""


class ExactRowStore:
    """One transaction per row; a duplicate is always an error, never a skip."""

    def __init__(
        self, path: Path, manifest: Mapping[str, Any], *, resume: bool,
        expected_row_keys: frozenset[str],
    ) -> None:
        self.path = Path(path)
        if not isinstance(expected_row_keys, frozenset) or not expected_row_keys:
            raise OfflineGridContractError("store requires a non-empty frozen expected-key set")
        self.expected_row_keys = expected_row_keys
        validate_run_manifest(manifest)
        exists = self.path.exists()
        if resume and not exists:
            raise OfflineGridContractError(f"--resume store does not exist: {self.path}")
        if not resume and exists:
            raise OfflineGridContractError(f"create-only row store exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=60.0)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA journal_mode=WAL")
        if exists:
            self._verify_metadata(manifest)
            self.audit_rows()
        else:
            try:
                self.connection.executescript(DDL)
                payload = canonical_json_bytes(manifest)
                self.connection.execute(
                    "INSERT INTO metadata VALUES (1,?,?,?,?,?)",
                    (
                        payload,
                        manifest["run_manifest_sha256"],
                        manifest["run_binding_sha256"],
                        manifest["selection_manifest_sha256"],
                        manifest["reward_spec_sha256"],
                    ),
                )
                self.connection.commit()
            except Exception:
                self.connection.close()
                raise
        self.manifest = dict(manifest)

    def _verify_metadata(self, manifest: Mapping[str, Any]) -> None:
        row = self.connection.execute(
            "SELECT run_manifest_json, run_manifest_sha256, run_binding_sha256, "
            "selection_manifest_sha256, reward_spec_sha256 FROM metadata WHERE singleton=1"
        ).fetchone()
        if row is None or self.connection.execute("SELECT COUNT(*) FROM metadata").fetchone()[0] != 1:
            raise OfflineGridContractError("row store metadata is missing or duplicated")
        expected = (
            canonical_json_bytes(manifest),
            manifest["run_manifest_sha256"],
            manifest["run_binding_sha256"],
            manifest["selection_manifest_sha256"],
            manifest["reward_spec_sha256"],
        )
        observed = (bytes(row[0]), *row[1:])
        if observed != expected:
            raise OfflineGridContractError("resume refused: run/checkpoint/spec binding differs")

    def __enter__(self) -> "ExactRowStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.connection.close()

    @property
    def row_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM quality_rows").fetchone()[0])

    def completed_keys(self) -> frozenset[str]:
        return frozenset(
            str(row[0])
            for row in self.connection.execute("SELECT row_key_sha256 FROM quality_rows")
        )

    def insert(self, row: Mapping[str, Any]) -> None:
        validate_row(row)
        if row["row_key_sha256"] not in self.expected_row_keys:
            raise OfflineGridContractError("row key is outside the selected frozen grid")
        for name in ("run_binding_sha256", "selection_manifest_sha256", "reward_spec_sha256"):
            if row[name] != self.manifest[name]:
                raise OfflineGridContractError(f"row {name} differs from store metadata")
        values = (
            row["row_key_sha256"],
            row["episode_id"],
            row["sample_id"],
            row["frame_id"],
            row["grid_split"],
            row["mode_id"],
            row["family"],
            row["quantizer"],
            row["q_e4"],
            row["row_sha256"],
            canonical_json_bytes(row),
        )
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO quality_rows VALUES (?,?,?,?,?,?,?,?,?,?,?)", values
                )
        except sqlite3.IntegrityError as exc:
            raise OfflineGridContractError(
                f"duplicate/conflicting row refused: {row['row_key_sha256']}"
            ) from exc

    def audit_rows(self) -> dict[str, Any]:
        count = 0
        observed_keys: set[str] = set()
        for key, row_sha, payload in self.connection.execute(
            "SELECT row_key_sha256, row_sha256, row_json FROM quality_rows ORDER BY row_key_sha256"
        ):
            try:
                row = json.loads(bytes(payload).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OfflineGridContractError(f"corrupt stored row JSON: {key}") from exc
            validate_row(row)
            if row["row_key_sha256"] != key or row["row_sha256"] != row_sha:
                raise OfflineGridContractError(f"stored row index/content mismatch: {key}")
            if key not in self.expected_row_keys:
                raise OfflineGridContractError(f"stored row is outside the selected grid: {key}")
            observed_keys.add(str(key))
            count += 1
        if count > len(self.expected_row_keys):
            raise OfflineGridContractError("store has more rows than the frozen grid")
        return {
            "rows": count,
            "expected_rows": len(self.expected_row_keys),
            "complete": observed_keys == self.expected_row_keys,
            "sqlite_main_bytes": self.path.stat().st_size,
        }
