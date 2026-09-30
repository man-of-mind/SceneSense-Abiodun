"""Crash-consistent Run-5 boundary bundles and append-only ledgers.

Publication (one bundle = one boundary, weights and events together)::

    <parent>/.staging-<name>-<pid>-<token>/   written file by file, each fsynced
        event.json  training_state.pt  actor_state_dict.pt  channel_state.json
        manifest.json   (SHA-256 + size of every payload)
        COMMITTED       (SHA-256 of manifest.json)
    fsync(staging); rename(staging -> <parent>/<name>); fsync(parent)
    LATEST.tmp -> fsync -> replace(LATEST) -> fsync(parent)

A directory named ``checkpoint_NNNNNN`` / ``emergency_NNNNNN`` therefore only
ever appears complete.  Readers additionally verify the marker, the manifest
and every payload hash, ignore ``.staging-*`` directories and refuse anything
partial, corrupt or ambiguous.

LATEST policy on resume: the highest-update verified bundle is chosen.  LATEST
must name a verified bundle with the matching manifest digest; a pointer that
lags behind a newer *verified* bundle (crash between rename and pointer
update) is repaired and reported, while a missing, corrupt, foreign or
ahead-of-the-evidence pointer is refused.  Any final-named directory that
fails verification refuses the whole resume.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

BUNDLE_SCHEMA = "splitfusion.run5.boundary_bundle.v1"
MANIFEST = "manifest.json"
COMMITTED = "COMMITTED"
LATEST = "LATEST"
STAGING_PREFIX = ".staging-"
NAME = re.compile(r"^(checkpoint|emergency|final_actor)_(\d{6})$")


class BundleError(RuntimeError):
    pass


class BundleCorrupt(BundleError):
    pass


def require(condition: bool, message: str, error: type = BundleError) -> None:
    if not condition:
        raise error(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def torch_from_bytes(data: bytes) -> Any:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


class BundleIO:
    """Real filesystem effects; tests subclass this to inject faults."""

    def mkdir(self, path: Path) -> None:
        os.mkdir(path, 0o700)

    def write_file(self, path: Path, data: bytes) -> None:
        with open(path, "xb") as stream:
            self.write(stream, data)
            stream.flush()
            self.fsync(stream.fileno())

    def write(self, stream, data: bytes) -> None:
        stream.write(data)

    def fsync(self, descriptor: int) -> None:
        os.fsync(descriptor)

    def fsync_dir(self, path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            self.fsync(descriptor)
        finally:
            os.close(descriptor)

    def rename(self, source: Path, target: Path) -> None:
        os.rename(source, target)

    def replace(self, source: Path, target: Path) -> None:
        os.replace(source, target)


REAL_IO = BundleIO()


def bundle_name(kind: str, update: int) -> str:
    require(kind in ("checkpoint", "emergency", "final_actor"), f"unknown bundle kind {kind}")
    return f"{kind}_{update:06d}"


def publish_bundle(parent: Path, name: str, payloads: Mapping[str, bytes],
                   manifest: Mapping[str, Any], *, bundle_io: BundleIO = REAL_IO,
                   update_latest: bool = True) -> str:
    """Atomically publish one complete bundle; returns the manifest digest."""
    parent = Path(parent)
    require(NAME.match(name) is not None, f"bundle name {name!r} is not registered")
    target = parent / name
    require(not (target.exists() or target.is_symlink()), f"{name} already exists")
    require(not set(payloads) & {MANIFEST, COMMITTED}, "reserved payload name")
    staging = parent / f"{STAGING_PREFIX}{name}-{os.getpid()}-{secrets.token_hex(6)}"
    renamed = False
    try:
        bundle_io.mkdir(staging)
        files = {}
        for filename in sorted(payloads):
            data = payloads[filename]
            bundle_io.write_file(staging / filename, data)
            files[filename] = {"sha256": sha256_bytes(data), "size_bytes": len(data)}
        document = {**manifest, "bundle_schema": BUNDLE_SCHEMA, "name": name, "files": files}
        manifest_bytes = canonical_bytes(document)
        manifest_sha = sha256_bytes(manifest_bytes)
        bundle_io.write_file(staging / MANIFEST, manifest_bytes)
        bundle_io.write_file(staging / COMMITTED, (manifest_sha + "\n").encode("ascii"))
        bundle_io.fsync_dir(staging)
        require(not (target.exists() or target.is_symlink()), f"{name} appeared concurrently")
        bundle_io.rename(staging, target)
        renamed = True
        bundle_io.fsync_dir(parent)
    except BaseException:
        if not renamed and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    if update_latest:
        write_latest(parent, name, manifest_sha, bundle_io=bundle_io)
    return manifest_sha


def write_latest(parent: Path, name: str, manifest_sha: str, *,
                 bundle_io: BundleIO = REAL_IO) -> None:
    temporary = Path(parent) / f".{LATEST}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    try:
        bundle_io.write_file(temporary, canonical_bytes({"name": name,
                                                         "manifest_sha256": manifest_sha}))
        bundle_io.replace(temporary, Path(parent) / LATEST)
        bundle_io.fsync_dir(parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_write_file(path: Path, data: bytes, *, bundle_io: BundleIO = REAL_IO) -> None:
    """Create-or-replace one small file atomically (tmp, fsync, replace, fsync dir)."""
    path = Path(path)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}"
    try:
        bundle_io.write_file(temporary, data)
        bundle_io.replace(temporary, path)
        bundle_io.fsync_dir(path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


@dataclass(frozen=True)
class VerifiedBundle:
    path: Path
    name: str
    kind: str
    update: int
    manifest: Mapping[str, Any]
    manifest_sha256: str

    def payload(self, filename: str) -> bytes:
        data = (self.path / filename).read_bytes()
        entry = self.manifest["files"][filename]
        require(sha256_bytes(data) == entry["sha256"] and len(data) == entry["size_bytes"],
                f"{self.name}/{filename} changed after verification", BundleCorrupt)
        return data


def verify_bundle(path: Path) -> VerifiedBundle:
    path = Path(path)
    match = NAME.match(path.name)
    require(match is not None, f"{path.name} is not a registered bundle name", BundleCorrupt)
    require(path.is_dir() and not path.is_symlink(), f"{path.name} is not a real directory",
            BundleCorrupt)
    members = {item.name for item in path.iterdir()}
    require(COMMITTED in members, f"{path.name} lacks the COMMITTED marker", BundleCorrupt)
    require(MANIFEST in members, f"{path.name} lacks manifest.json", BundleCorrupt)
    manifest_bytes = (path / MANIFEST).read_bytes()
    marker = (path / COMMITTED).read_bytes()
    require(marker == (sha256_bytes(manifest_bytes) + "\n").encode("ascii"),
            f"{path.name} COMMITTED does not match its manifest", BundleCorrupt)
    try:
        manifest = json.loads(manifest_bytes)
    except ValueError as exc:
        raise BundleCorrupt(f"{path.name} manifest is not JSON") from exc
    require(canonical_bytes(manifest) == manifest_bytes, f"{path.name} manifest not canonical",
            BundleCorrupt)
    require(manifest.get("bundle_schema") == BUNDLE_SCHEMA and manifest.get("name") == path.name,
            f"{path.name} manifest identity differs", BundleCorrupt)
    files = manifest["files"]
    require(members == {MANIFEST, COMMITTED, *files}, f"{path.name} member set differs",
            BundleCorrupt)
    for filename, entry in files.items():
        data = (path / filename).read_bytes()
        require(len(data) == entry["size_bytes"] and sha256_bytes(data) == entry["sha256"],
                f"{path.name}/{filename} hash or size differs", BundleCorrupt)
    return VerifiedBundle(path, path.name, match.group(1), int(match.group(2)), manifest,
                          sha256_bytes(manifest_bytes))


@dataclass(frozen=True)
class ResumeSelection:
    bundle: Optional[VerifiedBundle]
    ignored_staging: tuple[str, ...]
    latest_status: str


def select_resume(parent: Path, *, bundle_io: BundleIO = REAL_IO,
                  repair_lag: bool = True) -> ResumeSelection:
    parent = Path(parent)
    staging, verified, corrupt = [], [], []
    for item in sorted(parent.iterdir()) if parent.is_dir() else []:
        if item.name.startswith(STAGING_PREFIX) or item.name.startswith(f".{LATEST}.tmp-"):
            staging.append(item.name)
            continue
        match = NAME.match(item.name)
        if match is None or match.group(1) == "final_actor":
            continue
        try:
            verified.append(verify_bundle(item))
        except (BundleError, OSError, KeyError) as exc:
            corrupt.append(f"{item.name}: {exc}")
    require(not corrupt, f"refusing resume, corrupt candidates: {corrupt}", BundleCorrupt)
    latest_path = parent / LATEST
    if not verified:
        require(not latest_path.exists(), "LATEST exists but no verified bundle does",
                BundleCorrupt)
        return ResumeSelection(None, tuple(staging), "NO_BUNDLE")
    updates = [bundle.update for bundle in verified]
    require(len(updates) == len(set(updates)), "two bundles claim the same update",
            BundleCorrupt)
    best = max(verified, key=lambda bundle: bundle.update)
    require(latest_path.is_file(), "verified bundles exist but LATEST is missing", BundleCorrupt)
    try:
        pointer = json.loads(latest_path.read_bytes())
    except ValueError as exc:
        raise BundleCorrupt("LATEST is not JSON") from exc
    named = next((b for b in verified if b.name == pointer.get("name")), None)
    require(named is not None, f"LATEST names a missing or unverified bundle {pointer}",
            BundleCorrupt)
    require(named.manifest_sha256 == pointer.get("manifest_sha256"),
            "LATEST manifest digest differs from the bundle it names", BundleCorrupt)
    status = "LATEST_CURRENT"
    if named.update < best.update:
        require(repair_lag, "LATEST lags a newer verified bundle", BundleCorrupt)
        write_latest(parent, best.name, best.manifest_sha256, bundle_io=bundle_io)
        status = f"LATEST_LAGGED_REPAIRED_FROM_{named.name}"
    return ResumeSelection(best, tuple(staging), status)


class AppendOnlyJsonl:
    """Keyed, append-only, resume-safe JSONL ledger.

    Record ``i`` is line ``i``.  Re-emitting an existing index after a resume
    must reproduce the stored line exactly (verified, not appended), so a
    restart can neither lose nor duplicate history.  A torn final fragment
    without a newline (a crash inside ``write``) is not a record and is
    truncated on open; complete lines are never rewritten.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lines: list[bytes] = []
        if self.path.exists():
            data = self.path.read_bytes()
            complete, _, fragment = data.rpartition(b"\n")
            if fragment:
                with open(self.path, "r+b") as stream:
                    stream.truncate(len(complete) + (1 if complete else 0))
                    stream.flush()
                    os.fsync(stream.fileno())
            self.lines = [line for line in complete.split(b"\n")] if complete else []

    def __len__(self) -> int:
        return len(self.lines)

    def record(self, index: int, value: Mapping[str, Any]) -> None:
        line = canonical_bytes({"index": index, **value})
        if index < len(self.lines):
            require(self.lines[index] == line,
                    f"{self.path.name}[{index}] differs from the resumed recomputation",
                    BundleCorrupt)
            return
        require(index == len(self.lines), f"{self.path.name} index gap at {index}")
        with open(self.path, "ab") as stream:
            stream.write(line + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.lines.append(line)

    def prefix(self, count: int) -> dict[str, Any]:
        require(count <= len(self.lines), f"{self.path.name} has fewer than {count} records")
        return {"count": count, "sha256": sha256_bytes(b"".join(
            line + b"\n" for line in self.lines[:count]))}

    def require_prefix(self, expected: Mapping[str, Any]) -> None:
        require(self.prefix(int(expected["count"])) == dict(expected),
                f"{self.path.name} committed prefix differs from the bundle", BundleCorrupt)
