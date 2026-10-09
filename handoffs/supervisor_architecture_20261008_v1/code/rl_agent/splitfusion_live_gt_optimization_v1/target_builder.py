"""Code-only compilation cache and frame-private temporary target output.

Every invocation validates native inputs and all original source hashes. Only
immutable code objects are cached; each original module/function receives fresh
globals. The original build_contract, PNG/CSV writers/readers, and hash helpers
run unchanged. Their temporary output is memory-backed; durable transport,
fsync, target restoration, evidence and reward evaluation remain untouched.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import types
from typing import Any, Mapping

import cv2
import numpy as np

from rl_agent.splitfusion_live_target_parity_v1 import adapter as A

SCHEMA = "splitfusion.live_gt_optimization.target_builder.v1"
TABLES = ("object_boxes.csv", "object_ignore_regions.csv", "target_manifest.csv")


def _require(condition, message):
    if not condition:
        raise A.TargetInputError(message)


def _verified_sources(root):
    """Read once; the exact verified bytes are also the compilation inputs."""
    root = Path(root).resolve(strict=True)
    sources, payloads = {}, {}
    for name, (relative, expected) in A.SOURCES.items():
        raw = (root/relative).read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        _require(actual == expected, f"pinned {name} source changed: {actual}")
        sources[name] = {"path": relative, "sha256": actual}
        payloads[name] = raw
    return root, sources, payloads


@lru_cache(maxsize=24)
def _code(path, raw, function=None):
    tree = ast.parse(raw, filename=path)
    if function is not None:
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
        _require(len(nodes) == 1, "pinned original function not unique")
        tree = ast.Module(body=nodes, type_ignores=[])
    return compile(tree, path, "exec")


def _modules(root, payloads):
    modules = {}
    for role, name in (("builder", "_private_original_v010_builder"), ("visibility", "_private_original_v3_visibility")):
        path = str(root/A.SOURCES[role][0])
        module = types.ModuleType(name)
        module.__file__, module.__package__ = path, ""
        exec(_code(path, payloads[role]), module.__dict__)
        modules[role] = module
    visibility = modules["visibility"]
    ns = {"np": np, "math": math, "Any": Any, "Mapping": Mapping,
          "reconstruct_consistent_mask": visibility.reconstruct_consistent_mask}
    path = str(root/A.SOURCES["reconciliation"][0])
    exec(_code(path, payloads["reconciliation"], "roi_mask"), ns)
    recon = types.SimpleNamespace(roi_mask=ns["roi_mask"], decode_depth_bgra=visibility.decode_depth_bgra,
                                  depth_is_plausible=visibility.depth_is_plausible)
    return modules["builder"], visibility, recon


def prewarm(*, root=A.ROOT):
    """Verify and compile originals only; never accept a frame or touch history."""
    root, sources, payloads = _verified_sources(root)
    _modules(root, payloads)
    return {"schema": SCHEMA, "sources": sources, "frame_history_advanced": False,
            "frame_inputs_consumed": 0, "cache": _code.cache_info()._asdict()}


class _TextOutput(io.StringIO):
    def __init__(self, filesystem, name):
        super().__init__(newline="")
        self.filesystem, self.name = filesystem, name

    def close(self):
        if not self.closed:
            self.filesystem.files[self.name] = self.getvalue().encode("utf-8")
        super().close()


class _MemoryPath:
    """Minimal exact surface exercised by original temporary-output helpers."""
    def __init__(self, filesystem, path):
        self.filesystem, self.path = filesystem, PurePosixPath(path)
        _require(".." not in self.path.parts and self.path.is_relative_to(filesystem.prefix), "unsafe temporary target path")

    def __truediv__(self, child):
        return _MemoryPath(self.filesystem, self.path/str(child))

    def __str__(self):
        return str(self.path)

    @property
    def parent(self):
        return _MemoryPath(self.filesystem, self.path.parent)

    def exists(self):
        return str(self) in self.filesystem.files or self.path in self.filesystem.directories

    def mkdir(self, parents=False, exist_ok=False):
        _require(self.path in self.filesystem.allowed_directories, "unregistered temporary target directory")
        if self.exists():
            if exist_ok and self.path in self.filesystem.directories:
                return
            raise FileExistsError(str(self))
        if not parents and self.path.parent not in self.filesystem.directories:
            raise FileNotFoundError(str(self.parent))
        self.filesystem.directories.update(p for p in (self.path, *self.path.parents)
            if p.is_relative_to(self.filesystem.prefix))

    def open(self, mode="r", encoding=None, newline=None):
        _require(str(self) in self.filesystem.allowed_files, "unregistered temporary target file")
        _require(mode in ("x", "r", "rb") and encoding in (None, "utf-8") and newline in (None, ""),
                 "unregistered temporary target file operation")
        if mode == "x":
            if self.exists():
                raise FileExistsError(str(self))
            if self.path.parent not in self.filesystem.directories:
                raise FileNotFoundError(str(self.parent))
            self.filesystem.files[str(self)] = b""
            return _TextOutput(self.filesystem, str(self))
        if str(self) not in self.filesystem.files:
            raise FileNotFoundError(str(self))
        raw = self.filesystem.files[str(self)]
        return io.BytesIO(raw) if mode == "rb" else io.StringIO(raw.decode("utf-8"), newline=newline)


class _MemoryFiles:
    def __init__(self, sample, split):
        self.prefix = PurePosixPath("/_private_frame_target")
        target = self.prefix/"contracts/v010"/split
        self.allowed_directories = {self.prefix, self.prefix/"contracts", self.prefix/"contracts/v010", target,
                                    target/"segmentation_masks", target/"object_ignore_masks"}
        self.allowed_files = {str(target/name) for name in TABLES} | {
            str(target/"segmentation_masks"/(sample+".png")), str(target/"object_ignore_masks"/(sample+".png"))}
        self.files = {}; self.directories = {self.prefix}
        self.root = _MemoryPath(self, self.prefix)

    def write_png(self, path, image):
        path = str(path)
        _require(path in self.allowed_files and path.endswith(".png"), "unregistered temporary PNG write")
        if path in self.files:
            raise FileExistsError(path)
        if PurePosixPath(path).parent not in self.directories:
            raise FileNotFoundError(str(PurePosixPath(path).parent))
        success, encoded = cv2.imencode(".png", image)
        if success:
            self.files[path] = encoded.tobytes()
        return success


def build_complete_v010(inputs, *, root=A.ROOT):
    """Call original numerical and serialization functions in fresh namespaces."""
    A._validate(inputs)
    root, sources, payloads = _verified_sources(root)
    builder, _visibility, recon = _modules(root, payloads)
    frame = dict(inputs.frame)
    source_view = builder.FROZEN/"views"/inputs.split
    reads = {str(source_view/frame["experiment_id"]/"depth"/(frame["sample_id"]+".png")): inputs.depth_bgra,
             str(source_view/frame["instance_raw_path"]): inputs.raw_semantic,
             str(source_view/frame["mask_path"]): inputs.original_v3_mask}
    _require(len(reads) == 3, "three original image paths must be distinct")
    output = _MemoryFiles(frame["sample_id"], inputs.split)
    class Cv2Inputs:
        def __getattr__(self, name):
            return getattr(cv2, name)
        def imread(self, path, flags):
            path = str(path)
            if path in reads:
                return reads[path].copy()
            raise A.TargetInputError(f"unregistered original image read: {path}")
        def imwrite(self, path, image):
            return output.write_png(path, image)
    builder.cv2 = Cv2Inputs()
    summary = builder.build_contract(contract="v010", split=inputs.split, manifest=[frame],
        boxes=[dict(r) for r in inputs.actor_boxes], visibility=[dict(r) for r in inputs.actor_visibility],
        static_rows=[dict(r) for r in inputs.static_candidates], quarantine=inputs.quarantine,
        recon=recon, output_root=output.root)
    directory = output.root/"contracts/v010"/inputs.split
    tables = {name: builder.read_csv(directory/name) for name in TABLES}
    _require(set(output.files) == output.allowed_files, "incomplete temporary target output")
    seg = output.files[str(directory/"segmentation_masks"/(frame["sample_id"]+".png"))]
    obj = output.files[str(directory/"object_ignore_masks"/(frame["sample_id"]+".png"))]
    # Keep the second source revalidation of the baseline adapter. No filesystem
    # or mutable source change after prewarm can silently reuse stale code.
    _require(A.verify_sources(root) == sources, "original source changed during target construction")
    return {"contract": A.CONTRACT, "live_qualified": False, "sources": sources,
            "provenance": dict(inputs.provenance), "summary": summary, "tables": tables,
            "segmentation": cv2.imdecode(np.frombuffer(seg, np.uint8), cv2.IMREAD_UNCHANGED),
            "object_ignore": cv2.imdecode(np.frombuffer(obj, np.uint8), cv2.IMREAD_UNCHANGED),
            "segmentation_png": seg, "object_ignore_png": obj}


@contextmanager
def install_target_builder(*, root=A.ROOT):
    """UE producer scope only; retain original E.prepare_target and validation."""
    root = Path(root).resolve(strict=True)
    original = A.build_complete_v010
    def installed(inputs, *, root=root):
        return build_complete_v010(inputs, root=root)
    A.build_complete_v010 = installed
    try:
        yield {"schema": SCHEMA, "root": str(root), "prewarm": lambda: prewarm(root=root),
               "cached_frame_state": False, "durable_transport_unchanged": True}
    finally:
        A.build_complete_v010 = original
