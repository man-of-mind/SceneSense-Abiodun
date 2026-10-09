"""Verify pinned sources once at startup and bind immutable original code.

The readiness path never re-reads or re-hashes source files; every original
numerical function is taken from namespaces that the retained v1 optimization
already proved bit-exact (strided nearest count, fast JSON normalisation).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import types
from typing import Any, Mapping

from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_live_parity_repair_v1 import conflicts as C
from rl_agent.splitfusion_live_parity_repair_v1 import evaluation as E
from rl_agent.splitfusion_live_parity_repair_v1 import source_geometry as S
from rl_agent.splitfusion_live_target_parity_v1 import adapter as A

SCHEMA = "splitfusion.numeric_gt_producer.v1"
PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
AVO_SOURCES = (
    "rl_agent/splitfusion_run5_live_avo_v1/live_gt.py",
    "rl_agent/splitfusion_run5_live_avo_v1/quality_adapter.py",
    "rl_agent/splitfusion_run5_live_avo_v1/gt_source_sync.py",
    "data_collection/route_b_publication_actor_volume_visibility_v1/core.py",
    "data_collection/route_b_publication_actor_volume_visibility_v1/scoring.py",
)


class ProducerBindingError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ProducerBindingError(message)


def file_sha256(path: Path) -> str:
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"regular source file required: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_sources() -> dict[str, str]:
    return {p.name: file_sha256(p) for p in sorted(PACKAGE.glob("*.py")) if not p.name.startswith("test_")}


@dataclass(frozen=True)
class Bound:
    """Immutable handles to the original code objects used on the readiness path."""
    pins: Mapping[str, Mapping[str, str]]
    source_pins: Mapping[str, Mapping[str, str]]   # exactly the retained producer's 9 ``verify_sources`` entries
    avo_pins: Mapping[str, str]
    package: Mapping[str, str]
    binding_sha256: str
    objects: Mapping[str, Any]          # private clone of the collector object-row namespace
    semantic: Mapping[str, Any]         # original training-mask namespace (shared, read-only use)
    static: Mapping[str, Any]           # private static-projection namespace (fast exact nearest count)
    visibility: Any                     # private visibility namespace
    paint: Any                          # original collector person-paint code object
    builder: Any                        # pinned original build_contract module (functions/constants)
    recon: Any                          # roi_mask / depth helpers used by the original builder
    conflicts: Mapping[str, Any]        # original conflict detector functions
    adapter: Any                        # private adapter namespace with fast exact json normalisation


def _clone_namespace(original: Mapping[str, Any]) -> dict:
    namespace = dict(original)
    for name, value in tuple(namespace.items()):
        if isinstance(value, types.FunctionType) and value.__globals__ is original:
            namespace[name] = V1S._function(value, namespace)
    return namespace


@lru_cache(maxsize=1)
def bound() -> Bound:
    pins = S.verify_sources()                      # 9 pinned sources (adapter + source geometry)
    evaluation_pins = E.verify_sources()           # adds detection scorer and quality_adapter pins
    avo_pins = {name: file_sha256(ROOT / name) for name in AVO_SOURCES}
    require(avo_pins["rl_agent/splitfusion_run5_live_avo_v1/quality_adapter.py"] == E.PERSON_SHA256,
            "quality_adapter pin differs from the evaluation pin")
    objects0, semantic, static, visibility, paint = V1S._original_private()
    objects = _clone_namespace(objects0)
    builder, _vis_module, recon = A._original_modules(ROOT)
    conflicts = C._original_functions()
    require(builder.VEHICLE_TAGS == static["VEHICLE_SEMANTIC_TAGS"] == frozenset((14, 15, 16)),
            "builder and static vehicle tag sets differ; mask hoisting is not exact")
    require(builder.MODEL_SIZE == (static["MODEL_WIDTH"], static["MODEL_HEIGHT"]) == (768, 432),
            "model grid differs from the frozen 768x432 target")
    adapter = V1S._adapter_namespace()
    package = package_sources()
    binding = {"schema": SCHEMA, "pins": {**pins, **{k: v for k, v in evaluation_pins.items() if k not in pins}},
               "avo_pins": avo_pins, "package": package}
    digest = hashlib.sha256(A.canonical(A._json_value(binding))).hexdigest()
    return Bound(pins=types.MappingProxyType({**pins, **evaluation_pins}), source_pins=types.MappingProxyType(dict(pins)), avo_pins=types.MappingProxyType(avo_pins),
                 package=types.MappingProxyType(package), binding_sha256=digest, objects=objects,
                 semantic=semantic, static=static, visibility=visibility, paint=paint, builder=builder,
                 recon=recon, conflicts=conflicts, adapter=adapter)


def startup_binding() -> dict:
    """Serializable record of what was verified and bound at startup."""
    value = bound()
    return {"schema": SCHEMA, "binding_sha256": value.binding_sha256, "pins": dict(value.pins),
            "avo_pins": dict(value.avo_pins), "package": dict(value.package),
            "fixed_grid_proved": bool(V1S._fixed_grid_proof()),
            "readiness_path_reverifies_source_files": False}
