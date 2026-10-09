"""``NumericSourceProducerV2``: the v1 numeric producer with exact frame-local reuse (see ``exact_frame``).

Same constructor contract, validators, staging (compute/commit/cancel), history digests, identities and evidence as
``splitfusion_numeric_gt_producer_v1.source.NumericSourceProducer``. ``compute`` is the v1 code object executed in a
private copy of its module namespace in which only ``static_rows`` is replaced, plus three private substitutions:
* the producer's ``bound.visibility`` view (ROI-limited ``visibility_from_corners``, same return contract);
* a per-frame memo of ``actor_bbox_world_points`` installed around the call (always restored);
* the conflict state built from the already qualified exact private conflict namespace.
Nothing global is mutated beyond the per-call, restored ``objects`` entry that v1 already uses the same way for
its hoisted radar transform. ``bound.binding_sha256`` still names the pinned originals; this package's own source
hashes are reported by ``package_sources()`` and must be bound by any future registration.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import types

from rl_agent.splitfusion_live_gt_optimization_v1 import source_geometry as V1S
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS

from . import exact_frame as X

PACKAGE = Path(__file__).resolve().parent


def package_sources():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.glob("*.py"))
            if not p.name.startswith("test_")}


def _compute_v2():
    namespace = dict(vars(NS))
    stats = {}

    def static_rows(bound, **kwargs):
        rows, ambiguity_rows, summary, frame_stats = X.static_rows(bound, **kwargs)
        stats.clear()
        stats.update(frame_stats)
        return rows, ambiguity_rows, summary
    namespace["static_rows"] = static_rows
    return V1S._function(NS.NumericSourceProducer.compute, namespace), stats


class NumericSourceProducerV2(NS.NumericSourceProducer):
    VARIANT = "NUMERIC_PRODUCER_V2_EXACT_FRAME_LOCAL_REUSE"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        visibility = dict(vars(self.bound.visibility))
        visibility["visibility_from_corners"] = X.make_visibility_from_corners(self.bound.visibility)
        self.bound = replace(self.bound, visibility=types.SimpleNamespace(**visibility))
        conflicts = V1S._conflict_namespace(self.bound.adapter)
        self.conflict_state = conflicts.IncrementalConflictState(self.catalogue, max_frames=self.max_frames)
        if self.catalogue_digest != self.conflict_state.catalogue_digest:
            raise NS.SourceGeometryError("catalogue digest differs between producer and exact private conflict state")
        self._compute, self.frame_stats = _compute_v2()
        self.last_frame_stats = None

    def compute(self, **kwargs):
        with X.frame_bbox_cache(self.bound.objects) as calls:
            staged = self._compute(self, **kwargs)
        self.last_frame_stats = dict(self.frame_stats, bbox_calls=calls[0], bbox_computed=calls[1])
        return staged
