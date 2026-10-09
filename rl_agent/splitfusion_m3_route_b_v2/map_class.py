"""Map-side class preservation (M3.1 v2, map process only).

The Run-4 map update carries the model's prediction records (``class_name`` = "vehicle"/"person", ``internal_class``),
but the frozen baseline normalizer reads ``type`` and defaults to "Unknown" (v1: 45,731/45,731 installed objects).
This conversion runs in the map process, on a COPY of the update, immediately before the frozen install:
each record gains ``type`` from its own ``class_name`` (vehicle -> Vehicle, person -> Pedestrian); a record whose class
is not one of the two keeps no type (baseline "Unknown") and is counted; a record that already carries a different
``type`` is refused. No GT is read. Count, order, scores, coordinates and every other field are unchanged; ingest
validation, the ACK, the edge records and the scorer inputs never see the copy. Counters are written at service close
next to the ingest CSV (``m3_map_class_conversion.json``).
"""
import json
import threading

from . import contract as C

MAPPING = dict(C.MAP_CLASS["mapping"])
_LOCK = threading.Lock()
COUNTS = dict(updates=0, objects=0, typed=0, unknown_class=0, by_type={}, by_internal_class={})


class MapClassError(RuntimeError):
    pass


def convert(document):
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
    if document.get("schema") != MP.RUN4_UPDATE_SCHEMA:
        return document
    records, out, typed, unknown, by_type, by_internal = document.get("records") or [], [], 0, 0, {}, {}
    for record in records:
        value = MAPPING.get(record.get("class_name"))
        if "type" in record and record["type"] != value:
            raise MapClassError(f"record already carries type {record['type']!r} != {value!r}")
        copy = dict(record)
        if value is None:
            unknown += 1
        else:
            copy["type"] = value
            typed += 1
            by_type[value] = by_type.get(value, 0) + 1
            key = f"{record.get('class_name')}:{record.get('internal_class')}"
            by_internal[key] = by_internal.get(key, 0) + 1
        out.append(copy)
    with _LOCK:
        COUNTS["updates"] += 1
        COUNTS["objects"] += len(records)
        COUNTS["typed"] += typed
        COUNTS["unknown_class"] += unknown
        for k, v in by_type.items():
            COUNTS["by_type"][k] = COUNTS["by_type"].get(k, 0) + v
        for k, v in by_internal.items():
            COUNTS["by_internal_class"][k] = COUNTS["by_internal_class"].get(k, 0) + v
    return {**dict(document), "records": out}


def install_map_class_conversion():
    """Wrap phase6_map_server_v2.install (called by map_profiled_v1.main) in this process."""
    from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import phase6_map_server_v2 as M6
    original_install = M6.install
    if getattr(original_install, "__m3_map_class__", False):
        raise MapClassError("map class conversion already installed")

    def install(contract):
        server = original_install(contract)
        inner = server.install_under_state_lock

        def install_with_class(document, ingest_at):
            return inner(convert(document), ingest_at)

        server.install_under_state_lock = install_with_class
        service = server.DirectMapIngestService

        class ClassCountingService(service):
            def close(self):
                try:
                    return super().close()
                finally:
                    if self.ingest_csv is not None:
                        path = self.ingest_csv.with_name("m3_map_class_conversion.json")
                        if not path.exists():
                            with _LOCK:
                                doc = dict(schema="splitfusion.m3_route_b_v2.map_class.v1", rule=C.MAP_CLASS,
                                           counts=json.loads(json.dumps(COUNTS)))
                            path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")

        server.DirectMapIngestService = ClassCountingService
        return server

    install.__m3_map_class__ = True
    M6.install = install
    return original_install
