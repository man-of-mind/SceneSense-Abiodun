"""Bounded, owned prediction snapshots; disk flushing occurs after evaluation.

Snapshots are taken on ticket submission, including tickets whose GT never
arrives. This module does not generate GT or change any reward/terminal.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

SCHEMA = "splitfusion.parity_repair.prediction_archive.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class PredictionArchive:
    def __init__(self, *, max_frames: int, max_bytes: int = 512_000_000):
        if type(max_frames) is not int or not 1 <= max_frames <= 1000:
            raise ValueError("explicit 1..1000 frame bound required")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 1_000_000_000:
            raise ValueError("explicit bounded archive memory required")
        self.max_frames, self.max_bytes = max_frames, max_bytes
        self._lock = threading.Lock()
        self._items = {}
        self._bytes = 0
        self._closed = False

    @staticmethod
    def key(identity):
        # Enqueue time is evidence, not a new ticket identity.
        return hashlib.sha256(canonical({k: v for k, v in identity.items()
                                         if k != "enqueued_wall_ns"})).hexdigest()

    @staticmethod
    def identity(ticket):
        from rl_agent.splitfusion_hybrid_sac_live_route_b_v2 import run4_map_protocol_v2 as MP
        action = MP.run4_identity(ticket.envelope)
        MP.validate_run4_identity(action)
        if (ticket.context.frame_id != ticket.envelope.frame_id
                or ticket.context.capture_timestamp_ns != ticket.envelope.capture_timestamp_ns
                or ticket.gt_identity["frame_id"] != ticket.context.frame_id
                or ticket.gt_identity["capture_timestamp_ns"] != ticket.context.capture_timestamp_ns
                or ticket.gt_identity["stream_id"] != ticket.context.stream_id
                or ticket.context.sequence_id != ticket.envelope.tensor_seq):
            raise ValueError("prediction/ticket identity mismatch")
        return {"gt_identity": dict(ticket.gt_identity), "action_identity": action,
                "capture_timestamp_ns": ticket.context.capture_timestamp_ns,
                "enqueued_wall_ns": ticket.enqueued_wall_ns}

    def add(self, ticket):
        identity = self.identity(ticket)
        if not identity["action_identity"]["reward_requested"]:
            raise ValueError("archive takes reward tickets, not held-frame pseudo-tickets")
        labels = np.asarray(ticket.predicted_mask)
        if (labels.dtype != np.uint8 or labels.ndim != 2
                or labels.shape[0] > 448 or labels.shape[1] > 768
                or labels.size == 0 or not np.isin(labels, (0, 1, 2)).all()):
            raise ValueError("invalid dense prediction")
        if len(ticket.records) > 4096:
            raise ValueError("prediction record bound exceeded")
        records = canonical(list(ticket.records))  # own primitive values, reject NaN
        labels = labels.copy(order="C")
        labels.flags.writeable = False
        key = self.key(identity)
        digest = hashlib.sha256(labels.tobytes() + records).hexdigest()
        cost = labels.nbytes + len(records) + len(canonical(identity))
        with self._lock:
            if self._closed:
                raise ValueError("prediction archive already closed")
            if key in self._items:
                if self._items[key]["prediction_digest"] != digest:
                    raise ValueError("conflicting duplicate prediction")
                return key
            if len(self._items) >= self.max_frames or self._bytes + cost > self.max_bytes:
                raise ValueError("prediction archive capacity exceeded; never silently drop")
            self._items[key] = {"identity": identity, "labels": labels, "records": records,
                "prediction_digest": digest, "snapshot_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
                "evaluation": None}
            self._bytes += cost
        return key

    def evaluated(self, ticket, evidence):
        key = self.key(self.identity(ticket))
        payload = canonical(evidence)
        with self._lock:
            if self._closed or key not in self._items:
                raise ValueError("no open prediction snapshot for evaluation")
            item = self._items[key]
            if item["evaluation"] is not None:
                if item["evaluation"] != payload:
                    raise ValueError("conflicting evaluation evidence")
                return
            if self._bytes + len(payload) > self.max_bytes:
                raise ValueError("evaluation evidence memory bound exceeded")
            item["evaluation"] = payload
            self._bytes += len(payload)

    def owned_ticket(self, ticket):
        """Isolate asynchronous scoring from callers mutating their containers."""
        key = self.key(self.identity(ticket))
        with self._lock:
            if self._closed or key not in self._items:
                raise ValueError("no open snapshot for owned scoring ticket")
            item = self._items[key]
            labels = item["labels"].copy()
            labels.flags.writeable = False
            records = tuple(json.loads(item["records"]))
        return replace(ticket, records=records, predicted_mask=labels,
                       gt_identity=dict(ticket.gt_identity))

    def flush(self, directory: Path):
        """Call only after evaluator drain. Manifest last; partial failure stays visible."""
        directory = Path(directory)
        with self._lock:
            if self._closed:
                raise ValueError("prediction archive already flushed")
            directory.mkdir(parents=True, exist_ok=False)
            self._closed = True
            manifest = {}
            for key, item in self._items.items():
                buffer = io.BytesIO()
                np.save(buffer, item["labels"], allow_pickle=False)
                payloads = {f"{key}.labels.npy": buffer.getvalue(),
                    f"{key}.records.json": item["records"],
                    f"{key}.identity.json": canonical({"schema": SCHEMA, **item["identity"],
                        "prediction_digest": item["prediction_digest"],
                        "snapshot_raw_ns": item["snapshot_raw_ns"]})}
                if item["evaluation"] is not None:
                    payloads[f"{key}.evaluation.json"] = item["evaluation"]
                for name, payload in payloads.items():
                    with (directory / name).open("xb") as stream:
                        stream.write(payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    manifest[name] = hashlib.sha256(payload).hexdigest()
            result = {"schema": SCHEMA, "frames": len(self._items), "owned_bytes": self._bytes,
                "evaluated_frames": sum(v["evaluation"] is not None for v in self._items.values()),
                "sha256": manifest, "includes_gt_unavailable_tickets": True,
                "flush_after_feedback": True, "quality_and_terminals_unchanged": True}
            with (directory / "MANIFEST.json").open("xb") as stream:
                stream.write(canonical(result))
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return result
