"""Verify actual mixed masks; original scheduling is a labeled reference only."""
import json
from pathlib import Path
from rl_agent.splitfusion_live_gt_placement_trial_v1 import contract as T
from . import contract as C


def require(value, message):
    if not value:
        raise ValueError(message)


def original_scheduling_reference(context, pid, *, mode=None):
    """Return the original reference after validating every actual TID mask.

    This does not alter or relabel context['scheduling']. Treatment INIT/RUN
    intentionally differs from CLOSE; only the original reference is invariant.
    """
    T.validate_context(context, T.sources())
    placement = context.get("runtime_placement", {})
    actual_mode = placement.get("trial_mode")
    require(placement.get("schema") == T.SCHEMA
            and actual_mode in ("control", "p4") and (mode is None or actual_mode == mode),
            "worker trial mode differs")
    expected = C.trial(actual_mode)
    require(context.get("pid") == placement.get("worker_pid") == placement.get("main_tid") == pid,
            "worker placement owner differs")
    require(placement.get("requested_cpu") == expected["placement_cpu"]
            and placement.get("affinity_scope") == expected["affinity_scope"]
            and placement.get("mode") == ("main_tid_affinity" if actual_mode == "p4" else "control_unbound")
            and placement.get("native_pool_preparation") is False
            and placement.get("helper_affinity_mutated") is False
            and placement.get("parent_affinity_mutated") is False
            and placement.get("archive_affinity_mutated") is False
            and placement.get("errors") == [], "worker placement scope differs")
    original = placement["original_scheduling"]
    closed = placement.get("closed") is True
    require(placement.get("restored") is closed, "placement restoration differs")
    active = dict(original, affinity=[4]) if actual_mode == "p4" and not closed else original
    require(context["scheduling"] == active and str(pid) in context["threads"], "main affinity differs")
    for tid, row in context["threads"].items():
        require(row == (active if int(tid) == pid else original), "helper affinity/scheduling changed")
    for field in ("archive_start", "archive_end"):
        archive = placement.get(field)
        require(isinstance(archive, dict) and archive.get("pid") != pid and archive.get("threads"),
                "archive process observation missing")
        require(all(row.get("parent_pid") == pid and all(row.get(key) == value for key, value in original.items())
                    for row in archive["threads"].values()), "archive scheduling or owner changed")
    require(placement.get("set_calls") == ([pid] if actual_mode == "p4" else []),
            "foreign affinity mutation")
    if closed:
        require(placement.get("restore_calls") == ([pid] if actual_mode == "p4" else []),
                "foreign affinity restoration")
    gc = context.get("runtime_gc", {})
    require(gc.get("mode") == "control" and gc.get("enabled") is True
            and gc.get("thresholds") == [700, 10, 10] and gc.get("freeze_calls") == 0
            and gc.get("freeze_count") == 0, "matched control GC policy differs")
    return original


def verify_trial_scope(cell, *, mode):
    cell = Path(cell)
    timeline = json.loads((cell / "CPU_GT_PROCESS_TIMELINE.json").read_bytes())
    require(len(timeline["clients"]) == 1, "one trial worker required")
    client = timeline["clients"][0]
    pid = client["pid"]
    contexts = [client["ready"]["process_context"]]
    contexts += [row["process_context"] for row in timeline["requests"]]
    contexts += [client["close"]["process_context"]]
    original = original_scheduling_reference(contexts[0], pid, mode=mode)
    for index, context in enumerate(contexts):
        require(original_scheduling_reference(context, pid, mode=mode) == original,
                "original scheduling reference changed")
        place = context["runtime_placement"]
        closed = index == len(contexts)-1
        require(place["closed"] is closed and place["request_count"] == min(index, len(timeline["requests"])),
                "worker lifecycle/causal request count differs")
        if 0 < index < len(contexts)-1:
            samples = place["request_samples"]
            require(len(samples) == 1, "one bounded request resource sample required")
            sample = samples[0]
            require(sample["frame_id"] == timeline["requests"][index-1]["identity"]["frame_id"]
                    and sample["outcome"] == "OK", "request resource sample identity differs")
            for side in ("before", "after"):
                require(sample[side]["native_tid"] == pid
                        and sample[side]["affinity"] == ([4] if mode == "p4" else original["affinity"])
                        and sample[side]["cpu"] in sample[side]["affinity"], "request CPU/mask differs")
    receipt = client["close"].get("extended_close_receipt", {})
    path = Path(receipt.get("path", ""))
    require(path.is_absolute() and path.resolve().is_relative_to(cell.resolve())
            and path.name == "TRIAL_WORKER_CLOSE.json" and not path.is_symlink()
            and C.sha(path) == receipt.get("sha256"), "full native worker close receipt invalid")
    document = json.loads(path.read_bytes())
    require(document.get("schema") == T.SCHEMA and document.get("trial_mode") == mode,
            "full close trial identity differs")
    full = document["full_close"]
    timing = dict(full["timing"])
    extensions = {key: timing.pop(key) for key in ("runtime_gc_scope", "runtime_placement_scope")}
    require(document.get("removed_extension_keys") == sorted(extensions)
            and document.get("extensions") == extensions, "close extension projection differs")
    native = dict(full, timing=timing)
    projected = dict(client["close"])
    projected.pop("extended_close_receipt")
    require(native == projected, "full and projected native close differ")
    records = dict(extensions, native_drain=document["native_drain"])
    documents = {}
    for key, record in records.items():
        owned = Path(record["path"])
        require(owned.is_absolute() and owned.resolve().is_relative_to(cell.resolve())
                and not owned.is_symlink() and C.sha(owned) == record["sha256"],
                "native close extension file/hash differs")
        documents[key] = json.loads(owned.read_bytes())
    require(documents["native_drain"] == timing, "native drain differs from close projection")
    gc = documents["runtime_gc_scope"]
    require(gc["mode"] == "control" and gc["closed"] is True and gc["settings_restored"] is True
            and gc["freeze_calls"] == gc["unfreeze_calls"] == gc["freeze_count"] == 0
            and gc["errors"] == [] and gc["callback_errors"] == gc["dropped"] == 0
            and gc["native_close_succeeded"] is True
            and gc["startup_collection"]["generation"] == 2
            and gc["startup_collection"]["end_raw_ns"] >= gc["startup_collection"]["start_raw_ns"],
            "matched control GC startup/close receipt differs")
    placement = documents["runtime_placement_scope"]
    require(placement["closed"] is True and placement["restored"] is True
            and placement["errors"] == [] and placement["trial_mode"] == mode
            and placement["request_count"] == len(timeline["requests"]),
            "durable placement restoration receipt differs")
    return dict(passed=True, trial=C.trial(mode), worker_pid=pid,
        original_scheduling_reference=original, effective_contexts_unchanged=True,
        main_singleton_is_intentional_init_run_exception=mode == "p4",
        actual_restored_close=contexts[-1]["scheduling"], checked_contexts=len(contexts),
        extended_close_receipt=receipt, main_thread_cpu_excludes_native_helpers=True)
