"""Registered capacity substitutions for the M3 full-route ceiling (no pinned file is edited).

Each entry of ``contract.CAPACITY_SUBSTITUTIONS`` names one function of a pinned module, the process that owns it and
the exact literal replacements. ``install(ids)`` reads the function's source from the (registration-verified) file,
requires every replacement text to occur EXACTLY once, refuses functions that use zero-argument ``super()`` or closure
variables (they cannot be recompiled faithfully), compiles the result under a registered pseudo-filename in the
function's own module globals and installs it on the owning class. Every installation is recorded (target, original
and substituted source digests) and can be uninstalled. Nothing else in the function changes.
"""
import __future__
import hashlib
import importlib
import inspect
import json
import textwrap
from pathlib import Path

from . import contract as C

FILENAME = "<m3-capacity-substitution:{id}>"


class CapacitySubstitutionError(RuntimeError):
    pass


REGISTRY = {}          # additional registered entries (route_completion adds the child stop-rule substitution)


def _entry(substitution_id):
    matches = [dict(item) for item in (*C.CAPACITY_SUBSTITUTIONS, *C.POSE_SUBSTITUTIONS, *REGISTRY.values())
               if item["id"] == substitution_id]
    if len(matches) != 1:
        raise CapacitySubstitutionError("unregistered capacity substitution " + repr(substitution_id))
    return matches[0]


def _resolve(target):
    module_name, qualified = target.split(":")
    module = importlib.import_module(module_name)
    if "." in qualified:
        owner_name, function_name = qualified.split(".")
        owner = getattr(module, owner_name)
    else:
        owner, function_name = module, qualified
    function = vars(owner).get(function_name)
    function = getattr(function, "__m3_original__", function)      # resolve the retained function when installed
    if not inspect.isfunction(function):
        raise CapacitySubstitutionError(f"{target} is not a plain function defined on its class")
    return module, owner, function_name, function


def substituted_source(substitution_id):
    """(original source, substituted source) for one registered entry; raises unless every replacement occurs once."""
    entry = _entry(substitution_id)
    _, _, _, function = _resolve(entry["target"])
    base = inspect.unwrap(function)                 # @contextmanager etc.: the source and code of the defined function
    original = textwrap.dedent(inspect.getsource(base))
    # zero-argument super()/closures in the function's OWN body cannot be recompiled faithfully; super() inside classes
    # that the recompiled body itself defines is fine (registered per entry as nested_classes)
    if "super" in base.__code__.co_names or base.__code__.co_freevars or (
            "super()" in original and not entry.get("nested_classes")):
        raise CapacitySubstitutionError(entry["target"] + " uses super()/closures; refusing to recompile")
    text = original
    for old, new in entry["replacements"]:
        if text.count(old) != 1:
            raise CapacitySubstitutionError(f"{entry['target']}: {old!r} occurs {text.count(old)} times (expected 1)")
        text = text.replace(old, new)
    return original, text


class Installed:
    def __init__(self):
        self.records, self._restore = [], []

    def uninstall(self):
        while self._restore:
            owner, name, function = self._restore.pop()
            setattr(owner, name, function)

    def document(self):
        return dict(schema="splitfusion.m3_route_b_v3.capacity_substitutions.v1", installed=list(self.records),
                    restored=not self._restore)


def install(ids, installed=None):
    installed = installed or Installed()
    for substitution_id in ids:
        entry = _entry(substitution_id)
        module, owner, name, function = _resolve(entry["target"])
        if getattr(vars(owner).get(name), "__m3_capacity_substitution__", None):
            raise CapacitySubstitutionError("capacity substitution already installed: " + substitution_id)
        original, text = substituted_source(substitution_id)
        namespace = {}
        flags = 0
        for feature in __future__.all_feature_names:          # recompile under the module's own future flags
            flag = getattr(__future__, feature).compiler_flag
            if inspect.unwrap(function).__code__.co_flags & flag:
                flags |= flag
        exec(compile(text, FILENAME.format(id=substitution_id), "exec", flags=flags, dont_inherit=True),
             inspect.unwrap(function).__globals__, namespace)       # the defining module's globals (not a decorator's)
        replacement = namespace[name]
        if inspect.unwrap(replacement).__code__.co_freevars or replacement.__name__ != function.__name__:
            raise CapacitySubstitutionError("recompiled function shape differs: " + substitution_id)
        replacement.__qualname__ = function.__qualname__
        replacement.__module__ = function.__module__
        replacement.__m3_capacity_substitution__ = substitution_id
        replacement.__m3_original__ = function
        setattr(owner, name, replacement)
        installed._restore.append((owner, name, function))
        installed.records.append(dict(
            id=substitution_id, process=entry["process"], target=entry["target"],
            module_file=str(Path(module.__file__).resolve()),
            module_file_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            substituted_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            replacements=entry["replacements"]))
    return installed


def write_record(installed, path):
    with open(path, "x") as stream:
        json.dump(installed.document(), stream, indent=1, sort_keys=True)
        stream.write("\n")


UE_CHILD = ("C4_UE_TICKETS", "C5_UE_POSE_ARCHIVE")
GT_SERVICE = ("C3_SERVICE_ARCHIVE_SIZING",)
PREP_WORKER = ("C1_PREP_DEFERRED_ENGINE", "C2_PREP_PRODUCER_SOURCE")

POSE = ("P1_OWNED_POSE_RECONSTRUCTED", "P2_WIRE_SOURCE_CAMERA")
POSE_GT_WORKER = ("P3_GT_LIVE_CONTEXT_SOURCE_CAMERA",)
