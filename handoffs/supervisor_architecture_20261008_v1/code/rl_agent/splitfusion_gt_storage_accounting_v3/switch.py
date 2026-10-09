"""Reversible, prep-worker-only installation of one exact V3 accounting session (default OFF)."""
from types import FunctionType

from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN

from . import accounting as A

ORIGINAL_PAYLOAD_STORAGE_BYTES = OWN.payload_storage_bytes
SUBSTITUTION = dict(target="rl_agent.splitfusion_numeric_gt_integration_v2.ownership:payload_storage_bytes",
                    replacement="rl_agent.splitfusion_gt_storage_accounting_v3.accounting:AccountingSession",
                    process="GT prep worker only", default="OFF")
_SAVED = []
_SESSION = None


def is_original(function):
    """True only for the function object defined in ownership.py (never a session or another wrapper)."""
    return (type(function) is FunctionType and function.__module__ == OWN.__name__
            and function.__qualname__ == "payload_storage_bytes"
            and function.__code__.co_filename == OWN.__file__)


if not is_original(ORIGINAL_PAYLOAD_STORAGE_BYTES):
    raise RuntimeError("REFUSED: ownership.payload_storage_bytes was already substituted at switch import")


def install(max_objects=A.DEFAULT_MAX_OBJECTS):
    global _SESSION
    if _SAVED or _SESSION is not None:
        raise RuntimeError("storage-accounting switch already installed")
    if OWN.payload_storage_bytes is not ORIGINAL_PAYLOAD_STORAGE_BYTES:
        raise RuntimeError("owner storage accounting is not the original payload_storage_bytes")
    session = A.AccountingSession(max_objects=max_objects)
    _SAVED.append(OWN.payload_storage_bytes)
    _SESSION = session
    OWN.payload_storage_bytes = session
    return session


def uninstall():
    global _SESSION
    if _SAVED:
        OWN.payload_storage_bytes = _SAVED.pop()
    if _SESSION is not None:
        _SESSION.close()
        _SESSION = None


def installed():
    return bool(_SAVED) and _SESSION is not None and OWN.payload_storage_bytes is _SESSION


def session():
    return _SESSION
