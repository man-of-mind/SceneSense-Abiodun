"""Reversible, prep-worker-only installation of one bounded accounting session."""
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN

from . import accounting as A

ORIGINAL_PAYLOAD_STORAGE_BYTES = OWN.payload_storage_bytes
SUBSTITUTION = dict(target="rl_agent.splitfusion_numeric_gt_integration_v2.ownership:payload_storage_bytes",
                    replacement="rl_agent.splitfusion_gt_storage_accounting_v2.accounting:AccountingSession",
                    process="GT prep worker only", default="OFF")
_SAVED = []
_SESSION = None


def install(max_entries=A.DEFAULT_MAX_ENTRIES):
    global _SESSION
    if _SAVED or _SESSION is not None:
        raise RuntimeError("storage-accounting switch already installed")
    if OWN.payload_storage_bytes is not ORIGINAL_PAYLOAD_STORAGE_BYTES:
        raise RuntimeError("owner storage accounting is not the original payload_storage_bytes")
    session = A.AccountingSession(max_entries=max_entries)
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
