"""Process-local, reversible switch: the GT prep owner's storage accounting uses the exact candidate.

One explicit substitution, restored by ``uninstall()`` (default OFF), the producer-v2 / tilt ``switch`` pattern:
``splitfusion_numeric_gt_integration_v2.ownership.payload_storage_bytes`` -> ``accounting.payload_storage_bytes``.
Its only production caller is ``ProducerOwner._account`` (``ownership.py:243-244``), which resolves the module global
at call time; ``DeferredProducerOwner`` inherits it unchanged.  Nothing else is rebound.  Any live use must register
this substitution and the candidate sources; this module does not do that.
"""
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OWN

from . import accounting as A

ORIGINAL_PAYLOAD_STORAGE_BYTES = OWN.payload_storage_bytes
SUBSTITUTION = dict(target="rl_agent.splitfusion_numeric_gt_integration_v2.ownership:payload_storage_bytes",
                    replacement="rl_agent.splitfusion_gt_storage_accounting_v1.accounting:payload_storage_bytes",
                    process="GT prep worker only")
_SAVED = []


def install():
    if _SAVED:
        raise RuntimeError("storage-accounting switch already installed")
    if OWN.payload_storage_bytes is not ORIGINAL_PAYLOAD_STORAGE_BYTES:
        raise RuntimeError("owner storage accounting is not the original payload_storage_bytes")
    _SAVED.append(OWN.payload_storage_bytes)
    OWN.payload_storage_bytes = A.payload_storage_bytes


def uninstall():
    if _SAVED:
        OWN.payload_storage_bytes = _SAVED.pop()


def installed():
    return bool(_SAVED) and OWN.payload_storage_bytes is A.payload_storage_bytes
