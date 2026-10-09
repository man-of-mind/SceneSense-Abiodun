"""Process-local, reversible switch: the GT prep composition computes AVO with the tilt-aware candidate.

One explicit substitution, restored by ``uninstall()`` (default OFF), the producer-v2 ``worker_switch`` pattern:
``splitfusion_numeric_gt_producer_v1.avo.numeric_avo`` -> ``quality.numeric_avo``.  The only production caller is
``splitfusion_numeric_gt_integration_v2.composition.assemble`` (``NA.numeric_avo(...)``, resolved at call time), run
by ``DeferredProducerEngine.prepare_unbound`` in the GT prep worker.  Any live use must register this substitution
and the candidate sources; this module does not do that.
"""
from rl_agent.splitfusion_numeric_gt_producer_v1 import avo as NA

from . import quality as Q

ORIGINAL_NUMERIC_AVO = NA.numeric_avo
_SAVED = []


def install():
    if _SAVED:
        raise RuntimeError("tilt-aware AVO switch already installed")
    if NA.numeric_avo is not ORIGINAL_NUMERIC_AVO:
        raise RuntimeError("producer AVO step is not the original numeric_avo")
    _SAVED.append(NA.numeric_avo)
    NA.numeric_avo = Q.numeric_avo


def uninstall():
    if _SAVED:
        NA.numeric_avo = _SAVED.pop()


def installed():
    return bool(_SAVED) and NA.numeric_avo is Q.numeric_avo
