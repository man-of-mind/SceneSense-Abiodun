"""Process-local, reversible switch: make the exact-GT prep engine construct and own ``NumericSourceProducerV2``.

Two explicit substitutions, both restored by ``uninstall()`` (default OFF):
* ``splitfusion_exact_gt_service_proto_v2.deferred.NS`` -> a proxy module exposing V2 as ``NumericSourceProducer``;
* ``splitfusion_numeric_gt_integration_v2.ownership.NumericSourceProducer`` -> V2. The owner keeps its EXACT type
  check (``type(producer) is ...``): with the switch on it accepts only V2 and refuses v1 or any other class.
Any live use must register these two substitutions and the V2 package sources; this module does not do that.
"""
import types

from rl_agent.splitfusion_exact_gt_service_proto_v2 import deferred as D
from rl_agent.splitfusion_numeric_gt_integration_v2 import ownership as OW
from rl_agent.splitfusion_numeric_gt_producer_v1 import source as NS

from .source_v2 import NumericSourceProducerV2

_SAVED = []


def install():
    if _SAVED:
        raise RuntimeError("producer switch already installed")
    if D.NS is not NS or OW.NumericSourceProducer is not NS.NumericSourceProducer:
        raise RuntimeError("engine/owner do not reference the v1 producer")
    proxy = types.ModuleType("numeric_producer_v2_proxy")
    proxy.__dict__.update({k: v for k, v in vars(NS).items() if not k.startswith("__")})
    proxy.NumericSourceProducer = NumericSourceProducerV2
    _SAVED.append((D.NS, OW.NumericSourceProducer))
    D.NS, OW.NumericSourceProducer = proxy, NumericSourceProducerV2


def uninstall():
    if _SAVED:
        D.NS, OW.NumericSourceProducer = _SAVED.pop()


def installed():
    return bool(_SAVED)
