"""Same verified V5 startup/IPC methods, with only a private binding namespace."""
from types import FunctionType
from rl_agent.splitfusion_live_gt_optimization_v5.client import Client as V5Client
from . import contract as C


def _bound(function):
    namespace = dict(function.__globals__, C=C)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class Client(V5Client):
    # _exchange's inherited __class__ cell is V5Client: super() therefore
    # calls V4.Client, exactly as in the sealed startup repair.
    __init__ = _bound(V5Client.__init__)
    _exchange = _bound(V5Client._exchange)
