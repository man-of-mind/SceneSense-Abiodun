"""Private namespaces for retained code; never edit retained module globals."""
from contextlib import contextmanager
import inspect
from types import FunctionType


def function(original, namespace):
    result = FunctionType(original.__code__, namespace, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result


def namespace(module, **overrides):
    result = dict(vars(module), __package__=__package__, **overrides)
    result["__spec__"] = None
    for name, value in vars(module).items():
        if name in overrides:
            continue
        if isinstance(value, FunctionType) and value.__globals__ is vars(module):
            result[name] = function(value, result)
        elif (isinstance(value, FunctionType) and hasattr(value, "__wrapped__")
              and value.__wrapped__.__globals__ is vars(module)
              and inspect.isgeneratorfunction(value.__wrapped__)):
            result[name] = contextmanager(function(value.__wrapped__, result))
    return result


def specialize(original, target, replacements):
    source = inspect.getsource(original)
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("retained adapter seam changed: " + old)
        source = source.replace(old, new)
    exec(compile(source, "<registered-v9-adapter:" + original.__name__ + ">", "exec"), target)
    return target[original.__name__]
