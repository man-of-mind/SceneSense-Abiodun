"""Instance-local dependency binding of retained functions; no module mutation.

Code objects/closures/defaults are retained. Explicit global dependencies live
in a private dictionary. This is not an import hook or a process-global seam.
Construction-only source substitutions use the retained capacity convention:
each exact clause must occur once; the resulting source is recorded.
"""
import hashlib
import inspect
import textwrap
import types

from .identity import require


def bind(function, **dependencies):
    free = function.__code__.co_freevars
    require(set(dependencies) <= set(function.__globals__) | set(free), 'unknown dependency')
    cells = None if function.__closure__ is None else tuple(
        types.CellType(dependencies[name]) if name in dependencies else cell
        for name, cell in zip(free, function.__closure__))
    result = types.FunctionType(function.__code__, dict(function.__globals__, **dependencies),
                               function.__name__, function.__defaults__, cells)
    result.__kwdefaults__ = function.__kwdefaults__
    result.__annotations__ = function.__annotations__
    result.__m4_original__ = function
    return result


def construct(function, substitutions, **dependencies):
    require(not function.__code__.co_freevars, 'constructor factory has free variables')
    original = textwrap.dedent(inspect.getsource(function))
    source = original
    for old, new in substitutions:
        require(source.count(old) == 1, 'construction clause drift: ' + old)
        source = source.replace(old, new)
    namespace = dict(function.__globals__, **dependencies)
    exec(compile(source, '<m4-construction:' + function.__qualname__ + '>', 'exec',
                 flags=function.__code__.co_flags & 0x1000000, dont_inherit=True), namespace)
    result = namespace[function.__name__]
    result.__m4_binding__ = dict(retained=function.__module__ + '.' + function.__qualname__,
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        constructed_sha256=hashlib.sha256(source.encode()).hexdigest(), substitutions=substitutions,
        dependencies=sorted(dependencies))
    return result
