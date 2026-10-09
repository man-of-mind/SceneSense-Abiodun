"""Policy-neutral single-host CPU/GPU contention profiler (instrumentation only).

Nothing in this package selects an action, changes a reward, a timeout or a
terminal, or alters precision, scheduling or rendering.  The policy, its state
schema and its checkpoint are *recorded* from the loaded binding, never
assumed.  Importing any module here starts nothing and does not initialize
CUDA.
"""
