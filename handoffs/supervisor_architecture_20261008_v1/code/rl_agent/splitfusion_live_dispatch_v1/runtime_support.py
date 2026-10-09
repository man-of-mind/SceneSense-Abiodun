"""Shared startup-only module preparation and observable operation counters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class OperationSnapshot:
    startup_model_load_operations: int
    startup_model_construction_operations: int
    startup_preloaded_objects: int
    startup_module_device_moves: int
    startup_eval_transitions: int
    startup_parameter_freezes: int
    hot_path_model_load_operations: int
    hot_path_model_construction_operations: int
    frames_attempted: int
    frames_completed: int
    ranker_dispatches: int
    ae_encoder_dispatches: int
    ae_decoder_dispatches: int
    tail_dispatches: int


class OperationCounters:
    def __init__(
        self,
        *,
        startup_model_load_operations: int,
        startup_model_construction_operations: int,
        startup_preloaded_objects: int,
    ) -> None:
        for name, value in (
            ("startup_model_load_operations", startup_model_load_operations),
            ("startup_model_construction_operations", startup_model_construction_operations),
            ("startup_preloaded_objects", startup_preloaded_objects),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.startup_model_load_operations = startup_model_load_operations
        self.startup_model_construction_operations = startup_model_construction_operations
        self.startup_preloaded_objects = startup_preloaded_objects
        self.startup_module_device_moves = 0
        self.startup_eval_transitions = 0
        self.startup_parameter_freezes = 0
        self.frames_attempted = 0
        self.frames_completed = 0
        self.ranker_dispatches = 0
        self.ae_encoder_dispatches = 0
        self.ae_decoder_dispatches = 0
        self.tail_dispatches = 0

    def snapshot(self) -> OperationSnapshot:
        return OperationSnapshot(
            startup_model_load_operations=self.startup_model_load_operations,
            startup_model_construction_operations=self.startup_model_construction_operations,
            startup_preloaded_objects=self.startup_preloaded_objects,
            startup_module_device_moves=self.startup_module_device_moves,
            startup_eval_transitions=self.startup_eval_transitions,
            startup_parameter_freezes=self.startup_parameter_freezes,
            hot_path_model_load_operations=0,
            hot_path_model_construction_operations=0,
            frames_attempted=self.frames_attempted,
            frames_completed=self.frames_completed,
            ranker_dispatches=self.ranker_dispatches,
            ae_encoder_dispatches=self.ae_encoder_dispatches,
            ae_decoder_dispatches=self.ae_decoder_dispatches,
            tail_dispatches=self.tail_dispatches,
        )


def prepare_preloaded_module(
    module: Any,
    device: torch.device,
    counters: OperationCounters,
) -> Any:
    """Move/freeze an existing object once during runtime construction."""
    if hasattr(module, "to"):
        moved = module.to(device)
        if moved is not None and moved is not module:
            raise ValueError("preloaded module .to() replaced the supplied object")
        counters.startup_module_device_moves += 1
    if hasattr(module, "eval"):
        evaluated = module.eval()
        if evaluated is not None and evaluated is not module:
            raise ValueError("preloaded module .eval() replaced the supplied object")
        counters.startup_eval_transitions += 1
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        values: Iterable[Any] = parameters()
        for parameter in values:
            if hasattr(parameter, "requires_grad_"):
                parameter.requires_grad_(False)
                counters.startup_parameter_freezes += 1
    return module
