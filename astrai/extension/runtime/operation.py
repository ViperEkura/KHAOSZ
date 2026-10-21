"""Shared call contract; every operation implements its own selection rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch

from astrai.extension.runtime.context import op_context


@dataclass(frozen=True)
class OpSpec(ABC):
    """Immutable metadata needed to decide one forward call."""

    device: torch.device
    dtype: torch.dtype
    phase: str
    needs_backward: bool

    @property
    @abstractmethod
    def operation(self) -> str:
        """The operation whose call this specification describes."""


class ExplicitSelectionError(RuntimeError):
    """A selected implementation cannot handle this call."""


SpecT = TypeVar("SpecT", bound=OpSpec)


class OpDispatcher(Generic[SpecT], ABC):
    """Read a choice once, then delegate capability rules to one operation."""

    @property
    @abstractmethod
    def operation(self) -> str:
        """Context key for this operation."""

    @abstractmethod
    def build_spec(self, *args: Any, **kwargs: Any) -> SpecT:
        """Capture call metadata without retaining tensors in the spec."""

    @abstractmethod
    def select(
        self, spec: SpecT, choice: Any, origin: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Choose one implementation using operation-specific rules."""

    def dispatch(self, *args: Any, explicit: Any = None, **kwargs: Any) -> Any:
        spec = self.build_spec(*args, **kwargs)
        choice, origin = op_context.snapshot().choice(self.operation)
        if explicit is not None:
            choice, origin = explicit, "explicit"
        if isinstance(choice, str) and choice == "auto":
            choice, origin = None, "chain"
        return self.select(spec, choice, origin, *args, **kwargs)
