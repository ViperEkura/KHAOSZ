"""Shared callback protocol and registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astrai.factory import BaseFactory

if TYPE_CHECKING:
    from astrai.trainer.train_context import TrainContext


@runtime_checkable
class TrainCallback(Protocol):
    """
    Callback interface for trainer.
    """

    def on_train_begin(self, context: TrainContext):
        """Called at the beginning of training."""

    def on_train_end(self, context: TrainContext):
        """Called at the end of training."""

    def on_epoch_begin(self, context: TrainContext):
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, context: TrainContext):
        """Called at the end of each epoch."""

    def on_batch_begin(self, context: TrainContext):
        """Called at the beginning of each batch."""

    def on_batch_end(self, context: TrainContext):
        """Called at the end of each batch."""

    def before_optimizer_step(self, context: TrainContext):
        """Called immediately before every optimizer step (sync step only)."""

    def after_optimizer_step(self, context: TrainContext):
        """Called after the optimizer and scheduler step (sync step only)."""

    def on_stop_requested(self, context: TrainContext):
        """Called when a graceful stop is agreed; the run is ending normally."""

    def on_failure(self, context: TrainContext):
        """Called when training raises; the context may be partially advanced."""


class CallbackFactory(BaseFactory[TrainCallback]):
    """Factory for registering and creating training callbacks.

    Example:
        @CallbackFactory.register("my_callback")
        class MyCallback(TrainCallback):
            ...

        callback = CallbackFactory.create("my_callback", **kwargs)
    """
