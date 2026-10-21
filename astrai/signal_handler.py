import logging
import os
import signal
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_early_stop = threading.Event()
_active_context = None
_previous_handlers = None
_parent_notified = False


@contextmanager
def ignore_worker_deaths():
    """Ignore SIGCHLD, so dying DataLoader workers cannot abort a shutdown save.

    PyTorch turns a dead worker into a ``RuntimeError`` from its SIGCHLD
    handler, which can land in the middle of the emergency save. During
    shutdown the workers are gone and nobody reads from them any more.
    """
    if not hasattr(signal, "SIGCHLD"):
        yield
        return
    try:
        previous = signal.getsignal(signal.SIGCHLD)
        signal.signal(signal.SIGCHLD, lambda signum, frame: None)
    except ValueError:  # not the main thread: signals cannot be replaced
        yield
        return
    try:
        yield
    finally:
        if callable(previous) or previous in (signal.SIG_DFL, signal.SIG_IGN):
            signal.signal(signal.SIGCHLD, previous)


def _notify_parent(signum: int) -> None:
    """Forward a rank's signal to the launcher so every rank stops together."""
    global _parent_notified
    if _parent_notified:
        return
    _parent_notified = True
    try:
        os.kill(os.getppid(), signum)
    except OSError:
        pass


def _early_handler(signum: int, frame):
    sig = signal.Signals(signum)
    logger.warning(
        "Received %s (pid=%d), requesting graceful training stop...",
        sig.name,
        os.getpid(),
    )
    _early_stop.set()
    if _active_context is not None:
        _active_context.request_stop()
    if os.environ.get("ASTRAI_RANK_CHILD"):
        _notify_parent(signum)


def install_early_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _early_handler)
    _unblock_signals()


def _unblock_signals():
    try:
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        blocked = {signal.SIGTERM, signal.SIGINT} & mask
        if blocked:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, blocked)
    except (AttributeError, OSError):
        pass


def register_signal_handlers(context):
    global _active_context, _previous_handlers
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    _active_context = context
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _early_handler)
    except BaseException:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _active_context = None
        raise
    _previous_handlers = previous
    if _early_stop.is_set():
        context.request_stop()
        logger.warning("Signal was received during initialization, stopping...")


def unregister_signal_handlers():
    global _active_context, _previous_handlers, _parent_notified
    try:
        if _previous_handlers is not None:
            for sig, handler in _previous_handlers.items():
                signal.signal(sig, handler)
    finally:
        _previous_handlers = None
        _active_context = None
        _parent_notified = False
        _early_stop.clear()
