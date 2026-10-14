"""In-process request and event boundary for the inference frontend.

The client delegates request and lifecycle operations to EngineCore.
Both remain in the training model's process; request submission and
cancellation share the core's operation lock with model execution.
"""

from typing import List

from astrai.extension.runtime.context import OpSelection
from astrai.inference.core.engine_core import EngineCore
from astrai.inference.core.scheduler import Scheduler


class EngineCoreClient:
    """Interface between the frontend and the engine core."""

    def set_event_sink(self, sink) -> None:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    @property
    def max_batch_size(self) -> int:
        raise NotImplementedError

    @property
    def op_selection(self) -> OpSelection:
        raise NotImplementedError

    @property
    def cuda_graph_enabled(self) -> bool:
        raise NotImplementedError

    def score_ids(self, prompt_ids_list, continuation_ids_list, per_token=False):
        raise NotImplementedError

    def send_request(self, **kwargs) -> str:
        """Submit one generation request; returns its request id."""
        raise NotImplementedError

    def send_requests(self, prompts: List[str], **kwargs) -> List[str]:
        """Submit a batch; returns the request ids in order."""
        raise NotImplementedError

    def abort_request(self, request_id: str) -> bool:
        """Cancel one request; False when it was already gone."""
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """Direct client of an in-process EngineCore."""

    def __init__(self, core: EngineCore):
        self._core = core

    @property
    def scheduler(self) -> Scheduler:
        return self._core.scheduler

    def set_event_sink(self, sink) -> None:
        self._core.set_event_sink(sink)

    def start(self) -> None:
        self._core.start()

    @property
    def max_batch_size(self) -> int:
        return self._core.max_batch_size

    @property
    def op_selection(self) -> OpSelection:
        return self._core.op_selection

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._core.cuda_graph_enabled

    def score_ids(self, prompt_ids_list, continuation_ids_list, per_token=False):
        return self._core.score_ids(
            prompt_ids_list, continuation_ids_list, per_token=per_token
        )

    def send_request(self, **kwargs) -> str:
        return self._core.add_request(**kwargs)

    def send_requests(self, prompts: List[str], **kwargs) -> List[str]:
        return self._core.add_requests(prompts, **kwargs)

    def abort_request(self, request_id: str) -> bool:
        return self._core.cancel_request(request_id)

    def stats(self) -> dict:
        return self._core.get_stats()

    def shutdown(self) -> None:
        self._core.stop()
