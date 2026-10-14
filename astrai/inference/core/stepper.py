"""Scheduling policy and compatibility entry points; execution lives in EngineCore."""

from typing import TYPE_CHECKING, List, Optional

from astrai.inference.contracts import ExecutionRequest
from astrai.inference.core.events import FINISH_ABORTED, FINISH_LENGTH
from astrai.inference.core.request import Request, RequestStatus

if TYPE_CHECKING:
    from astrai.inference.core.scheduler import Scheduler


class SchedulerStep:
    """Plan prefill windows and decode slots, without calling a model.

    The existing per-prefill-group budget is deliberately preserved here.
    A global token budget / mixed forward is a separate scheduling change.
    """

    def __init__(self, scheduler: "Scheduler", token_budget: Optional[int] = None):
        if token_budget is not None and token_budget <= 0:
            raise ValueError("token_budget must be positive or None")
        self.scheduler = scheduler
        self._token_budget = token_budget

    def plan(self, requests: List[Request]) -> List[ExecutionRequest]:
        scheduler = self.scheduler
        kv = scheduler._kv_manager
        entries = []
        groups = {}
        decode = []
        for request in requests:
            if request.terminal_emitted:
                continue
            request.input_tokens = len(request.prompt_ids)
            if request.is_finished(scheduler.stop_ids):
                scheduler.finish(request, scheduler.finish_reason(request))
                continue
            if not request.prefill_complete:
                start = min(
                    max(
                        request.num_computed_tokens,
                        kv.cached_tokens(request.request_id),
                    ),
                    len(request.prompt_ids) - 1,
                )
                groups.setdefault(start, []).append(request)
            elif request.next_pos >= scheduler.max_seq_len:
                scheduler.finish(request, FINISH_LENGTH)
            else:
                decode.append(request)

        for start, group in groups.items():
            remaining = self._token_budget
            for request in group:
                need = len(request.prompt_ids) - start
                count = need if remaining is None else min(need, remaining)
                if count <= 0:
                    break
                entries.append(request.execution("prefill", start, count))
                if remaining is not None:
                    remaining -= count

        if decode:
            extended = kv.extend_slots_batch(
                [r.request_id for r in decode], [r.next_pos for r in decode]
            )
            for request, ok in zip(decode, extended):
                if ok:
                    entries.append(request.execution("decode", request.next_pos, 1))
                else:
                    scheduler.finish(
                        request,
                        FINISH_ABORTED,
                        error_reason="kv_cache_extension_failed",
                    )
        return entries

    def step(self, requests: List[Request], return_logprobs: bool = False):
        """Compatibility shell used by synchronous rollout and focused tests."""
        self.scheduler.engine_core.execute_synchronously(requests, return_logprobs)
        return (
            [r for r in requests if r.status != RequestStatus.ABORTED],
            [r for r in requests if r.status == RequestStatus.ABORTED],
        )
