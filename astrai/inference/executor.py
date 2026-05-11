import logging
from typing import List, Optional

import torch
from torch import Tensor

from astrai.inference.cache import PagedCache
from astrai.inference.sample import sample
from astrai.inference.task import STOP, Task, TaskStatus
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


class Executor:
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        page_cache: PagedCache,
        page_size: int = 64,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.page_cache = page_cache
        self.page_size = page_size
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

    def allocate_pages_for_activation(self, task: Task) -> bool:
        prompt_len = len(task.prompt_ids)
        hit_pages = self.page_cache.lookup_prefix(task.prompt_ids)
        cached_tokens = len(hit_pages) * self.page_size
        for p in hit_pages:
            self.page_cache.inc_ref(p)

        remaining = prompt_len - cached_tokens
        n_new = self._n_pages_for(remaining) if remaining > 0 else 0
        new_pages = self.page_cache.alloc_n(n_new) if n_new > 0 else []

        if remaining > 0 and not new_pages:
            for p in hit_pages:
                self.page_cache.free(p)
            return False

        task.page_table = hit_pages + new_pages
        task.n_pages = len(task.page_table)
        task._prefix_cached_tokens = cached_tokens
        return True

    def free_task_pages(self, task: Task) -> None:
        if task._pages_freed:
            return
        for idx in task.page_table:
            self.page_cache.free(idx)
        task.page_table.clear()
        task.n_pages = 0
        task._pages_freed = True

    def execute_prefill(
        self, tasks: List[Task], prompt_len: int, start_pos: int = 0
    ) -> None:
        if start_pos >= prompt_len:
            return

        tasks = sorted(tasks, key=lambda t: t.task_id)
        batch_sz = len(tasks)

        seq_len = prompt_len - start_pos
        input_ids = torch.empty(batch_sz, seq_len, dtype=torch.long, device=self.device)
        input_mask = torch.ones(
            batch_sz, prompt_len, dtype=torch.bool, device=self.device
        )

        for i, t in enumerate(tasks):
            input_ids[i] = torch.tensor(
                t.prompt_ids[start_pos:prompt_len], device=self.device
            )

        page_tables = self._make_page_table_tensor(tasks)

        with torch.inference_mode():
            self.model(
                input_ids,
                input_mask=input_mask,
                start_pos=start_pos,
                paged_cache=self.page_cache.bind(page_tables, total_len=prompt_len),
            )

        start_logical_page = start_pos // self.page_size
        for t in tasks:
            self._record_page_hashes(t, start_logical_page=start_logical_page)

    def execute_decode(self, tasks: List[Task], start_pos: int) -> None:
        if not tasks:
            return

        tasks = sorted(tasks, key=lambda t: t.task_id)

        valid: List[Task] = []
        for t in tasks:
            if self._maybe_alloc_page(t, start_pos):
                valid.append(t)
            else:
                t.status = TaskStatus.ABORTED
                if t.stream_callback:
                    t.stream_callback(STOP)

        if not valid:
            return

        tasks = valid
        batch_sz = len(tasks)

        input_ids = torch.tensor(
            [t.output_ids[-1] if t.output_ids else t.prompt_ids[-1] for t in tasks],
            dtype=torch.long,
            device=self.device,
        )

        active_mask = torch.ones((batch_sz, 1), dtype=torch.bool, device=self.device)

        page_tables = self._make_page_table_tensor(tasks)
        total_len = start_pos + 1

        temperatures = torch.tensor([t.temperature for t in tasks], device=self.device)
        top_ks = torch.tensor([t.top_k for t in tasks], device=self.device)
        top_ps = torch.tensor([t.top_p for t in tasks], device=self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids.unsqueeze(1),
                input_mask=active_mask,
                paged_cache=self.page_cache.bind(page_tables, total_len=total_len),
                start_pos=start_pos,
            )
            logits = outputs["logits"][:, -1, :]

        next_tokens = sample(
            logits,
            temperature=temperatures,
            top_k=top_ks,
            top_p=top_ps,
        ).tolist()

        for t, ntok in zip(tasks, next_tokens):
            t.output_ids.append(ntok)
            t.output_tokens += 1
            pos = t.input_tokens + t.output_tokens
            self._maybe_alloc_page(t, pos)
            if t.stream_callback:
                t.stream_callback(self.tokenizer.decode([ntok]))

        for t in tasks:
            if t.is_finished(self.tokenizer.stop_ids):
                if t.stream_callback:
                    t.stream_callback(STOP)

    def _n_pages_for(self, n_tokens: int) -> int:
        return (n_tokens + self.page_size - 1) // self.page_size

    def _make_page_table_tensor(self, tasks: List[Task]) -> Tensor:
        max_pages = max(t.n_pages for t in tasks)
        rows = [t.page_table + [-1] * (max_pages - t.n_pages) for t in tasks]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _record_page_hashes(self, task: Task, start_logical_page: int = 0) -> None:
        full_pages = len(task.prompt_ids) // self.page_size
        for i in range(start_logical_page, full_pages):
            self.page_cache.record_page(task.page_table[i], task.prompt_ids, i)

    def _maybe_alloc_page(self, task: Task, pos: int) -> bool:
        needed = self._n_pages_for(pos + 1)
        while task.n_pages < needed:
            p = self.page_cache.alloc()
            if p < 0:
                return False
            task.page_table.append(p)
            task.n_pages += 1
        return True
