"""Metric logging and validation callback."""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import torch
import torch.distributed as dist

from astrai.parallel import only_on_rank
from astrai.parallel.setup import get_current_device
from astrai.trainer.callbacks.base import CallbackFactory, TrainCallback
from astrai.trainer.callbacks.metric_util import (
    ctx_get_grad_norm,
    ctx_get_grad_snr,
    ctx_get_loss,
    ctx_get_lr,
    ctx_get_moe_metric,
    ctx_get_val_loss,
)

if TYPE_CHECKING:
    from astrai.trainer.train_context import TrainContext


@CallbackFactory.register("metric")
class MetricCallback(TrainCallback):
    def __init__(
        self,
        ckpt_dir: str,
        save_interval: int,
        metrics: List[str] = None,
        val_step: int = 0,
        grad_snr_interval: int = 1,
    ):
        if grad_snr_interval <= 0:
            raise ValueError("grad_snr_interval must be positive")
        self.last_log_flush_step = None
        self.save_interval = save_interval
        self.metrics = metrics or ["loss", "lr"]
        self.val_step = val_step
        self.grad_snr_interval = grad_snr_interval
        self._next_val_step = 0

        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else Path.cwd() / "checkpoint"
        # A run log, not checkpoint state: the checkpoint directory is published
        # atomically by its own writer, and a second writer inside it made a
        # re-published step drop the records of the steps before it.
        self.log_path = self.ckpt_dir / "metric.jsonl"

        self.log_cache = []

        self._metric_funcs = {
            "loss": ctx_get_loss,
            "lr": ctx_get_lr,
            "val_loss": ctx_get_val_loss,
            "grad_norm": ctx_get_grad_norm,
            "grad_snr": ctx_get_grad_snr,
            "moe_aux_loss": partial(ctx_get_moe_metric, key="aux_loss"),
            "router_entropy": partial(ctx_get_moe_metric, key="router_entropy"),
            "dead_expert_fraction": partial(
                ctx_get_moe_metric, key="dead_expert_fraction"
            ),
            "load_imbalance_mean": partial(
                ctx_get_moe_metric, key="load_imbalance_mean"
            ),
            "load_imbalance_max": partial(ctx_get_moe_metric, key="load_imbalance_max"),
        }

    def _metrics(self, context: TrainContext, names):
        metrics = dict(context.metrics)
        for name in names:
            metric_fn = self._metric_funcs.get(name)
            if metric_fn is None:
                continue
            value = metric_fn(context)
            if value is not None:
                metrics[name] = value
        selected = set(context.metrics) | set(names)
        selected.discard("*")
        result = {name: metrics[name] for name in selected if name in metrics}
        if result and context.dp_size > 1 and dist.is_initialized():
            metric_names = sorted(result)
            values = torch.tensor(
                [result[name] for name in metric_names],
                dtype=torch.float32,
                device=get_current_device(),
            )
            # dp-dimension average only: cp peers hold values derived from
            # the same batch, so summing them in would double-count.
            values = context.topology.reduce_mean(values)
            result.update(zip(metric_names, values.tolist()))
        return result

    @only_on_rank(0)
    def _append(self, event_type: str, context: TrainContext, **extra):
        entry = {
            "type": event_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epoch": context.epoch,
            "step": context.optimizer_step,
            "consumed_samples": context.consumed_samples,
            **extra,
        }
        self.log_cache.append(entry)

    def _run_validation(self, context: TrainContext) -> float:
        context.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in context.val_dataloader:
                # Online strategies evaluate a one-off rollout (leaving
                # the replay cache untouched) via the public hook; None
                # means offline — validate the batch directly.
                loss_output = context.strategy.validate_online(batch)
                if loss_output is None:
                    loss_output = context.strategy(batch)
                total_loss += loss_output["loss"].item()
                num_batches += 1

        # Sum (loss, batches) across dp replicas and take the ratio; the
        # reduce is a no-op single-process, where the clamp preserves the
        # local zero-batch behavior.
        stats = torch.tensor(
            [total_loss, float(num_batches)], device=get_current_device()
        )
        stats = context.topology.reduce_sum(stats)
        avg_loss = (stats[0] / stats[1].clamp(min=1.0)).item()

        context.model.train()
        return avg_loss

    def _run_rollout_validation(self, context: TrainContext) -> Dict[str, float]:
        """Validation for online strategies via :class:`RolloutEvaluator`.

        Reports reward statistics under the evaluator's own sampling
        params instead of the training RL loss (degenerate under greedy
        decode or group_size == 1).  The generator toggles the model into
        eval mode itself, so no global mode switch is needed here; the
        replay cache and its cadence stay untouched by construction.
        """
        totals: Dict[str, float] = {}
        num_batches = 0
        for batch in context.val_dataloader:
            metrics = context.val_evaluator.evaluate(batch)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            num_batches += 1
        if num_batches == 0:
            return {}
        return {name: total / num_batches for name, total in totals.items()}

    def on_train_begin(self, context: TrainContext):
        self.last_log_flush_step = context.optimizer_step
        # A resumed run replays the steps the crashed one logged after its last
        # checkpoint; drop those records so the log holds one pass per step.
        if context.checkpoint is not None:
            self._drop_records_after(context.optimizer_step)

    @only_on_rank(0)
    def _drop_records_after(self, step: int):
        if not self.log_path.exists():
            return
        kept = []
        for line in self.log_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(record.get("step", 0)) <= step:
                kept.append(line)
        self.log_path.write_text("".join(f"{line}\n" for line in kept))

    @only_on_rank(0)
    def _flush(self):
        if not self.log_cache:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.writelines(json.dumps(log) + "\n" for log in self.log_cache)
        self.log_cache.clear()

    def before_optimizer_step(self, context):
        # GradSNR keeps model-sized first-moment state. Only update it when
        # requested and at the configured sampling interval.
        if (
            "grad_snr" in self.metrics
            and context.grad_snr_tracker is not None
            and context.optimizer_step % self.grad_snr_interval == 0
        ):
            context.grad_snr_tracker.update(
                context.model, step_span=self.grad_snr_interval
            )
            context.grad_snr_value = context.grad_snr_tracker.snr

        if (
            context.val_dataloader is not None
            and self.val_step > 0
            and context.optimizer_step >= self._next_val_step
        ):
            if context.val_evaluator is not None:
                context.val_loss = None
                val_metrics = self._run_rollout_validation(context)
                self._next_val_step = context.optimizer_step + self.val_step
                self._append("validation", context, **val_metrics)
            else:
                context.val_loss = self._run_validation(context)
                self._next_val_step = context.optimizer_step + self.val_step
                self._append("validation", context, val_loss=context.val_loss)

        step_metrics = [m for m in self.metrics if m != "val_loss"]
        self._append("step", context, **self._metrics(context, step_metrics))

    def after_optimizer_step(self, context):
        if context.optimizer_step - self.last_log_flush_step >= self.save_interval:
            self._flush()
            self.last_log_flush_step = context.optimizer_step

    def on_epoch_end(self, context):
        self._append("epoch", context)

    def on_train_end(self, context):
        self._flush()
        self.last_log_flush_step = context.optimizer_step

    def on_stop_requested(self, context):
        self._flush()

    def on_failure(self, context):
        self._flush()
