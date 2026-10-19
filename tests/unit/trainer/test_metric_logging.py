import json
from types import SimpleNamespace

from astrai.trainer.callbacks import MetricCallback


def _callback(tmp_path, flush_interval=2, checkpoint_interval=3, val_step=0):
    return MetricCallback(
        ckpt_dir=tmp_path,
        flush_interval=flush_interval,
        checkpoint_interval=checkpoint_interval,
        metrics=["loss"],
        val_step=val_step,
    )


def _context(step=0, samples=0, checkpoint=None):
    return SimpleNamespace(
        checkpoint=checkpoint,
        optimizer_step=step,
        consumed_samples=samples,
        epoch=0,
        metrics={},
        loss=1.0,
        val_dataloader=None,
        grad_snr_tracker=None,
        dp_size=1,
    )


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_resume_trims_logs_by_step_and_data_cursor(tmp_path):
    path = tmp_path / "metric.jsonl"
    records = [
        {"type": "step", "step": 4, "consumed_samples": 8},
        {"type": "step", "step": 5, "consumed_samples": 10},
        {"type": "epoch", "step": 5, "consumed_samples": 12},
        {"type": "step", "step": 6, "consumed_samples": 14},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    callback = _callback(tmp_path)
    callback.on_train_begin(_context(step=5, samples=10, checkpoint=object()))

    assert _records(path) == records[:2]


def test_failed_optimizer_step_does_not_flush_pending_metrics(tmp_path):
    callback = _callback(tmp_path, flush_interval=1)
    context = _context(step=1, samples=2)
    callback.on_train_begin(_context())

    callback.before_optimizer_step(context)
    callback.on_shutdown(context)

    assert not (tmp_path / "metric.jsonl").exists()


def test_metrics_flush_on_interval_and_before_checkpoint(tmp_path):
    callback = _callback(tmp_path)
    context = _context()
    callback.on_train_begin(context)
    path = tmp_path / "metric.jsonl"

    for step in range(1, 4):
        context.optimizer_step = step
        context.consumed_samples = step * 2
        callback.before_optimizer_step(context)
        assert len(callback.pending_records) == 1
        callback.after_optimizer_step(context)
        if step == 1:
            assert not path.exists()
        else:
            assert [record["step"] for record in _records(path)] == list(
                range(1, step + 1)
            )


def test_validation_cadence_uses_global_step_after_resume(tmp_path):
    callback = _callback(tmp_path, val_step=10)
    callback.on_train_begin(_context(step=11, samples=22, checkpoint=object()))

    assert callback._next_val_step == 21
