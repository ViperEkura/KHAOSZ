"""Checkpoint save/resume state coherence.

A checkpoint must describe one training state: the saved weights already include
the optimizer step of the saving batch, so the saved cursor must include it too.
"""

import json

import pytest
import torch

from astrai.parallel.topology import ParallelTopology
from astrai.serialization import Checkpoint
from astrai.trainer.callbacks import TrainCallback
from astrai.trainer.components import build_dataloader
from astrai.trainer.trainer import Trainer
from tests.support.rollout import RandomTokenDataset


class _StopAfterSteps(TrainCallback):
    """Request a stop once N optimizer steps have been committed."""

    def __init__(self, steps: int):
        self.steps = steps

    def after_optimizer_step(self, context):
        if context.optimizer_step >= self.steps:
            context.request_stop()


class _StopAfterBatches(TrainCallback):
    """Request a stop after N data batches, i.e. inside an accumulation window."""

    def __init__(self, batches: int):
        self.batches = batches
        self.seen = 0
        self.synced_at_request = None

    def on_batch_end(self, context):
        self.seen += 1
        if self.seen == self.batches:
            # A batch that ends an accumulation window commits; any other one
            # leaves the cursor where the last commit put it.
            self.synced_at_request = context.executor.sync_gradients
            context.request_stop()


class _StopAfterFirstEpoch(TrainCallback):
    def on_epoch_end(self, context):
        if context.epoch == 0:
            context.request_stop()


def _checkpoints(directory):
    paths = list(directory.glob("epoch_*_step_*"))
    return sorted(
        paths,
        key=lambda p: (
            int(p.name.split("_")[1]),
            int(p.name.rsplit("_", 1)[1]),
        ),
    )


def _final_state(directory):
    paths = _checkpoints(directory)
    assert paths, f"no checkpoint written under {directory}"
    return Checkpoint.load(str(paths[-1])).state_dict


def _train(config, callbacks=None, param_path=None, resume=False):
    Trainer(config, callbacks=callbacks).train(param_path=param_path, resume=resume)


def test_saved_cursor_matches_committed_steps(
    train_config_factory, trainer_dataset, tmp_path
):
    """Every checkpoint must report the step its weights actually belong to."""
    config = train_config_factory(
        trainer_dataset,
        str(tmp_path),
        device="cpu",
        n_epoch=1,
        batch_per_device=2,
        grad_accum_steps=2,
        ckpt_interval=1,
    )
    _train(config)

    per_step = config.batch_per_device * config.grad_accum_steps
    paths = _checkpoints(tmp_path)
    assert len(paths) >= 2, "expected at least two interval checkpoints"

    for path in paths:
        meta = json.loads((path / "meta.json").read_text())
        step = int(path.name.rsplit("_", 1)[1])
        assert meta["optimizer_step"] == step, (
            f"{path.name}: meta reports step {meta['optimizer_step']}"
        )
        assert meta["consumed_samples"] == step * per_step, (
            f"{path.name}: cursor {meta['consumed_samples']} does not match "
            f"{step} committed steps x {per_step} samples"
        )


@pytest.mark.parametrize(
    "stop_callback, expected_step",
    [
        pytest.param(_StopAfterSteps(3), 3, id="step-boundary"),
        pytest.param(_StopAfterBatches(5), 3, id="mid-window"),
    ],
)
def test_resume_reproduces_the_uninterrupted_run(
    train_config_factory, tmp_path, stop_callback, expected_step
):
    """Interrupting, saving and resuming must be bitwise equivalent to running on."""
    baseline_dir = tmp_path / "baseline"
    split_dir = tmp_path / "split"
    per_step_total = 2 * 2  # batch_per_device * grad_accum_steps, dp_size == 1

    def config_for(directory):
        return train_config_factory(
            RandomTokenDataset(length=64, max_length=8, vocab_size=100),
            str(directory),
            device="cpu",
            n_epoch=2,
            batch_per_device=2,
            grad_accum_steps=2,
            # Only stop/end checkpoints, so the split leg resumes from a known one.
            ckpt_interval=1000,
            random_seed=1234,
        )

    _train(config_for(baseline_dir))

    _train(config_for(split_dir), callbacks=[stop_callback])
    first_leg = _checkpoints(split_dir)
    assert len(first_leg) == 1, (
        f"expected a single emergency checkpoint, got {first_leg}"
    )
    checkpoint = first_leg[0]
    step = int(checkpoint.name.rsplit("_", 1)[1])
    assert step == expected_step
    meta = json.loads((checkpoint / "meta.json").read_text())
    assert meta["consumed_samples"] == step * per_step_total, (
        "saved cursor must cover exactly the committed steps"
    )
    mid_window = getattr(stop_callback, "synced_at_request", None)
    if mid_window is not None:
        assert mid_window is False, "the stop must land mid-window"

    _train(config_for(split_dir), param_path=str(checkpoint), resume=True)

    baseline = _final_state(baseline_dir)
    resumed = _final_state(split_dir)
    assert set(baseline) == set(resumed)
    differing = [
        key for key in baseline if not torch.equal(baseline[key], resumed[key])
    ]
    assert not differing, f"resume diverged from the uninterrupted run: {differing[:5]}"


def test_loader_iterator_leaves_global_rng_untouched(train_config_factory, tmp_path):
    """Creating a loader iterator must not shift the model's RNG stream."""
    dataset = RandomTokenDataset(length=32, max_length=8, vocab_size=100)
    config = train_config_factory(
        dataset,
        str(tmp_path),
        device="cpu",
        num_workers=2,
    )
    loader = build_dataloader(
        dataset, config.dataset, config, ParallelTopology(1), epoch=0, start_iter=0
    )
    assert loader.generator is not None

    torch.manual_seed(7)
    before = torch.get_rng_state()
    iterator = iter(loader)
    next(iterator)
    del iterator
    assert torch.equal(torch.get_rng_state(), before), (
        "creating a loader iterator consumed the global RNG"
    )


@pytest.mark.parametrize("length", [9, 10])
def test_short_epoch_tail_is_discarded_and_resumes_at_next_epoch(
    train_config_factory, tmp_path, length
):
    """An incomplete accumulation window is skipped at the epoch boundary."""
    baseline_dir = tmp_path / "baseline_tail"
    split_dir = tmp_path / "split_tail"

    def config_for(directory):
        return train_config_factory(
            RandomTokenDataset(length=length, max_length=8, vocab_size=100),
            str(directory),
            device="cpu",
            n_epoch=2,
            batch_per_device=2,
            grad_accum_steps=2,
            ckpt_interval=1000,
            random_seed=1234,
        )

    _train(config_for(baseline_dir))
    _train(config_for(split_dir), callbacks=[_StopAfterFirstEpoch()])
    first = _checkpoints(split_dir)[-1]
    checkpoint = Checkpoint.load(str(first))
    assert checkpoint.meta["optimizer_step"] == 2
    assert checkpoint.consumed_samples == length
    assert first.name == "epoch_1_step_2"

    _train(config_for(split_dir), param_path=str(first), resume=True)
    baseline = Checkpoint.load(str(_checkpoints(baseline_dir)[-1]))
    resumed = Checkpoint.load(str(_checkpoints(split_dir)[-1]))
    assert baseline.meta["optimizer_step"] == resumed.meta["optimizer_step"] == 4
    assert baseline.consumed_samples == resumed.consumed_samples == 2 * length
    for key in baseline.state_dict:
        torch.testing.assert_close(
            baseline.state_dict[key], resumed.state_dict[key], rtol=0, atol=0
        )
    assert torch.equal(
        baseline.extra["rng_state"]["torch"], resumed.extra["rng_state"]["torch"]
    )


def test_discarded_tail_is_not_loaded(train_config_factory, tmp_path):
    dataset = RandomTokenDataset(length=9, max_length=8, vocab_size=100, stop_after=9)
    config = train_config_factory(
        dataset,
        str(tmp_path),
        device="cpu",
        n_epoch=1,
        batch_per_device=2,
        grad_accum_steps=2,
        ckpt_interval=1000,
    )
    _train(config)
    checkpoint = Checkpoint.load(str(_checkpoints(tmp_path)[-1]))
    assert checkpoint.meta["optimizer_step"] == 2
    assert checkpoint.consumed_samples == 9


def test_resume_from_checkpoint_before_discarded_tail(train_config_factory, tmp_path):
    baseline_dir = tmp_path / "baseline"
    stopped_dir = tmp_path / "stopped"
    resumed_dir = tmp_path / "resumed"

    def config_for(directory):
        return train_config_factory(
            RandomTokenDataset(length=9, max_length=8, vocab_size=100),
            str(directory),
            device="cpu",
            n_epoch=2,
            batch_per_device=2,
            grad_accum_steps=2,
            ckpt_interval=1,
            random_seed=1234,
        )

    _train(config_for(baseline_dir))
    _train(config_for(stopped_dir), callbacks=[_StopAfterFirstEpoch()])
    before_tail = stopped_dir / "epoch_0_step_2"
    assert before_tail.is_dir()
    assert Checkpoint.load(str(before_tail)).consumed_samples == 8

    _train(config_for(resumed_dir), param_path=str(before_tail), resume=True)
    baseline = Checkpoint.load(str(_checkpoints(baseline_dir)[-1]))
    resumed = Checkpoint.load(str(_checkpoints(resumed_dir)[-1]))
    assert baseline.meta["optimizer_step"] == resumed.meta["optimizer_step"] == 4
    assert baseline.consumed_samples == resumed.consumed_samples == 18
    for key in baseline.state_dict:
        torch.testing.assert_close(
            baseline.state_dict[key], resumed.state_dict[key], rtol=0, atol=0
        )


def test_negative_cursor_is_refused(train_config_factory, tmp_path):
    """A negative sample cursor cannot describe a resumable state."""
    directory = tmp_path / "run"
    config = train_config_factory(
        RandomTokenDataset(length=64, max_length=8, vocab_size=100),
        str(directory),
        device="cpu",
        n_epoch=2,
        batch_per_device=2,
        grad_accum_steps=2,
        ckpt_interval=1000,
        random_seed=1234,
    )
    _train(config)
    checkpoint = _checkpoints(directory)[-1]
    meta_path = checkpoint / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["consumed_samples"] = -1
    meta_path.write_text(json.dumps(meta))

    # Keep the manifest file-size table consistent with the edited metadata.
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["meta.json"]["size"] = len(meta_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="non-negative sample count"):
        _train(config, param_path=str(checkpoint), resume=True)
