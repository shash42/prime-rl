import asyncio
from types import SimpleNamespace

from prime_rl.transport.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample


def test_training_batch_mailbox_is_removed_after_decode(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_default"
    sender = FileSystemTrainingBatchSender(run_dir)
    batch = TrainingBatch(
        examples=[
            TrainingSample(
                token_ids=[1],
                mask=[True],
                logprobs=[-0.1],
                temperatures=[0.7],
                env_name="test",
            )
        ],
        step=1,
    )
    asyncio.run(sender.send(batch))
    manager = SimpleNamespace(
        used_idxs=[0],
        ready_to_update=[False],
        progress=[SimpleNamespace(step=1)],
        get_run_dir=lambda _: run_dir,
    )
    monkeypatch.setattr(
        "prime_rl.transport.filesystem.get_multi_run_manager", lambda: manager
    )
    receiver = FileSystemTrainingBatchReceiver()
    path = receiver._get_batch_path(0)

    assert receiver.receive()[0].step == 1
    assert not path.exists()


def test_micro_batch_mailbox_is_removed_after_decode(tmp_path):
    batch = MicroBatch(
        input_ids=[1],
        loss_mask=[True],
        advantages=[1.0],
        inference_logprobs=[-0.1],
        position_ids=[0],
        sequence_lengths=[1],
        temperatures=[0.7],
        env_names=["test"],
    )
    sender = FileSystemMicroBatchSender(tmp_path, data_world_size=1, current_step=1)
    sender.send([[batch]])
    receiver = FileSystemMicroBatchReceiver(
        tmp_path, data_rank=0, current_step=1
    )
    path = receiver._get_micro_batch_path()

    assert receiver.receive()[0].input_ids == [1]
    assert not path.exists()
