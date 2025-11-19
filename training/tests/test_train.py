import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from training.src.train import train
from training.tests.helpers.generate_dummy_data import generate_dummy_chexpert_dataset
from training.tests.helpers.run_if import RunIf

tensorboard = {
    "tensorboard": {
        "_target_": "lightning.pytorch.loggers.tensorboard.TensorBoardLogger",
        "save_dir": "${paths.output_dir}/tensorboard/",
        "name": None,
        "log_graph": False,
        "default_hp_metric": True,
        "prefix": "",
    }
}


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    generate_dummy_chexpert_dataset(base_dir=cfg_train.data.data_dir)
    with open_dict(cfg_train):
        cfg_train.logger = tensorboard
        cfg_train.data.batch_size = 1
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.limit_val_batches = 1.0
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.logger = tensorboard
        cfg_train.data.batch_size = 1
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.limit_val_batches = 1.0
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    generate_dummy_chexpert_dataset(base_dir=cfg_train.data.data_dir)
    with open_dict(cfg_train):
        cfg_train.logger = tensorboard
        cfg_train.data.batch_size = 1
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
        cfg_train.trainer.limit_val_batches = 1.0
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    generate_dummy_chexpert_dataset(base_dir=cfg_train.data.data_dir)
    with open_dict(cfg_train):
        cfg_train.logger = tensorboard
        cfg_train.data.batch_size = 1
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
        cfg_train.trainer.limit_val_batches = 1.0
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    generate_dummy_chexpert_dataset(base_dir=cfg_train.data.data_dir)
    with open_dict(cfg_train):
        cfg_train.logger = tensorboard
        cfg_train.data.batch_size = 1
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
        cfg_train.trainer.limit_val_batches = 1.0
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.logger = tensorboard
        cfg_train.data.batch_size = 1
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_val_batches = 1.0

    HydraConfig().set_config(cfg_train)
    generate_dummy_chexpert_dataset(base_dir=cfg_train.data.data_dir)
    metric_dict_1, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    metric_dict_2, _ = train(cfg_train)

    files = os.listdir(tmp_path / "checkpoints")
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    assert metric_dict_1["train/acc"] < metric_dict_2["train/acc"]
    assert metric_dict_1["val/acc"] < metric_dict_2["val/acc"]
