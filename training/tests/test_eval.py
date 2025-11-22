import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from training.src.eval import evaluate
from training.src.train import train
from training.tests.helpers.generate_dummy_data import generate_dummy_chexpert_dataset


@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir
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
    with open_dict(cfg_train):
        cfg_train.logger = tensorboard
        cfg_train.data.batch_size = 1
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_val_batches = 1.0

    HydraConfig().set_config(cfg_train)
    generate_dummy_chexpert_dataset(base_dir=cfg_train.data.data_dir)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
        cfg_eval.logger = tensorboard
        cfg_eval.data.batch_size = 1
        cfg_eval.test = True
        cfg_eval.trainer.limit_val_batches = 1.0

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/acc"] > 0.0
    assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001
