from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from training.src.data.datamodule import CXRDataModule
from training.tests.helpers.generate_dummy_data import generate_dummy_chexpert_dataset

COLUMNS = [
    "Path",
    "Sex",
    "Age",
    "Frontal/Lateral",
    "AP/PA",
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "patient_id",
    "fold",
]


def create_dummy_image(path: Path):
    arr = (np.random.rand(224, 224) * 255).astype("uint8")
    img = Image.fromarray(arr)
    img.save(path)


@pytest.fixture
def dummy_cxr_dataset(tmp_path):
    generate_dummy_chexpert_dataset(base_dir=tmp_path, num_train=20, num_val=10)
    return tmp_path


def test_cxr_datamodule(dummy_cxr_dataset):
    dm = CXRDataModule(
        data_dir=str(dummy_cxr_dataset),
        batch_size=2,
        num_workers=2,
        pin_memory=False,
    )

    # Must be None before setup
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None

    dm.setup()

    # After setup, datasets should be loaded
    assert len(dm.data_train) == 20
    assert len(dm.data_val) == 10
    # test loader loads val.csv again (per your code)
    assert len(dm.data_test) == 10

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    x, y = batch

    # shape checks
    assert x.shape[0] == 2
    assert x.ndim == 4  # B C H W
    assert y.shape[0] == 2

    # dtype checks
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32 or y.dtype == torch.int64
