from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from src.data.datamodule import CXRDataModule

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
    # Make directories
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    # Create dummy images + CSV entries
    train_rows = []
    val_rows = []

    for i in range(3):
        img_path = train_dir / f"img{i}.jpg"
        create_dummy_image(img_path)
        row = [
            str(img_path),
            "Male",
            33,
            "Frontal",
            "PA",
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            f"patient{i}",
            0,
        ]
        train_rows.append(row)

    for i in range(2):
        img_path = val_dir / f"img{i}.jpg"
        create_dummy_image(img_path)
        row = [
            str(img_path),
            "Female",
            29,
            "Frontal",
            "AP",
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            f"patientV{i}",
            0,
        ]
        val_rows.append(row)

    pd.DataFrame(train_rows, columns=COLUMNS).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame(val_rows, columns=COLUMNS).to_csv(tmp_path / "val.csv", index=False)

    return tmp_path


def test_cxr_datamodule(dummy_cxr_dataset):
    dm = CXRDataModule(
        data_dir=str(dummy_cxr_dataset),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    # Must be None before setup
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None

    dm.setup()

    # After setup, datasets should be loaded
    assert len(dm.data_train) == 3
    assert len(dm.data_val) == 2
    # test loader loads val.csv again (per your code)
    assert len(dm.data_test) == 2

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
