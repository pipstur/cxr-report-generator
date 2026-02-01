import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image


def generate_dummy_chexpert_dataset(
    base_dir: str,
    num_train: int = 128,
    num_val: int = 128,
    num_test: int = 128,
    img_size: Tuple[int, int] = (224, 224),
) -> None:
    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "../", "test")
    test_img_dir = os.path.join(test_dir, "images")

    for d in [train_dir, val_dir, test_img_dir]:
        os.makedirs(d, exist_ok=True)

    label_cols = [
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
    ]

    def random_image(path: str) -> None:
        arr = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
        Image.fromarray(arr).save(path)

    def make_rows(
        split_dir: str,
        prefix: str,
        n: int,
    ):
        rows = []
        for i in range(n):
            patient_id = f"{prefix}{i + 1}"
            img_name = f"{patient_id}_frontal.jpg"
            img_path = os.path.join(split_dir, img_name)

            random_image(img_path)

            row = {
                "Path": img_path,
                "Sex": np.random.choice(["Male", "Female"]),
                "Age": int(np.random.randint(20, 90)),
                "Frontal/Lateral": "Frontal",
                "AP/PA": np.random.choice(["AP", "PA"]),
            }

            for c in label_cols:
                row[c] = int(np.random.randint(0, 2))

            rows.append(row)

        return rows

    train_df = pd.DataFrame(make_rows(train_dir, "patientT", num_train))
    val_df = pd.DataFrame(make_rows(val_dir, "patientV", num_val))
    test_df = pd.DataFrame(make_rows(test_img_dir, "patientX", num_test))

    train_df.to_csv(os.path.join(base_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(base_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(test_dir, "test.csv"), index=False)

    print(f"\n✅ Dummy CheXpert dataset created at: {base_dir}")
    print(f"   Train samples: {num_train}")
    print(f"   Val samples:   {num_val}")
    print(f"   Test samples:  {num_test}\n")
