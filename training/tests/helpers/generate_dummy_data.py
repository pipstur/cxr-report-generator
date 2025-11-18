import os

import numpy as np
import pandas as pd
from PIL import Image


def generate_dummy_chexpert_dataset(base_dir, num_train=20, num_val=10, img_size=(224, 224)):
    """
    Generates a dummy CheXpert-like dataset with the required structure.

    base_dir = cfg_train.data.data_dir
    """

    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

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

    def random_image(path):
        arr = np.random.randint(0, 255, (img_size[0], img_size[1], 3), dtype=np.uint8)
        Image.fromarray(arr).save(path)

    train_rows = []
    for i in range(num_train):
        patient_id = f"patient{i+1}"
        img_name = f"{patient_id}_frontal.jpg"
        img_path = os.path.join(train_dir, img_name)

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

        train_rows.append(row)

    val_rows = []
    for i in range(num_val):
        patient_id = f"patientV{i+1}"
        img_name = f"{patient_id}_frontal.jpg"
        img_path = os.path.join(val_dir, img_name)

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

        val_rows.append(row)

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)

    train_df.to_csv(os.path.join(base_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(base_dir, "val.csv"), index=False)

    print(f"\n✅ Dummy dataset created at: {base_dir}")
    print(f"   Train images: {num_train}")
    print(f"   Val images:   {num_val}\n")
