from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CXRDataset(Dataset):
    def __init__(
        self, csv_file, dataset: str, label_selection: List[str] = ["all"], transform=None
    ):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        if dataset.lower() == "chexpert":
            self.label_cols = (
                self.df.columns.difference(
                    ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", "patient_id", "fold"]
                )
                if label_selection[0] == "all"
                else label_selection
            )
        elif dataset.lower() == "nih":
            self.label_cols = (
                self.df.columns.difference(
                    [
                        "Image Index",
                        "Finding Labels",
                        "Follow-up #",
                        "Patient ID",
                        "Patient Age",
                        "Patient Gender",
                        "View Position",
                        "OriginalImage[Width,Height]",
                        "OriginalImagePixelSpacing[x,y]",
                        "Unnamed: 11",
                        "fold",
                    ]
                )
                if label_selection[0] == "all"
                else label_selection
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Choose 'chexpert' or 'nih'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["Path"]
        try:
            img = Image.open(path).convert("L")
        except Exception as e:
            print(f"[ERROR OPENING IMAGE] {path} - {e}")
            raise e
        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(row[self.label_cols].to_numpy(dtype="float32"), dtype=torch.float32)
        return img, labels
