import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CXRDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.label_cols = self.df.columns.difference(
            ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", "patient_id", "fold"]
        )
        # map -1 to 0
        self.df[self.label_cols] = self.df[self.label_cols].replace(-1, 0).astype("float32")

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
